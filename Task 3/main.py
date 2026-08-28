import cv2
import numpy as np
import os


def process_image(img_path, output_path):
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: Could not read image {img_path}")
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = img.shape[:2]

    pothole_count = 0
    obstacle_count = 0

    # =============================================================
    # 1. POTHOLE DETECTION
    # =============================================================

    _, white_mask = cv2.threshold(
        gray, 215, 255, cv2.THRESH_BINARY
    )

    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )

    white_mask_cleaned = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        kernel_small
    )

    contours_white, _ = cv2.findContours(
        white_mask_cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours_white:

        area = cv2.contourArea(cnt)

        if area > 25:

            x, y, w, h = cv2.boundingRect(cnt)

            perimeter = cv2.arcLength(cnt, True)

            if perimeter == 0:
                continue

            circularity = 4 * np.pi * (
                area / (perimeter * perimeter)
            )

            aspect_ratio = float(w) / h

            if (
                circularity > 0.18
                and 0.25 < aspect_ratio < 4.0
                and w < width * 0.45
                and h < height * 0.45
            ):

                # Ignore objects touching image borders
                if (
                    x > 1
                    and y > 1
                    and x + w < width - 1
                    and y + h < height - 1
                ):

                    pothole_count += 1

                    # Draw bounding box
                    cv2.rectangle(
                        img,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )

                    # Print coordinates
                    cv2.putText(
                        img,
                        f"Pothole ({x},{y})",
                        (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1
                    )

                    print(
                        f"Pothole {pothole_count}: "
                        f"({x}, {y})"
                    )

       # =============================================================
    # 2. OBSTACLE DETECTION
    # =============================================================

    masks = []

    # Yellow / Gold / Brownish Yellow
    masks.append(
        cv2.inRange(
            hsv,
            np.array([10, 40, 40]),
            np.array([38, 255, 255])
        )
    )

    # Dark Blue / Light Blue
    masks.append(
        cv2.inRange(
            hsv,
            np.array([90, 60, 30]),
            np.array([135, 255, 255])
        )
    )

    # Green
    masks.append(
        cv2.inRange(
            hsv,
            np.array([38, 40, 30]),
            np.array([85, 255, 255])
        )
    )

    # -------------------------------------------------------------
    # Combine obstacle colors
    # -------------------------------------------------------------

    combined_obstacle_mask = masks[0]

    for m in masks[1:]:
        combined_obstacle_mask = cv2.bitwise_or(
            combined_obstacle_mask,
            m
        )

    # -------------------------------------------------------------
    # IMPORTANT:
    # DO NOT use MORPH_CLOSE here.
    #
    # Closing was causing nearby obstacles to merge into one.
    # A very small opening is used only to remove isolated
    # single-pixel noise while preserving small objects.
    # -------------------------------------------------------------

    kernel_noise = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    combined_obstacle_mask = cv2.morphologyEx(
        combined_obstacle_mask,
        cv2.MORPH_OPEN,
        kernel_noise
    )

    # -------------------------------------------------------------
    # Ignore bottom UI/text area
    # -------------------------------------------------------------

    bottom_ignore_start = int(height * 0.92)

    combined_obstacle_mask[
        bottom_ignore_start:height, :
    ] = 0

    # -------------------------------------------------------------
    # Find initial contours
    # -------------------------------------------------------------

    contours_obs, _ = cv2.findContours(
        combined_obstacle_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours_obs:

        area = cv2.contourArea(cnt)

        # NO minimum obstacle-area threshold
        if area <= 0:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Ignore anything in the bottom UI region
        if y >= bottom_ignore_start:
            continue

        if y + h > bottom_ignore_start:
            continue

        aspect_ratio = float(w) / h

        # =========================================================
        # CHECK WHETHER THIS COULD BE A MERGED GROUP
        # =========================================================

        # Large/wide blobs are candidates for containing multiple
        # nearby obstacles.
        #
        # Small objects, including the small ball, are handled
        # normally.
        # =========================================================

        is_possible_merged_group = (
            area > 1000 and
            aspect_ratio > 1.15
        )

        if is_possible_merged_group:

            # -----------------------------------------------------
            # Extract this contour's region
            # -----------------------------------------------------

            roi_x1 = max(0, x - 2)
            roi_y1 = max(0, y - 2)
            roi_x2 = min(width, x + w + 2)
            roi_y2 = min(height, y + h + 2)

            roi = combined_obstacle_mask[
                roi_y1:roi_y2,
                roi_x1:roi_x2
            ]

            # -----------------------------------------------------
            # Distance transform
            # -----------------------------------------------------

            dist = cv2.distanceTransform(
                roi,
                cv2.DIST_L2,
                5
            )

            max_dist = dist.max()

            if max_dist > 0:

                # Peaks inside each object become separate markers
                sure_foreground = np.uint8(
                    dist > (0.35 * max_dist)
                )

                # Find connected marker regions
                marker_count, markers = cv2.connectedComponents(
                    sure_foreground
                )

                # -------------------------------------------------
                # If multiple distinct peaks exist, use watershed
                # -------------------------------------------------

                if marker_count > 2:

                    # Background marker
                    markers = markers + 1

                    unknown = cv2.subtract(
                        roi,
                        sure_foreground
                    )

                    markers[unknown == 255] = 0

                    # Create a 3-channel image for watershed
                    roi_color = cv2.cvtColor(
                        roi,
                        cv2.COLOR_GRAY2BGR
                    )

                    cv2.watershed(
                        roi_color,
                        markers
                    )

                    # -------------------------------------------------
                    # Extract each watershed object
                    # -------------------------------------------------

                    detected_regions = []

                    for label in range(2, marker_count + 1):

                        region = np.uint8(
                            markers == label
                        ) * 255

                        region_contours, _ = cv2.findContours(
                            region,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )

                        for region_cnt in region_contours:

                            rx, ry, rw, rh = cv2.boundingRect(
                                region_cnt
                            )

                            if rw <= 0 or rh <= 0:
                                continue

                            # Convert ROI coordinates back to image
                            # coordinates
                            abs_x = roi_x1 + rx
                            abs_y = roi_y1 + ry

                            detected_regions.append(
                                (abs_x, abs_y, rw, rh)
                            )

                    # -------------------------------------------------
                    # Use the watershed results if we actually got
                    # multiple objects
                    # -------------------------------------------------

                    if len(detected_regions) >= 2:

                        for ox, oy, ow, oh in detected_regions:

                            obstacle_count += 1

                            cv2.rectangle(
                                img,
                                (ox, oy),
                                (ox + ow, oy + oh),
                                (0, 0, 255),
                                2
                            )

                            cv2.putText(
                                img,
                                f"Obstacle ({ox},{oy})",
                                (ox, max(15, oy - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (0, 0, 255),
                                1
                            )

                            print(
                                f"Obstacle {obstacle_count}: "
                                f"({ox}, {oy})"
                            )

                        # We have already processed this contour
                        continue

        # =========================================================
        # NORMAL SINGLE OBSTACLE
        # =========================================================

        obstacle_count += 1

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

        cv2.putText(
            img,
            f"Obstacle ({x},{y})",
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1
        )

        print(
            f"Obstacle {obstacle_count}: "
            f"({x}, {y})"
        )


    # =============================================================
    # 3. SUMMARY
    # =============================================================

    summary = (
        f"Total Potholes: {pothole_count} | "
        f"Total Obstacles: {obstacle_count}"
    )

    cv2.rectangle(
        img,
        (10, 10),
        (520, 50),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        img,
        summary,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # =============================================================
    # 4. SAVE IMAGE
    # =============================================================

    cv2.imwrite(output_path, img)

    print(
        f"Done: {os.path.basename(img_path)} -> "
        f"Potholes: {pothole_count}, "
        f"Obstacles: {obstacle_count}"
    )

    print("-" * 60)


# =============================================================
# DYNAMIC PATH SETUP
# =============================================================

script_dir = os.path.dirname(
    os.path.abspath(__file__)
)

input_dir = os.path.join(
    script_dir,
    "input"
)

output_dir = os.path.join(
    script_dir,
    "output"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


# =============================================================
# PROCESS IMAGES: 1.png, 2.png, 3.png ...
# =============================================================

for i in range(1, 11):

    file_name = f"{i}.png"

    in_file_path = os.path.join(
        input_dir,
        file_name
    )

    out_file_path = os.path.join(
        output_dir,
        f"{i}_output.png"
    )

    if os.path.exists(in_file_path):

        process_image(
            in_file_path,
            out_file_path
        )

    else:

        print(
            f"Skipped: {file_name} "
            f"not found in {input_dir}"
        )