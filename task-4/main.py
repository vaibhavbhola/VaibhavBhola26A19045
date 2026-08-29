import cv2
import math
import heapq
import numpy as np
from pathlib import Path
import networkx as nx

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# ----------------------------- Road -----------------------------
ROAD_BRIGHTNESS_THRESHOLD = 89
MAX_CHANNEL_SPREAD = 18
ROAD_OPEN_KERNEL = 7
ROAD_CLOSE_KERNEL = 25
ROAD_FINAL_CLOSE_KERNEL = 31
ROAD_CLEARANCE_KERNEL = 9

# ---------------------------- Hazards ---------------------------
# Potholes are dark, nearly neutral, compact circular/elliptical marks.
POTHOLE_GRAY_THRESHOLD = 82
POTHOLE_SAT_THRESHOLD = 75
POTHOLE_MIN_AREA = 180
POTHOLE_MAX_AREA = 2600
POTHOLE_MIN_SIZE = 18
POTHOLE_MAX_SIZE = 85

# Coloured obstacle pieces.
OBSTACLE_SAT_THRESHOLD = 45
OBSTACLE_MIN_AREA = 70
OBSTACLE_MAX_AREA = 6500
OBSTACLE_MAX_SIZE = 120

# Light/cream/white square obstacle pieces.
LIGHT_GRAY_THRESHOLD = 115
LIGHT_SAT_MAX = 75
LIGHT_MIN_AREA = 150
LIGHT_MAX_AREA = 2600
LIGHT_MIN_SIZE = 14
LIGHT_MAX_SIZE = 75

# Neutral gray obstacle pieces used in some images.
GRAY_OBSTACLE_MIN_AREA = 90
GRAY_OBSTACLE_MAX_AREA = 1800
GRAY_OBSTACLE_MAX_SIZE = 80
GRAY_OBSTACLE_LOW = 35
GRAY_OBSTACLE_HIGH = 82
GRAY_OBSTACLE_SAT_MAX = 75

# Do not let the START graphic become an obstacle.
START_IGNORE_RADIUS = 115

# Safety margin around every detected hazard.
HAZARD_DILATION_KERNEL = 15

# ---------------------------- Planning --------------------------
CENTERLINE_SAMPLE_SPACING = 4
ROAD_CENTER_PREFERENCE = 0.018
ASTAR_SCALE = 2
DETOUR_CORRIDOR_WIDTH = 105
DETOUR_CORRIDOR_DILATION = 45
HAZARD_SEARCH_MARGIN_POINTS = 14

# ---------------------------- Drawing ---------------------------
ROUTE_OUTER_COLOR = (25, 25, 25)       # dark lining
ROUTE_INNER_COLOR = (0, 255, 0)        # green route
OBSTACLE_COLOR = (0, 90, 255)          # orange/red
POTHOLE_COLOR = (0, 0, 0)             # black
POTHOLE_RING_INNER = (255, 255, 255)
ROAD_BOUNDARY_COLOR = (255, 200, 0)    # cyan-ish
ROAD_BOUNDARY_OUTER_COLOR = (20, 20, 20)
ROAD_BOUNDARY_OUTER_THICKNESS = 9
INNER_BOUNDARY_MIN_AREA_RATIO = 0.015
PARTICLE_COLOR = (255, 255, 255)      # white particle
PARTICLE_CORE_COLOR = (0, 255, 255)   # yellow core
ARROW_COLOR = (255, 255, 255)

ROUTE_OUTER_THICKNESS = 15
ROUTE_INNER_THICKNESS = 8
ROAD_BOUNDARY_THICKNESS = 3
DIRECTION_ARROW_SPACING = 80
DIRECTION_ARROW_SIZE = 12

def ellipse(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

def largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return np.zeros_like(mask), 0
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask)
    out[labels == idx] = 255
    return out, int(stats[idx, cv2.CC_STAT_AREA])

def get_contour_metrics(contour):
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 0.0 if perimeter <= 0 else 4 * math.pi * area / (perimeter * perimeter)
    rectangularity = area / float(max(w * h, 1))
    return area, x, y, w, h, circularity, rectangularity

def detect_road(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    spread = img.max(axis=2).astype(np.int16) - img.min(axis=2).astype(np.int16)
    neutral = spread <= MAX_CHANNEL_SPREAD
    bright = ((blurred >= ROAD_BRIGHTNESS_THRESHOLD) & neutral).astype(np.uint8) * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, ellipse(ROAD_OPEN_KERNEL))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, ellipse(ROAD_CLOSE_KERNEL))
    bright, bright_area = largest_component(bright)
    road = bright
    if bright_area < 0.05 * img.shape[0] * img.shape[1]:
        relaxed = ((blurred >= ROAD_BRIGHTNESS_THRESHOLD - 4) & neutral).astype(np.uint8) * 255
        relaxed = cv2.morphologyEx(relaxed, cv2.MORPH_OPEN, ellipse(ROAD_OPEN_KERNEL))
        relaxed = cv2.morphologyEx(relaxed, cv2.MORPH_CLOSE, ellipse(ROAD_CLOSE_KERNEL))
        road, _ = largest_component(relaxed)
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, ellipse(ROAD_FINAL_CLOSE_KERNEL))
    road = cv2.morphologyEx(road, cv2.MORPH_OPEN, ellipse(5))
    road, _ = largest_component(road)
    return road

def get_road_boundaries(road_mask):
    contours, hierarchy = cv2.findContours(road_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours or hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    outer_idx = max(
        [i for i, h in enumerate(hierarchy) if h[3] == -1],
        key=lambda i: cv2.contourArea(contours[i]),
        default=-1
    )
    if outer_idx < 0:
        return []
    outer_area = cv2.contourArea(contours[outer_idx])
    result = []

    def simplify(c):
        epsilon = max(1.5, 0.0025 * cv2.arcLength(c, True))
        return cv2.approxPolyDP(c, epsilon, True)

    result.append(simplify(contours[outer_idx]))
    child = hierarchy[outer_idx][2]
    while child != -1:
        area = cv2.contourArea(contours[child])
        if area >= outer_area * INNER_BOUNDARY_MIN_AREA_RATIO:
            result.append(simplify(contours[child]))
        child = hierarchy[child][0]
    return result

def find_start_arrow(img, road_mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bright = cv2.inRange(gray, 180, 255)
    kernel = np.ones((3, 3), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    lines = cv2.HoughLinesP(
        bright,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=15,
        maxLineGap=10
    )
    candidates = []
    if lines is not None:
        arr = np.asarray(lines)
        if arr.size >= 4:
            arr = arr.reshape(-1, 4)
            h, w = gray.shape[:2]

            for vals in arr:
                try:
                    lx1, ly1, lx2, ly2 = [int(v) for v in vals]
                except (TypeError, ValueError):
                    continue

                dx = lx2 - lx1
                dy = ly2 - ly1
                length = float(np.hypot(dx, dy))

                if length < 10:
                    continue

                mx = int(round((lx1 + lx2) / 2))
                my = int(round((ly1 + ly2) / 2))

                if not (0 <= mx < w and 0 <= my < h):
                    continue
                yy0, yy1 = max(0, my - 4), min(h, my + 5)
                xx0, xx1 = max(0, mx - 4), min(w, mx + 5)

                road_near = np.count_nonzero(road_mask[yy0:yy1, xx0:xx1])
                score = length + 0.25 * road_near

                candidates.append((score, lx1, ly1, lx2, ly2))

    if candidates:
        _, x1, y1, x2, y2 = max(candidates, key=lambda z: z[0])
        h, w = road_mask.shape[:2]

        def road_distance(x, y):
            r = 25
            xa, xb = max(0, x-r), min(w, x+r+1)
            ya, yb = max(0, y-r), min(h, y+r+1)
            patch = road_mask[ya:yb, xa:xb]
            ys, xs = np.where(patch > 0)
            if len(xs) == 0:
                return float("inf")
            gx = xs + xa
            gy = ys + ya
            return float(np.min((gx-x)**2 + (gy-y)**2))

        d1 = road_distance(x1, y1)
        d2 = road_distance(x2, y2)

        if d1 <= d2:
            tip = np.array([x1, y1], dtype=float)
            other = np.array([x2, y2], dtype=float)
        else:
            tip = np.array([x2, y2], dtype=float)
            other = np.array([x1, y1], dtype=float)
        direction = tip - other
        norm = np.linalg.norm(direction)

        if norm > 1e-6:
            direction /= norm
            return tip, direction
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bright, 8)
    best = None
    h, w = road_mask.shape[:2]

    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < 10 or area > 5000:
            continue
        cx, cy = centroids[i]
        r = 60
        xa, xb = max(0, int(cx)-r), min(w, int(cx)+r+1)
        ya, yb = max(0, int(cy)-r), min(h, int(cy)+r+1)
        near_road = np.count_nonzero(road_mask[ya:yb, xa:xb])

        score = area + 0.02 * near_road
        if best is None or score > best[0]:
            best = (score, cx, cy, ww, hh)

    if best is not None:
        _, cx, cy, ww, hh = best
        tip = np.array([cx, cy], dtype=float)
        ys, xs = np.where(road_mask > 0)
        if len(xs):
            idx = np.argmin((xs-cx)**2 + (ys-cy)**2)
            target = np.array([xs[idx], ys[idx]], dtype=float)
            direction = target - tip
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                return tip, direction / norm
    ys, xs = np.where(road_mask > 0)
    if len(xs):
        start = np.array([np.mean(xs), np.mean(ys)], dtype=float)
    else:
        start = np.array([w/2, h/2], dtype=float)

    return start, np.array([1.0, 0.0], dtype=float)

def detect_potholes(img, road_mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    pothole_region = cv2.dilate(road_mask, ellipse(75))
    dark = ((gray < POTHOLE_GRAY_THRESHOLD) &
            (sat < POTHOLE_SAT_THRESHOLD) &
            (pothole_region > 0)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, ellipse(3))
    pothole_mask = np.zeros_like(dark)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if not (POTHOLE_MIN_AREA <= area <= POTHOLE_MAX_AREA):
            continue
        if not (POTHOLE_MIN_SIZE <= min(w, h) <= POTHOLE_MAX_SIZE):
            continue
        if max(w, h) > POTHOLE_MAX_SIZE:
            continue
        component = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        _, _, _, _, _, circularity, _ = get_contour_metrics(contour)
        if circularity >= 0.28:
            pothole_mask[labels == i] = 255
    blurred = cv2.GaussianBlur(gray, (9, 9), 1.5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=35,
        param1=80, param2=18, minRadius=10, maxRadius=38
    )
    if circles is not None:
        for c in np.round(circles[0]).astype(int):
            cx, cy, r = map(int, c)
            if not (0 <= cx < img.shape[1] and 0 <= cy < img.shape[0]):
                continue
            if pothole_region[cy, cx] == 0:
                continue
            yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
            disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
            ring = ((xx - cx) ** 2 + (yy - cy) ** 2 >= max(0, r - 6) ** 2) & disk
            mean_gray = float(np.mean(gray[disk]))
            mean_sat = float(np.mean(sat[disk]))
            mean_ring = float(np.mean(gray[ring])) if np.any(ring) else 255
            center_gray = float(gray[cy, cx])
            if (mean_gray < 100 and mean_sat < 80 and
                    mean_ring < 90 and center_gray < 95):
                cv2.circle(pothole_mask, (cx, cy), r, 255, -1)
    clean = np.zeros_like(pothole_mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(pothole_mask, 8)
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if 120 <= area <= 4500 and max(w, h) <= 95:
            clean[labels == i] = 255

    return clean

def detect_obstacles(img, road_mask, pothole_mask, start_tip):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sat = hsv[:, :, 1]
    raw = np.zeros_like(gray, dtype=np.uint8)
    coloured = ((sat >= OBSTACLE_SAT_THRESHOLD) & (gray > 30)).astype(np.uint8) * 255
    raw[coloured > 0] = 255
    light = ((gray >= LIGHT_GRAY_THRESHOLD) & (sat <= LIGHT_SAT_MAX)).astype(np.uint8) * 255
    light = cv2.morphologyEx(light, cv2.MORPH_OPEN, ellipse(3))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(light, 8)
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if not (LIGHT_MIN_AREA <= area <= LIGHT_MAX_AREA):
            continue
        if not (LIGHT_MIN_SIZE <= min(w, h) <= LIGHT_MAX_SIZE):
            continue
        if max(w, h) > LIGHT_MAX_SIZE:
            continue
        aspect = w / float(max(h, 1))
        rectangularity = area / float(max(w * h, 1))
        if 0.55 <= aspect <= 1.8 and rectangularity >= 0.45:
            raw[labels == i] = 255
    gray_neutral = ((gray >= GRAY_OBSTACLE_LOW) &
                    (gray < GRAY_OBSTACLE_HIGH) &
                    (sat < GRAY_OBSTACLE_SAT_MAX)).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(gray_neutral, 8)
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if not (GRAY_OBSTACLE_MIN_AREA <= area <= GRAY_OBSTACLE_MAX_AREA):
            continue
        if max(w, h) > GRAY_OBSTACLE_MAX_SIZE:
            continue
        aspect = w / float(max(h, 1))
        if not (0.45 <= aspect <= 2.2):
            continue
        raw[labels == i] = 255
    # Remove potholes from the obstacle class.
    raw[pothole_mask > 0] = 0
    # Remove the START arrow/text neighbourhood from obstacle candidates.
    if start_tip is not None:
        cv2.circle(raw, tuple(np.rint(start_tip).astype(int)), START_IGNORE_RADIUS, 0, -1)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, ellipse(3))
    obstacle_mask = np.zeros_like(raw)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if not (OBSTACLE_MIN_AREA <= area <= OBSTACLE_MAX_AREA):
            continue
        if max(w, h) > OBSTACLE_MAX_SIZE:
            continue
        obstacle_mask[labels == i] = 255
    obstacle_mask = cv2.dilate(obstacle_mask, ellipse(3))
    obstacle_mask[pothole_mask > 0] = 0
    return obstacle_mask

def hazard_centers(mask):
    centers = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 70:
            continue
        m = cv2.moments(c)
        if abs(m["m00"]) < 1e-6:
            continue
        centers.append((m["m10"] / m["m00"], m["m01"] / m["m00"]))
    return centers

def skeletonize_mask(binary_mask):
    img = (binary_mask > 0).astype(np.uint8)
    changed = True

    while changed:
        changed = False
        to_remove = []
        ys, xs = np.where(img == 1)
        for y, x in zip(ys, xs):
            if y == 0 or x == 0 or y == img.shape[0] - 1 or x == img.shape[1] - 1:
                continue
            p2 = img[y-1, x]
            p3 = img[y-1, x+1]
            p4 = img[y,   x+1]
            p5 = img[y+1, x+1]
            p6 = img[y+1, x]
            p7 = img[y+1, x-1]
            p8 = img[y,   x-1]
            p9 = img[y-1, x-1]
            neighbors = [p2,p3,p4,p5,p6,p7,p8,p9]
            B = sum(neighbors)
            if B < 2 or B > 6:
                continue
            transitions = sum(
                neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1
                for i in range(8)
            )
            if transitions != 1:
                continue
            if p2 * p4 * p6 != 0:
                continue
            if p4 * p6 * p8 != 0:
                continue
            to_remove.append((y, x))
        if to_remove:
            changed = True
            for y, x in to_remove:
                img[y, x] = 0
        to_remove = []
        ys, xs = np.where(img == 1)
        for y, x in zip(ys, xs):
            if y == 0 or x == 0 or y == img.shape[0] - 1 or x == img.shape[1] - 1:
                continue
            p2 = img[y-1, x]
            p3 = img[y-1, x+1]
            p4 = img[y,   x+1]
            p5 = img[y+1, x+1]
            p6 = img[y+1, x]
            p7 = img[y+1, x-1]
            p8 = img[y,   x-1]
            p9 = img[y-1, x-1]
            neighbors = [p2,p3,p4,p5,p6,p7,p8,p9]
            B = sum(neighbors)
            if B < 2 or B > 6:
                continue
            transitions = sum(
                neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1
                for i in range(8)
            )
            if transitions != 1:
                continue
            if p2 * p4 * p8 != 0:
                continue
            if p2 * p6 * p8 != 0:
                continue
            to_remove.append((y, x))
        if to_remove:
            changed = True
            for y, x in to_remove:
                img[y, x] = 0

    return img.astype(bool)

def extract_main_cycle(road_mask):
    skeleton = skeletonize_mask(road_mask)
    ys, xs = np.where(skeleton)
    if len(xs) == 0:
        return None
    node_at = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs))}
    graph = nx.Graph()
    graph.add_nodes_from(range(len(xs)))
    for i, (y, x) in enumerate(zip(ys, xs)):
        y, x = int(y), int(x)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                j = node_at.get((y + dy, x + dx))
                if j is not None:
                    graph.add_edge(i, j, weight=math.hypot(dx, dy))
    cycles = nx.cycle_basis(graph)
    if not cycles:
        return None

    def cycle_length(c):
        c = np.asarray(c).reshape(-1)
        if len(c) < 3:
            return 0.0
        return sum(
            graph[int(c[k])][int(c[(k + 1) % len(c)])]["weight"]
            for k in range(len(c))
        )
    best = max(cycles, key=cycle_length)
    best = np.asarray(best, dtype=np.int64).reshape(-1)
    return np.array([[xs[int(i)], ys[int(i)]] for i in best], dtype=float)

def orient_cycle(cycle, start_tip, start_direction):
    idx = int(np.argmin(np.sum((cycle - start_tip) ** 2, axis=1)))
    cycle = np.roll(cycle, -idx, axis=0)
    look = min(50, len(cycle) - 1)
    forward = cycle[look] - cycle[0]
    backward = cycle[-look] - cycle[0]
    if np.dot(backward, start_direction) > np.dot(forward, start_direction):
        cycle = cycle[::-1]
    return cycle

def resample_cycle(cycle, spacing=CENTERLINE_SAMPLE_SPACING):
    closed = np.vstack([cycle, cycle[:1]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    total = cumulative[-1]
    distances = np.arange(0, total, spacing)
    result = []
    for d in distances:
        j = np.searchsorted(cumulative, d, side="right") - 1
        j = min(j, len(cycle) - 1)
        frac = (d - cumulative[j]) / max(lengths[j], 1e-6)
        result.append(cycle[j] * (1 - frac) + cycle[(j + 1) % len(cycle)] * frac)
    return np.asarray(result, dtype=float)

def astar(start, goal, allowed, cost_map, scale=2):
    h, w = allowed.shape
    sh, sw = h // scale, w // scale
    if sh < 2 or sw < 2:
        return None
    a_allowed = allowed[:sh * scale, :sw * scale]
    a_cost = cost_map[:sh * scale, :sw * scale]
    grid = a_allowed.reshape(sh, scale, sw, scale).all(axis=(1, 3))
    cost = a_cost.reshape(sh, scale, sw, scale).mean(axis=(1, 3))
    sx, sy = np.rint(start / scale).astype(int)
    gx, gy = np.rint(goal / scale).astype(int)
    sx, sy = int(np.clip(sx, 0, sw - 1)), int(np.clip(sy, 0, sh - 1))
    gx, gy = int(np.clip(gx, 0, sw - 1)), int(np.clip(gy, 0, sh - 1))
    free_y, free_x = np.where(grid)
    if len(free_x) == 0:
        return None

    def snap(y, x):
        if grid[y, x]:
            return y, x
        k = np.argmin((free_y - y) ** 2 + (free_x - x) ** 2)
        return int(free_y[k]), int(free_x[k])
    sy, sx = snap(sy, sx)
    gy, gx = snap(gy, gx)
    pq = [(0.0, sy, sx)]
    dist = {(sy, sx): 0.0}
    prev = {}
    moves = ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1))
    while pq:
        _, y, x = heapq.heappop(pq)
        if (y, x) == (gy, gx):
            break
        current = dist[(y, x)]
        for dy, dx in moves:
            ny, nx_ = y + dy, x + dx
            if not (0 <= ny < sh and 0 <= nx_ < sw) or not grid[ny, nx_]:
                continue
            step = math.hypot(dx, dy)
            nd = current + step * (1.0 + float(cost[ny, nx_]))
            if nd < dist.get((ny, nx_), 1e30):
                dist[(ny, nx_)] = nd
                prev[(ny, nx_)] = (y, x)
                heuristic = math.hypot(gx - nx_, gy - ny)
                heapq.heappush(pq, (nd + heuristic, ny, nx_))
    else:
        return None

    cur = (gy, gx)
    path = []
    while cur != (sy, sx):
        path.append(cur)
        if cur not in prev:
            return None
        cur = prev[cur]
    path.append((sy, sx))
    path.reverse()
    return np.asarray([[x * scale + scale / 2, y * scale + scale / 2] for y, x in path], dtype=float)

def plan_safe_loop(road_mask, obstacle_mask, pothole_mask, cycle):
    checkpoints = resample_cycle(cycle)
    n = len(checkpoints)
    safe = cv2.erode(road_mask, ellipse(ROAD_CLEARANCE_KERNEL)) > 0
    hazards = cv2.bitwise_or(obstacle_mask, pothole_mask)
    blocked = cv2.dilate(hazards, ellipse(HAZARD_DILATION_KERNEL)) > 0
    safe &= ~blocked
    centerline_img = np.zeros_like(road_mask)
    cv2.polylines(centerline_img, [np.int32(checkpoints)], True, 255, 3, cv2.LINE_AA)
    distance_from_center = cv2.distanceTransform(255 - centerline_img, cv2.DIST_L2, 5)
    center_cost = np.clip(distance_from_center * ROAD_CENTER_PREFERENCE, 0, 4)
    unsafe = np.zeros(n, dtype=bool)
    for i, p in enumerate(checkpoints):
        x, y = np.rint(p).astype(int)
        unsafe[i] = not (0 <= x < safe.shape[1] and 0 <= y < safe.shape[0] and safe[y, x])

    route_parts = []
    i = 0
    detours = 0

    while i < n:
        if not unsafe[i]:
            route_parts.append(checkpoints[i:i+1])
            i += 1
            continue

        j = i
        while j + 1 < n and unsafe[j + 1]:
            j += 1
        pre = max(0, i - HAZARD_SEARCH_MARGIN_POINTS)
        post = min(n - 1, j + HAZARD_SEARCH_MARGIN_POINTS)
        if post <= pre:
            raise RuntimeError("Could not bracket a blocked checkpoint section")
        arc_start = max(0, i - 30)
        arc_end = min(n - 1, j + 30)
        arc = checkpoints[arc_start:arc_end+1]
        corridor = np.zeros_like(road_mask)
        cv2.polylines(corridor, [np.int32(arc)], False, 255,
                      DETOUR_CORRIDOR_WIDTH, cv2.LINE_AA)
        local_allowed = safe & (cv2.dilate(corridor, ellipse(DETOUR_CORRIDOR_DILATION)) > 0)
        detour = astar(checkpoints[pre], checkpoints[post], local_allowed, center_cost, ASTAR_SCALE)
        if detour is None:
            detour = astar(checkpoints[pre], checkpoints[post], safe, center_cost * 2.5, ASTAR_SCALE)
        if detour is None:
            detour = astar(checkpoints[pre], checkpoints[post], safe, center_cost * 2.5, 1)
        if detour is None:
            raise RuntimeError(f"A* could not find a safe detour for checkpoint range {i}-{j}")

        route_parts.append(detour)
        i = j + 1
        detours += 1
    route = np.vstack(route_parts)
    if np.linalg.norm(route[-1] - checkpoints[0]) > 2:
        route = np.vstack([route, checkpoints[0]])
    return route, safe, detours

def route_is_safe(route, safe):
    if route is None or len(route) < 2:
        return False
    pts = np.rint(route).astype(int)
    inside = ((pts[:, 0] >= 0) & (pts[:, 0] < safe.shape[1]) &
              (pts[:, 1] >= 0) & (pts[:, 1] < safe.shape[0]))
    if not np.all(inside):
        return False
    return bool(np.all(safe[pts[:, 1], pts[:, 0]]))

def smooth_route(route, window=9):
    if len(route) < window * 2:
        return route
    kernel = np.ones(window, dtype=float) / window
    x = np.convolve(route[:, 0], kernel, mode="same")
    y = np.convolve(route[:, 1], kernel, mode="same")
    smoothed = np.column_stack([x, y])
    smoothed[0] = route[0]
    smoothed[-1] = route[-1]
    return smoothed

def draw_label(img, text, xy, colour=(255,255,255), scale=0.43):
    x, y = map(int, xy)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = max(3, min(x, img.shape[1] - tw - 6))
    y = max(th + base + 5, min(y, img.shape[0] - 4))
    cv2.rectangle(img, (x-3, y-th-base-4), (x+tw+3, y+3), (20,20,20), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)

def draw_arrowhead(img, p, direction, colour, size=12):
    d = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-6:
        return
    d /= norm
    side = np.array([-d[1], d[0]])
    tip = np.asarray(p) + d * size
    back = np.asarray(p) - d * size * 0.55
    left = back + side * size * 0.55
    right = back - side * size * 0.55
    pts = np.int32([tip, left, right])
    cv2.fillConvexPoly(img, pts, colour)
    cv2.polylines(img, [pts], True, (30,30,30), 1, cv2.LINE_AA)


def draw_route_direction(img, route):
    if len(route) < 4:
        return
    # Small white arrowheads tell which way the particle travels.
    distances = np.zeros(len(route))
    distances[1:] = np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))
    target = DIRECTION_ARROW_SPACING
    while target < distances[-1] - 15:
        idx = int(np.searchsorted(distances, target))
        idx = max(1, min(idx, len(route)-2))
        d = route[idx+1] - route[idx-1]
        draw_arrowhead(img, route[idx], d, ARROW_COLOR, DIRECTION_ARROW_SIZE)
        target += DIRECTION_ARROW_SPACING


def draw_hazards(out, obstacle_mask, pothole_mask):
    # Potholes: explicit black outline + white inner ring, with label.
    contours, _ = cv2.findContours(pothole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 80:
            continue
        cv2.drawContours(out, [c], -1, POTHOLE_COLOR, 7, cv2.LINE_AA)
        cv2.drawContours(out, [c], -1, POTHOLE_RING_INNER, 2, cv2.LINE_AA)
        x, y, w, h = cv2.boundingRect(c)
        draw_label(out, "POTHOLE", (x, max(22, y - 7)), (255,255,255), 0.38)

    # Obstacles: bright box AND contour so they remain visible against all
    # original marker colours.
    contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 60:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.drawContours(out, [c], -1, OBSTACLE_COLOR, 3, cv2.LINE_AA)
        cv2.rectangle(out, (x-4,y-4), (x+w+4,y+h+4), OBSTACLE_COLOR, 2, cv2.LINE_AA)
        draw_label(out, "OBSTACLE", (x, min(out.shape[0]-5, y+h+18)), OBSTACLE_COLOR, 0.33)

def draw_output(img, road_mask, obstacle_mask, pothole_mask, route, start_point, detours):
    out = img.copy()

    overlay = out.copy()
    overlay[road_mask > 0] = (80, 130, 80)
    out = cv2.addWeighted(overlay, 0.10, out, 0.90, 0)
    boundaries = get_road_boundaries(road_mask)
    for boundary in boundaries:
        cv2.polylines(out, [boundary], True, ROAD_BOUNDARY_OUTER_COLOR,
                      ROAD_BOUNDARY_OUTER_THICKNESS, cv2.LINE_AA)
        cv2.polylines(out, [boundary], True, ROAD_BOUNDARY_COLOR,
                      ROAD_BOUNDARY_THICKNESS, cv2.LINE_AA)

    # Hazard markings first; route is then drawn on top for clarity.
    draw_hazards(out, obstacle_mask, pothole_mask)

    route_i = np.int32(route)
    cv2.polylines(out, [route_i], False, ROUTE_OUTER_COLOR,
                  ROUTE_OUTER_THICKNESS, cv2.LINE_AA)
    cv2.polylines(out, [route_i], False, ROUTE_INNER_COLOR,
                  ROUTE_INNER_THICKNESS, cv2.LINE_AA)
    draw_route_direction(out, route)

    if start_point is not None:
        p = tuple(np.rint(start_point).astype(int))
        cv2.circle(out, p, 15, ROUTE_OUTER_COLOR, -1, cv2.LINE_AA)
        cv2.circle(out, p, 10, PARTICLE_COLOR, -1, cv2.LINE_AA)
        cv2.circle(out, p, 5, PARTICLE_CORE_COLOR, -1, cv2.LINE_AA)
        cv2.putText(out, "PARTICLE START",
                    (p[0] + 18, p[1] - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255,255,255), 2, cv2.LINE_AA)

        # A short initial arrow makes the particle's travel direction obvious.
        d = route[min(12, len(route)-1)] - route[0]
        draw_arrowhead(out, np.asarray(p, dtype=float) + d / max(np.linalg.norm(d),1e-6) * 25,
                       d, ARROW_COLOR, 15)

    if start_point is not None:
        p = tuple(np.rint(start_point).astype(int))
        cv2.putText(out, "END (1 LOOP)",
                    (p[0] + 18, p[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                    (255,255,255), 2, cv2.LINE_AA)

    cv2.rectangle(out, (15, 15), (480, 150), (15,15,15), -1)
    cv2.putText(out, "SAFE ONE-LOOP PATH", (28, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.67, (255,255,255), 2, cv2.LINE_AA)
    cv2.line(out, (30,62), (95,62), ROUTE_OUTER_COLOR, 12, cv2.LINE_AA)
    cv2.line(out, (30,62), (95,62), ROUTE_INNER_COLOR, 7, cv2.LINE_AA)
    cv2.putText(out, "route: dark outer + green inner", (110,68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(out, "orange = obstacle", (30,92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, OBSTACLE_COLOR, 1, cv2.LINE_AA)
    cv2.putText(out, "black ring = pothole", (30,114),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(out, "white arrows = particle direction", (30,136),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, ARROW_COLOR, 1, cv2.LINE_AA)
    cv2.putText(out, f"A* detours: {detours}",
                (15, out.shape[0]-18), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (255,255,255), 2, cv2.LINE_AA)
    return out

def process_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError("OpenCV could not read the image")

    road_mask = detect_road(img)
    if cv2.countNonZero(road_mask) == 0:
        raise RuntimeError("Road/track could not be detected")

    start_tip, start_direction = find_start_arrow(img, road_mask)
    if start_tip is None:
        raise RuntimeError("START arrow could not be detected")

    cycle = extract_main_cycle(road_mask)
    if cycle is None:
        raise RuntimeError("Closed track centreline could not be extracted")

    cycle = orient_cycle(cycle, start_tip, start_direction)
    start_point = cycle[0].copy()
    obstacle_mask = detect_obstacles(img, road_mask, np.zeros_like(road_mask), start_tip)
    pothole_mask = detect_potholes(img, road_mask)
    obstacle_mask[pothole_mask > 0] = 0

    route, safe, detours = plan_safe_loop(
        road_mask, obstacle_mask, pothole_mask, cycle
    )

    smoothed = smooth_route(route, 7)
    if route_is_safe(smoothed, safe):
        route = smoothed
    elif not route_is_safe(route, safe):
        raise RuntimeError("Final route safety validation failed")

    output = draw_output(
        img, road_mask, obstacle_mask, pothole_mask,
        route, start_point, detours
    )
    return output, route, start_point, detours, len(hazard_centers(obstacle_mask)), len(hazard_centers(pothole_mask))

def main():
    print("=" * 72)
    print("SAFE ONE-LOOP PATH PLANNER")
    print("Road + pothole + obstacle detection + checkpoint/A* planning")
    print("=" * 72)

    if not INPUT_DIR.exists():
        print(f"ERROR: input directory not found: {INPUT_DIR}")
        print("Create input/ and place the course images inside it.")
        return

    files = sorted(p for p in INPUT_DIR.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    print(f"Found {len(files)} image(s).")
    if not files:
        print("No supported images found.")
        return

    success = 0
    for path in files:
        print(f"\nProcessing: {path.name}")
        try:
            output, route, start, detours, obstacles, potholes = process_image(path)
            destination = OUTPUT_DIR / path.name
            if not cv2.imwrite(str(destination), output):
                raise IOError("OpenCV failed to write output")

            print("  Status: SAFE LOOP PLANNED")
            print(f"  Route points: {len(route)}")
            print(f"  Obstacles marked: {obstacles}")
            print(f"  Potholes marked: {potholes}")
            print(f"  A* detours: {detours}")
            print(f"  Start: ({start[0]:.1f}, {start[1]:.1f})")
            print(f"  Saved: {destination}")
            success += 1
        except Exception as exc:
            import traceback
            print(f"  [ERROR] {type(exc).__name__}: {exc}")
            print("  [TRACEBACK]")
            traceback.print_exc()
    print("\n" + "=" * 72)
    print(f"Finished: {success}/{len(files)} image(s) planned successfully.")
    print("=" * 72)

if __name__ == "__main__":
    main()