import cv2
import numpy as np
import os

class LaneDetector:
    def __init__(self):
        # ROI: Wide at bottom, converges at the horizon (approx 62% down)
        self.roi_v = [
            [0.0, 1.0],   # Bottom Left
            [0.46, 0.62], # Top Left
            [0.54, 0.62], # Top Right
            [1.0, 1.0]    # Bottom Right
        ]

    def isolate_lanes(self, img):
        """
        Uses multiple color spaces to find lines in any lighting.
        - Lab space 'b' channel for Yellow
        - HLS space 'L' channel for White
        """
        # 1. Normalize lighting
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        # 2. Yellow Detection (Lab space)
        lab = cv2.cvtColor(img_norm, cv2.COLOR_BGR2Lab)
        yellow_lab = lab[:,:,2] # 'b' channel: yellow to blue
        _, yellow_bin = cv2.threshold(yellow_lab, 145, 255, cv2.THRESH_BINARY)
        
        # 3. White Detection (HLS space)
        hls = cv2.cvtColor(img_norm, cv2.COLOR_BGR2HLS)
        white_hls = hls[:,:,1] # 'L' channel: Lightness
        _, white_bin = cv2.threshold(white_hls, 200, 255, cv2.THRESH_BINARY)
        
        # 4. Structural Edges (Canny)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        canny = cv2.Canny(blur, 50, 150)
        
        # Combine: Color helps find the line, Canny helps find the shape
        combined = cv2.bitwise_or(yellow_bin, white_bin)
        final_map = cv2.bitwise_or(combined, canny)
        
        return final_map

    def get_roi(self, img):
        h, w = img.shape[:2]
        mask = np.zeros_like(img)
        pts = np.array([[
            [w*self.roi_v[0][0], h*self.roi_v[0][1]],
            [w*self.roi_v[1][0], h*self.roi_v[1][1]],
            [w*self.roi_v[2][0], h*self.roi_v[2][1]],
            [w*self.roi_v[3][0], h*self.roi_v[3][1]]
        ]], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)
        return cv2.bitwise_and(img, mask)

    def robust_fit(self, pts, y_min, y_max):
        """
        Fits a stable straight line while ignoring noise.
        """
        if len(pts) < 5: return None
        
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        
        # Perform linear regression
        try:
            fit = np.polyfit(y, x, 1)
            # Predict X coordinates
            x_start = int(fit[0] * y_max + fit[1])
            x_end = int(fit[0] * y_min + fit[1])
            return [x_start, y_max, x_end, y_min]
        except:
            return None

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        
        # 1. Advanced Feature Extraction
        binary = self.isolate_lanes(frame)
        
        # 2. Area of Interest
        masked = self.get_roi(binary)
        
        # 3. Hough Line Segments
        lines = cv2.HoughLinesP(masked, 1, np.pi/180, 20, 
                                minLineLength=15, maxLineGap=150)
        
        left_pts, right_pts = [], []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                if x1 == x2: continue
                slope = (y2 - y1) / (x2 - x1)
                
                # Logic: Left lane segments have negative slope and are on left side
                # Right lane segments have positive slope and are on right side
                if slope < -0.3 and x1 < w * 0.6:
                    left_pts.extend([(x1, y1), (x2, y2)])
                elif slope > 0.3 and x1 > w * 0.4:
                    right_pts.extend([(x1, y1), (x2, y2)])

        # 4. Extrapolation
        y_max = h
        y_min = int(h * self.roi_v[1][1])
        
        left = self.robust_fit(left_pts, y_min, y_max)
        right = self.robust_fit(right_pts, y_min, y_max)

        # 5. Rendering
        overlay = np.zeros_like(frame)
        
        # Drivable Area
        if left and right:
            # Ensure lines don't intersect by clamping their top X coordinates
            # This handles the perspective convergence issue
            if left[2] > right[2]: 
                center = (left[2] + right[2]) // 2
                left[2], right[2] = center - 5, center + 5
                
            poly = np.array([[ [left[0], left[1]], [left[2], left[3]], 
                               [right[2], right[3]], [right[0], right[1]] ]], dtype=np.int32)
            cv2.fillPoly(overlay, poly, (0, 255, 100)) # Green
            
        if left: cv2.line(overlay, (left[0], left[1]), (left[2], left[3]), (255, 0, 0), 12)
        if right: cv2.line(overlay, (right[0], right[1]), (right[2], right[3]), (0, 0, 255), 12)

        return cv2.addWeighted(frame, 1.0, overlay, 0.35, 0)

def main():
    in_dir, out_dir = './Task 2/input/', './Task 2/output/'
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    detector = LaneDetector()
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(in_dir) if f.lower().endswith(exts)]
    
    if not files:
        print("No images found in ./Task 2/input/")
        return

    print(f"Processing {len(files)} images...")
    for filename in files:
        img = cv2.imread(os.path.join(in_dir, filename))
        if img is None: continue
        
        result = detector.process_frame(img)
        cv2.imwrite(os.path.join(out_dir, filename), result)
        print(f"Success: {filename}")

if __name__ == "__main__":
    main()