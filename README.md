# VaibhavBhola26A19045
UGV DTU Software Department Final Round Submission
## 📅 Daily Progress Log

### 📅 25 August 2026
*   **What I Did:** 
    * Read through the recruitment guidelines and explored what the tasks required.
    * Researched about the uses and functionality of Git, and OpenCV, as I had never used them before.
*   **Problems Faced:** 
    * I did not have Git installed on my laptop and was completely unfamiliar with version control.
    * I had zero prior experience with computer vision or image processing and felt intimidated about how a program could "see" and detect lanes or obstacles on a road.
*   **What I Learnt:** 
    * Understood that Git is a local version control tool to track code changes, while GitHub is a cloud platform to host and share that code.
    * Learnt that OpenCV is a highly powerful, classical library used for image processing and computer vision, which means I don't need complex machine learning to detect objects or paths.

---

### 📅 26 August 2026
*   **What I Did:** 
    * Installed Git on my laptop and configured my local profile.
    * Created my official GitHub repository online named with my roll number.
    * Linked and cloned the online repository to my local laptop using VS Code and Git Bash.
    * Set up the initial empty directory structures for the upcoming tasks.
*   **Problems Faced:** 
    * Had a hard time navigating folders inside the terminal interface (command line) and figuring out how to connect my laptop's Git to my online GitHub account.
*   **What I Learnt:** 
    * Gained confidence with basic terminal commands (`cd`, `mkdir`, `ls`, `pwd`).
    * Mastered the core Git cycle: staging changes (`git add`), capturing snapshots locally (`git commit`), and uploading them to the cloud (`git push`).
    * Learnt how to integrate Git Bash directly inside VS Code to make my daily workflow seamless.

---

### 📅 27 August 2026
*   **What I Did:** 
    * Created the file structure for `task-2` (nested `input` and `output` folders, along with `main.py`).
    * Downloaded the official road images dataset and placed the raw images inside `task-2/input/`.
    * Developed and successfully executed a classical Python OpenCV pipeline inside `task-2/main.py` to highlight road lanes and save them to `task-2/output/`.
*   **Problems Faced:** 
    * **Path Error:** The script initially crashed because Python could not find the `input` directory when run from the main repository folder.
    * **CV Concepts:** I struggled to understand how a pixel grid is converted into mathematical lines.
    * **Parameters Tuning:** Finding the right low and high threshold values for edge detection so that shadows and grass textures didn't interfere with the lanes.
*   **What I Learnt:** 
    * Resolved the path issue professionally by using Python's built-in `os` module (`os.path.abspath` and `os.path.dirname`) to dynamically locate files relative to the script itself.
    * Understood the mathematical progression of a classical computer vision pipeline:
      1. **Grayscale conversion** (reduces detail)
      2. **Gaussian Blur** (smoothes out noise)
      3. **Canny Edge Detection** (identifies sharp changes in contrast)
      4. **Region of Interest (ROI) Masking** (isolates the road triangular zone ahead)
      5. **Hough Lines & np.polyfit** (connects edge pixels into solid, extrapolated boundary lines)
      6. **Alpha Blending** (using `cv2.addWeighted` to paint a semi-transparent green drivable area overlay between the lines).

---

### 📅 28 August 2026
*   **What I Did:** 
    * Completed Task 3 (Obstacle & Pothole Detection).
    * Created the file structure for `task-3` containing `input` and `output` subfolders.
    * Wrote a Python script in `task-3/main.py` using OpenCV to isolate white circular blobs on the track and draw rectangular bounding boxes around them.
    * Automatically printed the total object count and center coordinates for each obstacle onto the final output images.
*   **Problems Faced:** 
    * Filtering out small specks of dust, noise, or track textures that were being falsely detected as obstacles.
    * Calculating the exact mathematical center of irregular circular blobs without the program crashing when calculating image moments.
*   **What I Learnt:** 
    * Learnt how to use **Binarisation/Thresholding** (`cv2.threshold` / Otsu's thresholding) to isolate objects of a specific brightness from a dark background.
    * Learnt how **Morphological Operations** (opening and closing via `cv2.morphologyEx`) clean up small salt-and-pepper noise in an image.
    * Understood **Contour Detection** (`cv2.findContours`) and how to extract spatial dimensions using bounding rectangles (`cv2.boundingRect`) and image moments (`cv2.moments`) to pinpoint exact center coordinates.

    *   **Task 3 Code Revision (Update):** 
    *   **What I Did:** Refined the threshold limits and contour filtering logic in `task-3/main.py` to improve the detection accuracy of the white circular obstacles.
    *   **Problems Faced:** Found that minor lighting variations and small non-obstacle pixels were occasionally causing false positives.
    *   **What I Learnt:** Learnt how to adjust the minimum area threshold filter (`cv2.contourArea`) dynamically to exclude background noise, ensuring only true obstacles/potholes are bounded.