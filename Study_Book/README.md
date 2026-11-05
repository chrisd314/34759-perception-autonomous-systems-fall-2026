# Perception for Autonomous Systems (34759): study book

One chapter per class. Each chapter has:
- the concepts from the lecture slides;
- the methods used in the exercises, with the key lines of code and the results;
- the failure cases met along the way;
- comparison tables and key takeaways.

The full code stays in the exercise notebooks; the book quotes only the lines that carry the idea.

## Chapters

| Class | Chapter | Status |
|---|---|---|
| 1 | [Basic image processing](01_basic_image_processing.md): images, colour spaces, filtering, thresholding, morphology, contours | done |
| 2 | [Edges, features, matching, fitting and optical flow](02_feature_detection.md): Canny, Harris/Shi-Tomasi, SIFT/ORB, Hough, RANSAC, Lucas–Kanade, Farnebäck | done |
| 3 | [Stereo vision and template matching](03_stereo_vision.md): epipolar geometry, rectified stereo, disparity and depth, block matching | done |
| 4 | [Camera calibration and epipolar geometry](04_calibration_and_epipolar.md): camera model, DLT, homography, distortion, E and F, rectification | done |
| 5 | [Point clouds: pose estimation and registration](05_point_cloud_registration.md): pose, Kabsch, ICP, normals, FPFH, RANSAC in 3-D | done |
| 6 | [Regression, ranging and clustering](06_regression_and_clustering.md): least squares, lidar, k-means, mean shift, DBSCAN | done |
| 7 | [State estimation: histogram and Kalman filters](07_state_estimation.md): Bayes filters, Kalman, EKF, UKF, particle filters | done |
| 8 | [Classification](08_classification.md): PCA, k-NN, SVMs and kernels, grid search, sliding window, IoU and NMS | done |
| 9 | [Visual odometry](09_visual_odometry.md): orientation, tracking, 3-D to 3-D, PnP, 2-D to 2-D, drift, bundle adjustment, VIO, stereo VO | done |
| 10 | [Final project: tracking the Husky](10_final_project.md): approaches, tests, results and limitations | done; the report and the upload are still to do |

[CHECKLIST.md](CHECKLIST.md) lists everything still to review: open items, things not done, and every small change made along the way.

[reference.md](reference.md) lists every function (with the class it was introduced in, and whether it is *taught*, *project*, *industry* or *figures* only) and every term, with a link to the chapter section.

## Conventions

- **L1 s.12** means lecture 1, slide 12; slide numbers equal PDF page numbers. **L2** is the Class 2 lecture PDF, **L3** the Class 3 one, **L4a** and **L4b** the two Class 4 ones, and so on.
- **cell 4.3** means the exercise cell under the heading "4.3" in the notebook.
- From chapter 3 on, the "When it fails" tables say whether the notebook applies each fix.
- Images are NumPy arrays indexed `[y, x]` (row, column); OpenCV points are `(x, y)` and sizes `(w, h)`.
- `cv2.imread` returns **BGR**; the notebooks convert to RGB before displaying.
- OpenCV 8-bit HSV: H 0–179 (degrees / 2), S and V 0–255.

## Figures

All figures are in `figures/classN/`.
- `ex*` figures are the outputs of exercise cells.
- `concept_*` figures are drawn from the exercise images or from small synthetic examples, to show a lecture concept.

To rebuild them after changing a notebook (the notebooks are executed in memory; the files on disk are not modified):

```
.venv/bin/python Study_Book/tools/build_figures.py      # all chapters
.venv/bin/python Study_Book/tools/build_figures.py 1    # chapter 1 only
```

The script also prints the numbers quoted in the chapters.

## Course facts (from the introduction lecture)

- **Assessment:**
  - weekly projects: one notebook per group of 5, Pass/Fail; at least 7 passes out of 9 give 20 % of the grade;
  - final project report: 10 ± 2 pages with a video link, Pass/Fail, 20 %;
  - exam: on campus, online, multiple choice, 2 hours, 60 %.
- **Dates:** final project description on 2 November, report hand-in on 3 December, exam on 8 December.
- **Books:** R. Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed., 2022; D. Forsyth and J. Ponce, *Computer Vision: A Modern Approach*, 2nd ed.
- **Learning objectives:**
  - describe the steps of 3D reconstruction from multiple views;
  - define image feature extraction and matching techniques;
  - discuss ranging sensors;
  - process 3D point clouds;
  - combine visual and 3D input with state estimation;
  - compare classical and learning-based classification;
  - describe visual odometry.
