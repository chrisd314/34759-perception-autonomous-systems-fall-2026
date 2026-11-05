# Checklist · Exercises and study book

Everything still to review for the exercises and this book, for classes 1 to 9 and the final project. Items marked `[x]` are finished, and say what was done. Tick the others as you go through them.

**Contents**
1. [Final project](#final-project)
2. [Overall status](#overall-status)
3. [Open items](#open-items)
4. [Not done](#not-done)
5. [Limits of the verification](#limits-of-the-verification)
6. [Checks never done](#checks-never-done)
7. [Changes to template code](#changes-to-template-code)
8. [Files added](#files-added)
9. [Functions no earlier template showed](#functions-no-earlier-template-showed)
10. [More than the minimum](#more-than-the-minimum)
11. [Tuned to one dataset](#tuned-to-one-dataset)

---

## Final project

**Done**

- [x] **Calibration, rectification, disparity, 3-D scene, detection and tracking in 2-D and 3-D** for all four sequences, in [Final Project.ipynb](../Final_project/Final%20Project.ipynb), with course methods and no deep learning.
- [x] **The comparisons behind each choice,** in [Approach tests.ipynb](../Final_project/Approach%20tests.ipynb).
- [x] **The video:** [husky_tracking.mp4](../Final_project/results/husky_tracking.mp4), 1,594 frames, 39 MB. Only 6 of its frames were looked at.
- [x] **The study book's chapter:** [chapter 10](10_final_project.md).
- [x] **The board's square size:** 80 × 80 mm, as given by the professor. The project description doesn't state it, so cite the professor in the report. The robot then measures 0.956 m, 3.4 % short of the published 0.99 m.

**Your decision**

- [ ] **`cornerSubPix`.** No course material shows it, so it is off, but most 3-D tests are better with it. [The comparison](10_final_project.md#the-cornersubpix-decision) lists both. To switch it on, set `USE_SUBPIX = True`, delete `results/` and rerun (about 7 minutes).

**Still to do**

- [ ] **The report** of 10 ± 2 pages, with its Discussion and Limitations section, due 3 December 2026. Chapter 10 holds the material, but isn't the report.
- [ ] **Upload the video,** preferably to YouTube, and put the link in the report.
- [ ] **Check the Husky's 0.99 m length** against its datasheet. It is the independent check of the scale.
- [ ] **Watch the whole video,** since only a few frames were checked.

**Known limits** (details in [chapter 10](10_final_project.md#discussion-and-limitations))

- [ ] **Depth tilts by about 7°** on paths of constant distance, and about 3° with `cornerSubPix`. This points to the calibration.
- [ ] **Far depths are uncertain.** The circle comes out 25 % longer in depth than in width, and the height test suggests a 1 px disparity offset, but removing it worsens the other tests. So it isn't applied.
- [ ] **The floor marking's blob** in the first frames of sequence 2 is rejected, but why it appears wasn't investigated.
- [ ] **Colour, optical flow and the sliding window** were only tested, not used.

**Images viewed for the project.** The rule against reading images was relaxed here, because a perception project can't be checked otherwise. These were the only ones, all small sheets: 2 of the scene and early boxes, 2 of random boxes, 1 of video frames, and 2 of result plots.

---

## Overall status

Your summary of the work, checked claim by claim:

- [ ] **All exercises and weekly projects solved: no.** The biggest gap is Class 1. See [Not done](#not-done).
- [ ] **Verified: partly.** All 31 notebooks run without errors. Only Class 9 has an expected output or ground truth to compare with; the others rely on numeric checks. See [Limits of the verification](#limits-of-the-verification).
- [ ] **No functions from later classes: almost.** The one real exception, `findHomography` in Class 2 cell 3.4, is now removed. Minor items remain. See [Functions no earlier template showed](#functions-no-earlier-template-showed).
- [ ] **Minimal and robust: mostly minimal, not always robust.** See [More than the minimum](#more-than-the-minimum) and [Tuned to one dataset](#tuned-to-one-dataset).

---

## Open items

**Fixed**

- [x] **Class 1 cell 4.3, the crescent left on the red sphere.** The cell now uses both red hue bands (H 165–179 and H 0–10) and keeps the blob with the most crimson pixels. The mask covers 94 % of the sphere, up from 77 %. The 2,562 px it misses are all unsaturated shadow, and it touches no other sphere. Without the blob check, the low band would erase 29,326 px of the orange sphere. [Notebook](../Class_1_weekly_project/Exercises_week_1/Exercise%202.ipynb), [chapter 1](01_basic_image_processing.md).
- [x] **Class 1 cell texts 3.1, 4.1, 4.2, 4.3 and 5.1.** They made inaccurate claims and now give measured numbers. For example, 4.1 gives the selected share (41 %) and the share failing each limit.
- [x] **Class 2 Exercise 1 cell 19.** The text about circle voting was wrong. It now says that direct voting would need a 3-D accumulator, while `HOUGH_GRADIENT` votes for centres along the gradient in a 2-D accumulator and then chooses the radius. [Notebook](../Class_2_feature_detection_and_description/Exercises_week_2/Exercise%201%20-%20Hough%20transform.ipynb).
- [x] **Class 2 Exercise 3 cell 3.4.** Removed, because `findHomography` comes from Class 4. The inlier numbers stay in the notebook's text and in [chapter 2](02_feature_detection.md), labelled as measured outside the notebook. [Notebook](../Class_2_feature_detection_and_description/Exercises_week_2/Exercise%203%20-%20Feature%20Matching.ipynb).
- [x] **Class 9 and chapter 9.** Exercise 1 and the Weekly Project are solved. The path ends 2.11 m off after 84.6 m, against the true poses. [Chapter 9](09_visual_odometry.md).
- [x] **Class 8 and chapter 8.** Exercises 1 and 2 and the digit classifier are solved, with SVM and PCA from Class 8's own templates and no deep learning. [Chapter 8](08_classification.md).

**Still open**

- [ ] **Class 9 Weekly Project, the first 14 frames.** The estimated steps are up to 21 % shorter than `00.txt`, and more than half of the final error arises there. Tracking isn't the cause, and later frames match, but the cause isn't established. [Chapter 9](09_visual_odometry.md#weekly-project-stereo-visual-odometry) has the numbers.
- [ ] **Class 9, rotation sign against the hint.** The positions match the template's hint, but the rotations have the opposite sign. The notebook prints the camera's orientation in the world (−`rvec`), and the hint prints the raw `rvec`. To reproduce the hint exactly, delete the line `rvec = -rvec`. [Notebook](../Class_9_visual_odom/Day%2017%20Exercises/Exercise%201.ipynb).
- [ ] **Class 4 weekly project, the `rs` camera is already rectified.** Rectifying barely changes its error (0.09 → 0.10 px), so the run doesn't show what rectification does. The mynteye camera does (3.57 → 0.65 px). To try it, set `folder='mynteye'` and `max_disp` to about 200; it takes about 10 minutes. [Notebook](../Class_4_weekly_project/Weekly%20Project%204.ipynb).
- [ ] **Class 6 weekly project, the choice of k.** k = 6 and k = 4 are nearly tied on silhouette (0.534 and 0.533). Starting the search at k = 3 was my choice, not the exercise's. [Notebook](../Class_6_weekly_project/Weeklyproject6.ipynb).
- [ ] **Class 5 Exercise 2 D, the map's point count.** The map has 51,000 points from colour clouds, against 58,000 or 88,000 from grayscale clouds. Check which one you expect. [Notebook](../Class_5_point_cloud_processing/Exercises_week_5/week5/Exercise2.ipynb).
- [ ] **Chapters 1 and 2 have no "Applied in the notebook?" column.** Their "When it fails" tables predate it; all the later chapters have it.
- [ ] **Chapter 9, answers to the lecture's open questions.** The slides ask why 2-D to 2-D uses 5 points, why 2-D to 2-D and 3-D to 2-D beat 3-D to 3-D, and why stereo VO is better, and their text gives no answers. The slide text also only names loosely and tightly coupled VIO, without defining them. The book's answers and definitions are my own, so compare them with what was said in the lecture.

---

## Not done

- [ ] **Class 1, Exercise 1:** the task that uses your own picture.
- [ ] **Class 1, Weekly Project part 1 (apples):** not started. This project is your group's work, so I didn't solve it.
- [ ] **Class 1, Weekly Project part 2:** only composites over flat green, with no new background.
- [ ] **Class 2, Weekly Project:** the challenge.
- [ ] **Class 4:** FLANN, and the weekly project on the mynteye camera.
- [ ] **Class 5:** the car reconstruction.
- [ ] **Class 6:** mean shift.
- [ ] **Class 7:** multiple balls.
- [ ] **Class 8, Weekly Project:** the `lfw` faces challenge. The digit classifier itself is done, at 98.68 %, which also meets the 98 % challenge.
- [ ] **Class 9, Weekly Project:** the challenge (keeping old features), and the larger dataset behind the SharePoint links.

---

## Limits of the verification

- [ ] **I didn't look at images, figures or windows in classes 1 to 9.** The one exception is `algorithm_3.png` in Class 9. Results were checked through numbers only, so a plot can run without errors and still look wrong.
- [ ] **Display windows were switched off in every run** (`cv2.imshow` and the Open3D viewers), so that code has never been run.
- [ ] **177 of the 254 code cells have no saved output,** so their results only appear once you run them.

---

## Checks never done

- [ ] **Viewing the figures.** Of the book's 137 figures, only 4 of chapter 10's have been looked at.
- [ ] **Opening the `.ply` files in Meshlab.** They were only tested in Open3D, because Meshlab isn't installed here.
- [ ] **The chart colour validator.** It couldn't run, because Node isn't installed here.
- [ ] **Slide formulas stored as images.** The text extraction can't read them, so the book uses the standard forms, and the notation may differ from the slides.

---

## Changes to template code

- [ ] **Class 3:** a file-name parameter in `export_pointcloud`, the RGB order, and the PLY header.
- [ ] **Class 4:** a print of the results in Exercise 1.
- [ ] **Class 5:** `import time`, and `True` for `mutual_filter`, which Open3D 0.19 requires.
- [ ] **Class 6:** `create_mobius`, the name in Open3D 0.19; the `fragment.ply` path; and 10,000 points instead of 100,000.
- [ ] **Class 7:** `ndimage.convolve1d`, because SciPy removed `ndimage.filters`.
- [ ] **Class 9 Weekly Project:** `tvec[0, 0]` instead of `tvec[0]`, because newer NumPy rejects the old form; and `playImageSequence` now returns the estimated positions, for the plots.

---

## Files added

- [ ] **Weekly project notebooks** for classes 2, 3 and 4.
- [ ] **`out.ply` and `out_storage.ply`**, the point clouds exported in Class 3.
- [ ] **Unpacked exercise folders** for Class 7 and Class 9. The solutions in them were not opened.
- [ ] **Class 8 templates**, now unpacked in `Class_8_classification/week8/`. The solutions zip was not opened.
- [ ] **MNIST,** downloaded to `~/scikit_learn_data` (about 15 MB) for the Class 8 Weekly Project.
- [ ] **Final project:** the two notebooks, and `Final_project/results/` (64 MB).
- [ ] **Class 9 rows** in [reference.md](reference.md).

---

## Functions no earlier template showed

- [ ] **Class 1 industry cells, which you asked for:** `inRange`, `morphologyEx`, `getStructuringElement`, `contourArea`, `inpaint`, `bitwise_or`, `bitwise_not`.
- [ ] **Class 1, from a later class:** `COLOR_BGR2RGB` and `plt.subplots` (both Class 3).
- [ ] **Class 2 Weekly Project:** `cap.get`, `COLOR_HSV2BGR`, `COLOR_BGR2GRAY`, `np.hstack` (Class 3), `plt.axis` (Class 6).
- [ ] **Class 2 Exercise 3:** `ORB_create`, `NORM_HAMMING`.
- [ ] **Class 4 Weekly Project:** the four functions you approved (`stereoCalibrate`, `stereoRectify`, `initUndistortRectifyMap`, `remap`) and their constants.
- [ ] **Class 5 Weekly Project:** `NORM_L2`.
- [ ] **Plot helpers:** `plt.colorbar`, `plt.xlabel`, `plt.ylabel`, `plt.legend`, `plt.ylim`, `plt.gca`.
- [ ] **Class 9:** `cv2.Rodrigues`, which neither the template nor the lecture text names, and `cv2.solvePnPRansac`, where the template only says PnP with RANSAC.
- [ ] **Class 8 Weekly Project:** `metrics.accuracy_score` and `metrics.confusion_matrix`; the template imports `metrics` but names neither.
- [ ] **Final project:** `cv2.findHomography` for the corners, and the four stereo functions from Weekly Project 4. `cv2.VideoWriter` only saves the video. `decision_function` is only in the approach tests, and `cv2.cornerSubPix` is off.
- [ ] **Book only, in no notebook:** `cv2.findHomography` for the chapter 2 table; `cv2.solvePnP`, `cv2.findEssentialMat` and `cv2.recoverPose` for the chapter 9 comparisons.

---

## More than the minimum

- [ ] **Class 2 Weekly Project:** dense outlines.
- [ ] **Class 4 Weekly Project:** extra checks.
- [ ] **Class 6:** the purity helper.
- [ ] **Class 7 Weekly Project and Class 9:** extra plots.

---

## Tuned to one dataset

These values work on the course's images and may not work on others.

- [ ] **Class 1:** the RGB limits.
- [ ] **Class 3:** the `StereoBM` values and `max_disp`.
- [ ] **Final project:** the foreground threshold, the size limits in metres, and the motion noises.
- [ ] **Class 7 Weekly Project:** the Hough settings and the red check.
- [ ] **Open3D RANSAC:** not tuned, but its results vary slightly between runs.
