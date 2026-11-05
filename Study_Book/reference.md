# Reference: functions and terms

This list grows with every chapter. **Class** is the class where a function first appears. **Level** says where it may be used:
- **taught**: used in the provided tutorial cells of a notebook, or named by the exercise text. These are the functions the exercise solutions are allowed to use.
- **industry**: used only in the "industry standard" cells. Not allowed in later solutions.
- **project**: not shown in a template, but allowed by the text of a weekly project (Weekly Project 4 allows OpenCV for undistorting and rectifying).
- **figures**: used only to draw this book's figures, to show a lecture concept.

Images are NumPy arrays: shape `(h, w)` for gray and `(h, w, 3)` for colour, type `uint8` (0–255). NumPy indexing is `[y, x]`; OpenCV points are `(x, y)` and sizes `(w, h)`.

## Functions

### Reading, arrays, display

| Function | Class | Level | What it does | Parameters and pitfalls |
|---|---|---|---|---|
| `cv2.imread(path)` | 1 | taught | reads an image file into a `uint8` array, **BGR** channel order | returns `None` (no error) if the path is wrong; later calls then fail |
| `cv2.split(img)`, `cv2.merge([a, b, c])` | 1 | taught | separates / stacks the channels | `merge([r, g, b])` turns BGR into RGB |
| `img.shape` | 1 | taught | `(h, w)` or `(h, w, c)` | rows come first |
| `img[y, x]`, `img[y0:y1, x0:x1]` | 1 | taught | one pixel / a region of interest | a slice is a view: writing into it changes the original |
| `img.copy()` | 1 | taught | independent copy | needed before drawing on an image you want to keep |
| `np.zeros_like(a)` | 1 | taught | zeros with the same shape and type | empty mask or canvas |
| `np.sum(mask == 255)` | 1 | taught | counts the selected pixels | an area in pixels |
| `plt.imshow(img, cmap='gray')` | 1 | taught | displays an image | expects RGB; 2-D arrays need `cmap='gray'`; values are auto-scaled unless `vmin`/`vmax` are given |
| `plt.figure(figsize=...)`, `plt.subplot(r, c, i)`, `plt.title`, `plt.show` | 1 | taught | figure layout | `subplot` indices start at 1 |
| `cv2.cvtColor(img, code)` | 1 | taught | converts colour spaces | taught code: `COLOR_RGB2GRAY`; industry codes: `COLOR_BGR2RGB`, `COLOR_RGB2HSV` |

### Geometry

| Function | Class | Level | What it does | Parameters and pitfalls |
|---|---|---|---|---|
| `cv2.resize(img, (w, h))` | 1 | taught | new pixel grid, interpolated | size is `(width, height)`; a fixed size distorts the aspect ratio |
| `imutils.resize(img, width=...)` | 1 | taught | resize keeping the aspect ratio | give either `width` or `height` |
| `cv2.getRotationMatrix2D(center, angle, scale)` | 1 | taught | 2×3 affine rotation matrix | degrees; positive = counter-clockwise |
| `cv2.warpAffine(img, M, (w, h))` | 1 | taught | applies an affine matrix | output size is fixed, so rotated corners are cut off |
| `imutils.rotate(img, angle)` | 1 | taught | rotation in one call | positive = counter-clockwise; corners cut off |
| `imutils.rotate_bound(img, angle)` | 1 | taught | rotation on an enlarged canvas | positive = **clockwise** |

### Filtering

| Function | Class | Level | What it does | Parameters and pitfalls |
|---|---|---|---|---|
| `cv2.GaussianBlur(img, (k, k), sigma)` | 1 | taught | Gaussian smoothing | k odd; `sigma=0` derives σ from k: 0.3·((k−1)/2 − 1) + 0.8 |
| `cv2.blur(img, (k, k))` | 1 | figures | mean (box) filter | — |
| `cv2.medianBlur(img, k)` | 1 | figures | median filter | k odd; removes salt-and-pepper noise |

### Thresholding, colour ranges, edges

| Function | Class | Level | What it does | Parameters and pitfalls |
|---|---|---|---|---|
| `cv2.threshold(src, T, maxval, type)` | 1 | taught | global threshold; returns `(T_used, mask)` | `[1]` is the mask. `THRESH_BINARY`: > T → maxval; `THRESH_BINARY_INV`: > T → 0; `+ cv2.THRESH_OTSU` picks T itself (returned in `[0]`) |
| `cv2.inRange(img, lower, upper)` | 1 | industry | 255 where every channel is within [lower, upper] | bounds are inclusive; OpenCV hue is 0–179 |
| `cv2.Canny(gray, low, high)` | 1 (theory in 2) | taught | edge map (0/255) | high = strong-edge threshold, low = hysteresis threshold; does not blur first |

### Morphology

| Function | Class | Level | What it does | Parameters and pitfalls |
|---|---|---|---|---|
| `cv2.erode(mask, kernel, iterations=n)` | 1 | taught | erosion | `kernel=None` = 3×3 square; n iterations ≈ one (2n+1)² square |
| `cv2.dilate(mask, kernel, iterations=n)` | 1 | taught | dilation | as above |
| `cv2.getStructuringElement(shape, (k, k))` | 1 | industry | builds an SE | `MORPH_RECT`, `MORPH_ELLIPSE`, `MORPH_CROSS` |
| `cv2.morphologyEx(mask, op, kernel)` | 1 | industry | opening / closing | `MORPH_OPEN`, `MORPH_CLOSE` |

### Contours and components

| Function | Class | Level | What it does | Parameters and pitfalls |
|---|---|---|---|---|
| `cv2.findContours(mask, mode, method)` | 1 | taught | borders of the non-zero regions as (x, y) point arrays | `RETR_EXTERNAL` = outer borders only; `CHAIN_APPROX_SIMPLE` = keep only the corner points |
| `imutils.grab_contours(result)` | 1 | taught | takes the contour list out of the `findContours` result | needed because the return signature differs between OpenCV versions |
| `cv2.drawContours(img, cnts, idx, colour, thickness)` | 1 | taught | draws contours | `idx=-1` = all of them; `thickness=-1` = filled |
| `cv2.contourArea(c)` | 1 | industry | area of the contour polygon | slightly smaller than the pixel count; holes are not subtracted |
| `cv2.connectedComponents(mask, connectivity=8)` | 1 | figures | labels the components | returns `(n_labels, labels)`; label 0 is the background |

### Masks, removal, drawing

| Function | Class | Level | What it does | Parameters and pitfalls |
|---|---|---|---|---|
| `cv2.bitwise_and(a, b, mask=m)` | 1 | taught | AND; with `mask`, keeps a & b where m ≠ 0 and sets the rest to 0 | `bitwise_and(img, img, mask=m)` applies a mask |
| `cv2.bitwise_or(a, b)`, `cv2.bitwise_not(a)` | 1 | industry | OR / NOT | NOT of a 0/255 mask is the same as `threshold(a, 127, 255, THRESH_BINARY_INV)` |
| `cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)` | 1 | industry | fills the masked pixels from their surroundings | the mask must cover the whole object, rim included |
| `cv2.rectangle(img, p1, p2, colour, t)` | 1 | taught | draws in place | points are (x, y); `t=-1` fills; the colour follows the array's channel order |
| `cv2.circle(img, center, r, colour, t)` | 1 | taught | draws in place | as above |
| `cv2.line(img, p1, p2, colour, t)` | 1 | taught | draws in place | as above |
| `cv2.putText(img, text, org, font, scale, colour, t)` | 1 | taught | draws text in place | `org` = bottom-left corner of the text |

### Edges, lines and circles (Class 2)

| Function | Class | Level | What it does | Parameters and pitfalls |
|---|---|---|---|---|
| `cv2.HoughLines(edges, rho, theta, threshold)` | 2 | taught | lines as `(rho, theta)` pairs, shape `(N, 1, 2)` | input = an edge image; `theta` in radians (`np.pi/180` = 1°); `threshold` = minimum votes |
| `cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, minDist, param1=, param2=, minRadius=, maxRadius=)` | 2 | taught | circles as `[x, y, r]`, shape `(1, N, 3)` | input = the gray image (runs Canny itself, `param1` = upper threshold); returns `None` if nothing is found |
| `np.cos`, `np.sin`, `np.linspace`, `np.pi` | 2 | taught | polar → cartesian conversion | angles in radians |
| `cv2.Sobel(gray, cv2.CV_32F, dx, dy, ksize=3)` | 2 | figures | derivative filter | use a float output type, or the negative values are lost |
| `cv2.Laplacian(img, cv2.CV_32F)` | 2 | figures | second derivative | smooth first (LoG) |

### Corners and features (Class 2)

| Function | Class | Level | What it does | Parameters and pitfalls |
|---|---|---|---|---|
| `cv2.cornerHarris(gray, blockSize, ksize, k)` | 2 | taught | Harris score for every pixel (float image) | threshold relative to `dst.max()`; gives clusters of pixels, not points |
| `cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance)` | 2 | taught | Shi-Tomasi corners, shape `(N, 1, 2)` | `qualityLevel` is relative to the best corner; may return `None` |
| `cv2.SIFT_create()`, `sift.detectAndCompute(gray, None)` | 2 | taught | SIFT keypoints + N × 128 `float32` descriptors | second argument = optional mask |
| `cv2.ORB_create(nfeatures=500)`, `orb.detectAndCompute(gray, None)` | 2 | taught | ORB keypoints + N × 32-byte binary descriptors | match with `NORM_HAMMING` |
| `cv2.drawKeypoints(img, kp, out, flags)` | 2 | taught | draws keypoints | `DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS` shows size and orientation |
| `cv2.BFMatcher(norm)`, `bf.match(des1, des2)` | 2 | taught | brute-force nearest neighbour for every descriptor of image 1 | `NORM_L2` (default) for SIFT, `NORM_HAMMING` for ORB; `crossCheck=True` = mutual check |
| `sorted(matches, key=lambda x: x.distance)` | 2 | taught | best matches first | — |
| `cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=...)` | 2 | taught | draws match lines between two images | `DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS` hides unmatched keypoints |
| `cv2.findHomography(pts1, pts2, cv2.RANSAC, thr)` | 2 | figures | homography + inlier mask | needs ≥ 4 pairs; `thr` in px |

### Optical flow and video (Class 2)

| Function | Class | Level | What it does | Parameters and pitfalls |
|---|---|---|---|---|
| `cv2.calcOpticalFlowPyrLK(gray1, gray2, pts1, None)` | 2 | taught | pyramidal Lucas–Kanade: `(pts2, status, err)` | use only points with `status == 1` |
| `cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.5, 0)` | 2 | taught | dense flow, shape `(h, w, 2)` | slow on large frames: downscale |
| `cv2.cartToPolar(dx, dy)` | 2 | taught | `(magnitude, angle)` | angle in radians, 0 to 2π; y axis points down |
| `cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)` / `COLOR_HSV2BGR` | 2 | taught | HSV image → displayable colours | 8-bit hue 0–179 |
| `cv2.VideoCapture(path)`, `cap.read()`, `cap.get(prop)`, `cap.release()` | 2 | taught | reads a video frame by frame | `read()` returns `(ret, frame)`; frames are BGR |
| `cv2.imshow(name, img)`, `cv2.waitKey(ms)`, `cv2.destroyAllWindows()` | 2 | taught | shows frames in a window | expects BGR; always end with `destroyAllWindows()` in Jupyter |
| `np.hstack([a, b])` | 2 | taught | puts two images side by side | same height and type required |

### Template and stereo matching (Class 3)

| Function | Class | Level | What it does | Parameters and pitfalls |
|---|---|---|---|---|
| `cv2.matchTemplate(gray, template, method)` | 3 | taught | score of the template at every position; map size (H − h + 1, W − w + 1) | both gray; `TM_CCORR_NORMED`: maximum is best; `TM_SQDIFF`: minimum is best. Banned in Weekly Projects 3 and 4 |
| `cv2.minMaxLoc(map)` | 3 | taught | `(min_val, max_val, min_loc, max_loc)` | locations are `(x, y)` |
| `cv2.StereoBM_create(numDisparities=, blockSize=)` | 3 | taught | block matcher | `numDisparities` a multiple of 16; `blockSize` odd. Banned in Weekly Projects 3 and 4 |
| `stereo.setMinDisparity`, `setUniquenessRatio`, `setSpeckleWindowSize`, `setSpeckleRange`, `setDisp12MaxDiff` | 3 | taught | search start; ambiguity check; speckle filter; left-right check | [chapter 3](03_stereo_vision.md#exercise-2-stereo-block-matching) has the measured effects |
| `stereo.compute(gray_left, gray_right)` | 3 | taught | disparity, `int16` in 1/16 px | divide by 16.0; invalid pixels = `minDisparity − 1` |
| `cv2.reprojectImageTo3D(disparity, Q)` | 3 | taught | a 3-D point per pixel | `disparity` as `float32` |
| `cv2.resize(img, size, interpolation=cv2.INTER_AREA)` | 3 | taught | shrinking without aliasing | `size` is `(w, h)` |
| `cv2.cvtColor(img, code)` with `COLOR_BGR2RGB`, `COLOR_BGR2GRAY`, `COLOR_GRAY2BGR` | 3 | taught | more conversion codes, shown in the Class 3 and 4 templates | `COLOR_BGR2RGB` was industry level in Class 1; it is taught from here on |
| `plt.gray()` | 3 | taught | gray as the default colour map | — |
| `plt.subplots(r, c, figsize=)`, `plt.colorbar` | 3 | taught | figure layout, colour scale | — |
| `np.savetxt`, `np.hstack` | 3 | taught | writes the PLY rows; joins arrays side by side | — |

### Calibration and two-view geometry (Class 4)

| Function | Class | Level | What it does | Parameters and pitfalls |
|---|---|---|---|---|
| `cv2.imread(path, 0)` | 4 | taught | reads an image as gray | — |
| `glob.glob('imgs/*.png')` | 4 | taught | list of file names | not sorted: use `sorted` to pair left and right |
| `np.mgrid[0:a, 0:b].T.reshape(-1, 2)` | 4 | taught | the (x, y) grid of the board's corners | — |
| `cv2.findChessboardCorners(gray, (nx, ny))` | 4 | taught | `(ret, corners)` for the **inner** corners; refines them internally | in this OpenCV version the shape is `(N, 2)`, so use `corners.reshape(-1, 2)` |
| `cv2.drawChessboardCorners(img, (nx, ny), corners, ret)` | 4 | taught | draws them | — |
| `cv2.calibrateCamera(obj, img, size, None, None)` | 4 | taught | `(rms, K, dist, rvecs, tvecs)` | `size` = `gray.shape[::-1]`; `dist` = k₁, k₂, p₁, p₂, k₃ |
| `cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (w, h))` | 4 | taught | camera matrix of the corrected image, and `roi` | alpha 1 keeps every pixel; alpha 0 leaves no black |
| `cv2.undistort(img, K, dist, None, newK)` | 4 | taught | removes the distortion | — |
| `cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)` | 4 | taught | `(F, mask)` | 3 px threshold and 1000 tries by default; mask = 1 for inliers |
| `cv2.computeCorrespondEpilines(pts, which_image, F)` | 4 | taught | line (a, b, c) in the other image, a² + b² = 1 | `which_image` is 1 or 2; `pts` shape `(N, 1, 2)` |
| `cv2.FlannBasedMatcher` | 4 | taught | approximate nearest neighbours | named by the optional exercise; not used yet |
| `np.linalg.svd`, `np.linalg.norm`, `np.trace` | 4 | taught | epipole, baseline, rotation angle | plain NumPy |
| `cv2.stereoCalibrate(obj, imgL, imgR, K1, d1, K2, d2, size, flags=cv2.CALIB_FIX_INTRINSIC)` | 4 | project | `(rms, K1, d1, K2, d2, R, T, E, F)` | `T` is in the units of `objp` |
| `cv2.stereoRectify(K1, d1, K2, d2, size, R, T, alpha=0)` | 4 | project | `(R1, R2, P1, P2, Q, roi1, roi2)` | `P2[0,3]` = −fB |
| `cv2.initUndistortRectifyMap(K, d, R1, P1, size, cv2.CV_32FC1)` | 4 | project | the lookup map, as two arrays | undistorts and rectifies in one map |
| `cv2.remap(img, map1, map2, cv2.INTER_LINEAR)` | 4 | project | applies the map | — |

### Point clouds and registration (Class 5)

| Function | Class | Level | What it does | Parameters and pitfalls |
|---|---|---|---|---|
| `o3d.io.read_point_cloud(path)`, `o3d.io.read_image(path)` | 5 | taught | reads a cloud (`.pcd`, `.ply`) or an image | — |
| `np.asarray(cloud.points)`, `.colors`, `.normals` | 5 | taught | the data as an N × 3 array | — |
| `cloud.voxel_down_sample(v)` | 5 | taught | one averaged point per cube of side v | returns a new cloud |
| `cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=, max_nn=))` | 5 | taught | normals from each point's neighbours (PCA) | the radius sets the scale: about 2 voxels; the sign is arbitrary |
| `o3d.pipelines.registration.compute_fpfh_feature(cloud, search_param)` | 5 | taught | a 33-value descriptor per point | needs normals; `.data` has shape (33, N) |
| `registration_ransac_based_on_feature_matching(a, b, fa, fb, True, dist, method, checkers=[...])` | 5 | taught | global registration | `mutual_filter` (the `True`) is required in this version; no `seed` argument |
| `TransformationEstimationPointToPoint(False)`, `TransformationEstimationPointToPlane()` | 5 | taught | the error measure | point-to-plane needs normals and a small rotation |
| `CorrespondenceCheckerBasedOnEdgeLength`, `CorrespondenceCheckerBasedOnDistance`, `CorrespondenceCheckerBasedOnNormal` | 5 | taught | reject samples early | much faster |
| `registration_icp(a, b, threshold, trans_init, method)` | 5 | taught | local registration | `.transformation`, `.fitness`, `.inlier_rmse` |
| `evaluate_registration(a, b, threshold, T)` | 5 | taught | fitness and inlier RMSE of a pose | use the same threshold when comparing |
| `o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=)` | 5 | taught | pairs the images | depth in mm, cut at 3 m by default |
| `o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera)` | 5 | taught | back-projects to 3-D | `PinholeCameraIntrinsicParameters.PrimeSenseDefault` |
| `cloud.transform(T)`, `a + b` | 5 | taught | moves the cloud in place; merges two clouds | — |
| `cloud.paint_uniform_color`, `o3d.visualization.draw_geometries`, `copy.deepcopy` | 5 | taught | viewing | the window blocks until it is closed |

### Regression and clustering (Class 6)

| Function | Class | Level | What it does | Parameters and pitfalls |
|---|---|---|---|---|
| `LinearRegression().fit(X, y)`, `.predict(X)` | 6 | taught | least squares | `X` has shape (N, K) |
| `np.linalg.inv`, `np.column_stack`, `@` | 6 | taught | OLS by hand | `(X.T @ X)` must be invertible |
| `linear_model.RANSACRegressor(residual_threshold=)`, `.inlier_mask_`, `.estimator_` | 6 | taught | a robust line | the default threshold is the data's spread: often too large |
| `KMeans(n_clusters=, init='random', n_init=, max_iter=, tol=, random_state=)`, `.fit_predict(X)` | 6 | taught | k-means | banned in Weekly Project 6 |
| `metrics.silhouette_score(X, labels)` | 6 | taught | quality of a clustering, −1 to 1 | slow on many points: `sample_size=` |
| `cloud.cluster_dbscan(eps=, min_points=)` | 6 | taught | DBSCAN labels; −1 = noise | very sensitive to `eps` |
| `o3d.geometry.TriangleMesh.create_box`, `create_tetrahedron`, `create_octahedron`, `create_icosahedron`, `create_torus`, `create_mobius`, `.sample_points_uniformly(n)`, `o3d.utility.Vector3dVector` | 6 | taught | test shapes | the templates' `create_moebius` is `create_mobius` here |
| `np.genfromtxt(path, delimiter=',')`, `np.deg2rad`, `np.linspace` | 6 | taught | reads and converts the lidar scan | remove the "no return" values |
| `plt.axes(projection='3d')`, `ax.scatter3D`, `ax.plot_surface`, `np.meshgrid` | 6 | taught | 3-D plots | — |

### State estimation (Class 7)

| Function | Class | Level | What it does | Parameters and pitfalls |
|---|---|---|---|---|
| `np.roll(p, shift, axis=)` | 7 | taught | an exact shift on a circular grid | — |
| `ndimage.convolve1d(q, kernel, mode='wrap')` | 7 | taught | 1-D uncertain motion | `ndimage.filters.convolve1d` no longer exists |
| `signal.convolve2d(q, kernel, boundary='wrap', mode='same')` | 7 | taught | 2-D uncertain motion | the weights must sum to 1 |
| `np.dot`, `np.transpose`, `np.linalg.pinv`, `np.eye` | 7 | taught | the Kalman equations | — |
| `plt.bar` | 7 | taught | a histogram of a belief | — |

### Classification (Class 8)

| Function | Class | Level | What it does | Parameters and pitfalls |
|---|---|---|---|---|
| `sk.datasets.load_iris()`, `pd.DataFrame(data, columns=)`, `.iloc[:, :2]` | 8 | taught | the Iris dataset as a table | — |
| `StandardScaler().fit_transform(X)` | 8 | taught | each feature to mean 0, standard deviation 1 | — |
| `train_test_split(X, y, test_size=, random_state=)` | 8 | taught | `(X_train, X_test, y_train, y_test)` | `random_state` fixes the split |
| `LinearSVC(penalty=, loss=, random_state=, max_iter=)` | 8 | taught | linear SVM, 1-vs-all | `max_iter` must be an integer |
| `svm.SVC(kernel=, C=, gamma=, degree=)` | 8 | taught | SVM with a kernel, 1-vs-1 | also reached as `sk.svm.SVC`; `'linear'`, `'rbf'`, `'poly'` |
| `clf.fit(X, y)`, `clf.score(X, y)`, `clf.predict(X)` | 8 | taught | train; accuracy; labels | — |
| `PCA(n_components=)`, `.fit_transform(X)`, `.transform(X)`, `.explained_variance_ratio_` | 8 | taught | principal components | fit on the training data only |
| `GridSearchCV(clf, param_grid=, cv=, n_jobs=, scoring=)`, `.best_params_`, `.best_score_` | 8 | taught | tries each value by cross-validation | `.score(X, y)` uses the best model |
| `Axes3D(fig, elev=, azim=)`, `fig.add_axes(ax)`, `ax.scatter(x, y, z, c=)` | 8 | taught | 3-D scatter plot | `add_axes` is needed in newer Matplotlib |
| `fetch_openml('mnist_784', version=1)` | 8 | taught | downloads MNIST | slow server; cached in `~/scikit_learn_data` |
| `metrics.accuracy_score(y, p)`, `metrics.confusion_matrix(y, p)` | 8 | taught | accuracy; counts per true and predicted class | the template imports `metrics` but names neither function |
| `plt.xticks`, `plt.yticks`, `plt.title`, `plt.tight_layout`, `plt.contourf` | 8 | taught | plot helpers from the templates | — |
| `KNeighborsClassifier(n_neighbors=)` | 8 | figures | k-NN | only used for the comparisons in chapter 8 |

### Visual odometry (Class 9)

| Function | Class | Level | What it does | Parameters and pitfalls |
|---|---|---|---|---|
| `cv2.solvePnPRansac(points_3d, points_2d, K, None)` | 9 | taught | PnP with RANSAC: `(ok, rvec, tvec, inliers)` | the pose maps world to camera; camera position = −Rᵀt; 8 px threshold by default |
| `cv2.Rodrigues(rvec)` | 9 | taught | axis-angle → rotation matrix: `(R, jacobian)` | not named by the template or the lecture text; the lecture covers axis-angle |
| `cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=, maxLevel=, criteria=)` | 9 | taught | Class 2's tracker with explicit settings | remove points with `status` 0 from all paired arrays |
| `bf.knnMatch(des1, des2, k=2)` | 9 | taught | the two best matches per descriptor, for the ratio test | `m.distance < 0.75 * n.distance`; `kp1[m.queryIdx].pt`, `kp2[m.trainIdx].pt` |
| `cv2.triangulatePoints(M_left, M_right, pts_left, pts_right)` | 9 | taught | 4 × N homogeneous 3-D points | points as 2 × N arrays; divide by the fourth row |
| `np.genfromtxt(file, delimiter=' ', dtype=None)` | 9 | taught | reads the true poses, 12 numbers per line | a pose is 3 × 4; the position is at indices 3, 7 and 11 |
| `cv2.solvePnP`, `cv2.findEssentialMat`, `cv2.recoverPose` | 9 | figures | PnP without RANSAC; the 2-D to 2-D case | only used to measure the comparisons in chapter 9 |

### Final project

| Function | Class | Level | What it does | Parameters and pitfalls |
|---|---|---|---|---|
| `cv2.findHomography(pts1, pts2, cv2.RANSAC, 1.0)` | 10 | project | removes wrong board corners by their mask | Class 2's figures showed it; homographies are taught in Class 4 |
| `cv2.stereoCalibrate`, `cv2.stereoRectify`, `cv2.initUndistortRectifyMap`, `cv2.remap` | 10 | project | as in Weekly Project 4 | `alpha` above 0.15 folds the map here |
| `linear_model.RANSACRegressor` with two input columns | 10 | taught | the floor plane Y = aX + bZ + c | `.estimator_.coef_`, `.intercept_` |
| `np.linalg.lstsq(A, b, rcond=None)` | 10 | taught | linear least squares, here for a circle | plain NumPy |
| `clf.decision_function(X)` | 10 | project | an SVM's score per window | only in `Approach tests.ipynb` |
| `cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)`, `.write(frame)`, `.release()` | 10 | project | saves the result video | not a perception method; `frame` must have exactly `size` |
| `cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)` | 10 | project | moves corners to sub-pixel positions, from the image gradients in a window | no course material shows it, so `USE_SUBPIX` is off; see [chapter 10](10_final_project.md#the-cornersubpix-decision) |

## Terms

| Term | Definition | Chapter |
|---|---|---|
| Affine transformation | p′ = A p + t; keeps straight lines straight and parallel lines parallel | [1](01_basic_image_processing.md#geometric-operations) |
| Aspect ratio | width / height of an image | [1](01_basic_image_processing.md#geometric-operations) |
| Bayer pattern | colour filter over the sensor: 50 % G, 25 % R, 25 % B; the missing values are interpolated | [1](01_basic_image_processing.md#digital-images) |
| Binary mask | image with only two values (0 / 255): not selected / selected | [1](01_basic_image_processing.md#masks-and-bitwise-operations) |
| Channel | one 2-D component of a multi-channel image (R, G, B or H, S, V) | [1](01_basic_image_processing.md#digital-images) |
| Closing | dilation followed by erosion: fills small holes and gaps | [1](01_basic_image_processing.md#morphology) |
| Connected component | maximal group of pixels linked through 4- or 8-neighbours with the same value | [1](01_basic_image_processing.md#connected-components-and-contours) |
| Connectivity (4 / 8) | which neighbours count as touching: up, down, left, right; or those plus the diagonals | [1](01_basic_image_processing.md#connected-components-and-contours) |
| Contour | ordered list of the border points of a region | [1](01_basic_image_processing.md#connected-components-and-contours) |
| Convolution | weighted sum with the kernel flipped: Σ w(u,v) F(i−u, j−v) | [1](01_basic_image_processing.md#linear-filtering) |
| Cross-correlation | weighted sum with the kernel as is: Σ w(u,v) F(i+u, j+v) | [1](01_basic_image_processing.md#linear-filtering) |
| Dilation | morphological maximum: a pixel becomes 1 if the SE touches any foreground pixel | [1](01_basic_image_processing.md#morphology) |
| Erosion | morphological minimum: a pixel stays 1 only if the SE fits inside the foreground | [1](01_basic_image_processing.md#morphology) |
| Foreground / background | the pixels of interest / the rest | [1](01_basic_image_processing.md#overview) |
| Gaussian filter | linear filter with Gaussian weights; separable; strength set by σ | [1](01_basic_image_processing.md#linear-filtering) |
| Histogram | number of pixels at each value | [1](01_basic_image_processing.md#non-linear-filters-and-thresholding) |
| HSV | hue (which colour, an angle), saturation (distance from gray), value (brightness) | [1](01_basic_image_processing.md#colour-spaces) |
| Hysteresis | two thresholds: strong pixels start an edge, weak pixels are kept only if they connect to one | [1](01_basic_image_processing.md#edges-and-inpainting) |
| Inpainting | filling a masked region with values estimated from its surroundings | [1](01_basic_image_processing.md#edges-and-inpainting) |
| Interpolation | computing a value between known samples (nearest, bilinear, …) | [1](01_basic_image_processing.md#geometric-operations) |
| Kernel (filter) | the weights of a linear filter, size (2k+1)² | [1](01_basic_image_processing.md#linear-filtering) |
| Linear filter | output = weighted sum of the neighbours, same weights everywhere | [1](01_basic_image_processing.md#linear-filtering) |
| Mean filter | linear filter with equal weights 1/(2k+1)² | [1](01_basic_image_processing.md#linear-filtering) |
| Median filter | non-linear: each pixel becomes the median of its neighbourhood | [1](01_basic_image_processing.md#non-linear-filters-and-thresholding) |
| Non-maximum suppression | keep a pixel only if it is the maximum along the gradient direction | [1](01_basic_image_processing.md#edges-and-inpainting) |
| Opening | erosion followed by dilation: removes small objects and thin bridges | [1](01_basic_image_processing.md#morphology) |
| Otsu's method | automatic threshold that minimises the within-class variance | [1](01_basic_image_processing.md#non-linear-filters-and-thresholding) |
| Pixel | one sample of the image grid | [1](01_basic_image_processing.md#digital-images) |
| Quantisation | rounding sample values to a fixed set of integers (0–255 for 8 bits) | [1](01_basic_image_processing.md#digital-images) |
| Rectification | g = max(f, 0) (ReLU) | [1](01_basic_image_processing.md#non-linear-filters-and-thresholding) |
| ROI | region of interest: the sub-array that is processed | [1](01_basic_image_processing.md#geometric-operations) |
| Salt-and-pepper noise | isolated pixels set to 0 or 255 | [1](01_basic_image_processing.md#non-linear-filters-and-thresholding) |
| Segmentation | dividing the pixels into object(s) and background | [1](01_basic_image_processing.md#overview) |
| Separable kernel | a 2-D kernel equal to the product of two 1-D kernels | [1](01_basic_image_processing.md#linear-filtering) |
| Sharpening (unsharp masking) | F + α (F − blur(F)) | [1](01_basic_image_processing.md#linear-filtering) |
| Structuring element | small binary shape that defines a morphological operation | [1](01_basic_image_processing.md#morphology) |
| Threshold | value T that splits the pixels into two classes | [1](01_basic_image_processing.md#non-linear-filters-and-thresholding) |
| Accumulator | grid of parameter cells in which the Hough transform counts votes | [2](02_feature_detection.md#fitting-a-model-with-outliers) |
| Aperture problem | one pixel only fixes the motion along the gradient; corners are needed for full motion | [2](02_feature_detection.md#optical-flow) |
| Brightness constancy | a moving point keeps its intensity: Iₓu + I_yv + I_t = 0 | [2](02_feature_detection.md#optical-flow) |
| Corner | point where the intensity changes in every direction (both eigenvalues of M large) | [2](02_feature_detection.md#corners-harris-and-shi-tomasi) |
| Descriptor | vector (or bit string) that describes the patch around a keypoint | [2](02_feature_detection.md#scale-space-dog-and-sift) |
| Difference of Gaussians (DoG) | L(kσ) − L(σ); approximates the scale-normalised LoG | [2](02_feature_detection.md#scale-space-dog-and-sift) |
| Edge | place of a sharp intensity change; large gradient magnitude | [2](02_feature_detection.md#edges) |
| Gradient | (∂I/∂x, ∂I/∂y): direction and size of the fastest intensity increase | [2](02_feature_detection.md#edges) |
| Hamming distance | number of differing bits between two binary descriptors | [2](02_feature_detection.md#binary-descriptors-orb) |
| Homography | 3 × 3 matrix mapping points of a plane between two views; 8 DOF, 4 point pairs | [2](02_feature_detection.md#matching) |
| Hough transform | model fitting by voting in a bounded parameter space | [2](02_feature_detection.md#fitting-a-model-with-outliers) |
| Inlier / outlier | a measurement that fits / does not fit the model within the threshold | [2](02_feature_detection.md#fitting-a-model-with-outliers) |
| kd-tree | tree that splits the descriptor space in halves, for fast nearest-neighbour search | [2](02_feature_detection.md#scale-space-dog-and-sift) |
| Keypoint (feature point) | distinctive image point that can be found again in other images | [2](02_feature_detection.md#corners-harris-and-shi-tomasi) |
| Laplacian of Gaussian (LoG) | second derivative of the smoothed image; zero crossings at edges; blob detector | [2](02_feature_detection.md#edges) |
| Least squares | fit that minimises the sum of squared residuals | [2](02_feature_detection.md#fitting-a-model-with-outliers) |
| Mutual check | keep a match only if it is the best in both directions | [2](02_feature_detection.md#matching) |
| Octave | group of scale-space levels at one image size; the image is halved between octaves | [2](02_feature_detection.md#scale-space-dog-and-sift) |
| Optical flow (sparse / dense) | apparent motion of image content: for selected points / for every pixel | [2](02_feature_detection.md#optical-flow) |
| Image pyramid | the image at successively halved sizes; used coarse to fine | [2](02_feature_detection.md#optical-flow) |
| RANSAC | fit minimal random samples, keep the model with the most inliers, refit | [2](02_feature_detection.md#fitting-a-model-with-outliers) |
| Ratio test | keep a match only if best distance / second-best distance is small (Lowe: < 0.8) | [2](02_feature_detection.md#matching) |
| Scale space | the image blurred at a range of σ, searched for the scale of each feature | [2](02_feature_detection.md#scale-space-dog-and-sift) |
| SIFT | DoG keypoints + orientation + 128-value gradient-histogram descriptor | [2](02_feature_detection.md#scale-space-dog-and-sift) |
| Sobel operator | 3 × 3 derivative kernels: derivative in one direction, smoothing in the other | [2](02_feature_detection.md#edges) |
| Structure tensor M | Σ [[Iₓ², IₓI_y], [IₓI_y, I_y²]] over a window; used by Harris, Shi-Tomasi and Lucas–Kanade | [2](02_feature_detection.md#corners-harris-and-shi-tomasi) |
| Baseline | the line connecting the two camera centres; its length is B | [3](03_stereo_vision.md#epipolar-geometry) |
| Disparity | x_L − x_R: the shift of a point between the two rectified images | [3](03_stereo_vision.md#depth-from-disparity) |
| Epipolar line | intersection of an epipolar plane with an image plane; the match of a point lies on it | [3](03_stereo_vision.md#epipolar-geometry) |
| Epipolar plane | plane through the two camera centres and a scene point | [3](03_stereo_vision.md#epipolar-geometry) |
| Epipole | where the baseline intersects an image plane; all epipolar lines pass through it | [3](03_stereo_vision.md#epipolar-geometry) |
| Global stereo method | minimises a data term plus a smoothness term over a row or the image | [3](03_stereo_vision.md#global-matching) |
| Local stereo method | decides each pixel from its own window | [3](03_stereo_vision.md#local-matching-windows-and-metrics) |
| NCC | normalised cross-correlation: Σ TI / √(Σ T² Σ I²) | [3](03_stereo_vision.md#local-matching-windows-and-metrics) |
| Rectification | reprojecting both images so that epipolar lines become the same rows | [3](03_stereo_vision.md#rectified-stereo), [4](04_calibration_and_epipolar.md#rectification) |
| SAD, SSD | sum of absolute / squared differences | [3](03_stereo_vision.md#local-matching-windows-and-metrics) |
| Template matching | a similarity score of a small image at every position of a larger one | [3](03_stereo_vision.md#exercise-1-template-matching) |
| Barrel / pincushion distortion | k₁ < 0: lines bend outwards / k₁ > 0: lines bend inwards | [4](04_calibration_and_epipolar.md#lens-distortion) |
| Camera matrix K | intrinsics: f_x, f_y, c_x, c_y, skew | [4](04_calibration_and_epipolar.md#camera-model) |
| DLT | Direct Linear Transform: stack the equations as Ap = 0, solve with SVD | [4](04_calibration_and_epipolar.md#calibration-with-known-3-d-points) |
| Essential matrix E | [T]× R: relates calibrated points; 5 DOF, rank 2 | [4](04_calibration_and_epipolar.md#the-essential-matrix) |
| Extrinsics | rotation R and translation T of the camera relative to the world | [4](04_calibration_and_epipolar.md#camera-model) |
| Fundamental matrix F | K⁻ᵀ E K′⁻¹: relates raw pixels; 7 DOF, rank 2 | [4](04_calibration_and_epipolar.md#the-fundamental-matrix) |
| Intrinsics | the camera's own parameters, K (and distortion) | [4](04_calibration_and_epipolar.md#camera-model) |
| Levenberg–Marquardt | nonlinear least squares, used to refine calibration | [4](04_calibration_and_epipolar.md#calibration-with-a-planar-pattern) |
| Projection matrix P | K[R \| T], 3 × 4, 11 DOF: maps world points to pixels | [4](04_calibration_and_epipolar.md#camera-model) |
| Reprojection error | distance between a detected point and where the model projects it | [4](04_calibration_and_epipolar.md#calibration-with-a-planar-pattern) |
| SVD | A = UDVᵀ; the last column of V solves Ax = 0 in the least-squares sense | [4](04_calibration_and_epipolar.md#calibration-with-known-3-d-points) |
| Fitness | share of source points with a target point within the threshold | [5](05_point_cloud_registration.md#local-alignment-icp) |
| FPFH | Fast Point Feature Histogram: a 33-value descriptor of the local surface | [5](05_point_cloud_registration.md#global-alignment-3-d-descriptors) |
| Global alignment | a rough pose from any start: features + RANSAC | [5](05_point_cloud_registration.md#overview) |
| ICP | Iterative Closest Point: alternate nearest points and pose | [5](05_point_cloud_registration.md#local-alignment-icp) |
| Inlier RMSE | error of the pairs within the threshold | [5](05_point_cloud_registration.md#local-alignment-icp) |
| Kabsch (Procrustes) | best rotation and translation for matched points, through the SVD | [5](05_point_cloud_registration.md#the-pose-from-matched-points-kabsch) |
| Local alignment | the exact pose from a good start: ICP | [5](05_point_cloud_registration.md#overview) |
| PFH | Point Feature Histogram: angles between all pairs in a neighbourhood | [5](05_point_cloud_registration.md#global-alignment-3-d-descriptors) |
| Point cloud | a list of 3-D points, sometimes with colours and normals | [5](05_point_cloud_registration.md#point-clouds-and-pose) |
| Point-to-plane / point-to-point | ICP error along the normal / between the points | [5](05_point_cloud_registration.md#local-alignment-icp) |
| Pose | rotation and translation, 6 DOF, a 4 × 4 matrix | [5](05_point_cloud_registration.md#point-clouds-and-pose) |
| Spin image | descriptor: a grid spun around the normal, counting neighbours | [5](05_point_cloud_registration.md#global-alignment-3-d-descriptors) |
| Surface normal | direction perpendicular to the surface; the least-variance direction of PCA | [5](05_point_cloud_registration.md#surface-normals) |
| Voxel downsampling | one averaged point per cube | [5](05_point_cloud_registration.md#exercise-1-global-registration) |
| Attraction basin | all starts whose mean shift reaches the same mode | [6](06_regression_and_clustering.md#clustering-mean-shift) |
| Core / border / noise point | DBSCAN's three point types | [6](06_regression_and_clustering.md#clustering-dbscan) |
| DBSCAN | density clustering with ε and MinPts | [6](06_regression_and_clustering.md#clustering-dbscan) |
| Distortion | k-means' cost: sum of squared distances to the centres | [6](06_regression_and_clustering.md#clustering-k-means) |
| Elbow method | choose k where the distortion stops falling quickly | [6](06_regression_and_clustering.md#clustering-k-means) |
| k-means | clustering by alternating assignment and centroids | [6](06_regression_and_clustering.md#clustering-k-means) |
| Mean shift | clustering by climbing the density to its modes | [6](06_regression_and_clustering.md#clustering-mean-shift) |
| MSE | mean squared error | [6](06_regression_and_clustering.md#regression) |
| OLS | ordinary least squares: β = (XᵀX)⁻¹Xᵀy | [6](06_regression_and_clustering.md#regression) |
| Regression | predicting a number from a weighted combination of inputs | [6](06_regression_and_clustering.md#regression) |
| Silhouette | (b − a) / max(a, b): how well a point fits its cluster | [6](06_regression_and_clustering.md#clustering-k-means) |
| Belief | probability distribution over the state | [7](07_state_estimation.md#overview) |
| Bayes' rule | posterior ∝ likelihood × prior | [7](07_state_estimation.md#the-histogram-filter) |
| Covariance matrix P | uncertainty of the state and the links between its components | [7](07_state_estimation.md#the-kalman-filter) |
| EKF | Kalman filter with local Jacobians of nonlinear models | [7](07_state_estimation.md#nonlinear-models-ekf-and-ukf) |
| Histogram filter | belief as a probability per grid cell | [7](07_state_estimation.md#the-histogram-filter) |
| Innovation | y = z − Hx′: measurement minus expected measurement | [7](07_state_estimation.md#the-kalman-filter) |
| Jacobian | matrix of the partial derivatives of a function at a point | [7](07_state_estimation.md#nonlinear-models-ekf-and-ukf) |
| Kalman filter | exact Gaussian filter for linear models | [7](07_state_estimation.md#the-kalman-filter) |
| Kalman gain K | how much of the innovation is applied | [7](07_state_estimation.md#the-kalman-filter) |
| Likelihood | probability of the measurement given a state | [7](07_state_estimation.md#the-histogram-filter) |
| Particle filter | belief as weighted samples: sample motion, weight, resample | [7](07_state_estimation.md#several-hypotheses-the-particle-filter) |
| Q, R | covariance of unmodelled motion / of the measurement noise | [7](07_state_estimation.md#the-kalman-filter) |
| State | the quantities being estimated | [7](07_state_estimation.md#overview) |
| Total probability | prediction: sum over all previous states | [7](07_state_estimation.md#the-histogram-filter) |
| UKF | sigma points through nonlinear models | [7](07_state_estimation.md#nonlinear-models-ekf-and-ukf) |
| Background subtraction | finding what differs from an image of the empty scene | [10](10_final_project.md#detecting-the-robot) |
| Gate (Kalman filter) | rejecting a measurement that is too far from the prediction, by Mahalanobis distance | [10](10_final_project.md#tracking-in-2-d-and-3-d) |
| Hidden-detection test | hiding known detections from a filter, and comparing its prediction with them | [10](10_final_project.md#tests-without-ground-truth) |
| Visible edge | using the unhidden edge of a cut-off box, plus half the usual width, as the position | [10](10_final_project.md#tracking-in-2-d-and-3-d) |
| 1-vs-all, 1-vs-1 | multi-class SVMs: one classifier per class, or per pair of classes | [8](08_classification.md#support-vector-machines) |
| Classification | predicting a nominal label from labelled examples | [8](08_classification.md#what-classification-is) |
| Confusion matrix | counts of each true class against each predicted class | [8](08_classification.md#weekly-project-handwritten-digits) |
| Cross-validation | training on k − 1 parts and scoring on the last, in turn | [8](08_classification.md#exercise-2-pca-and-grid-search) |
| Curse of dimensionality | too many dimensions make the data sparse | [8](08_classification.md#dimensionality-reduction-with-pca) |
| Decision boundary | the surface between classes in feature space | [8](08_classification.md#what-classification-is) |
| Explained variance | a component's eigenvalue as a share of all eigenvalues | [8](08_classification.md#dimensionality-reduction-with-pca) |
| Feature vector | the measurable properties of one sample | [8](08_classification.md#what-classification-is) |
| Grid search | trying many parameter values by cross-validation | [8](08_classification.md#exercise-2-pca-and-grid-search) |
| HOG | histogram of oriented gradients; the features the lecture pairs with an SVM | [8](08_classification.md#object-detection) |
| Intersection over union (IoU) | overlap area divided by union area of two boxes | [8](08_classification.md#object-detection) |
| k-nearest neighbours (k-NN) | a vote among the k nearest training samples | [8](08_classification.md#k-nearest-neighbours) |
| Kernel trick | a kernel function replaces working in a higher-dimensional space | [8](08_classification.md#support-vector-machines) |
| Margin | the distance between the boundary and the nearest samples | [8](08_classification.md#support-vector-machines) |
| Non-maximum suppression (NMS) | keeps the best box and removes the ones that overlap it | [8](08_classification.md#object-detection) |
| Normalization | scaling features to mean 0 and standard deviation 1 | [8](08_classification.md#exercise-1-svms-on-iris) |
| Overfitting | learning the training samples without a rule for new ones | [8](08_classification.md#what-classification-is) |
| Principal component analysis (PCA) | keeps the eigenvectors of the covariance with the largest eigenvalues | [8](08_classification.md#dimensionality-reduction-with-pca) |
| Sliding window | classifying a fixed-size window at many positions and scales | [8](08_classification.md#object-detection) |
| Support vector | a sample on the margin; these define the SVM's boundary | [8](08_classification.md#support-vector-machines) |
| Support vector machine (SVM) | a classifier that maximises the margin | [8](08_classification.md#support-vector-machines) |
| Axis-angle | a rotation as a vector: its direction is the axis, its length the angle | [9](09_visual_odometry.md#orientation) |
| Bundle adjustment | optimising poses and 3-D points together over several frames | [9](09_visual_odometry.md#drift-and-bundle-adjustment) |
| Drift | the growing error from chaining relative poses | [9](09_visual_odometry.md#drift-and-bundle-adjustment) |
| Euler angles | a rotation as three angles about successive axes; has gimbal lock | [9](09_visual_odometry.md#orientation) |
| Gimbal lock | two Euler axes align, and one degree of freedom is lost | [9](09_visual_odometry.md#orientation) |
| P3P | PnP from 3 points; up to 4 solutions | [9](09_visual_odometry.md#relative-pose-from-3-d-to-2-d-pnp) |
| PnP | perspective from n points: the pose from known 3-D points and their 2-D projections | [9](09_visual_odometry.md#relative-pose-from-3-d-to-2-d-pnp) |
| Quaternion | a rotation as 4 numbers; no gimbal lock; smooth interpolation | [9](09_visual_odometry.md#orientation) |
| Scale ambiguity | one moving camera gives geometry only up to an unknown scale | [9](09_visual_odometry.md#relative-pose-from-2-d-to-2-d) |
| Triangulation | finding a 3-D point from its image positions in two views | [9](09_visual_odometry.md#weekly-project-stereo-visual-odometry) |
| Visual odometry (VO) | estimating a camera's pose over time from the apparent motion of a static world | [9](09_visual_odometry.md#what-visual-odometry-is) |
| Visual-inertial odometry (VIO) | VO combined with an IMU; loosely or tightly coupled | [9](09_visual_odometry.md#visual-inertial-odometry) |
