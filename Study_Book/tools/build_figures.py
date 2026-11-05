"""Rebuild the figures of the study book.

Run from the workspace root with the course venv:
    .venv/bin/python Study_Book/tools/build_figures.py       # every chapter
    .venv/bin/python Study_Book/tools/build_figures.py 1     # chapter 1 only

Two kinds of figures are made:
- exercise results: the notebook is executed in memory (the .ipynb on disk is
  never written) and the image output of the chosen cells is saved as a PNG;
- explanatory figures: drawn here from the exercise images or from small
  synthetic examples.
The script also prints the numbers quoted in the chapters.
"""
import sys
import base64
import pathlib
import warnings

import cv2
import numpy as np
import imutils
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIGURES = ROOT / "Study_Book" / "figures"
EX1 = ROOT / "Class_1_weekly_project" / "Exercises_week_1"
EX2 = ROOT / "Class_2_feature_detection_and_description" / "Exercises_week_2"
WP2 = ROOT / "Class_2_weekly_project"
EX3 = ROOT / "Class_3_multiple_view_geometry_1" / "Exercise week 3"
WP3 = ROOT / "Class_3_weekly_project" / "Weekly project"
EX4 = ROOT / "Class_4_epipolar_geom_2_and_calibration" / "Exercises_week_4" / "Exercises_week_4"
WP4 = ROOT / "Class_4_weekly_project"

# reference chart palette (light surface); categorical slots in fixed order
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CHANNEL = {"R": "#e34948", "G": "#008300", "B": "#2a78d6"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "savefig.dpi": 100, "savefig.bbox": "tight",
    "text.color": INK, "axes.titlecolor": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "lines.linewidth": 2, "legend.frameon": False,
})

# first cell added to every executed notebook: fixed resolution, opaque background
SETUP_CELL = (
    "%matplotlib inline\n"
    "from IPython.core.pylabtools import select_figure_formats\n"
    "select_figure_formats(get_ipython(), {'png'}, bbox_inches='tight', dpi=100, facecolor='white')\n"
)


# ---------------------------------------------------------------- helpers
def rgb(path):
    """Read an image like the tutorials do: BGR from disk, channels swapped to RGB."""
    b, g, r = cv2.split(cv2.imread(str(path)))
    return cv2.merge([r, g, b])


def show(ax, img, title):
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")


def report(path):
    head = path.read_bytes()[:24]
    w, h = int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
    print(f"  saved {path.relative_to(ROOT)}  ({w}x{h} px, {path.stat().st_size // 1024} KB)")


def save(fig, chapter, name):
    path = FIGURES / f"class{chapter}" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    report(path)


def external_contours(mask):
    return imutils.grab_contours(cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))


def label_lines(ax, x, values, names, gap):
    """Direct labels at column x, pushed apart vertically so they don't overlap."""
    order = np.argsort(values)
    placed = []
    for i in order:
        y = values[i] if not placed else max(values[i], placed[-1] + gap)
        placed.append(y)
        ax.text(x, y, " " + names[i], color=INK, va="center", ha="left", fontsize=10,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.5, alpha=0.8))


# ---------------------------------------------------------------- notebook outputs
def first_image(cell, n=0):
    images = [out["data"]["image/png"] for out in cell.get("outputs", []) if "image/png" in out.get("data", {})]
    return images[n] if len(images) > n else None


def cell_image(nb, key):
    """Image of a cell, found by key:
    'code:<text>'  first code cell whose source contains <text>;
    '<heading>'    first image in the code cells after the markdown cell that starts with
                   <heading>, up to the next markdown heading.
    A suffix '@n' takes the n-th image of that cell instead of the first."""
    key, _, suffix = key.partition("@")
    n = int(suffix) if suffix else 0
    if key.startswith("code:"):
        for cell in nb.cells:
            if cell.cell_type == "code" and key[5:] in "".join(cell.source):
                return first_image(cell, n)
        return None
    start = None
    for i, cell in enumerate(nb.cells):
        src = "".join(cell.source)
        if cell.cell_type == "markdown" and (src.strip() == key or src.startswith(key + " ") or src.startswith(key + "\n")):
            start = i
            break
    if start is None:
        return None
    for cell in nb.cells[start + 1:]:
        if cell.cell_type == "markdown" and "".join(cell.source).startswith("#"):
            return None
        if cell.cell_type == "code" and first_image(cell, n) is not None:
            return first_image(cell, n)
    return None


def notebook_figures(chapter, notebooks):
    for entry in notebooks:
        path, wanted = entry[0], entry[1]
        replacements = entry[2] if len(entry) > 2 else {}
        print(f"  executing {path.relative_to(ROOT)} (in memory)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nb = nbformat.read(path, as_version=4)
        for cell in nb.cells:
            for old, new in replacements.items():
                cell.source = cell.source.replace(old, new)
        nb.cells.insert(0, nbformat.v4.new_code_cell(SETUP_CELL))
        NotebookClient(nb, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": str(path.parent)}}).execute()
        for cell in nb.cells:                              # printed results, for the chapter text
            for out in cell.get("outputs", []):
                if out.get("output_type") == "stream":
                    print("    | " + "".join(out["text"]).strip().replace("\n", "\n    | "))
        for heading, name in wanted:
            png = cell_image(nb, heading)
            if png is None:
                print(f"  WARNING: no image under '{heading}' in {path.name}")
                continue
            out = FIGURES / f"class{chapter}" / f"{name}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(base64.b64decode(png))
            report(out)


# ---------------------------------------------------------------- chapter 1
CH1_NOTEBOOKS = [
    (EX1 / "Exercise 2.ipynb", [
        ("## 1.1", "ex2_1-1_yellow_rgb_range"),
        ("## 1.2", "ex2_1-2_yellow_cluster_size"),
        ("## 1.3", "ex2_1-3_yellow_industry"),
        ("## 2.1", "ex2_2-1_spheres_tutorial"),
        ("## 4.1", "ex2_4-1_red_rgb_range"),
        ("## 4.2", "ex2_4-2_red_cluster"),
        ("## 4.3", "ex2_4-3_red_industry"),
        ("## 5.1", "ex2_5-1_green_rgb_range"),
        ("## 5.2", "ex2_5-2_green_cluster"),
        ("## 5.3", "ex2_5-3_green_industry"),
    ]),
]
CROP = (slice(40, 240), slice(270, 470))      # 200 x 200 px around the face in jp.png


def ch1_bgr_vs_rgb():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    show(axes[0], cv2.imread(str(EX1 / "jp.png")), "BGR array passed to plt.imshow")
    show(axes[1], rgb(EX1 / "jp.png"), "after split + merge([r, g, b])")
    save(fig, 1, "concept_bgr_vs_rgb")


def ch1_channels():
    image = rgb(EX1 / "jp.png")
    fig, axes = plt.subplots(1, 4, figsize=(17, 2.7))
    show(axes[0], image, "colour image, shape h x w x 3")
    for i, name in enumerate("RGB"):
        show(axes[i + 1], image[:, :, i], f"{name}(x, y) = image[:, :, {i}]")
    save(fig, 1, "concept_channels")


def ch1_geometry():
    image = rgb(EX1 / "jp.png")
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), -45, 1.0)
    panels = [(image, "original"),
              (image[60:160, 320:420], "ROI image[60:160, 320:420]"),
              (cv2.resize(image, (200, 200)), "resize(image, (200, 200))"),
              (cv2.warpAffine(image, M, (w, h)), "warpAffine, angle -45"),
              (imutils.rotate_bound(image, 45), "rotate_bound, angle 45")]
    ratios = [p.shape[1] / p.shape[0] for p, _ in panels]
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.1), gridspec_kw={"width_ratios": ratios})
    for ax, (p, title) in zip(axes, panels):
        show(ax, p, f"{title}\n{p.shape[1]} x {p.shape[0]} px")
    save(fig, 1, "concept_geometry")


def ch1_gaussian_blur():
    image = rgb(EX1 / "jp.png")
    sizes = [3, 11, 31]
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.6), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.5]})
    show(axes[0], image[CROP], "original (crop)")
    for ax, k in zip(axes[1:4], sizes):
        sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        show(ax, cv2.GaussianBlur(image, (k, k), 0)[CROP], f"GaussianBlur, ({k}, {k}), sigma {sigma:.1f}")
    ax = axes[4]
    for k, colour in zip(sizes, SERIES):
        weights = cv2.getGaussianKernel(k, 0).ravel()
        offsets = np.arange(k) - k // 2
        ax.plot(offsets, weights, color=colour, marker="o" if k < 31 else None, markersize=6,
                label=f"{k} x {k}")
        ax.text(1, weights.max(), f"  {k} x {k}", color=INK, va="center")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_xlabel("offset from the centre pixel")
    ax.set_ylabel("weight")
    ax.set_title("1-D kernel weights (each sums to 1)")
    ax.legend(loc="upper right")
    save(fig, 1, "concept_gaussian_blur")


def ch1_sharpening():
    gray = cv2.cvtColor(rgb(EX1 / "jp.png"), cv2.COLOR_RGB2GRAY).astype(np.float32)
    blurred = cv2.blur(gray, (5, 5))                     # mean filter, as on the slides
    detail = gray - blurred
    sharp = np.clip(gray + detail, 0, 255)               # alpha = 1: 2*original - blurred
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.1))
    show(axes[0], gray[CROP], "original f")
    show(axes[1], blurred[CROP], "blurred: mean 5 x 5")
    axes[2].imshow(detail[CROP], cmap="gray", vmin=-40, vmax=40)
    axes[2].set_title("detail = f - blurred (mid-grey = 0)")
    axes[2].axis("off")
    show(axes[3], sharp[CROP], "sharpened = f + 1 x detail")
    save(fig, 1, "concept_sharpening")


def ch1_mean_vs_median():
    gray = cv2.cvtColor(rgb(EX1 / "jp.png"), cv2.COLOR_RGB2GRAY)
    noise = np.random.default_rng(0).random(gray.shape)
    noisy = gray.copy()
    noisy[noise < 0.04] = 0
    noisy[noise > 0.96] = 255
    panels = [(gray, "original"), (noisy, "8 % salt-and-pepper noise"),
              (cv2.blur(noisy, (3, 3)), "mean 3 x 3"),
              (cv2.GaussianBlur(noisy, (3, 3), 0), "Gaussian 3 x 3"),
              (cv2.medianBlur(noisy, 3), "median 3 x 3")]
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.1))
    for ax, (p, title) in zip(axes, panels):
        show(ax, p[CROP], title)
    save(fig, 1, "concept_mean_vs_median")
    for p, title in panels[1:]:
        err = np.abs(p.astype(int) - gray.astype(int)).mean()
        print(f"  mean |error| vs original, {title}: {err:.1f} gray levels")


def ch1_morphology():
    img = np.zeros((150, 300), np.uint8)
    cv2.rectangle(img, (20, 25), (120, 125), 255, -1)           # square
    cv2.circle(img, (225, 75), 52, 255, -1)                       # disk
    cv2.rectangle(img, (121, 73), (172, 76), 255, -1)             # 4-px thick bridge between them
    for x, y in [(45, 50), (85, 95), (95, 45), (210, 55), (240, 95)]:
        cv2.rectangle(img, (x, y), (x + 3, y + 3), 0, -1)         # 4 x 4 holes
    rng = np.random.default_rng(1)
    for x, y in zip(rng.integers(5, 292, 60), rng.integers(5, 142, 60)):
        if img[max(y - 5, 0):y + 8, max(x - 5, 0):x + 8].max() == 0:
            cv2.rectangle(img, (int(x), int(y)), (int(x) + 2, int(y) + 2), 255, -1)   # 3 x 3 specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    results = [("original", img),
               ("erosion", cv2.erode(img, kernel)),
               ("dilation", cv2.dilate(img, kernel)),
               ("opening = erosion, then dilation", cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)),
               ("closing = dilation, then erosion", cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel))]
    fig, axes = plt.subplots(2, 3, figsize=(15, 5.6))
    se = axes[0, 0]
    se.imshow(kernel * 255, cmap="gray", vmin=0, vmax=255)
    se.set_xticks(np.arange(-0.5, 7, 1), minor=True)
    se.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    se.grid(which="minor", color=MUTED, linewidth=0.8)
    se.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for spine in se.spines.values():
        spine.set_visible(False)
    se.set_title("structuring element: 7 x 7 ellipse")
    # orange line = outline of the original, drawn over every result
    for ax, (title, res) in zip([axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]], results):
        show(ax, res, title)
        if title != "original":
            ax.contour(img > 0, levels=[0.5], colors=[SERIES[1]], linewidths=0.9)
    save(fig, 1, "concept_morphology")
    for title, res in results:
        n_obj = cv2.connectedComponents(res, connectivity=8)[0] - 1
        n_holes = cv2.connectedComponents(255 - res, connectivity=4)[0] - 2
        print(f"  morphology {title}: {n_obj} objects, {n_holes} holes, {int(np.sum(res == 255))} px")


def ch1_connectivity():
    grid = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 0, 0, 0, 0, 0],
                     [0, 1, 1, 0, 0, 0, 1, 0],
                     [0, 0, 0, 1, 1, 0, 1, 0],
                     [0, 0, 0, 1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0]], np.uint8)
    n4, lab4 = cv2.connectedComponents(grid, connectivity=4)
    n8, lab8 = cv2.connectedComponents(grid, connectivity=8)
    # colour follows the object: an 8-component takes the colour of the 4-component of its first pixel
    colour8 = {}
    for k in range(1, n8):
        ys, xs = np.nonzero(lab8 == k)
        colour8[k] = lab4[ys[0], xs[0]]

    def cells(ax, labels, colour_of, title):
        rows, cols = labels.shape
        for y in range(rows):
            for x in range(cols):
                k = labels[y, x]
                face = SURFACE if k == 0 else SERIES[(colour_of(k) - 1) % len(SERIES)]
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor=face, edgecolor=GRID, linewidth=1))
                if k:
                    ink = INK if face == SERIES[3] else "#ffffff"
                    ax.text(x + 0.5, y + 0.5, str(k), color=ink, ha="center", va="center", fontsize=11)
        ax.set_xlim(0, cols)
        ax.set_ylim(rows, 0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title)

    def stencil(ax, offsets, title):
        for y in range(3):
            for x in range(3):
                if (y, x) == (1, 1):
                    face, text = INK2, "p"
                elif (y - 1, x - 1) in offsets:
                    face, text = SERIES[0], ""
                else:
                    face, text = SURFACE, ""
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor=face, edgecolor=GRID, linewidth=1))
                if text:
                    ax.text(x + 0.5, y + 0.5, text, color="#ffffff", ha="center", va="center", fontsize=12)
        ax.set_xlim(0, 3)
        ax.set_ylim(3, 0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title)

    n4_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    n8_offsets = n4_offsets + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), gridspec_kw={"width_ratios": [1, 1, 2.2, 2.2]})
    stencil(axes[0], n4_offsets, "4-neighbours of p")
    stencil(axes[1], n8_offsets, "8-neighbours of p")
    cells(axes[2], lab4, lambda k: k, f"4-connectivity: {n4 - 1} components")
    cells(axes[3], lab8, lambda k: colour8[k], f"8-connectivity: {n8 - 1} components")
    save(fig, 1, "concept_connectivity")


def ch1_tetris_tutorial():
    image = rgb(EX1 / "tetris_blocks.png")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edged = cv2.Canny(gray, 30, 150)
    thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
    cnts = external_contours(thresh)
    output = image.copy()
    for c in cnts:
        cv2.drawContours(output, [c], -1, (0, 0, 0), 3)
    text = "I found {} objects!".format(len(cnts))
    cv2.putText(output, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (155, 0, 155), 2)
    panels = [(image, "input"), (gray, "cvtColor RGB2GRAY"), (edged, "Canny(gray, 30, 150)"),
              (thresh, "threshold 225, BINARY_INV"), (output, "findContours + drawContours"),
              (cv2.erode(thresh, None, iterations=5), "erode, 5 iterations"),
              (cv2.dilate(thresh, None, iterations=5), "dilate, 5 iterations"),
              (cv2.bitwise_and(image, image, mask=thresh), "bitwise_and(image, image, mask)")]
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.2))
    for ax, (p, title) in zip(axes.ravel(), panels):
        show(ax, p, title)
    save(fig, 1, "ex2_tutorial_tetris")
    yellow = (image[:, :, 0] > 200) & (image[:, :, 1] > 200) & (image[:, :, 2] <= 100)
    print(f"  tetris: {len(cnts)} objects; yellow piece mean RGB {image[yellow].mean(axis=0).round()}, "
          f"mean gray {gray[yellow].mean():.0f}; background gray {np.median(gray[thresh == 0]):.0f}")


def spheres_cropped():
    return rgb(EX1 / "spheres.png")[1:-1, 1:-1]


def ch1_threshold_choice():
    image = spheres_cropped()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[0]
    counts = np.bincount(gray.ravel(), minlength=256).astype(float)
    counts[counts == 0] = np.nan
    x = np.arange(256)
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.1], hspace=0.45)
    ax_full = fig.add_subplot(gs[0, :3])
    ax_zoom = fig.add_subplot(gs[0, 3], sharey=ax_full)
    lines = [(otsu, ":", f"Otsu: T = {otsu:.0f}"), (225, "--", "tutorial: T = 225"), (245, "-", "suggested: T = 245")]
    for ax, (lo, hi) in [(ax_full, (0, 255)), (ax_zoom, (200, 255))]:
        ax.fill_between(x, np.nan_to_num(counts, nan=0.0), step="mid", color=SERIES[0], alpha=0.3, linewidth=0)
        ax.step(x, counts, where="mid", color=SERIES[0], linewidth=1.2)
        ax.set_yscale("log")
        ax.set_xlim(lo - 0.5, hi + 0.5)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_xlabel("gray level")
        for t, style, label in lines:
            if lo <= t <= hi:
                ax.axvline(t, color=INK, linestyle=style, linewidth=1.3)
                if ax is ax_full:
                    ax.text(t - 1.5, np.nanmax(counts), label, rotation=90, va="top", ha="right", color=INK)
    ax_full.set_ylim(0.8, np.nanmax(counts) * 3)
    ax_full.set_ylabel("number of pixels (log scale)")
    ax_full.set_title("histogram of the gray image (frame cropped)")
    ax_zoom.set_title("zoom: gray 200-255")
    ax_zoom.text(254, np.nanmax(counts), "background\n(255)", ha="right", va="top", color=INK2)
    print(f"  spheres: Otsu threshold = {otsu:.0f}")
    for i, t in enumerate([otsu, 225, 245, 252]):
        thresh = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)[1]
        cnts = external_contours(thresh)
        big = [c for c in cnts if cv2.contourArea(c) > 1000]
        ax = fig.add_subplot(gs[1, i])
        name = "Otsu" if i == 0 else ""
        show(ax, thresh, f"T = {t:.0f} {name}\n{len(big)} objects > 1000 px, {len(cnts) - len(big)} specks")
    for t in [otsu, 225, 235, 240, 245, 248, 250, 252, 254]:
        thresh = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)[1]
        cnts = external_contours(thresh)
        areas = sorted(int(cv2.contourArea(c)) for c in cnts if cv2.contourArea(c) > 1000)
        print(f"  spheres T={t:.0f}: {len(areas)} objects > 1000 px (contour areas {areas[:1]}..{areas[-1:]}), "
              f"{len(cnts) - len(areas)} small, foreground {int(np.sum(thresh == 255))} px")
    save(fig, 1, "ex2_3-1_threshold_choice")


def ch1_border_frame():
    image = rgb(EX1 / "spheres.png")
    gray_full = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    print(f"  spheres: size {image.shape[1]} x {image.shape[0]}, gray values on the outer frame "
          f"{sorted(set(gray_full[0].tolist()) | set(gray_full[:, 0].tolist()))[:5]}, "
          f"just inside {sorted(set(gray_full[1, 1:-1].tolist()))[:5]}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (name, img) in zip(axes, [("full image", image), ("image[1:-1, 1:-1]", image[1:-1, 1:-1])]):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)[1]
        cnts = external_contours(thresh)
        out = img.copy()
        cv2.drawContours(out, cnts, -1, (0, 0, 0), 3)
        show(ax, out, f"{name}, T = 245: {len(cnts)} external contour(s)")
    save(fig, 1, "ex2_3-1_border_frame")


def ch1_rgb_vs_hsv():
    image = spheres_cropped()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    r, g, b = [image[:, :, i].astype(int) for i in range(3)]
    h, s, v = [hsv[:, :, i].astype(int) for i in range(3)]
    rgb_rule = (r > 130) & (g <= 80) & (b > 20) & (b <= 130)              # limits of cell 4.1
    crimson = (h >= 165) & (h <= 179) & (s >= 70) & (v >= 50)              # first band of cell 4.3
    hsv_rule = crimson | ((h <= 10) & (s >= 70) & (v >= 50))               # both bands of cell 4.3
    fg = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)[1]
    best_n, disk = -1, None
    for c in external_contours(fg):
        filled = np.zeros_like(gray)
        cv2.drawContours(filled, [c], -1, 255, -1)
        n = int(np.sum(crimson & (filled == 255)))
        if n > best_n:
            best_n, disk = n, filled == 255
    ys, xs_disk = np.nonzero(disk)
    cy, cx = ys.mean(), xs_disk.mean()
    radius = np.sqrt(disk.sum() / np.pi)
    yy, xx = np.indices(gray.shape)
    inner = disk & ((yy - cy) ** 2 + (xx - cx) ** 2 < (0.85 * radius) ** 2)      # away from the rim
    low_hue = disk & (h <= 10) & (s >= 70) & (v >= 50)                          # red past the 0/180 wrap
    y_a = int(np.argmax((disk & crimson).sum(axis=1)))                          # row with most H 165-179
    y_b = int(np.argmax(low_hue.sum(axis=1)))                                   # row with most H 0-10
    x0, x1 = max(int(cx - radius) - 25, 0), min(int(cx + radius) + 25, gray.shape[1] - 1)
    y0, y1 = max(int(cy - radius) - 25, 0), min(int(cy + radius) + 25, gray.shape[0] - 1)
    xs = np.arange(x0, x1 + 1)
    yb_, xb_ = np.unravel_index(np.argmax(np.where(inner, gray, 0)), gray.shape)
    print(f"  red sphere: {int(disk.sum())} px, radius ~{radius:.0f} px, centre row {cy:.0f}; selected by the "
          f"RGB limits {rgb_rule[disk].mean():.0%}, by the two HSV bands {hsv_rule[disk].mean():.0%}, "
          f"H 0-10 {low_hue[disk].mean():.0%}; brightest inner pixel gray {gray[yb_, xb_]} at row {yb_}, "
          f"S there {s[yb_, xb_]}; inner pixels with S < 70: {np.mean(s[inner] < 70):.1%}")

    fig = plt.figure(figsize=(18, 8.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.7, 1.7], hspace=0.6)
    ax_img = fig.add_subplot(gs[:, 0])
    show(ax_img, image[y0:y1 + 1, x0:x1 + 1], "red sphere and the two profile rows")
    for y, name in [(y_a, "a"), (y_b, "b")]:
        ax_img.plot([0, x1 - x0], [y - y0, y - y0], color=INK, linewidth=1)
        ax_img.text(3, y - y0 - 3, name, color=INK, fontsize=12, va="bottom",
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
    rows = [(y_a, "row a (most H 165-179)"), (y_b, "row b (most H 0-10)")]
    for i, (y, row_name) in enumerate(rows):
        cols = np.nonzero(disk[y])[0]
        x_label = cols.min() + (cols.max() - cols.min()) // 4
        inside = slice(cols.min(), cols.max() + 1)
        print(f"  {row_name}, y = {y}: inside the sphere median H {np.median(h[y, inside]):.0f}, "
              f"median S {np.median(s[y, inside]):.0f}; RGB limits pass {rgb_rule[y, inside].mean():.0%}, "
              f"crimson band alone {crimson[y, inside].mean():.0%}, both HSV bands {hsv_rule[y, inside].mean():.0%}")
        for j, (rule, rule_name, series, what) in enumerate([
                (rgb_rule, "selected by the RGB limits of 4.1",
                 [("R", r, CHANNEL["R"], "-"), ("G", g, CHANNEL["G"], "--"), ("B", b, CHANNEL["B"], ":")],
                 "R, G, B"),
                (hsv_rule, "selected by the two HSV bands of 4.3",
                 [("H", h, INK, "-"), ("S", s, INK2, "--"), ("V", v, INK2, ":")],
                 "H, S, V")]):
            ax = fig.add_subplot(gs[i, j + 1])
            ax.fill_between(xs, 0, 260, where=rule[y, xs], step="mid", color=GRID, linewidth=0, label=rule_name)
            for name, channel, colour, style in series:
                ax.plot(xs, channel[y, xs], color=colour, linestyle=style, label=name)
            label_lines(ax, x_label, [channel[y, x_label] for _, channel, _, _ in series],
                        [name for name, _, _, _ in series], gap=18)
            ax.set_ylim(0, 260)
            ax.set_xlim(x0, x1)
            ax.set_xlabel("x (pixel column)")
            ax.set_ylabel("8-bit value")
            ax.set_title(f"{what} along {row_name}")
            if i == 1:
                ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.3))
    save(fig, 1, "ex2_4_rgb_vs_hsv_profile")


def ch1_hue_bands():
    image = spheres_cropped()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    fg = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)[1]
    spheres = []
    for c in external_contours(fg):
        if cv2.contourArea(c) < 1000:
            continue
        filled = np.zeros_like(gray)
        cv2.drawContours(filled, [c], -1, 255, -1)
        sel = (filled == 255) & (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 50)
        x, y = c[0][0]
        if sel.sum() < 500:
            print(f"  sphere at ({x}, {y}): no saturated pixels (grey sphere)")
            continue
        hue = int(np.median(hsv[:, :, 0][sel]))
        colour = np.median(image[sel], axis=0)
        spheres.append((hue, colour / 255))
        print(f"  sphere at ({x}, {y}): median H = {hue}, median RGB = {colour.astype(int).tolist()}")
    hues = np.arange(180, dtype=np.uint8)
    bar = cv2.cvtColor(np.dstack([hues, np.full(180, 255, np.uint8), np.full(180, 255, np.uint8)]),
                       cv2.COLOR_HSV2RGB)
    fig, ax = plt.subplots(figsize=(15, 3.6))
    ax.imshow(bar, extent=(-0.5, 179.5, 0, 1), aspect="auto")
    bands = [(25, 35, "1.3 yellow\n(tetris)", "-"), (40, 65, "5.3 green", "-"),
             (165, 179, "4.3 red,\ncrimson side", "-"), (0, 10, "4.3 red,\nother side", "-")]
    for lo, hi, label, style in bands:
        ax.add_patch(Rectangle((lo - 0.5, 0), hi - lo + 1, 1, fill=False, edgecolor=INK,
                               linewidth=1.8, linestyle=style))
        ax.text((lo + hi) / 2, 1.08, f"{label}\nH {lo}-{hi}", ha="center", va="bottom", color=INK)
    previous = -10
    level = 0
    for hue, colour in sorted(spheres):
        level = 1 - level if hue - previous < 6 else 0
        previous = hue
        ax.scatter([hue], [-0.35], s=170, color=[colour], edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.text(hue, -0.7 - 0.3 * level, str(hue), ha="center", va="top", color=INK)
    ticks = np.arange(0, 181, 15)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}\n({2 * t} deg)" for t in ticks])
    ax.set_xlim(-2, 182)
    ax.set_ylim(-1.35, 2.0)
    ax.set_yticks([-0.35, 0.5])
    ax.set_yticklabels(["median hue\nof each sphere", "H at S = V = 255"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("OpenCV hue H (8-bit images: H = degrees / 2)")
    save(fig, 1, "ex2_hue_bands")


# ---------------------------------------------------------------- chapter 2
CH2_NOTEBOOKS = [
    (EX2 / "Exercise 1 - Hough transform.ipynb", [
        ("code:edges = cv2.Canny(gray, 20, 200)", "ex1_canny_edges"),
        ("Finally, we can show the image:", "ex1_hough_lines"),
        ("code:circles found", "ex1_hough_circles"),
    ]),
    (EX2 / "Exercise 2 - Feature Detection.ipynb", [
        ("#### 2a", "ex2_2a_harris_checkers"),
        ("#### 2c", "ex2_2c_harris_vs_shitomasi"),
    ]),
    (EX2 / "Exercise 3 - Feature Matching.ipynb", [
        ("code:kp_img = cv2.drawKeypoints(gray, kp, img", "ex3_sift_keypoints"),
        ("#### 3.1", "ex3_3-1_orb_keypoints"),
        ("code:img3 = cv2.drawMatches(gray,kp,gray2,kp2,matches[:100]", "ex3_sift_matches_100"),
        ("#### 3.2", "ex3_3-2_orb_matches_100"),
        ("#### 3.3", "ex3_3-3_matches_300"),
    ]),
    (EX2 / "Exercise 4 - Optical Flow.ipynb", [
        ("code:for i in range(len(feat1)):", "ex4_sparse_all"),
        ("#### 4a", "ex4_4a_sparse_moving"),
        ("#### 4b.1", "ex4_4b1_flow_colour"),
        ("#### 4b.2", "ex4_4b2_flow_arrows"),
    ]),
    (WP2 / "Weekly Project 2.ipynb", [
        ("## Sparse optical flow", "wp2_sparse_frames"),
        ("## Dense optical flow", "wp2_dense_frames"),
    ], {"show_video = True": "show_video = False"}),      # no OpenCV windows while building
]
BLUES = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blues", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def gray_of(path):
    return cv2.cvtColor(rgb(path), cv2.COLOR_RGB2GRAY)


def signed(ax, img, title, pct=99):
    m = np.percentile(np.abs(img), pct)
    ax.imshow(img, cmap="gray", vmin=-m, vmax=m)
    ax.set_title(title)
    ax.axis("off")


def ch2_derivative_noise():
    x = np.arange(200)
    f = np.where(x < 100, 40.0, 160.0) + np.random.default_rng(3).normal(0, 6, x.size)
    kernel = cv2.getGaussianKernel(41, 4).ravel()
    smooth = np.convolve(np.pad(f, 20, mode="edge"), kernel, mode="valid")
    panels = [(f, "noisy step f(x)", "f"),
              (np.gradient(f), "derivative f'(x): the step is lost in the noise", "f'"),
              (np.gradient(smooth), "derivative after Gaussian smoothing (sigma 4)", "(G * f)'")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.4))
    for ax, (y, title, name) in zip(axes, panels):
        ax.plot(x, y, color=SERIES[0], linewidth=1.6)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel(name)
    save(fig, 2, "concept_derivative_noise")


def ch2_gradients():
    gray = gray_of(EX2 / "Lego.jpg")
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    log = cv2.Laplacian(cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2), cv2.CV_32F, ksize=3)
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.1))
    show(axes[0], gray, "gray image I")
    signed(axes[1], gx, "Sobel x: dI/dx (gray = 0)")
    signed(axes[2], gy, "Sobel y: dI/dy (gray = 0)")
    mag = np.sqrt(gx ** 2 + gy ** 2)
    axes[3].imshow(mag, cmap="gray", vmin=0, vmax=np.percentile(mag, 99))
    axes[3].set_title("magnitude sqrt(Gx^2 + Gy^2)")
    axes[3].axis("off")
    signed(axes[4], log, "Laplacian of Gaussian, sigma 2")
    save(fig, 2, "concept_gradients")


def ch2_canny_thresholds():
    gray = gray_of(EX2 / "Lego.jpg")
    settings = [(5, 20, "too low"), (50, 150, "useful range"), (250, 500, "too high")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (lo, hi, label) in zip(axes, settings):
        edges = cv2.Canny(gray, lo, hi)
        n = int(np.sum(edges == 255))
        show(ax, edges, f"Canny({lo}, {hi}): {label}\n{n} edge pixels")
        print(f"  Canny({lo}, {hi}) on Lego: {n} edge pixels")
    save(fig, 2, "concept_canny_thresholds")


def ch2_harris_regions():
    l1, l2 = np.meshgrid(np.linspace(0, 10, 400), np.linspace(0, 10, 400))
    k = 0.04
    R = l1 * l2 - k * (l1 + l2) ** 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, score, thr, title in [(axes[0], R, 4.0, "Harris: R = l1 l2 - k (l1 + l2)^2 > 4"),
                                  (axes[1], np.minimum(l1, l2), 2.0, "Shi-Tomasi: min(l1, l2) > 2")]:
        ax.contourf(l1, l2, (score > thr).astype(float), levels=[0.5, 1.5], colors=[SERIES[0]], alpha=0.25)
        ax.contour(l1, l2, score, levels=[thr], colors=[SERIES[0]], linewidths=2)
        if score is R:
            ax.contour(l1, l2, R, levels=[0], colors=[INK2], linestyles="--", linewidths=1.2)
            ax.text(9.8, 0.25, "R < 0 below the dashed line", ha="right", color=INK2)
        ax.text(0.6, 0.6, "flat", color=INK, fontsize=12)
        ax.text(8.2, 1.0, "edge", color=INK, fontsize=12)
        ax.text(0.6, 9.0, "edge", color=INK, fontsize=12)
        ax.text(7.0, 7.0, "corner", color=INK, fontsize=12)
        ax.set_xlabel("lambda 1")
        ax.set_ylabel("lambda 2")
        ax.set_aspect("equal")
        ax.set_title(title)
    save(fig, 2, "concept_harris_regions")


def ch2_scale_space():
    gray = gray_of(EX2 / "Book.jpg").astype(np.float32)
    h, w = gray.shape
    crop = (slice(h // 2 - 150, h // 2 + 150), slice(w // 2 - 150, w // 2 + 150))
    sigmas = [1.6 * 2 ** (i / 2) for i in range(5)]                  # k = sqrt(2) between levels
    blurred = [cv2.GaussianBlur(gray, (0, 0), s) for s in sigmas]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.6))
    for i, (s, b) in enumerate(zip(sigmas, blurred)):
        show(axes[0, i], b[crop], f"Gaussian, sigma {s:.1f}")
    for i in range(4):
        signed(axes[1, i], (blurred[i + 1] - blurred[i])[crop],
               f"DoG: sigma {sigmas[i + 1]:.1f} minus {sigmas[i]:.1f}")
    axes[1, 4].axis("off")
    save(fig, 2, "concept_scale_space")


def ch2_hough_accumulator():
    # (a) and (b): three collinear points and their sinusoids in (theta, rho) space
    theta0, rho0 = np.deg2rad(60), 50.0
    base = rho0 * np.array([np.cos(theta0), np.sin(theta0)])
    direction = np.array([-np.sin(theta0), np.cos(theta0)])
    points = [base + t * direction for t in (-40, 0, 40)]
    thetas = np.deg2rad(np.arange(0, 180, 0.5))
    # (c): accumulator of the checkerboard edges, as in Exercise 1
    gray = gray_of(EX2 / "checkers.png")
    edges = cv2.Canny(gray, 20, 200)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    ys, xs = np.nonzero(edges)
    diag = int(np.ceil(np.hypot(*gray.shape)))
    acc = np.zeros((2 * diag + 1, 180), np.int64)
    for t in range(180):
        rho = np.round(xs * np.cos(np.deg2rad(t)) + ys * np.sin(np.deg2rad(t))).astype(int) + diag
        acc[:, t] = np.bincount(rho, minlength=2 * diag + 1)
    print(f"  Hough: {len(xs)} edge pixels, accumulator {acc.shape[0]} x {acc.shape[1]}, "
          f"max votes {acc.max()}, lines found by HoughLines(threshold 200): {len(lines)}")
    print(f"  Hough line angles (deg): {sorted(set(np.round(np.rad2deg(lines[:, 0, 1])).astype(int).tolist()))}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw={"width_ratios": [1, 1.3, 1.6]})
    ax = axes[0]
    ends = [base + t * direction for t in (-70, 70)]
    ax.plot([ends[0][0], ends[1][0]], [ends[0][1], ends[1][1]], color=MUTED, linewidth=1)
    for i, p in enumerate(points):
        ax.scatter([p[0]], [p[1]], s=70, color=SERIES[i], edgecolors=SURFACE, linewidths=1.5, zorder=3,
                   label=f"point {i + 1}")
    ax.plot([0, base[0]], [0, base[1]], color=INK, linestyle=":", linewidth=1.2)
    ax.text(base[0] / 2 + 2, base[1] / 2 - 4, "rho", color=INK)
    ax.set_xlim(-60, 90)
    ax.set_ylim(-30, 110)
    ax.invert_yaxis()                                  # image convention: y points down
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("image space: 3 points on one line")
    ax.legend(loc="lower left")
    ax = axes[1]
    for i, p in enumerate(points):
        ax.plot(np.rad2deg(thetas), p[0] * np.cos(thetas) + p[1] * np.sin(thetas), color=SERIES[i],
                label=f"point {i + 1}")
    ax.scatter([60], [rho0], s=120, facecolors="none", edgecolors=INK, linewidths=1.5, zorder=3)
    ax.text(63, rho0 + 6, "theta = 60, rho = 50", color=INK)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_xlabel("theta [deg]")
    ax.set_ylabel("rho")
    ax.set_title("parameter space: rho = x cos(theta) + y sin(theta)")
    ax.legend(loc="lower left")
    ax = axes[2]
    ax.imshow(np.log1p(acc), cmap=BLUES, aspect="auto", origin="lower",
              extent=(-0.5, 179.5, -diag - 0.5, diag + 0.5))
    ax.scatter(np.rad2deg(lines[:, 0, 1]), lines[:, 0, 0], s=60, facecolors="none",
               edgecolors=SERIES[1], linewidths=1.3)
    ax.set_xlabel("theta [deg]")
    ax.set_ylabel("rho [px]")
    ax.set_title(f"checkers.png: votes (log scale); rings = the {len(lines)} lines found")
    save(fig, 2, "concept_hough_voting")


def ch2_ransac_line():
    rng = np.random.default_rng(7)
    x_in = rng.uniform(0, 10, 40)
    y_in = 0.5 * x_in + 1 + rng.normal(0, 0.25, 40)
    x_out = rng.uniform(0, 10, 25)
    y_out = rng.uniform(0, 9, 25)
    x = np.concatenate([x_in, x_out])
    y = np.concatenate([y_in, y_out])

    def least_squares(xv, yv):                                       # closed form, L2 s.43
        b1 = np.sum((xv - xv.mean()) * (yv - yv.mean())) / np.sum((xv - xv.mean()) ** 2)
        return b1, yv.mean() - b1 * xv.mean()

    ls = least_squares(x, y)
    best = None
    for _ in range(200):                                             # RANSAC, L2 s.50-51
        i, j = rng.choice(len(x), 2, replace=False)
        if x[i] == x[j]:
            continue
        m = (y[j] - y[i]) / (x[j] - x[i])
        c = y[i] - m * x[i]
        inliers = np.abs(m * x - y + c) / np.sqrt(m ** 2 + 1) < 0.5
        if best is None or inliers.sum() > best.sum():
            best = inliers
    rs = least_squares(x[best], y[best])                             # refit on all inliers
    print(f"  line fit: true y = 0.50 x + 1.00; least squares y = {ls[0]:.2f} x + {ls[1]:.2f}; "
          f"RANSAC y = {rs[0]:.2f} x + {rs[1]:.2f} with {best.sum()} inliers of {len(x)} points")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(x[~best], y[~best], s=36, color=MUTED, label="outliers (RANSAC)")
    ax.scatter(x[best], y[best], s=36, color=SERIES[0], label="inliers (RANSAC)")
    xs = np.array([0, 10])
    ax.plot(xs, ls[0] * xs + ls[1], color=SERIES[1], linestyle="--", label="least squares, all points")
    ax.plot(xs, rs[0] * xs + rs[1], color=SERIES[0], label="RANSAC, refit on the inliers")
    ax.text(10.1, ls[0] * 10 + ls[1], " least squares", color=INK, va="center")
    ax.text(10.1, rs[0] * 10 + rs[1], " RANSAC", color=INK, va="center")
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_xlim(0, 11.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("40 points on y = 0.5 x + 1 plus 25 outliers")
    ax.legend(loc="upper left")
    save(fig, 2, "concept_ransac_line")


def ch2_match_quality():
    g1, g2 = gray_of(EX2 / "Book.jpg"), gray_of(EX2 / "More_books.jpg")
    methods = [("SIFT", cv2.SIFT_create(), cv2.NORM_L2),
               ("ORB, 500 features", cv2.ORB_create(nfeatures=500), cv2.NORM_HAMMING),
               ("ORB, 3000 features", cv2.ORB_create(nfeatures=3000), cv2.NORM_HAMMING)]
    ns = np.arange(25, 1001, 25)
    fig, ax = plt.subplots(figsize=(10, 5))
    ends = []
    for (name, det, norm), colour in zip(methods, SERIES):
        k1, d1 = det.detectAndCompute(g1, None)
        k2, d2 = det.detectAndCompute(g2, None)
        ms = sorted(cv2.BFMatcher(norm).match(d1, d2), key=lambda m: m.distance)
        a = np.float32([k1[m.queryIdx].pt for m in ms])
        b = np.float32([k2[m.trainIdx].pt for m in ms])
        inl = cv2.findHomography(a, b, cv2.RANSAC, 5.0)[1].ravel().astype(bool)
        n_ok = ns[ns <= len(ms)]
        share = [inl[:n].mean() for n in n_ok]
        ax.plot(n_ok, share, color=colour, label=name)
        ends.append((n_ok[-1], share[-1], name))
        print(f"  {name}: {len(ms)} matches; correct among best 100 / 300 / 500: "
              f"{inl[:100].mean():.0%} / {inl[:300].mean():.0%} / {inl[:500].mean():.0%}")
    for xe, ye, name in ends:
        ax.text(xe + 10, ye, " " + name, color=INK, va="center")
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1250)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("number of best matches (sorted by descriptor distance)")
    ax.set_ylabel("share that agrees with the homography")
    ax.set_title("Book.jpg vs More_books.jpg: correct matches (RANSAC inliers, 5 px)")
    ax.legend(loc="lower left")
    save(fig, 2, "concept_match_quality")


def ch2_flow_wheel():
    size, c = 301, 150
    yy, xx = np.indices((size, size))
    mag, ang = cv2.cartToPolar((xx - c).astype(np.float32), (yy - c).astype(np.float32))
    hsv = np.zeros((size, size, 3), np.uint8)
    hsv[:, :, 0] = ang * 180 / np.pi / 2
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.clip(mag / c * 255, 0, 255)
    wheel = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    wheel[mag > c] = (252, 252, 251)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(wheel)
    for text, (x, y) in [("right\n0 deg", (size + 8, c)), ("down\n90 deg", (c, size + 22)),
                         ("left\n180 deg", (-8, c)), ("up\n270 deg", (c, -22))]:
        ax.text(x, y, text, ha="center" if x == c else ("left" if x > c else "right"), va="center", color=INK)
    ax.set_xlim(-90, size + 90)
    ax.set_ylim(size + 50, -50)
    ax.axis("off")
    ax.set_title("flow colour key: hue = direction, brightness = length\n(image y axis points down)")
    save(fig, 2, "concept_flow_colour_key")


# ---------------------------------------------------------------- chapter 3
CH3_NOTEBOOKS = [
    (EX3 / "Exercise 1 - Template Matching.ipynb", [
        ("code:res = cv2.matchTemplate(gray_left, template, cv2.TM_CCORR_NORMED)", "ex1_similarity_map"),
        ("code:crocodile = cv2.rectangle", "ex1_crocodile"),
        ("#### 1.2", "ex1_1-2_bucket"),
        ("#### 1.3", "ex1_1-3_metrics"),
    ]),
    (EX3 / "Exercise 2 - Stereo Block Matching.ipynb", [
        ("#### 2.1", "ex2_2-1_disparity"),
        ("#### 2.3", "ex2_2-3_storage"),
    ]),
    (WP3 / "Weekly Project 3.ipynb", [
        ("## Step 2:", "wp3_step2_noses"),
        ("## Step 3:", "wp3_step3_cost"),
        ("## Challenge:", "wp3_disparity"),
    ]),
]


def sad_volume(left, right, block, max_disp):
    """Disparity with the smallest SAD. Same costs as the notebook's loops, computed with a box filter."""
    a, b = left.astype(np.float32), right.astype(np.float32)
    h, w = a.shape
    volume = np.full((max_disp + 1, h, w), 1e12, np.float32)
    for d in range(max_disp + 1):
        volume[d, :, d:] = cv2.boxFilter(np.abs(a[:, d:] - b[:, :w - d]), -1, (block, block), normalize=False,
                                         borderType=cv2.BORDER_REPLICATE)
    return volume.argmin(axis=0)


def sift_disparities(left, right):
    """Reliable sparse disparities: ratio test, same row in both images. Returns x, y, disparity."""
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(left, None)
    k2, d2 = sift.detectAndCompute(right, None)
    good = [a for a, b in cv2.BFMatcher().knnMatch(d1, d2, k=2) if a.distance < 0.7 * b.distance]
    p = np.array([[k1[m.queryIdx].pt[0], k1[m.queryIdx].pt[1], k2[m.trainIdx].pt[0], k2[m.trainIdx].pt[1]] for m in good])
    p = p[np.abs(p[:, 1] - p[:, 3]) < 1.5]
    return p[:, 0], p[:, 1], p[:, 0] - p[:, 2]


def ch3_epipolar_geometry():
    O, O2, P = np.array([1.0, 1.0]), np.array([7.0, 1.0]), np.array([4.0, 5.5])
    s = 0.45                                      # the image planes lie at this fraction from each centre
    to_left = lambda q: O + s * (q - O)
    to_right = lambda q: O2 + s * (q - O2)
    near, far = O + 0.75 * (P - O), O + 1.15 * (P - O)            # other depths along the ray through p
    e, e2 = to_left(O2), to_right(O)
    p = to_left(P)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.fill([O[0], O2[0], P[0]], [O[1], O2[1], P[1]], color=SERIES[0], alpha=0.12, linewidth=0, label="epipolar plane")
    for xs in ([1.2, 3.95, 3.95, 1.2], [6.8, 4.05, 4.05, 6.8]):
        ax.fill(xs, [0.3, 0.65, 3.6, 3.25], facecolor="none", edgecolor=INK2, linewidth=1.2)
    ax.plot([O[0], O2[0]], [O[1], O2[1]], color=INK, linewidth=1.6, label="baseline")
    ax.plot([O[0], far[0]], [O[1], far[1]], color=INK2, linewidth=1.2)
    for q in (near, P, far):
        ax.plot([O2[0], q[0]], [O2[1], q[1]], color=MUTED, linewidth=1, linestyle="--")
        ax.scatter(*q, s=45, color=INK, zorder=3)
        ax.scatter(*to_right(q), s=30, color=SERIES[1], zorder=3)
    end = e + 1.15 * (p - e)
    ax.plot([e[0], end[0]], [e[1], end[1]], color=SERIES[1], label="epipolar lines")
    far_r = to_right(far)
    ax.plot([e2[0], far_r[0]], [e2[1], far_r[1]], color=SERIES[1])
    ax.scatter(*p, s=45, color=SERIES[0], zorder=3)
    for q in (e, e2):
        ax.scatter(*q, s=45, color=INK, marker="s", zorder=3)
    for text, (x, y), kw in [("O", O + (-0.05, -0.32), {}), ("O'", O2 + (0.05, -0.32), {}), ("P ?", P + (0.15, 0.05), {}),
                             ("p", p + (-0.2, 0.05), {}), ("e", e + (-0.2, -0.3), {}), ("e'", e2 + (0.1, -0.3), {}),
                             ("all these depths of P\nproject onto one line", (6.0, 4.6), {"ha": "left"})]:
        ax.text(x, y, text, color=INK, fontsize=11, **kw)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper left")
    ax.set_title("general case: p in camera 1 restricts p' to a line in camera 2")
    ax = axes[1]
    for x0 in (0, 5):
        ax.add_patch(Rectangle((x0, 0), 4, 3, fill=False, edgecolor=INK2, linewidth=1.2))
        for y in (0.6, 1.5, 2.4):
            ax.plot([x0, x0 + 4], [y, y], color=SERIES[1], linewidth=1.5)
    ax.scatter([2.4, 6.2], [1.5, 1.5], s=45, color=SERIES[0], zorder=3)
    ax.text(2.4, 1.62, "$x_L$", ha="center", color=INK, fontsize=11)
    ax.text(6.2, 1.62, "$x_R$", ha="center", color=INK, fontsize=11)
    ax.text(4.5, -0.5, "epipolar lines are rows; the epipoles are at infinity", ha="center", color=INK)
    ax.set_xlim(-0.2, 9.2)
    ax.set_ylim(-0.8, 3.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("rectified case: search along the same row")
    save(fig, 3, "concept_epipolar_geometry")


def ch3_depth_from_disparity():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    f, B, P = 1.0, 4.0, np.array([1.4, 5.0])
    for x0 in (0.0, B):
        ax.plot([x0 - 1, x0 + 1], [f, f], color=INK2, linewidth=2)            # image plane
        ax.plot([x0, x0], [0, 5.6], color=MUTED, linewidth=1, linestyle="--")  # optical axis
        ax.plot([x0, P[0]], [0, P[1]], color=SERIES[0], linewidth=1.6)         # ray to P
        hit = x0 + (P[0] - x0) * f / P[1]
        ax.scatter([hit], [f], s=40, color=SERIES[1], zorder=3)
    ax.scatter(*P, s=60, color=INK, zorder=3)
    ax.text(P[0] + 0.15, P[1], "P", color=INK, fontsize=12)
    ax.text(0.14, f + 0.14, "$x_L$", color=INK, fontsize=11)
    ax.text(3.74, f + 0.14, "$x_R$", color=INK, fontsize=11)
    for (a, b, text, tx, ty) in [((0, -0.45), (B, -0.45), "B", B / 2, -0.8), ((-1.35, 0), (-1.35, f), "f", -1.65, f / 2),
                                 ((-2.1, 0), (-2.1, P[1]), "Z", -2.45, P[1] / 2)]:
        ax.annotate("", xy=a, xytext=b, arrowprops=dict(arrowstyle="<->", color=INK2))
        ax.text(tx, ty, text, color=INK, fontsize=12, ha="center", va="center")
    ax.scatter([0, B], [0, 0], s=50, color=INK, marker="s", zorder=3)
    ax.set_xlim(-2.8, 5.5)
    ax.set_ylim(-1.1, 5.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("top view: $d = x_L - x_R$ and $Z = fB/d$")
    ax = axes[1]
    fB = 375.5 * 49.8
    d = np.arange(8, 81)
    ax.plot(d, fB / d, color=SERIES[0])
    for d0 in (10, 60):
        ax.scatter([d0], [fB / d0], s=50, color=SERIES[0], zorder=3)
        ax.text(d0 + 2, fB / d0 + 40, f"d = {d0} px: 1 px of error = {fB / d0 - fB / (d0 + 1):.0f} mm", color=INK)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_xlabel("disparity d [px]")
    ax.set_ylabel("depth Z [mm]")
    ax.set_title("f = 375 px, B = 50 mm (the camera of Weekly Project 4)")
    save(fig, 3, "concept_depth_from_disparity")


def ch3_metrics():
    t = gray_of(WP3 / "nose_left.png").astype(np.float64)
    span = gray_of(WP3 / "nose_span.png").astype(np.float64)
    w = t.shape[1]
    xs = np.arange(span.shape[1] - w + 1)
    patches = [span[:, x:x + w] for x in xs]
    sad = np.array([np.abs(t - p).sum() for p in patches])
    ssd = np.array([((t - p) ** 2).sum() for p in patches])
    ncc = np.array([(t * p).sum() / np.sqrt((t ** 2).sum() * (p ** 2).sum()) for p in patches])
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.4))
    for ax, y, title, best in [(axes[0], sad, "SAD: smallest is best", sad.argmin()),
                               (axes[1], ssd, "SSD: smallest is best", ssd.argmin()),
                               (axes[2], ncc, "NCC: largest is best", ncc.argmax())]:
        ax.plot(xs, y, color=SERIES[0], linewidth=1.5)
        ax.axvline(best, color=INK, linestyle=":", linewidth=1.2)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_xlabel("column of nose_span.png")
        ax.set_title(f"{title} (column {best})")
    print(f"  metrics on the nose span: SAD best {sad.argmin()}, SSD best {ssd.argmin()}, NCC best {ncc.argmax()} "
          f"(NCC {ncc.max():.4f}, median {np.median(ncc):.3f}; second-best SAD {np.sort(sad)[1]:.0f}, best {sad.min():.0f})")
    save(fig, 3, "concept_metrics")


def ch3_window_size():
    left, right = gray_of(WP3 / "tsukuba_left.png"), gray_of(WP3 / "tsukuba_right.png")
    x, y, gt = sift_disparities(left, right)
    u, v = x.round().astype(int), y.round().astype(int)
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.9))
    show(axes[0], left, "tsukuba_left")
    for ax, block in zip(axes[1:], (3, 7, 21)):
        disp = sad_volume(left, right, block, 64)
        err = np.abs(disp[v, u] - gt)
        ax.imshow(disp, cmap="gray", vmin=0, vmax=64)
        ax.set_title(f"W = {block}: {np.mean(err <= 1):.0%} of SIFT points within 1 px")
        ax.axis("off")
        print(f"  tsukuba W={block}: {len(gt)} SIFT points, within 1 px {np.mean(err <= 1):.0%}, median error {np.median(err):.2f}")
    print(f"  tsukuba SIFT disparities percentiles 1/50/99: {np.percentile(gt, [1, 50, 99]).round(1)}")
    limited = sad_volume(left, right, 7, 20)
    print(f"  tsukuba with max_disp 20: {np.mean(limited == 20):.0%} of the pixels stop at the limit")
    save(fig, 3, "concept_window_size")


def ch3_stereobm_params():
    def small(path):
        img = cv2.imread(str(path))
        return cv2.cvtColor(cv2.resize(img, (img.shape[1] // 5, img.shape[0] // 5), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    left, right = small(EX3 / "playtable_left.png"), small(EX3 / "playtable_right.png")
    x, y, gt = sift_disparities(left, right)
    u, v = x.round().astype(int), y.round().astype(int)
    print(f"  playtable SIFT disparities: {len(gt)} points, percentiles 1/50/99: {np.percentile(gt, [1, 50, 99]).round(1)}")
    settings = [("block 5", 5, 1, 3, 200), ("block 15", 15, 1, 3, 200), ("block 31", 31, 1, 3, 200),
                ("block 15, uniqueness 10", 15, 10, 3, 200), ("block 15, uniqueness 10, speckle window 100", 15, 10, 100, 200),
                ("the tuned values of cell 2.1", 15, 10, 100, 1)]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.4))
    for ax, (name, block, unique, window, diff) in zip(axes.ravel(), settings):
        stereo = cv2.StereoBM_create(numDisparities=48, blockSize=block)
        stereo.setMinDisparity(7)
        stereo.setDisp12MaxDiff(diff)
        stereo.setUniquenessRatio(unique)
        stereo.setSpeckleRange(3 if window == 3 else 2)
        stereo.setSpeckleWindowSize(window)
        disp = stereo.compute(left, right).astype(np.float32) / 16
        here = disp[v, u]
        ok = here >= 7
        good = np.mean(np.abs(here[ok] - gt[ok]) <= 2)
        ax.imshow(disp, cmap="gray", vmin=6, vmax=55)
        ax.set_title(f"{name}\n{np.mean(disp >= 7):.0%} valid, {good:.0%} of SIFT points within 2 px")
        ax.axis("off")
        print(f"  StereoBM {name}: valid {np.mean(disp >= 7):.0%}, within 2 px {good:.0%}")
    save(fig, 3, "concept_stereobm_parameters")


def read_ply(path):
    with open(path) as f:
        for n, line in enumerate(f):
            if line.strip() == "end_header":
                break
    return np.loadtxt(path, skiprows=n + 1)


def ch3_point_cloud():
    fig = plt.figure(figsize=(16, 7))
    rng = np.random.default_rng(0)
    for i, (name, title) in enumerate([("out.ply", "playtable: out.ply"), ("out_storage.ply", "storage: out_storage.ply")]):
        data = read_ply(EX3 / name)
        print(f"  {name}: {len(data)} points, {(EX3 / name).stat().st_size // 1024} KB")
        chosen = rng.choice(len(data), min(40000, len(data)), replace=False)
        x, y, z, colours = data[chosen, 0], data[chosen, 1], -data[chosen, 2], data[chosen, 3:6] / 255
        keep = z < np.percentile(z, 98)                              # a few points with tiny disparity lie very far away
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(x[keep], z[keep], y[keep], c=colours[keep], s=1.2, linewidths=0)
        ax.set_box_aspect((np.ptp(x[keep]), np.ptp(z[keep]), np.ptp(y[keep])))
        ax.view_init(elev=18, azim=-68)
        ax.set_xlabel("X")
        ax.set_ylabel("depth")
        ax.set_zlabel("Y")
        ax.set_title(title)
    save(fig, 3, "concept_point_cloud")


# ---------------------------------------------------------------- chapter 4
CH4_NOTEBOOKS = [
    (EX4 / "Exercise 1 - Monochrome camera calibration.ipynb", [
        ("code:dst = cv2.undistort(img, mtx, dist, None, newcameramtx)", "ex1_undistort"),
        ("code:dst = dst[y:y+h, x:x+w]", "ex1_crop"),
    ], {"cv2.imshow('img',img)": "pass", "cv2.waitKey(500)": "pass", "cv2.destroyAllWindows()": "pass"}),   # no windows
    (EX4 / "Exercise 2 - Epipolar lines.ipynb", [
        ("code:lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2 ,F)", "ex2_2-3_epipolar_lines"),
        ("#### 2.3b", "ex2_2-3b_number_of_matches"),
    ]),
    (WP4 / "Weekly Project 4.ipynb", [
        ("## Rectification", "wp4_rectification_steps"),
        ("## Rectification@1", "wp4_rectified_pair"),
        ("## Block matching", "wp4_disparity"),
        ("## From disparity to depth", "wp4_depth"),
    ]),
]


def calibration_views():
    objp = np.zeros((42, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    objpoints, imgpoints = [], []
    for path in sorted((EX4 / "imgs").glob("*.png")):
        gray = cv2.imread(str(path), 0)
        ok, corners = cv2.findChessboardCorners(gray, (7, 6))
        if ok:
            objpoints.append(objp)
            imgpoints.append(corners)
    return objpoints, imgpoints, gray.shape[::-1]


def ch4_distortion():
    def distort(x, y, k1, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        return x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

    def grid(ax, xlim, ylim, transform):
        t = np.linspace(0, 1, 60)
        for x0 in np.linspace(*xlim, 9):
            ax.plot(*transform(np.full_like(t, x0), ylim[0] + t * (ylim[1] - ylim[0])), color=SERIES[0], linewidth=1.2)
        for y0 in np.linspace(*ylim, 7):
            ax.plot(*transform(xlim[0] + t * (xlim[1] - xlim[0]), np.full_like(t, y0)), color=SERIES[0], linewidth=1.2)

    objpoints, imgpoints, size = calibration_views()
    rms, K, d, _, _ = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)
    k1, k2, p1, p2, k3 = d.ravel()[:5]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    for ax, k, title in [(axes[0], -0.25, "barrel: $k_1 < 0$, lines bend outwards"),
                         (axes[1], 0.25, "pincushion: $k_1 > 0$, lines bend inwards")]:
        grid(ax, (-1, 1), (-0.75, 0.75), lambda x, y: distort(x, y, k))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title)
    ax = axes[2]
    grid(ax, (-1.2, 1.2), (-0.85, 0.85), lambda x, y: (lambda a, b: (K[0, 0] * a + K[0, 2], K[1, 1] * b + K[1, 2]))(*distort(x, y, k1, k2, p1, p2, k3)))
    ax.add_patch(Rectangle((0, 0), size[0], size[1], fill=False, edgecolor=INK, linewidth=1.5))
    ax.set_xlim(-60, size[0] + 60)
    ax.set_ylim(size[1] + 60, -60)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"camera of Exercise 1 ($k_1$ = {k1:.2f}): straight world lines, in pixels")
    print(f"  Exercise 1 camera: RMS {rms:.3f}, k1 {k1:.3f}, k2 {k2:.3f}, k3 {k3:.3f}")
    save(fig, 4, "concept_distortion")


def ch4_views():
    objpoints, imgpoints, size = calibration_views()
    rng = np.random.default_rng(0)
    counts = [3, 4, 5, 7, 10, 15, 20]
    fx, err = [], []
    for k in counts:
        fs, rs = [], []
        for _ in range(25):
            idx = rng.choice(len(objpoints), k, replace=False)
            r, K, *_ = cv2.calibrateCamera([objpoints[i] for i in idx], [imgpoints[i] for i in idx], size, None, None)
            fs.append(K[0, 0])
            rs.append(r)
        fx.append(fs)
        err.append(rs)
        print(f"  {k} views: fx median {np.median(fs):.1f}, range {min(fs):.1f}..{max(fs):.1f}; RMS median {np.median(rs):.3f}")
    full, K_all, *_ = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)
    print(f"  all {len(objpoints)} views: fx {K_all[0, 0]:.1f}, RMS {full:.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    for ax, values, reference, label, title in [
            (axes[0], fx, K_all[0, 0], "$f_x$ [px]", "focal length from random subsets of the views"),
            (axes[1], err, full, "RMS reprojection error [px]", "reprojection error of the same subsets")]:
        values = np.array(values)
        ax.fill_between(counts, values.min(axis=1), values.max(axis=1), color=SERIES[0], alpha=0.2, linewidth=0, label="smallest to largest of 25 subsets")
        ax.plot(counts, np.median(values, axis=1), color=SERIES[0], marker="o", markersize=6, label="median")
        ax.axhline(reference, color=INK2, linestyle="--", linewidth=1.2, label=f"all {len(objpoints)} views")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_xlabel("number of views used")
        ax.set_ylabel(label)
        ax.set_title(title)
        ax.legend()
    axes[0].set_ylim(420, 500)                       # 3 and 4 views reach about 3000 px, off this scale
    save(fig, 4, "concept_number_of_views")


def ch4_fundamental_sweep():
    img1, img2 = cv2.imread(str(EX4 / "img_left.jpg"), 0), cv2.imread(str(EX4 / "img_right.jpg"), 0)
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)
    ms = sorted(cv2.BFMatcher().match(d1, d2), key=lambda m: m.distance)
    p1 = np.int32([k1[m.queryIdx].pt for m in ms])
    p2 = np.int32([k2[m.trainIdx].pt for m in ms])
    counts = [8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 650, 815]
    median, share = [], []
    for n in counts:
        F, mask = cv2.findFundamentalMat(p1[:n], p2[:n], cv2.FM_RANSAC)
        lines = cv2.computeCorrespondEpilines(p2[:200].reshape(-1, 1, 2), 2, F).reshape(-1, 3)
        median.append(np.median(np.abs(lines[:, 0] * p1[:200, 0] + lines[:, 1] * p1[:200, 1] + lines[:, 2])))
        share.append(mask.mean())
        print(f"  F from {n} matches: inliers {mask.mean():.0%}, median distance of the best 200 matches {median[-1]:.2f} px")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    for ax, values, label, title in [(axes[0], median, "median distance to the line [px]", "error on the best 200 matches"),
                                     (axes[1], np.array(share) * 100, "inliers [%]", "share of the matches RANSAC accepts")]:
        ax.plot(counts, values, color=SERIES[0], marker="o", markersize=6)
        ax.axvline(200, color=INK2, linestyle="--", linewidth=1.2)
        ax.set_xscale("log")
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_xlabel("number of best matches used (dashed: 200)")
        ax.set_ylabel(label)
        ax.set_title(title)
    axes[0].set_yscale("log")
    save(fig, 4, "concept_fundamental_sweep")


# ---------------------------------------------------------------- chapter 5
import open3d as o3d                                     # noqa: E402  (only chapters 5 and 6 need it)
REG = o3d.pipelines.registration
EX5 = ROOT / "Class_5_point_cloud_processing" / "Exercises_week_5" / "week5"
W5 = ROOT / "Class_5_weekly_project" / "weekly_project_5"
EX6 = ROOT / "Class_6_pcp_2" / "Exercises_week_6" / "week6"
W6 = ROOT / "Class_6_weekly_project"
EX7 = ROOT / "Class_7_state_estimation" / "Exercises week 7"
W7 = ROOT / "Class_7_weekly_project" / "Project week 7"
NO_WINDOWS = {"o3d.visualization.draw_geometries([source_temp, target_temp])": "pass",
              "o3d.visualization.draw_geometries([bedroom])": "pass",
              "o3d.visualization.draw_geometries([pcl_temp])": "pass",
              "o3d.visualization.draw_geometries([pcl])": "pass"}
ORANGE, BLUE = (1, 0.706, 0), (0, 0.651, 0.929)          # the colours of draw_registrations

CH5_NOTEBOOKS = [
    (EX5 / "Exercise1.ipynb", [], NO_WINDOWS),
    (EX5 / "Exercise2.ipynb", [("code:rgbd_image0 = o3d.geometry.RGBDImage.create_from_color_and_depth", "ex2_rgbd")], NO_WINDOWS),
    (W5 / "Weekly Project.ipynb", [], NO_WINDOWS),
]


def prepare_cloud(cloud, v):
    s = cloud.voxel_down_sample(v)
    s.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v * 2, max_nn=30))
    return s, REG.compute_fpfh_feature(s, o3d.geometry.KDTreeSearchParamHybrid(radius=v * 5, max_nn=100))


def registered(a, b):
    sa, fa = prepare_cloud(a, 0.05)
    sb, fb = prepare_cloud(b, 0.05)
    r = REG.registration_ransac_based_on_feature_matching(sa, sb, fa, fb, True, 0.075, REG.TransformationEstimationPointToPoint(False))
    i = REG.registration_icp(sa, sb, 0.05, r.transformation, REG.TransformationEstimationPointToPlane())
    return sa, sb, fa, fb, i.transformation, i


def moved(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


def ch5_registration():
    r1, r2 = o3d.io.read_point_cloud(str(EX5 / "ICP/r1.pcd")), o3d.io.read_point_cloud(str(EX5 / "ICP/r2.pcd"))
    sa, sb, _, _, T, res = registered(r1, r2)
    a, b = np.asarray(sa.points), np.asarray(sb.points)
    print(f"  r1 to r2 after RANSAC + ICP: fitness {res.fitness:.3f}, rmse {res.inlier_rmse:.4f}, rotation {np.degrees(np.arccos((np.trace(T[:3, :3]) - 1) / 2)):.0f} deg")
    fig = plt.figure(figsize=(14, 6))
    for i, (p, title) in enumerate([(a, "before: r1 (orange) and r2 (blue)"), (moved(a, T), "after global registration and ICP")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], color=[ORANGE], s=1.5, linewidths=0)
        ax.scatter(b[:, 0], b[:, 1], b[:, 2], color=[BLUE], s=1.5, linewidths=0)
        all_points = np.vstack([p, b])
        ax.set_box_aspect(np.ptp(all_points, axis=0))
        ax.view_init(elev=25, azim=-60)
        ax.set_title(title)
    save(fig, 5, "concept_registration")


def ch5_icp_steps():
    t = np.linspace(0, 10, 200)
    target = np.column_stack([t, 1.5 * np.sin(0.8 * t) + 0.1 * t])
    rng = np.random.default_rng(0)
    source = target[::2] + rng.normal(0, 0.03, (100, 2))

    def rot(deg, shift):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        R = np.array([[c, -s], [s, c]])
        return (source - source.mean(0)) @ R.T + source.mean(0) + shift

    def icp_step(p):
        d = np.sum((p[:, None] - target[None]) ** 2, axis=2)
        q = target[d.argmin(axis=1)]
        pm, qm = p.mean(0), q.mean(0)
        U, _, Vt = np.linalg.svd((p - pm).T @ (q - qm))
        D = np.diag([1, np.linalg.det(Vt.T @ U.T)])
        R = Vt.T @ D @ U.T
        return (p - pm) @ R.T + qm, q, np.sqrt(d.min(axis=1).mean())

    fig, axes = plt.subplots(2, 4, figsize=(17, 5.6))
    for row, (deg, shift, label) in enumerate([(15, (0.5, 0.7), "good start"), (130, (0, 0), "bad start: 130 deg")]):
        p = rot(deg, np.array(shift))
        step = 0
        for ax, show_at in zip(axes[row], (0, 1, 3, 30)):
            while step < show_at:
                p, _, _ = icp_step(p)
                step += 1
            _, q, err = icp_step(p)
            ax.plot(target[:, 0], target[:, 1], color=BLUE, linewidth=2)
            for a, b in zip(p[::6], q[::6]):
                ax.plot([a[0], b[0]], [a[1], b[1]], color=GRID, linewidth=0.8)
            ax.scatter(p[:, 0], p[:, 1], color=[ORANGE], s=10, zorder=3)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{label}, iteration {show_at}: error {err:.2f}")
            print(f"  ICP {label}, iteration {show_at}: rms nearest-point distance {err:.3f}")
    save(fig, 5, "concept_icp_steps")


def rgbd_cloud(i, v=None, normals=False):
    cam = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.io.read_image(str(EX5 / f"RGBD/color/{i:06d}.jpg")), o3d.io.read_image(str(EX5 / f"RGBD/depth/{i:06d}.png")),
        convert_rgb_to_intensity=False)
    c = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam)
    c.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    if v is not None:
        c = c.voxel_down_sample(v)
    if normals:
        c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30), fast_normal_computation=True)
    return c


def ch5_icp_limits():
    p2p, p2pl = REG.TransformationEstimationPointToPoint(False), REG.TransformationEstimationPointToPlane()
    first = rgbd_cloud(0, 0.02, True)
    gaps = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300]
    fits = {"point-to-point": [], "point-to-plane": []}
    for g in gaps:
        other = rgbd_cloud(g, 0.02, True)
        for name, est in (("point-to-point", p2p), ("point-to-plane", p2pl)):
            r = REG.registration_icp(first, other, 0.02, np.eye(4), est)
            fits[name].append(REG.evaluate_registration(first, other, 0.02, r.transformation).fitness)
    far = rgbd_cloud(300, 0.02, True)
    thresholds = [0.02, 0.05, 0.1, 0.2, 0.5]
    by_thr = {"point-to-point": [], "point-to-plane": []}
    for t in thresholds:
        for name, est in (("point-to-point", p2p), ("point-to-plane", p2pl)):
            r = REG.registration_icp(first, far, t, np.eye(4), est)
            by_thr[name].append(REG.evaluate_registration(first, far, 0.02, r.transformation).fitness)
    print("  fitness at 2 cm by gap:", {k: np.round(v, 2).tolist() for k, v in fits.items()})
    print("  fitness at 2 cm by threshold (0 to 300):", {k: np.round(v, 2).tolist() for k, v in by_thr.items()})
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    for ax, xs, data, label, title in [(axes[0], gaps, fits, "frames between the two images", "threshold 0.02: a larger gap breaks ICP"),
                                       (axes[1], thresholds, by_thr, "ICP threshold [m]", "frames 0 and 300: effect of the threshold")]:
        for (name, values), colour, style in zip(data.items(), SERIES, ("-", "--")):
            ax.plot(xs, values, color=colour, linestyle=style, marker="o", markersize=6, label=name)
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_xlabel(label)
        ax.set_ylabel("fitness at 2 cm")
        ax.set_title(title)
        ax.legend()
    save(fig, 5, "concept_icp_limits")


def ch5_map():
    p2pl = REG.TransformationEstimationPointToPlane()
    T = np.eye(4)
    previous = rgbd_cloud(0, 0.02, True)
    bedroom = rgbd_cloud(0, 0.02)
    path = [T[:3, 3].copy()]
    for i in range(10, 400, 10):
        current = rgbd_cloud(i, 0.02, True)
        T = T @ REG.registration_icp(current, previous, 0.05, np.eye(4), p2pl).transformation
        path.append(T[:3, 3].copy())
        bedroom = bedroom + rgbd_cloud(i, 0.02).transform(T)
        if i % 100 == 0:
            bedroom = bedroom.voxel_down_sample(0.02)
        previous = current
    bedroom = bedroom.voxel_down_sample(0.02)
    points, colours, path = np.asarray(bedroom.points), np.asarray(bedroom.colors), np.array(path)
    print(f"  map: {len(points)} points, size {np.round(np.ptp(points, axis=0), 2)}, path {np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)):.2f} m")
    keep = np.random.default_rng(0).choice(len(points), min(45000, len(points)), replace=False)
    fig = plt.figure(figsize=(15, 6.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(points[keep, 0], points[keep, 2], points[keep, 1], c=colours[keep], s=1.2, linewidths=0)
    ax.plot(path[:, 0], path[:, 2], path[:, 1], color=SERIES[7], linewidth=2.5)
    ax.set_box_aspect(np.ptp(points[keep][:, [0, 2, 1]], axis=0))
    ax.view_init(elev=30, azim=-70)
    ax.set_title("the reconstructed map, red line: camera path")
    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(points[keep, 0], points[keep, 2], c=colours[keep], s=1.5, linewidths=0)
    ax.plot(path[:, 0], path[:, 2], color=SERIES[7], linewidth=2.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("top view")
    save(fig, 5, "concept_map")


def ch5_fpfh():
    r1, r2 = o3d.io.read_point_cloud(str(EX5 / "ICP/r1.pcd")), o3d.io.read_point_cloud(str(EX5 / "ICP/r2.pcd"))
    sa, sb, fa, fb, T, _ = registered(r1, r2)
    a, b = moved(np.asarray(sa.points), T), np.asarray(sb.points)
    Fa, Fb = fa.data.T, fb.data.T
    d = np.sqrt(((a[:, None] - b[None]) ** 2).sum(axis=2))
    partner, gap = d.argmin(axis=1), d.min(axis=1)
    close = np.nonzero(gap < 0.02)[0]
    score = np.linalg.norm(Fa[close] - Fa.mean(axis=0), axis=1)        # a distinctive point
    i = close[score.argmax()]
    j = partner[i]
    k = d[i].argmax()
    print(f"  FPFH distance: same surface point {np.linalg.norm(Fa[i] - Fb[j]):.1f}, far point {np.linalg.norm(Fa[i] - Fb[k]):.1f}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.8), sharey=True)
    for ax, other, title, colour in [(axes[0], Fb[j], "the same surface point in the two clouds", SERIES[0]),
                                     (axes[1], Fb[k], "a point on a different surface", SERIES[1])]:
        ax.plot(Fa[i], color=INK, linewidth=1.5, label="point in r1")
        ax.plot(other, color=colour, linestyle="--", label="point in r2")
        for edge in (10.5, 21.5):
            ax.axvline(edge, color=AXIS, linewidth=1)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_xlabel("histogram bin (3 sub-histograms of 11 bins)")
        ax.set_title(title)
        ax.legend()
    axes[0].set_ylabel("value")
    save(fig, 5, "concept_fpfh")


# ---------------------------------------------------------------- chapter 6
CH6_NOTEBOOKS = [
    (EX6 / "Exercise1.ipynb", [("#### A)", "ex1_a_ols"), ("code:predicted_zz", "ex1_b_plane")]),
    (EX6 / "Exercise2.ipynb", [], NO_WINDOWS),
    (EX6 / "Exercise 3 - Ranging.ipynb", [("#### 3.1", "ex3_3-1_scan"),
                                          ("code:plt.scatter(x[ransac.inlier_mask_]", "ex3_default_fit"),
                                          ("#### The second wall", "ex3_two_walls")]),
    (W6 / "Weeklyproject6.ipynb", [("#### Finding the number of clusters", "wp6_choosing_k")], NO_WINDOWS),
]


def ch6_kmeans_steps():
    rng = np.random.default_rng(3)
    blobs = [rng.normal(c, 0.7, (60, 2)) for c in ((0, 0), (5, 1), (2, 5))]
    p = np.vstack(blobs)
    centers = p[[3, 8, 70]].copy()                     # a poor start on purpose: two centres in one blob
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    step = 0
    for ax, show_at in zip(axes, (0, 1, 3, 8)):
        while step < show_at:
            labels = ((p[:, None] - centers[None]) ** 2).sum(axis=2).argmin(axis=1)
            centers = np.array([p[labels == j].mean(axis=0) for j in range(3)])
            step += 1
        labels = ((p[:, None] - centers[None]) ** 2).sum(axis=2).argmin(axis=1)
        dist = ((p - centers[labels]) ** 2).sum()
        for j in range(3):
            ax.scatter(p[labels == j, 0], p[labels == j, 1], color=SERIES[j], s=18)
        ax.scatter(centers[:, 0], centers[:, 1], color=INK, marker="X", s=130, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"iteration {show_at}: distortion {dist:.0f}")
        print(f"  k-means iteration {show_at}: distortion {dist:.1f}")
    save(fig, 6, "concept_kmeans_steps")


def ch6_meanshift():
    rng = np.random.default_rng(1)
    p = np.vstack([rng.normal(c, s, (n, 2)) for c, s, n in (((0, 0), 0.8, 120), ((5, 2), 0.6, 80), ((1, 5), 0.9, 100))])
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    ax.scatter(p[:, 0], p[:, 1], color=AXIS, s=14)
    modes = []
    for start in ((-1.5, -1.5), (1.5, 1.5), (3.6, 3.2), (6.2, 0.8), (2.5, 6.5), (-0.5, 3.5)):
        w = np.array(start, float)
        track = [w.copy()]
        for _ in range(40):
            weights = np.exp(-((p - w) ** 2).sum(axis=1) / (2 * 1.0 ** 2))
            new = (weights[:, None] * p).sum(axis=0) / weights.sum()
            if np.linalg.norm(new - w) < 1e-3:
                break
            w = new
            track.append(w.copy())
        track = np.array(track)
        ax.plot(track[:, 0], track[:, 1], color=SERIES[0], linewidth=1.8, marker="o", markersize=3)
        modes.append(tuple(np.round(w, 1)))
        ax.scatter(*w, color=INK, marker="X", s=120, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    print("  mean shift: the 6 starts end in the modes", sorted(set(modes)))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("mean shift, bandwidth 1: each window climbs to a mode (X)")
    save(fig, 6, "concept_mean_shift")


def ch6_dbscan_types():
    p = np.array([[0, 0], [0.5, 0.2], [-0.4, 0.4], [0.2, -0.5], [0.6, -0.4], [-0.3, -0.3], [1.0, 0.7], [1.45, 1.2], [3.2, 0.9], [-1.8, 1.6]], float)
    eps, min_pts = 1.0, 4
    d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(axis=2))
    counts = (d <= eps).sum(axis=1)
    core = counts >= min_pts
    border = ~core & np.any(d[:, core] <= eps, axis=1)
    noise = ~core & ~border
    print(f"  DBSCAN example: core {core.sum()}, border {border.sum()}, noise {noise.sum()}")
    fig, ax = plt.subplots(figsize=(8, 5))
    for mask, colour, name in [(core, SERIES[0], "core"), (border, SERIES[1], "border"), (noise, MUTED, "noise")]:
        ax.scatter(p[mask, 0], p[mask, 1], s=70, color=colour, label=name, zorder=3)
        i = np.nonzero(mask)[0][-1]
        ax.add_patch(plt.Circle(p[i], eps, fill=False, edgecolor=colour, linestyle="--", linewidth=1.2))
        ax.text(p[i, 0] + 0.08, p[i, 1] + 0.08, f"{name}: {counts[i]} in reach", color=INK)
    ax.set_aspect("equal")
    ax.set_xlim(-3.2, 4.6)
    ax.set_ylim(-1.8, 2.9)
    ax.axis("off")
    ax.legend(loc="lower right")
    ax.set_title(f"eps = {eps:.0f}, MinPts = {min_pts}; dashed circle = eps")
    save(fig, 6, "concept_dbscan_types")


def shapes_cloud(n):
    d = 4
    G = o3d.geometry.TriangleMesh
    mesh = G.create_tetrahedron().translate((-d, 0, 0))
    mesh += G.create_octahedron().translate((0, 0, 0))
    mesh += G.create_icosahedron().translate((d, 0, 0))
    mesh += G.create_torus().translate((-d, -d, 0))
    mesh += G.create_mobius(twists=1).translate((0, -d, 0))
    mesh += G.create_mobius(twists=2).translate((d, -d, 0))
    return mesh.sample_points_uniformly(n)


def colour_scatter(ax, p, labels, title):
    cmap = plt.get_cmap("tab20")
    colours = cmap(labels % 20)
    colours[labels < 0] = (0.8, 0.8, 0.8, 1)
    ax.scatter(p[:, 0], p[:, 1], color=colours, s=3, linewidths=0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)


def ch6_clusters():
    from sklearn.cluster import KMeans
    pcl = o3d.geometry.TriangleMesh.create_box().sample_points_uniformly(10000)
    pcl.estimate_normals()
    xyz, n = np.asarray(pcl.points), np.asarray(pcl.normals)
    faces = np.argmin(np.abs(np.concatenate((xyz, 1 - xyz), axis=1)), axis=1)
    km = KMeans(n_clusters=6, init="random", n_init=10, max_iter=300, tol=1e-04, random_state=0)
    fig = plt.figure(figsize=(17, 4.6))
    cmap = plt.get_cmap("tab20")
    for i, (name, F) in enumerate([("xyz", xyz), ("normals", n), ("xyz + normals", np.concatenate((xyz, n), 1)),
                                   ("xyz + 0.5 normals", np.concatenate((xyz, 0.5 * n), 1)), ("xyz + |normals|", np.concatenate((xyz, np.abs(n)), 1))]):
        labels = km.fit_predict(F)
        purity = sum(np.bincount(faces[labels == k]).max() for k in range(6)) / len(labels)
        print(f"  cube {name}: purity {purity:.3f}")
        ax = fig.add_subplot(1, 5, i + 1, projection="3d")
        ax.scatter(xyz[::4, 0], xyz[::4, 1], xyz[::4, 2], color=cmap(labels[::4] * 2 % 20), s=2, linewidths=0)
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(f"{name}\npurity {purity:.2f}")
    save(fig, 6, "concept_cube_kmeans")

    shapes = shapes_cloud(10000)
    p = np.asarray(shapes.points)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    colour_scatter(axes[0], p, KMeans(n_clusters=2, init="random", n_init=10, random_state=0).fit_predict(p), "k-means, k = 2: the two rows")
    colour_scatter(axes[1], p, KMeans(n_clusters=6, init="random", n_init=10, random_state=0).fit_predict(p), "k-means, k = 6: the six figures")
    colour_scatter(axes[2], p, np.array(shapes.cluster_dbscan(eps=0.5, min_points=5)), "DBSCAN, eps = 0.5: six clusters, no k needed")
    save(fig, 6, "concept_shapes")

    frag = o3d.io.read_point_cloud(str(EX6 / "TestData/fragment.ply")).voxel_down_sample(0.02)
    q = np.asarray(frag.points)
    labels_d = np.array(frag.cluster_dbscan(eps=0.03, min_points=10))
    print(f"  fragment: {len(q)} points; DBSCAN eps 0.03: {labels_d.max() + 1} clusters, noise {np.mean(labels_d < 0):.0%}")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    axes[0].scatter(q[:, 0], q[:, 1], c=np.asarray(frag.colors), s=2, linewidths=0)
    axes[0].set_aspect("equal")
    axes[0].axis("off")
    axes[0].set_title("fragment.ply, its own colours")
    colour_scatter(axes[1], q, KMeans(n_clusters=8, init="random", n_init=5, random_state=0).fit_predict(q), "k-means on xyz, k = 8")
    colour_scatter(axes[2], q, labels_d, "DBSCAN, eps = 0.03 (gray = noise)")
    save(fig, 6, "concept_fragment")


# ---------------------------------------------------------------- chapter 7
CH7_NOTEBOOKS = [
    (EX7 / "Exercise1 - Histogram filter.ipynb", [("#### The 2-D histogram filter", "ex1_2d_belief")]),
    (EX7 / "Exercise2 - Kalman Filter.ipynb", [("#### 2a", "ex2_2a_update"), ("#### 2b", "ex2_2b_filter"), ("#### 2d", "ex2_2d_path")]),
    (W7 / "Project week 7.ipynb", [("#### Results", "wp7_frames"), ("#### Results@1", "wp7_path")], {"show_video = True": "show_video = False"}),
]


def ch7_histogram_cycle():
    world = ["green", "red", "red", "green", "green"]
    p = np.full(5, 0.2)

    def sense(p, z):
        q = p * np.array([0.6 if w == z else 0.2 for w in world])
        return q / q.sum()

    def move(p, u):
        q = np.roll(p, u)
        return 0.8 * q + 0.1 * np.roll(q, 1) + 0.1 * np.roll(q, -1)

    states = [("start: all cells equal", p)]
    p = sense(p, "red")
    states.append(("sense red", p))
    p = move(p, 1)
    states.append(("move 1 to the right", p))
    p = sense(p, "red")
    states.append(("sense red again", p))
    print("  5-cell filter:", {k: np.round(v, 3).tolist() for k, v in states})
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2), sharey=True)
    for ax, (title, q) in zip(axes, states):
        ax.bar(range(5), q, color=[CHANNEL["R"] if w == "red" else CHANNEL["G"] for w in world], width=0.9)
        for i, v in enumerate(q):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", color=INK)
        ax.set_xticks(range(5))
        ax.set_title(title)
        ax.set_xlabel("cell (bar colour = colour of the cell)")
    axes[0].set_ylabel("probability")
    axes[0].set_ylim(0, 0.65)
    save(fig, 7, "concept_histogram_cycle")


def gauss(x, m, s2):
    return np.exp(-(x - m) ** 2 / (2 * s2)) / np.sqrt(2 * np.pi * s2)


def ch7_gaussians():
    x = np.linspace(0, 30, 800)
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
    for ax, (m, s2, v, r2), lim in [(axes[0], (10, 4, 12, 4), (2, 20)), (axes[1], (10, 8, 13, 2), (2, 20))]:
        mu = (r2 * m + s2 * v) / (s2 + r2)
        var = s2 * r2 / (s2 + r2)
        ax.plot(x, gauss(x, m, s2), color=SERIES[0], label=f"prior: {m}, {s2}")
        ax.plot(x, gauss(x, v, r2), color=SERIES[1], linestyle="--", label=f"measurement: {v}, {r2}")
        ax.plot(x, gauss(x, mu, var), color=SERIES[2], linestyle="-.", label=f"posterior: {mu:.1f}, {var:.1f}")
        ax.set_xlim(*lim)
        ax.set_title("measurement update")
        ax.legend()
    ax = axes[2]
    ax.plot(x, gauss(x, 8, 4), color=SERIES[0], label="prior: 8, 4")
    ax.plot(x, gauss(x, 18, 10), color=SERIES[1], linestyle="--", label="after motion +10 (noise 6): 18, 10")
    ax.set_xlim(0, 30)
    ax.set_title("prediction: the means add, the variances add")
    ax.legend()
    for ax in axes:
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_xlabel("position")
    axes[0].set_ylabel("probability density")
    save(fig, 7, "concept_gaussians")


def ch7_gain():
    x, P = np.zeros((2, 1)), np.eye(2) * 1000.0
    F, H, R = np.array([[1.0, 1], [0, 1]]), np.array([[1.0, 0]]), 1.0
    gains, pos, vel, est = [], [], [], []
    for z in [1, 2.1, 2.9, 4.1, 5.0, 5.9, 7.1, 8.1]:
        S = H @ P @ H.T + R
        K = P @ H.T / S
        x = x + K * (z - H @ x)
        P = (np.eye(2) - K @ H) @ P
        gains.append(K.ravel().copy())
        est.append(x[1, 0])
        x, P = F @ x, F @ P @ F.T
        pos.append(P[0, 0])
        vel.append(P[1, 1])
    gains = np.array(gains)
    steps = np.arange(1, 9)
    print("  gains", np.round(gains, 3).tolist())
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.6))
    axes[0].plot(steps, gains[:, 0], color=SERIES[0], marker="o", markersize=6, label="gain for position")
    axes[0].plot(steps, gains[:, 1], color=SERIES[1], marker="o", markersize=6, linestyle="--", label="gain for velocity")
    axes[0].set_title("Kalman gain: trust in new measurements")
    axes[0].legend()
    axes[1].plot(steps, pos, color=SERIES[0], marker="o", markersize=6, label="position")
    axes[1].plot(steps, vel, color=SERIES[1], marker="o", markersize=6, linestyle="--", label="velocity")
    axes[1].set_yscale("log")
    axes[1].set_title("predicted variance (log scale)")
    axes[1].legend()
    axes[2].plot(steps, est, color=SERIES[0], marker="o", markersize=6)
    axes[2].axhline(1, color=INK2, linestyle="--", linewidth=1.2)
    axes[2].set_title("estimated velocity (dashed: true value 1)")
    for ax in axes:
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_xlabel("measurement number")
    save(fig, 7, "concept_kalman_gain")


def ch7_ekf():
    px = np.linspace(-1, 8, 200)
    py = 2.0
    h = np.sqrt(px ** 2 + py ** 2)
    x0 = 3.0
    r0 = np.sqrt(x0 ** 2 + py ** 2)
    slope = x0 / r0
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))
    ax = axes[0]
    ax.plot(px, h, color=SERIES[0], label="true range h(x)")
    ax.plot(px, r0 + slope * (px - x0), color=SERIES[1], linestyle="--", label=f"tangent at x = {x0:.0f}: slope {slope:.2f} (the Jacobian)")
    ax.scatter([x0], [r0], s=60, color=INK, zorder=3)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_xlabel("position x of the target (y = 2)")
    ax.set_ylabel("range r")
    ax.set_title("EKF: near the estimate, h is replaced by its tangent")
    ax.legend()
    ax = axes[1]
    rng = np.random.default_rng(0)
    samples = rng.normal(x0, 1.0, 200000)
    bins = np.linspace(0.5, 7.5, 80)
    ax.hist(np.sqrt(samples ** 2 + py ** 2), bins=bins, density=True, color=SERIES[0], alpha=0.35, label="range of a Gaussian x (true)")
    ax.plot(bins, gauss(bins, r0, slope ** 2), color=SERIES[1], linestyle="--", label="Gaussian from the tangent (EKF)")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_xlabel("range r")
    ax.set_title("the tangent's Gaussian only approximates the result")
    ax.legend()
    save(fig, 7, "concept_ekf")


def ch7_particles():
    length = 100.0
    doors = np.array([15.0, 40.0, 75.0])
    rng = np.random.default_rng(4)
    true_x = 15.0
    n = 2000
    particles = rng.uniform(0, length, n)

    def at_door(x):
        d = np.abs(x[:, None] - doors[None])
        return np.minimum(d, length - d).min(axis=1)

    def weigh(x, sees_door):
        like = 0.05 + 0.95 * np.exp(-at_door(x) ** 2 / (2 * 3.0 ** 2))
        return like if sees_door else 1.05 - like

    def resample(x, w):
        return x[rng.choice(len(x), len(x), p=w / w.sum())]

    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8), sharey=True)
    bins = np.linspace(0, length, 101)
    stages = []
    particles = resample(particles, weigh(particles, True))
    stages.append(("sees a door: one mode per door", particles.copy(), true_x))
    for step in (25.0,):
        true_x = (true_x + step) % length
        particles = (particles + step + rng.normal(0, 2.0, n)) % length
        particles = resample(particles, weigh(particles, True))
    stages.append(("moves 25, sees a door again", particles.copy(), true_x))
    true_x = (true_x + 35.0) % length
    particles = (particles + 35.0 + rng.normal(0, 2.0, n)) % length
    particles = resample(particles, weigh(particles, True))
    stages.append(("moves 35, sees a door again", particles.copy(), true_x))
    for ax, (title, q, tx) in zip(axes, stages):
        ax.hist(q, bins=bins, color=SERIES[0], alpha=0.5)
        for door in doors:
            ax.axvline(door, color=INK2, linestyle=":", linewidth=1.2)
        ax.axvline(tx, color=SERIES[7], linewidth=1.8)
        ax.axvline(q.mean(), color=SERIES[1], linestyle="--", linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("position (dotted: doors, red: true, dashed orange: mean of the particles)")
        print(f"  particles, {title}: mean {q.mean():.1f}, true {tx:.0f}, share within 5 of the truth {np.mean(np.abs(q - tx) < 5):.0%}")
    axes[0].set_ylabel("number of particles")
    save(fig, 7, "concept_particles")


# ---------------------------------------------------------------- chapter 9
EX9 = ROOT / "Class_9_visual_odom" / "Day 17 Exercises"
WP9 = ROOT / "Class_9_weekly_project" / "Day 18 weekly project"
WP9_WINDOWS_OFF = {'cv2.imshow("Trajectory", vis)': "pass", "k = cv2.waitKey(1) & 0xFF": "k = 0",
                   "cv2.waitKey(0)": "pass", "cv2.destroyAllWindows()": "pass"}
CH9_NOTEBOOKS = [(EX9 / "Exercise 1.ipynb", [("#### 1c", "ex1_1c_path")]),
                 (WP9 / "weekly project.ipynb", [("#### Result", "wp_result")], WP9_WINDOWS_OFF)]
K9 = np.array([[7.188560e+02, 0, 6.071928e+02], [0, 7.188560e+02, 1.852157e+02], [0, 0, 1]])
LK9 = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def vo_step(t):
    """One step of the exercise: track the points of frame t-1 into frame t, then PnP with RANSAC."""
    previous, current = np.load(EX9 / f"img_{t - 1}.npy"), np.load(EX9 / f"img_{t}.npy")
    world, before = np.load(EX9 / f"landmark_{t - 1}.npy"), np.load(EX9 / f"reference_2D_{t - 1}.npy")
    after, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, before, None, **LK9)
    found = status.ravel() == 1
    world, before, after = world[found], before[found], after[found]
    _, rvec, tvec, inliers = cv2.solvePnPRansac(world, after, K9, None)
    R, _ = cv2.Rodrigues(rvec)
    uv = (K9 @ (R @ world.T + tvec)).T
    errors = np.linalg.norm(uv[:, :2] / uv[:, 2:] - after, axis=1)
    return current, world, before, after, R, tvec, inliers.ravel(), errors


def ch9_tracking():
    current, world, before, after, R, tvec, inliers, errors = vo_step(3)
    is_inlier = np.zeros(len(after), bool)
    is_inlier[inliers] = True
    print(f"  step 3: {len(after)} tracked, {is_inlier.sum()} inliers; mean flow {np.linalg.norm(after - before, axis=1).mean():.1f} px; "
          f"errors: inliers mean {errors[is_inlier].mean():.2f} px, outliers median {np.median(errors[~is_inlier]):.1f} px")
    fig, ax = plt.subplots(figsize=(15, 4.8))
    ax.imshow(current, cmap="gray", vmin=0, vmax=255)
    for (a, b), good in zip(zip(before, after), is_inlier):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=SERIES[2] if good else SERIES[7], linewidth=1.3 if good else 2)
    ax.scatter(after[~is_inlier, 0], after[~is_inlier, 1], s=26, color=SERIES[7], label="outliers", zorder=3)
    ax.scatter(after[is_inlier, 0], after[is_inlier, 1], s=8, color=SERIES[2], label="inliers", zorder=3)
    ax.axis("off")
    ax.legend(loc="lower right", facecolor=SURFACE, frameon=True)
    ax.set_title("frame 3: lines show how far each tracked point moved since frame 2")
    save(fig, 9, "concept_tracking")


def ch9_reprojection():
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))
    _, world, _, after, R, tvec, inliers, errors = vo_step(3)
    bins = np.linspace(0, 12, 49)
    axes[0].hist(np.minimum(errors, 11.99), bins=bins, color=SERIES[0], alpha=0.6)
    axes[0].axvline(8, color=INK, linestyle="--", linewidth=1.3)
    axes[0].text(8.1, axes[0].get_ylim()[1] * 0.9, "8 px threshold", color=INK)
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", color=GRID, linewidth=0.6)
    axes[0].set_xlabel("reprojection error [px] (the last bin holds everything above 12 px)")
    axes[0].set_ylabel("number of points (log scale)")
    axes[0].set_title(f"step 3: {len(inliers)} of {len(errors)} points are inliers")
    # how much RANSAC matters: replace a share of the image points by random positions
    rng = np.random.default_rng(0)
    clean = (-R.T @ tvec).ravel()
    shares = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    plain, robust = [], []
    for share in shares:
        p_err, r_err = [], []
        for _ in range(15):
            noisy = after.copy()
            bad = rng.choice(len(noisy), int(share * len(noisy)), replace=False)
            noisy[bad] = rng.uniform((0, 0), (1241, 376), (len(bad), 2))
            _, rv, tv = cv2.solvePnP(world, noisy, K9, None)
            p_err.append(np.linalg.norm((-cv2.Rodrigues(rv)[0].T @ tv).ravel() - clean))
            _, rv, tv, _ = cv2.solvePnPRansac(world, noisy, K9, None)
            r_err.append(np.linalg.norm((-cv2.Rodrigues(rv)[0].T @ tv).ravel() - clean))
        plain.append(np.median(p_err))
        robust.append(np.median(r_err))
    print("  wrong matches", shares, "| position error without RANSAC", np.round(plain, 3).tolist(), "| with RANSAC", np.round(robust, 3).tolist())
    x = np.array(shares) * 100
    axes[1].plot(x, plain, color=SERIES[1], linestyle="--", marker="o", markersize=6, label="PnP on all points")
    axes[1].plot(x, robust, color=SERIES[0], marker="o", markersize=6, label="PnP with RANSAC")
    axes[1].set_yscale("log")
    axes[1].grid(color=GRID, linewidth=0.6)
    axes[1].set_xlabel("share of image points replaced by random positions [%]")
    axes[1].set_ylabel("error in the camera position")
    axes[1].set_title("wrong matches: median of 15 trials")
    axes[1].legend()
    save(fig, 9, "concept_reprojection")


def ch9_scene():
    path, clouds = [], []
    for t in range(1, 6):
        _, world, _, _, R, tvec, _, _ = vo_step(t)
        path.append((-R.T @ tvec).ravel())
        clouds.append(world)
    path = np.array(path)
    points = np.vstack(clouds)
    print(f"  landmarks: {len(points)} in total; depth percentiles 5/50/95 {np.percentile(points[:, 2], [5, 50, 95]).round(1)}; x percentiles 5/95 {np.percentile(points[:, 0], [5, 95]).round(1)}")
    fig, ax = plt.subplots(figsize=(7, 6))
    keep = (np.abs(points[:, 0]) < 40) & (points[:, 2] < 60)
    ax.scatter(points[keep, 0], points[keep, 2], s=4, color=AXIS, label="landmarks")
    ax.plot(path[:, 0], path[:, 2], color=SERIES[0], marker="o", markersize=6, label="camera")
    ax.set_aspect("equal")
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("top view: 3-D landmarks and the five camera positions")
    ax.legend(loc="lower right")
    save(fig, 9, "concept_scene")


def ch9_drift():
    rng = np.random.default_rng(0)
    steps, runs, length = 300, 40, 0.7
    ends = np.zeros((runs, steps + 1, 2))
    for r in range(runs):
        pos, heading = np.zeros(2), 0.0
        for k in range(steps):
            heading += np.radians(rng.normal(0, 0.05))          # small error in each relative rotation
            d = length * (1 + rng.normal(0, 0.01))              # 1 % error in each relative translation
            pos = pos + d * np.array([np.sin(heading), np.cos(heading)])
            ends[r, k + 1] = pos
    ideal = np.arange(steps + 1) * length
    err = np.linalg.norm(ends - np.stack([np.zeros(steps + 1), ideal], axis=1), axis=2)
    med = np.median(err, axis=0)
    print(f"  drift: median end error after 50 / 150 / 300 steps: {med[50]:.2f} / {med[150]:.2f} / {med[300]:.2f} over paths of {ideal[50]:.0f} / {ideal[150]:.0f} / {ideal[300]:.0f}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    for r in range(runs):
        axes[0].plot(ends[r, :, 0], ends[r, :, 1], color=SERIES[0], linewidth=0.8, alpha=0.5)
    axes[0].plot([0, 0], [0, ideal[-1]], color=INK, linestyle="--", linewidth=1.3, label="true path")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("z")
    axes[0].set_title("40 simulated runs: each pose is built on the previous one")
    axes[0].legend(loc="lower right")
    axes[1].plot(np.arange(steps + 1), med, color=SERIES[0])
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("distance from the true position")
    axes[1].set_title("median error grows faster than the path length")
    for ax in axes:
        ax.grid(color=GRID, linewidth=0.6)
    save(fig, 9, "concept_drift")


def ch9_two_view():
    """2-D to 2-D between two frames, for comparison with the exercise's PnP (book only)."""
    for t in (2, 3, 4, 5):
        _, world, before, after, R, tvec, _, _ = vo_step(t)
        _, _, _, _, R0, tvec0, _, _ = vo_step(t - 1)
        pnp_step = (-R.T @ tvec).ravel() - (-R0.T @ tvec0).ravel()
        E, mask = cv2.findEssentialMat(before.astype(np.float64), after.astype(np.float64), K9, cv2.RANSAC, 0.999, 1.0)
        _, R2, t2, _ = cv2.recoverPose(E, before.astype(np.float64), after.astype(np.float64), K9, mask=mask)
        direction = (-R2.T @ t2).ravel()
        cosine = abs(direction @ pnp_step) / np.linalg.norm(pnp_step)
        print(f"  frames {t - 1}-{t}: 2-D to 2-D translation length {np.linalg.norm(t2):.3f} (always 1), direction {np.round(direction, 3).tolist()}, inliers {int(mask.sum())}/{len(mask)}; "
              f"PnP step length {np.linalg.norm(pnp_step):.3f}; angle between the directions {np.degrees(np.arccos(np.clip(cosine, -1, 1))):.2f} deg")


def wp9_data():
    lefts = [cv2.imread(str(p), 0) for p in sorted((WP9 / "left").glob("*.png"))]
    rights = [cv2.imread(str(p), 0) for p in sorted((WP9 / "right").glob("*.png"))]
    sift = cv2.SIFT_create()
    features = [(sift.detectAndCompute(a, None), sift.detectAndCompute(b, None)) for a, b in zip(lefts, rights)]
    return lefts, features, np.genfromtxt(WP9 / "00.txt", delimiter=" ")


def wp9_stereo(feature, ratio=0.75):
    """Steps 1.2, 1.3 and 2.4: 3-D points in camera coordinates, with their positions in the left image."""
    (k1, d1), (k2, d2) = feature
    good = [m for m, n in cv2.BFMatcher().knnMatch(d1, d2, k=2) if m.distance < ratio * n.distance]
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    left_matrix = K9 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    right_matrix = K9 @ np.hstack((np.eye(3), np.array([[-0.54, 0, 0]]).T))
    points = cv2.triangulatePoints(left_matrix, right_matrix, p1.T, p2.T)
    return (points[:3] / points[3]).T, p1


def ch9_wp_variants():
    lefts, features, true = wp9_data()
    def run(ratio=0.75, ransac=True, status=True):
        world, points = wp9_stereo(features[0], ratio)
        out, turn, counts = [], [], []
        for i in range(1, 101):
            tracked, found, _ = cv2.calcOpticalFlowPyrLK(lefts[i - 1], lefts[i], points, None, **LK9)
            keep = found.ravel() == 1 if status else np.ones(len(points), bool)
            if ransac:
                _, rvec, tvec, inliers = cv2.solvePnPRansac(world[keep], tracked[keep], K9, None)
                counts.append(len(inliers))
            else:
                _, rvec, tvec = cv2.solvePnP(world[keep], tracked[keep], K9, None)
                counts.append(keep.sum())
            R, _ = cv2.Rodrigues(rvec)
            centre = -R.T @ tvec
            out.append(centre.ravel())
            pose = true[i].reshape(3, 4)
            turn.append(np.degrees(np.arccos(np.clip((np.trace(R @ pose[:3, :3]) - 1) / 2, -1, 1))))
            fresh, points = wp9_stereo(features[i], ratio)
            world = (np.hstack((R.T, centre)) @ np.vstack((fresh.T, np.ones(len(fresh))))).T
        errors = np.linalg.norm(np.array(out) - true[1:101][:, [3, 7, 11]], axis=1)
        return errors, turn, counts
    for name, options in (("ratio 0.6", dict(ratio=0.6)), ("ratio 0.7", dict(ratio=0.7)), ("ratio 0.75", dict()), ("ratio 0.8", dict(ratio=0.8)),
                          ("ratio 0.9", dict(ratio=0.9)), ("no RANSAC", dict(ransac=False)), ("no status filter", dict(status=False))):
        errors, turn, counts = run(**options)
        print(f"  {name}: end {errors[-1]:.2f} m, largest {errors.max():.2f} m, rotation at the end {turn[-1]:.2f} deg, median inliers {np.median(counts):.0f}")
        if not options:
            print(f"    error at frames 14, 25, 50, 75, 100: {errors[[13, 24, 49, 74, 99]].round(2)}")


def ch9_wp_flow_check():
    """Compares the image motion with what the true pose and the stereo depth predict, frame by frame."""
    lefts, features, true = wp9_data()
    def pose(i):
        matrix = np.eye(4)
        matrix[:3] = true[i].reshape(3, 4)
        return matrix
    flow, steps = [], []
    for i in range(1, 101):
        world, points = wp9_stereo(features[i - 1])
        tracked, found, _ = cv2.calcOpticalFlowPyrLK(lefts[i - 1], lefts[i], points, None, **LK9)
        keep = found.ravel() == 1
        world, points, tracked = world[keep], points[keep], tracked[keep]
        _, _, tvec, inliers = cv2.solvePnPRansac(world, tracked, K9, None)
        inliers = inliers.ravel()
        world, points, tracked = world[inliers], points[inliers], tracked[inliers]
        relative = np.linalg.inv(pose(i)) @ pose(i - 1)
        moved = (K9 @ (relative[:3, :3] @ world.T + relative[:3, 3:])).T
        predicted = moved[:, :2] / moved[:, 2:]
        size = np.linalg.norm(predicted - points, axis=1)
        chosen = size > 1
        flow.append(np.median(np.linalg.norm(tracked[chosen] - points[chosen], axis=1) / size[chosen]))
        steps.append(np.linalg.norm(tvec) / np.linalg.norm(relative[:3, 3]))
    flow, steps = np.array(flow), np.array(steps)
    print(f"  measured / predicted image motion: frames 1-5 {np.median(flow[:5]):.2f}, 6-14 {np.median(flow[5:14]):.2f}, 15-30 {np.median(flow[14:30]):.2f}, 31-100 {np.median(flow[30:]):.2f}")
    print(f"  estimated / true step: frames 1-5 {np.median(steps[:5]):.2f}, 6-14 {np.median(steps[5:14]):.2f}, 15-30 {np.median(steps[14:30]):.2f}, 31-100 {np.median(steps[30:]):.2f}")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    frames = np.arange(1, 101)
    ax.plot(frames, flow, color=SERIES[0], label="measured / predicted image motion")
    ax.plot(frames, steps, color=SERIES[1], linestyle="--", label="estimated / true step length")
    ax.axhline(1, color=AXIS, linewidth=1)
    ax.set_ylim(0.5, 1.3)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_xlabel("frame")
    ax.set_ylabel("ratio")
    ax.set_title("the image motion agrees with 00.txt, except in the first frames")
    ax.legend(loc="lower right")
    save(fig, 9, "wp_flow_check")


# ---------------------------------------------------------------- chapter 8
EX8 = ROOT / "Class_8_classification" / "week8"
WP8 = ROOT / "Class_8_weekly_project"
CH8_NOTEBOOKS = [
    (EX8 / "Exercise1.ipynb", [("#### Normalization", "ex1_scaled"), ("#### LinearSVC", "ex1_linearsvc"),
                               ("#### 1) Kernels@1", "ex1_rbf"), ("#### 2) Three features", "ex1_3d")]),
    (EX8 / "Exercise2.ipynb", [("#### Plotting both", "ex2_compare"),
                               ("#### 1) SVM on the principal components", "ex2_fit"),
                               ("#### 2) Three principal components", "ex2_3d")]),
    (WP8 / "WeeklyProject.ipynb", [("code:show_some_digits(images, targets, 12)", "wp_digits"),
                                   ("#### Confusion matrix", "wp_confusion"), ("#### Wrong digits", "wp_wrong")]),
]


def ch8_boundaries(ax, data, labels, model, colours):
    """Fill the classes that a fitted model predicts, and draw the samples over them."""
    xx, yy = np.meshgrid(np.linspace(data[:, 0].min() - 0.5, data[:, 0].max() + 0.5, 300),
                         np.linspace(data[:, 1].min() - 0.5, data[:, 1].max() + 0.5, 300))
    zone = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    for k, colour in enumerate(colours):
        ax.contourf(xx, yy, zone == k, levels=[0.5, 1], colors=[colour], alpha=0.16)
        ax.scatter(*data[labels == k].T, s=18, color=colour, edgecolor=SURFACE, linewidth=0.4)


def ch8_pca():
    rng = np.random.default_rng(0)
    data = rng.multivariate_normal([0, 0], [[3, 1.6], [1.6, 1.2]], 200)
    centred = data - data.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(centred.T))
    values, vectors = values[::-1], vectors[:, ::-1]
    share = values / values.sum()
    print(f"  PCA: eigenvalues {values.round(2)}, shares {share.round(3)}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    axes[0].scatter(*centred.T, s=12, color=AXIS)
    for k, colour in enumerate(SERIES[:2]):
        tip = vectors[:, k] * 2 * np.sqrt(values[k])
        axes[0].annotate("", xy=tip, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.5))
        axes[0].text(*(tip * 1.12), f"component {k + 1}: {share[k] * 100:.0f} %", color=INK, ha="center")
    axes[0].set_aspect("equal")
    axes[0].set_title("eigenvectors of the covariance matrix")
    axes[1].scatter(centred @ vectors[:, 0], np.zeros(len(centred)), s=14, color=SERIES[0], alpha=0.5)
    axes[1].set_yticks([])
    axes[1].spines["left"].set_visible(False)
    axes[1].set_xlabel("position along component 1")
    axes[1].set_title(f"each sample reduced to one number, keeping {share[0] * 100:.0f} % of the variance")
    save(fig, 8, "concept_pca")


def ch8_knn():
    from sklearn.datasets import load_iris
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    iris = load_iris()
    data = StandardScaler().fit_transform(iris.data[:, :2])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, k in zip(axes, (1, 15, 75)):
        ch8_boundaries(ax, data, iris.target, KNeighborsClassifier(n_neighbors=k).fit(data, iris.target), SERIES[:3])
        ax.set_title(f"k = {k}")
        ax.set_xlabel("sepal length (scaled)")
    axes[0].set_ylabel("sepal width (scaled)")
    save(fig, 8, "concept_knn")


def ch8_margin():
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    iris = load_iris()
    keep = iris.target < 2
    data = StandardScaler().fit_transform(iris.data[keep][:, [0, 1]])
    labels = iris.target[keep]
    model = SVC(kernel="linear", C=1000).fit(data, labels)
    w, b = model.coef_[0], model.intercept_[0]
    print(f"  margin: {len(model.support_)} support vectors of {len(data)}, margin width 2/|w| = {2 / np.linalg.norm(w):.3f}")
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, colour in enumerate(SERIES[:2]):
        ax.scatter(*data[labels == k].T, s=24, color=colour, label=iris.target_names[k])
    x = np.array([data[:, 0].min(), data[:, 0].max()])
    for level, style, label in ((0, "-", "boundary"), (1, "--", "margin"), (-1, "--", None)):
        ax.plot(x, (level - b - w[0] * x) / w[1], color=INK, linestyle=style, linewidth=1.3, label=label)
    ax.scatter(*data[model.support_].T, s=110, facecolor="none", edgecolor=INK, linewidth=1.3, label="support vectors")
    ax.set_ylim(data[:, 1].min() - 0.3, data[:, 1].max() + 0.3)
    ax.set_xlabel("sepal length (scaled)")
    ax.set_ylabel("sepal width (scaled)")
    ax.legend(loc="lower right")
    ax.set_title("only the support vectors define the boundary")
    save(fig, 8, "concept_margin")


def ch8_kernels():
    from sklearn.svm import SVC
    rng = np.random.default_rng(0)
    angle = rng.uniform(0, 2 * np.pi, 300)
    labels = (np.arange(300) % 2)
    radius = np.where(labels == 0, 1.0, 2.5) + rng.normal(0, 0.2, 300)
    data = np.c_[radius * np.cos(angle), radius * np.sin(angle)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, (name, model) in zip(axes, (("linear", SVC(kernel="linear")), ("RBF", SVC(kernel="rbf", gamma="scale")))):
        model.fit(data, labels)
        ch8_boundaries(ax, data, labels, model, SERIES[:2])
        ax.set_aspect("equal")
        ax.set_title(f"{name} kernel: {model.score(data, labels) * 100:.0f} % correct")
        print(f"  kernels: {name} {model.score(data, labels):.3f}")
    save(fig, 8, "concept_kernels")


def ch8_nms():
    def iou(a, b):
        w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
        h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
        overlap = max(w, 0) * max(h, 0)
        return overlap / (a[2] * a[3] + b[2] * b[3] - overlap)
    boxes = [(10, 10, 40, 40, 0.95), (15, 14, 40, 40, 0.80), (6, 6, 42, 42, 0.70), (70, 18, 34, 34, 0.90), (75, 22, 34, 34, 0.60)]
    kept, rest = [], sorted(boxes, key=lambda b: -b[4])
    while rest:
        best = rest.pop(0)
        kept.append(best)
        rest = [b for b in rest if iou(best, b) <= 0.5]
    print(f"  NMS: IoU of the first two boxes {iou(boxes[0], boxes[1]):.2f}; {len(boxes)} boxes -> {len(kept)}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, shown, title in ((axes[0], boxes, "before: many windows fire on each object"), (axes[1], kept, "after non-maximum suppression (IoU > 0.5 removed)")):
        for x, y, w, h, score in shown:
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=SERIES[0], linewidth=1.6))
            ax.text(x + 1, y + h - 3, f"{score:.2f}", color=INK, fontsize=9)
        ax.set_xlim(0, 115)
        ax.set_ylim(65, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
    save(fig, 8, "concept_nms")


def ch8_mnist_pca():
    from sklearn.datasets import fetch_openml
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    mnist = fetch_openml("mnist_784", version=1)
    images, targets = mnist.data.to_numpy() / 255.0, mnist.target.to_numpy()
    train, test = images[:60000], images[60000:]
    pca = PCA(n_components=300, random_state=0).fit(train)
    total = np.cumsum(pca.explained_variance_ratio_)
    print(f"  MNIST: variance at 10/50/100/152/300 components {total[[9, 49, 99, 151, 299]].round(3)}")
    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(np.arange(1, 301), total * 100, color=SERIES[0])
    for n in (50, 152):
        ax.scatter(n, total[n - 1] * 100, s=50, color=SERIES[0], zorder=3)
        ax.annotate(f"{n} components: {total[n - 1] * 100:.0f} %", (n, total[n - 1] * 100), xytext=(8, -14), textcoords="offset points", color=INK)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_xlabel("number of components (the image has 784 pixels)")
    ax.set_ylabel("explained variance [%]")
    ax.set_title("how much variance the first components keep")
    sample = test[0:1]
    for k, n in enumerate((10, 50, 152, None)):
        panel = fig.add_subplot(1, 8, 5 + k)
        picture = sample[0]
        if n is not None:
            picture = pca.mean_ + (sample - pca.mean_) @ pca.components_[:n].T @ pca.components_[:n]
        panel.imshow(np.asarray(picture).reshape(28, 28), cmap="gray_r")
        panel.set_title("original" if n is None else f"{n} comp.")
        panel.axis("off")
    save(fig, 8, "concept_mnist_pca")
    low = PCA(n_components=50, random_state=0).fit(train)
    a, b = low.transform(train), low.transform(test)
    for k in (1, 5, 15):
        print(f"  MNIST k-NN on 50 components: k={k} accuracy {KNeighborsClassifier(n_neighbors=k).fit(a, targets[:60000]).score(b, targets[60000:]):.4f}")


# ---------------------------------------------------------------- chapter 10 (final project)
FINAL = ROOT / "Final_project"
FINAL_RESULTS = FINAL / "results"
FINAL_SEQUENCES = ["1_no_obstacle", "2_obstacle", "3_curved_bend", "4_circle"]


def ch10_saved_outputs():
    """The final project's notebook takes minutes to run, so its saved outputs are used as they are."""
    nb = nbformat.read(FINAL / "Final Project.ipynb", as_version=4)
    for key, name in (("#### 3.", "rectified_pair"), ("#### 5.", "detections"), ("#### 6.", "disparity_maps"),
                      ("#### 8.@0", "tracks_2d"), ("#### 8.@1", "tracks_3d"),
                      ("#### 10.@0", "blackout"), ("#### 10.@1", "occlusion")):
        png = cell_image(nb, key)
        if png is None:
            print(f"  WARNING: no image under '{key}'")
            continue
        out = FIGURES / "class10" / f"{name}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(png))
        report(out)


def ch10_corner_residuals():
    left, right = np.load(FINAL_RESULTS / "corners.npy", allow_pickle=True)
    grid = np.mgrid[0:7, 0:5].T.reshape(-1, 2).astype(np.float32)
    before, after, removed = [], [], []
    for corners in list(left) + list(right):
        if corners is None:
            continue
        H, _ = cv2.findHomography(grid, corners, 0)
        fit = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), H).reshape(-1, 2)
        before.append(np.sqrt(np.mean(np.sum((fit - corners) ** 2, axis=1))))
        H, mask = cv2.findHomography(grid, corners, cv2.RANSAC, 1.0)
        keep = mask.ravel() == 1
        fit = cv2.perspectiveTransform(grid[keep].reshape(-1, 1, 2), H).reshape(-1, 2)
        after.append(np.sqrt(np.mean(np.sum((fit - corners[keep]) ** 2, axis=1))))
        removed.append(35 - keep.sum())
    before, after = np.array(before), np.array(after)
    print(f"  corners: {len(before)} views; homography residual before: median {np.median(before):.2f}, max {before.max():.2f} px; after: median {np.median(after):.2f}, max {after.max():.2f} px; removed per view: median {np.median(removed):.0f}, max {np.max(removed)}")
    order = np.argsort(before)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(before[order], color=SERIES[1], marker="o", markersize=8, linestyle="none", label="all corners")
    ax.plot(after[order], color=SERIES[0], marker="o", markersize=8, linestyle="none", label="after removing the outliers with RANSAC")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_xticks([])
    ax.set_xlabel("calibration views, sorted by their error before")
    ax.set_ylabel("residual from one homography [px]")
    ax.legend()
    save(fig, 10, "corner_residuals")


def ch10_height_test():
    from sklearn import linear_model
    calibration = np.load(FINAL_RESULTS / "calibration.npz")
    height, disparity, source = [], [], []
    for k, name in enumerate(FINAL_SEQUENCES):
        rows = np.load(FINAL_RESULTS / f"{name}_detections.npy")
        centre = rows[:, 1] + rows[:, 3] / 2
        good = (rows[:, 0] >= 1000) & (rows[:, 1] > 58) & (rows[:, 1] + rows[:, 3] < 1277) & (rows[:, 5] > 0)
        if k == 1:
            good &= (centre < 230) | (centre > 1000)
        height.append(rows[good, 4])
        disparity.append(rows[good, 5])
        source.append(np.full(good.sum(), k))
    height, disparity, source = map(np.concatenate, (height, disparity, source))
    model = linear_model.RANSACRegressor(residual_threshold=8, random_state=0).fit(disparity.reshape(-1, 1), height)
    a, b = np.polyfit(disparity[model.inlier_mask_], height[model.inlier_mask_], 1)
    print(f"  height test: a {a:.2f}, b {b:.2f}, offset {-b / a:.2f} px; inliers {model.inlier_mask_.mean():.3f}")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for k, name in enumerate(FINAL_SEQUENCES):
        ax.scatter(disparity[source == k], height[source == k], s=10, color=SERIES[k], label=name)
    x = np.array([0, disparity.max()])
    ax.plot(x, a * x + b, color=INK, linewidth=1.3, label="fitted line")
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_ylim(-25, None)
    ax.set_xlim(0, None)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_xlabel("measured disparity [px]")
    ax.set_ylabel("height of the box [px]")
    ax.set_title(f"a rigid robot's line should pass through zero; it crosses at {-b / a:.2f} px")
    ax.legend()
    save(fig, 10, "height_test")


def ch10_scene():
    header = 10
    data = np.loadtxt(FINAL_RESULTS / "2_obstacle_scene.ply", skiprows=header)
    data = data[::3]
    track = np.genfromtxt(FINAL_RESULTS / "2_obstacle_track.csv", delimiter=",", skip_header=1)
    print(f"  scene: {len(data)} points drawn; depth percentiles 5/50/95 {np.percentile(data[:, 2], [5, 50, 95]).round(2)} m")
    fig, ax = plt.subplots(figsize=(10, 6))
    near = data[:, 2] < 14
    ax.scatter(data[near, 0], data[near, 2], s=1, c=data[near, 3:] / 255.0)
    ax.plot(track[:, 3], track[:, 5], color=SERIES[7], linewidth=2, label="the robot's 3-D track, including the occlusion")
    ax.scatter(0, 0, marker="^", s=70, color=INK, label="camera")
    ax.set_aspect("equal")
    ax.set_xlim(-9, 9)
    ax.set_ylim(-0.5, 14)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("sequence 2 from above: the coloured 3-D scene points and the track")
    ax.legend(loc="upper right", facecolor=SURFACE, frameon=True)
    save(fig, 10, "scene_top_view")


def ch10_video_frames():
    capture = cv2.VideoCapture(str(FINAL_RESULTS / "husky_tracking.mp4"))
    frames = []
    for i in (150, 256 + 136, 256 + 165, 256 + 291 + 230, 256 + 291 + 397 + 240, 256 + 291 + 397 + 400):
        capture.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame = capture.read()
        frames.append(frame[:, :, ::-1])
    capture.release()
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for ax, frame in zip(axes.ravel(), frames):
        ax.imshow(frame)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    save(fig, 10, "video_frames")


CHAPTERS = {
    10: ([], [ch10_saved_outputs, ch10_corner_residuals, ch10_height_test, ch10_scene, ch10_video_frames]),
    8: (CH8_NOTEBOOKS, [ch8_pca, ch8_knn, ch8_margin, ch8_kernels, ch8_nms, ch8_mnist_pca]),
    9: (CH9_NOTEBOOKS, [ch9_tracking, ch9_reprojection, ch9_scene, ch9_drift, ch9_two_view, ch9_wp_variants, ch9_wp_flow_check]),
    3: (CH3_NOTEBOOKS, [ch3_epipolar_geometry, ch3_depth_from_disparity, ch3_metrics, ch3_window_size,
                        ch3_stereobm_params, ch3_point_cloud]),
    4: (CH4_NOTEBOOKS, [ch4_distortion, ch4_views, ch4_fundamental_sweep]),
    5: (CH5_NOTEBOOKS, [ch5_registration, ch5_icp_steps, ch5_icp_limits, ch5_map, ch5_fpfh]),
    6: (CH6_NOTEBOOKS, [ch6_kmeans_steps, ch6_meanshift, ch6_dbscan_types, ch6_clusters]),
    7: (CH7_NOTEBOOKS, [ch7_histogram_cycle, ch7_gaussians, ch7_gain, ch7_ekf, ch7_particles]),
    1: (CH1_NOTEBOOKS, [ch1_bgr_vs_rgb, ch1_channels, ch1_geometry, ch1_gaussian_blur, ch1_sharpening,
                        ch1_mean_vs_median, ch1_morphology, ch1_connectivity, ch1_tetris_tutorial,
                        ch1_threshold_choice, ch1_border_frame, ch1_rgb_vs_hsv, ch1_hue_bands]),
    2: (CH2_NOTEBOOKS, [ch2_derivative_noise, ch2_gradients, ch2_canny_thresholds, ch2_harris_regions,
                        ch2_scale_space, ch2_hough_accumulator, ch2_ransac_line, ch2_match_quality,
                        ch2_flow_wheel]),
}


if __name__ == "__main__":
    chosen = [int(a) for a in sys.argv[1:]] or sorted(CHAPTERS)
    for chapter in chosen:
        notebooks, drawings = CHAPTERS[chapter]
        print(f"chapter {chapter}")
        notebook_figures(chapter, notebooks)             # first: some drawings read files the notebooks write
        for draw in drawings:
            print(f" {draw.__name__}")
            draw()
