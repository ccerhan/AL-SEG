import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from mmseg.visualization.local_visualizer import SegLocalVisualizer


def show_uncertainty_image(
        uncertainty_img: np.ndarray,
        show_delay: int = None,
        win_name: str = "DEBUG",
        save_path: str = None,
        colored: bool = False,
):
    img_cv = uncertainty_img
    if len(img_cv.shape) > 2:
        img_cv = np.transpose(img_cv, (1, 2, 0))
    img_cv = np.clip(img_cv * 255, 0, 255)
    img_cv = img_cv.astype(np.uint8)
    if colored:
        img_cv = cv2.applyColorMap(img_cv, cv2.COLORMAP_JET)
    if show_delay is not None and show_delay >= 0:
        cv2.imshow(win_name, img_cv)
        cv2.waitKey(show_delay)
    if save_path is not None:
        cv2.imwrite(save_path, img_cv)
    return img_cv


def create_debug_image(
        uncert_img,
        seg_result,
        ignore_mask,
        cooccur_pdf,
        cls_names,
        palette,
        uncert_map_scale=1.5,
        fontsize=6,
        dpi=200
):
    h, w = uncert_img.shape
    ratio = h / w
    size = 4
    img_size = (size, size * ratio)
    figsize = (img_size[0] + img_size[1], img_size[1])
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=(img_size[0], img_size[1]))
    ax_img = fig.add_subplot(gs[0, 0])
    ax_mat = fig.add_subplot(gs[0, 1], anchor="S")

    # Draw original image
    image = cv2.imread(seg_result.img_path)
    if ignore_mask is not None:
        image[ignore_mask, :] = 0

    # Draw labels and predictions
    vis = SegLocalVisualizer(alpha=0.7)
    gt = vis._draw_sem_seg(image, seg_result.gt_sem_seg, cls_names, palette, True)
    preds = vis._draw_sem_seg(image, seg_result.pred_sem_seg, cls_names, palette, False)
    if ignore_mask is not None:
        preds[ignore_mask] = 0

    # Draw risk map
    if ignore_mask is not None:
        uncert_img[ignore_mask] = 0
    uncert_img = show_uncertainty_image(uncert_img * uncert_map_scale, show_delay=None, colored=True)
    uncert_img = cv2.cvtColor(uncert_img, cv2.COLOR_BGR2RGB)

    top = np.hstack([image, uncert_img])
    bot = np.hstack([gt, preds])
    img = np.vstack([top, bot])

    ax_img.imshow(img)
    ax_img.set_yticks([])
    ax_img.set_xticks([])
    ax_img.set_yticklabels([])
    ax_img.set_xticklabels([])
    ax_img.set_aspect("equal", "box")

    h, w, c = gt.shape
    ax_img.text(0, 0, "Image", bbox=dict(facecolor="white", edgecolor="black", pad=2), fontsize=fontsize, va="top")
    ax_img.text(0, h, "Labels", bbox=dict(facecolor="white", edgecolor="black", pad=2), fontsize=fontsize, va="top")
    ax_img.text(w, 0, "Risk Map", bbox=dict(facecolor="white", edgecolor="black", pad=2), fontsize=fontsize, va="top")
    ax_img.text(w, h, "Prediction", bbox=dict(facecolor="white", edgecolor="black", pad=2), fontsize=fontsize, va="top")

    # Draw matrix
    reversed_cmap = plt.cm.get_cmap("rocket").reversed()
    # mat = ax_mat.matshow(conf_pdf, cmap=reversed_cmap)
    mat = ax_mat.matshow(cooccur_pdf, cmap=reversed_cmap, vmin=0, vmax=1)
    ax_mat.set_yticks(np.arange(len(cls_names)), [f"{n} - {c:2}" for c, n in enumerate(cls_names)], fontsize=fontsize)
    ax_mat.set_xticks(np.arange(len(cls_names)), [f"{c:2}" for c, n in enumerate(cls_names)], fontsize=fontsize)
    ax_mat.xaxis.set_tick_params(rotation=90, bottom=False)
    # ax_mat.xaxis.set_tick_params(bottom=False)
    cbar = ax_mat.figure.colorbar(mat, ax=ax_mat, aspect=35, shrink=0.8, orientation="vertical")
    # cbar.ax.set_yticks([0, 1])
    cbar.ax.tick_params(labelsize=fontsize - 1)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = cv2.cvtColor(np.asarray(canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    fig.clf()
    return img
