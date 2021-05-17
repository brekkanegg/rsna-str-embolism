import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import warnings


class Visualizer:
    def __init__(self, args, mode, save_dir, writer=None):
        self.args = args
        self.mode = mode
        self.save_dir = save_dir
        self.writer = writer

    def summary_fig(self, fp, img, pred, bbox, iteration, th=0.5):

        warnings.filterwarnings("ignore")

        gt_bbox, gt_class = bbox[:, :4], bbox[:, 4]
        pred_score, pred_class, pred_bbox = pred
        pred_idx = pred_score > th
        pred_score = pred_score[pred_idx]
        pred_class = pred_class[pred_idx]
        pred_bbox = pred_bbox[pred_idx]
        # print(pred_bbox)

        fp = fp.replace("/", "-")

        fig = plt.figure()
        fig.suptitle(fp)
        plt.axis("off")

        ax00 = fig.add_subplot(1, 3, 1)
        ax00.imshow(img, cmap="gray")
        # ax00.imshow(self.apply_windowing(img), cmap="gray")

        ax01 = fig.add_subplot(1, 3, 2)
        # ax01.imshow(self.apply_windowing(img), cmap="gray")
        ax01.imshow(img, cmap="gray")
        for i, (i_gtc, i_gtb) in enumerate(zip(gt_class, gt_bbox)):
            edgecolor = "g"
            if i_gtc > 0.5:
                edgecolor = "b"

            if i_gtc == -1:
                continue

            x0, y0, x1, y1 = [int(ii) for ii in i_gtb]
            w, h = (x1 - x0), (y1 - y0)
            rect = patches.Rectangle(
                (x0, y0), w, h, linewidth=1, edgecolor=edgecolor, facecolor="none"
            )

            ax01.add_patch(rect)

        # 겹치면- 분홍, seg만- 보라, pred만- 빨강
        ax02 = fig.add_subplot(1, 3, 3)
        # ax02.imshow(self.apply_windowing(img), cmap="gray")
        ax02.imshow(img, cmap="gray")
        for i, (i_ps, i_pc, i_pb) in enumerate(zip(pred_score, pred_class, pred_bbox)):
            x0, y0, x1, y1 = [int(ii) for ii in i_pb]
            w, h = (x1 - x0), (y1 - y0)
            rect = patches.Rectangle(
                (x0, y0), w, h, linewidth=1, edgecolor="r", facecolor="none"
            )
            i_ps = "{:.2f}".format(i_ps)
            plt.text(x0, y0, i_ps, bbox={"facecolor": "r", "alpha": 0.5, "pad": 0})
            ax02.add_patch(rect)

        self.writer.add_figure("{}_img".format(self.mode), fig, iteration, close=True)

    def apply_windowing(self, img):
        eps = 1e-6
        center = img.mean()
        width = img.std() * 4
        low = center - width / 2
        high = center + width / 2
        img = (img - low) / (high - low + eps)
        img[img < 0.0] = 0
        img[img > 1.0] = 1
        img = img * 255

        return img
