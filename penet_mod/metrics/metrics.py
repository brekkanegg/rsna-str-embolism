import numpy as np
from skimage.measure import regionprops, label
import cv2


def calc_iou(bbox_a, bbox_b):
    """
    :param a: bbox list [min_y, min_x, max_y, max_x]
    :param b: bbox list [min_y, min_x, max_y, max_x]
    :return:
    """
    size_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    size_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])

    min_ab_y = max(bbox_a[0], bbox_b[0])
    min_ab_x = max(bbox_a[1], bbox_b[1])
    max_ab_y = min(bbox_a[2], bbox_b[2])
    max_ab_x = min(bbox_a[3], bbox_b[3])

    inter_ab = max(0, max_ab_y - min_ab_y) * max(0, max_ab_x - min_ab_x)

    return inter_ab / (size_a + size_b - inter_ab)


def eval(args, pred, gt, iou_th=0.15, prob_ths=[0.5]):

    pred_score, pred_class, pred_bbox = pred

    pred_bboxes = {
        k: {thi: [] for thi in range(len(prob_ths))} for k in range(args.num_classes)
    }
    gt_bboxes = {k: [] for k in range(args.num_classes)}

    for bi in range(len(pred_bbox)):
        for thi in range(len(prob_ths)):
            if pred_score[bi] > prob_ths[thi]:
                pbi = [int(x) for x in pred_bbox[bi]]
                pred_bboxes[pred_class[bi]][thi].append(pbi)

    for g in gt:
        if g[-1] != -1:
            gb = [int(x) for x in g[:4]]
            gt_bboxes[g[-1]].append(gb)

    # 초기화
    gt_nums = np.array([len(gt_bboxes[c]) for c in range(args.num_classes)])
    pred_nums = np.zeros((len(prob_ths), args.num_classes))
    for thi in range(len(prob_ths)):
        for c in range(args.num_classes):
            pred_nums[thi, c] = len(pred_bboxes[c][thi])

    tp_nums = np.zeros((len(prob_ths), args.num_classes))
    fp_nums = pred_nums.copy()  # .copy() 없으면 포인터가 같아짐

    # Gt-Pred Bbox Iou Matrix
    for c in range(args.num_classes):
        for thi in range(len(prob_ths)):
            if (gt_nums[c] == []) or (pred_nums[thi][c] == 0):  # np array 이상함;
                continue

            iou_matrix = np.zeros((gt_nums[c], int(pred_nums[thi][c])))
            for gi, gr in enumerate(gt_bboxes[c]):
                for pi, pr in enumerate(pred_bboxes[c][thi]):
                    iou_matrix[gi, pi] = calc_iou(gr, pr)

            tp_nums[thi][c] = np.sum(np.any((iou_matrix >= iou_th), axis=1))
            fp_nums[thi][c] -= np.sum(np.any((iou_matrix > iou_th), axis=0))

    return gt_nums, pred_nums, tp_nums, fp_nums


def calc_metrics(tp_nums, gt_nums, fp_nums, nums_tot, num_classes, mod=0):
    if mod:
        fp_nums *= mod

    recall = tp_nums / (gt_nums + 1e-6)
    precision = tp_nums / (fp_nums + tp_nums + 1e-6)

    # NOTE: Not really f1
    f1 = (2 * recall * precision) / (recall + precision + 1e-6)
    fppi = fp_nums / (nums_tot + 1e-6)

    max_f1_index = np.argmax(f1, axis=0)

    f1 = np.max(f1, axis=0)
    recall = [recall[i, j] for i, j in zip(max_f1_index, range(num_classes))]
    precision = [precision[i, j] for i, j in zip(max_f1_index, range(num_classes))]
    fppi = [fppi[i, j] for i, j in zip(max_f1_index, range(num_classes))]

    return recall, precision, f1, fppi

