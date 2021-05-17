import json
import numpy as np
import torch
from collections import OrderedDict
from util import tools, visualizer, logger, writer
from opts import losses
from configs import *


class CommonerALL(object):
    def __init__(self, args):
        self.args = args
        with open(DATA_DIR + "/train_positive_ratio.json", "rb") as f:
            self.positive_ratio_dict = json.load(f)

    def calc_loss(self, fps, anns, outputs):

        """        
        "negative_exam_for_pe": 0.0736196319,
        "indeterminate": 0.09202453988,
        "chronic_pe": 0.1042944785,
        "acute_and_chronic_pe": 0.1042944785,
        "central_pe": 0.1877300613,
        "leftsided_pe": 0.06257668712,
        "rightsided_pe": 0.06257668712,
        "rv_lv_ratio_gte_1": 0.2346625767,
        "rv_lv_ratio_lt_1": 0.0782208589,
        """

        exam_label_weights = np.array(
            [
                [
                    0.0736196319,
                    0.09202453988,
                    0.1042944785,
                    0.1042944785,
                    0.1877300613,
                    0.06257668712,
                    0.06257668712,
                    0.2346625767,
                    0.0782208589,
                ]
            ]
        )
        exam_label_weights = torch.from_numpy(exam_label_weights).cuda()

        # TODO:              ##############################
        # batch_n, seq_n = anns.shape[:2]
        # anns_reshaped = torch.reshape(anns, (batch_n * seq_n, -1))
        # valid_idxs = anns_reshaped != -1
        # outputs_reshaped = torch.reshape(outputs, (batch_n * seq_n, -1))

        # anns = anns_reshaped[valid_idxs]
        # outputs = outputs_reshaped[valid_idxs]

        # Exam:  NOTE: batch_size = 1
        exam_anns = anns[:, :9, 0]
        exam_outputs = outputs[:, :9, 0]

        f_len = anns.shape[1] - 9
        image_anns = anns[:, 9:, 0]
        image_outputs = outputs[:, 9:, 0]

        if self.args.phase == "exam_feat_all":
            # Image: NOTE: batch_size = 1 , 1 exam on one train

            criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

            exam_loss = criterion(exam_outputs, exam_anns) * exam_label_weights
            exam_loss_sum = torch.sum(exam_loss)
            exam_weight_sum = torch.tensor(
                1
            ).cuda()  # torch.sum(exam_label_weights).item()

            image_weight = torch.tensor(self.positive_ratio_dict[fps[0]]).cuda()
            image_loss = criterion(image_outputs, image_anns) * image_weight
            image_loss_sum = torch.sum(image_loss)
            image_weight_sum = image_weight * f_len

        elif self.args.phase == "image_exam":  # FIXME: Hot-fix for code re-using
            criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
            exam_loss_sum = criterion(exam_outputs, exam_anns)
            exam_weight_sum = torch.tensor(1).cuda()
            image_loss_sum = criterion(image_outputs, image_anns)
            image_weight_sum = torch.tensor(1).cuda()

        return exam_loss_sum, exam_weight_sum, image_loss_sum, image_weight_sum

    def init_results(self):

        # self.total_loss = tools.AverageMeter()

        # Exam
        self.exam_tot_nums = tools.AverageMeter()
        self.exam_losses = tools.AverageMeter()
        self.exam_weights = tools.AverageMeter()
        self.exam_gt_nums = tools.AverageMeter()
        self.exam_tp_nums = tools.AverageMeter()
        self.exam_pred_nums = tools.AverageMeter()
        self.exam_correct_nums = tools.AverageMeter()

        # Image
        self.image_tot_nums = tools.AverageMeter()
        self.image_losses = tools.AverageMeter()
        self.image_weights = tools.AverageMeter()
        self.image_gt_nums = tools.AverageMeter()
        self.image_tp_nums = tools.AverageMeter()
        self.image_pred_nums = tools.AverageMeter()
        self.image_correct_nums = tools.AverageMeter()

    def update_results(self, fps, anns, outputs, loss):
        # NOTE: batch_size = 1,

        # Losses - exam, image
        exam_loss_sum, exam_weight_sum, image_loss_sum, image_weight_sum = loss

        self.exam_losses.update(exam_loss_sum.item())
        self.exam_weights.update(exam_weight_sum.item())  # 1
        self.image_losses.update(image_loss_sum.item())
        self.image_weights.update(image_weight_sum.item())

        # Total
        # self.total_loss.update(
        #     (self.exam_losses.sum + self.image_losses.sum)
        #     / (self.exam_weights.sum + self.image_weights.sum)
        # )

        gts = anns.detach().cpu().numpy()
        preds = torch.sigmoid(outputs)

        # FIXME: modify rv_lv_ratio_gte_1, rv_lv_ratio_lt_1 with softmax
        # preds[:, 7:9, :] = torch.softmax(preds[:, 7:9, :], dim=1)
        preds = (preds > 0.5).detach().cpu().numpy().astype(np.float32)

        # Metrics - exam, pe-only
        # NOTE: it is negative_exam_for_pe, not is_positive
        exam_anns = gts[:, :9, :]
        exam_outputs = preds[:, :9, :]
        # note only use first label
        self.exam_gt_nums.update(np.sum(exam_anns[:, 0, :] == 0))
        self.exam_tp_nums.update(
            np.sum((exam_anns[:, 0, :] == 0) * (exam_outputs[:, 0, :] == 0))
        )
        self.exam_pred_nums.update(np.sum(exam_outputs[:, 0, :] == 0))
        self.exam_correct_nums.update(
            np.sum(exam_anns[:, 0, :] == exam_outputs[:, 0, :])
        )
        if self.args.phase == "exam_feat_all":
            self.exam_tot_nums.update(1)  # NOTE: batch_size = 1
        elif self.args.phase == "image_exam":
            # Assume each exam is different
            self.exam_tot_nums.update(exam_anns.shape[0])

        # Metrics - image, pe-positive
        image_anns = gts[:, 9:, :]
        image_outputs = preds[:, 9:, :]
        self.image_gt_nums.update(np.sum(image_anns == 1))
        self.image_tp_nums.update(np.sum(image_anns * image_outputs))
        self.image_pred_nums.update(np.sum(image_outputs == 1))
        self.image_correct_nums.update(np.sum(image_anns == image_outputs))
        if self.args.phase == "exam_feat_all":
            self.image_tot_nums.update(image_anns.shape[1])
        elif self.args.phase == "image_exam":
            self.image_tot_nums.update(image_anns.shape[0])

    def get_results(self):
        exam_pc = self.exam_tp_nums.sum / (self.exam_pred_nums.sum + 1e-6)
        exam_rc = self.exam_tp_nums.sum / (self.exam_gt_nums.sum + 1e-6)
        image_pc = self.image_tp_nums.sum / (self.image_pred_nums.sum + 1e-6)
        image_rc = self.image_tp_nums.sum / (self.image_gt_nums.sum + 1e-6)

        result = {
            "loss": (self.exam_losses.sum + self.image_losses.sum)
            / (self.exam_weights.sum + self.image_weights.sum + 1e-6),
            "exam_loss": self.exam_losses.sum / (self.exam_weights.sum + 1e-6),
            "exam_precision": exam_pc,
            "exam_recall": exam_rc,
            "exam_f1": (2 * exam_rc * exam_pc) / (exam_rc + exam_pc + 1e-6),
            "exam_acc": self.exam_correct_nums.sum / (self.exam_tot_nums.sum + 1e-6),
            "image_loss": self.image_losses.sum / (self.image_weights.sum + 1e-6),
            "image_precision": image_pc,
            "image_recall": image_rc,
            "image_f1": (2 * image_rc * image_pc) / (image_rc + image_pc + 1e-6),
            "image_acc": self.image_correct_nums.sum / (self.image_tot_nums.sum + 1e-6),
        }

        return result
