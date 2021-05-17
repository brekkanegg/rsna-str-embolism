import json
import numpy as np
import torch

from util import tools, visualizer, logger, writer
from opts import losses
from configs import *


class Commoner(object):
    def __init__(self, args):
        self.args = args
        with open(DATA_DIR + "/train_positive_ratio.json", "rb") as f:
            self.positive_ratio_dict = json.load(f)

    def calc_comp_metric(self, fps, anns, outputs):
        """
        output should be logits
        """

        image_metric_weights = 0
        image_metric_losses = 0
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

        if (self.args.phase == "exam_feat") or (self.args.phase == "exam_end2end"):
            for bi in range(len(fps)):
                pr = self.positive_ratio_dict[fps[bi]]
                if pr == 0:
                    continue

                bi_valid_idxs = (anns[bi] != -1).detach().cpu().numpy()
                bi_anns = anns[bi][bi_valid_idxs]
                bi_outputs = outputs[bi][bi_valid_idxs]

                with torch.no_grad():
                    image_metric_weights += pr * len(bi_anns)
                    image_metric_losses += np.sum(
                        pr * (criterion(bi_outputs, bi_anns).detach().cpu().numpy())
                    )

            return image_metric_weights, image_metric_losses

        elif self.args.phase == "exam_feat_all":
            # FIXME: not implemented - no need
            return image_metric_weights, image_metric_losses

        else:
            for bi in range(len(fps)):
                sid = fps[bi].split("_")[0]
                pr = self.positive_ratio_dict[sid]

                if pr == 0:
                    continue

                with torch.no_grad():
                    image_metric_weights += pr
                    image_metric_losses += (
                        pr * criterion(outputs[bi, 0], anns[bi, 0]).item()
                    )

            return image_metric_weights, image_metric_losses

    def calc_loss(self, fps, anns, outputs):

        if self.args.phase == "chunk_end2end":
            outputs = torch.max(outputs, dim=1)[0]
            pw = None
            if self.args.pos_weight is not None:
                pw = torch.tensor(self.args.pos_weight).cuda()

            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
            loss = criterion(outputs, anns)

        elif self.args.phase == "image":
            pw = None
            if self.args.pos_weight is not None:
                pw = torch.tensor(self.args.pos_weight).cuda()

            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
            loss = criterion(outputs, anns)

        elif (self.args.phase == "exam_feat") or ((self.args.phase == "exam_end2end")):

            batch_n, seq_n = anns.shape[:2]
            anns_reshaped = torch.reshape(anns, (batch_n * seq_n, -1))
            valid_idxs = anns_reshaped != -1
            outputs_reshaped = torch.reshape(outputs, (batch_n * seq_n, -1))

            anns = anns_reshaped[valid_idxs]
            outputs = outputs_reshaped[valid_idxs]

            if self.args.loss_type == "bce":
                pw = None
                if self.args.pos_weight is not None:
                    pw = torch.tensor(self.args.pos_weight).cuda()

                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
                loss = criterion(outputs, anns)

        return loss

    def init_results(self):
        self.tot_nums = tools.AverageMeter()
        self.loss = tools.AverageMeter()

        self.gt_nums = tools.AverageMeter()
        self.tp_nums = tools.AverageMeter()
        self.pred_nums = tools.AverageMeter()
        self.correct_nums = tools.AverageMeter()

        self.comp_metric_weight = tools.AverageMeter()
        self.comp_metric_loss = tools.AverageMeter()

    def update_results(self, fps, anns, outputs, loss):

        # NOTE; outputs should be logits
        comp_metric_weights, comp_metric_losses = self.calc_comp_metric(
            fps, anns, outputs
        )

        self.comp_metric_weight.update(comp_metric_weights)
        self.comp_metric_loss.update(comp_metric_losses)

        gts = anns.detach().cpu().numpy()

        if self.args.phase == "chunk_end2end":
            outputs = torch.max(torch.sigmoid(outputs), dim=1)[0]

        preds = (torch.sigmoid(outputs).detach().cpu().numpy() > 0.5).astype(np.float32)

        if (self.args.phase == "exam_feat") or (self.args.phase == "exam_end2end"):

            batch_n, seq_n = gts.shape[:2]
            gts = np.reshape(gts, (batch_n * seq_n, -1))
            valid_idxs = gts != -1
            preds = np.reshape(preds, (batch_n * seq_n, -1))
            gts = gts[valid_idxs]
            preds = preds[valid_idxs]

        self.tot_nums.update(len(gts))
        self.loss.update(loss.item())

        self.correct_nums.update(np.sum(gts == preds))
        self.gt_nums.update(np.sum(gts == 1))
        self.pred_nums.update(np.sum(preds))
        self.tp_nums.update(np.sum(gts * preds))

    def get_results(self):
        pc = self.tp_nums.sum / (self.pred_nums.sum + 1e-6)
        rc = self.tp_nums.sum / (self.gt_nums.sum + 1e-6)

        result = {
            "loss": self.loss.avg,
            "precision": pc,
            "recall": rc,
            "f1": (2 * rc * pc) / (rc + pc + 1e-6),
            "acc": self.correct_nums.sum / self.tot_nums.sum,
        }

        result["comp_metric"] = self.comp_metric_loss.sum / (
            self.comp_metric_weight.sum + 1e-15
        )

        return result

