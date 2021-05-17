import matplotlib

matplotlib.use("Agg")  # tensorboardX
import os, sys
import json
import torch
import torch.backends.cudnn as cudnn

from args import args


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if len(args.gpu.split(",")) > 1:
        args.multi_gpu = True

    torch.autograd.set_detect_anomaly(True)
    cudnn.deterministic = True
    cudnn.benchmark = False

    if args.mode == "train":
        from scripts.train import Trainer

        Trainer(args).do_train()

    elif args.mode == "val":  # 'test' 와 동일,  /train/trainval/val(test)
        from scripts.validate import Validator

        Validator(args, is_trainval=False).do_validate()
