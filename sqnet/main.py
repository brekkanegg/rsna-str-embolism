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

    if args.run == "train":
        if args.phase == "exam_feat_all":
            from scripts.train_all import TrainerALL as Trainer

        elif args.phase == "image_exam":
            from scripts.train_all import TrainerALL as Trainer

        else:
            from scripts.train import Trainer

        Trainer(args).do_train()

    elif args.run == "val":  # 'test' 와 동일,  /train/trainval/val(test)
        from scripts.validate import Validator

        Validator(args, is_trainval=False).do_validate()

    elif args.run == "extract":  # 'test' 와 동일,  /train/trainval/val(test)
        if args.phase == "public":
            from scripts.public_extract import PublicExtractor

            PublicExtractor(args).do_extract()
        else:
            from scripts.extract import Extractor

            ex = Extractor(args)
            ex.do_extract(is_train=True)
            ex.do_extract(is_train=False)

    elif args.run == "test":  # 'test' 와 동일,  /train/trainval/val(test)
        from scripts.test_all import TestorALL

        TestorALL(args).do_test()
