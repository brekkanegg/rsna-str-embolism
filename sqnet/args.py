import os
import argparse
import collections


def str2bool(v):
    if v is None:
        return None
    return v.lower() in ("true")


# control here
parser = argparse.ArgumentParser()


# Task - Speicifc
parser.add_argument("--phase", type=str, required=True)
parser.add_argument("--extract_train", type=str2bool, default=True)
parser.add_argument("--save_whatever", type=str2bool, default=False)


# GPU, CPU
parser.add_argument("--gpu", "--g", type=str, default="0")
parser.add_argument("--num_workers", "--w", type=int, default=4)
parser.add_argument("--apex", type=str2bool, default=False)
parser.add_argument("--distributed", "--dist", type=str2bool, default=False)


# Default
parser.add_argument("--run", type=str, default="train")
parser.add_argument("--verbosity", type=int, default=1)


# Dataset
parser.add_argument("--fold", "--f", type=int, default=0)  # 1, 2, 3, ..
# 7fold_final
parser.add_argument("--fold_json", "--fj", type=str, default="temp_fold.json")
parser.add_argument("--num_classes", "--nc", type=int, default=1)
parser.add_argument("--augment", "--aug", type=str, default="augment_3d")
parser.add_argument("--data_format", "--df", type=str, default="h5")
parser.add_argument("--image_size", type=int, default=448)
parser.add_argument("--posneg_ratio", "--pnr", type=float, default=None)
parser.add_argument("--windowing", type=str, default="3")
parser.add_argument("--use_only_positive", "--uop", type=str2bool, default=False)


# TRAIN
parser.add_argument(
    "--load_pretrained_ckpt", "--lp", type=str, default=None,
)

parser.add_argument("--resume_train", "--rt", type=str2bool, default=False)
parser.add_argument("--max_epoch", "--mep", type=int, default=1000)
parser.add_argument("--batch_size", "--bs", type=int, default=16)
parser.add_argument("--display_interval", "--di", type=int, default=4)
parser.add_argument("--samples_per_epoch", "--spe", type=int, default=50000)
parser.add_argument("--endurance", "--ed", type=int, default=50)


# Optimization

parser.add_argument("--optimizer", "--opt", type=str, default="adam")
parser.add_argument("--lr_scheduler", "--sched", type=str, default=None)
parser.add_argument("--lr_warmup_steps", "--warmup", type=int, default=None)
parser.add_argument("--lr_decay_step", type=int, default=None)
parser.add_argument("--learning_rate", "--lr", type=float, default=1e-4)
# parser.add_argument("--warmup_epoch", "--wep", type=int, default=4)

parser.add_argument("--weight_decay", "--wd", type=float, default=5e-6)
parser.add_argument("--adam_beta_1", "--b1", type=float, default=0.9)
parser.add_argument("--adam_beta_2", "--b2", type=float, default=0.999)
parser.add_argument("--sgd_momentum", "--sgdm", type=float, default=0.9)
parser.add_argument("--sgd_dampening", "--sgdd", type=float, default=0)

# parser.add_argument("--grad_clip", "--gc", type=int, default=None)
# parser.add_argument("--grad_accum_steps", "--gas", type=int, default=1)


# Loss
parser.add_argument("--loss_type", "--lt", type=str, default="bce")
parser.add_argument("--pos_weight", "--pw", type=float, default=None)


# Validation
parser.add_argument("--val_epoch", "--vep", type=int, default=-1)
parser.add_argument("--val_interval", "--vi", type=int, default=1)
parser.add_argument("--val_th", "--vth", type=float, default=0.5)
parser.add_argument("--best", type=str, default="loss")


# Options
parser.add_argument("--is_debugging", "--debug", type=str2bool, default=False)
parser.add_argument("--print_io", "--pio", type=str2bool, default=False)
parser.add_argument("--name", type=str, default=None)


# Model
parser.add_argument("--model", default="ResNext", type=str)
parser.add_argument("--dense", "--dn", type=str2bool, default=False)
# parser.add_argument("--slice_num", "--sn", type=int, default=7)
parser.add_argument("--units", "--lus", default=256, type=int)
# parser.add_argument("--encoder", "--enc", default="resnext101", type=str)


# Phase - End2End
# parser.add_argument("--embed_size", "--es", default=2048, type=int)

# Temp

parser.add_argument("--test_save_dir", "--tsd", default=None, type=str)

# Phase - image
# parser.add_argument("--follow_penet", default=False, type=str2bool)

# args = parser.parse_args()
args, _ = parser.parse_known_args()


# Phase


# if args.phase == "lstm_end2end_chunk":


if args.phase == "image":
    args.model = "ResNext101"
    args.augment = "augment_2d"
    args.best = "loss"
    args.posneg_ratio = 1.0  # FIXME:

elif args.phase == "image_exam":
    args.model = "EIResNext50"
    args.augment = "augment_2d"
    args.best = "loss"
    args.fold_json = "7fold_final.json"
    args.posneg_ratio = 1.0  # FIXME:


# elif args.phase == "lstm_feat_exam":
elif args.phase == "exam_feat":
    args.augment = None
    args.best = "comp_metric"

elif args.phase == "exam_feat_all":
    args.batch_size = 1  # NOTE:
    args.augment = None
    args.best = "loss"

# elif args.phase == "chunk_end2end":
#     args.freeze_encoder = False
#     args.encoder = "resnext101"
#     args.augment = "augment_3d"


if args.run == "extract":
    if args.phase != "public":
        extract_train = input("Extract from train+val: True/False ")
        if extract_train == "True":
            args.extract_train = True
        elif extract_train == "False":
            args.extract_train = False

    # args.fold = int(input("Check once again the fold: "))
    args.extract_epoch = int(input("Extract epoch: "))
    args.samples_per_epoch = None


if args.run == "val":
    args.load_pretrained_ckpt = input("Validate which ckpt: ")
    assert args.fold == int(
        args.load_pretrained_ckpt.split("/fold-")[1].split("_phase-")[0]
    )
    assert (
        args.phase == args.load_pretrained_ckpt.split("_phase-")[1].split("_model-")[0]
    )

elif args.run == "test":
    args.fold_json = "temp_fold.json"  # FIXME:
    if args.load_pretrained_ckpt is None:
        args.load_pretrained_ckpt = input("Test which ckpt: ")
    assert args.fold == int(
        args.load_pretrained_ckpt.split("/fold-")[1].split("_phase-")[0]
    )
    assert (
        args.phase == args.load_pretrained_ckpt.split("_phase-")[1].split("_model-")[0]
    )


print("\n")
print("### PHASE:   ", args.phase)
print("### MODEL:   ", args.model)
print("### FOLD:    ", args.fold)

print("\n")


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if len(args.gpu.split(",")) > 1:
    args.multi_gpu = True
else:
    args.multi_gpu = False

