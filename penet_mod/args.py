import os
import argparse
import collections


def str2bool(v):
    if v is None:
        return None
    return v.lower() in ("true")


# control here
parser = argparse.ArgumentParser()

# FIXME:

# NEW
parser.add_argument("--windowing", type=str, default="3")
# parser.add_argument("--normalization", type=str, default="norm_01")
# parser.add_argument("--standardization", type=str, default="standard_z")


# GPU, CPU
parser.add_argument("--gpu", "--g", type=str, default="0")
parser.add_argument("--num_workers", "--w", type=int, default=16)


# Default
parser.add_argument("--mode", type=str, default="train")
parser.add_argument("--verbosity", type=int, default=1)


# Dataset
parser.add_argument("--fold", "--f", type=int, default=0)  # 1, 2, 3, ..
parser.add_argument("--num_classes", "--nc", type=int, default=1)
parser.add_argument("--augment", "--aug", type=str, default="augment_3d")
parser.add_argument("--data_format", "--df", type=str, default="h5")


parser.add_argument(
    "--load_pretrained_ckpt",
    "--lp",
    type=str,
    default="/nfs3/minki/kaggle/rsna-str-embolism/penet_best.pth.tar",
)

parser.add_argument("--resume_train", "--rt", type=str2bool, default=False)
parser.add_argument("--max_epoch", "--mep", type=int, default=250)
parser.add_argument("--batch_size", "--bs", type=int, default=16)
parser.add_argument("--display_interval", "--di", type=int, default=4)
parser.add_argument("--samples_per_epoch", "--spe", type=int, default=None)
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

# Validation
parser.add_argument("--val_epoch", "--vep", type=int, default=-1)
parser.add_argument("--val_interval", "--vi", type=int, default=1)
parser.add_argument("--val_th", "--vth", type=float, default=0.5)
parser.add_argument("--best", type=str, default="acc")


# Options
parser.add_argument("--is_debugging", "--debug", type=str2bool, default=False)
parser.add_argument("--name", type=str, default=None, help="save_dir naming")


parser.add_argument("--slice_num", type=int, default=24)
parser.add_argument("--image_size", type=int, default=448)


# Model
parser.add_argument("--model", default="PENetClassifier", type=str)
parser.add_argument("--model_depth", default=50, type=int)
parser.add_argument(
    "--init_method",
    type=str,
    default="kaiming",
    choices=("kaiming", "normal", "xavier"),
    help="Initialization method to use for conv kernels and linear weights.",
)

parser.add_argument("--follow_penet", default=False, type=str2bool)

# args = parser.parse_args()
args, _ = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if len(args.gpu.split(",")) > 1:
    args.multi_gpu = True
else:
    args.multi_gpu = False


if args.follow_penet:
    args.loss_type = "focal"
    args.optimizer = "sgd"
    args.learning_rate = 1e-2
    args.weight_decay = 1e-3
    args.use_pretrained = True
    args.lr_decay_step = 600000
    args.lr_scheduler = "cosine_warmup"
    args.lr_warmup_steps = 10000
