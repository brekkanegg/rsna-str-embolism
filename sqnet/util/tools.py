import time
import collections
import os


def set_save_dir(args):
    save_dict = collections.OrderedDict()
    save_dict["fold"] = args.fold  # 1,2,3
    save_dict["phase"] = args.phase  # 1,2,3
    save_dict["model"] = args.model  # 1,2,3

    if args.name:
        save_dict["name"] = args.name

    save_dir = ["{}-{}".format(key, save_dict[key]) for key in save_dict.keys()]
    save_dir = "/nfs3/minki/kaggle/rsna-str-embolism/ckpt/" + "_".join(save_dir)
    save_dir = save_dir.replace("True", "true").replace("False", "false")

    os.makedirs(save_dir, exist_ok=True)

    return save_dir


def apply_windowing(img):
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


def draw_rect(img, coord):
    # x0, y0, x1, y1 = [c - 1 for c in coord]
    x0, y0, x1, y1 = [c for c in coord]
    img[y0 : y0 + 2, x0:x1] = 1.0 * 255
    img[y1 - 2 : y1, x0:x1] = 1.0 * 255
    img[y0:y1, x0 : x0 + 2] = 1.0 * 255
    img[y0:y1, x1 - 2 : x1] = 1.0 * 255

    return img


def convert_time(time):
    # time = float(input("Input time in seconds: "))
    day = time // (24 * 3600)
    time = time % (24 * 3600)
    hour = time // 3600
    time %= 3600
    minutes = time // 60
    time %= 60
    seconds = time

    # ("%dD:%dH:%dM:%dS" % (day, hour, minutes, seconds))
    return "%1dD %2dH %2dM" % (day, hour, minutes)


class AverageMeter(object):
    """Computes and stores the average and current value.

    Adapted from:
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.avg = 0
        self.val = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.__init__()

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

