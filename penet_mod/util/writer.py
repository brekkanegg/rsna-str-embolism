import sys
import time
import numpy as np
from tensorboardX import SummaryWriter


class Writer(SummaryWriter):
    def write_scalar(self, scalar_dict, iteration):
        for k, v in scalar_dict.items():
            self.add_scalar(k, v, iteration)

    def write_scalars(self, scalars_dict, iteration):
        for k1, v1 in scalars_dict.items():
            self.add_scalars(k1, {k2: v2 for k2, v2 in v1.items()}, iteration)

