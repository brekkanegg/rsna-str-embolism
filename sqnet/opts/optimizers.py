import math
import torch

from torch.optim.optimizer import Optimizer, required


class adam(torch.optim.Adam):
    def __init__(self, parameters, args):
        super(adam, self).__init__(
            params=parameters,
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )


class sgd(torch.optim.SGD):
    def __init__(self, parameters, args):
        super(sgd, self).__init__(
            params=parameters,
            lr=args.learning_rate,
            momentum=args.beta1,
            weight_decay=args.weight_decay,
        )
