import torch
import torch.nn as nn
import torch.nn.functional as F
import math


import torchvision.models as tvmodels


class ResNext(nn.Module):
    def __init__(self, args, use_pretrained=True):

        super(ResNext, self).__init__()

        if use_pretrained:
            self.embedder = torch.hub.load(
                "facebookresearch/WSL-Images", "resnext101_32x8d_wsl"
            )
        else:
            self.embedder = tvmodels.resnext101_32x8d(pretrained=False)

        # elif args.encoder == "resnet50":
        #     self.embedder = tvmodels.resnet50_32x4d(pretrained=True)

        self.embedder.fc = nn.Linear(2048, args.num_classes)
        for m in self.embedder.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        return self.embedder(x)


# NOTE: For Exam_Image Case -- label is 9 + 1
class EIResNext50(nn.Module):
    def __init__(self, args, use_pretrained=True):

        super(EIResNext50, self).__init__()

        if use_pretrained:
            self.embedder = tvmodels.resnext50_32x4d(pretrained=True)
        else:
            self.embedder = tvmodels.resnext50_32x4d(pretrained=False)

        self.embedder.fc = nn.Linear(2048, 9 + args.num_classes)
        for m in self.embedder.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.embedder(x)
        x = x.unsqueeze(2)  # shape: batch x 10 x 1

        return x


class EfficientNet(nn.Module):
    def __init__(self, args, use_pretrained=True):
        super(EfficientNet, self).__init__()
        from efficientnet_pytorch import EfficientNet as EFNet

        self.model = EFNet.from_pretrained(
            "efficientnet-b4", num_classes=args.num_classes
        )

    def forward(self, x):
        out = self.model(x)

        return out
