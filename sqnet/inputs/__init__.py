import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler as DS


def collater(data):

    fps = [s["fp"] for s in data]
    imgs = torch.FloatTensor([s["img"] for s in data])

    collated_data = {"fp": fps, "img": imgs}

    anns = torch.tensor([s["anns"] for s in data])
    collated_data["anns"] = anns

    return collated_data


def collater_feat_pad(data):
    max_len = max([l["img"].shape[0] for l in data])
    img_size = data[0]["img"].shape[1:]
    # has_anns = "anns" in data[0].keys()
    # if has_anns:
    anns_dim = data[0]["anns"].shape[1]

    for b in data:
        dummy_len = max_len - len(b["img"])
        b["img"] = np.vstack((b["img"], (np.zeros([dummy_len] + list(img_size)))))
        # if has_anns:
        b["anns"] = np.vstack((b["anns"], (np.ones((dummy_len, anns_dim)) * -1)))

    fps = [s["fp"] for s in data]
    imgs = torch.FloatTensor([s["img"] for s in data])
    collated_data = {"fp": fps, "img": imgs}
    # if has_anns:
    anns = torch.tensor([s["anns"] for s in data])
    collated_data["anns"] = anns

    return collated_data


def collater_pad(data):
    max_len = max([l["img"].shape[1] for l in data])
    img_size = data[0]["img"].shape[2:]
    anns_dim = data[0]["anns"].shape[1]

    for b in data:
        dummy_len = max_len - b["img"].shape[1]
        b["img"] = np.concatenate(
            (b["img"], np.zeros((3, dummy_len, img_size[0], img_size[1]))), axis=1
        )
        b["anns"] = np.concatenate(
            (b["anns"], (np.ones((dummy_len, anns_dim)) * -1)), axis=0
        )

    fps = [s["fp"] for s in data]
    imgs = torch.FloatTensor([s["img"] for s in data])
    collated_data = {"fp": fps, "img": imgs}

    anns = torch.tensor([s["anns"] for s in data])
    collated_data["anns"] = anns

    return collated_data


def get_dataloader(args, transform=None):
    if args.phase == "image":
        from .ct_images import CTImages as data_class

        collate_fn = collater

    elif args.phase == "image_exam":
        from .ct_image_exams import CTImage_Exams as data_class

        collate_fn = collater

    elif args.phase == "chunk_end2end":
        from .ct_chunks import CTChunks as data_class

        collate_fn = collater

    elif (args.phase == "exam_feat") or (args.phase == "exam_feat_all"):
        from .ct_feats import CTFeats as data_class

        collate_fn = collater_feat_pad

    elif args.phase == "exam_end2end":
        from .ct_exams import CTExams as data_class

        collate_fn = collater_pad

    else:
        print("wrong phase!")
        raise

    if args.run == "train":
        data_set = data_class(args, transform=transform, mode="train")
        train_loader = DataLoader(
            dataset=data_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
        )

        data_set = data_class(args, transform=None, mode="trainval")
        if args.phase == "exam_feat_all":
            bs = args.batch_size
        else:
            bs = args.batch_size * 4

        val_loader = DataLoader(
            dataset=data_set,
            batch_size=bs,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
        )

        return (train_loader, val_loader)

    elif args.run == "val":
        data_set = data_class(args, transform=None, mode="val")
        val_loader = DataLoader(
            dataset=data_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
        )

        return val_loader

    elif args.run == "extract":
        data_set = data_class(args, transform=None, mode="train")
        train_loader = DataLoader(
            dataset=data_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
        )
        data_set = data_class(args, transform=None, mode="trainval")
        val_loader = DataLoader(
            dataset=data_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
        )

        return (train_loader, val_loader)

    elif args.run == "test":
        data_set = data_class(args, transform=None, mode="test")
        test_loader = DataLoader(
            dataset=data_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
        )

        return test_loader
