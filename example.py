import os
import random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2, InterpolationMode
from tqdm import tqdm

from loss import create_yolov7_loss
from metrics import f_score
from models.model_factory import create_yolov7_model
from utils import WarmupLinearSchedule, seed_all, seed_worker

seed_all(0)


class DetectionDataset(Dataset):
    def __init__(self, df, transform):
        self.transform = transform
        self.df = []
        image_paths = sorted(df["image_path"].unique())
        for image_path in image_paths:
            rows = df[df["image_path"] == image_path]
            boxes = torch.as_tensor(
                rows[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32
            )
            class_ids = torch.as_tensor(rows["class_id"].values, dtype=torch.float32)
            self.df.append(
                {
                    "image_path": image_path,
                    "bounding_boxes": boxes,
                    "class_ids": class_ids,
                }
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rows = self.df[idx]

        img = Image.open(rows["image_path"])
        h, w = img.height, img.width
        boxes = tv_tensors.BoundingBoxes(
            rows["bounding_boxes"], format="XYXY", canvas_size=(h, w)
        )

        transformed = self.transform(
            {"image": img, "boxes": boxes, "labels": rows["class_ids"]}
        )
        transformed["boxes"][:, [1, 3]] /= transformed["image"].shape[1]
        transformed["boxes"][:, [0, 2]] /= transformed["image"].shape[2]

        return transformed


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)

    image_ids = torch.cat(
        [torch.full((len(b["boxes"]), 1), i) for i, b in enumerate(batch)], dim=0
    )
    labels = torch.cat([b["labels"] for b in batch], dim=0).unsqueeze(1)
    boxes = torch.cat([b["boxes"] for b in batch], dim=0)

    out = torch.cat((image_ids, labels, boxes), dim=1)

    return images, out


def main(
    image_size=640,
    batch_size=8,
    device=torch.device("cuda:0"),
    num_classes=1,
    arch_name="yolov7-tiny",
    num_epochs=30,
):
    df = pd.read_csv("train_solution_bounding_boxes (1).csv")
    df.loc[:, "class_id"] = 0
    df["image_path"] = df["image"].apply(lambda x: os.path.join("training_images", x))

    file_names = sorted(df.image_path.unique())
    validation_files = set(random.sample(file_names, int(len(df) * 0.2)))
    train_df = df[~df.image_path.isin(validation_files)]
    val_df = df[df.image_path.isin(validation_files)]

    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ClampBoundingBoxes(),
            v2.SanitizeBoundingBoxes(),
            v2.Resize(size=(image_size, image_size), antialias=True, interpolation=InterpolationMode.BILINEAR),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    val_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ClampBoundingBoxes(),
            v2.SanitizeBoundingBoxes(),
            v2.Resize(size=(image_size, image_size), antialias=True, interpolation=InterpolationMode.BILINEAR),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    g = torch.Generator()
    g.manual_seed(0)

    train_dataset = DetectionDataset(train_df, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_dataset = DetectionDataset(val_df, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    model = create_yolov7_model(
        architecture=arch_name, num_classes=num_classes, pretrained=True
    )
    model.to(device)

    loss_func = create_yolov7_loss(
        model,
        image_size=image_size,
        box_loss_weight=0.05,
        cls_loss_weight=0.3,
        obj_loss_weight=0.7,
        ota_loss=True,
    )
    loss_func.to(device)

    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=1e-2, weight_decay=0, momentum=0.9, nesterov=True
    )
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_epochs=5, train_epochs=num_epochs, train_loader=train_loader
    )

    best_score = 0
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch: {epoch}/{num_epochs}")

        model.train()
        loss_func.train()

        train_loss = 0
        for images, boxes in tqdm(train_loader):
            images, boxes = images.to(device), boxes.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(
                enabled=True, dtype=torch.float16, cache_enabled=True
            ):
                out = model(images)
                loss, _ = loss_func(fpn_heads_outputs=out, targets=boxes, images=images)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        print(f"Train loss: {train_loss / len(train_loader)}")

        model.eval()
        loss_func.eval()

        val_preds = []
        val_gts = []
        image_id = 0
        with torch.no_grad():
            for images, boxes in tqdm(val_loader):
                images = images.to(device)

                with torch.cuda.amp.autocast(
                    enabled=True, dtype=torch.float16, cache_enabled=True
                ):
                    out = model(images)
                out = model.postprocess(
                    out, conf_thres=0.3, nms_thres=0.3, max_detections=5000
                )

                for i, pred in enumerate(out):
                    gt = boxes[boxes[:, 0] == i]
                    for p in gt:
                        val_gts.append(
                            {
                                "image_id": image_id,
                                "x0": p[2].item() * image_size,
                                "y0": p[3].item() * image_size,
                                "x1": p[4].item() * image_size,
                                "y1": p[5].item() * image_size,
                                "class_id": p[1].item(),
                            }
                        )

                    for p in pred:
                        val_preds.append(
                            {
                                "image_id": image_id,
                                "x0": p[0].item(),
                                "y0": p[1].item(),
                                "x1": p[2].item(),
                                "y1": p[3].item(),
                                "conf": p[4].item(),
                                "class_id": p[5].item(),
                            }
                        )
                    image_id += 1

        if not val_preds:
            print("No predictions found")
            continue

        val_preds = pd.DataFrame(val_preds)
        val_gts = pd.DataFrame(val_gts)
        f, precision, recall = f_score(
            val_preds, val_gts, eval_iou_threshold=0.3, beta=1
        )
        print(f"F1: {f}, precision {precision}, recall: {recall}")

        if f > best_score:
            best_score = f
            print("New best F-score")

            torch.save(model.state_dict(), "model.pt")

    print(f"Best F-score: {best_score}")


if __name__ == "__main__":
    main()
