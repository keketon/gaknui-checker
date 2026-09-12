"""学ぬい分類用転移学習スクリプト

MobileNetV3 Large をベースに転移学習することで、学ぬいを分類するモデルをつくる。

方針:
- `data/processed/` 以下のディレクトリ構造（`{category_name}/*.jpg`）をそのまま `ImageFolder` として利用する
- train/val は層化(stratify)分割する
- train用とval用でtransform（データ拡張の有無）を分ける
- まずはバックボーンを凍結し、分類層のみ学習する
- 保存時は重みと一緒に `class_to_idx` も保存する（推論側でラベル対応が必要になるため）
"""

import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

CONFIG = {
    "data_dir": Path("data/processed"),
    "model_dir": Path("models"),
    "seed": 42,
    "val_ratio": 0.2,
    "batch_size": 16,
    "num_epochs": 15,
    "lr": 1e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def stratified_split(
    dataset, val_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    """クラスごとにシャッフルしてval_ratio分をvalに回し、train/valのインデックスを返す"""
    rng = random.Random(seed)

    class_indices = defaultdict(list)
    for idx, label in enumerate(dataset.targets):
        class_indices[label].append(idx)

    train_indices = []
    val_indices = []

    for label_indices in class_indices.values():
        rng.shuffle(label_indices)
        val_count = int(len(label_indices) * val_ratio)
        val_indices += label_indices[:val_count]
        train_indices += label_indices[val_count:]

    return (train_indices, val_indices)


def build_model(num_classes: int) -> nn.Module:
    model = mobilenet_v3_large(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier[0].parameters():
        param.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    """1エポック分学習し、平均lossを返す"""
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """評価して (平均loss, accuracy) を返す"""
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    return (total_loss / len(loader.dataset), correct / len(loader.dataset))


if __name__ == "__main__":
    random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    print(CONFIG)

    # 事前学習のパラメータを使用する
    # 転移学習では、画像のRGB値（[0,1)）の分布（平均と偏差）を事前学習のデータセットと同じものとする必要があるため
    weights = MobileNet_V3_Large_Weights.DEFAULT
    preset = weights.transforms()
    mean, std = preset.mean, preset.std

    train_transform = transforms.Compose(
        [
            # ぬいの位置を画像内である程度ランダムに動かすことで、ユーザーがぬいをカメラの端で捉えたときにも適応できるようにする、という理解
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            # Horizontal Flipはスマホのインカメラ使用時などを想定して必要そう
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # ↑みたいな幾何処理がいらない場合はpresetをそのまま使うと楽
    # preprocessで画像サイズをそろえているのでリサイズはここでしなくていいのも貢献してる。
    val_transform = weights.transforms()

    dataset_train_view = ImageFolder(CONFIG["data_dir"], transform=train_transform)
    dataset_val_view = ImageFolder(CONFIG["data_dir"], transform=val_transform)

    print("classes:", dataset_train_view.classes)
    print("class_to_idx:", dataset_train_view.class_to_idx)

    train_indices, val_indices = stratified_split(
        dataset_train_view, CONFIG["val_ratio"], CONFIG["seed"]
    )

    train_dataset = Subset(dataset_train_view, train_indices)
    val_dataset = Subset(dataset_val_view, val_indices)

    print(f"train: {len(train_dataset)} / val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False
    )

    model = build_model(num_classes=len(dataset_train_view.classes)).to(
        CONFIG["device"]
    )

    criterion = nn.CrossEntropyLoss()

    # 凍結していないパラメータのみoptimizerに渡す
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr"],
    )

    best_val_acc = 0.0
    CONFIG["model_dir"].mkdir(parents=True, exist_ok=True)

    for epoch in range(CONFIG["num_epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, CONFIG["device"]
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, CONFIG["device"])

        print(
            f"epoch {epoch + 1}/{CONFIG['num_epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # ベスト更新時にチェックポイントを保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": dataset_train_view.class_to_idx,
                },
                CONFIG["model_dir"] / "mobilenet_v3_large_nui.pth",
            )
