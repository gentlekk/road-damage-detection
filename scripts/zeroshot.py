"""
================================================================
实验二：YOLO-World 开放词汇零样本检测
================================================================
创新点 1 的核心实验：不训练，直接设词表跑推理

对比三类方案：
  A. 零样本        —— 不训练，直接推理
  B. 少样本微调     —— 用 10% / 30% / 50% / 100% 训练集
  C. 全监督         —— YOLO11n 全量训练（用 train.py 跑）

用法：
  python zeroshot.py                          # 只跑零样本
  python zeroshot.py --fraction 0.1           # 用 10% 数据微调
  python zeroshot.py --fraction 0.3 --epochs 50
================================================================
"""

import argparse
import random
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


# ------------------------------------------------------------------
# 词表：把 RDD2022 的类别名映射成开放词汇提示词
# ------------------------------------------------------------------
# ⚠️ 提示词设计是零样本效果的关键，可以多试几组写进论文做消融
PROMPT_SETS = {
    "简短语": ["longitudinal crack", "transverse crack", "alligator crack", "pothole"],
    "描述语": [
        "a long crack running along the road direction",
        "a crack crossing the road perpendicularly",
        "a spider web pattern of cracks on asphalt",
        "a hole in the asphalt road surface",
    ],
    "带上下文": [
        "longitudinal crack on road pavement",
        "transverse crack on road pavement",
        "alligator cracking on asphalt road",
        "pothole on road surface",
    ],
}


def build_subset(data_yaml: str, fraction: float, out_dir: str, seed: int = 42):
    """
    从完整训练集中随机抽取 fraction 比例，构建子集数据集。
    用于画「样本量 - mAP」曲线。
    """
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["path"])
    train_images = sorted((root / cfg["train"]).glob("*.*"))
    train_labels_dir = root / cfg["train"].replace("images", "labels")

    random.seed(seed)                       # 固定种子，保证实验可复现
    n = max(1, int(len(train_images) * fraction))
    picked = random.sample(train_images, n)

    out_root = Path(out_dir)
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)

    for img in picked:
        shutil.copy2(img, out_root / "images" / "train" / img.name)
        lbl = train_labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, out_root / "labels" / "train" / lbl.name)

    # 验证集用完整验证集，保证评估口径一致
    new_cfg = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": str((root / cfg["val"]).resolve()),
        "nc": cfg["nc"],
        "names": cfg["names"],
    }
    out_yaml = out_root / "data.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(new_cfg, f, allow_unicode=True)

    print(f"[子集] 抽取 {n} / {len(train_images)} 张 ({fraction:.0%})")
    print(f"[子集] 配置已写入 {out_yaml}")
    return str(out_yaml)


# ------------------------------------------------------------------
def run_zeroshot(data_yaml: str, prompt_key: str, weights: str,
                 out_dir: str, conf: float = 0.25):
    """方案 A：零样本，完全不训练"""
    print("\n" + "=" * 60)
    print("  方案 A：开放词汇零样本检测（不训练）")
    print("=" * 60)

    classes = PROMPT_SETS[prompt_key]
    print(f"[词表] 使用「{prompt_key}」提示词组:")
    for i, c in enumerate(classes):
        print(f"  {i}: {c}")

    model = YOLO(weights)
    model.set_classes(classes)

    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    val_path = str(Path(cfg["path"]) / cfg["val"])

    print(f"\n[推理] 验证集: {val_path}")
    results = model.predict(
        source=val_path,
        conf=conf,
        save=True,
        save_txt=True,
        project=out_dir,
        name=f"zeroshot_{prompt_key}",
        exist_ok=True,
        verbose=False,
    )

    print(f"\n[完成] 零样本推理结束，共 {len(results)} 张")
    print(f"  可视化结果: {out_dir}/zeroshot_{prompt_key}/")
    print("\n[注意] YOLO-World 的 val() 需要词汇表对齐，")
    print("       建议用 04_对比脚本.py 统一统计 mAP")
    return results


# ------------------------------------------------------------------
def run_finetune(data_yaml: str, fraction: float, epochs: int,
                 batch: int, imgsz: int, weights: str, project: str):
    """方案 B：少样本微调"""
    print("\n" + "=" * 60)
    print(f"  方案 B：微调（{fraction:.0%} 训练数据）")
    print("=" * 60)

    sub_yaml = build_subset(data_yaml, fraction, out_dir=f"subsets/frac_{fraction}")

    model = YOLO(weights)
    model.train(
        data=sub_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        optimizer="SGD",
        lr0=0.01,
        patience=20,
        amp=True,
        fliplr=0.0,                 # ⚠️ 同上，必须为 0
        flipud=0.0,
        degrees=0.0,
        mosaic=1.0,
        mixup=0.0,
        project=project,
        name=f"world_finetune_{fraction}",
        exist_ok=True,
        plots=True,
    )
    save_dir = Path(project) / f"world_finetune_{fraction}"
    print(f"\n[完成] 权重: {save_dir / 'weights' / 'best.pt'}")
    return str(save_dir)


# ------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="YOLO-World 开放词汇实验")
    p.add_argument("--data", default="scripts/data.yaml")
    p.add_argument("--weights", default="yolov8s-worldv2.pt",
                   help="YOLO-World 权重，小显存可用 yolov8s-worldv2.pt")
    p.add_argument("--prompt", default="简短语", choices=list(PROMPT_SETS.keys()),
                   help="提示词组，可做消融实验")
    p.add_argument("--fraction", type=float, default=None,
                   help="微调用的数据比例，如 0.1；留空则只跑零样本")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", default="runs/road_damage")
    p.add_argument("--project", default="runs/road_damage")
    args = p.parse_args()

    # 方案 A：零样本
    run_zeroshot(args.data, args.prompt, args.weights, args.out)

    # 方案 B：少样本微调（可选）
    if args.fraction:
        run_finetune(args.data, args.fraction, args.epochs,
                     args.batch, args.imgsz, args.weights, args.project)
        print("\n[提示] 要画「样本量-mAP」曲线，请对 0.1 / 0.3 / 0.5 / 1.0 各跑一次")
        print("       把结果填入 00_开题方案.md 的实验二表格")


if __name__ == "__main__":
    main()
