"""
================================================================
实验一：YOLO11 基线训练
================================================================
功能：
  1. 自动检测显存，选择合适的 batch size
  2. 强制设置 fliplr=0（RDD2022 必须，否则标签语义出错）
  3. 支持一次训练多个模型做对比

用法：
  python train.py                      # 训练 YOLO11n
  python train.py --model yolo11s.pt   # 训练 YOLO11s
  python train.py --all                # 依次训练 n / s / v8n 三个模型
================================================================
"""

import argparse
import gc
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


# ------------------------------------------------------------------
# 显存自适应
# ------------------------------------------------------------------
def detect_batch_size():
    """根据显卡显存自动选择 batch size 和 imgsz"""
    if not torch.cuda.is_available():
        print("[警告] 未检测到 CUDA，将使用 CPU 训练（非常慢）")
        return 4, 416, "cpu"

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[环境] 显卡: {gpu_name}")
    print(f"[环境] 显存: {vram_gb:.1f} GB")

    if vram_gb >= 24:
        batch, imgsz = 32, 640
    elif vram_gb >= 10:
        batch, imgsz = 16, 640
    elif vram_gb >= 7:
        batch, imgsz = 12, 640
    elif vram_gb >= 5:
        batch, imgsz = 8, 640
    else:  # 4GB 及以下
        batch, imgsz = 4, 640

    print(f"[环境] 自动配置: batch={batch}, imgsz={imgsz}")
    if vram_gb < 6:
        print("[提示] 显存较小，若仍 OOM 请把 imgsz 降到 512 或 batch 降到 2")
    return batch, imgsz, 0


# ------------------------------------------------------------------
# 训练主函数
# ------------------------------------------------------------------
def train_one(model_name: str, data_yaml: str, epochs: int,
              batch: int, imgsz: int, device, project: str):
    print("\n" + "=" * 60)
    print(f"  开始训练: {model_name}")
    print("=" * 60)

    model = YOLO(model_name)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,

        # ---- 优化器 ----
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,

        # ---- 早停与保存 ----
        patience=20,
        save=True,
        save_period=-1,
        val=True,

        # ---- 精度与速度 ----
        amp=True,               # 混合精度，省显存提速
        cache=False,            # 32GB 内存可设 True 加速（首次加载慢）

        # ---- 数据增强（RDD2022 专用配置）----
        fliplr=0.0,             # ⚠️ 必须为 0！翻转会改变横/纵裂缝语义
        flipud=0.0,             # 上下翻转同理
        degrees=0.0,            # 裂缝方向敏感，不做旋转
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        mosaic=1.0,
        mixup=0.0,              # 裂缝场景 MixUp 易产生伪目标
        copy_paste=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.4,

        # ---- 输出 ----
        project=project,
        name=model_name.replace(".pt", ""),
        exist_ok=True,
        plots=True,             # 生成 PR 曲线、混淆矩阵等论文用图
        verbose=True,
    )

    save_dir = Path(project) / model_name.replace(".pt", "")
    print(f"\n[完成] {model_name} 训练结束")
    print(f"  权重: {save_dir / 'weights' / 'best.pt'}")
    print(f"  曲线: {save_dir / 'results.png'}")
    print(f"  混淆矩阵: {save_dir / 'confusion_matrix.png'}")

    return results


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="道路病害检测 - 基线训练")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="模型权重，如 yolo11n.pt / yolo11s.pt / yolov8n.pt")
    parser.add_argument("--data", default="scripts/data.yaml",
                        help="数据集配置文件路径")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=None,
                        help="留空则按显存自动选择")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--project", default="runs/road_damage")
    parser.add_argument("--all", action="store_true",
                        help="依次训练 yolo11n / yolo11s / yolov8n 三个模型")
    parser.add_argument("--cross-domain", action="store_true",
                        help="实验三：按国家划分跑跨域泛化评估（留一法，6 组配置）")
    args = parser.parse_args()

    # ---- 实验三：委托给 cross_domain.py ----
    if args.cross_domain:
        here = Path(__file__).parent
        sys.path.insert(0, str(here))
        import cross_domain
        argv = ["--data", args.data, "--model", args.model,
                "--epochs", str(args.epochs)]
        if args.batch:
            argv += ["--batch", str(args.batch)]
        if args.imgsz:
            argv += ["--imgsz", str(args.imgsz)]
        cross_domain.main(argv)
        return

    # 检查数据集配置是否存在
    if not Path(args.data).exists():
        print(f"[错误] 找不到数据集配置: {args.data}")
        print("       请先修改 scripts/data.yaml 里的 path 字段")
        sys.exit(1)

    auto_batch, auto_imgsz, auto_device = detect_batch_size()
    batch = args.batch or auto_batch
    imgsz = args.imgsz or auto_imgsz

    if args.all:
        models = ["yolo11n.pt", "yolo11s.pt", "yolov8n.pt"]
    else:
        models = [args.model]

    for m in models:
        train_one(m, args.data, args.epochs, batch, imgsz, auto_device, args.project)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("  全部训练完成")
    print("=" * 60)
    print("\n下一步：运行 validate.py 汇总对比表，或直接看各 runs 目录下的 results.png")


if __name__ == "__main__":
    main()
