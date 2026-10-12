"""
================================================================
实验三：跨域泛化能力评估（留一法设计）
================================================================
功能：
  自动按国家划分数据集，构建 6 组训练/测试配置，依次训练并汇总结果。

设计要点（重要）：
  跨域测试中，目标域必须「训练时不可见」，否则就是域内测试，不可比。
  因此第 ⑤ 行采用「留一法」：训练时排除挪威，测试时才用挪威。

用法：
  python cross_domain.py                      # 跑全部 6 组
  python cross_domain.py --only 03 05         # 只跑第 3、5 组
  python cross_domain.py --epochs 50
  python cross_domain.py --dry-run            # 只划分数据不训练（先验证划分配置）

前置条件：
  数据集需保留 RDD2022 原始文件名（文件名含国家前缀），例如：
      China_Drone_000001.jpg
      China_MotorBike_000001.jpg
      Japan_000001.jpg
      Norway_000001.jpg
      United_States_000001.jpg
      Czech_000001.jpg
      India_000001.jpg
================================================================
"""

import argparse
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml


# ------------------------------------------------------------------
# 项目根目录 / 默认输出目录
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ⚠️ 输出目录必须用【绝对路径】
#    ultralytics 规则：project 是相对路径时会被拼到全局设置 runs_dir 后面，
#    而全局 runs_dir 可能被别的软件（如 SD WebUI）改过。
DEFAULT_WORK = str(PROJECT_ROOT / "cross_domain")


# ------------------------------------------------------------------
# 国家识别：靠文件名前缀
# ------------------------------------------------------------------
COUNTRY_PREFIXES = {
    "China":  ["China_Drone", "China_MotorBike"],
    "Japan":  ["Japan"],
    "India":  ["India"],
    "Czech":  ["Czech"],
    "Norway": ["Norway"],
    "USA":    ["United_States"],
}

ALL_COUNTRIES = list(COUNTRY_PREFIXES.keys())


# ------------------------------------------------------------------
# 六组实验配置
# ------------------------------------------------------------------
#  字段：(编号, 训练国家, 测试国家, 说明)
EXPERIMENTS = [
    ("01", ["China"], "China",  "基线：域内上界（单源训练 / 域内测试）"),
    ("02", ["China"], "Japan",  "跨国家（单源训练 / 跨域测试）"),
    ("03", ["China"], "Norway", "跨国家+跨设备（单源训练 / 跨域测试）"),
    ("04", ["China"], "USA",    "跨国家（单源训练 / 跨域测试）"),
    ("05", ["China", "Japan", "India", "Czech", "USA"], "Norway",
     "多源训练 / 跨域测试（挪威留出）← 与 03 对比 = 多源训练的收益"),
    ("06", ALL_COUNTRIES, "China",
     "全域训练 / 域内测试（中国参与训练）← 与 01 对比 = 加域的成本"),
]


def classify_country(stem: str):
    """根据文件名前缀判断属于哪个国家"""
    for country, prefixes in COUNTRY_PREFIXES.items():
        for pre in prefixes:
            if stem.startswith(pre):
                return country
    return None


def scan_split(images_dir: Path):
    """扫描一个 images 目录，按国家分组"""
    groups = defaultdict(list)
    unknown = 0
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for p in images_dir.iterdir():
        if p.suffix.lower() not in exts:
            continue
        c = classify_country(p.stem)
        if c:
            groups[c].append(p)
        else:
            unknown += 1
    return groups, unknown


def link_or_copy(src: Path, dst: Path):
    """优先硬链接（NTFS 同盘免管理员权限、不占额外空间），失败则复制"""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def label_path_for(img: Path, images_root: Path, labels_root: Path) -> Path:
    """由图像路径推出对应的标注路径"""
    rel = img.relative_to(images_root)
    return labels_root / rel.with_suffix(".txt")


# ------------------------------------------------------------------
# 构建一组实验的数据子集
# ------------------------------------------------------------------
def build_config(cfg_dir: Path, train_countries, test_country,
                 all_train_imgs, all_train_lbls, all_val_imgs, all_val_lbls,
                 train_groups, val_groups, nc, names):
    """把指定国家的图像/标注硬链接到子集目录，并写 data.yaml"""

    train_dst_img = cfg_dir / "train" / "images"
    train_dst_lbl = cfg_dir / "train" / "labels"
    val_dst_img = cfg_dir / "val" / "images"
    val_dst_lbl = cfg_dir / "val" / "labels"
    for d in (train_dst_img, train_dst_lbl, val_dst_img, val_dst_lbl):
        d.mkdir(parents=True, exist_ok=True)

    n_train = n_val = 0

    for country in train_countries:
        for img in train_groups.get(country, []):
            link_or_copy(img, train_dst_img / img.name)
            lbl = label_path_for(img, all_train_imgs, all_train_lbls)
            if lbl.exists():
                link_or_copy(lbl, train_dst_lbl / lbl.name)
            n_train += 1

    # 验证集固定为「测试国」的验证集，保证评估口径一致
    for img in val_groups.get(test_country, []):
        link_or_copy(img, val_dst_img / img.name)
        lbl = label_path_for(img, all_val_imgs, all_val_lbls)
        if lbl.exists():
            link_or_copy(lbl, val_dst_lbl / lbl.name)
        n_val += 1

    cfg = {
        "path": str(cfg_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": nc,
        "names": names,
    }
    yaml_file = cfg_dir / "data.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    return str(yaml_file), n_train, n_val


# ------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="实验三：跨域泛化评估（留一法）")
    p.add_argument("--data", default="scripts/data.yaml",
                   help="原始数据集配置（用于读取 path / nc / names）")
    p.add_argument("--work", default=DEFAULT_WORK,
                   help="子集与权重的输出根目录（绝对路径）")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--only", nargs="*", default=None,
                   help="只跑指定编号，如 --only 03 05")
    p.add_argument("--dry-run", action="store_true",
                   help="只划分数据并打印统计，不训练")
    args = p.parse_args(argv)

    # ---- 读取原始配置 ----
    if not Path(args.data).exists():
        print(f"[错误] 找不到 {args.data}")
        sys.exit(1)
    with open(args.data, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    root = Path(base["path"])
    all_train_imgs = root / base["train"]
    all_val_imgs = root / base["val"]
    # 标注目录 = 图像目录的同级 labels 目录（不用字符串替换，避免父目录名含 "images" 时误伤）
    all_train_lbls = all_train_imgs.parent / "labels"
    all_val_lbls = all_val_imgs.parent / "labels"
    nc, names = base["nc"], base["names"]

    print("=" * 64)
    print("  实验三：跨域泛化能力评估（留一法设计）")
    print("=" * 64)
    print(f"[数据] 训练集目录: {all_train_imgs}")
    print(f"[数据] 验证集目录: {all_val_imgs}")

    # ---- 按国家扫描 ----
    train_groups, unk_tr = scan_split(all_train_imgs)
    val_groups, unk_va = scan_split(all_val_imgs)

    print("\n[统计] 训练集按国家分布：")
    total = sum(len(v) for v in train_groups.values())
    for c in ALL_COUNTRIES:
        n = len(train_groups.get(c, []))
        bar = "█" * int(40 * n / max(1, total))
        print(f"   {c:<8} {n:>7,}  {bar}")
    print(f"   {'合计':<8} {total:>7,}")

    print("\n[统计] 验证集按国家分布：")
    for c in ALL_COUNTRIES:
        print(f"   {c:<8} {len(val_groups.get(c, [])):>7,}")

    if unk_tr or unk_va:
        print(f"\n[警告] 有 {unk_tr} 张训练图 / {unk_va} 张验证图无法识别国家前缀，已跳过")
        print("       若数量很多，说明文件名不含国家前缀，本脚本无法按域划分。")

    missing = [c for c in ALL_COUNTRIES
               if not train_groups.get(c) or not val_groups.get(c)]
    if missing:
        print(f"\n[错误] 以下国家缺少数据，无法完成全部实验: {missing}")
        print("       可先用 --only 跑已有的组合。")
        if not args.dry_run:
            sys.exit(1)

    # ---- 逐组实验 ----
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    summary = []

    for num, train_c, test_c, desc in EXPERIMENTS:
        if args.only and num not in args.only:
            continue

        print("\n" + "-" * 64)
        print(f"  第 {num} 组：训练={'+'.join(train_c)}  测试={test_c}")
        print(f"  说明：{desc}")
        print("-" * 64)

        cfg_dir = work / f"exp{num}_{test_c.lower()}"
        data_yaml, n_tr, n_va = build_config(
            cfg_dir, train_c, test_c,
            all_train_imgs, all_train_lbls, all_val_imgs, all_val_lbls,
            train_groups, val_groups, nc, names)

        print(f"  训练样本 {n_tr:,} 张 | 验证样本 {n_va:,} 张")
        print(f"  配置: {data_yaml}")

        if args.dry_run:
            summary.append((num, "+".join(train_c), test_c, n_tr, "-", "-"))
            continue

        from ultralytics import YOLO
        model = YOLO(args.model)
        model.train(
            data=data_yaml, epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, optimizer="SGD", lr0=0.01, patience=20,
            amp=True,
            fliplr=0.0, flipud=0.0, degrees=0.0,   # ⚠️ RDD2022 必须关闭翻转
            mosaic=1.0, mixup=0.0,
            project=str(work / "runs"), name=f"exp{num}_{test_c.lower()}",
            exist_ok=True, plots=True,
        )

        # 取最终 mAP
        metrics = model.val(data=data_yaml, verbose=False)
        mAP50 = float(metrics.box.map50)
        mAP = float(metrics.box.map)
        print(f"\n  [结果] mAP@0.5 = {mAP50:.4f}   mAP@0.5:0.95 = {mAP:.4f}")

        summary.append((num, "+".join(train_c), test_c, n_tr,
                        f"{mAP50:.4f}", f"{mAP:.4f}"))

        # 立刻落盘，防止中途中断丢失结果
        _write_summary(work, summary)

    # ---- 汇总 ----
    _write_summary(work, summary)
    _print_analysis(work, summary)


def _write_summary(work: Path, summary):
    out = work / "summary.csv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("编号,训练集,测试集,训练样本数,mAP@0.5,mAP@0.5:0.95\n")
        for row in summary:
            f.write(",".join(str(x) for x in row) + "\n")


def _print_analysis(work: Path, summary):
    d = {row[0]: row for row in summary}

    print("\n" + "=" * 64)
    print("  汇总结果")
    print("=" * 64)
    print(f"{'编号':<5}{'训练集':<40}{'测试集':<10}{'训练样本':<12}{'mAP@0.5':<10}")
    for num, tr, te, n, m50, m in summary:
        print(f"{num:<5}{tr:<40}{te:<10}{n:<12,}{m50:<10}")

    # ---- dry-run 模式：mAP 是占位符 "-"，跳过数值分析 ----
    if any(str(row[4]) == "-" for row in summary):
        print("\n" + "-" * 64)
        print("  [dry-run 模式] 未实际训练，跳过数值分析")
        print("-" * 64)
        print("  划分配置验证完毕。确认上面的样本数无误后，去掉 --dry-run 正式训练：")
        print("      uv run python scripts/cross_domain.py --epochs 100")
        print(f"\n结果已保存至: {work}/summary.csv")
        return

    print("\n" + "-" * 64)
    print("  【对照 A】多源训练的收益    对比 03 与 05")
    print("-" * 64)
    if "03" in d and "05" in d:
        a, b = d["03"][4], d["05"][4]
        print(f"   03 中国单源 → 挪威      mAP@0.5 = {a}")
        print(f"   05 五国多源 → 挪威      mAP@0.5 = {b}")
        print(f"   → 多源训练的跨域收益：{'提升' if b > a else '未提升'}")
    else:
        print("   （缺少 03 或 05 的结果）")

    print("\n" + "-" * 64)
    print("  【对照 B】全域训练的成本    对比 01 与 06")
    print("-" * 64)
    if "01" in d and "06" in d:
        a, b = d["01"][4], d["06"][4]
        print(f"   01 中国单源 → 中国      mAP@0.5 = {a}   ← 域内上界")
        print(f"   06 全域联合 → 中国      mAP@0.5 = {b}   ← 加入其他域的代价")
        print(f"   → 域内精度代价：{'下降' if b < a else '未下降'}")
    else:
        print("   （缺少 01 或 06 的结果）")

    print("\n" + "-" * 64)
    print("  【跨域衰减】单源训练跨到各国的衰减幅度")
    print("-" * 64)
    if "01" in d:
        base = float(d["01"][4])
        for num, label in [("02", "日本"), ("03", "挪威"), ("04", "美国")]:
            if num in d:
                v = float(d[num][4])
                drop = base - v
                print(f"   中国 → {label:<4}  mAP@0.5 = {v:.4f}    "
                      f"较域内下降 {drop:.4f} ({drop/base*100:.1f}%)")

    print(f"\n结果已保存至: {work}/summary.csv")
    print(f"训练曲线与混淆矩阵: {work}/runs/")


if __name__ == "__main__":
    main()
