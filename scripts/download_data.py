"""
================================================================
数据集下载与验证
================================================================
数据集固定放在项目里，不用记路径：

    graduation-project/                      ← 项目根 = kagglehub 缓存根
    └── datasets/                            ← kagglehub 自动创建
        └── sreekaraditya/                   ← Kaggle 用户名
            └── rdd2022-yolo-crackscan-v2/   ← 数据集名
                └── versions/1/              ← 版本号
                    ├── train/images/ + train/labels/
                    ├── valid/images/ + valid/labels/
                    └── test/images/  + test/labels/

脚本会自动把这个路径写进 scripts/data.yaml，你不需要手动填。

用法：
  uv run python scripts/download_data.py                   # 下载 + 验证 + 写配置
  uv run python scripts/download_data.py --check-only PATH # 只验证已有目录

重复运行是安全的（kagglehub 自带缓存，第二次秒完成）。

前置条件：Kaggle 凭证，二选一
  1) ~/.kaggle/kaggle.json
  2) 环境变量 KAGGLE_USERNAME + KAGGLE_KEY
  获取：https://www.kaggle.com/settings → API → Create New Token
================================================================
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

# 项目根目录（scripts/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 把 kagglehub 的「缓存根」指向项目根目录，数据就落在项目里了。
#
# kagglehub 的固定布局是： <缓存根>/datasets/<作者>/<数据集>/versions/<版本>/
# 所以缓存根设成项目根，最终路径就是：
#     <项目>/datasets/sreekaraditya/rdd2022-yolo-crackscan-v2/versions/1/
#
# ⚠️ 注意别设成 PROJECT_ROOT/"datasets"，那样会变成 datasets/datasets/ 两层
os.environ.setdefault("KAGGLEHUB_CACHE", str(PROJECT_ROOT))

# 复用 cross_domain.py 里已验证过的国家识别逻辑
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from cross_domain import COUNTRY_PREFIXES, classify_country
except ImportError:
    print("[错误] 找不到 cross_domain.py，请确认脚本在同一目录下")
    sys.exit(1)

DATASET_ID = "sreekaraditya/rdd2022-yolo-crackscan-v2"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ==================================================================
def check_credentials() -> bool:
    """下载前预检 Kaggle 凭证"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_file = kaggle_json.exists()
    has_env = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))

    print("=" * 62)
    print("  预检：Kaggle 凭证")
    print("=" * 62)
    print(f"\n  kaggle.json: {kaggle_json}  →  {'存在' if has_file else '不存在'}")
    print(f"  环境变量凭证: {'已设置' if has_env else '未设置'}")

    if has_file or has_env or os.getenv("KAGGLE_API_TOKEN"):
        print("\n  [通过]\n")
        return True

    print("\n  [未通过] 没有 Kaggle 凭证，下载会失败\n")
    print("  配置步骤（只需一次）：")
    print("    1. 登录 https://www.kaggle.com/settings")
    print("    2. 滚动到 'API' 区块 → 点 'Create New Token'")
    print("    3. 下载得到的 kaggle.json 放到：")
    print(f"       {kaggle_json.parent}")
    print("       （没有 .kaggle 文件夹就自己建一个）")
    print("    4. 重新运行本脚本\n")
    return False


# ==================================================================
def download() -> Path:
    """下载数据集，返回版本目录"""
    if not check_credentials():
        sys.exit(1)

    try:
        import kagglehub
    except ImportError:
        print("[错误] 未安装 kagglehub，请先执行：uv sync")
        sys.exit(1)

    print("=" * 62)
    print("  下载数据集")
    print("=" * 62)
    print(f"\n  数据集：{DATASET_ID}")
    print(f"  存放位置：{PROJECT_ROOT / 'datasets'}")
    print("\n  [提示] 约 12GB，视网速需要 20~40 分钟")
    print("         中途断了可以直接重跑，会续传/复用已下载部分")
    print("         已下过的话是秒完成\n")

    try:
        return Path(kagglehub.dataset_download(DATASET_ID))
    except Exception as e:
        print(f"\n[下载失败] {e}\n")
        if any(k in str(e) for k in ("403", "401", "credential")):
            print("这通常是 Kaggle 凭证问题，见上方预检部分的配置步骤")
        sys.exit(1)


# ==================================================================
def find_data_root(root: Path):
    """在下载目录里找到含 train/images 的那一层"""
    if not root.exists():
        return None
    for c in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        if (c / "train" / "images").is_dir():
            return c
    return None


def scan_structure(root: Path) -> dict:
    """找出 train / val / test 的图像与标注目录"""
    found = {}
    for d in root.rglob("*"):
        if not d.is_dir() or d.name.lower() != "images":
            continue
        split = {"train": "train", "valid": "val", "val": "val",
                 "validation": "val", "test": "test"}.get(d.parent.name.lower())
        if not split:
            continue
        labels = d.parent / "labels"
        found[split] = {"images": d, "labels": labels if labels.exists() else None}
    return found


def count_files(d) -> int:
    return sum(1 for f in Path(d).iterdir() if f.is_file()) if d and Path(d).exists() else 0


# ==================================================================
def check_structure(struct: dict) -> bool:
    """检查一：目录结构与标注完整性"""
    print("=" * 62)
    print("  检查一：目录结构与标注完整性")
    print("=" * 62 + "\n")

    if not struct:
        print("  [错误] 没找到 train/val/test 目录结构")
        return False

    ok = True
    for split in ("train", "val", "test"):
        if split not in struct:
            print(f"  {split:<6} [缺失]")
            ok &= split == "test"          # 缺 test 不影响训练
            continue

        n_img = count_files(struct[split]["images"])
        lbl = struct[split]["labels"]
        n_lbl = count_files(lbl) if lbl else 0

        if lbl is None:
            print(f"  {split:<6} 图像 {n_img:>7,}  标注 [缺失]")
            ok = False
        elif n_img == n_lbl:
            print(f"  {split:<6} 图像 {n_img:>7,}  标注 {n_lbl:>7,}   [OK]")
        else:
            print(f"  {split:<6} 图像 {n_img:>7,}  标注 {n_lbl:>7,}   [不一致 差{abs(n_img-n_lbl)}]")
            ok = False
    return ok


def check_filenames(images_dir: Path, sample: int = 5) -> bool:
    """检查二：文件名是否含国家前缀 —— 决定实验三能否做"""
    files = [f for f in images_dir.iterdir()
             if f.is_file() and f.suffix.lower() in IMG_EXTS]

    print("\n" + "=" * 62)
    print("  检查二：文件名格式（决定实验三能否做）")
    print("=" * 62)

    if not files:
        print("\n  [错误] 目录下没有图像")
        return False

    print(f"\n  抽样 {sample} 个：")
    for f in files[:sample]:
        c = classify_country(f.stem)
        print(f"    {f.name:<38} -> {c if c else '(无法识别)'}")

    dist, unknown = Counter(), 0
    for f in files:
        c = classify_country(f.stem)
        dist[c] += 1 if c else 0
        unknown += 0 if c else 1

    total = len(files)
    print(f"\n  国家分布（共 {total:,} 张）：")
    for c in COUNTRY_PREFIXES:
        if dist.get(c):
            print(f"    {c:<8}{dist[c]:>8,}  {'#' * int(36 * dist[c] / total)}")
    if unknown:
        print(f"    {'未识别':<8}{unknown:>8,}  ({unknown/total*100:.1f}%)")

    print()
    if unknown == 0 or unknown / total < 0.02:
        print("  [通过] 文件名含国家前缀，实验三可以做")
        return True

    print(f"  [不通过] {unknown} 张（{unknown/total*100:.1f}%）无法识别国家前缀")
    print("           → 实验三（跨域泛化评估）无法进行")
    print("           → 可以放弃实验三，或去下官方按国家分包的 zip：")
    print("             https://crddc2022.sekllab.global/data/")
    return False


# ==================================================================
def write_config(root: Path, struct: dict):
    """把路径写进 scripts/data.yaml"""
    cfg = Path(__file__).resolve().parent / "data.yaml"
    if not cfg.exists():
        print(f"\n[跳过] 找不到 {cfg}")
        return

    out = []
    for line in cfg.read_text(encoding="utf-8").split("\n"):
        s = line.strip()
        if s.startswith("path:"):
            out.append(f"path: {root.as_posix()}")
        elif s.startswith("train:") and "train" in struct:
            out.append(f"train: {struct['train']['images'].relative_to(root).as_posix()}")
        elif s.startswith("val:") and "val" in struct:
            out.append(f"val: {struct['val']['images'].relative_to(root).as_posix()}")
        elif s.startswith("test:") and "test" in struct:
            out.append(f"test: {struct['test']['images'].relative_to(root).as_posix()}")
        else:
            out.append(line)

    cfg.write_text("\n".join(out), encoding="utf-8")
    print(f"\n[完成] 配置已写入：{cfg}")
    print(f"       path = {root.as_posix()}")


# ==================================================================
def main():
    p = argparse.ArgumentParser(description="RDD2022 下载与验证")
    p.add_argument("--check-only", metavar="PATH",
                   help="跳过下载，只验证指定目录")
    p.add_argument("--no-set-config", action="store_true",
                   help="不自动写入 scripts/data.yaml（默认会写）")
    args = p.parse_args()

    if args.check_only:
        root = Path(args.check_only).resolve()
        if not root.exists():
            print(f"[错误] 目录不存在：{root}")
            sys.exit(1)
        print(f"[验证已有目录] {root}\n")
    else:
        root = find_data_root(download())
        if root is None:
            print("[错误] 下载完成但找不到 train/images")
            sys.exit(1)

    struct = scan_structure(root)
    ok_struct = check_structure(struct)
    ok_names = check_filenames(struct["train"]["images"]) if "train" in struct else False

    print("\n" + "=" * 62)
    print("  总结")
    print("=" * 62)
    print(f"\n  数据集路径：{root.as_posix()}")
    print(f"  目录结构：  {'OK' if ok_struct else '有问题'}")
    print(f"  实验三可用：{'是' if ok_names else '否'}")

    if ok_struct and not args.no_set_config:
        write_config(root, struct)
        print("\n  下一步：跑 1 个 epoch 验证整条链路")
        print("      uv run python scripts/train.py --epochs 1")


if __name__ == "__main__":
    main()
