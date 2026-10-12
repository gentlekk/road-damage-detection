# 道路病害智能检测与养护决策系统

> 本科毕业设计 | 基于 YOLO 与多模态大模型协同

## 文件说明

| 文件 | 用途 |
|---|---|
| `scripts/download_data.py` | ⭐ **数据集下载与验证**（自动检查文件名格式） |
| `scripts/data.yaml` | 数据集配置（由上面脚本自动写入） |
| `scripts/train.py` | 实验一：YOLO11 基线训练（显存自适应） |
| `scripts/zeroshot.py` | 实验二：YOLO-World 零样本 vs 微调 |
| `scripts/cross_domain.py` | 实验三：跨域泛化评估（留一法，6 组） |
| `scripts/vlm_review.py` | 实验四：VLM 复核 + 养护建议生成 |
| `scripts/app.py` | 系统界面（Gradio） |
| `pyproject.toml` | uv 环境配置（依赖 + CUDA 索引 cu121） |
| `.python-version` | **锁定 Python 3.10**（告诉 uv 用哪个版本） |
| `.env.example` | API key 模板（复制成 `.env`） |
| `requirements.txt` | pip 回退用（uv 不读它） |
| `CLAUDE.md` | AI 助手上下文（非交付物） |

---

## 快速开始（5 步）

### 第 1 步：环境（用 uv）

```bash
# 装 uv（已装 Python 的话）
pip install uv

# 装 Python 3.10（.python-version 文件已锁定用这个版本）
uv python install 3.10

# ⚠️ 先改 pyproject.toml：把 cu121 改成你显卡支持的版本
#    查版本：nvidia-smi（看右上角 CUDA Version）
#    要改 4 处（index 的 name/url、sources 的 torch/torchvision）

# 一键装依赖
uv sync
```

**关于 Python 版本**：项目里有两个地方管版本，分工不同：

| 位置 | 作用 | 当前值 |
|---|---|---|
| `.python-version` | **指定实际用哪个**（uv 读它） | `3.10` |
| `pyproject.toml` 的 `requires-python` | 声明兼容范围（给别人看） | `>=3.10,<3.13` |

**验证实际用的是哪个**：

```bash
uv run python --version     # 应该输出 Python 3.10.x
```

验证：

```bash
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

**必须输出 `CUDA: True`**，否则 CUDA 版本填错了，回上一步改。

> 注意：以后所有命令都要加 `uv run` 前缀，例如 `uv run python scripts/train.py`。

### 第 2 步：下载并验证数据集（一条命令）

```bash
uv run python scripts/download_data.py
```

**这条命令会自动做四件事**：

1. 用 `kagglehub` 下载 RDD2022（不用手动配 token）
2. **检查文件名是否含国家前缀** ← 决定实验三能不能做
3. 检查目录结构与标注完整性
4. 把路径自动写进 `scripts/data.yaml`

**看输出的两个关键结论**：

```
  目录结构：  OK
  实验三可用：是        ← 这个必须是「是」，否则实验三做不了
```

如果显示 **「实验三可用：否」**，说明文件名被改写过、丢失了国家信息，有两个选择：
1. 放弃实验三（它是可选项）
2. 去下官方按国家分包的 zip（VOC 格式，需转换）：https://crddc2022.sekllab.global/data/

**前置条件**：需要 Kaggle 凭证。先去 https://www.kaggle.com/settings → API → Create New Token，把下载的 `kaggle.json` 放到 `C:/Users/你的用户名/.kaggle/kaggle.json`。

> **已经手动下好了？** 跳过下载，只验证：
> ```bash
> uv run python scripts/download_data.py --check-only "D:/你的数据集目录"
> ```

### 第 3 步：试训 1 个 epoch（验证整条链路）

**整个流程中最重要的一次验证**——用几分钟换掉后面几天的风险。

```bash
uv run python scripts/train.py --epochs 1
```

成功标志：结束出现 `runs/road_damage/yolo11n/weights/best.pt`

4GB 显存报 OOM 的话降配：

```bash
uv run python scripts/train.py --epochs 1 --batch 4 --imgsz 512
```

### 第 4 步：跑基线训练（实验一）

```bash
# 一次训练三个模型做对比
uv run python scripts/train.py --all --epochs 100 --batch 8
```

产出在 `runs/road_damage/yolo11n/`：
- `weights/best.pt` —— 部署用
- `results.png` —— 训练曲线（论文用）
- `confusion_matrix.png` —— 混淆矩阵（论文用）

### 第 5 步：跑零样本对比（实验二）

```bash
# 三组提示词做消融（建议都跑）
uv run python scripts/zeroshot.py --prompt 简短语
uv run python scripts/zeroshot.py --prompt 描述语
uv run python scripts/zeroshot.py --prompt 带上下文
```

**为什么要跑三组**：零样本效果完全取决于提示词。只跑一组的话，万一结果很差，会被人质疑「是不是你提示词没写好」。三组都不行，才能证明是**方法本身的固有限制**。

---

## 运行 VLM 复核（实验四）

### ⚠️ 模型 ID 必须带 `-flash` 才是免费的

```
glm-4v-flash      ✅ 免费（永久）
glm-4.6v-flash    ✅ 免费（永久）
glm-4v            ❌ 按量计费，会扣钱
```

### 配置（推荐用 .env，设一次就够）

```bash
cp .env.example .env
```

然后编辑 `.env`：

```ini
VLM_PROVIDER=zhipu
VLM_API_KEY=你申请的key
```

**查看所有免费服务商**：`uv run python scripts/vlm_review.py --list-providers`

**换服务商**：改 `.env` 里的 `VLM_PROVIDER` 一行即可（`zhipu` / `siliconflow` / `dashscope` / `modelscope` / `openrouter`）

### 运行

```bash
uv run python scripts/vlm_review.py --image test.jpg
```

免费方案对比：

| 平台 | 免费模型 | 免费形式 | 国内直连 |
|---|---|---|---|
| **智谱 AI** | `glm-4v-flash` | ⭐ **永久免费** | ✅ |
| **硅基流动** | `Qwen2.5-VL-7B` 等 | ⭐ **永久免费** | ✅ |
| 阿里云百炼 | `qwen3-vl-flash` | 新用户额度（3 个月） | ✅ |
| OpenRouter | `qwen2.5-vl-7b:free` | 50 次/天 | ⚠️ 需代理 |

> 💡 **为什么推荐"永久免费"而不是"送额度"**：毕设跨几个月，送的额度会过期。

---

## 启动系统界面

```bash
uv run python scripts/app.py
```

浏览器打开 http://127.0.0.1:7860

启用 VLM 复核（需先设好 `VLM_API_KEY`）：

```bash
set USE_VLM=1
uv run python scripts/app.py
```

---

## ⚠️ 三个必看的坑

### 坑 1：`fliplr` 必须设为 0

水平翻转会把「纵向裂缝」变成「横向裂缝」，**标签语义直接出错**。所有训练脚本里已经设好了，但如果你手敲 `yolo train` 命令，记得加 `fliplr=0.0`。

### 坑 2：CUDA 版本不匹配

`torch.cuda.is_available()` 返回 `False` 是最常见的翻车点。

**PyTorch 不走 PyPI，走独立索引**——`pyproject.toml` 里已经配好了（默认 `cu121`）。如果你的显卡驱动不是 CUDA 12.1，要改**四处**：

```toml
[[tool.uv.index]]
name = "pytorch-cu121"                              # ← 改
url = "https://download.pytorch.org/whl/cu121"      # ← 改
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu121" }                 # ← 改
torchvision = { index = "pytorch-cu121" }           # ← 改
```

改完删掉 `.venv` 和 `uv.lock`，重新 `uv sync`。

**⚠️ 绝对不要 `pip install torch`**——那会装成 CPU 版，GPU 用不了。

### 坑 3：答辩现场断网

VLM 走 API 时，断网 = 演示失败。**必须提前录屏**，并且演示时优先走纯 YOLO 路径（`USE_VLM=0`）。

---

## 实验产出对照表

> ⚠️ **先分清「训练哪个模型」**——四个实验涉及的模型不同，最容易搞混的是实验二。

| 实验 | 脚本 | **训练哪个模型** | 产出 |
|---|---|---|---|---|
| **一、基线对比** | `train.py --all` | YOLOv8n / YOLO11n / YOLO11s | mAP/FPS 对比表 + 训练曲线 |
| **二、零样本 vs 微调** | `zeroshot.py` | 方案 A/B → **YOLO-World**<br>方案 C → YOLO11n | 样本量-mAP 曲线 |
| **三、跨域泛化** | `cross_domain.py`<br>（或 `train.py --cross-domain`） | YOLO11n（6 组不同训练集划分） | 跨域 mAP 下降表 + `summary.csv` |
| **四、VLM 复核** | `vlm_review.py` | YOLO11n + VLM<br>（**VLM 全程不训练**） | 误检率对比表 + 案例展示 |

### 实验三的用法

```bash
# 先 dry-run 验证划分配置（不训练，几秒出结果）
uv run python scripts/cross_domain.py --dry-run

# 跑全部 6 组
uv run python scripts/cross_domain.py --epochs 100

# 只跑第 3、5 组
uv run python scripts/cross_domain.py --only 03 05 --epochs 100
```

**⚠️ 前置条件**：数据集必须保留 RDD2022 原始文件名（文件名含国家前缀），例如 `China_Drone_000001.jpg`、`Norway_000123.jpg`、`United_States_000007.jpg`。脚本靠文件名识别国家，若文件名已被改写则无法按域划分。

**留一法设计**：第 ⑤ 组训练时**排除挪威**，确保"挪威未见"与第 ③ 组口径一致——否则拿域内成绩和跨域成绩对比是无效的。

### 两个易混淆点

**① 实验二微调的是 YOLO-World，不是 YOLO11n**

| 方案 | 训练哪个模型 | 训练数据 |
|---|---|---|
| A. 零样本 | YOLO-World | 不训练 |
| B. 少样本微调 | **YOLO-World** | RDD2022 训练集的 10%/30%/50%/100% 子集 |
| C. 全监督 | YOLO11n | RDD2022 训练集全量 |

数据**不用另找**，是从 RDD2022 训练集里按固定随机种子抽的子集。

**② 实验四的 VLM 不训练**

VLM 全程零样本使用，只做提示词调优。若需微调 VLM 提升分级准确率，需要先构造带严重程度标注的数据集，属于另一个量级的工作。

## 显存参考（RTX 3050 4GB）

| 场景 | 配置 | 是否可跑 |
|---|---|---|
| YOLO11n 训练 | `batch=4~8, imgsz=640, amp=True` | ✅ |
| YOLO11s 训练 | `batch=4, imgsz=512` | ⚠️ 勉强 |
| YOLO 推理 | 默认 | ✅ 流畅 |
| VLM 本地 | — | ❌ 需服务器 |
| 界面演示（纯 YOLO） | 默认 | ✅ |
