"""
================================================================
实验四：多模态大模型语义复核 + 养护决策生成
================================================================
创新点 2 & 3 的核心实现

流水线：
  YOLO 检出框 → 裁剪 ROI → VLM 复核 → 结构化决策 → 中文报告

支持两种 VLM 后端（一行配置切换）：
  backend="api"    → 云端 API（推荐，中文效果好）
  backend="local"  → 本地部署（需服务器 / 大显存）

用法：
  python vlm_review.py --image test.jpg
  python vlm_review.py --image test.jpg --backend local
  python vlm_review.py --dir test_images/ --export report.xlsx
================================================================
"""

import argparse
import json
import os
import re
from pathlib import Path

import cv2
from ultralytics import YOLO

# 项目根目录（scripts/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 默认权重：实验一训出来的 YOLO11n（绝对路径，不受运行目录影响）
DEFAULT_WEIGHTS = str(PROJECT_ROOT / "runs" / "road_damage" / "yolo11n" / "weights" / "best.pt")
DEFAULT_OUT = str(PROJECT_ROOT / "runs" / "vlm_review")

# 自动从 .env 读取 API key（找不到 .env 也不会报错）
try:
    from dotenv import load_dotenv
    for _p in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"):
        if _p.exists():
            load_dotenv(_p)
            break
except ImportError:
    pass  # 没装 python-dotenv 就用系统环境变量


# ==================================================================
# 免费 VLM 服务商预设
# ==================================================================
# 切换服务商只需改 .env 里的 VLM_PROVIDER，或命令行加 --provider
# 或直接跑：python vlm_review.py --list-providers
# ==================================================================
PROVIDER_PRESETS = {
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-4v-flash",
        "free": "永久免费（glm-4v-flash / glm-4.6v-flash）",
        "signup": "https://open.bigmodel.cn/",
        "note": "注意模型 ID 必须带 -flash 才是免费，不带会按量计费",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "free": "9B 以下模型永久免费",
        "signup": "https://siliconflow.cn/",
        "note": "新用户送 2000 万 token；模型名带 Pro 前缀为付费版",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-max",
        "free": "新用户免费额度（约 3 个月有效）",
        "signup": "https://bailian.console.aliyun.com/",
        "note": "额度会过期，不适合跨月的毕设",
    },
    "modelscope": {
        "base_url": "https://api-inference.modelscope.cn/v1",
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "free": "每日免费额度",
        "signup": "https://modelscope.cn/",
        "note": "需在魔搭开通 API 推理服务",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "qwen/qwen2.5-vl-7b-instruct:free",
        "free": "50 次/天（充值 $10 后 1000 次/天）",
        "signup": "https://openrouter.ai/",
        "note": "国内需代理；模型名必须带 :free 后缀",
    },
}

DEFAULT_PROVIDER = "zhipu"


# ==================================================================
# 提示词模板（这是创新点 2 的核心，可多设计几组做消融）
# ==================================================================
PROMPT_TEMPLATE = """这是一张从道路巡检图像中裁剪出的区域，目标检测模型将其识别为「{cls_name}」。

请依次完成三个判断：
1. 【真伪】该区域是真实的道路病害，还是阴影、油渍、路面接缝、车道线等干扰？
2. 【分级】若为真实病害，严重程度属于：轻微 / 中等 / 严重？
3. 【建议】给出一句中文养护处置建议。

请严格按以下 JSON 格式输出，不要输出任何其他内容：
{{
  "is_real_damage": true,
  "severity": "moderate",
  "reason": "该区域呈现不规则网状裂纹，符合龟裂特征",
  "suggestion": "建议3个月内进行铣刨重铺处理"
}}"""


# ==================================================================
# VLM 后端抽象 —— 一行切换，答辩可随时降级
# ==================================================================
class VLMBackend:
    """统一接口：analyze(roi_image, prompt) -> dict"""

    def analyze(self, roi_image, prompt: str) -> dict:
        raise NotImplementedError


class APIBackend(VLMBackend):
    """
    云端 API 后端（本项目采用）

    服务商由 .env 里的 VLM_PROVIDER 决定，或命令行 --provider 覆盖。
    全部走 OpenAI 兼容接口，切换服务商不用改代码。
    """

    def __init__(self, model: str = None, provider: str = None):
        from openai import OpenAI

        # 优先级：命令行 > .env 里的 VLM_MODEL/BASE_URL > 服务商预设
        prov = provider or os.getenv("VLM_PROVIDER") or DEFAULT_PROVIDER
        if prov not in PROVIDER_PRESETS:
            raise RuntimeError(
                f"未知的服务商 '{prov}'。\n"
                f"可用：{', '.join(PROVIDER_PRESETS)}\n"
                f"运行 --list-providers 查看详情"
            )
        preset = PROVIDER_PRESETS[prov]

        api_key = os.getenv("VLM_API_KEY")
        base_url = os.getenv("VLM_BASE_URL") or preset["base_url"]
        model = model or os.getenv("VLM_MODEL") or preset["model"]

        if not api_key or api_key.startswith("在这里填"):
            raise RuntimeError(
                "未设置 VLM_API_KEY。\n\n"
                "【推荐做法】复制 .env.example 为 .env，填入 key：\n"
                "    cp .env.example .env\n"
                "    （然后编辑 .env，把 VLM_API_KEY 改成你的 key）\n\n"
                "【临时做法】设置环境变量（关窗口就失效）：\n"
                "    export VLM_API_KEY=你的key\n\n"
                f"免费 key 申请地址（{prov}）：{preset['signup']}"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.provider = prov
        print(f"[VLM] 服务商: {prov}")
        print(f"[VLM] 模型:   {model}")
        print(f"[VLM] 地址:   {base_url}")

    def analyze(self, roi_image, prompt: str) -> dict:
        import base64
        import io

        # PIL Image → base64
        buf = io.BytesIO()
        roi_image.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=512,
                temperature=0.1,        # 低温度保证输出稳定
            )
            return _parse_json(resp.choices[0].message.content)
        except Exception as e:
            return {"is_real_damage": None, "error": str(e)}


class LocalBackend(VLMBackend):
    """本地部署后端（Qwen2.5-VL，需服务器）"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                 load_4bit: bool = False):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        print(f"[VLM] 后端: 本地 ({model_name})")
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            kwargs.pop("torch_dtype")
            print("[VLM] 启用 4bit 量化")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.torch = torch

    def analyze(self, roi_image, prompt: str) -> dict:
        from qwen_vl_utils import process_vision_info

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": roi_image},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

        with self.torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=256,
                                            do_sample=False)

        trimmed = [out[len(inp):] for inp, out in
                   zip(inputs.input_ids, generated)]
        output = self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]

        return _parse_json(output)


def _parse_json(text: str) -> dict:
    """从 VLM 输出里把 JSON 抠出来（模型常带 markdown 围栏或多余文字）"""
    if not text:
        return {"is_real_damage": None, "error": "empty output"}

    # 去掉 ```json ... ``` 围栏
    text = re.sub(r"```(?:json)?", "", text).strip()

    # 直接尝试
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 正则抠第一个 {...}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    return {"is_real_damage": None, "raw": text[:200]}


# ==================================================================
# 级联流水线
# ==================================================================
class CascadePipeline:
    def __init__(self, yolo_weights: str, vlm: VLMBackend,
                 conf_thres: float = 0.4, vlm_min_conf: float = 0.5,
                 padding: int = 10):
        self.detector = YOLO(yolo_weights)
        self.vlm = vlm
        self.conf_thres = conf_thres
        self.vlm_min_conf = vlm_min_conf   # ⭐ 优化点：低置信度不调 VLM，省调用
        self.padding = padding
        self.call_count = 0                # 统计 VLM 调用次数，写论文用

    def detect(self, image):
        r = self.detector.predict(image, conf=self.conf_thres, verbose=False)[0]
        dets = []
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            dets.append({
                "class_id": int(box.cls[0]),
                "class_name": r.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [x1, y1, x2, y2],
            })
        return dets, r.orig_img

    def crop_roi(self, image, bbox):
        from PIL import Image
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        p = self.padding
        x1, y1 = max(0, x1 - p), max(0, y1 - p)
        x2, y2 = min(w, x2 + p), min(h, y2 + p)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        return Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

    def review(self, image_path: str):
        dets, img = self.detect(image_path)

        for d in dets:
            # ⭐ 优化：只对高置信度框调 VLM
            if d["confidence"] < self.vlm_min_conf:
                d["vlm"] = {"skipped": True,
                            "reason": f"conf<{self.vlm_min_conf}，跳过VLM"}
                continue

            roi = self.crop_roi(img, d["bbox"])
            if roi is None:
                continue

            self.call_count += 1
            prompt = PROMPT_TEMPLATE.format(cls_name=d["class_name"])
            d["vlm"] = self.vlm.analyze(roi, prompt)

        return dets, img

    def draw(self, img, dets):
        """在原图上画框 + VLM 结论"""
        out = img.copy()
        colors = {0: (0, 255, 0), 1: (255, 200, 0),
                  2: (0, 165, 255), 3: (0, 0, 255)}
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            c = colors.get(d["class_id"], (255, 255, 255))
            cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)

            vlm = d.get("vlm", {})
            if vlm.get("is_real_damage") is False:
                c, tag = (128, 128, 128), "误检"
            elif vlm.get("severity") == "severe":
                c, tag = (0, 0, 255), "严重"
            else:
                tag = d["class_name"][:6]

            label = f"{tag} {d['confidence']:.2f}"
            cv2.putText(out, label, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        return out


# ==================================================================
def main():
    p = argparse.ArgumentParser(description="YOLO + VLM 级联复核")
    p.add_argument("--image", help="单张图片路径")
    p.add_argument("--dir", help="图片目录，批量处理")
    p.add_argument("--yolo", default=DEFAULT_WEIGHTS,
                   help="检测器权重（默认实验一的 YOLO11n best.pt）")
    p.add_argument("--backend", default="api", choices=["api", "local"],
                   help="api=云端（默认）；local=本地部署（需大显存）")
    p.add_argument("--provider", default=None,
                   choices=list(PROVIDER_PRESETS.keys()),
                   help=f"VLM 服务商，默认读 .env 的 VLM_PROVIDER，再默认 {DEFAULT_PROVIDER}")
    p.add_argument("--vlm-model", default=None,
                   help="留空则用服务商预设的免费模型")
    p.add_argument("--list-providers", action="store_true",
                   help="列出所有支持的免费服务商")
    p.add_argument("--conf", type=float, default=0.4, help="YOLO 置信度阈值")
    p.add_argument("--vlm-min-conf", type=float, default=0.5,
                   help="低于此置信度不调 VLM（省调用）")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="结果输出目录（绝对路径）")
    p.add_argument("--json", action="store_true", help="输出 JSON 结果")
    args = p.parse_args()

    # ---- 列出服务商 ----
    if args.list_providers:
        print("\n可用的免费 VLM 服务商：\n")
        for name, cfg in PROVIDER_PRESETS.items():
            mark = " ← 默认" if name == DEFAULT_PROVIDER else ""
            print(f"  {name}{mark}")
            print(f"      模型:   {cfg['model']}")
            print(f"      免费:   {cfg['free']}")
            print(f"      注册:   {cfg['signup']}")
            print(f"      注意:   {cfg['note']}\n")
        print("切换方式：改 .env 里的 VLM_PROVIDER，或命令行加 --provider <名字>\n")
        return

    # 构建 VLM 后端
    if args.backend == "api":
        vlm = APIBackend(args.vlm_model, provider=args.provider)
    else:
        vlm = LocalBackend(args.vlm_model or "Qwen/Qwen2.5-VL-3B-Instruct")

    pipe = CascadePipeline(args.yolo, vlm,
                           conf_thres=args.conf,
                           vlm_min_conf=args.vlm_min_conf)

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # 收集待处理图片
    if args.image:
        files = [Path(args.image)]
    elif args.dir:
        files = sorted([f for f in Path(args.dir).iterdir()
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    else:
        p.error("请指定 --image 或 --dir")

    all_results = {}
    for f in files:
        dets, img = pipe.review(str(f))
        vis = pipe.draw(img, dets)
        cv2.imwrite(str(Path(args.out) / f"{f.stem}_review.jpg"), vis)

        real = sum(1 for d in dets if d.get("vlm", {}).get("is_real_damage") is True)
        fake = sum(1 for d in dets if d.get("vlm", {}).get("is_real_damage") is False)
        print(f"\n[{f.name}] 检出 {len(dets)} 个框 | 复核确认 {real} | 判为误检 {fake}")
        for d in dets:
            v = d.get("vlm", {})
            if v.get("is_real_damage"):
                print(f"  · {d['class_name']} ({d['confidence']:.2f}) "
                      f"[{v.get('severity','?')}] {v.get('suggestion','')}")

        all_results[f.name] = dets

    print(f"\n{'='*60}")
    print(f"总计：处理 {len(files)} 张图，VLM 调用 {pipe.call_count} 次")
    if len(files):
        print(f"平均每张图调用 {pipe.call_count/len(files):.1f} 次")
    print(f"结果图输出至: {args.out}/")

    if args.json:
        with open(Path(args.out) / "results.json", "w", encoding="utf-8") as fp:
            json.dump(all_results, fp, ensure_ascii=False, indent=2)
        print(f"JSON 结果: {args.out}/results.json")


if __name__ == "__main__":
    main()
