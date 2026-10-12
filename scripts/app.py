"""
================================================================
系统界面：道路病害智能检测与养护决策系统
================================================================
用 Gradio 实现，几十行出界面，最省力。

功能：
  Tab 1 - 单图检测（检测框 + VLM 养护建议）
  Tab 2 - 批量检测（统计表 + 导出）
  Tab 3 - 视频检测

运行：
  python app.py
  浏览器打开 http://127.0.0.1:7860

⚠️ 答辩前务必录屏！VLM 走 API 时有断网风险。
================================================================
"""

import os
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import pandas as pd
from ultralytics import YOLO

# 自动从 .env 读取配置（API key、VLM_PROVIDER 等）
try:
    from dotenv import load_dotenv
    for _p in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"):
        if _p.exists():
            load_dotenv(_p)
            break
except ImportError:
    pass

# ------------------------------------------------------------------
# 配置区（改这里就行）
# ------------------------------------------------------------------
YOLO_WEIGHTS = "runs/road_damage/yolo11n/weights/best.pt"
USE_VLM = os.getenv("USE_VLM", "0") == "1"   # 设为 1 启用 VLM 复核
CONF_THRESHOLD = 0.4

CLASS_CN = {
    "Longitudinal": "纵向裂缝",
    "Transverse": "横向裂缝",
    "Alligator": "龟裂",
    "Pothole": "坑洞",
}
SEVERITY_CN = {"minor": "轻微", "moderate": "中等", "severe": "严重"}

# 延迟加载，避免启动就占显存
_model = None
_vlm = None


def get_model():
    global _model
    if _model is None:
        if not Path(YOLO_WEIGHTS).exists():
            raise FileNotFoundError(
                f"找不到权重 {YOLO_WEIGHTS}\n"
                f"请先运行 train.py 训练，或修改 app.py 顶部的 YOLO_WEIGHTS")
        _model = YOLO(YOLO_WEIGHTS)
    return _model


def get_vlm():
    """懒加载 VLM 后端（服务商读 .env 里的 VLM_PROVIDER）"""
    global _vlm
    if _vlm is None:
        import vlm_review
        _vlm = vlm_review.APIBackend()
    return _vlm


# ------------------------------------------------------------------
# 核心：检测 + 绘制
# ------------------------------------------------------------------
COLORS = {0: (0, 255, 0), 1: (255, 200, 0), 2: (0, 165, 255), 3: (0, 0, 255)}


def detect_and_draw(image, use_vlm_flag: bool):
    """返回 (标注图, 统计DataFrame, 报告文本)"""
    global _vlm
    model = get_model()
    r = model.predict(image, conf=CONF_THRESHOLD, verbose=False)[0]

    img = r.orig_img.copy()
    rows = []
    report_lines = []

    for i, box in enumerate(r.boxes, 1):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        cls_name = r.names[cls_id]
        conf = float(box.conf[0])
        cn = CLASS_CN.get(cls_name, cls_name)

        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS.get(cls_id, (255, 255, 255)), 2)
        cv2.putText(img, f"{cn} {conf:.2f}", (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    COLORS.get(cls_id, (255, 255, 255)), 2)

        row = {"序号": i, "类别": cn, "置信度": round(conf, 3),
               "位置": f"({x1},{y1})-({x2},{y2})"}

        # ---- VLM 复核 ----
        if use_vlm_flag and conf >= 0.5:
            try:
                import vlm_review
                from PIL import Image as PILImage
                roi = img[max(0, y1 - 10):y2 + 10, max(0, x1 - 10):x2 + 10]
                roi_pil = PILImage.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                prompt = vlm_review.PROMPT_TEMPLATE.format(cls_name=cn)
                res = get_vlm().analyze(roi_pil, prompt)

                is_real = res.get("is_real_damage")
                row["复核"] = "真实病害" if is_real else ("误检" if is_real is False else "未判定")
                row["严重度"] = SEVERITY_CN.get(res.get("severity", ""), "-")
                if res.get("suggestion"):
                    report_lines.append(f"{i}. 【{cn}】{res['suggestion']}")
            except Exception as e:
                # VLM 出错不能让整个检测挂掉
                row["复核"] = "调用失败"
                report_lines.append(f"{i}. 【{cn}】大模型调用失败：{e}")
                _vlm = None   # 下次重试时重建后端

        rows.append(row)

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["序号", "类别", "置信度", "位置"])

    # ---- 汇总报告 ----
    summary = [f"共检出 {len(rows)} 处病害"]
    if rows:
        counts = df["类别"].value_counts()
        summary.append("分类统计：" + "、".join(f"{k} {v} 处" for k, v in counts.items()))
    if report_lines:
        summary.append("\n【养护建议】")
        summary.extend(report_lines)

    return img, df, "\n".join(summary)


# ------------------------------------------------------------------
# 批量
# ------------------------------------------------------------------
def batch_detect(files):
    if not files:
        return pd.DataFrame(), "请先上传图片"

    model = get_model()
    all_rows = []
    for f in files:
        path = f.name if hasattr(f, "name") else f
        r = model.predict(path, conf=CONF_THRESHOLD, verbose=False)[0]
        name = Path(path).name
        for box in r.boxes:
            cls_id = int(box.cls[0])
            all_rows.append({
                "文件": name,
                "类别": CLASS_CN.get(r.names[cls_id], r.names[cls_id]),
                "置信度": round(float(box.conf[0]), 3),
            })

    if not all_rows:
        return pd.DataFrame(), "未检出任何病害"

    df = pd.DataFrame(all_rows)
    pivot = df.pivot_table(index="文件", columns="类别",
                           values="置信度", aggfunc="count", fill_value=0)
    summary = (f"共处理 {len(files)} 张图，检出 {len(df)} 处病害\n\n"
               f"分类汇总：\n{df['类别'].value_counts().to_string()}")
    return pivot, summary


# ------------------------------------------------------------------
# 视频
# ------------------------------------------------------------------
def video_detect(video_path, progress=gr.Progress()):
    if not video_path:
        return None, "请先上传视频"

    model = get_model()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "无法打开视频"

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = Path(tempfile.gettempdir()) / "rdd_output.mp4"
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    total_det = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % max(1, int(fps)) == 0:   # 每秒抽 1 帧检测，提速
            progress(idx / max(1, total), desc=f"处理中 {idx}/{total}")

        r = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        total_det += len(r.boxes)
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          COLORS.get(cls_id, (255, 255, 255)), 2)
        writer.write(frame)

    cap.release()
    writer.release()
    return str(out_path), f"处理 {idx} 帧，累计检出 {total_det} 处病害"


# ------------------------------------------------------------------
# 界面
# ------------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="道路病害智能检测与养护决策系统") as demo:
        gr.Markdown("# 🛣️ 道路病害智能检测与养护决策系统")
        gr.Markdown("基于 YOLO 与多模态大模型协同 | RDD2022 数据集")

        with gr.Tab("单图检测"):
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(label="上传路面图像", type="numpy")
                    use_vlm = gr.Checkbox(
                        label="启用大模型养护建议（需联网）",
                        value=USE_VLM and bool(os.getenv("VLM_API_KEY")))
                    btn = gr.Button("开始检测", variant="primary")
                with gr.Column():
                    img_out = gr.Image(label="检测结果")
                    report = gr.Textbox(label="检测报告", lines=8)
            table = gr.Dataframe(label="病害明细")

            btn.click(detect_and_draw, [img_in, use_vlm], [img_out, table, report])

        with gr.Tab("批量检测"):
            files = gr.Files(label="上传多张图片", file_count="multiple")
            btn2 = gr.Button("批量检测", variant="primary")
            out_table = gr.Dataframe(label="分类统计表")
            out_text = gr.Textbox(label="汇总", lines=6)
            btn2.click(batch_detect, [files], [out_table, out_text])

        with gr.Tab("视频检测"):
            vid_in = gr.Video(label="上传视频")
            btn3 = gr.Button("开始检测", variant="primary")
            vid_out = gr.Video(label="检测结果")
            vid_txt = gr.Textbox(label="统计")
            btn3.click(video_detect, [vid_in], [vid_out, vid_txt])

    return demo


if __name__ == "__main__":
    print("[系统] 启动中...")
    print(f"[系统] YOLO 权重: {YOLO_WEIGHTS}")
    print(f"[系统] VLM 复核: {'启用' if USE_VLM else '关闭'}")
    build_ui().launch(server_name="127.0.0.1", server_port=7860, show_error=True)
