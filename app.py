import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO, StringIO
from PIL import Image

from face_detector import detect_faces, extract_face_roi
from expression_recognizer import ExpressionRecognizer, EMOTIONS
from analyzer import ResultAnalyzer
from utils import (
    draw_face_boxes, apply_mood_filter,
    make_emoji_overlay, EMOTION_COLORS
)

# ── 页面配置 ──────────────────────────────────
st.set_page_config(
    page_title="表情识别系统",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 全局状态初始化 ────────────────────────────
if "analyzer" not in st.session_state:
    st.session_state.analyzer = ResultAnalyzer()
if "recognizer" not in st.session_state:
    st.session_state.recognizer = ExpressionRecognizer()

analyzer = st.session_state.analyzer
recognizer = st.session_state.recognizer

# ── 侧边栏 ─────────────────────────────────────
with st.sidebar:
    st.title("🎭 表情识别系统")
    st.markdown("---")

    mode = st.radio(
        "📌 选择模式",
        ["🪞 魔镜模式", "📊 仪表盘模式"],
    )

    st.markdown("---")
    input_type = st.radio(
        "📥 输入来源",
        ["📷 上传图片", "🎬 上传视频", "🎥 实时摄像头"],
    )

    confidence_threshold = st.slider(
        "🎚️ 检测阈值",
        min_value=0.3, max_value=0.9, value=0.5, step=0.05,
    )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 重置统计", use_container_width=True):
            analyzer.clear()
            st.rerun()
    with col2:
        if analyzer.records:
            buf = StringIO()
            analyzer.export_csv(buf)
            st.download_button(
                "📥 导出CSV",
                data=buf.getvalue(),
                file_name=f"emotion_records_{datetime.now():%Y%m%d_%H%M%S}.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ── 处理函数 ──────────────────────────────────
@st.cache_resource
def get_recognizer():
    return ExpressionRecognizer()


def process_frame(image: np.ndarray):
    """处理单帧：检测人脸 → 识别表情"""
    faces = detect_faces(image)
    emotions = []
    for face in faces:
        if face.confidence < confidence_threshold:
            continue
        roi = extract_face_roi(image, face)
        probs = recognizer.recognize(roi)
        emotions.append(probs)
    return faces, emotions


def draw_results_mirror(image, faces, emotions):
    """魔镜模式渲染"""
    img = image.copy()
    for face, probs in zip(faces, emotions):
        top_emo = max(probs, key=probs.get)
        img = apply_mood_filter(img, top_emo)
        img = make_emoji_overlay(img, face, top_emo)
    return img


def draw_results_dashboard(image, faces, emotions):
    """仪表盘模式渲染"""
    return draw_face_boxes(image, faces, emotions)


def save_records(faces, emotions, source_name):
    """保存检测记录到分析器"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i, (face, probs) in enumerate(zip(faces, emotions)):
        top_emo = max(probs, key=probs.get)
        top_conf = probs[top_emo]
        analyzer.add_record(ts, source_name, i + 1, top_emo, top_conf)


def render_stats_panel():
    """渲染统计面板"""
    summary = analyzer.get_summary()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👥 检测人次", summary["total_people"])
    with col2:
        st.metric("🎯 主导表情", summary["main_expression"])
    with col3:
        active = sum(1 for v in summary["expression_count"].values() if v > 0)
        st.metric("🌈 表情种类", active)

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        # 环形图
        data = {e: summary["expression_count"][e] for e in EMOTIONS
                if summary["expression_count"][e] > 0}
        if data:
            colors_rgb = []
            for e in data.keys():
                # EMOTION_COLORS is BGR, convert to RGB hex
                bgr = EMOTION_COLORS.get(e, (128, 128, 128))
                rgb = f"rgb({int(bgr[2]*0.8)},{int(bgr[1]*0.8)},{int(bgr[0]*0.8)})"
                colors_rgb.append(rgb)
            fig = go.Figure(data=[go.Pie(
                labels=list(data.keys()),
                values=list(data.values()),
                hole=0.5,
                marker_colors=colors_rgb,
            )])
            fig.update_layout(
                title="情绪占比",
                height=320,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        # 柱状图
        if data:
            fig = px.bar(
                x=list(data.keys()),
                y=list(data.values()),
                title="各表情计数",
                labels={"x": "表情", "y": "次数"},
            )
            fig.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    # CSV 记录表格
    if analyzer.records:
        st.markdown("---")
        st.subheader("📋 检测记录")
        df = pd.DataFrame(analyzer.records)
        st.dataframe(df, use_container_width=True, height=200)


# ── 主界面 ─────────────────────────────────────
st.title("🎭 智能表情识别系统")
st.caption("人脸表情识别 — IoT 综合大作业")

if input_type == "📷 上传图片":
    uploaded = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        faces, emotions = process_frame(image)

        st.markdown("---")
        col_img, col_stats = st.columns([2, 1])

        with col_img:
            if "🪞" in mode:
                result = draw_results_mirror(image, faces, emotions)
            else:
                result = draw_results_dashboard(image, faces, emotions)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                     caption=f"检测到 {len(faces)} 张人脸",
                     use_container_width=True)

        with col_stats:
            save_records(faces, emotions, uploaded.name)
            render_stats_panel()

elif input_type == "🎬 上传视频":
    uploaded = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])
    if uploaded:
        tmp_path = f"_tmp_{uploaded.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())

        cap = cv2.VideoCapture(tmp_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.info(f"视频共 {frame_count} 帧，正在处理...")

        progress = st.progress(0)
        frame_placeholder = st.empty()
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed += 1
            if processed % 5 != 0:  # 每5帧处理一次
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces, emotions = process_frame(rgb)

            if "🪞" in mode:
                result = draw_results_mirror(rgb, faces, emotions)
            else:
                result = draw_results_dashboard(rgb, faces, emotions)

            frame_placeholder.image(result, use_container_width=True)
            save_records(faces, emotions, uploaded.name)
            progress.progress(min(processed / frame_count, 1.0))

        cap.release()
        progress.empty()
        st.success("视频处理完成!")

elif input_type == "🎥 实时摄像头":
    camera_photo = st.camera_input("📸 点击拍照", key="emotion_camera")

    if camera_photo:
        pil_img = Image.open(camera_photo).convert("RGB")
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        faces, emotions = process_frame(image)

        st.markdown("---")
        col_img, col_stats = st.columns([2, 1])

        with col_img:
            if "🪞" in mode:
                result = draw_results_mirror(image, faces, emotions)
            else:
                result = draw_results_dashboard(image, faces, emotions)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                     caption=f"检测到 {len(faces)} 张人脸",
                     use_container_width=True)

        with col_stats:
            save_records(faces, emotions, "camera_capture")
            render_stats_panel()


# ── 底部统计 ──────────────────────────────────
with st.sidebar:
    st.markdown("---")
    if analyzer.records:
        summary = analyzer.get_summary()
        st.subheader("📊 实时统计")
        st.write(f"检测人次: **{summary['total_people']}**")
        st.write(f"主导表情: **{summary['main_expression']}**")
        for e in EMOTIONS:
            if summary["expression_count"][e] > 0:
                st.write(f"{e}: {summary['expression_count'][e]} "
                         f"({summary['expression_ratio'][e]}%)")
