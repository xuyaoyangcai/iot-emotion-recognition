import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import StringIO
from PIL import Image

from face_detector import detect_faces, extract_face_roi
from expression_recognizer import ExpressionRecognizer, EMOTIONS
from analyzer import ResultAnalyzer
from utils import (
    draw_face_boxes, apply_mood_filter,
    make_emoji_overlay, EMOTION_COLORS
)

st.set_page_config(
    page_title="表情识别系统",
    page_icon="🎭",
    layout="wide",
)

# 初始化
if "analyzer" not in st.session_state:
    st.session_state.analyzer = ResultAnalyzer()
if "recognizer" not in st.session_state:
    with st.spinner("正在加载模型..."):
        st.session_state.recognizer = ExpressionRecognizer()

analyzer = st.session_state.analyzer

# ── 侧边栏 ──
with st.sidebar:
    st.title("🎭 表情识别系统")
    mode = st.radio("选择模式", ["🪞 魔镜模式", "📊 仪表盘模式"])
    st.markdown("---")
    input_type = st.radio("输入来源", ["📷 上传图片", "🎬 上传视频", "🎥 摄像头"])
    threshold = st.slider("检测阈值", 0.3, 0.9, 0.5, 0.05)
    st.markdown("---")
    if st.button("🔄 重置统计"):
        analyzer.clear()
        st.rerun()
    if analyzer.records:
        buf = StringIO()
        analyzer.export_csv(buf)
        st.download_button("📥 导出CSV", buf.getvalue(),
                           f"records_{datetime.now():%Y%m%d_%H%M%S}.csv",
                           mime="text/csv")

# ── 核心函数 ──
def process_frame(image):
    faces = detect_faces(image)
    emotions = []
    for face in faces:
        if face.confidence < threshold:
            continue
        roi = extract_face_roi(image, face)
        probs = st.session_state.recognizer.recognize(roi)
        emotions.append(probs)
    return faces, emotions

def save_records(faces, emotions, source):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i, (face, probs) in enumerate(zip(faces, emotions)):
        top = max(probs, key=probs.get)
        analyzer.add_record(ts, source, i + 1, top, probs[top])

def show_stats():
    s = analyzer.get_summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("检测人次", s["total_people"])
    c2.metric("主导表情", s["main_expression"])
    c3.metric("表情种类", sum(1 for v in s["expression_count"].values() if v > 0))
    st.markdown("---")
    data = {e: s["expression_count"][e] for e in EMOTIONS if s["expression_count"][e] > 0}
    if data:
        left, right = st.columns(2)
        with left:
            fig = go.Figure(data=[go.Pie(labels=list(data.keys()), values=list(data.values()), hole=0.5)])
            fig.update_layout(title="情绪占比", height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.bar(x=list(data.keys()), y=list(data.values()), title="各表情计数",
                         labels={"x": "表情", "y": "次数"})
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
    if analyzer.records:
        st.markdown("---")
        st.subheader("检测记录")
        st.dataframe(pd.DataFrame(analyzer.records), use_container_width=True, height=200)

# ── 主界面 ──
st.title("🎭 智能表情识别系统")
st.caption("物联网综合大作业 — 人脸表情识别")

if input_type == "📷 上传图片":
    file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
    if file:
        image = cv2.cvtColor(np.array(Image.open(file).convert("RGB")), cv2.COLOR_RGB2BGR)
        faces, emotions = process_frame(image)
        col_a, col_b = st.columns([2, 1])
        with col_a:
            if mode == "🪞 魔镜模式":
                result = image.copy()
                for face, probs in zip(faces, emotions):
                    top = max(probs, key=probs.get)
                    result = apply_mood_filter(result, top)
                    result = make_emoji_overlay(result, face, top)
            else:
                result = draw_face_boxes(image, faces, emotions)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                     caption=f"检测到 {len(faces)} 张人脸", use_container_width=True)
        with col_b:
            save_records(faces, emotions, file.name)
            show_stats()
    else:
        st.info("👆 请上传一张图片开始识别")

elif input_type == "🎬 上传视频":
    file = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])
    if file:
        tmp = f"_tmp_{file.name}"
        with open(tmp, "wb") as f:
            f.write(file.read())
        cap = cv2.VideoCapture(tmp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.info(f"共 {total} 帧，正在处理...")
        bar = st.progress(0)
        slot = st.empty()
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if n % 5 != 0:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces, emotions = process_frame(rgb)
            if mode == "🪞 魔镜模式":
                res = rgb.copy()
                for face, probs in zip(faces, emotions):
                    res = apply_mood_filter(res, max(probs, key=probs.get))
                    res = make_emoji_overlay(res, face, max(probs, key=probs.get))
            else:
                res = draw_face_boxes(rgb, faces, emotions)
            slot.image(res, use_container_width=True)
            save_records(faces, emotions, file.name)
            bar.progress(min(n / total, 1.0))
        cap.release()
        bar.empty()
        st.success("处理完成!")
    else:
        st.info("👆 请上传一个视频文件")

elif input_type == "🎥 摄像头":
    photo = st.camera_input("拍照")
    if photo:
        image = cv2.cvtColor(np.array(Image.open(photo).convert("RGB")), cv2.COLOR_RGB2BGR)
        faces, emotions = process_frame(image)
        col_a, col_b = st.columns([2, 1])
        with col_a:
            if mode == "🪞 魔镜模式":
                result = image.copy()
                for face, probs in zip(faces, emotions):
                    top = max(probs, key=probs.get)
                    result = apply_mood_filter(result, top)
                    result = make_emoji_overlay(result, face, top)
            else:
                result = draw_face_boxes(image, faces, emotions)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                     caption=f"检测到 {len(faces)} 张人脸", use_container_width=True)
        with col_b:
            save_records(faces, emotions, "camera")
            show_stats()
    else:
        st.info("👆 请点击上方按钮拍照")
