import sys, os
sys.dont_write_bytecode = True
_base = os.path.dirname(os.path.abspath(__file__))
for _m in ['classroom_state', 'face_detector', 'analyzer', 'utils', 'expression_recognizer']:
    _p = os.path.join(_base, '__pycache__', f'{_m}.cpython-311.pyc')
    if os.path.exists(_p):
        os.remove(_p)

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import queue
import threading
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import StringIO
from PIL import Image
import time

import face_detector
from expression_recognizer import ExpressionRecognizer, EMOTIONS
from analyzer import ResultAnalyzer
from utils import (
    draw_face_boxes, apply_mood_filter,
    make_emoji_overlay, EMOTION_COLORS, CLASSROOM_STATE_COLORS,
    COMPOSITE_EMOTION_COLORS, COMPOSITE_EMOTION_EMOJI,
)
from gaze_emotion import classify_classroom_emotion, top_classroom_emotion, CLASSROOM_EMOTIONS
from classroom_state import (
    aggregate_per_frame, compute_sliding_window, WarningTracker,
    classify_classroom_state,
)

st.set_page_config(page_title="课堂状态分析系统", page_icon="🏫", layout="wide")

# ── 初始化 ──
if "analyzer" not in st.session_state:
    st.session_state.analyzer = ResultAnalyzer()
if "recognizer" not in st.session_state:
    with st.spinner("加载模型中..."):
        st.session_state.recognizer = ExpressionRecognizer()
if "cam" not in st.session_state:
    st.session_state.cam = None
if "cam_list" not in st.session_state:
    st.session_state.cam_list = None
if "warning_tracker" not in st.session_state:
    st.session_state.warning_tracker = WarningTracker()

analyzer = st.session_state.analyzer
tracker = st.session_state.warning_tracker

# ── 侧边栏 ──
with st.sidebar:
    st.title("🏫 课堂状态分析")
    mode = st.radio("显示模式", ["📊 仪表盘模式", "🪞 魔镜模式"])
    st.markdown("---")
    input_type = st.radio("输入来源", ["📷 上传图片", "🎬 上传视频", "🎥 实时摄像头"])
    threshold = st.slider("检测阈值", 0.3, 0.9, 0.4, 0.05)
    st.markdown("---")

    # 课堂状态实时显示
    state = analyzer.get_latest_classroom_state()
    if state != "N/A":
        color = CLASSROOM_STATE_COLORS.get(state, "#888")
        st.markdown(f"**当前课堂状态:** :{'green' if '良好' in state else 'orange' if '平稳' in state else 'red' if '低落' in state or '波动' in state else 'gray'}[{state}]")

    warning_level = tracker.current_level
    if warning_level != "Normal":
        level_emoji = {"Green": "🟢", "Yellow": "🟡", "Red": "🔴"}.get(warning_level, "⚪")
        st.markdown(f"**预警等级: {level_emoji} {warning_level}**")

    st.markdown("---")
    if st.button("🔄 重置统计"):
        analyzer.clear()
        st.session_state.warning_tracker = WarningTracker()
        st.rerun()

    # CSV 导出
    if analyzer.records:
        buf = StringIO()
        analyzer.export_csv(buf)
        st.download_button("📥 导出课堂状态CSV", buf.getvalue(),
                           f"classroom_{datetime.now():%Y%m%d_%H%M%S}.csv",
                           mime="text/csv")

    # 时序 CSV 导出（有帧记录时显示）
    if analyzer.frame_records:
        ts_buf = StringIO()
        window_results = compute_sliding_window(analyzer.frame_records) \
            if len(analyzer.frame_records) >= 5 else None
        analyzer.export_time_series_csv(ts_buf, window_results)
        st.download_button("📈 导出时序分析CSV", ts_buf.getvalue(),
                           f"timeline_{datetime.now():%Y%m%d_%H%M%S}.csv",
                           mime="text/csv")

def _classify(per_frame):
    """课堂状态分类，传入抬头率"""
    return classify_classroom_state(per_frame.counts, per_frame.total_faces, per_frame.head_up_rate)


# ── 处理函数 ──
def process_frame(image):
    faces = face_detector.detect_faces(image)
    emotions = []
    composite_emotions = []
    valid_faces = []
    head_up = 0
    head_down = 0
    for face in faces:
        if face.confidence < threshold:
            continue
        roi = face_detector.extract_face_roi(image, face)
        if roi.size == 0:
            continue
        try:
            # 头姿态估计（可能因关键点质量失败，不影响表情识别）
            try:
                pitch, yaw, _ = face_detector.estimate_head_pose(face, image.shape)
                far = face_detector.face_aspect_ratio(face)
                status = face_detector.classify_head_pose(pitch, face_ar=far)
            except Exception:
                pitch, yaw = 0.0, 0.0
                status = "正常"
            probs = st.session_state.recognizer.recognize(roi, head_status=status)
            emotions.append(probs)

            # 课堂复合情绪（表情 + 头姿态）
            comp = classify_classroom_emotion(probs, yaw, pitch)
            comp_top = max(comp, key=comp.get) if comp else "N/A"
            composite_emotions.append({"scores": comp, "top": comp_top, "yaw": yaw, "pitch": pitch})

            valid_faces.append(face)
            if status == "低头":
                head_down += 1
            else:
                head_up += 1  # 抬头 + 正常 = 在听讲
        except Exception:
            continue
    return valid_faces, emotions, head_up, head_down, composite_emotions


def save_records(faces, emotions, source):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i, (face, probs) in enumerate(zip(faces, emotions)):
        top = max(probs, key=probs.get)
        analyzer.add_record(ts, source, i + 1, top, probs[top])


def render_result(image, faces, emotions, composite_emotions=None):
    if mode == "🪞 魔镜模式":
        result = image.copy()
        for face, probs in zip(faces, emotions):
            top = max(probs, key=probs.get)
            result = apply_mood_filter(result, top)
            result = make_emoji_overlay(result, face, top)
        return result
    else:
        # 仪表盘模式 — 画基本表情框 + 复合情绪标签
        img = image.copy()
        comp_list = composite_emotions or []
        for i, (face, probs) in enumerate(zip(faces, emotions)):
            x1, y1, x2, y2 = face.bbox
            top_emo = max(probs, key=probs.get)
            color = EMOTION_COLORS.get(top_emo, (0, 255, 0))

            # 画人脸框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 第一行：基础表情
            label1 = f"{top_emo} ({face.confidence:.0%})"
            (tw, th), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label1, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # 第二行：复合情绪（上方，留足间距不遮挡基础表情）
            if i < len(comp_list) and comp_list[i]["top"] != "N/A":
                comp_top = comp_list[i]["top"]
                comp_color = COMPOSITE_EMOTION_COLORS.get(comp_top, (0, 255, 0))
                label2 = f"{comp_top}"
                (tw2, th2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_DUPLEX, 0.55, 2)
                gap = th + 14
                cv2.rectangle(img, (x1, y1 - gap - th2 - 6), (x1 + tw2 + 6, y1 - gap),
                            comp_color, -1)
                cv2.putText(img, label2, (x1 + 3, y1 - gap - 2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 2)
        return img


def show_stats():
    s = analyzer.get_summary()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("检测人次", s["total_people"])
    c2.metric("主导表情", s["main_expression"])
    c3.metric("表情种类", sum(1 for v in s["expression_count"].values() if v > 0))
    state = analyzer.get_latest_classroom_state()
    c4.metric("课堂状态", state)
    # 最新帧抬头率
    if analyzer.frame_records:
        hr = analyzer.frame_records[-1].head_up_rate
        c5.metric("抬头率", f"{hr*100:.0f}%")
    else:
        c5.metric("抬头率", "N/A")

    st.markdown("---")
    data = {e: s["expression_count"][e] for e in EMOTIONS if s["expression_count"][e] > 0}
    if data:
        left, right = st.columns(2)
        with left:
            fig = go.Figure(data=[go.Pie(labels=list(data.keys()), values=list(data.values()), hole=0.5)])
            fig.update_layout(title="情绪占比", height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.bar(x=list(data.keys()), y=list(data.values()), title="各表情计数",
                         labels={"x": "表情", "y": "次数"})
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
    if analyzer.records:
        st.markdown("---")
        st.subheader("检测记录")
        st.dataframe(pd.DataFrame(analyzer.records), use_container_width=True, height=200)


def show_time_series_chart():
    """显示时序折线图 + 预警标记"""
    fr_data = analyzer.get_per_frame_data()
    if len(fr_data) < 2:
        return

    window_results = compute_sliding_window(fr_data) if len(fr_data) >= 5 else []

    fig = go.Figure()

    plot_emotions = ["Happy", "Neutral", "Sad", "Angry", "Surprise"]
    colors = {"Happy": "#FFD700", "Neutral": "#AAAAAA", "Sad": "#4488CC",
              "Angry": "#DD4444", "Surprise": "#FF8800"}

    for emo in plot_emotions:
        y_vals = [fr.ratios.get(emo, 0) * 100 for fr in fr_data]
        x_vals = [fr.timestamp_seconds for fr in fr_data]
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='lines+markers',
            name=emo, line=dict(color=colors.get(emo), width=2),
            marker=dict(size=4),
        ))

    # 抬头率曲线（虚线）
    hr_vals = [fr.head_up_rate * 100 for fr in fr_data]
    fig.add_trace(go.Scatter(
        x=x_vals, y=hr_vals, mode='lines',
        name="抬头率", line=dict(color="#00BCD4", width=2, dash="dash"),
        yaxis="y",
    ))

    # 预警区域标记
    if window_results:
        wr = window_results[-1]
        if wr.warning_level != "Normal":
            y_max = 100
            color_map = {"Green": "rgba(0,200,0,0.1)", "Yellow": "rgba(255,200,0,0.15)",
                         "Red": "rgba(255,0,0,0.15)"}
            # window_start/end 是 frame_number，需要映射到 fr_data 索引
            idx_map = {fr.frame_number: i for i, fr in enumerate(fr_data)}
            x0_idx = idx_map.get(wr.window_start)
            x1_idx = idx_map.get(wr.window_end)
            if x0_idx is not None and x1_idx is not None and x0_idx < len(fr_data) and x1_idx < len(fr_data):
                fig.add_hrect(
                    y0=0, y1=y_max,
                    x0=fr_data[x0_idx].timestamp_seconds,
                    x1=fr_data[x1_idx].timestamp_seconds,
                    fillcolor=color_map.get(wr.warning_level, "rgba(128,128,128,0.1)"),
                    layer="below", line_width=0,
                    annotation_text=wr.warning_level,
                    annotation_position="top right",
                )

    fig.update_layout(
        title="📈 课堂情绪时间序列",
        xaxis_title="时间 (秒)",
        yaxis_title="占比 (%)",
        yaxis_range=[0, 100],
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # 预警日志表格
    if window_results:
        st.markdown("---")
        st.subheader("🚨 预警日志")
        warn_log = []
        for wr in window_results:
            if wr.warning_level != "Normal":
                warn_log.append({
                    "中心帧": wr.center_frame,
                    "窗口": f"{wr.window_start}-{wr.window_end}",
                    "Happy均值": f"{wr.window_mean.get('Happy', 0)*100:.0f}%",
                    "Neutral均值": f"{wr.window_mean.get('Neutral', 0)*100:.0f}%",
                    "Sad均值": f"{wr.window_mean.get('Sad', 0)*100:.0f}%",
                    "预警": wr.warning_level,
                })
        if warn_log:
            st.dataframe(pd.DataFrame(warn_log), use_container_width=True, height=150)


# ── 主界面 ──
st.title("🏫 基于表情识别的课堂状态分析系统")
st.caption("多人人脸检测与表情识别的课堂状态分析")

if input_type == "📷 上传图片":
    file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
    if file:
        image = cv2.cvtColor(np.array(Image.open(file).convert("RGB")), cv2.COLOR_RGB2BGR)
        faces, emotions, head_up, head_down, composite_emotions = process_frame(image)

        # 帧级聚合
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        per_frame = aggregate_per_frame(emotions, 1, 0, ts, file.name)
        _hp = head_up + head_down
        per_frame.head_up_count = head_up
        per_frame.head_down_count = head_down
        per_frame.head_up_rate = round(head_up / _hp, 3) if _hp > 0 else 1.0
        per_frame.classroom_state = _classify(per_frame)
        analyzer.add_frame_record(per_frame)
        tracker.feed(per_frame.classroom_state)

        result = render_result(image, faces, emotions, composite_emotions)
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                 caption=f"检测: {per_frame.total_faces}人 | 抬头率: {per_frame.head_up_rate*100:.0f}% | 课堂状态: {per_frame.classroom_state}",
                 use_container_width=True)
        save_records(faces, emotions, file.name)

        # 显示复合情绪分布
        if composite_emotions:
            comp_counts = {}
            for ce in composite_emotions:
                t = ce["top"]
                comp_counts[t] = comp_counts.get(t, 0) + 1
            comp_str = " | ".join(f"{COMPOSITE_EMOTION_EMOJI.get(k,'')} {k}:{v}" for k, v in sorted(comp_counts.items(), key=lambda x: -x[1]))
            st.caption(f"课堂情绪: {comp_str}")
        show_stats()
    else:
        st.info("👆 请上传一张包含多人的课堂图片")

elif input_type == "🎬 上传视频":
    file = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])
    if file:
        tmp = f"_tmp_{file.name}"
        with open(tmp, "wb") as f:
            f.write(file.read())
        cap = cv2.VideoCapture(tmp)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # 1fps 采样：计算帧间隔
        frame_interval = max(1, int(fps))
        sampled = total_frames // frame_interval

        st.info(f"视频: {duration:.0f}秒, {fps:.0f}fps, 采样 ~{sampled} 帧 (每 {frame_interval} 帧采1帧)")

        bar = st.progress(0)
        slot_img = st.empty()
        slot_chart = st.empty()
        slot_status = st.empty()

        n = 0       # 已读总帧数
        processed = 0  # 已处理帧数
        frame_data = []  # 积累的 PerFrameResult

        # 清空之前的帧记录和 tracker
        analyzer.frame_records.clear()
        st.session_state.warning_tracker = WarningTracker()
        tracker_local = st.session_state.warning_tracker

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if (n - 1) % frame_interval != 0:
                continue

            processed += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces, emotions, head_up, head_down, composite_emotions = process_frame(rgb)

            elapsed = processed * (frame_interval / fps)

            per_frame = aggregate_per_frame(
                emotions, processed, elapsed,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                file.name,
            )
            _hp = head_up + head_down
            per_frame.head_up_count = head_up
            per_frame.head_down_count = head_down
            per_frame.head_up_rate = round(head_up / _hp, 3) if _hp > 0 else 1.0
            per_frame.classroom_state = _classify(per_frame)
            analyzer.add_frame_record(per_frame)
            frame_data.append(per_frame)
            save_records(faces, emotions, file.name)

            # 预警追踪
            warning_level = tracker_local.feed(per_frame.classroom_state)
            level_emoji = {"Green": "🟢", "Yellow": "🟡", "Red": "🔴"}.get(warning_level, "⚪")

            # 显示当前帧
            result = render_result(rgb, faces, emotions, composite_emotions)
            slot_img.image(result, use_container_width=True)
            slot_status.markdown(
                f"**帧 {processed}/{sampled}** | 检测: {per_frame.total_faces}人 | "
                f"抬头率: {per_frame.head_up_rate*100:.0f}% | "
                f"课堂状态: {per_frame.classroom_state} | "
                f"预警: {level_emoji} {warning_level} "
                f"({tracker_local.good_streak}G/{tracker_local.stable_streak}S/"
                f"{tracker_local.low_streak}L/{tracker_local.volatile_streak}V)"
            )

            # 实时更新时序图
            if len(frame_data) >= 2:
                with slot_chart.container():
                    show_time_series_chart()

            bar.progress(min(n / total_frames, 1.0))

        cap.release()
        bar.empty()
        st.success(f"处理完成! 共 {processed} 帧")

        # 最终时序图
        st.markdown("---")
        show_time_series_chart()

        # 趋势分析结论
        st.markdown("---")
        st.subheader("📋 趋势分析结论")
        if analyzer.frame_records:
            states = [fr.classroom_state for fr in analyzer.frame_records]
            good_pct = sum(1 for s in states if "良好" in s) / len(states) * 100
            low_pct = sum(1 for s in states if "低落" in s) / len(states) * 100
            volatile_pct = sum(1 for s in states if "波动" in s) / len(states) * 100
            stable_pct = sum(1 for s in states if "平稳" in s) / len(states) * 100

            avg_head_up = sum(fr.head_up_rate for fr in analyzer.frame_records) / len(analyzer.frame_records) * 100
            st.write(f"- 平均抬头率: **{avg_head_up:.1f}%**")
            st.write(f"- 状态良好占比: **{good_pct:.1f}%**")
            st.write(f"- 状态平稳占比: **{stable_pct:.1f}%**")
            st.write(f"- 需关注占比: **{low_pct:.1f}%**")
            st.write(f"- 注意力波动占比: **{volatile_pct:.1f}%**")

            warning_levels = [w for w in
                [compute_sliding_window(analyzer.frame_records)]
                if w]
            if warning_levels:
                red_count = sum(1 for wr in warning_levels[0] if wr.warning_level == "Red")
                yellow_count = sum(1 for wr in warning_levels[0] if wr.warning_level == "Yellow")
                if red_count > 0:
                    st.error(f"⚠️ 检测到 {red_count} 次红色预警（低落/波动），建议关注该时段")
                elif yellow_count > 0:
                    st.warning(f"⚡ 检测到 {yellow_count} 次黄色预警（中性占比过高），课堂互动可能需要加强")
                else:
                    st.success("✅ 课堂整体状态正常，未触发预警")
    else:
        st.info("👆 请上传一段课堂视频（≥1分钟），系统将进行时序分析")

elif input_type == "🎥 实时摄像头":
    analysis_sec = st.slider("⏱ 分析间隔(秒)", 1, 5, 2)

    if "cam_result_queue" not in st.session_state:
        st.session_state.cam_result_queue = queue.Queue()

    # 捕获参数到闭包（避免 WebRTC 回调线程访问 st.session_state 不稳定）
    _cam_recognizer = st.session_state.recognizer
    _cam_threshold = threshold
    _cam_analysis_sec = analysis_sec
    _cam_result_queue = st.session_state.cam_result_queue

    class CamProcessor(VideoProcessorBase):
        def __init__(self):
            self._lock = threading.Lock()
            self._last_emotions = []
            self._last_caption = ""
            self._last_analysis_ts = 0.0
            self._analysis_busy = False
            self._stop_worker = False
            self._recv_count = 0
            self._face_count = 0
            self._analysis_count = 0
            self._worker = threading.Thread(target=self._analysis_worker, daemon=True)
            self._worker.start()

        def _analysis_worker(self):
            """后台线程：定期对最新帧执行表情识别+头部姿态（不阻塞视频流）"""
            while not self._stop_worker:
                time.sleep(0.1)
                now = time.time()
                if now - self._last_analysis_ts < _cam_analysis_sec:
                    continue
                if self._analysis_busy:
                    continue

                with self._lock:
                    snap = getattr(self, "_snap_img", None)
                    snap_faces = getattr(self, "_snap_faces", None) or []
                if snap is None or not snap_faces:
                    continue

                self._analysis_busy = True
                self._last_analysis_ts = now

                try:
                    emotions = []
                    composite_emotions = []
                    head_up, head_down = 0, 0
                    for face in snap_faces:
                        roi = face_detector.extract_face_roi(snap, face)
                        if roi.size == 0:
                            continue
                        try:
                            # 头姿态（可能失败，不影响表情识别）
                            try:
                                pitch, yaw, _ = face_detector.estimate_head_pose(face, snap.shape)
                                far = face_detector.face_aspect_ratio(face)
                                status = face_detector.classify_head_pose(pitch, face_ar=far)
                            except Exception:
                                pitch, yaw = 0.0, 0.0
                                status = "正常"
                            probs = _cam_recognizer.recognize(roi, head_status=status)
                            emotions.append(probs)

                            # 课堂复合情绪
                            comp = classify_classroom_emotion(probs, yaw, pitch)
                            comp_top = max(comp, key=comp.get) if comp else "N/A"
                            composite_emotions.append({"scores": comp, "top": comp_top, "yaw": yaw, "pitch": pitch})

                            if status == "低头":
                                head_down += 1
                            else:
                                head_up += 1  # 抬头 + 正常 = 在听讲
                        except Exception:
                            continue

                    if emotions:
                        with self._lock:
                            self._last_emotions = emotions
                            self._last_composite = composite_emotions
                            self._analysis_count += 1
                            hr = round(head_up / (head_up + head_down), 3) \
                                if (head_up + head_down) > 0 else 1.0
                            # 显示复合情绪占比
                            comp_counts = {}
                            for ce in composite_emotions:
                                comp_counts[ce["top"]] = comp_counts.get(ce["top"], 0) + 1
                            comp_summary = " ".join(f"{COMPOSITE_EMOTION_EMOJI.get(k,'')}{v}" for k, v in sorted(comp_counts.items(), key=lambda x: -x[1])[:2])
                            self._last_caption = (
                                f"实时 | 人脸:{len(emotions)} | 抬头率:{hr*100:.0f}% | {comp_summary}"
                                f" | 分析#{self._analysis_count}"
                            )
                        _cam_result_queue.put({
                            "emotions": emotions,
                            "composite_emotions": composite_emotions,
                            "head_up": head_up,
                            "head_down": head_down,
                            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "faces": list(snap_faces),
                        })
                except Exception:
                    pass
                finally:
                    self._analysis_busy = False

        def recv(self, frame):
            try:
                img = frame.to_ndarray(format="bgr24")
                h, w = img.shape[:2]

                faces = face_detector.detect_faces(img)
                faces = [f for f in faces if f.confidence >= _cam_threshold]

                with self._lock:
                    self._recv_count += 1
                    self._face_count = len(faces)
                    self._snap_img = img.copy()
                    self._snap_faces = list(faces)
                    emo_disp = list(self._last_emotions)
                    comp_disp = list(getattr(self, "_last_composite", []) or [])
                    caption = self._last_caption

                # 始终画框（即使还没分析结果，也显示复合情绪标签）
                if faces:
                    if emo_disp:
                        disp = emo_disp[:len(faces)]
                        comp = comp_disp[:len(faces)]
                    else:
                        disp = []
                        comp = []
                    while len(disp) < len(faces):
                        disp.append({"Neutral": 1.0, "Happy": 0.0, "Sad": 0.0,
                                    "Angry": 0.0, "Surprise": 0.0, "Fear": 0.0, "Disgust": 0.0})
                    # 画框 + 复合情绪标签
                    for i, face in enumerate(faces):
                        x1, y1, x2, y2 = face.bbox
                        probs = disp[i] if i < len(disp) else {"Neutral": 1.0}
                        top_emo = max(probs, key=probs.get) if probs else "Neutral"
                        color = EMOTION_COLORS.get(top_emo, (0, 255, 0))

                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        label1 = f"{top_emo} ({face.confidence:.0%})"
                        (tw, th), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                        cv2.putText(img, label1, (x1 + 2, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                        # 第二行：复合情绪
                        if i < len(comp) and comp[i].get("top") != "N/A":
                            comp_top = comp[i]["top"]
                            comp_color = COMPOSITE_EMOTION_COLORS.get(comp_top, (0, 255, 0))
                            label2 = f"{comp_top}"
                            (tw2, th2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_DUPLEX, 0.55, 2)
                            gap = th + 14
                            cv2.rectangle(img, (x1, y1 - gap - th2 - 6), (x1 + tw2 + 6, y1 - gap),
                                        comp_color, -1)
                            cv2.putText(img, label2, (x1 + 3, y1 - gap - 2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # 左下角调试信息 + 顶部状态栏
                dbg = f"F#{self._recv_count} | Faces:{len(faces)} | A#{self._analysis_count}"
                cv2.putText(img, dbg, (10, h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                if caption:
                    cv2.putText(img, caption, (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                return av.VideoFrame.from_ndarray(img, format="bgr24")
            except Exception:
                return frame

    webrtc_streamer(
        key="cam-webrtc",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=CamProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # 统计面板放在 fragment 中，每2秒自动刷新，不影响视频流
    @st.fragment(run_every=2)
    def _cam_stats_panel():
        result_queue = st.session_state.cam_result_queue
        processed = 0
        while not result_queue.empty() and processed < 10:
            try:
                r = result_queue.get_nowait()
                per_frame = aggregate_per_frame(r["emotions"], len(analyzer.frame_records) + 1,
                                                time.time(), r["ts"], "camera")
                _hp = r["head_up"] + r["head_down"]
                per_frame.head_up_count = r["head_up"]
                per_frame.head_down_count = r["head_down"]
                per_frame.head_up_rate = round(r["head_up"] / _hp, 3) if _hp > 0 else 1.0
                per_frame.classroom_state = _classify(per_frame)
                analyzer.add_frame_record(per_frame)
                tracker.feed(per_frame.classroom_state)
                save_records(r["faces"], r["emotions"], "camera")
                processed += 1
            except queue.Empty:
                break
        if processed > 0 or analyzer.frame_records:
            show_stats()
            if len(analyzer.frame_records) >= 3:
                show_time_series_chart()

    _cam_stats_panel()
