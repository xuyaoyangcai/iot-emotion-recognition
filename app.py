import sys, os
sys.dont_write_bytecode = True
_base = os.path.dirname(os.path.abspath(__file__))
for _m in ['classroom_state', 'face_detector', 'analyzer', 'utils', 'expression_recognizer']:
    _p = os.path.join(_base, '__pycache__', f'{_m}.cpython-311.pyc')
    if os.path.exists(_p):
        os.remove(_p)

import streamlit as st
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
)
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

# ── 辅助函数 ──
def _classify(per_frame):
    """调用 classify_classroom_state，兼容新旧签名"""
    import inspect
    try:
        sig = inspect.signature(classify_classroom_state)
        if len(sig.parameters) >= 3:
            return classify_classroom_state(per_frame.counts, per_frame.total_faces, per_frame.head_up_rate)
        else:
            return classify_classroom_state(per_frame.counts, per_frame.total_faces)
    except Exception:
        return classify_classroom_state(per_frame.counts, per_frame.total_faces)


# ── 处理函数 ──
def process_frame(image):
    faces = face_detector.detect_faces(image)
    emotions = []
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
            probs = st.session_state.recognizer.recognize(roi)
            emotions.append(probs)
            valid_faces.append(face)
            # 头部姿态估计
            pitch, yaw, roll = face_detector.estimate_head_pose(face, image.shape)
            status = face_detector.classify_head_pose(pitch)
            if status == "低头":
                head_down += 1
            elif status == "抬头":
                head_up += 1
            else:
                head_up += 1  # 正常算抬头
        except Exception:
            continue
    return valid_faces, emotions, head_up, head_down


def save_records(faces, emotions, source):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i, (face, probs) in enumerate(zip(faces, emotions)):
        top = max(probs, key=probs.get)
        analyzer.add_record(ts, source, i + 1, top, probs[top])


def render_result(image, faces, emotions):
    if mode == "🪞 魔镜模式":
        result = image.copy()
        for face, probs in zip(faces, emotions):
            top = max(probs, key=probs.get)
            result = apply_mood_filter(result, top)
            result = make_emoji_overlay(result, face, top)
        return result
    else:
        return draw_face_boxes(image, faces, emotions)


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
            fig.add_hrect(
                y0=0, y1=y_max,
                x0=fr_data[wr.window_start - 1].timestamp_seconds,
                x1=fr_data[wr.window_end - 1].timestamp_seconds,
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
        faces, emotions, head_up, head_down = process_frame(image)

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

        result = render_result(image, faces, emotions)
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                 caption=f"检测: {per_frame.total_faces}人 | 抬头率: {per_frame.head_up_rate*100:.0f}% | 课堂状态: {per_frame.classroom_state}",
                 use_container_width=True)
        save_records(faces, emotions, file.name)
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
            faces, emotions, head_up, head_down = process_frame(rgb)

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
            result = render_result(rgb, faces, emotions)
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
    # 扫描可用摄像头（仅首次）
    if st.session_state.cam_list is None:
        cam_list = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                backend = cap.getBackendName()
                if ret:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cam_list.append((i, f"📷 摄像头 {i} — {w}x{h} ({backend})"))
                else:
                    cam_list.append((i, f"📷 摄像头 {i} ({backend}，无画面)"))
                cap.release()
        if not cam_list:
            cam_list = [(0, "📷 摄像头 0 (未检测到)")]
        st.session_state.cam_list = cam_list

    cam_idx_map = {label: idx for idx, label in st.session_state.cam_list}
    cam_choice = st.selectbox("选择摄像头", list(cam_idx_map.keys()),
                              help="DroidCam虚拟摄像头通常显示为摄像头1或2")
    cam_index = cam_idx_map[cam_choice]

    running = st.session_state.cam is not None
    if st.button("⏹ 关闭摄像头" if running else "▶ 开启实时摄像头"):
        if running:
            st.session_state.cam.release()
            st.session_state.cam = None
        else:
            st.session_state.cam = cv2.VideoCapture(cam_index)
            # 重置帧计数
            st.session_state.cam_frame = 0
        st.rerun()

    if st.session_state.cam is not None:
        cam = st.session_state.cam
        ret, frame = cam.read()
        if ret:
            frame_count = st.session_state.get("cam_frame", 0) + 1
            st.session_state.cam_frame = frame_count

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces, emotions, head_up, head_down = process_frame(rgb)

            # 帧级聚合
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            per_frame = aggregate_per_frame(
                emotions, frame_count, frame_count * 0.08,
                ts, "camera",
            )
            _hp = head_up + head_down
            per_frame.head_up_count = head_up
            per_frame.head_down_count = head_down
            per_frame.head_up_rate = round(head_up / _hp, 3) if _hp > 0 else 1.0
            per_frame.classroom_state = _classify(per_frame)
            analyzer.add_frame_record(per_frame)
            tracker.feed(per_frame.classroom_state)

            result = render_result(rgb, faces, emotions)
            level_emoji = {"Green": "🟢", "Yellow": "🟡", "Red": "🔴"}.get(
                tracker.current_level, "⚪")
            st.image(result,
                     caption=f"实时 | 人脸:{len(faces)} | 抬头率:{per_frame.head_up_rate*100:.0f}% | 课堂:{per_frame.classroom_state} | 预警:{level_emoji}",
                     use_container_width=True)
            if faces:
                save_records(faces, emotions, "camera")
            show_stats()

            # 时序图（积累足够帧后显示）
            if len(analyzer.frame_records) >= 3:
                show_time_series_chart()

            time.sleep(0.08)
            st.rerun()
        else:
            st.error("无法读取摄像头，请检查权限")
            st.session_state.cam.release()
            st.session_state.cam = None
    else:
        st.info("👆 点击「开启实时摄像头」开始实时课堂状态检测")
