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
    draw_face_boxes, EMOTION_COLORS, CLASSROOM_STATE_COLORS,
    COMPOSITE_EMOTION_COLORS, COMPOSITE_EMOTION_EMOJI,
)
from gaze_emotion import classify_classroom_emotion, top_classroom_emotion, CLASSROOM_EMOTIONS
from classroom_state import (
    aggregate_per_frame, compute_sliding_window, WarningTracker,
    classify_classroom_state,
)

st.set_page_config(page_title="课堂状态分析系统", page_icon="🏫", layout="wide")

# ── 自定义CSS ──
st.markdown("""
<style>
    /* 指标卡片 */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        border-radius: 12px;
        padding: 12px 16px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 3px 8px rgba(0,0,0,0.10);
        border-color: #c0c0c0;
    }
    [data-testid="stMetric"] label {
        font-size: 0.78rem;
        color: #666;
        font-weight: 500;
    }
    [data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
    }
    /* 侧边栏 */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
    }
    section[data-testid="stSidebar"] .stDownloadButton > button {
        width: 100%;
        margin-bottom: 4px;
    }
    /* 主标题 */
    h1 { font-weight: 700 !important; letter-spacing: -0.5px; }
    /* 空状态引导 */
    .empty-guide {
        text-align: center;
        padding: 60px 20px;
        color: #999;
        background: #fafafa;
        border-radius: 16px;
        border: 2px dashed #e0e0e0;
    }
    .empty-guide h2 { color: #666; margin-bottom: 8px; }
    /* 分区分隔 */
    .section-divider { margin: 1.5rem 0; border-top: 1px solid #eee; }
</style>
""", unsafe_allow_html=True)

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
    input_type = st.radio("输入来源", ["📷 上传图片", "🎬 上传视频", "🎥 实时摄像头"], label_visibility="collapsed")
    st.markdown("---")
    threshold = st.slider("检测阈值", 0.3, 0.9, 0.4, 0.05)
    st.markdown("---")
    show_intro = st.checkbox("📖 显示系统简介", value=False)

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
    img = image.copy()
    comp_list = composite_emotions or []
    for i, (face, probs) in enumerate(zip(faces, emotions)):
        x1, y1, x2, y2 = face.bbox
        top_emo = max(probs, key=probs.get)
        color = EMOTION_COLORS.get(top_emo, (0, 255, 0))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label1 = f"{top_emo} ({face.confidence:.0%})"
        (tw, th), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label1, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

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


def show_status_bar():
    """课堂状态 + 预警 + 重置 + 导出（在主区域顶部）"""
    state = analyzer.get_latest_classroom_state()
    warning_level = tracker.current_level

    sc1, sc2, sc3, sc4, sc5 = st.columns([2, 1.5, 1, 1, 1])
    with sc1:
        if state != "N/A":
            color = CLASSROOM_STATE_COLORS.get(state, "#888")
            st.markdown(f"**课堂状态:** :{'green' if '良好' in state else 'orange' if '平稳' in state else 'red' if '低落' in state or '波动' in state else 'gray'}[{state}]")
        else:
            st.caption("课堂状态: 等待数据")
    with sc2:
        if warning_level != "Normal":
            level_emoji = {"Green": "🟢", "Yellow": "🟡", "Red": "🔴"}.get(warning_level, "⚪")
            st.markdown(f"**预警: {level_emoji} {warning_level}**")
        else:
            st.caption("预警: 正常")
    with sc3:
        if st.button("🔄 重置", use_container_width=True, key="reset_btn_main"):
            analyzer.clear()
            st.session_state.warning_tracker = WarningTracker()
            st.rerun()
    with sc4:
        if analyzer.records:
            buf = StringIO()
            analyzer.export_csv(buf)
            st.download_button("📊 课堂CSV", buf.getvalue(),
                               f"classroom_{datetime.now():%Y%m%d_%H%M%S}.csv",
                               mime="text/csv", use_container_width=True, key="csv_btn_main")
    with sc5:
        if analyzer.frame_records:
            ts_buf = StringIO()
            window_results = compute_sliding_window(analyzer.frame_records) \
                if len(analyzer.frame_records) >= 5 else None
            analyzer.export_time_series_csv(ts_buf, window_results)
            st.download_button("📈 时序CSV", ts_buf.getvalue(),
                               f"timeline_{datetime.now():%Y%m%d_%H%M%S}.csv",
                               mime="text/csv", use_container_width=True, key="ts_csv_btn_main")


def show_stats():
    s = analyzer.get_summary()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("检测人次", s["total_people"])
    c2.metric("主导表情", s["main_expression"])
    c3.metric("表情种类", sum(1 for v in s["expression_count"].values() if v > 0))
    state = analyzer.get_latest_classroom_state()
    c4.metric("课堂状态", state)
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
            pie_colors = [
                "#FFD700" if e == "Happy" else "#AAAAAA" if e == "Neutral" else
                "#4488CC" if e == "Sad" else "#DD4444" if e == "Angry" else
                "#FF8800" if e == "Surprise" else "#8844AA" if e == "Fear" else
                "#44AA55" if e == "Disgust" else "#44CCCC"
                for e in data.keys()
            ]
            fig = go.Figure(data=[go.Pie(labels=list(data.keys()), values=list(data.values()),
                                         hole=0.5, marker=dict(colors=pie_colors))])
            fig.update_layout(title="情绪占比", height=320, margin=dict(l=10, r=10, t=40, b=10),
                            template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            bar_colors = [
                "#FFD700" if e == "Happy" else "#AAAAAA" if e == "Neutral" else
                "#4488CC" if e == "Sad" else "#DD4444" if e == "Angry" else
                "#FF8800" if e == "Surprise" else "#8844AA" if e == "Fear" else
                "#44AA55" if e == "Disgust" else "#44CCCC"
                for e in data.keys()
            ]
            fig = px.bar(x=list(data.keys()), y=list(data.values()), title="各表情计数",
                         labels={"x": "表情", "y": "次数"})
            fig.update_traces(marker_color=bar_colors)
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10),
                            template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
    if analyzer.records:
        st.markdown("---")
        st.subheader("检测记录")
        try:
            st.dataframe(pd.DataFrame(analyzer.records), use_container_width=True, height=200)
        except ValueError:
            st.dataframe(pd.DataFrame.from_records(analyzer.records), use_container_width=True, height=200)


def show_time_series_chart(key_suffix: str = ""):
    """显示时序折线图 + 预警标记"""
    fr_data = analyzer.get_per_frame_data()
    if len(fr_data) < 2:
        return

    window_results = compute_sliding_window(fr_data) if len(fr_data) >= 5 else []

    fig = go.Figure()

    plot_emotions = ["Happy", "Neutral", "Sad", "Angry", "Surprise", "Fear", "Disgust", "Contempt"]
    colors = {"Happy": "#FFD700", "Neutral": "#AAAAAA", "Sad": "#4488CC",
              "Angry": "#DD4444", "Surprise": "#FF8800", "Fear": "#8844AA",
              "Disgust": "#44AA55", "Contempt": "#44CCCC"}

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
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True,
                    key=f"ts_chart_{key_suffix}" if key_suffix else None)

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

if show_intro:
    tab1, tab2, tab3, tab4 = st.tabs(["基础表情", "学生状态", "课堂状态判断", "预警规则"])

    with tab1:
        st.subheader("系统可识别的 8 种基本表情")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown("**😊 Happy 开心**\n\n愉悦、满意")
        c2.markdown("**😐 Neutral 平静**\n\n无表情、正常")
        c3.markdown("**😢 Sad 悲伤**\n\n低落、难过")
        c4.markdown("**😡 Angry 生气**\n\n不满、愤怒")
        c1.markdown("**😲 Surprise 惊讶**\n\n意外、吃惊")
        c2.markdown("**😨 Fear 害怕**\n\n紧张、恐惧")
        c3.markdown("**🤢 Disgust 厌恶**\n\n反感、嫌弃")
        c4.markdown("**😏 Contempt 轻蔑**\n\n不屑、无聊")
        st.info("系统通过摄像头画面分析每张人脸，输出以上 8 种表情的概率分布")

    with tab2:
        st.subheader("综合基础表情和头部姿态推导出 8 种学生个人状态")
        data = [
            ["🎯 Focused 专注",    "Happy / Neutral",   "面朝前方",   "正在认真听讲"],
            ["🙋 Engaged 投入",    "Happy",             "面朝前方",   "积极互动、理解内容"],
            ["💭 Thinking 思考",   "Neutral / Sad",     "低头",      "做笔记、做题、思考"],
            ["😴 Tired 困倦",      "Sad",               "低头",      "打瞌睡、精神状态差"],
            ["🥱 Bored 走神",      "Neutral / Contempt", "低头",      "玩手机、发呆、不想听"],
            ["👀 Distracted 分心", "Neutral / Sad",      "大幅偏头",  "看窗外、和同学说话"],
            ["🤔 Confused 困惑",   "Surprise / Fear",    "头部微偏",  "听不懂、有疑问"],
            ["😰 Anxious 焦虑",    "Fear / Sad",         "频繁转头",  "考试/提问时紧张"],
        ]
        st.dataframe(
            data, use_container_width=True, hide_index=True,
            column_config={
                "0": st.column_config.TextColumn("状态"),
                "1": st.column_config.TextColumn("伴随表情"),
                "2": st.column_config.TextColumn("头部动作"),
                "3": st.column_config.TextColumn("课堂含义"),
            }
        )
        st.info("关键区分：同样是低头，表情平静 = 在思考做题，表情悲伤 = 困了打瞌睡，表情轻蔑 = 无聊走神")

    with tab3:
        st.subheader("统计全班学生状态，按优先级判断整体课堂氛围")
        rules = [
            ["① 大面积低头", "抬头率 < 50%", "需要关注 ⚠️"],
            ["② 积极健康", "(Happy + Neutral) / 总数 ≥ 70% 且 抬头率 ≥ 60%", "状态良好 ✅"],
            ["③ 负面情绪多", "(Sad + Angry + Contempt) / 总数 ≥ 40%", "需要关注 ⚠️"],
            ["④ Contempt 突出", "Contempt 是唯一人数最多的表情", "注意力波动 📈"],
            ["⑤ Surprise 突出", "Surprise 是唯一人数最多的表情", "注意力波动 📈"],
            ["⑥ 平静为主", "Neutral 是人数最多的表情", "状态平稳 ➡️"],
            ["⑦ 其他情况", "—", "状态平稳 ➡️"],
        ]
        st.dataframe(
            rules, use_container_width=True, hide_index=True,
            column_config={
                "0": st.column_config.TextColumn("判断条件"),
                "1": st.column_config.TextColumn("具体规则"),
                "2": st.column_config.TextColumn("课堂状态"),
            }
        )

    with tab4:
        st.subheader("不只判断单帧，还跟踪时间趋势变化")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📊 滑动窗口预警**\n（5帧为一个窗口，逐帧滑动）")
            trends = [
                ["🟢 Green", "(Happy + Neutral) > 60%\n且 抬头率 > 70%"],
                ["🟡 Yellow", "Neutral > 60%\n或 Neutral > 55% 且 抬头率 < 70%"],
                ["🔴 Red", "抬头率 < 40%\n或 (Sad+Angry+Fear+Contempt) > 40%"],
            ]
            st.dataframe(
                trends, use_container_width=True, hide_index=True,
                column_config={
                    "0": st.column_config.TextColumn("等级"),
                    "1": st.column_config.TextColumn("触发条件"),
                }
            )
        with c2:
            st.markdown("**⏱ 连续帧追踪预警**\n（累积连续同类帧数）")
            tracks = [
                ["🟢 Green", "连续 ≥ 4 帧状态良好"],
                ["🟡 Yellow", "连续 ≥ 3 帧状态平稳"],
                ["🔴 Red", "连续 ≥ 2 帧低落或波动"],
            ]
            st.dataframe(
                tracks, use_container_width=True, hide_index=True,
                column_config={
                    "0": st.column_config.TextColumn("等级"),
                    "1": st.column_config.TextColumn("触发条件"),
                }
            )
        st.info("滑动窗口检测短期突变，连续追踪检测长期趋势。任一触发 Red 即报警。")

    st.markdown("---")

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
        show_status_bar()
        show_stats()
    else:
        st.markdown(
            '<div class="empty-guide"><h2>📷 请上传课堂图片</h2>'
            '<p>支持 JPG/PNG 格式，包含多人的课堂照片效果最佳</p></div>',
            unsafe_allow_html=True)

elif input_type == "🎬 上传视频":
    file = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])
    if file:
        # 判断是否新文件
        is_new = ("vid_state" not in st.session_state
                  or st.session_state.vid_state.get("file_name") != file.name)
        if is_new:
            tmp = f"_tmp_{file.name}"
            with open(tmp, "wb") as f:
                f.write(file.read())
            cap = cv2.VideoCapture(tmp)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            analyzer.clear()
            st.session_state.warning_tracker = WarningTracker()
            st.session_state.vid_state = {
                "file_name": file.name,
                "tmp": tmp,
                "fps": fps,
                "total_frames": total_frames,
                "processed": 0,
                "running": True,
                "last_emotions": [],
                "last_composite": [],
                "last_analysis_frame": -999,
            }
            st.session_state.vid_display = None

        vs = st.session_state.vid_state
        duration = vs["total_frames"] / vs["fps"]
        tracker_local = st.session_state.warning_tracker

        # ── 顶部控制栏 ──
        ctl1, ctl2, ctl3, ctl4 = st.columns([2.5, 1, 1, 1])
        pct = vs["processed"] / max(vs["total_frames"], 1) * 100
        finished = vs["processed"] >= vs["total_frames"]
        with ctl1:
            elapsed_s = vs["processed"] / vs["fps"]
            st.markdown(f"🎬 **{vs['file_name']}** | {vs['fps']:.0f}fps | 进度 {elapsed_s:.0f}s/{duration:.0f}s ({pct:.0f}%)")
        with ctl2:
            analysis_sec = st.selectbox("分析间隔", [0.5, 1, 2, 3, 5], index=2,
                                        key="vid_interval",
                                        help="每隔多少秒做一次完整表情分析，其余帧仅做人脸检测保持流畅")
        with ctl3:
            if finished:
                st.button("✅ 已完成", use_container_width=True, key="vid_done", disabled=True)
            elif vs["running"]:
                if st.button("⏹ 停止处理", use_container_width=True, key="vid_stop", type="primary"):
                    vs["running"] = False
                    st.rerun()
            else:
                if st.button("▶ 继续处理", use_container_width=True, key="vid_resume", type="primary"):
                    vs["running"] = True
                    st.rerun()
        with ctl4:
            if st.button("🔄 重置", use_container_width=True, key="vid_reset"):
                vs["running"] = False
                st.session_state.vid_display = None
                del st.session_state["vid_state"]
                st.rerun()

        # ── 实时指标卡片 ──
        if analyzer.frame_records:
            last_fr = analyzer.frame_records[-1]
            mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
            mc1.metric("检测人数", last_fr.total_faces)
            mc2.metric("主导表情", last_fr.main_emotion)
            mc3.metric("课堂状态", last_fr.classroom_state)
            mc4.metric("抬头率", f"{last_fr.head_up_rate*100:.0f}%")
            wl = tracker_local.current_level
            wl_emoji = {"Green": "🟢", "Yellow": "🟡", "Red": "🔴"}.get(wl, "⚪")
            mc5.metric("预警等级", f"{wl_emoji} {wl}")
            mc6.metric("分析次数", len(analyzer.frame_records))
        else:
            st.caption("等待首次分析...")

        bar = st.progress(min(pct / 100, 1.0))

        # ── 分栏：帧画面 + 时序图 ──
        col_img, col_chart = st.columns([1, 1.2])

        # 先用 session_state 中的缓存图占位，避免画面消失
        with col_img:
            if st.session_state.get("vid_display") is not None:
                st.image(st.session_state.vid_display, use_container_width=True)
            else:
                st.info("等待视频帧...")

        with col_chart:
            if len(analyzer.frame_records) >= 2:
                show_time_series_chart(key_suffix="live")
            elif len(analyzer.frame_records) == 1:
                st.info("至少需要 2 个分析点才能显示时序图")
            else:
                st.info("等待分析数据...")

        if vs["running"]:
            cap = cv2.VideoCapture(vs["tmp"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, vs["processed"])

            analysis_interval = max(1, int(vs["fps"] * analysis_sec))
            # 每批显示的帧数（仅做人脸检测，保证停止按钮响应）
            batch_size = 10
            last_display = None

            for _ in range(batch_size):
                ok, frame = cap.read()
                if not ok:
                    vs["running"] = False
                    break
                vs["processed"] += 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 每帧做人脸检测（轻量），保证框跟随人脸移动、画面流畅
                faces = face_detector.detect_faces(rgb)
                faces = [f for f in faces if f.confidence >= threshold]

                # 每隔 analysis_interval 帧做一次完整分析（表情 + 头姿态 + 统计）
                do_analysis = (vs["processed"] - vs.get("last_analysis_frame", -999)) >= analysis_interval

                if do_analysis and faces:
                    emotions = []
                    composite_emotions = []
                    head_up, head_down = 0, 0
                    for face in faces:
                        roi = face_detector.extract_face_roi(rgb, face)
                        if roi.size == 0:
                            continue
                        try:
                            try:
                                pitch, yaw, _ = face_detector.estimate_head_pose(face, rgb.shape)
                                far = face_detector.face_aspect_ratio(face)
                                status = face_detector.classify_head_pose(pitch, face_ar=far)
                            except Exception:
                                pitch, yaw = 0.0, 0.0
                                status = "正常"
                            probs = st.session_state.recognizer.recognize(roi, head_status=status)
                            emotions.append(probs)
                            comp = classify_classroom_emotion(probs, yaw, pitch)
                            comp_top = max(comp, key=comp.get) if comp else "N/A"
                            composite_emotions.append({"scores": comp, "top": comp_top, "yaw": yaw, "pitch": pitch})
                            if status == "低头":
                                head_down += 1
                            else:
                                head_up += 1
                        except Exception:
                            continue

                    vs["last_emotions"] = emotions
                    vs["last_composite"] = composite_emotions
                    vs["last_analysis_frame"] = vs["processed"]

                    elapsed = vs["processed"] / vs["fps"]
                    per_frame = aggregate_per_frame(
                        emotions, vs["processed"], elapsed,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        file.name,
                    )
                    _hp = head_up + head_down
                    per_frame.head_up_count = head_up
                    per_frame.head_down_count = head_down
                    per_frame.head_up_rate = round(head_up / _hp, 3) if _hp > 0 else 1.0
                    per_frame.classroom_state = _classify(per_frame)
                    analyzer.add_frame_record(per_frame)
                    save_records(faces, emotions, file.name)
                    tracker_local.feed(per_frame.classroom_state)
                else:
                    # 非分析帧：复用上次缓存的识别结果来标注人脸框
                    emotions = list(vs.get("last_emotions", []))
                    composite_emotions = list(vs.get("last_composite", []))
                    # 如果缓存表情不够（首次、或人脸数变了），用 Neutral 补齐
                    while len(emotions) < len(faces):
                        emotions.append({"Neutral": 1.0, "Happy": 0.0, "Sad": 0.0,
                                         "Angry": 0.0, "Surprise": 0.0, "Fear": 0.0,
                                         "Disgust": 0.0, "Contempt": 0.0})

                last_display = render_result(rgb, faces, emotions, composite_emotions)

            cap.release()

            if last_display is not None:
                st.session_state.vid_display = last_display

            if vs["running"]:
                st.rerun()

        # ── 处理完毕后展示最终结果 ──
        if not vs["running"]:
            st.markdown("---")
            if vs["processed"] >= vs["total_frames"]:
                st.success(f"✅ 处理完成! 共 {len(analyzer.frame_records)} 次分析 (扫描 {vs['total_frames']} 帧)")
            else:
                st.info(f"⏸ 已停止 ({len(analyzer.frame_records)} 次分析, 扫描 {vs['processed']}/{vs['total_frames']} 帧)")

            show_status_bar()

            st.markdown("---")
            if len(analyzer.frame_records) >= 2:
                show_time_series_chart(key_suffix="final")
            elif analyzer.frame_records:
                st.info("至少需要 2 个分析点才能显示时序图")

            # 趋势分析结论
            if analyzer.frame_records:
                st.markdown("---")
                st.subheader("📋 趋势分析结论")
                states = [fr.classroom_state for fr in analyzer.frame_records]
                good_pct = sum(1 for s in states if "良好" in s) / len(states) * 100
                low_pct = sum(1 for s in states if "低落" in s) / len(states) * 100
                volatile_pct = sum(1 for s in states if "波动" in s) / len(states) * 100
                stable_pct = sum(1 for s in states if "平稳" in s) / len(states) * 100

                avg_head_up = sum(fr.head_up_rate for fr in analyzer.frame_records) / len(analyzer.frame_records) * 100

                tc1, tc2 = st.columns(2)
                with tc1:
                    st.write(f"- 平均抬头率: **{avg_head_up:.1f}%**")
                    st.write(f"- 状态良好占比: **{good_pct:.1f}%**")
                    st.write(f"- 状态平稳占比: **{stable_pct:.1f}%**")
                with tc2:
                    st.write(f"- 需关注占比: **{low_pct:.1f}%**")
                    st.write(f"- 注意力波动占比: **{volatile_pct:.1f}%**")

                window_results = compute_sliding_window(analyzer.frame_records)
                if window_results:
                    red_count = sum(1 for wr in window_results if wr.warning_level == "Red")
                    yellow_count = sum(1 for wr in window_results if wr.warning_level == "Yellow")
                    if red_count > 0:
                        st.error(f"⚠️ 检测到 {red_count} 次红色预警（低落/波动），建议关注该时段")
                    elif yellow_count > 0:
                        st.warning(f"⚡ 检测到 {yellow_count} 次黄色预警（中性占比过高），课堂互动可能需要加强")
                    else:
                        st.success("✅ 课堂整体状态正常，未触发预警")
    else:
        st.markdown(
            '<div class="empty-guide"><h2>🎬 请上传课堂视频</h2>'
            '<p>支持 MP4/AVI/MOV 格式，≥1分钟视频可进行完整的时序分析和预警</p></div>',
            unsafe_allow_html=True)

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
                                    "Angry": 0.0, "Surprise": 0.0, "Fear": 0.0,
                                    "Disgust": 0.0, "Contempt": 0.0})
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

    ctx = webrtc_streamer(
        key="cam-webrtc",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=CamProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.session_state.cam_active = ctx.state.playing

    # 统计面板放在 fragment 中，每2秒自动刷新，不影响视频流
    @st.fragment(run_every=2)
    def _cam_stats_panel():
        result_queue = st.session_state.cam_result_queue
        cam_active = st.session_state.get("cam_active", False)

        if not cam_active:
            # 摄像头已停止：清空队列残留，不再新增分析记录
            while not result_queue.empty():
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    break
            if analyzer.frame_records:
                show_status_bar()
                show_stats()
                if len(analyzer.frame_records) >= 3:
                    show_time_series_chart(key_suffix="cam")
            return

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
            show_status_bar()
            show_stats()
            if len(analyzer.frame_records) >= 3:
                show_time_series_chart(key_suffix="cam")

    _cam_stats_panel()
