"""课堂状态分析模块 — 状态分类、帧聚合、滑动窗口、预警追踪（含抬头率）"""
from dataclasses import dataclass
from expression_recognizer import EMOTIONS


@dataclass
class PerFrameResult:
    """单帧表情识别汇总"""
    frame_number: int
    timestamp_seconds: float
    timestamp_str: str
    image_name: str
    total_faces: int
    counts: dict         # {"Happy": 2, "Neutral": 3, ...}
    ratios: dict         # {"Happy": 0.4, "Neutral": 0.6, ...}
    main_emotion: str
    classroom_state: str
    head_up_count: int = 0     # 抬头人数
    head_down_count: int = 0   # 低头人数
    head_up_rate: float = 1.0  # 抬头率


@dataclass
class WindowResult:
    """滑动窗口统计结果"""
    center_frame: int
    window_start: int
    window_end: int
    window_mean: dict       # 各表情占比均值
    window_variance: dict   # 各表情占比方差
    head_up_mean: float     # 窗口内抬头率均值
    warning_level: str      # Green / Yellow / Red / Normal


def classify_classroom_state(counts: dict, total: int,
                             head_up_rate: float = 1.0) -> str:
    """
    根据表情分布 + 抬头率综合判断课堂状态，按优先级评估
    head_up_rate: 抬头率 0~1，默认1.0（无低头检测时）
    """
    if total == 0:
        return "未检测到学生"

    happy = counts.get("Happy", 0)
    neutral = counts.get("Neutral", 0)
    sad = counts.get("Sad", 0)
    angry = counts.get("Angry", 0)
    surprise = counts.get("Surprise", 0)
    contempt = counts.get("Contempt", 0)
    fear = counts.get("Fear", 0)
    disgust = counts.get("Disgust", 0)
    max_count = max(counts.values())

    # 规则1: 低头率过高 (>50%) → 需关注，优先级最高
    if head_up_rate < 0.50:
        return "课堂状态较低落或需要关注"

    # 规则2: Happy + Neutral >= 70% + 抬头率 ≥ 60% → 良好
    if (happy + neutral) / total >= 0.70 and head_up_rate >= 0.60:
        return "课堂状态良好"

    # 规则3: Sad + Angry + Contempt >= 40% → 需关注
    if (sad + angry + contempt) / total >= 0.40:
        return "课堂状态较低落或需要关注"

    if max_count == 0:
        return "未检测到学生"

    # 规则4: Contempt 占比最高 → 注意力波动（轻蔑/不屑）
    if contempt == max_count and contempt > 0:
        return "课堂注意力波动较大"

    # 规则5: Surprise 占比最高 → 注意力波动
    if surprise == max_count:
        for emo in EMOTIONS:
            if emo != "Surprise" and counts.get(emo, 0) == max_count:
                break
        else:
            return "课堂注意力波动较大"

    # 规则6: Neutral 占比最高 → 平稳
    if neutral == max_count:
        return "课堂状态平稳"

    return "课堂状态平稳"


def aggregate_per_frame(emotions: list[dict], frame_number: int,
                        timestamp_seconds: float, timestamp_str: str,
                        image_name: str,
                        head_up: int = 0, head_down: int = 0) -> PerFrameResult:
    """将单帧中所有人脸的表情结果汇总为帧级统计"""
    total = len(emotions)
    counts = {e: 0 for e in EMOTIONS}

    for probs in emotions:
        top = max(probs, key=probs.get)
        counts[top] += 1

    ratios = {e: (counts[e] / total if total > 0 else 0.0) for e in EMOTIONS}
    main_emotion = max(counts, key=counts.get) if total > 0 else "N/A"

    # 抬头率
    if total > 0 and (head_up + head_down) > 0:
        head_up_rate = head_up / (head_up + head_down)
    else:
        head_up_rate = 1.0

    return PerFrameResult(
        frame_number=frame_number,
        timestamp_seconds=timestamp_seconds,
        timestamp_str=timestamp_str,
        image_name=image_name,
        total_faces=total,
        counts=counts,
        ratios=ratios,
        main_emotion=main_emotion,
        classroom_state=classify_classroom_state(counts, total, head_up_rate),
        head_up_count=head_up,
        head_down_count=head_down,
        head_up_rate=round(head_up_rate, 3),
    )


def compute_sliding_window(per_frame_data: list[PerFrameResult],
                           window_size: int = 5) -> list[WindowResult]:
    """滑动窗口统计：计算窗口内各表情占比的均值和方差"""
    results = []
    for i in range(window_size - 1, len(per_frame_data)):
        window = per_frame_data[i - window_size + 1: i + 1]

        mean_vals = {e: 0.0 for e in EMOTIONS}
        var_vals = {e: 0.0 for e in EMOTIONS}

        for e in EMOTIONS:
            vals = [fr.ratios.get(e, 0.0) for fr in window]
            mean_vals[e] = sum(vals) / window_size
            var_vals[e] = sum((v - mean_vals[e]) ** 2 for v in vals) / window_size

        head_up_vals = [fr.head_up_rate for fr in window]
        head_up_mean = sum(head_up_vals) / window_size

        warning = _window_warning(mean_vals, head_up_mean)

        results.append(WindowResult(
            center_frame=per_frame_data[i].frame_number,
            window_start=window[0].frame_number,
            window_end=window[-1].frame_number,
            window_mean=mean_vals,
            window_variance=var_vals,
            head_up_mean=round(head_up_mean, 3),
            warning_level=warning,
        ))
    return results


def _window_warning(mean_vals: dict, head_up_mean: float = 1.0) -> str:
    """基于窗口均值和抬头率判断预警等级"""
    sad = mean_vals.get("Sad", 0)
    angry = mean_vals.get("Angry", 0)
    fear = mean_vals.get("Fear", 0)
    contempt = mean_vals.get("Contempt", 0)
    happy = mean_vals.get("Happy", 0)
    neutral = mean_vals.get("Neutral", 0)

    # 低头率过高 → Red
    if head_up_mean < 0.40:
        return "Red"
    # 负面情绪高 → Red
    if sad + angry + fear + contempt > 0.40:
        return "Red"
    # 中性占比过高 + 低头率偏高 → Yellow
    if neutral > 0.55 and head_up_mean < 0.70:
        return "Yellow"
    # 中性占比过高 → Yellow
    if neutral > 0.60:
        return "Yellow"
    # 良好
    if happy + neutral > 0.60 and head_up_mean > 0.70:
        return "Green"
    return "Normal"


class WarningTracker:
    """三级预警追踪器：连续帧状态累积触发预警"""

    def __init__(self):
        self.good_streak = 0
        self.stable_streak = 0
        self.low_streak = 0
        self.volatile_streak = 0
        self.current_level = "Normal"

    def feed(self, classroom_state: str) -> str:
        """喂入一帧状态，内部累加连续计数，返回预警等级"""
        good = classroom_state == "课堂状态良好"
        stable = classroom_state == "课堂状态平稳"
        low = classroom_state == "课堂状态较低落或需要关注"
        volatile = classroom_state == "课堂注意力波动较大"

        if good:
            self.good_streak += 1
            self.stable_streak = 0
            self.low_streak = 0
            self.volatile_streak = 0
        elif stable:
            self.good_streak = 0
            self.stable_streak += 1
            self.low_streak = 0
            self.volatile_streak = 0
        elif low:
            self.good_streak = 0
            self.stable_streak = 0
            self.low_streak += 1
            self.volatile_streak = 0
        elif volatile:
            self.good_streak = 0
            self.stable_streak = 0
            self.low_streak = 0
            self.volatile_streak += 1

        if self.low_streak >= 2 or self.volatile_streak >= 2:
            self.current_level = "Red"
        elif self.stable_streak >= 3:
            self.current_level = "Yellow"
        elif self.good_streak >= 4:
            self.current_level = "Green"
        else:
            self.current_level = "Normal"

        return self.current_level
