"""课堂状态分析模块 — 状态分类、帧聚合、滑动窗口、预警追踪"""
from dataclasses import dataclass, field
from expression_recognizer import EMOTIONS


@dataclass
class PerFrameResult:
    """单帧表情识别汇总"""
    frame_number: int
    timestamp_seconds: float
    timestamp_str: str
    image_name: str
    total_faces: int
    counts: dict  # {"Happy": 2, "Neutral": 3, ...}
    ratios: dict  # {"Happy": 0.4, "Neutral": 0.6, ...}
    main_emotion: str
    classroom_state: str


@dataclass
class WindowResult:
    """滑动窗口统计结果"""
    center_frame: int
    window_start: int
    window_end: int
    window_mean: dict   # 各表情占比均值
    window_variance: dict  # 各表情占比方差
    warning_level: str  # Green / Yellow / Red / Normal


def classify_classroom_state(counts: dict, total: int) -> str:
    """根据表情分布判断课堂状态，按优先级评估"""
    if total == 0:
        return "未检测到学生"

    happy = counts.get("Happy", 0)
    neutral = counts.get("Neutral", 0)
    sad = counts.get("Sad", 0)
    angry = counts.get("Angry", 0)
    surprise = counts.get("Surprise", 0)

    # 规则1: Happy + Neutral >= 70% → 良好
    if (happy + neutral) / total >= 0.70:
        return "课堂状态良好"

    # 规则2: Sad + Angry >= 40% → 需关注 (优先级高于Neutral最高)
    if (sad + angry) / total >= 0.40:
        return "课堂状态较低落或需要关注"

    # 找占比最高的表情
    max_count = max(counts.values())
    if max_count == 0:
        return "未检测到学生"

    # 规则3: Surprise 占比最高 → 注意力波动
    if surprise == max_count:
        for emo in EMOTIONS:
            if emo != "Surprise" and counts.get(emo, 0) == max_count:
                break
        else:
            return "课堂注意力波动较大"

    # 规则4: Neutral 占比最高 → 平稳
    if neutral == max_count:
        return "课堂状态平稳"

    return "课堂状态平稳"


def aggregate_per_frame(emotions: list[dict], frame_number: int,
                        timestamp_seconds: float, timestamp_str: str,
                        image_name: str) -> PerFrameResult:
    """将单帧中所有人脸的表情结果汇总为帧级统计"""
    total = len(emotions)
    counts = {e: 0 for e in EMOTIONS}

    for probs in emotions:
        top = max(probs, key=probs.get)
        counts[top] += 1

    ratios = {e: (counts[e] / total if total > 0 else 0.0) for e in EMOTIONS}
    main_emotion = max(counts, key=counts.get) if total > 0 else "N/A"

    return PerFrameResult(
        frame_number=frame_number,
        timestamp_seconds=timestamp_seconds,
        timestamp_str=timestamp_str,
        image_name=image_name,
        total_faces=total,
        counts=counts,
        ratios=ratios,
        main_emotion=main_emotion,
        classroom_state=classify_classroom_state(counts, total),
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

        # 基于窗口均值判断预警
        warning = _window_warning(mean_vals)

        results.append(WindowResult(
            center_frame=per_frame_data[i].frame_number,
            window_start=window[0].frame_number,
            window_end=window[-1].frame_number,
            window_mean=mean_vals,
            window_variance=var_vals,
            warning_level=warning,
        ))
    return results


def _window_warning(mean_vals: dict) -> str:
    """基于窗口均值判断预警等级"""
    sad = mean_vals.get("Sad", 0)
    angry = mean_vals.get("Angry", 0)
    fear = mean_vals.get("Fear", 0)
    happy = mean_vals.get("Happy", 0)
    neutral = mean_vals.get("Neutral", 0)

    if sad + angry + fear > 0.40:
        return "Red"
    elif neutral > 0.60:
        return "Yellow"
    elif happy + neutral > 0.60:
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

    def update(self, classroom_state: str) -> str:
        """输入最新帧的课堂状态，返回当前预警等级"""
        # 重置所有计数器
        self.good_streak = 0
        self.stable_streak = 0
        self.low_streak = 0
        self.volatile_streak = 0

        # 根据当前状态累加对应计数器
        if classroom_state == "课堂状态良好":
            self.good_streak = 1
            # 需要从历史恢复，这里简化：只追踪当前状态
            # 实际通过外部累加实现
        elif classroom_state == "课堂状态平稳":
            self.stable_streak = 1
        elif classroom_state in ("课堂状态较低落或需要关注",):
            self.low_streak = 1
        elif classroom_state in ("课堂注意力波动较大",):
            self.volatile_streak = 1

        # 简化版：直接根据单帧状态判断
        # 连续追踪由 app.py 中的循环累加实现
        return self.current_level

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
        else:
            # 未检测到学生 — 不重置也不累加
            pass

        # 优先级: Red > Yellow > Green > Normal
        if self.low_streak >= 3 or self.volatile_streak >= 3:
            self.current_level = "Red"
        elif self.stable_streak >= 5:
            self.current_level = "Yellow"
        elif self.good_streak >= 10:
            self.current_level = "Green"
        else:
            self.current_level = "Normal"

        return self.current_level
