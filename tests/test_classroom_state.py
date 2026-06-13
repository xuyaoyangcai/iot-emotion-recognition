"""课堂状态分析模块单元测试"""
from classroom_state import (
    classify_classroom_state, aggregate_per_frame,
    compute_sliding_window, WarningTracker,
    PerFrameResult, WindowResult,
)


class TestClassifyClassroomState:

    def test_no_students(self):
        assert classify_classroom_state({}, 0) == "未检测到学生"

    def test_good(self):
        # Happy + Neutral >= 70%
        counts = {"Happy": 4, "Neutral": 3, "Sad": 1, "Angry": 0,
                  "Surprise": 1, "Fear": 0, "Disgust": 0}
        assert classify_classroom_state(counts, 9) == "课堂状态良好"

    def test_low_priority(self):
        # Sad + Angry >= 40%, overrides Neutral highest
        counts = {"Happy": 1, "Neutral": 2, "Sad": 4, "Angry": 1,
                  "Surprise": 0, "Fear": 0, "Disgust": 0}
        assert classify_classroom_state(counts, 8) == "课堂状态较低落或需要关注"

    def test_stable(self):
        # Neutral 占比最高，且 Happy+Neutral < 70%
        counts = {"Happy": 0, "Neutral": 3, "Sad": 1, "Angry": 1,
                  "Surprise": 1, "Fear": 1, "Disgust": 0}
        assert classify_classroom_state(counts, 7) == "课堂状态平稳"

    def test_volatile_surprise(self):
        # Surprise highest
        counts = {"Happy": 0, "Neutral": 1, "Sad": 0, "Angry": 0,
                  "Surprise": 4, "Fear": 0, "Disgust": 0}
        assert classify_classroom_state(counts, 5) == "课堂注意力波动较大"

    def test_fallback_stable_when_happy_highest(self):
        # Happy 最高但 Happy+Neutral < 70%, Sad+Angry < 40%
        counts = {"Happy": 2, "Neutral": 1, "Sad": 1, "Angry": 1,
                  "Surprise": 1, "Fear": 1, "Disgust": 0}
        assert classify_classroom_state(counts, 7) == "课堂状态平稳"


class TestAggregatePerFrame:

    def test_basic(self):
        emotions = [
            {"Happy": 0.8, "Neutral": 0.1, "Sad": 0.05, "Angry": 0.02,
             "Surprise": 0.01, "Fear": 0.01, "Disgust": 0.01},
            {"Happy": 0.9, "Neutral": 0.05, "Sad": 0.02, "Angry": 0.01,
             "Surprise": 0.01, "Fear": 0.005, "Disgust": 0.005},
            {"Neutral": 0.6, "Happy": 0.2, "Sad": 0.1, "Angry": 0.05,
             "Surprise": 0.02, "Fear": 0.02, "Disgust": 0.01},
        ]
        result = aggregate_per_frame(emotions, 1, 0.0, "2026-06-13 10:00:00", "test.jpg")
        assert result.frame_number == 1
        assert result.total_faces == 3
        assert result.counts["Happy"] == 2
        assert result.counts["Neutral"] == 1
        assert result.main_emotion == "Happy"
        assert result.ratios["Happy"] == 2 / 3
        assert "课堂状态" in result.classroom_state

    def test_head_up_rate(self):
        """有低头时抬头率降低"""
        emotions = [
            {"Happy": 0.8, "Neutral": 0.1, "Sad": 0.0, "Angry": 0.0,
             "Surprise": 0.05, "Fear": 0.03, "Disgust": 0.02},
        ]
        result = aggregate_per_frame(emotions, 1, 0.0, "ts", "test.jpg",
                                     head_up=1, head_down=3)
        assert result.head_up_rate == 0.25
        assert result.head_up_count == 1
        assert result.head_down_count == 3

    def test_empty(self):
        result = aggregate_per_frame([], 1, 0.0, "2026-06-13 10:00:00", "test.jpg")
        assert result.total_faces == 0
        assert result.main_emotion == "N/A"
        assert result.classroom_state == "未检测到学生"


class TestClassifyHeadPose:

    def test_low_head_up_triggers_concern(self):
        """抬头率 < 50% → 需关注，哪怕表情是好的"""
        counts = {"Happy": 8, "Neutral": 2, "Sad": 0, "Angry": 0,
                  "Surprise": 0, "Fear": 0, "Disgust": 0}
        assert classify_classroom_state(counts, 10, head_up_rate=0.3) == \
            "课堂状态较低落或需要关注"

    def test_good_needs_head_up(self):
        """表情好但抬头率不够 → 不判为良好"""
        counts = {"Happy": 5, "Neutral": 3, "Sad": 1, "Angry": 1,
                  "Surprise": 0, "Fear": 0, "Disgust": 0}
        # Happy+Neutral=80% but head_up_rate=55%
        assert classify_classroom_state(counts, 10, head_up_rate=0.55) != \
            "课堂状态良好"


class TestSlidingWindow:

    def test_window(self):
        frames = []
        for i in range(7):
            happy = 0.5 if i < 5 else 0.1
            neutral = 0.3 if i < 5 else 0.7
            sad = 0.2 if i < 5 else 0.2
            emotions = [{"Happy": happy, "Neutral": neutral, "Sad": sad,
                         "Angry": 0.0, "Surprise": 0.0, "Fear": 0.0, "Disgust": 0.0}]
            fr = aggregate_per_frame(emotions, i + 1, float(i), f"ts_{i}", "test.mp4")
            frames.append(fr)

        results = compute_sliding_window(frames)
        assert len(results) == 3  # frames 5-7
        assert results[0].center_frame == 5
        assert results[-1].center_frame == 7


class TestWarningTracker:

    def test_green_after_10_good(self):
        tracker = WarningTracker()
        for i in range(10):
            level = tracker.feed("课堂状态良好")
            if i < 9:
                assert level == "Normal"
        assert level == "Green"

    def test_yellow_after_5_stable(self):
        tracker = WarningTracker()
        for i in range(5):
            level = tracker.feed("课堂状态平稳")
            if i < 4:
                assert level == "Normal"
        assert level == "Yellow"

    def test_red_after_3_low(self):
        tracker = WarningTracker()
        for i in range(3):
            level = tracker.feed("课堂状态较低落或需要关注")
            if i < 2:
                assert level == "Normal"
        assert level == "Red"

    def test_red_after_3_volatile(self):
        tracker = WarningTracker()
        for i in range(3):
            level = tracker.feed("课堂注意力波动较大")
            if i < 2:
                assert level == "Normal"
        assert level == "Red"

    def test_streak_reset(self):
        tracker = WarningTracker()
        for _ in range(4):
            tracker.feed("课堂状态良好")
        tracker.feed("课堂状态较低落或需要关注")  # resets
        assert tracker.good_streak == 0
        assert tracker.low_streak == 1

    def test_priority_red_over_yellow(self):
        """Red (3 low) triggers before Yellow (5 stable)"""
        tracker = WarningTracker()
        # 2 low + 5 stable should still be Normal, not Yellow (because low resets stable)
        tracker.feed("课堂状态较低落或需要关注")
        tracker.feed("课堂状态较低落或需要关注")
        # stable resets low_streak
        tracker.feed("课堂状态平稳")
        assert tracker.low_streak == 0
        level = tracker.feed("课堂状态平稳")
        assert level == "Normal"
