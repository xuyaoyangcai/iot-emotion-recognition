import os
import tempfile
from analyzer import ResultAnalyzer
from expression_recognizer import EMOTIONS
from classroom_state import aggregate_per_frame


def test_add_and_summary():
    analyzer = ResultAnalyzer()
    analyzer.add_record("2026-06-13 10:20:00", "test.jpg", 1, "Happy", 0.85)
    analyzer.add_record("2026-06-13 10:20:00", "test.jpg", 2, "Neutral", 0.72)
    analyzer.add_record("2026-06-13 10:20:01", "test.jpg", 1, "Happy", 0.90)
    analyzer.add_record("2026-06-13 10:20:01", "test.jpg", 2, "Sad", 0.65)

    summary = analyzer.get_summary()
    assert summary["total_people"] == 4
    assert summary["expression_count"]["Happy"] == 2
    assert summary["expression_count"]["Neutral"] == 1
    assert summary["expression_count"]["Sad"] == 1
    assert summary["expression_count"]["Angry"] == 0
    assert summary["expression_ratio"]["Happy"] == 50.0
    assert summary["expression_ratio"]["Neutral"] == 25.0
    assert summary["expression_ratio"]["Sad"] == 25.0
    assert summary["main_expression"] == "Happy"


def test_export_csv_new_format():
    """验证新 CSV 格式（中文表头）"""
    analyzer = ResultAnalyzer()
    analyzer.add_record("2026-06-13 10:20:00", "test.jpg", 1, "Happy", 0.85)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="", encoding="utf-8") as f:
        tmp = f.name
    try:
        analyzer.export_csv(tmp)
        with open(tmp, "r", encoding="utf-8-sig") as f:
            content = f.read()
        assert "时间" in content
        assert "检测人数" in content
        assert "Happy人数" in content
        assert "Neutral人数" in content
        assert "主要表情" in content
        assert "课堂状态" in content
        # 旧表头不应出现
        assert "Timestamp" not in content
        assert "Person_ID" not in content
        assert "Dominant" not in content
    finally:
        os.unlink(tmp)


def test_export_csv_with_frame_records():
    """帧记录优先导出"""
    analyzer = ResultAnalyzer()
    emotions = [{"Happy": 0.8, "Neutral": 0.1, "Sad": 0.05, "Angry": 0.02,
                 "Surprise": 0.01, "Fear": 0.01, "Disgust": 0.01}]
    fr = aggregate_per_frame(emotions, 1, 0.0, "2026-06-13 10:20:00", "test.jpg")
    analyzer.add_frame_record(fr)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="", encoding="utf-8") as f:
        tmp = f.name
    try:
        analyzer.export_csv(tmp)
        with open(tmp, "r", encoding="utf-8-sig") as f:
            content = f.read()
        assert "课堂状态良好" in content  # 1人Happy → 良好
        assert "1" in content
    finally:
        os.unlink(tmp)


def test_export_time_series_csv():
    """时序 CSV 导出"""
    analyzer = ResultAnalyzer()
    for i in range(7):
        emotions = [{"Happy": 0.5, "Neutral": 0.3, "Sad": 0.2, "Angry": 0.0,
                     "Surprise": 0.0, "Fear": 0.0, "Disgust": 0.0}]
        fr = aggregate_per_frame(emotions, i + 1, float(i),
                                 f"2026-06-13 10:0{i}:00", "test.mp4")
        analyzer.add_frame_record(fr)

    from classroom_state import compute_sliding_window
    window_results = compute_sliding_window(analyzer.frame_records)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="", encoding="utf-8") as f:
        tmp = f.name
    try:
        analyzer.export_time_series_csv(tmp, window_results)
        with open(tmp, "r", encoding="utf-8-sig") as f:
            content = f.read()
        assert "帧号" in content
        assert "预警等级" in content
        assert "Happy均值" in content
    finally:
        os.unlink(tmp)


def test_clear():
    analyzer = ResultAnalyzer()
    analyzer.add_record("2026-06-13 10:20:00", "test.jpg", 1, "Happy", 0.85)
    emotions = [{"Happy": 0.8, "Neutral": 0.1, "Sad": 0.05, "Angry": 0.02,
                 "Surprise": 0.01, "Fear": 0.01, "Disgust": 0.01}]
    fr = aggregate_per_frame(emotions, 1, 0.0, "2026-06-13 10:20:00", "test.jpg")
    analyzer.add_frame_record(fr)

    analyzer.clear()
    summary = analyzer.get_summary()
    assert summary["total_people"] == 0
    assert len(analyzer.frame_records) == 0


def test_get_latest_classroom_state():
    analyzer = ResultAnalyzer()
    assert analyzer.get_latest_classroom_state() == "N/A"
    emotions = [{"Happy": 0.8, "Neutral": 0.1, "Sad": 0.05, "Angry": 0.02,
                 "Surprise": 0.01, "Fear": 0.01, "Disgust": 0.01}]
    fr = aggregate_per_frame(emotions, 1, 0.0, "2026-06-13 10:20:00", "test.jpg")
    analyzer.add_frame_record(fr)
    assert analyzer.get_latest_classroom_state() == "课堂状态良好"
