import os
import tempfile
from analyzer import ResultAnalyzer
from expression_recognizer import EMOTIONS


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


def test_export_csv():
    analyzer = ResultAnalyzer()
    analyzer.add_record("2026-06-13 10:20:00", "test.jpg", 1, "Happy", 0.85)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="", encoding="utf-8") as f:
        tmp = f.name
    try:
        analyzer.export_csv(tmp)
        with open(tmp, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Timestamp" in content
        assert "Happy" in content
    finally:
        os.unlink(tmp)


def test_clear():
    analyzer = ResultAnalyzer()
    analyzer.add_record("2026-06-13 10:20:00", "test.jpg", 1, "Happy", 0.85)
    analyzer.clear()
    summary = analyzer.get_summary()
    assert summary["total_people"] == 0
