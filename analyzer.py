import csv
from datetime import datetime
from expression_recognizer import EMOTIONS
from classroom_state import PerFrameResult, classify_classroom_state


class ResultAnalyzer:
    def __init__(self):
        self.records: list[dict] = []
        self.frame_records: list[PerFrameResult] = []

    # ── 逐人脸记录 (向后兼容) ──

    def add_record(self, timestamp: str, image_name: str,
                   person_id: int, emotion: str, confidence: float):
        self.records.append({
            "timestamp": timestamp,
            "image_name": image_name,
            "person_id": person_id,
            "emotion": emotion,
            "confidence": round(confidence, 4),
        })

    # ── 逐帧记录 (课堂分析用) ──

    def add_frame_record(self, result: PerFrameResult):
        self.frame_records.append(result)

    def get_per_frame_data(self) -> list[PerFrameResult]:
        return self.frame_records

    def get_latest_classroom_state(self) -> str:
        if self.frame_records:
            return self.frame_records[-1].classroom_state
        return "N/A"

    # ── 统计 ──

    def get_summary(self) -> dict:
        if not self.records:
            return {
                "total_people": 0,
                "expression_count": {e: 0 for e in EMOTIONS},
                "expression_ratio": {e: 0.0 for e in EMOTIONS},
                "main_expression": "N/A",
            }
        total = len(self.records)
        counts = {e: 0 for e in EMOTIONS}
        for r in self.records:
            em = r["emotion"]
            if em in counts:
                counts[em] += 1

        ratios = {e: round(counts[e] / total * 100, 1) for e in EMOTIONS}
        main = max(counts, key=counts.get)
        return {
            "total_people": total,
            "expression_count": counts,
            "expression_ratio": ratios,
            "main_expression": main,
        }

    # ── CSV 导出 (大作业规范格式) ──

    def export_csv(self, filepath_or_buffer):
        """导出为课堂状态 CSV 格式"""
        fieldnames = ["时间", "图片名称", "检测人数",
                      "Happy人数", "Neutral人数", "Sad人数", "Angry人数",
                      "主要表情", "课堂状态"]

        if isinstance(filepath_or_buffer, str):
            f = open(filepath_or_buffer, "w", newline="", encoding="utf-8-sig")
            _close = True
        else:
            f = filepath_or_buffer
            _close = False
        try:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            if self.frame_records:
                for fr in self.frame_records:
                    writer.writerow({
                        "时间": fr.timestamp_str,
                        "图片名称": fr.image_name,
                        "检测人数": fr.total_faces,
                        "Happy人数": fr.counts.get("Happy", 0),
                        "Neutral人数": fr.counts.get("Neutral", 0),
                        "Sad人数": fr.counts.get("Sad", 0),
                        "Angry人数": fr.counts.get("Angry", 0),
                        "主要表情": fr.main_emotion,
                        "课堂状态": fr.classroom_state,
                    })
            else:
                # 无帧记录时从 per-face records 聚合
                groups = {}
                for r in self.records:
                    key = (r["timestamp"], r["image_name"])
                    if key not in groups:
                        groups[key] = {"counts": {e: 0 for e in EMOTIONS}, "total": 0}
                    em = r["emotion"]
                    if em in groups[key]["counts"]:
                        groups[key]["counts"][em] += 1
                    groups[key]["total"] += 1

                for (ts, img), data in groups.items():
                    counts = data["counts"]
                    total = data["total"]
                    main = max(counts, key=counts.get) if total > 0 else "N/A"
                    state = classify_classroom_state(counts, total)
                    writer.writerow({
                        "时间": ts,
                        "图片名称": img,
                        "检测人数": total,
                        "Happy人数": counts.get("Happy", 0),
                        "Neutral人数": counts.get("Neutral", 0),
                        "Sad人数": counts.get("Sad", 0),
                        "Angry人数": counts.get("Angry", 0),
                        "主要表情": main,
                        "课堂状态": state,
                    })
        finally:
            if _close:
                f.close()

    def export_time_series_csv(self, filepath_or_buffer,
                               window_results: list = None):
        """导出时序分析 CSV：帧号、时间戳、窗口均值、预警等级"""
        fieldnames = ["帧号", "时间_秒", "时间", "图片名称", "检测人数",
                      "Happy均值", "Neutral均值", "Sad均值", "Angry均值",
                      "预警等级"]

        if isinstance(filepath_or_buffer, str):
            f = open(filepath_or_buffer, "w", newline="", encoding="utf-8-sig")
            _close = True
        else:
            f = filepath_or_buffer
            _close = False
        try:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # 构建 window map
            win_map = {}
            if window_results:
                for wr in window_results:
                    win_map[wr.center_frame] = wr

            for fr in self.frame_records:
                wr = win_map.get(fr.frame_number)
                row = {
                    "帧号": fr.frame_number,
                    "时间_秒": round(fr.timestamp_seconds, 1),
                    "时间": fr.timestamp_str,
                    "图片名称": fr.image_name,
                    "检测人数": fr.total_faces,
                    "Happy均值": round(wr.window_mean.get("Happy", 0) * 100, 1) if wr else "",
                    "Neutral均值": round(wr.window_mean.get("Neutral", 0) * 100, 1) if wr else "",
                    "Sad均值": round(wr.window_mean.get("Sad", 0) * 100, 1) if wr else "",
                    "Angry均值": round(wr.window_mean.get("Angry", 0) * 100, 1) if wr else "",
                    "预警等级": wr.warning_level if wr else "",
                }
                writer.writerow(row)
        finally:
            if _close:
                f.close()

    def clear(self):
        self.records.clear()
        self.frame_records.clear()

    def get_timeline(self) -> list[dict]:
        timeline = []
        for r in self.records:
            timeline.append({
                "timestamp": r["timestamp"],
                "emotion": r["emotion"],
            })
        return timeline
