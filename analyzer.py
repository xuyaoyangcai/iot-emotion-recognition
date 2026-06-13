import csv
from expression_recognizer import EMOTIONS


class ResultAnalyzer:
    def __init__(self):
        self.records: list[dict] = []

    def add_record(self, timestamp: str, image_name: str,
                   person_id: int, emotion: str, confidence: float):
        self.records.append({
            "timestamp": timestamp,
            "image_name": image_name,
            "person_id": person_id,
            "emotion": emotion,
            "confidence": round(confidence, 4),
        })

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

    def export_csv(self, filepath_or_buffer):
        fieldnames = ["Timestamp", "Image", "Person_ID",
                      "Happy", "Neutral", "Sad", "Angry",
                      "Surprise", "Fear", "Disgust", "Dominant"]
        if isinstance(filepath_or_buffer, str):
            f = open(filepath_or_buffer, "w", newline="", encoding="utf-8")
            _close = True
        else:
            f = filepath_or_buffer
            _close = False
        try:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # 按时间分组聚合
            groups = {}
            for r in self.records:
                key = (r["timestamp"], r["image_name"])
                if key not in groups:
                    groups[key] = {"counts": {e: 0 for e in EMOTIONS}, "total": 0}
                groups[key][r["emotion"]] = groups[key]["counts"].get(r["emotion"], 0) + 1
                groups[key]["total"] += 1

            for (ts, img), data in groups.items():
                row = {"Timestamp": ts, "Image": img,
                       "Person_ID": data["total"]}
                for e in EMOTIONS:
                    row[e] = data["counts"][e]
                dominant = max(data["counts"], key=data["counts"].get)
                row["Dominant"] = dominant if data["counts"][dominant] > 0 else "N/A"
                writer.writerow(row)
        finally:
            if _close:
                f.close()

    def clear(self):
        self.records.clear()

    def get_timeline(self) -> list[dict]:
        timeline = []
        for r in self.records:
            timeline.append({
                "timestamp": r["timestamp"],
                "emotion": r["emotion"],
            })
        return timeline
