"""课堂复合情绪 — 融合表情识别 + 头姿态(yaw/pitch) 推导更丰富的课堂状态"""

CLASSROOM_EMOTIONS = [
    "Focused",       # looking at screen + neutral/happy
    "Distracted",    # large yaw (looking sideways)
    "Engaged",       # happy + looking forward
    "Confused",      # surprise/fear + head tilt
    "Thinking",      # head down + neutral
    "Tired",         # head down + sad
]

# (主导表情, 次选表情, yaw阈值, pitch阈值, 系数)
_RULES = {
    "Focused":      (["Neutral", "Happy"],    15, 10,  1.2),
    "Engaged":      (["Happy"],               15, 10,  1.3),
    "Distracted":   (["Neutral", "Sad"],      60, 99,  1.2),
    "Confused":     (["Surprise", "Fear"],    20, 15,  1.3),
    "Thinking":     (["Neutral", "Sad"],      15, 15,  1.1),
    "Tired":        (["Sad"],                 15, 15,  1.3),
}


def classify_classroom_emotion(emotion_probs: dict, yaw: float, pitch: float) -> dict:
    """
    综合基础表情概率 + 头部姿态 → 课堂复合情绪得分
    返回: {"Focused": 0.45, "Distracted": 0.12, ...}
    """
    scores = {}

    for label, (lead_emos, yaw_thresh, pitch_thresh, boost) in _RULES.items():
        score = 0.0

        for i, emo in enumerate(lead_emos):
            weight = 1.0 if i == 0 else 0.5
            score += emotion_probs.get(emo, 0) * weight

        yaw_match = 1.0 if abs(yaw) < yaw_thresh else max(0, 1.0 - (abs(yaw) - yaw_thresh) / 30)
        pitch_match = 1.0 if abs(pitch) < abs(pitch_thresh) else max(0, 1.0 - (abs(pitch) - abs(pitch_thresh)) / 20)

        if label == "Distracted":
            yaw_match = min(1.0, abs(yaw) / 60)

        if label in ("Thinking", "Tired"):
            pitch_match = min(1.0, max(pitch, 0) / 20)

        score *= boost * yaw_match * pitch_match
        scores[label] = round(score, 4)

    total = sum(scores.values())
    if total > 0:
        scores = {k: round(v / total, 4) for k, v in scores.items()}

    return scores


def top_classroom_emotion(emotion_probs: dict, yaw: float, pitch: float) -> str:
    """返回最匹配的课堂复合情绪标签"""
    scores = classify_classroom_emotion(emotion_probs, yaw, pitch)
    if not scores:
        return "N/A"
    return max(scores, key=scores.get)
