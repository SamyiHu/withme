"""状态向量的维度与取值范围（全项目唯一权威定义）。"""

# 顺序即向量顺序
DIMS = ["energy", "social_need", "curiosity", "care", "expression", "arousal", "valence"]

# valence 情绪效价取 [-1,1]，其余取 [0,1]
RANGES = {
    "energy": (0.0, 1.0),
    "social_need": (0.0, 1.0),
    "curiosity": (0.0, 1.0),
    "care": (0.0, 1.0),
    "expression": (0.0, 1.0),
    "arousal": (0.0, 1.0),
    "valence": (-1.0, 1.0),
}

DIM_CN = {
    "energy": "精力",
    "social_need": "社交需求",
    "curiosity": "好奇",
    "care": "关心",
    "expression": "表达欲",
    "arousal": "唤醒度",
    "valence": "情绪效价",
}


def clamp_dim(name: str, value: float) -> float:
    lo, hi = RANGES[name]
    return max(lo, min(hi, value))


def clamp_vector(state: dict) -> dict:
    return {k: clamp_dim(k, v) for k, v in state.items()}
