"""人格吸引子层（原则 A/C）：距离度量 + 回拉目标。

吸引子中心/容差/权重来自角色卡；距离长期必须收敛回容差内（P1 验收）。
"""
from .state_dims import DIMS


class PersonaAttractor:

    def __init__(self, persona, convergence_cfg):
        self.entries = persona.attractor_map()
        self.distance_function = convergence_cfg.distance_function
        self.total_weight = sum(e.weight for e in self.entries.values()) or 1.0
        # 参考容差：加权平均。距离收敛目标以它为尺度。
        self.reference_tolerance = (sum(e.weight * e.tolerance for e in self.entries.values())
                                    / self.total_weight)

    def distance(self, state_vector: dict) -> float:
        """加权平均绝对偏差（mad）或加权均方根（rms），量纲与单维容差可比。"""
        devs = []
        for name in DIMS:
            e = self.entries[name]
            devs.append((e.weight, abs(state_vector[name] - e.center)))
        if self.distance_function == "rms":
            return (sum(w * d * d for w, d in devs) / self.total_weight) ** 0.5
        return sum(w * d for w, d in devs) / self.total_weight

    def consistency(self, state_vector: dict) -> float:
        """人格一致性：距离在参考容差内为 1，线性衰减到 0。"""
        ratio = self.distance(state_vector) / self.reference_tolerance
        return max(0.0, 1.0 - ratio)

    def pull(self, state_vector: dict, alpha: float, max_pull: float) -> dict:
        """5.3 回拉：距离越远增益 g 越大，但单步回拉量被 max_pull 封顶。"""
        out = {}
        for name in DIMS:
            e = self.entries[name]
            gap = state_vector[name] - e.center
            g = min(1.0, abs(gap) / e.tolerance)
            p = alpha * (-gap) * g
            out[name] = max(-max_pull, min(max_pull, p))
        return out

    def per_dim_within_tolerance(self, state_vector: dict) -> dict:
        return {n: abs(state_vector[n] - self.entries[n].center) <= self.entries[n].tolerance
                for n in DIMS}
