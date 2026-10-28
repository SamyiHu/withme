"""状态演化层：多尺度惰性演化 + 事件脉冲。

- 多尺度：fast(弛豫粒度，闭式计算) / seed_tau(噪声步长) / slow(昼夜节律)。
- 惰性：只在被唤醒时按 Δt 补算，不要求后台常驻；内部按 300s 噪声步推进，
  步数过多时自动放粗步长（弛豫项是闭式的，放粗不破坏收敛）。
- 演化每步：弛豫向基线（energy 带昼夜调制）+ 孤独累积 + 有界噪声 + 吸引子回拉。
"""
import math
import time
from copy import deepcopy

from . import chaos
from .config import AppConfig, Persona
from .persona_attractor import PersonaAttractor
from .state_dims import DIMS, clamp_vector

MAX_STEPS = 2000


class StateEngine:

    def __init__(self, config: AppConfig, persona: Persona):
        self.cfg = config
        self.persona = persona
        self.attractor = PersonaAttractor(persona, config.convergence)

    # ---------------- 行结构 ----------------

    def new_row(self, now: float) -> dict:
        return {
            "state_vector": dict(self.persona.baseline_state),
            "last_tick": now,
            "noise_prev": None,
            "last_user_interaction": now,
            "relation_state": {
                "affinity": 0.5, "trust": 0.5, "familiarity": 0.3,
                "boundaries": list(self.persona.boundaries),
                "recent_active_effect": "尚无记录",
            },
            "last_active_time_by_level": {},
            "active_count_today": {"date": self._local_date(now), "L1": 0, "L2": 0, "L3": 0},
            "last_active_effect": "尚无记录",
            "cooldown_multiplier": {},  # intent_type -> 倍率（拒绝后翻倍，见 feedback）
        }

    @staticmethod
    def _local_date(t: float) -> str:
        return time.strftime("%Y-%m-%d", time.localtime(t))

    @staticmethod
    def _local_hour(t: float) -> float:
        tm = time.localtime(t)
        return tm.tm_hour + tm.tm_min / 60.0

    # ---------------- 惰性演化 ----------------

    def lazy_evolve(self, row: dict, now: float, *, memory_tension_hash: int = 0,
                    relation_hash: int = 0, last_action: str = "") -> dict:
        """按 Δt 从 last_tick 补算到 now。确定性：同输入必同输出。"""
        row = deepcopy(row)
        dt = now - row["last_tick"]
        if dt <= 0:
            row["last_tick"] = max(row["last_tick"], now)
            return row

        step = float(self.cfg.chaos.seed_tau)
        n_steps = int(math.ceil(dt / step))
        if n_steps > MAX_STEPS:
            n_steps = MAX_STEPS
            step = dt / n_steps

        conv = self.cfg.convergence
        alpha_step = min(conv.alpha * step / self.cfg.chaos.seed_tau, 0.5)
        sv = row["state_vector"]
        prev_vec = row.get("noise_prev")
        t = row["last_tick"]
        hour_cache = {}

        for k in range(n_steps):
            t_k = t + step * (k + 1)
            seed = chaos.make_seed(self.persona.role_id, t_k, self.cfg.chaos.seed_tau,
                                   memory_tension_hash, relation_hash, last_action)
            delta, prev_vec = chaos.delta_noise(
                seed, self.cfg.chaos.epsilon_base, DIMS,
                prev_vec=prev_vec, smoothing=self.cfg.chaos.noise_smoothing)

            hour = self._local_hour(t_k)
            if hour not in hour_cache:
                # slow 尺度：昼夜节律调制 energy 弛豫目标，峰值约在 14:00
                circadian = 0.75 + 0.25 * math.sin((hour - 8.0) / 24.0 * 2.0 * math.pi)
                lonely_hours = max(0.0, (t_k - row["last_user_interaction"]) / 3600.0)
                lonely_offset = min(
                    self.cfg.motivation.loneliness_rate_per_hour * lonely_hours,
                    self.cfg.motivation.loneliness_cap - self.persona.baseline_state["social_need"])
                hour_cache[hour] = (circadian, lonely_offset)
            circadian, lonely_offset = hour_cache[hour]

            relax_coeff = 1.0 - math.exp(-step / conv.decay_tau_seconds)
            for i, name in enumerate(DIMS):
                target = self.persona.baseline_state[name]
                if name == "energy":
                    target = target * circadian
                elif name == "social_need":
                    target = target + lonely_offset
                sv[name] += (target - sv[name]) * relax_coeff + delta[i]

            pull = self.attractor.pull(sv, alpha_step, conv.max_pull)
            for name in DIMS:
                sv[name] += pull[name]
            sv.update(clamp_vector(sv))

        row["state_vector"] = sv
        row["noise_prev"] = prev_vec
        row["last_tick"] = now
        return row

    # ---------------- 事件脉冲 ----------------

    IMPULSES = {
        "user_message": {"energy": 0.03, "social_need": -0.18, "arousal": 0.08,
                         "valence": 0.02, "curiosity": 0.02},
        "agent_proactive_sent": {"social_need": -0.10, "expression": -0.05},
        "user_positive_reply": {"valence": 0.08, "arousal": 0.05, "social_need": -0.05},
        "user_negative_reply": {"valence": -0.15, "arousal": 0.10, "social_need": -0.05},
        "user_no_reply": {"valence": -0.05},
        # 共鸣/冲撞（拟人反馈回路）：对方回复与我上一条的内心立场同频/顶撞
        "resonate": {"valence": 0.10, "care": 0.05, "arousal": 0.03, "expression": 0.04},
        "clash": {"valence": -0.12, "arousal": 0.12, "care": -0.05},
        "clash_streak": {"valence": -0.18, "arousal": 0.15, "care": -0.08,
                         "expression": 0.05},   # 连续被顶：真来火了，话变冲
    }

    def apply_event_impulse(self, row: dict, kind: str) -> dict:
        row = deepcopy(row)
        delta = self.IMPULSES.get(kind)
        if delta is None:
            raise ValueError(f"未知事件脉冲: {kind}")
        sv = row["state_vector"]
        for name, d in delta.items():
            sv[name] += d
        sv.update(clamp_vector(sv))
        return row
