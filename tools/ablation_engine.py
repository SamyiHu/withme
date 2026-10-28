# -*- coding: utf-8 -*-
"""引擎机制消融：弛豫项 vs 吸引子回拉，各自对"偏离收敛"的贡献。

条件：
  full          弛豫 + 吸引子（默认实现）
  no_attractor  仅弛豫（吸引子回拉置零）
  no_relax      仅吸引子（decay_tau 置极大，弛豫项≈0）

指标：
  t_ref  回到加权参考容差（≈0.2553）所需时间
  t_dim  所有维度都回到各自容差内所需时间
  gaps   到达 t_ref 时刻各维的残余偏差
输出：reports/ablation_engine.json
"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import load_config, load_persona          # noqa: E402
from src.state_dims import DIMS, clamp_vector            # noqa: E402
from src.state_engine import StateEngine                 # noqa: E402

STEP = 300.0
T0 = 1790000000.0
MAX_STEPS = 500
RUNS = 6


class _NoPull:
    """吸引子回拉置零，其余接口透传。"""

    def __init__(self, real):
        self._real = real
        self.reference_tolerance = real.reference_tolerance

    def pull(self, sv, alpha, max_pull):
        return {name: 0.0 for name in DIMS}

    def distance(self, sv):
        return self._real.distance(sv)


def build(mode):
    cfg = load_config(os.path.join(ROOT, "config", "config.yaml"))
    persona = load_persona(os.path.join(ROOT, "config", "persona.yaml"))
    cfg.chaos.epsilon_base = 0.05
    if mode == "no_relax":
        cfg.convergence.decay_tau_seconds = 1e12
    eng = StateEngine(cfg, persona)
    real = eng.attractor
    if mode == "no_attractor":
        eng.attractor = _NoPull(real)
    return eng, real


def run(mode):
    eng, real = build(mode)
    ref = real.reference_tolerance
    tol = {n: real.entries[n].tolerance for n in DIMS}
    center = {n: real.entries[n].center for n in DIMS}

    t_ref_all, t_dim_all, gap_at_ref = [], [], []
    for rep in range(RUNS):
        row = eng.new_row(T0 + rep * 173)
        for n in DIMS:
            row["state_vector"][n] = center[n] + 0.45
        row["state_vector"] = clamp_vector(row["state_vector"])
        t = row["last_tick"]
        t_ref = t_dim = None
        for step in range(1, MAX_STEPS + 1):
            t += STEP
            row = eng.lazy_evolve(row, t)
            sv = row["state_vector"]
            if t_ref is None and real.distance(sv) <= ref:
                t_ref = step * STEP / 3600.0
                gap_at_ref.append({n: round(abs(sv[n] - center[n]), 4) for n in DIMS})
            if t_dim is None and all(abs(sv[n] - center[n]) <= tol[n] for n in DIMS):
                t_dim = step * STEP / 3600.0
                break
        t_ref_all.append(t_ref if t_ref is not None else 99.0)
        t_dim_all.append(t_dim if t_dim is not None else 99.0)

    return {
        "mode": mode,
        "reference_tolerance": round(ref, 4),
        "t_ref_hours": {"median": round(statistics.median(t_ref_all), 3),
                        "min": min(t_ref_all), "max": max(t_ref_all)},
        "t_dim_hours": {"median": round(statistics.median(t_dim_all), 3),
                        "min": min(t_dim_all), "max": max(t_dim_all)},
        "gap_at_ref_sample": gap_at_ref[0] if gap_at_ref else {},
        "per_dim_tolerance": tol,
    }


def main():
    out = {"step_seconds": STEP, "epsilon": 0.05, "start_deviation": 0.45,
           "runs_per_mode": RUNS, "modes": {}}
    for mode in ("full", "no_attractor", "no_relax"):
        out["modes"][mode] = run(mode)
        m = out["modes"][mode]
        print(f"{mode:13s} t_ref={m['t_ref_hours']['median']:6.3f}h "
              f"t_dim={m['t_dim_hours']['median']:6.3f}h")
        print("   gaps@ref:", m["gap_at_ref_sample"])
    path = os.path.join(ROOT, "reports", "ablation_engine.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("written", path)


if __name__ == "__main__":
    main()
