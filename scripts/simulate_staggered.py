"""交错处理模拟演示（需求：还原真实效应 + 展示 TWFE 偏差）。

直接运行：``python scripts/simulate_staggered.py``

数据设定
========
* 6 个年度时期 t=0..5；
* 两个处理队列：g=1 与 g=3，各 200 个县；另有 200 个从未处理县；
* 真实处理效应随相对时期线性增长：τ_e = e（e=t-g≥0）；
* y_it = α_i + 0.5·ε_{i,t-1}+ε_it + (t-g)_+。

真实总体 ATT（所有处理后 (g,t) 按处理县数等权）= 1.625。
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from app.did import Row, estimate_did, estimate_twfe


def main() -> None:
    rng = np.random.default_rng(42)
    n, T = 200, 6
    cohorts = {"g1": 1, "g3": 3}
    rows = []
    units = [(f"{g}_{i}", gi) for g, gi in cohorts.items() for i in range(n)]
    units += [(f"never_{i}", None) for i in range(n)]
    for u, g in units:
        alpha = rng.normal(0, 1)
        ar = 0.0
        for t in range(T):
            ar = 0.5 * ar + rng.normal(0, 1)
            effect = (t - g) if (g is not None and t >= g) else 0.0
            rows.append(Row(u, t, alpha + ar + effect, g is not None and t >= g, ()))

    truth = 1.625
    did = estimate_did(rows, control="never_treated")
    twfe = estimate_twfe(rows)

    print(f"真实总体 ATT                 = {truth:.4f}")
    print(f"堆叠 DID（never_treated）ATT = {did['att']:.4f}  (SE={did['std_error']:.4f})")
    print(f"经典 TWFE 回归系数           = {twfe['coef_static']:.4f}  "
          f"(SE={twfe['se_static']:.4f})")
    print(f"TWFE 偏差                    = {twfe['coef_static'] - truth:+.4f}")
    print()
    print("动态效应（事件研究，累积口径）：")
    for e in did["event_study"]:
        tag = " (基准)" if e.get("baseline") else ""
        se = "" if e.get("baseline") else f"  SE={e['std_error']:.3f}"
        print(f"  e={e['relative_period']:+d}: {e['att']:+.3f}{se}{tag}")
    pt = did["pretrend_test"]
    print()
    print(f"处理前平行趋势联合 Wald：stat={pt['statistic']:.3f}, "
          f"df={pt['df']}, p={pt['p_value']:.3f}")


if __name__ == "__main__":
    main()
