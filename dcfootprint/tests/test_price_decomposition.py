"""Price decomposition (routing/agents.py) must reproduce the central LP (routing/lyapunov._lp_month).

1. clear_market vs linprog on random instances, including deliberate cost ties: same objective,
   pool cleared, bounds respected.
2. Full Q3 simulation on the canonical account: every controller policy x 10 seeds x legal on/off,
   agents vs central LP — identical penalty, carbon, scarcity and monthly basin queues.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dcfootprint.routing import lyapunov
from dcfootprint.routing.agents import clear_market

ROOT = Path(__file__).resolve().parents[2]
ACCOUNT = ROOT / "dcfootprint" / "results" / "round5_final" / "outputs" / "account_facility_month.parquet"
RTOL = 1e-9


@pytest.mark.parametrize("seed", range(50))
def test_clear_market_matches_lp(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 40))
    costs = np.round(rng.uniform(0, 5, n), int(rng.integers(0, 3)))    # coarse rounding -> ties
    head = rng.uniform(0, 10, n) * (rng.uniform(size=n) > 0.1)
    pool = float(rng.uniform(0, 1.2) * head.sum())
    x, _ = clear_market(costs, head, pool)
    x_lp = lyapunov._lp_month(costs, head, pool)
    assert np.all(x >= -1e-12) and np.all(x <= head + 1e-9)
    assert x.sum() == pytest.approx(min(pool, head.sum()), rel=1e-9, abs=1e-9)
    assert costs @ x == pytest.approx(costs @ x_lp, rel=1e-9, abs=1e-9)


@pytest.fixture(scope="module")
def prepared():
    if not (ROOT / "data" / "gem").is_dir():
        pytest.skip("needs the raw inputs in data/ (the routing setup validates their paths)")
    acct = pd.read_parquet(ACCOUNT)
    return {legal: lyapunov.prepare(acct, legal=legal) for legal in (True, False)}


@pytest.mark.parametrize("legal", [True, False])
@pytest.mark.parametrize("policy", ["greedy", "lyapunov", "lyapunov_pf"])
def test_agents_match_central_lp(prepared, legal, policy):
    P = prepared[legal]
    for seed in range(10):
        a = lyapunov.simulate(P, policy, seed=seed, solver="price_decomposition")
        c = lyapunov.simulate(P, policy, seed=seed, solver="central_lp")
        for k in ["penalty", "carbon_tco2", "scarcity_m3eq", "peak_basin_queue_m3", "end_backlog_m3"]:
            assert a[k] == pytest.approx(c[k], rel=RTOL, abs=1e-9), (policy, legal, seed, k)
        np.testing.assert_allclose(a["_qpeak_by_basin"], c["_qpeak_by_basin"], rtol=RTOL, atol=1e-6)
