"""L6 / Q3 — price decomposition of the drift-plus-penalty step as a multi-agent allocation.

Each month t the controller must place a flexible pool P(t) across facilities:

    min_x  sum_f c_f(t) x_f   s.t.  sum_f x_f = P(t),  0 <= x_f <= h_f(t)
    c_f(t) = V [ k_f(t) CI_z(f)(t) / c_bar + lambda s_f(t) / s_bar ] + (D_b(f)(t) / R_bar_b) w_f(t) / w_bar

The problem is separable in f except for the single demand constraint, so it decomposes by
prices (the dual / market reading of drift-plus-penalty, Neely 2010 ch. 3-4):

  GridAgent(z)      posts the carbon price  CI_z(t)   (forecast for the controller, true for
                    perfect foresight) — gCO2/kWh of zone z
  BasinAgent(b)     posts the water price   D_b(t) / R_bar_b  (its overdraft backlog in months
                    of the basin's own datacenter draw) and updates its queue after allocation:
                    D_b(t+1) = max(D_b(t) + W_b(t) - R_b(t), 0)
  FacilityAgent(f)  turns the prices it sees into its unit cost c_f(t) (its own PUE, legal
                    carbon share, scarcity and water intensities) and, given a demand price mu,
                    offers  x_f(mu) = h_f if c_f < mu, 0 if c_f > mu  (indifferent at c_f = mu)
  Coordinator       bisects mu until offered supply clears P(t); agents indifferent at the
                    clearing price share the residual in proportion to headroom

No agent sees another agent's costs; only prices and quantities cross agent boundaries.
With linear costs the clearing allocation is an optimum of the central LP (tests compare
cost, basin loads and totals against routing.lyapunov._lp_month).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GridAgent:
    zone: str
    ci: np.ndarray                                   # (12,) gCO2/kWh the agent posts

    def price(self, t: int) -> float:
        return float(self.ci[t])


@dataclass
class BasinAgent:
    basin_id: int
    budget: np.ndarray                               # (12,) litres available this month
    scale: float                                     # R_bar: mean monthly DC draw (price normaliser)
    queue: float = 0.0
    history: list = field(default_factory=list)

    def price(self) -> float:
        return self.queue / self.scale

    def settle(self, t: int, load_l: float) -> None:
        self.queue = max(self.queue + load_l - self.budget[t], 0.0)
        self.history.append(self.queue)


@dataclass
class FacilityAgent:
    facility_id: str
    zone: int                                        # index into grid agents
    basin: int                                       # index into basin agents
    k_carbon: np.ndarray                             # (12,) tCO2/MWh-IT per gCO2/kWh (PUE, legal share)
    u_scarcity: np.ndarray                           # (12,) L-eq / MWh-IT
    u_onsite: np.ndarray                             # (12,) L / MWh-IT (scope-1: the basin draw)
    banned: bool

    def unit_cost(self, t, carbon_price, water_price, V, lam, norms, use_queue) -> float:
        c_bar, s_bar, w_bar = norms
        c = V * (self.k_carbon[t] * carbon_price / c_bar + lam * self.u_scarcity[t] / s_bar)
        if use_queue:
            c += water_price * self.u_onsite[t] / w_bar
        return c


def clear_market(costs: np.ndarray, head: np.ndarray, pool: float, tol: float = 1e-12,
                 max_iter: int = 200) -> tuple[np.ndarray, float]:
    """Coordinator: find demand price mu with sum_f x_f(mu) = pool. Returns (x, mu)."""
    pool = min(pool, float(head.sum()))
    if pool <= 0:
        return np.zeros_like(head), float("nan")
    lo, hi = float(costs.min()) - 1.0, float(costs.max()) + 1.0
    offered = lambda mu: float(head[costs < mu].sum())
    for _ in range(max_iter):                        # offered(mu) is non-decreasing in mu
        mid = 0.5 * (lo + hi)
        if offered(mid) > pool:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol * max(1.0, abs(hi)):
            break
    # invariant: offered(lo) <= pool < offered(hi) -> the marginal agents are those in [lo, hi)
    mu = hi
    below = costs < lo
    marginal = (costs >= lo) & (costs < hi)
    x = np.where(below, head, 0.0)
    residual = pool - x.sum()
    if residual > 0 and head[marginal].sum() > 0:
        x[marginal] = residual * head[marginal] / head[marginal].sum()
    return x, mu


def run_month(t: int, facilities: list[FacilityAgent], grids: list[GridAgent], basins: list[BasinAgent],
              head: np.ndarray, pool: float, V: float, lam: float, norms: tuple, use_queue: bool) -> tuple[np.ndarray, float]:
    """One allocation round: agents post prices, facilities price themselves, coordinator clears."""
    carbon = [g.price(t) for g in grids]
    water = [b.price() for b in basins]
    costs = np.array([f.unit_cost(t, carbon[f.zone], water[f.basin], V, lam, norms, use_queue)
                      for f in facilities])
    return clear_market(costs, head, pool)
