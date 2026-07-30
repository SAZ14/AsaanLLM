"""
Cash Pulse — ATM cash tracking & rebalancing agent (framework-agnostic core).

No third-party dependencies.

    from agents.cash_pulse import CashPulse
    cp = CashPulse()
    status = cp.evaluate({"id":"LHR-07","location":"Wapda Town",
                          "capacity":4_000_000,"cash":640_000,"daily_dispense":1_500_000})
    summary = cp.network_summary(atm_list)
    plan    = cp.rebalance_plan(atm_list)   # concrete "move X from A to B" actions
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


def _get(a: dict, *keys, default=0):
    for k in keys:
        if k in a and a[k] is not None:
            return a[k]
    return default


@dataclass
class CashConfig:
    critical_days: float = 1.5     # < this many days of cash -> critical
    low_days: float = 3.0          # < this -> low
    overstock_days: float = 14.0   # > this -> overstocked (idle cash)
    buffer_days: float = 2.5       # target minimum cover to keep on the machine
    refill_target_days: float = 5.0  # top up to this many days of cover
    idle_cost_annual: float = 0.13   # carrying cost of idle cash (rate p.a.)
    dry_loss_factor: float = 0.35    # share of a day's dispense lost when a machine runs dry


class CashPulse:
    def __init__(self, config: CashConfig | None = None):
        self.c = config or CashConfig()

    def evaluate(self, atm: dict) -> dict:
        cfg = self.c
        cash = float(_get(atm, "cash", "current_cash", default=0))
        daily = float(_get(atm, "daily_dispense", "daily", "avg_daily", default=1)) or 1.0
        cap = float(_get(atm, "capacity", "cap", default=cash))
        days = cash / daily
        needed = daily * cfg.buffer_days
        refill_target = min(cap, daily * cfg.refill_target_days)

        status = cls = recommendation = ""
        load = idle = save = 0.0

        if days < cfg.critical_days:
            status, cls = "Critical", "critical"
            load = max(0, refill_target - cash)
            save = daily * cfg.dry_loss_factor
            recommendation = f"Load PKR {load:,.0f} today — runs dry in {days:.1f} days."
        elif days < cfg.low_days:
            status, cls = "Low", "low"
            load = max(0, refill_target - cash)
            save = daily * cfg.dry_loss_factor * 0.4
            recommendation = f"Top up PKR {load:,.0f} within 48h to avoid downtime."
        elif days > cfg.overstock_days:
            status, cls = "Overstocked", "overstocked"
            idle = max(0, cash - needed)
            save = idle * cfg.idle_cost_annual / 12
            recommendation = f"PKR {idle:,.0f} idle ({days:.0f} days cover). Redeploy to a critical site."
        else:
            status, cls = "Healthy", "healthy"
            recommendation = f"Balanced — {days:.1f} days cover, no action."

        return {
            "id": _get(atm, "id", default=None),
            "location": _get(atm, "location", "loc", default=""),
            "status": status,
            "class": cls,                      # critical | low | healthy | overstocked
            "days_to_empty": round(days, 2),
            "cash": int(cash),
            "capacity": int(cap),
            "fill_pct": round(min(100, cash / cap * 100) if cap else 0, 1),
            "recommended_load": int(round(load)),
            "idle_cash": int(round(idle)),
            "daily_loss_avoided": int(round(save)),
            "recommendation": recommendation,
        }

    def evaluate_all(self, atms: list[dict]) -> list[dict]:
        return [self.evaluate(a) for a in atms]

    def network_summary(self, atms: list[dict]) -> dict:
        cfg = self.c
        ev = self.evaluate_all(atms)
        total = sum(e["cash"] for e in ev)
        idle = sum(e["idle_cash"] for e in ev)
        critical = [e for e in ev if e["class"] in ("critical", "low")]
        dry_loss_day = sum(e["daily_loss_avoided"] for e in ev if e["class"] == "critical")
        return {
            "machines": len(ev),
            "network_cash": total,
            "idle_cash": idle,
            "idle_carry_cost_per_day": int(round(idle * cfg.idle_cost_annual / 365)),
            "need_attention": len(critical),
            "downtime_loss_risk_per_day": int(round(dry_loss_day)),
        }

    def rebalance_plan(self, atms: list[dict]) -> list[dict]:
        """Greedily match idle cash at overstocked machines to what critical/low machines need."""
        ev = self.evaluate_all(atms)
        donors = sorted([e for e in ev if e["class"] == "overstocked"],
                        key=lambda e: e["idle_cash"], reverse=True)
        needers = sorted([e for e in ev if e["class"] in ("critical", "low")],
                         key=lambda e: e["days_to_empty"])
        pool = [[d, d["idle_cash"]] for d in donors]  # [donor, remaining]
        moves = []
        for n in needers:
            need = n["recommended_load"]
            for slot in pool:
                if need <= 0:
                    break
                donor, remaining = slot
                if remaining <= 0:
                    continue
                amt = min(remaining, need)
                slot[1] -= amt
                need -= amt
                moves.append({
                    "from": donor["location"], "from_id": donor["id"],
                    "to": n["location"], "to_id": n["id"],
                    "amount": int(round(amt)),
                    "reason": f"{n['location']} at {n['days_to_empty']:.1f} days; "
                              f"{donor['location']} sitting on idle cash.",
                })
        return moves


default_agent = CashPulse()

def evaluate(atm: dict) -> dict:
    return default_agent.evaluate(atm)
