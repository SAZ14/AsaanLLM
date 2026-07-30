"""
Credit Sentinel — instant-loan decisioning agent (framework-agnostic core).

No third-party dependencies. Import and call from Flask, FastAPI, Django,
a script, a queue worker, or your LLM pipeline.

    from agents.credit_sentinel import CreditSentinel
    agent = CreditSentinel()
    result = agent.assess({
        "name": "Bilal Ahmed", "area": "Johar Town",
        "income": 88000, "balance": 6200, "balance_trend": -1,
        "account_age_months": 31, "existing_exposure": 1,
        "repayment_pct": 91, "missed_payments": 0, "defaulted": False,
    })
    # -> {"decision": "Approve · Instant Loan", "risk": 24, "offer": 140000, ...}
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class CreditPolicy:
    """Tune these to match your real product. Nothing here is hard-coded elsewhere."""
    # factor weights (must sum to ~1.0); repayment dominates by design
    w_repayment: float = 0.42
    w_exposure: float = 0.16
    w_balance: float = 0.12
    w_relation: float = 0.15
    w_income: float = 0.15
    # hard gates
    min_repayment_pct: float = 55.0          # below this -> decline
    approve_risk_cutoff: int = 42            # risk <= this -> eligible, else review
    strain_balance_income_ratio: float = 0.15  # below this (or falling balance) = "low on cash"
    # offer sizing
    instant_income_multiple_high_risk: float = 1.6
    instant_income_multiple_low_risk: float = 2.2
    low_risk_threshold: int = 28
    standard_income_multiple: float = 1.4
    offer_min: int = 20_000
    offer_max: int = 600_000
    # markup band (% p.a.)
    markup_min: float = 24.0
    markup_max: float = 42.0
    tenure_months_min: int = 3
    tenure_months_max: int = 12


# canonical input keys with sensible aliases so you can feed messy data
_ALIASES = {
    "income": ("income", "monthly_income"),
    "balance": ("balance", "avg_balance", "current_balance"),
    "balance_trend": ("balance_trend", "bal_trend", "balTrend"),
    "account_age_months": ("account_age_months", "age_months", "ageM", "tenure_months"),
    "existing_exposure": ("existing_exposure", "exposure", "open_loans"),
    "repayment_pct": ("repayment_pct", "repay", "on_time_pct"),
    "missed_payments": ("missed_payments", "missed"),
    "defaulted": ("defaulted", "def", "is_defaulter"),
}


def _get(c: dict, key: str, default: Any = 0) -> Any:
    for alias in _ALIASES.get(key, (key,)):
        if alias in c and c[alias] is not None:
            return c[alias]
    return default


class CreditSentinel:
    def __init__(self, policy: CreditPolicy | None = None):
        self.p = policy or CreditPolicy()

    # ---- individual factor scores (0..100, higher = healthier) ----
    def _factors(self, c: dict) -> dict[str, float]:
        income = float(_get(c, "income", 1)) or 1.0
        balance = float(_get(c, "balance", 0))
        trend = int(_get(c, "balance_trend", 0))
        age_m = float(_get(c, "account_age_months", 0))
        exposure = int(_get(c, "existing_exposure", 0))
        repay = float(_get(c, "repayment_pct", 0))
        missed = int(_get(c, "missed_payments", 0))
        defaulted = bool(_get(c, "defaulted", False))

        f_repay = 8.0 if defaulted else _clamp(repay - missed * 6, 0, 100)
        f_exposure = _clamp(100 - exposure * 24, 0, 100)
        trend_adj = -12 if trend < 0 else (10 if trend > 0 else 0)
        f_balance = _clamp(min(100, (balance / income) * 140) + trend_adj, 0, 100)
        f_relation = _clamp(min(100, age_m / 72 * 100), 0, 100)
        f_income = _clamp(min(100, income / 120000 * 100), 0, 100)
        return {
            "repayment_history": round(f_repay, 1),
            "existing_exposure": round(f_exposure, 1),
            "balance_stability": round(f_balance, 1),
            "relationship_length": round(f_relation, 1),
            "income_strength": round(f_income, 1),
        }

    def _markup(self, risk: int) -> float:
        return round(_clamp(22 + risk * 0.45, self.p.markup_min, self.p.markup_max), 1)

    def assess(self, customer: dict) -> dict:
        """Return a full decision object for one customer. Pure & deterministic."""
        p = self.p
        f = self._factors(customer)
        health = (f["repayment_history"] * p.w_repayment
                  + f["existing_exposure"] * p.w_exposure
                  + f["balance_stability"] * p.w_balance
                  + f["relationship_length"] * p.w_relation
                  + f["income_strength"] * p.w_income)
        risk = int(round(_clamp(100 - health, 1, 99)))

        income = float(_get(customer, "income", 1)) or 1.0
        balance = float(_get(customer, "balance", 0))
        trend = int(_get(customer, "balance_trend", 0))
        repay = float(_get(customer, "repayment_pct", 0))
        missed = int(_get(customer, "missed_payments", 0))
        defaulted = bool(_get(customer, "defaulted", False))
        strained = (balance / income) < p.strain_balance_income_ratio or trend < 0

        decision, cls, offer, markup, instant, reasons = "", "", 0, 0.0, False, []

        if defaulted or repay < p.min_repayment_pct:
            decision, cls = "Decline", "decline"
            reasons = [
                ("bad", "On record as a previous defaulter — hard block.") if defaulted
                else ("bad", f"Repayment reliability below policy floor ({p.min_repayment_pct:.0f}%)."),
                ("bad", f"{missed} missed instalments in the last 12 months.") if missed > 2
                else ("warn", "Weak relationship history with the bank."),
                ("warn", "Not eligible for instant credit; route to recovery / counselling."),
            ]
        elif risk <= p.approve_risk_cutoff and strained:
            decision, cls, instant = "Approve · Instant Loan", "approve", True
            mult = p.instant_income_multiple_low_risk if risk < p.low_risk_threshold else p.instant_income_multiple_high_risk
            offer = round(_clamp(income * mult, p.offer_min, p.offer_max) / 1000) * 1000
            markup = self._markup(risk)
            reasons = [
                ("ok", f"Excellent repayment history ({repay:.0f}% on-time)."),
                ("ok", "Low balance vs income signals a genuine short-term need."),
                ("info", f"Offer sized to income; markup {markup}% reflects assessed risk."),
            ]
        elif risk <= p.approve_risk_cutoff:
            decision, cls = "Approve · Standard", "approve"
            offer = round(_clamp(income * p.standard_income_multiple, p.offer_min, p.offer_max) / 1000) * 1000
            markup = round(_clamp(20 + risk * 0.4, 20, 34), 1)
            reasons = [
                ("ok", "Clean history and low existing exposure."),
                ("info", "Ample balance — instant-loan propensity is low."),
                ("info", "Pre-approved limit available on request."),
            ]
        else:
            decision, cls = "Manual Review", "review"
            reasons = [
                ("warn", "Risk score in the review band."),
                ("warn", f"{missed} missed instalment(s) recently.") if missed > 0
                else ("info", "Thin relationship history."),
                ("info", "Request income verification before approval."),
            ]

        return {
            "decision": decision,
            "class": cls,                     # approve | decline | review
            "eligible": cls == "approve",
            "instant": instant,               # True = auto-offer this customer
            "risk": risk,                     # 1..99, lower = safer
            "offer": int(offer),              # recommended amount (PKR), 0 if none
            "markup_pct": markup,             # annual %
            "tenure_months": [p.tenure_months_min, p.tenure_months_max],
            "factors": f,                     # per-factor 0..100 health scores
            "reasons": [{"level": lv, "text": tx} for lv, tx in reasons],
        }

    def assess_batch(self, customers: list[dict]) -> list[dict]:
        return [self.assess(c) for c in customers]

    def portfolio_summary(self, customers: list[dict]) -> dict:
        results = self.assess_batch(customers)
        approve = [r for r in results if r["class"] == "approve"]
        return {
            "total": len(results),
            "eligible": len(approve),
            "instant_offers": sum(1 for r in results if r["instant"]),
            "declined": sum(1 for r in results if r["class"] == "decline"),
            "review": sum(1 for r in results if r["class"] == "review"),
            "avg_risk": round(sum(r["risk"] for r in results) / max(1, len(results))),
            "potential_book": sum(r["offer"] for r in approve),
        }


# convenience singleton
default_agent = CreditSentinel()

def assess(customer: dict) -> dict:
    return default_agent.assess(customer)
