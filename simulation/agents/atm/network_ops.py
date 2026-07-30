"""Deterministic ATM network-operations policy engine.

Covers the 8 "network ops" task types in the fine-tuning dataset: alert
triage, replenishment prioritisation, CIT (cash-in-transit) route planning,
demand forecasting, end-of-day position reporting, month-on-month growth
analysis, share-of-total analysis, and generic site ranking.

Mirrors loan-agent's app/agents/loans/policy.py convention: small pure
dataclasses for inputs/outputs, one deterministic computation function per
task (no I/O, no randomness), and a matching prompt-fact-renderer function
per task that presents the computed numbers as authoritative ground truth
for a downstream LLM to narrate. Every formula below was reverse-engineered
from the worked examples in merged_finetune_train.jsonl / _val.jsonl
(synthetic_atm_ops).
"""
from __future__ import annotations

from dataclasses import dataclass

from .entities import AtmMachine
from .prompts import _rs, render_atm_status

# ── alert triage ──

# Severity policy: P1 = cash-out / hardware down / security; P2 = degraded
# service; P3 = non-customer-impacting; P4 = cosmetic. The (severity, action)
# pairs below and the sort rule (severity asc, then time-open desc) are taken
# directly from the worked examples. DOOR_OPEN sits at P2 by default — it
# only escalates to a security incident once confirmed unscheduled, which is
# exactly what the action text says to go check.
ALERT_POLICY: dict[str, tuple[int, str]] = {
    "DISPENSER_FAULT": (1, "take offline, raise vendor engineer ticket"),
    "COMM_DOWN": (1, "check backup 4G failover, raise ISP ticket"),
    "CASH_OUT": (1, "emergency cash load — escalate for immediate dispatch"),
    "DOOR_OPEN": (2, "verify against the CIT schedule, escalate to security if unscheduled"),
    "UPS_ON_BATTERY": (2, "confirm generator changeover before the UPS drains"),
    "CASH_LOW": (2, "schedule top-up on the next run"),
    "HIGH_REJECT_RATE": (3, "flag the note quality from the last load, plan an unfit-note swap"),
    "PRINTER_PAPER_OUT": (4, "add to the custodian's next routine visit"),
}
# Unseen alert codes default to P3 with a generic runbook action rather than
# crashing the triage — approximation noted since the dataset doesn't cover
# every possible alert type.
_DEFAULT_SEVERITY = (3, "review and action per standard NOC runbook")


@dataclass
class AlertRow:
    atm_id: str
    location: str
    alert_code: str
    description: str
    open_minutes: int


@dataclass
class TriagedAlert:
    atm_id: str
    location: str
    alert_code: str
    description: str
    open_minutes: int
    severity: int   # 1-4, 1 = most urgent
    action: str


@dataclass
class AlertTriageResult:
    triaged: list[TriagedAlert]
    dispatch_tonight: list[str]   # atm_ids at P1


def triage_alerts(alerts: list[AlertRow]) -> AlertTriageResult:
    triaged = []
    for a in alerts:
        severity, action = ALERT_POLICY.get(a.alert_code, _DEFAULT_SEVERITY)
        triaged.append(TriagedAlert(
            atm_id=a.atm_id, location=a.location, alert_code=a.alert_code,
            description=a.description, open_minutes=a.open_minutes,
            severity=severity, action=action,
        ))
    triaged.sort(key=lambda t: (t.severity, -t.open_minutes))
    dispatch_tonight = [t.atm_id for t in triaged if t.severity == 1]
    return AlertTriageResult(triaged=triaged, dispatch_tonight=dispatch_tonight)


def render_alert_triage_facts(r: AlertTriageResult) -> str:
    lines = ["ALERT TRIAGE (authoritative — do not contradict)"]
    for i, t in enumerate(r.triaged, 1):
        lines.append(
            f"- {i}. [P{t.severity}] {t.atm_id} ({t.location}) — {t.alert_code}, "
            f"open {t.open_minutes} min -> {t.action}"
        )
    lines.append(
        "- Dispatch tonight: " + (", ".join(r.dispatch_tonight) if r.dispatch_tonight else "none")
    )
    return "\n".join(lines)


# ── replenishment priority ──

@dataclass
class ReplenishmentCandidate:
    atm_id: str
    location: str
    cash_on_hand: int
    dispense_rate_per_hr: int


@dataclass
class RankedReplenishment:
    atm_id: str
    location: str
    hours_to_empty: float
    at_risk: bool


@dataclass
class ReplenishmentPriorityResult:
    ranked: list[RankedReplenishment]
    window_hours: float
    at_risk_count: int
    first_site: str   # atm_id to load first — always the most urgent overall


def prioritize_replenishment(
    candidates: list[ReplenishmentCandidate], window_hours: float,
) -> ReplenishmentPriorityResult:
    ranked = []
    for c in candidates:
        hours = round(c.cash_on_hand / c.dispense_rate_per_hr, 1) if c.dispense_rate_per_hr > 0 else float("inf")
        ranked.append(RankedReplenishment(
            atm_id=c.atm_id, location=c.location, hours_to_empty=hours,
            at_risk=hours < window_hours,
        ))
    ranked.sort(key=lambda r: r.hours_to_empty)
    at_risk_count = sum(1 for r in ranked if r.at_risk)
    first_site = ranked[0].atm_id if ranked else ""
    return ReplenishmentPriorityResult(
        ranked=ranked, window_hours=window_hours, at_risk_count=at_risk_count, first_site=first_site,
    )


def render_replenishment_facts(r: ReplenishmentPriorityResult) -> str:
    lines = ["REPLENISHMENT PRIORITY (authoritative — do not contradict)"]
    for i, s in enumerate(r.ranked, 1):
        risk = " — AT RISK" if s.at_risk else ""
        lines.append(f"- {i}. {s.atm_id} ({s.location}) — {s.hours_to_empty:.1f} h of cash left{risk}")
    lines.append(
        f"- {r.at_risk_count} site(s) empty inside the {r.window_hours:g}-hour window; "
        f"load {r.first_site} first"
    )
    return "\n".join(lines)


# ── CIT route planning ──

@dataclass
class CitCandidate:
    atm_id: str
    distance_km: float
    cash_left: int
    hours_to_empty: float
    requested_load: int


@dataclass
class CitRoutePlanResult:
    included: list[CitCandidate]     # in load order
    deferred: list[CitCandidate]     # in urgency order
    total_loaded: int
    cap: int
    spare_capacity: int
    sites_included: int
    sites_total: int


def plan_cit_route(candidates: list[CitCandidate], cap: int) -> CitRoutePlanResult:
    """Greedy first-fit by urgency: sort most-urgent-first (hours-to-empty
    ascending, cash-left ascending as the tiebreak — matches the dataset's
    ordering when several sites round to the same hours-to-empty), then walk
    the list loading every site that still fits under the vehicle cash-carry
    cap, skipping (deferring) any that would breach it and continuing to
    check the rest — not stopping at the first site that doesn't fit."""
    ordered = sorted(candidates, key=lambda c: (c.hours_to_empty, c.cash_left))
    included: list[CitCandidate] = []
    deferred: list[CitCandidate] = []
    total = 0
    for c in ordered:
        if total + c.requested_load <= cap:
            included.append(c)
            total += c.requested_load
        else:
            deferred.append(c)
    return CitRoutePlanResult(
        included=included, deferred=deferred, total_loaded=total, cap=cap,
        spare_capacity=cap - total, sites_included=len(included), sites_total=len(candidates),
    )


def render_cit_route_facts(r: CitRoutePlanResult) -> str:
    lines = [
        "CIT ROUTE PLAN (authoritative — do not contradict)",
        f"- Run plan: {r.sites_included} of {r.sites_total} sites, {_rs(r.total_loaded)} of the {_rs(r.cap)} cap",
    ]
    for i, c in enumerate(r.included, 1):
        lines.append(
            f"- {i}. {c.atm_id} — {_rs(c.requested_load)} ({c.distance_km:g} km, "
            f"{c.hours_to_empty:g} h of cash left)"
        )
    if r.deferred:
        lines.append("- Deferred to the next run:")
        for c in r.deferred:
            lines.append(f"    {c.atm_id} — {_rs(c.requested_load)} ({c.hours_to_empty:g} h left)")
    lines.append(f"- Spare capacity on the vehicle: {_rs(r.spare_capacity)}")
    return "\n".join(lines)


# ── demand forecast ──

@dataclass
class DemandForecastInput:
    machine: AtmMachine
    week_dispensed: list[tuple[str, int]]   # (day name, amount), 7 entries
    forecast_day: str
    seasonal_label: str
    seasonal_multiplier: float
    machine_capacity: int


@dataclass
class DemandForecastResult:
    same_day_last_week: int
    weekly_average: int
    day_of_week_factor: float
    seasonal_multiplier: float
    forecast: int
    within_capacity: bool
    topup_needed: int


def forecast_demand(inp: DemandForecastInput) -> DemandForecastResult:
    same_day = next(v for day, v in inp.week_dispensed if day == inp.forecast_day)
    weekly_avg = round(sum(v for _, v in inp.week_dispensed) / 7)
    factor = round(same_day / weekly_avg, 2) if weekly_avg else 0.0
    forecast = round(same_day * inp.seasonal_multiplier)
    within_capacity = forecast <= inp.machine_capacity
    topup = 0 if within_capacity else forecast - inp.machine_capacity
    return DemandForecastResult(
        same_day_last_week=same_day, weekly_average=weekly_avg, day_of_week_factor=factor,
        seasonal_multiplier=inp.seasonal_multiplier, forecast=forecast,
        within_capacity=within_capacity, topup_needed=topup,
    )


def render_demand_forecast_facts(inp: DemandForecastInput, r: DemandForecastResult) -> str:
    lines = [render_atm_status(inp.machine), "", "DEMAND FORECAST (authoritative — do not contradict)"]
    lines.append(f"- Forecast for {inp.forecast_day} at {inp.machine.atm_id}: {_rs(r.forecast)}")
    lines.append(f"- Same day last week: {_rs(r.same_day_last_week)}")
    lines.append(f"- Weekly average: {_rs(r.weekly_average)} (day-of-week factor {r.day_of_week_factor:g})")
    lines.append(f"- Seasonal multiplier ({inp.seasonal_label}): {r.seasonal_multiplier:g}x")
    if r.within_capacity:
        lines.append(f"- Within the machine's {_rs(inp.machine_capacity)} capacity — a single pre-day load covers it")
    else:
        lines.append(
            f"- Exceeds the machine's {_rs(inp.machine_capacity)} capacity — plan a same-day "
            f"top-up of about {_rs(r.topup_needed)} on top of a full load"
        )
    return "\n".join(lines)


# ── end-of-day position report ──

@dataclass
class EodSiteRow:
    atm_id: str
    location: str
    status: str   # ONLINE | OUT_OF_CASH | COMMS_DOWN | OFFLINE_FAULT | ...
    cash_on_hand: int
    dispensed_today: int


@dataclass
class EodActionItem:
    atm_id: str
    location: str
    reason: str


@dataclass
class EodPositionResult:
    availability_pct: float
    online_count: int
    total_count: int
    cash_on_hand_total: int
    dispensed_total: int
    action_items: list[EodActionItem]


LOW_CASH_THRESHOLD = 500_000


def report_eod_position(
    sites: list[EodSiteRow], low_cash_threshold: int = LOW_CASH_THRESHOLD,
) -> EodPositionResult:
    online = [s for s in sites if s.status == "ONLINE"]
    availability = round(len(online) / len(sites) * 100, 1) if sites else 0.0
    cash_total = sum(s.cash_on_hand for s in sites)
    dispensed_total = sum(s.dispensed_today for s in sites)

    # Action items: offline/faulted sites first (original table order), then
    # online-but-low-cash sites (original table order) — matches the dataset,
    # which does NOT sort by alert type, just preserves table order per group.
    offline_items = [
        EodActionItem(
            s.atm_id, s.location,
            "OUT_OF_CASH - emergency load first thing" if s.status == "OUT_OF_CASH"
            else f"{s.status} - engineer dispatch",
        )
        for s in sites if s.status != "ONLINE"
    ]
    low_cash_items = [
        EodActionItem(s.atm_id, s.location, f"only {_rs(s.cash_on_hand)} left - top up on the morning run")
        for s in sites if s.status == "ONLINE" and s.cash_on_hand < low_cash_threshold
    ]
    return EodPositionResult(
        availability_pct=availability, online_count=len(online), total_count=len(sites),
        cash_on_hand_total=cash_total, dispensed_total=dispensed_total,
        action_items=offline_items + low_cash_items,
    )


def render_eod_position_facts(r: EodPositionResult) -> str:
    lines = [
        "END-OF-DAY POSITION (authoritative — do not contradict)",
        f"- Availability: {r.availability_pct}% ({r.online_count} of {r.total_count} sites online)",
        f"- Cash on hand across the zone: {_rs(r.cash_on_hand_total)}",
        f"- Dispensed today: {_rs(r.dispensed_total)}",
        f"- Sites needing action: {len(r.action_items)}",
    ]
    for a in r.action_items:
        lines.append(f"- {a.atm_id} ({a.location}): {a.reason}")
    return "\n".join(lines)


# ── growth analysis (month-on-month) ──

@dataclass
class GrowthInput:
    atm_id: str
    metric_label: str    # e.g. "Off-us withdrawals", "Transaction count"
    old_period: str
    old_value: int
    new_period: str
    new_value: int
    is_money: bool = True


@dataclass
class GrowthResult:
    change: int
    pct_change: float


def analyze_growth(inp: GrowthInput) -> GrowthResult:
    change = inp.new_value - inp.old_value
    pct = round(change / inp.old_value * 100, 1) if inp.old_value else float("inf")
    return GrowthResult(change=change, pct_change=pct)


def _fmt(x: int, is_money: bool) -> str:
    return _rs(x) if is_money else f"{x:,}"


def render_growth_facts(inp: GrowthInput, r: GrowthResult) -> str:
    sign = "+" if r.change >= 0 else ""
    old_s, new_s, change_s = _fmt(inp.old_value, inp.is_money), _fmt(inp.new_value, inp.is_money), _fmt(abs(r.change), inp.is_money)
    return "\n".join([
        "GROWTH ANALYSIS (authoritative — do not contradict)",
        f"- {inp.metric_label} at {inp.atm_id} moved from {old_s} in {inp.old_period} to "
        f"{new_s} in {inp.new_period}",
        f"- Change: {sign}{change_s} ({sign}{r.pct_change}%)",
    ])


# ── share-of-total analysis ──

@dataclass
class ShareRow:
    atm_id: str
    location: str
    value: int


@dataclass
class ShareResult:
    atm_id: str
    location: str
    value: int
    total: int
    share_pct: float
    rank: int
    total_sites: int


def analyze_share(rows: list[ShareRow], target_atm_id: str) -> ShareResult:
    total = sum(r.value for r in rows)
    target = next(r for r in rows if r.atm_id == target_atm_id)
    ordered = sorted(rows, key=lambda r: -r.value)
    rank = next(i for i, r in enumerate(ordered, 1) if r.atm_id == target_atm_id)
    share_pct = round(target.value / total * 100, 1) if total else 0.0
    return ShareResult(
        atm_id=target.atm_id, location=target.location, value=target.value, total=total,
        share_pct=share_pct, rank=rank, total_sites=len(rows),
    )


def render_share_facts(r: ShareResult) -> str:
    return "\n".join([
        "SHARE ANALYSIS (authoritative — do not contradict)",
        f"- {r.atm_id} ({r.location}) dispensed {_rs(r.value)} of the {_rs(r.total)} total",
        f"- Share: {r.share_pct}%, ranking {r.rank} of {r.total_sites} sites",
    ])


# ── generic site ranking ──

@dataclass
class RankingRow:
    atm_id: str
    location: str
    value: float


@dataclass
class RankedSite:
    atm_id: str
    location: str
    value: float
    rank: int


@dataclass
class RankingResult:
    ranked: list[RankedSite]
    top_comment: str


def rank_sites(
    rows: list[RankingRow], metric_label: str, higher_is_worse: bool = False,
) -> RankingResult:
    ordered = sorted(rows, key=lambda r: -r.value)
    ranked = [RankedSite(r.atm_id, r.location, r.value, i) for i, r in enumerate(ordered, 1)]
    top = ranked[0]
    if higher_is_worse:
        comment = (
            f"{top.atm_id} in {top.location} tops the list - for a {metric_label} metric that "
            "is a problem site, not a star, so it goes to the top of the remediation queue."
        )
    else:
        comment = (
            f"{top.atm_id} in {top.location} leads, which makes it the site to protect on cash "
            "availability and uptime."
        )
    return RankingResult(ranked=ranked, top_comment=comment)


def render_ranking_facts(r: RankingResult, is_money: bool = False) -> str:
    lines = ["SITE RANKING (authoritative — do not contradict)"]
    for s in r.ranked:
        val = _fmt(int(s.value), is_money) if float(s.value).is_integer() else f"{s.value:,g}"
        lines.append(f"- {s.rank}. {s.atm_id} ({s.location}) — {val}")
    lines.append(f"- {r.top_comment}")
    return "\n".join(lines)
