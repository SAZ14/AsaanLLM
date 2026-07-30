"""Deterministic ATM customer-money policy engine.

Covers the 9 customer-facing "money" task types in the fine-tuning dataset:
ATM fee calculation, ATM recommendation, cash advance assessment, cash
affordability check, monthly ATM cost summary, spend pattern summary, travel
cash planning, plus the two ops-side sizing tasks cash carrying cost and
surge capacity planning.

Mirrors loan-agent's app/agents/loans/policy.py convention: small pure
dataclasses for inputs/outputs, one deterministic computation function per
task (no I/O, no randomness), and a matching prompt-fact-renderer function
per task that presents the computed numbers as authoritative ground truth
for a downstream LLM to narrate. Every formula below was reverse-engineered
from the worked examples in merged_finetune_train.jsonl / _val.jsonl
(synthetic_atm_customer / synthetic_atm_ops).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .entities import AtmMachine, CustomerAtmProfile, Transaction
from .prompts import _rs

# ── ATM fee calculation ──


@dataclass
class AtmFeeParams:
    free_offus_per_month: int
    offus_already_made: int
    planned_offus_count: int
    fee_per_txn: float
    fed_pct: float          # Federal Excise Duty, e.g. 0.16 for 16%
    per_txn_limit: int      # from the customer's profile, for the alternative
    amount_needed: int      # total cash needed this month, for the alternative


@dataclass
class AtmFeeResult:
    free_left: int
    chargeable: int
    fee: int
    fed: int
    total: int
    alt_count: int          # withdrawals needed at the per-txn limit instead
    alt_chargeable: int
    alt_fee: int
    alt_fed: int
    alt_total: int
    savings: int
    cheaper_route_available: bool


def calculate_atm_fee(p: AtmFeeParams) -> AtmFeeResult:
    free_left = max(0, p.free_offus_per_month - p.offus_already_made)
    chargeable = max(0, p.planned_offus_count - free_left)
    fee = round(chargeable * p.fee_per_txn)
    fed = round(fee * p.fed_pct)
    total = fee + fed

    alt_count = math.ceil(p.amount_needed / p.per_txn_limit) if p.per_txn_limit > 0 else p.planned_offus_count
    alt_chargeable = max(0, alt_count - free_left)
    alt_fee = round(alt_chargeable * p.fee_per_txn)
    alt_fed = round(alt_fee * p.fed_pct)
    alt_total = alt_fee + alt_fed

    cheaper = alt_total < total
    return AtmFeeResult(
        free_left=free_left, chargeable=chargeable, fee=fee, fed=fed, total=total,
        alt_count=alt_count, alt_chargeable=alt_chargeable, alt_fee=alt_fee, alt_fed=alt_fed,
        alt_total=alt_total, savings=(total - alt_total) if cheaper else 0,
        cheaper_route_available=cheaper,
    )


def render_atm_fee_facts(r: AtmFeeResult) -> str:
    lines = [
        "ATM FEE CALCULATION (authoritative — do not contradict)",
        f"- Free allowance left this month: {r.free_left}",
        f"- Chargeable withdrawals: {r.chargeable} x fee = {_rs(r.fee)}"
        + (f" + FED {_rs(r.fed)}" if r.fed else ""),
        f"- Total charge: {_rs(r.total)}",
    ]
    if r.cheaper_route_available:
        lines.append(
            f"- Cheaper route: {r.alt_count} larger withdrawal(s) at the per-transaction limit "
            f"costs {_rs(r.alt_total)}, saving {_rs(r.savings)}"
        )
    else:
        lines.append("- Fewer, larger withdrawals would not reduce the cost here")
    return "\n".join(lines)


# ── ATM recommendation ──


@dataclass
class NearbyAtm:
    atm_id: str
    bank: str
    location: str
    distance_km: float
    status: str            # ONLINE | OFFLINE | OUT_OF_CASH | COMMS_DOWN
    cash_on_hand: int
    denominations: list[int]
    queue: int


@dataclass
class AtmRecommendationParams:
    customer: CustomerAtmProfile
    amount_needed: int
    off_us_charge: float
    candidates: list[NearbyAtm]


@dataclass
class AtmRecommendationResult:
    recommended: NearbyAtm | None
    backup: NearbyAtm | None
    dispense_breakdown: list[tuple[int, int]]   # (denom, count)
    cost_to_customer: int
    survivors_count: int
    candidates_count: int


def _dispense_breakdown(amount: int, denominations: list[int]) -> list[tuple[int, int]]:
    """Greedy largest-denomination-first fill — matches every worked example,
    which always dispenses off a single (usually the largest) denomination."""
    breakdown: list[tuple[int, int]] = []
    remaining = amount
    for denom in sorted(denominations, reverse=True):
        if denom <= 0:
            continue
        count = remaining // denom
        if count > 0:
            breakdown.append((denom, count))
            remaining -= denom * count
    return breakdown


def recommend_atm(p: AtmRecommendationParams) -> AtmRecommendationResult:
    survivors = [
        c for c in p.candidates
        if c.status == "ONLINE" and c.cash_on_hand >= p.amount_needed
    ]

    def cost(c: NearbyAtm) -> int:
        return 0 if c.bank == p.customer.bank else round(p.off_us_charge)

    ranked = sorted(survivors, key=lambda c: (cost(c), c.distance_km, c.queue))
    best = ranked[0] if ranked else None
    backup = ranked[1] if len(ranked) > 1 else None
    breakdown = _dispense_breakdown(p.amount_needed, best.denominations) if best else []

    return AtmRecommendationResult(
        recommended=best, backup=backup, dispense_breakdown=breakdown,
        cost_to_customer=cost(best) if best else 0,
        survivors_count=len(survivors), candidates_count=len(p.candidates),
    )


def render_atm_recommendation_facts(r: AtmRecommendationResult) -> str:
    if r.recommended is None:
        return (
            "ATM RECOMMENDATION (authoritative — do not contradict)\n"
            f"- None of the {r.candidates_count} nearby ATMs can serve this request right now"
        )
    b = r.recommended
    dispense = ", ".join(f"{n} x Rs {d:,}" for d, n in r.dispense_breakdown)
    lines = [
        "ATM RECOMMENDATION (authoritative — do not contradict)",
        f"- Go to: {b.atm_id} ({b.location}, {b.distance_km:g} km) — {b.status}, "
        f"Rs {b.cash_on_hand:,} loaded, queue {b.queue}",
        f"- Dispenses as: {dispense}",
        f"- Cost to customer: {_rs(r.cost_to_customer)}",
        f"- {r.survivors_count} of {r.candidates_count} nearby ATMs can serve this request",
    ]
    if r.backup:
        lines.append(f"- Backup: {r.backup.atm_id} ({r.backup.location}, {r.backup.distance_km:g} km)")
    return "\n".join(lines)


# ── cash advance assessment ──


@dataclass
class CashAdvanceParams:
    customer: CustomerAtmProfile
    atm: AtmMachine
    requested_amount: int
    fee_pct: float
    fee_min: int
    monthly_markup_pct: float
    days_held: int


@dataclass
class CashAdvanceResult:
    sub_limit_remaining: int
    granted_amount: int
    fee: int
    markup: int
    total_cost: int
    pct_of_cash: float
    binding_constraint: str


def assess_cash_advance(p: CashAdvanceParams) -> CashAdvanceResult:
    c = p.customer
    sub_limit_remaining = round(c.credit_line * c.cash_advance_sub_limit_pct - c.cash_advance_drawn)

    constraints = {
        "requested amount": p.requested_amount,
        "per-transaction ATM limit": c.per_txn_limit,
        "daily ATM limit": c.daily_limit,
        "cash advance sub-limit remaining": sub_limit_remaining,
        "cash loaded in the machine": p.atm.usable_cash,
    }
    binding_constraint = min(constraints, key=lambda k: constraints[k])
    raw_max = max(0, constraints[binding_constraint])

    denoms = p.atm.usable_denominations
    smallest_denom = min(denoms) if denoms else 1
    granted = (raw_max // smallest_denom) * smallest_denom if smallest_denom > 0 else raw_max

    fee = round(max(granted * p.fee_pct, p.fee_min)) if granted > 0 else 0
    markup = round(granted * p.monthly_markup_pct * p.days_held / 30)
    total_cost = fee + markup
    pct_of_cash = round(total_cost / granted * 100, 1) if granted > 0 else 0.0

    return CashAdvanceResult(
        sub_limit_remaining=sub_limit_remaining, granted_amount=granted, fee=fee, markup=markup,
        total_cost=total_cost, pct_of_cash=pct_of_cash, binding_constraint=binding_constraint,
    )


def render_cash_advance_facts(r: CashAdvanceResult) -> str:
    return "\n".join([
        "CASH ADVANCE ASSESSMENT (authoritative — do not contradict)",
        f"- Cash advance sub-limit remaining: {_rs(r.sub_limit_remaining)}",
        f"- Amount available: {_rs(r.granted_amount)} (binding constraint: {r.binding_constraint})",
        f"- Cash advance fee: {_rs(r.fee)}",
        f"- Markup for the holding period: {_rs(r.markup)}",
        f"- Total cost: {_rs(r.total_cost)} ({r.pct_of_cash}% of the cash)",
    ])


# ── cash affordability check ──


@dataclass
class Bill:
    label: str
    amount: int
    due_date: str    # ISO "YYYY-MM-DD" so string comparison sorts correctly


@dataclass
class CashAffordabilityParams:
    customer: CustomerAtmProfile
    withdrawal_amount: int
    bills: list[Bill]
    salary_date: str | None = None   # bills due on/after this date are excluded


@dataclass
class CashAffordabilityResult:
    projected_balance: int
    total_bills_due: int
    headroom: int
    safe: bool
    max_affordable: int
    shortfall: int
    smallest_bill_label: str | None
    smallest_bill_amount: int | None


def check_cash_affordability(p: CashAffordabilityParams) -> CashAffordabilityResult:
    bills = p.bills
    if p.salary_date is not None:
        bills = [b for b in bills if b.due_date < p.salary_date]

    projected_balance = p.customer.available_balance - p.withdrawal_amount
    total_bills_due = sum(b.amount for b in bills)
    headroom = projected_balance - total_bills_due
    safe = headroom >= 0

    max_affordable = max(0, p.customer.available_balance - total_bills_due)
    shortfall = max(0, -headroom)

    smallest = min(bills, key=lambda b: b.amount) if bills else None
    return CashAffordabilityResult(
        projected_balance=projected_balance, total_bills_due=total_bills_due, headroom=headroom,
        safe=safe, max_affordable=max_affordable, shortfall=shortfall,
        smallest_bill_label=smallest.label if smallest else None,
        smallest_bill_amount=smallest.amount if smallest else None,
    )


def render_cash_affordability_facts(r: CashAffordabilityResult) -> str:
    lines = [
        "CASH AFFORDABILITY CHECK (authoritative — do not contradict)",
        f"- Projected balance after withdrawal: {_rs(r.projected_balance)}",
        f"- Bills due before the next salary credit: {_rs(r.total_bills_due)}",
        f"- Headroom: {_rs(r.headroom)}",
        f"- Safe to withdraw in full: {'yes' if r.safe else 'no'}",
    ]
    if not r.safe:
        lines.append(f"- Shortfall: {_rs(r.shortfall)}")
        lines.append(f"- Largest amount that can safely be withdrawn today: {_rs(r.max_affordable)}")
        if r.smallest_bill_label:
            lines.append(f"- Smallest (most flexible) bill: {r.smallest_bill_label} ({_rs(r.smallest_bill_amount)})")
    return "\n".join(lines)


# ── monthly ATM cost summary ──


@dataclass
class MonthlyAtmActivity:
    on_us_count: int
    off_us_count: int
    inquiries_count: int
    avg_withdrawal: int
    free_offus_allowance: int
    offus_fee: float
    inquiry_fee: float


@dataclass
class MonthlyAtmCostResult:
    chargeable_offus: int
    offus_fee_total: int
    inquiry_fee_total: int
    total_fee: int
    total_cash_accessed: int
    pct_of_cash: float


def summarize_monthly_atm_cost(a: MonthlyAtmActivity) -> MonthlyAtmCostResult:
    chargeable = max(0, a.off_us_count - a.free_offus_allowance)
    offus_total = round(chargeable * a.offus_fee)
    inquiry_total = round(a.inquiries_count * a.inquiry_fee)
    total_fee = offus_total + inquiry_total
    total_cash = round((a.on_us_count + a.off_us_count) * a.avg_withdrawal)
    pct = round(total_fee / total_cash * 100, 2) if total_cash else 0.0
    return MonthlyAtmCostResult(
        chargeable_offus=chargeable, offus_fee_total=offus_total, inquiry_fee_total=inquiry_total,
        total_fee=total_fee, total_cash_accessed=total_cash, pct_of_cash=pct,
    )


def render_monthly_atm_cost_facts(r: MonthlyAtmCostResult) -> str:
    return "\n".join([
        "MONTHLY ATM COST SUMMARY (authoritative — do not contradict)",
        f"- Chargeable off-us withdrawals: {r.chargeable_offus} = {_rs(r.offus_fee_total)}",
        f"- Balance inquiry charges: {_rs(r.inquiry_fee_total)}",
        f"- Total ATM charges: {_rs(r.total_fee)}",
        f"- Cash accessed this month: {_rs(r.total_cash_accessed)} ({r.pct_of_cash}% went to fees)",
    ])


# ── spend pattern summary ──


@dataclass
class SpendPatternResult:
    total_withdrawn: int
    withdrawal_count: int
    avg_withdrawal: int
    largest_withdrawal: int
    largest_withdrawal_date: str
    most_used_atm: str | None
    most_used_atm_visits: int
    cities: list[str]
    card_digital_spend: int
    cash_pct_of_outflow: int


_ATM_DESC_RE = re.compile(r"ATM withdrawal (\S+) \(([^)]*)\)")


def summarize_spend_pattern(transactions: list[Transaction]) -> SpendPatternResult:
    atm_txns = [t for t in transactions if t.channel == "ATM"]
    withdrawals = [abs(t.amount) for t in atm_txns]
    total_withdrawn = sum(withdrawals)
    count = len(atm_txns)
    avg = round(total_withdrawn / count) if count else 0

    largest_txn = max(atm_txns, key=lambda t: abs(t.amount)) if atm_txns else None

    atm_visit_counts: dict[str, int] = {}
    cities: list[str] = []
    for t in atm_txns:
        m = _ATM_DESC_RE.search(t.description)
        if not m:
            continue
        atm_id, location_str = m.group(1), m.group(2)
        atm_visit_counts[atm_id] = atm_visit_counts.get(atm_id, 0) + 1
        # location string is "<place>, <city>" — city is the last comma-part
        city = location_str.split(",")[-1].strip()
        if city and city not in cities:
            cities.append(city)

    most_used_atm = None
    most_used_visits = 0
    if atm_visit_counts:
        most_used_atm = max(atm_visit_counts, key=lambda k: atm_visit_counts[k])
        most_used_visits = atm_visit_counts[most_used_atm]

    card_digital_spend = sum(
        -t.amount for t in transactions if t.channel != "ATM" and t.amount < 0
    )
    outflow = total_withdrawn + card_digital_spend
    cash_pct = round(total_withdrawn / outflow * 100) if outflow else 0

    return SpendPatternResult(
        total_withdrawn=total_withdrawn, withdrawal_count=count, avg_withdrawal=avg,
        largest_withdrawal=abs(largest_txn.amount) if largest_txn else 0,
        largest_withdrawal_date=largest_txn.date if largest_txn else "",
        most_used_atm=most_used_atm, most_used_atm_visits=most_used_visits,
        cities=cities, card_digital_spend=card_digital_spend, cash_pct_of_outflow=cash_pct,
    )


def render_spend_pattern_facts(r: SpendPatternResult) -> str:
    lines = [
        "SPEND PATTERN SUMMARY (authoritative — do not contradict)",
        f"- Cash withdrawn: {_rs(r.total_withdrawn)} across {r.withdrawal_count} withdrawals",
        f"- Average withdrawal: {_rs(r.avg_withdrawal)}",
        f"- Largest single withdrawal: {_rs(r.largest_withdrawal)} on {r.largest_withdrawal_date}",
    ]
    if r.most_used_atm:
        lines.append(f"- Most used machine: {r.most_used_atm} ({r.most_used_atm_visits} visits)")
    if r.cities:
        lines.append(f"- Cities used: {', '.join(r.cities)}")
    lines.append(f"- Card and digital spending over the same period: {_rs(r.card_digital_spend)}")
    lines.append(f"- Cash is {r.cash_pct_of_outflow}% of total outflow")
    return "\n".join(lines)


# ── travel cash planning ──


@dataclass
class TravelCashParams:
    customer: CustomerAtmProfile
    days: int
    daily_spend: int
    working_atms_at_destination: int
    off_us_charge: float


@dataclass
class TravelCashResult:
    total_cash_needed: int
    withdrawal_count: int
    days_to_withdraw: int
    cost_if_drawn_at_destination: int
    coverage_classification: str   # "none" | "thin" | "reasonable"


def plan_travel_cash(p: TravelCashParams) -> TravelCashResult:
    total = p.daily_spend * p.days
    per_txn = p.customer.per_txn_limit
    daily_limit = p.customer.daily_limit
    withdrawal_count = math.ceil(total / per_txn) if per_txn > 0 else 1
    days_to_withdraw = math.ceil(total / daily_limit) if daily_limit > 0 else 1
    cost = round(withdrawal_count * p.off_us_charge)

    # Approximation: coverage classification thresholds are not stated
    # explicitly in the dataset (only 0/1/4 examples seen) — 0 machines =
    # "none", 1-2 = "thin", 3+ = "reasonable".
    if p.working_atms_at_destination <= 0:
        classification = "none"
    elif p.working_atms_at_destination < 3:
        classification = "thin"
    else:
        classification = "reasonable"

    return TravelCashResult(
        total_cash_needed=total, withdrawal_count=withdrawal_count,
        days_to_withdraw=days_to_withdraw, cost_if_drawn_at_destination=cost,
        coverage_classification=classification,
    )


def render_travel_cash_facts(r: TravelCashResult) -> str:
    return "\n".join([
        "TRAVEL CASH PLAN (authoritative — do not contradict)",
        f"- Total cash needed: {_rs(r.total_cash_needed)}",
        f"- Withdrawals needed at the per-transaction limit: {r.withdrawal_count}, "
        f"spread over {r.days_to_withdraw} day(s) under the daily limit",
        f"- Cost if drawn off-us at the destination: {_rs(r.cost_if_drawn_at_destination)}",
        f"- Destination ATM coverage: {r.coverage_classification}",
    ])


# ── cash carrying cost (ATM ops) ──


@dataclass
class CashCarryingCostParams:
    atm: AtmMachine
    idle_cash: int
    daily_dispense: int
    annual_rate: float
    cit_trip_cost: int
    window_days: int


@dataclass
class CashCarryingCostResult:
    carrying_cost: int
    reduced_load_amount: int
    funding_saved: int
    shortens_coverage_days: float
    extra_cit_cost: int
    net_benefit: int
    worth_it: bool


# Approximation: the dataset shows the "cut the average load and add trips"
# scenario, but the size of the trial cut is a planning-team judgment call
# not fully recoverable from the given fields. We test a 30% load cut against
# a monthly (30-day) CIT replenishment cycle — a standard treasury heuristic.
_TRIAL_CUT_FRACTION = 0.30
_CIT_CYCLE_DAYS = 30


def calculate_cash_carrying_cost(p: CashCarryingCostParams) -> CashCarryingCostResult:
    carrying_cost = round(p.idle_cash * p.annual_rate * p.window_days / 365)

    cut = round(p.idle_cash * _TRIAL_CUT_FRACTION)
    funding_saved = round(cut * p.annual_rate * p.window_days / 365)
    shortens_days = round(cut / p.daily_dispense, 1) if p.daily_dispense else 0.0

    cycles = p.window_days / _CIT_CYCLE_DAYS
    extra_trips_per_cycle = max(1, math.ceil(shortens_days))
    extra_cit_cost = round(extra_trips_per_cycle * cycles * p.cit_trip_cost)

    net = funding_saved - extra_cit_cost
    return CashCarryingCostResult(
        carrying_cost=carrying_cost, reduced_load_amount=cut, funding_saved=funding_saved,
        shortens_coverage_days=shortens_days, extra_cit_cost=extra_cit_cost,
        net_benefit=net, worth_it=net > 0,
    )


def render_cash_carrying_cost_facts(r: CashCarryingCostResult) -> str:
    return "\n".join([
        "CASH CARRYING COST (authoritative — do not contradict)",
        f"- Carrying cost of the idle cash: {_rs(r.carrying_cost)}",
        f"- Trial load cut: {_rs(r.reduced_load_amount)} -> funding saved {_rs(r.funding_saved)}, "
        f"shortens coverage by {r.shortens_coverage_days} day(s)",
        f"- Extra CIT cost from more frequent trips: {_rs(r.extra_cit_cost)}",
        f"- Net: {'+' if r.net_benefit >= 0 else ''}{_rs(r.net_benefit)} ({'worth doing' if r.worth_it else 'not worth doing'})",
    ])


# ── surge capacity planning (ATM ops) ──


@dataclass
class SurgeCapacityParams:
    atm: AtmMachine
    normal_daily_dispense: int
    season_name: str
    demand_multiplier: float
    closed_days: int
    max_capacity: int


@dataclass
class SurgeCapacityResult:
    peak_daily_demand: int
    total_cash_needed: int
    trips_required: int
    slack: int | None                 # only meaningful when trips_required == 1
    topup_count: int
    topup_interval_days: float | None  # only meaningful when trips_required > 1


def plan_surge_capacity(p: SurgeCapacityParams) -> SurgeCapacityResult:
    peak = round(p.normal_daily_dispense * p.demand_multiplier)
    total_needed = peak * p.closed_days
    trips = math.ceil(total_needed / p.max_capacity) if p.max_capacity > 0 else 1
    trips = max(1, trips)

    slack = p.max_capacity - total_needed if trips == 1 else None
    topup_count = trips - 1
    interval = round(p.max_capacity / peak, 1) if trips > 1 and peak else None

    return SurgeCapacityResult(
        peak_daily_demand=peak, total_cash_needed=total_needed, trips_required=trips,
        slack=slack, topup_count=topup_count, topup_interval_days=interval,
    )


def render_surge_capacity_facts(r: SurgeCapacityResult) -> str:
    lines = [
        "SURGE CAPACITY PLAN (authoritative — do not contradict)",
        f"- Expected daily dispense during the surge: {_rs(r.peak_daily_demand)}",
        f"- Total cash needed across the closure: {_rs(r.total_cash_needed)}",
        f"- CIT trips required: {r.trips_required}",
    ]
    if r.trips_required == 1 and r.slack is not None:
        lines.append(f"- Slack on a single full load: {_rs(r.slack)}")
    else:
        lines.append(
            f"- One full load is not enough: {r.topup_count} mid-holiday top-up(s), "
            f"roughly every {r.topup_interval_days} day(s)"
        )
    return "\n".join(lines)
