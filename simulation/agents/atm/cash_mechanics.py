"""Deterministic ATM cash-mechanics policy engine.

Covers the 8 "cash mechanics" task types in the fine-tuning dataset: cash
run-out forecasting, cassette status triage, cash reconciliation,
denomination mix planning, denomination dispensability, cash load planning,
withdrawal feasibility, and daily-limit-remaining checks.

Mirrors loan-agent's app/agents/loans/policy.py convention (also followed by
this package's network_ops.py): small pure dataclasses for inputs/outputs,
one deterministic computation function per task (no I/O, no randomness), and
a matching prompt-fact-renderer function per task that presents the computed
numbers as authoritative ground truth for a downstream LLM to narrate. Every
formula below was reverse-engineered from the worked examples in
merged_finetune_train.jsonl / _val.jsonl (synthetic_atm_ops /
synthetic_atm_customer).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .entities import AtmMachine, CustomerAtmProfile, Transaction
from .prompts import _rs, render_atm_status, render_customer_profile  # noqa: F401 (re-exported for callers)


# ── shared helpers ──

def _denom_notes(machine: AtmMachine) -> dict[int, int]:
    """Aggregate note counts per denomination across every non-FAULT
    cassette (LOW cassettes still dispense — only FAULT locks a cassette
    out, matching AtmMachine.usable_cash's own rule)."""
    notes: dict[int, int] = {}
    for c in machine.cassettes:
        if c.status == "FAULT":
            continue
        notes[c.denom] = notes.get(c.denom, 0) + c.notes
    return notes


def _compose_greedy(denom_notes: dict[int, int], target: int) -> tuple[int, list[tuple[int, int]]]:
    """Largest-denomination-first note composition (the dispensing logic
    the dataset always uses): walk denominations largest to smallest,
    taking as many notes of each as are available and still needed.
    Returns (amount actually composed, [(denom, notes_used), ...]) — the
    composed amount is target itself when it is exactly reachable, or the
    largest reachable amount below target otherwise (used both to test
    dispensability and to find the "nearest amount")."""
    if target <= 0:
        return 0, []
    remaining = target
    breakdown: list[tuple[int, int]] = []
    for denom in sorted(denom_notes, reverse=True):
        available = denom_notes[denom]
        if available <= 0 or denom <= 0:
            continue
        notes = min(remaining // denom, available)
        if notes > 0:
            breakdown.append((denom, notes))
            remaining -= notes * denom
    return target - remaining, breakdown


def _atm_used_today(transactions: list[Transaction]) -> int:
    """Sum of today's ATM cash withdrawals — POS/e-commerce spend never
    touches the ATM daily cash limit."""
    return sum(-t.amount for t in transactions if t.channel == "ATM" and t.amount < 0)


def _fmt_breakdown(bd: list[tuple[int, int]]) -> str:
    return " + ".join(f"{n} x Rs {d}" for d, n in bd) if bd else "none"


# ── 1. cash run-out forecast ──

@dataclass
class CashRunoutForecastInput:
    current_time: str              # "HH:MM", 24-hour
    dispense_rate_per_hour: float  # observed Rs/hour
    low_cash_threshold: int
    cit_time: str                  # scheduled CIT arrival, "HH:MM"
    cit_hours_from_now: float      # hours until that CIT visit


@dataclass
class CashRunoutForecastResult:
    total_cash: int
    dispense_rate_per_hour: float
    hours_to_low: float
    low_cash_time: str
    hours_to_empty: float
    empty_time: str
    cit_time: str
    cit_hours_from_now: float
    survives_to_cit: bool
    headroom_hours: float   # positive = spare hours past CIT; negative = deficit before CIT


def _parse_hhmm(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def _format_clock(total_minutes: float) -> str:
    m = round(total_minutes) % (24 * 60)
    h, mi = divmod(m, 60)
    return f"{h:02d}:{mi:02d}"


def _format_duration(hours: float) -> str:
    total_minutes = round(hours * 60)
    if total_minutes <= 0:
        return "0 hours"
    h, m = divmod(total_minutes, 60)
    return f"{h} hours {m} minutes"


def forecast_cash_runout(machine: AtmMachine, inp: CashRunoutForecastInput) -> CashRunoutForecastResult:
    total_cash = machine.usable_cash
    rate = inp.dispense_rate_per_hour
    if rate <= 0:
        hours_to_low = math.inf
        hours_to_empty = math.inf
    else:
        hours_to_low = max(0.0, (total_cash - inp.low_cash_threshold) / rate)
        hours_to_empty = total_cash / rate
    current_min = _parse_hhmm(inp.current_time)
    low_time = _format_clock(current_min + hours_to_low * 60) if math.isfinite(hours_to_low) else "never"
    empty_time = _format_clock(current_min + hours_to_empty * 60) if math.isfinite(hours_to_empty) else "never"
    survives = hours_to_empty >= inp.cit_hours_from_now
    headroom = hours_to_empty - inp.cit_hours_from_now if math.isfinite(hours_to_empty) else math.inf
    return CashRunoutForecastResult(
        total_cash=total_cash, dispense_rate_per_hour=rate,
        hours_to_low=hours_to_low, low_cash_time=low_time,
        hours_to_empty=hours_to_empty, empty_time=empty_time,
        cit_time=inp.cit_time, cit_hours_from_now=inp.cit_hours_from_now,
        survives_to_cit=survives, headroom_hours=round(headroom, 1) if math.isfinite(headroom) else headroom,
    )


def render_cash_runout_facts(r: CashRunoutForecastResult) -> str:
    lines = [
        "CASH RUN-OUT FORECAST (authoritative — do not contradict)",
        f"- Cash loaded: {_rs(r.total_cash)}, dispensing {_rs(r.dispense_rate_per_hour)}/hour",
        f"- Low-cash threshold hit in {_format_duration(r.hours_to_low)}, at about {r.low_cash_time}",
        f"- Fully empty in {_format_duration(r.hours_to_empty)}, at about {r.empty_time}",
        f"- CIT scheduled at {r.cit_time} (in {r.cit_hours_from_now:g} hours)",
    ]
    if r.survives_to_cit:
        lines.append(
            f"- Survives to CIT with {r.headroom_hours:.1f} hours of headroom — no emergency run needed"
        )
    else:
        lines.append(
            f"- Goes dry {abs(r.headroom_hours):.1f} hours before the CIT slot — "
            f"raise an emergency replenishment or pull the visit forward to about {r.low_cash_time}"
        )
    return "\n".join(lines)


# ── 2. cassette status triage ──

WITHDRAWAL_CAP = 200_000   # typical single-transaction cash ceiling used to size the "largest composable withdrawal"


@dataclass
class CassetteTriageResult:
    usable_cash: int
    working_cassette_count: int
    max_single_withdrawal: int
    max_withdrawal_breakdown: list[tuple[int, int]]
    keep_in_service: bool
    actions: list[str]


def triage_cassette_status(machine: AtmMachine, withdrawal_cap: int = WITHDRAWAL_CAP) -> CassetteTriageResult:
    usable_cash = machine.usable_cash
    working = sum(1 for c in machine.cassettes if c.status != "FAULT")
    denom_notes = _denom_notes(machine)
    composed, breakdown = _compose_greedy(denom_notes, min(withdrawal_cap, usable_cash))

    actions: list[str] = []
    for i, c in enumerate(machine.cassettes, 1):
        if c.status == "FAULT":
            actions.append(
                f"Cassette {i} (Rs {c.denom}): engineer visit - pick roller / transport path check, "
                "cassette is locked out"
            )
        elif c.status == "LOW":
            actions.append(f"Cassette {i} (Rs {c.denom}): {c.notes:,} notes left ({_rs(c.value)}) - top up")
    if not actions:
        actions = ["No action needed"]

    # Approximation: only take the machine offline once there is literally
    # nothing left to dispense — the dataset's worked examples never show a
    # take-offline verdict, so this is the clearest rule-based extrapolation.
    keep_in_service = usable_cash > 0

    return CassetteTriageResult(
        usable_cash=usable_cash, working_cassette_count=working,
        max_single_withdrawal=composed, max_withdrawal_breakdown=breakdown,
        keep_in_service=keep_in_service, actions=actions,
    )


def render_cassette_triage_facts(r: CassetteTriageResult) -> str:
    lines = [
        "CASSETTE TRIAGE (authoritative — do not contradict)",
        f"- Dispensable now: {_rs(r.usable_cash)} from {r.working_cassette_count} working cassette(s)",
        f"- Largest composable withdrawal: {_rs(r.max_single_withdrawal)} "
        f"({_fmt_breakdown(r.max_withdrawal_breakdown)})",
        f"- Recommendation: {'keep in service' if r.keep_in_service else 'take offline'}",
        "- Action list:",
    ]
    lines += [f"  {a}" for a in r.actions]
    return "\n".join(lines)


# ── 3. cash reconciliation ──

@dataclass
class CashReconciliationResult:
    expected_closing: int
    physical_count: int
    variance: int
    status: str   # "balanced" | "shortage" | "overage"


def reconcile_cash(opening_cash: int, loaded_this_cycle: int, dispensed_ej: int, physical_count: int) -> CashReconciliationResult:
    expected = opening_cash + loaded_this_cycle - dispensed_ej
    variance = physical_count - expected
    status = "balanced" if variance == 0 else ("shortage" if variance < 0 else "overage")
    return CashReconciliationResult(
        expected_closing=expected, physical_count=physical_count, variance=variance, status=status,
    )


def render_cash_reconciliation_facts(r: CashReconciliationResult, reject_bin_notes: int | None = None) -> str:
    lines = [
        "CASH RECONCILIATION (authoritative — do not contradict)",
        f"- Expected closing balance: {_rs(r.expected_closing)}",
        f"- Physical count: {_rs(r.physical_count)}",
        f"- Variance: {r.variance:+,} PKR ({r.status})",
    ]
    if reject_bin_notes is not None:
        lines.append(f"- Notes in reject bin: {reject_bin_notes}")
    return "\n".join(lines)


# ── 4. denomination mix planning ──

@dataclass
class DenominationMixPlanningInput:
    histogram: list[tuple[int, int]]         # (withdrawal amount, transaction count)
    denominations: list[int]                  # available note denominations
    cassette_capacity_notes: dict[int, int]   # denom -> capacity in notes


@dataclass
class DenominationMixPlanningResult:
    total_withdrawals: int
    total_value: int
    notes_per_denom: dict[int, int]
    value_per_denom: dict[int, int]
    capacity_notes: dict[int, int]
    over_capacity_denoms: list[int]
    all_fit_capacity: bool


def plan_denomination_mix(inp: DenominationMixPlanningInput) -> DenominationMixPlanningResult:
    denoms_sorted = sorted(inp.denominations, reverse=True)
    notes_per_denom: dict[int, int] = {d: 0 for d in denoms_sorted}
    total_withdrawals = 0
    total_value = 0
    for amount, count in inp.histogram:
        total_withdrawals += count
        total_value += amount * count
        remaining = amount
        for d in denoms_sorted:
            n = remaining // d
            if n:
                notes_per_denom[d] += n * count
                remaining -= n * d
        # Any leftover remainder is dropped — the dataset's histogram
        # buckets are always exact multiples of the smallest denomination.
    value_per_denom = {d: notes_per_denom[d] * d for d in denoms_sorted}
    over = [
        d for d in denoms_sorted
        if notes_per_denom[d] > inp.cassette_capacity_notes.get(d, 0)
    ]
    return DenominationMixPlanningResult(
        total_withdrawals=total_withdrawals, total_value=total_value,
        notes_per_denom=notes_per_denom, value_per_denom=value_per_denom,
        capacity_notes=dict(inp.cassette_capacity_notes),
        over_capacity_denoms=over, all_fit_capacity=len(over) == 0,
    )


def render_denomination_mix_facts(r: DenominationMixPlanningResult) -> str:
    lines = [
        "DENOMINATION MIX PLAN (authoritative — do not contradict)",
        f"- {r.total_withdrawals:,} withdrawals, {_rs(r.total_value)} total",
    ]
    for d, n in r.notes_per_denom.items():
        if n > 0:
            lines.append(
                f"- Rs {d}: {n:,} notes ({_rs(r.value_per_denom[d])}) - "
                f"cassette capacity {r.capacity_notes.get(d, 0):,} notes"
            )
    if r.all_fit_capacity:
        lines.append("- Every cassette can hold more than a full day of demand — a daily fill cycle is safe")
    else:
        for d in r.over_capacity_denoms:
            lines.append(
                f"- Rs {d} cassette cannot hold a full day of demand "
                f"({r.notes_per_denom[d]:,} notes needed vs {r.capacity_notes.get(d, 0):,} capacity)"
            )
    return "\n".join(lines)


# ── 5. denomination dispensability ──

@dataclass
class DenominationDispensabilityResult:
    requested_amount: int
    dispensable: bool
    breakdown: list[tuple[int, int]]
    nearest_amount: int
    nearest_breakdown: list[tuple[int, int]]


def check_denomination_dispensable(machine: AtmMachine, requested_amount: int) -> DenominationDispensabilityResult:
    denom_notes = _denom_notes(machine)
    composed, breakdown = _compose_greedy(denom_notes, requested_amount)
    if composed == requested_amount:
        return DenominationDispensabilityResult(
            requested_amount=requested_amount, dispensable=True,
            breakdown=breakdown, nearest_amount=requested_amount, nearest_breakdown=breakdown,
        )
    return DenominationDispensabilityResult(
        requested_amount=requested_amount, dispensable=False,
        breakdown=[], nearest_amount=composed, nearest_breakdown=breakdown,
    )


def render_denomination_dispensability_facts(r: DenominationDispensabilityResult) -> str:
    lines = [
        "DENOMINATION CHECK (authoritative — do not contradict)",
        f"- Requested: {_rs(r.requested_amount)}",
    ]
    if r.dispensable:
        lines.append(f"- Dispensable: yes ({_fmt_breakdown(r.breakdown)})")
    else:
        lines.append("- Dispensable: no")
        lines.append(f"- Nearest amount the machine can give: {_rs(r.nearest_amount)} ({_fmt_breakdown(r.nearest_breakdown)})")
    return "\n".join(lines)


# ── 6. cash load planning ──

@dataclass
class CashLoadPlanningInput:
    avg_daily_dispense: int
    days_until_cit: int
    policy_buffer_pct: float           # e.g. 0.15 for 15%
    cassette_specs: list[tuple[int, int]]   # (denom, capacity_notes), one entry per physical cassette
    withdrawal_mix: dict[int, float]         # denom -> historical share (fractions, need not sum to the machine's own denoms)


@dataclass
class CashLoadPlanningResult:
    demand: int
    required_with_buffer: int
    total_capacity: int
    load_amount: int
    capacity_binding: bool
    allocations: list[tuple[int, int, int]]   # (denom, notes, value)
    total_loaded: int
    coverage_days: float


def plan_cash_load(inp: CashLoadPlanningInput) -> CashLoadPlanningResult:
    demand = round(inp.avg_daily_dispense * inp.days_until_cit)
    required = round(demand * (1 + inp.policy_buffer_pct))
    total_capacity = sum(denom * cap for denom, cap in inp.cassette_specs)
    load_amount = min(required, total_capacity)
    capacity_binding = total_capacity < required

    capacity_by_denom: dict[int, int] = {}
    for denom, cap in inp.cassette_specs:
        capacity_by_denom[denom] = capacity_by_denom.get(denom, 0) + cap
    present_denoms = sorted(capacity_by_denom, reverse=True)

    # Renormalise the historical withdrawal mix over only the denominations
    # this machine actually has cassettes for — a machine missing a
    # denomination's cassette still needs the load split across what it
    # does have, so that share is redistributed proportionally. Matches the
    # dataset's own numbers when a listed mix denomination has no cassette.
    mix_sum = sum(inp.withdrawal_mix.get(d, 0.0) for d in present_denoms)

    allocations: list[tuple[int, int, int]] = []
    total_loaded = 0
    for d in present_denoms:
        weight = (inp.withdrawal_mix.get(d, 0.0) / mix_sum) if mix_sum > 0 else 0.0
        notes = int((load_amount * weight) // d)
        notes = min(notes, capacity_by_denom[d])
        value = notes * d
        allocations.append((d, notes, value))
        total_loaded += value

    coverage_days = round(total_loaded / inp.avg_daily_dispense, 1) if inp.avg_daily_dispense else 0.0
    return CashLoadPlanningResult(
        demand=demand, required_with_buffer=required, total_capacity=total_capacity,
        load_amount=load_amount, capacity_binding=capacity_binding, allocations=allocations,
        total_loaded=total_loaded, coverage_days=coverage_days,
    )


def render_cash_load_planning_facts(r: CashLoadPlanningResult) -> str:
    lines = [
        "CASH LOAD PLAN (authoritative — do not contradict)",
        f"- Demand over the interval: {_rs(r.demand)}, with buffer: {_rs(r.required_with_buffer)}",
        f"- Total cassette capacity: {_rs(r.total_capacity)}",
        f"- Load amount: {_rs(r.load_amount)}" + (" (capacity-bound)" if r.capacity_binding else " (demand-bound)"),
    ]
    for d, n, v in r.allocations:
        if n > 0:
            lines.append(f"- Rs {d} cassette: {n:,} notes = {_rs(v)}")
    lines.append(
        f"- Total loaded: {_rs(r.total_loaded)}, covers about {r.coverage_days:.1f} days "
        "at the trailing average"
    )
    return "\n".join(lines)


# ── 7. withdrawal feasibility ──

@dataclass
class WithdrawalFeasibilityResult:
    requested_amount: int
    binding_constraint: str
    binding_value: int
    feasible: bool
    dispense_amount: int
    dispense_breakdown: list[tuple[int, int]]
    balance_after: int
    daily_limit_remaining_after: int


def check_withdrawal_feasibility(
    profile: CustomerAtmProfile, machine: AtmMachine, requested_amount: int, transactions: list[Transaction],
) -> WithdrawalFeasibilityResult:
    remaining_daily = profile.daily_limit - _atm_used_today(transactions)
    usable_cash = machine.usable_cash

    ceilings = {
        "account balance": profile.available_balance,
        "per-transaction limit": profile.per_txn_limit,
        "remaining daily limit": remaining_daily,
        "cash available in ATM": usable_cash,
    }
    binding_label = min(ceilings, key=lambda k: ceilings[k])
    binding_value = ceilings[binding_label]

    denom_notes = _denom_notes(machine)
    target = max(0, min(requested_amount, binding_value))
    composed, breakdown = _compose_greedy(denom_notes, target)

    feasible = requested_amount <= binding_value and composed == requested_amount
    dispense_amount = composed
    balance_after = profile.available_balance - dispense_amount
    daily_after = remaining_daily - dispense_amount

    return WithdrawalFeasibilityResult(
        requested_amount=requested_amount, binding_constraint=binding_label, binding_value=binding_value,
        feasible=feasible, dispense_amount=dispense_amount, dispense_breakdown=breakdown,
        balance_after=balance_after, daily_limit_remaining_after=daily_after,
    )


def render_withdrawal_feasibility_facts(r: WithdrawalFeasibilityResult) -> str:
    return "\n".join([
        "WITHDRAWAL FEASIBILITY (authoritative — do not contradict)",
        f"- Requested: {_rs(r.requested_amount)}",
        f"- Feasible: {'yes' if r.feasible else 'no'}",
        f"- Binding constraint: {r.binding_constraint} ({_rs(r.binding_value)})",
        f"- Machine will dispense: {_rs(r.dispense_amount)} ({_fmt_breakdown(r.dispense_breakdown)})",
        f"- Balance after: {_rs(r.balance_after)}",
        f"- Daily limit remaining after: {_rs(r.daily_limit_remaining_after)}",
    ])


# ── 8. daily limit remaining ──

@dataclass
class DailyLimitRemainingResult:
    daily_limit: int
    atm_used_today: int
    remaining: int
    requested_amount: int | None = None
    feasible: bool | None = None


def remaining_daily_limit(
    profile: CustomerAtmProfile, transactions: list[Transaction], requested_amount: int | None = None,
) -> DailyLimitRemainingResult:
    used = _atm_used_today(transactions)
    remaining = profile.daily_limit - used
    feasible = (requested_amount <= remaining) if requested_amount is not None else None
    return DailyLimitRemainingResult(
        daily_limit=profile.daily_limit, atm_used_today=used, remaining=remaining,
        requested_amount=requested_amount, feasible=feasible,
    )


def render_daily_limit_facts(r: DailyLimitRemainingResult) -> str:
    lines = [
        "DAILY CASH LIMIT (authoritative — do not contradict)",
        f"- Daily ATM limit: {_rs(r.daily_limit)}",
        f"- Taken out at ATMs today: {_rs(r.atm_used_today)}",
        f"- Remaining today: {_rs(r.remaining)}",
    ]
    if r.requested_amount is not None:
        lines.append(
            f"- Requested {_rs(r.requested_amount)}: "
            + ("will go through" if r.feasible else "will be declined")
        )
    return "\n".join(lines)
