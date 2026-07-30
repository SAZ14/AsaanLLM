"""Deterministic ATM judgment/classification policy engine.

Covers the 8 remaining ATM task types in the fine-tuning dataset that are
rule-based classifiers rather than pure arithmetic: fault root-cause
diagnosis, security anomaly assessment, card fraud assessment, card
retention guidance, and failed-transaction dispute assessment — plus the
three pure-arithmetic tasks interbank settlement, trend summary, and
uptime SLA check.

Mirrors loan-agent's app/agents/loans/policy.py convention: small pure
dataclasses for inputs/outputs, one deterministic computation function per
task (no I/O, no randomness), and a matching prompt-fact-renderer function
per task that presents the computed numbers/verdicts as authoritative
ground truth for a downstream LLM to narrate. Every rule and formula below
was reverse-engineered from the worked examples in
merged_finetune_train.jsonl / merged_finetune_val.jsonl
(synthetic_atm_ops / synthetic_atm_customer).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from .entities import AtmMachine, CustomerAtmProfile, Transaction
from .prompts import _rs

# ════════════════════════════════════════════════════════════════════════
# fault root cause
# ════════════════════════════════════════════════════════════════════════

# Cause/action/ownership knowledge base, keyed by vendor fault code. The
# code -> meaning mapping itself is NOT hardcoded here — it comes from the
# vendor code reference table given in the prompt (see diagnose_fault) —
# but the operational playbook (why it happens, what to do, who owns it)
# is bank/vendor-support knowledge and lives in this lookup table so the
# logic is auditable rather than a black box. Every entry is a fault code
# that appears somewhere in the ATM-ops worked examples (fault_root_cause,
# security_anomaly_assessment, uptime_sla_check).
FAULT_CODE_PLAYBOOK: dict[str, tuple[str, str, str]] = {
    # code: (likely_cause, action, ownership)
    "R0005": (
        "Paper roll not replaced on last visit",
        "Replace roll during next custodian visit",
        "branch custodian / NOC (can be closed remotely or by site staff)",
    ),
    "P0022": (
        "Extended load-shedding beyond UPS backup",
        "Generator changeover / extend UPS bank",
        "branch custodian / NOC (can be closed remotely or by site staff)",
    ),
    "L0002": (
        "ISP or router failure at site",
        "Failover to backup 4G link, raise ISP ticket",
        "branch custodian / NOC (can be closed remotely or by site staff)",
    ),
    "S0015": (
        "Custodian did not close supervisor session",
        "Close session, retrain custodian",
        "branch custodian / NOC (can be closed remotely or by site staff)",
    ),
    "H0013": (
        "Note-present sensor miscounting, often after a mechanical jam",
        "Power-cycle the dispenser and run a test dispense; escalate to a vendor engineer if the mismatch persists",
        "field engineer / vendor support",
    ),
    "E0004": (
        "Electronic journal not downloaded/archived on schedule",
        "Download and clear the electronic journal, verify the archiving job",
        "branch custodian / NOC (can be closed remotely or by site staff)",
    ),
    "N0009": (
        "Worn/dirty notes or degraded feed rollers",
        "Clean the feed path, load a fresh note cassette, inspect rollers",
        "branch custodian / vendor engineer",
    ),
    "T0007": (
        "Tamper sensor triggered - possible physical interference at the PIN pad",
        "Take the ATM offline, inspect the PIN pad for skimming devices, verify against CCTV before resetting",
        "security team / vendor engineer",
    ),
    "D0087": (
        "Note jammed in the feed path, often from a damaged note",
        "Field engineer clears the jam and inspects the feed rollers",
        "field engineer / vendor support",
    ),
    "C0031": (
        "Card reader head dirty or faulty",
        "Clean and inspect the card reader, run diagnostics; replace the reader if the fault repeats",
        "field engineer / vendor support",
    ),
    "X0001": (
        "Cassette load depleted faster than the scheduled CIT visit",
        "Dispatch an emergency CIT/cash replenishment run",
        "cash management / CIT vendor",
    ),
    "D0011": (
        "Pick mechanism misfeed, often from a worn note or debris",
        "Field engineer clears and services the dispenser pick mechanism",
        "field engineer / vendor support",
    ),
}

CUSTOMER_NOTICE_MINUTES = 60  # past this, a lobby-door customer notice is warranted


@dataclass
class FaultDiagnosisInputs:
    atm: AtmMachine
    fault_code: str
    down_minutes: int
    failed_transactions: int
    vendor_table: dict[str, str]   # code -> meaning, as given in the prompt


@dataclass
class FaultRootCauseResult:
    atm_id: str
    code: str
    meaning: str
    likely_cause: str
    action: str
    ownership: str
    down_minutes: int
    failed_transactions: int
    past_notice_threshold: bool


def diagnose_fault(inputs: FaultDiagnosisInputs) -> FaultRootCauseResult:
    """Lookup-table match against the vendor code table given in the
    prompt for the meaning, and the bank's internal playbook for the
    cause/action/ownership. Unknown codes fall back to a generic escalation
    rather than crashing — the table doesn't claim to cover every possible
    vendor code."""
    meaning = inputs.vendor_table.get(inputs.fault_code)
    if meaning is None:
        raise ValueError(f"fault code {inputs.fault_code!r} not found in the vendor table")

    playbook = FAULT_CODE_PLAYBOOK.get(inputs.fault_code)
    if playbook is None:
        cause, action, ownership = (
            "Cause not catalogued for this code",
            "Escalate to vendor support / NOC for triage",
            "field engineer / vendor support",
        )
    else:
        cause, action, ownership = playbook

    return FaultRootCauseResult(
        atm_id=inputs.atm.atm_id, code=inputs.fault_code, meaning=meaning,
        likely_cause=cause, action=action, ownership=ownership,
        down_minutes=inputs.down_minutes, failed_transactions=inputs.failed_transactions,
        past_notice_threshold=inputs.down_minutes > CUSTOMER_NOTICE_MINUTES,
    )


def render_fault_diagnosis_facts(r: FaultRootCauseResult) -> str:
    lines = [
        "FAULT DIAGNOSIS (authoritative — do not contradict)",
        f"- {r.code} on {r.atm_id} means {r.meaning}",
        f"- Likely cause: {r.likely_cause}",
        f"- Action: {r.action}",
        f"- Ownership: {r.ownership}",
        f"- Down for: {r.down_minutes} minutes ({r.failed_transactions} failed transactions since fault)",
        "- Past the 60-minute customer-notice threshold — put a notice on the lobby door"
        if r.past_notice_threshold
        else "- Under the 60-minute customer-notice threshold — log against the uptime SLA and keep monitoring",
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# security anomaly assessment
# ════════════════════════════════════════════════════════════════════════

# A genuine surge is one where volume is elevated at least this multiple
# over the trailing 30-day average — below this, treat volume as normal.
SURGE_MULTIPLIER = 1.3
# Repeated "no cash received" complaints against completed dispenses, at
# or above this count, is enough to suspect a physical trap device rather
# than one-off customer error.
CASH_TRAP_REPORT_THRESHOLD = 3

JACKPOTTING_ACTIONS = [
    "Take the ATM offline right now and do not let anyone reboot it - the machine state is evidence",
    "Send physical security to site",
    "Pull CCTV for the full window",
    "Preserve the electronic journal and the hard disk image",
    "Notify the fraud and information-security teams",
    "Check every other ATM of the same model and firmware version in the region for the same signature",
]
SKIMMING_ACTIONS = [
    "Take the ATM offline and inspect the card reader and PIN pad for foreign devices",
    "Pull CCTV for the inspection window",
    "Block and reissue cards used at this ATM since the device was likely installed",
    "Notify the fraud and information-security teams",
]
CASH_TRAP_ACTIONS = [
    "Take the ATM offline and physically inspect the dispenser throat for a trap device",
    "Cross-check journal 'dispensed' entries against the cash reconciliation for a mismatch",
    "Refund affected customers once the trap is confirmed",
    "Notify the fraud team",
]
DEMAND_SURGE_ACTIONS = [
    "Raise the cash load for this site",
    "Add a mid-day top-up for payout/disbursement days",
]
MANUAL_REVIEW_ACTIONS = [
    "Escalate to the fraud/security team for manual review - the pattern doesn't cleanly match a known signature",
]

# Known ATM attack signatures, named for auditability. Each entry is a
# short description of the trigger condition implemented in
# assess_security_anomaly() below.
ATTACK_SIGNATURES: dict[str, str] = {
    "jackpotting_black_box": (
        "Dispense commands with no matching switch authorisation, combined with "
        "a physical-tamper indicator (safe door opened outside a scheduled CIT "
        "visit, or the USB port cover disturbed)."
    ),
    "skimming": "A foreign device or tamper flagged at the card reader / PIN pad.",
    "cash_trapping": "Repeated customer reports of no cash received against completed dispenses.",
    "genuine_demand_surge": (
        "Elevated transaction volume that is fully authorised, with normal "
        "ticket sizes and an explained cause (e.g. a nearby payout day)."
    ),
}


@dataclass
class SecurityObservation:
    atm: AtmMachine
    event_count: int                       # dispense commands or withdrawal count in the window
    window_start: str
    window_end: str
    total_amount: int = 0
    baseline_avg_count: int | None = None  # trailing 30-day average count for the same window
    avg_ticket_size: int | None = None
    unmatched_authorisation: bool = False  # True if dispenses have no matching switch auth
    safe_door_opened_time: str | None = None
    cit_visit_scheduled: bool = False
    usb_port_cover_open: bool = False
    payout_event_nearby: bool = False
    skimming_device_detected: bool = False
    cash_trap_reports: int = 0


@dataclass
class SecurityAnomalyAssessmentResult:
    atm_id: str
    signature: str        # key into ATTACK_SIGNATURES, or "unclassified"
    is_attack: bool
    headline: str
    findings: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


def assess_security_anomaly(obs: SecurityObservation) -> SecurityAnomalyAssessmentResult:
    atm_id = obs.atm.atm_id

    # Signature 1: jackpotting / black-box — unauthorised dispenses plus a
    # physical-tamper indicator.
    tamper_indicator = (
        (obs.safe_door_opened_time is not None and not obs.cit_visit_scheduled)
        or obs.usb_port_cover_open
    )
    if obs.unmatched_authorisation and tamper_indicator:
        findings = [f"{obs.event_count} dispense commands with no matching switch authorisation"]
        if obs.safe_door_opened_time and not obs.cit_visit_scheduled:
            findings.append(
                f"safe door sensor opened at {obs.safe_door_opened_time} with no CIT visit scheduled"
            )
        if obs.usb_port_cover_open:
            findings.append("USB port cover found open on the last inspection")
        return SecurityAnomalyAssessmentResult(
            atm_id=atm_id, signature="jackpotting_black_box", is_attack=True,
            headline="jackpotting / black-box attack",
            findings=findings, actions=list(JACKPOTTING_ACTIONS),
        )

    # Signature 2: skimming — tamper/foreign device at the card/PIN interface.
    if obs.skimming_device_detected:
        return SecurityAnomalyAssessmentResult(
            atm_id=atm_id, signature="skimming", is_attack=True,
            headline="skimming attack",
            findings=["Foreign device / tamper detected at the card reader or PIN pad"],
            actions=list(SKIMMING_ACTIONS),
        )

    # Signature 3: cash trapping — repeated non-receipt complaints against
    # completed journal entries.
    if obs.cash_trap_reports >= CASH_TRAP_REPORT_THRESHOLD:
        return SecurityAnomalyAssessmentResult(
            atm_id=atm_id, signature="cash_trapping", is_attack=True,
            headline="cash-trapping attack",
            findings=[
                f"{obs.cash_trap_reports} customer reports of no cash received against completed dispenses"
            ],
            actions=list(CASH_TRAP_ACTIONS),
        )

    # Signature 4: genuine demand surge — elevated volume, fully authorised,
    # explained by an external event.
    if (
        not obs.unmatched_authorisation
        and obs.baseline_avg_count
        and obs.event_count >= obs.baseline_avg_count * SURGE_MULTIPLIER
    ):
        findings = [
            f"{obs.event_count} withdrawals between {obs.window_start} and {obs.window_end}, "
            f"against a 30-day average of {obs.baseline_avg_count}",
            "all transactions carry valid switch authorisations",
        ]
        if obs.avg_ticket_size is not None:
            findings.append(f"average ticket size {_rs(obs.avg_ticket_size)}, in line with normal")
        if obs.payout_event_nearby:
            findings.append("a nearby payout/disbursement event explains the surge")
        return SecurityAnomalyAssessmentResult(
            atm_id=atm_id, signature="genuine_demand_surge", is_attack=False,
            headline="genuine demand surge, not an attack",
            findings=findings, actions=list(DEMAND_SURGE_ACTIONS),
        )

    # No known signature matched — flag for manual review rather than
    # guessing.
    return SecurityAnomalyAssessmentResult(
        atm_id=atm_id, signature="unclassified", is_attack=obs.unmatched_authorisation,
        headline="pattern does not match a known signature",
        findings=["insufficient corroborating evidence for an automatic classification"],
        actions=list(MANUAL_REVIEW_ACTIONS),
    )


def render_security_assessment_facts(r: SecurityAnomalyAssessmentResult) -> str:
    lines = [
        "SECURITY ANOMALY ASSESSMENT (authoritative — do not contradict)",
        f"- Assessment for {r.atm_id}: {r.headline}",
        f"- Is an attack: {'yes' if r.is_attack else 'no'}",
    ]
    lines += [f"- {f}" for f in r.findings]
    lines.append("- Actions: " + "; ".join(r.actions))
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# card fraud assessment
# ════════════════════════════════════════════════════════════════════════

# "Odd hours" for the velocity signature: transactions before this hour of
# the day are treated as night-time.
NIGHT_HOUR_CUTOFF = 5
# 3+ maximum-value withdrawals inside this window, at odd hours, is the
# classic post-compromise drain pattern.
VELOCITY_MIN_COUNT = 3
VELOCITY_WINDOW_MINUTES = 30
# Two ATM withdrawals in different cities inside this many minutes cannot
# both be the genuine cardholder.
IMPOSSIBLE_TRAVEL_MINUTES = 120

FRAUD_HIGH_RISK_ACTIONS = [
    "Block the card immediately - do not wait for the bank to call back",
    "Report the transactions as unauthorised and get a complaint reference number",
    "Ask for the card to be reissued with a new number, and change internet banking password and PIN",
    "Check whether anything else moved on the account",
]


def _extract_atm_city(description: str) -> str | None:
    """Pulls the city out of descriptions of the form
    'ATM withdrawal <City> - <Sub-location>'."""
    prefix = "ATM withdrawal "
    if not description.startswith(prefix):
        return None
    rest = description[len(prefix):]
    city = rest.split(" - ", 1)[0].strip()
    return city or None


def _parse_txn_dt(t: Transaction) -> datetime:
    return datetime.strptime(f"{t.date} {t.time}", "%Y-%m-%d %H:%M")


@dataclass
class CardFraudInputs:
    profile: CustomerAtmProfile
    transactions: list[Transaction]


@dataclass
class CardFraudAssessmentResult:
    verdict: str          # "high_risk_velocity" | "high_risk_impossible_travel" | "normal"
    is_high_risk: bool
    total_atm_outflow: int
    findings: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


def assess_card_fraud(inputs: CardFraudInputs) -> CardFraudAssessmentResult:
    p = inputs.profile
    atm_txns = [t for t in inputs.transactions if t.channel == "ATM" and t.amount < 0]
    total_atm_outflow = sum(-t.amount for t in atm_txns)

    parsed = sorted(
        ((t, _extract_atm_city(t.description), _parse_txn_dt(t)) for t in atm_txns),
        key=lambda x: x[2],
    )

    # Signature 1: velocity max-out at odd hours.
    maxed = [x for x in parsed if -x[0].amount >= p.per_txn_limit and x[2].hour < NIGHT_HOUR_CUTOFF]
    if len(maxed) >= VELOCITY_MIN_COUNT:
        span_minutes = (maxed[-1][2] - maxed[0][2]).total_seconds() / 60
        if span_minutes <= VELOCITY_WINDOW_MINUTES:
            findings = [
                f"{len(maxed)} maximum-value withdrawals of {_rs(p.per_txn_limit)} each, back to "
                f"back between {maxed[0][2]:%H:%M} and {maxed[-1][2]:%H:%M}",
                "classic pattern after a card and PIN have been compromised",
            ]
            return CardFraudAssessmentResult(
                verdict="high_risk_velocity", is_high_risk=True,
                total_atm_outflow=total_atm_outflow, findings=findings,
                actions=list(FRAUD_HIGH_RISK_ACTIONS),
            )

    # Signature 2: impossible travel — different cities too close together
    # in time for the same person to have travelled between them.
    for i in range(1, len(parsed)):
        prev_t, prev_city, prev_dt = parsed[i - 1]
        cur_t, cur_city, cur_dt = parsed[i]
        if prev_city and cur_city and prev_city != cur_city:
            gap_minutes = (cur_dt - prev_dt).total_seconds() / 60
            if 0 <= gap_minutes <= IMPOSSIBLE_TRAVEL_MINUTES:
                findings = [
                    f"card used in {prev_city} at {prev_dt:%H:%M} then {cur_city} at "
                    f"{cur_dt:%H:%M} - {gap_minutes:.0f} minutes apart, physically impossible"
                ]
                return CardFraudAssessmentResult(
                    verdict="high_risk_impossible_travel", is_high_risk=True,
                    total_atm_outflow=total_atm_outflow, findings=findings,
                    actions=list(FRAUD_HIGH_RISK_ACTIONS),
                )

    return CardFraudAssessmentResult(
        verdict="normal", is_high_risk=False, total_atm_outflow=total_atm_outflow,
        findings=["transaction pattern is consistent with normal usage"], actions=[],
    )


def render_card_fraud_facts(r: CardFraudAssessmentResult) -> str:
    lines = [
        "CARD FRAUD ASSESSMENT (authoritative — do not contradict)",
        f"- Verdict: {r.verdict} ({'high risk' if r.is_high_risk else 'not high risk'})",
        f"- Total ATM outflow across the transactions shown: {_rs(r.total_atm_outflow)}",
    ]
    lines += [f"- {f}" for f in r.findings]
    if r.actions:
        lines.append("- Do now: " + "; ".join(r.actions))
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# card retention guidance
# ════════════════════════════════════════════════════════════════════════

# (keywords to match in the on-screen reason text, category key, customer
# -facing label, whether the handling of the card while captured is
# uncertain — which changes the closing guidance line).
CAPTURE_REASON_RULES: list[tuple[tuple[str, ...], str, str, bool]] = [
    (("lost", "stolen", "hot-list", "hotlist", "hot list"), "hotlist", "hot-list capture", True),
    (("wrong pin", "incorrect pin", "pin retries", "pin attempts exceeded"), "wrong_pin", "wrong-PIN retention", True),
    (("power failure", "fault", "malfunction", "error", "jam"), "fault", "fault retention", False),
    (("left in the slot", "time-out", "timeout", "not collected", "unclaimed"), "timeout", "time-out retention", False),
]
_DEFAULT_CAPTURE_CATEGORY = ("other", "card retention", False)

_BANK_CODE_IN_PARENS_RE = re.compile(r"\(([^)]+)\)\s*$")


def _extract_bank_code(bank_field: str) -> str:
    """CustomerAtmProfile.bank is rendered like 'Sindh Bank (SNDB)' — pull
    the short code out of the trailing parens, falling back to the raw
    field if there isn't one."""
    m = _BANK_CODE_IN_PARENS_RE.search(bank_field)
    return m.group(1).strip() if m else bank_field.strip()


def _classify_capture_reason(reason: str) -> tuple[str, str, bool]:
    low = reason.lower()
    for keywords, category, label, uncertain in CAPTURE_REASON_RULES:
        if any(k in low for k in keywords):
            return category, label, uncertain
    return _DEFAULT_CAPTURE_CATEGORY


@dataclass
class CardRetentionInputs:
    profile: CustomerAtmProfile
    atm: AtmMachine              # the capturing ATM (atm.bank is the short code)
    atm_bank_full_name: str      # e.g. "Habib Metropolitan Bank"
    hours_since_capture: int
    capture_reason: str          # on-screen reason text
    hold_days: int


@dataclass
class CardRetentionGuidanceResult:
    category: str
    category_label: str
    own_bank: bool
    uncertain_handling: bool
    hold_days: int
    headline: str
    guidance: list[str] = field(default_factory=list)


def guide_card_retention(inputs: CardRetentionInputs) -> CardRetentionGuidanceResult:
    """Decision tree keyed on (a) capture reason category and (b) whether
    the capturing machine belongs to the customer's own bank."""
    category, label, uncertain = _classify_capture_reason(inputs.capture_reason)
    customer_bank_code = _extract_bank_code(inputs.profile.bank)
    own_bank = customer_bank_code == inputs.atm.bank

    guidance: list[str] = []
    if own_bank:
        headline = f"That was a {label}, at your own bank's machine."
        guidance.append(
            f"Since {inputs.atm_bank_full_name} ({inputs.atm.bank}) is your own bank, you can "
            "usually collect the card directly from that branch during working hours after "
            "verifying your identity - no interbank transfer needed."
        )
    else:
        headline = f"That was a {label}."
        guidance.append(
            f"This is the awkward case: the machine belongs to {inputs.atm_bank_full_name}, not "
            f"your bank, so that branch cannot hand a {customer_bank_code} card back to you over "
            "the counter. The captured card is logged and returned to the issuing bank through "
            "the interbank card-return process, which takes time."
        )
        guidance.append(
            f"The practical move is to stop waiting for the card. Call {customer_bank_code} now, "
            "report the card as captured, block it, and request a replacement. You keep your "
            "account and your money either way - only the piece of plastic is stuck."
        )

    if uncertain:
        guidance.append(
            "One thing worth doing whatever you decide: block the card first if the capture was "
            "a wrong-PIN or hot-list case, since in those situations you cannot be certain who "
            "last handled the card."
        )
    else:
        guidance.append(
            "Your money is not affected by the capture - the account stays open and your other "
            "channels (app, internet banking, Raast) keep working."
        )

    guidance.append(
        f"Bank policy: captured cards are held for {inputs.hold_days} working days at the branch "
        "before destruction, so do not leave it too long."
    )

    return CardRetentionGuidanceResult(
        category=category, category_label=label, own_bank=own_bank, uncertain_handling=uncertain,
        hold_days=inputs.hold_days, headline=headline, guidance=guidance,
    )


def render_card_retention_facts(r: CardRetentionGuidanceResult) -> str:
    lines = [
        "CARD RETENTION GUIDANCE (authoritative — do not contradict)",
        f"- {r.headline}",
        f"- Own bank's machine: {'yes' if r.own_bank else 'no'}",
        f"- Hold period before destruction: {r.hold_days} working days",
    ]
    lines += [f"- {g}" for g in r.guidance]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# failed transaction dispute
# ════════════════════════════════════════════════════════════════════════

@dataclass
class DisputeInputs:
    atm: AtmMachine
    date: str
    debited: int
    received: int
    journal_entry: str            # e.g. "DISPENSE FAILED - notes retracted to purge bin"
    reconciliation_excess: int    # excess cash found in the machine that day
    reversal_window_days: int     # bank policy days
    days_since_transaction: int   # how many days ago the transaction/complaint happened


@dataclass
class FailedTransactionDisputeResult:
    shortfall: int
    verdict: str                  # "no_claim" | "owed" | "insufficient_evidence"
    journal_supports_claim: bool
    reconciliation_corroborates: bool
    within_window: bool
    days_over: int
    findings: list[str] = field(default_factory=list)


def assess_failed_transaction_dispute(inputs: DisputeInputs) -> FailedTransactionDisputeResult:
    shortfall = inputs.debited - inputs.received
    journal_failed = "FAILED" in inputs.journal_entry.upper()
    reconciliation_corroborates = shortfall > 0 and inputs.reconciliation_excess == shortfall
    within_window = inputs.days_since_transaction <= inputs.reversal_window_days
    days_over = max(0, inputs.days_since_transaction - inputs.reversal_window_days)

    findings = [f"Debited: {_rs(inputs.debited)}, received: {_rs(inputs.received)}, shortfall: {_rs(shortfall)}"]

    if shortfall <= 0:
        verdict = "no_claim"
        findings.append("the journal and the machine reconciliation both balance - nothing to claim")
    elif journal_failed or reconciliation_corroborates:
        verdict = "owed"
        if journal_failed:
            findings.append("the ATM journal itself records a failed dispense")
        findings.append(
            f"the machine was found with {_rs(inputs.reconciliation_excess)} of excess cash that day"
            + (" - matches the shortfall exactly" if reconciliation_corroborates else "")
        )
    else:
        verdict = "insufficient_evidence"
        findings.append(
            "the journal records a successful dispense and the machine balanced - request CCTV "
            "and the full EJ before escalating further"
        )

    return FailedTransactionDisputeResult(
        shortfall=shortfall, verdict=verdict, journal_supports_claim=journal_failed,
        reconciliation_corroborates=reconciliation_corroborates, within_window=within_window,
        days_over=days_over, findings=findings,
    )


def render_dispute_facts(inputs: DisputeInputs, r: FailedTransactionDisputeResult) -> str:
    lines = [
        "FAILED TRANSACTION DISPUTE (authoritative — do not contradict)",
        f"- Verdict: {r.verdict}",
    ]
    lines += [f"- {f}" for f in r.findings]
    if r.verdict == "owed":
        lines.append(
            f"- File the dispute quoting date {inputs.date}, ATM ID {inputs.atm.atm_id}, and the "
            f"amount; policy is {inputs.reversal_window_days} working days for the reversal"
        )
        if r.within_window:
            lines.append(
                f"- {inputs.days_since_transaction} day(s) in, still inside the "
                f"{inputs.reversal_window_days}-day window"
            )
        else:
            lines.append(
                f"- {inputs.days_since_transaction} day(s) in, {r.days_over} day(s) past the "
                f"{inputs.reversal_window_days}-day window - escalate to the complaint unit"
            )
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# interbank settlement
# ════════════════════════════════════════════════════════════════════════

def _round_half_up(x: float) -> int:
    """Standard commercial (half-up) rounding — avoids Python's banker's
    rounding surprising us on exact .5 interchange totals."""
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


@dataclass
class InterbankSettlementInputs:
    bank_code: str
    on_us_count: int                # our cards on our ATMs — carries no interchange
    acquired_offus_count: int       # other banks' cards on our ATMs — we earn interchange
    issued_offus_count: int         # our cards on other banks' ATMs — we pay interchange
    interchange_rate: float         # Rs per off-us cash withdrawal


@dataclass
class InterbankSettlementResult:
    acquiring_income: int
    issuing_cost: int
    net_position: int               # negative = net payable
    net_payer: bool


def calculate_interbank_settlement(inputs: InterbankSettlementInputs) -> InterbankSettlementResult:
    acquiring_raw = inputs.acquired_offus_count * inputs.interchange_rate
    issuing_raw = inputs.issued_offus_count * inputs.interchange_rate
    acquiring_income = _round_half_up(acquiring_raw)
    issuing_cost = _round_half_up(issuing_raw)
    # Net is rounded from the raw (unrounded) difference, not from the two
    # already-rounded line items — matches the dataset's worked examples,
    # where the net can differ by a rupee from acquiring_income - issuing_cost.
    net_position = _round_half_up(acquiring_raw - issuing_raw)
    return InterbankSettlementResult(
        acquiring_income=acquiring_income, issuing_cost=issuing_cost,
        net_position=net_position, net_payer=net_position < 0,
    )


def render_interbank_settlement_facts(inputs: InterbankSettlementInputs, r: InterbankSettlementResult) -> str:
    lines = [
        "INTERBANK SETTLEMENT (authoritative — do not contradict)",
        f"- Net interchange position: {_rs(abs(r.net_position))} {'payable' if r.net_payer else 'receivable'}",
        f"- Acquiring income: {inputs.acquired_offus_count:,} x {_rs(inputs.interchange_rate)} = {_rs(r.acquiring_income)}",
        f"- Issuing cost: {inputs.issued_offus_count:,} x {_rs(inputs.interchange_rate)} = {_rs(r.issuing_cost)}",
        f"- Net: {r.net_position:,} PKR",
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# trend summary
# ════════════════════════════════════════════════════════════════════════

# CIT visits should land this many hours ahead of the observed peak hour
# so the machine is full going into it.
CIT_LEAD_HOURS = 2


@dataclass
class HourlyLogEntry:
    hour: str            # "HH:00"
    balance_after: int
    dispensed: int


@dataclass
class TrendSummaryInputs:
    atm: AtmMachine
    opening_balance: int
    log: list[HourlyLogEntry]


@dataclass
class TrendSummaryResult:
    total_dispensed: int
    closing_balance: int
    pct_of_load_dispensed: float
    peak_hour: str
    peak_amount: int
    cit_recommendation_hour: str


def summarize_trend(inputs: TrendSummaryInputs) -> TrendSummaryResult:
    total_dispensed = sum(e.dispensed for e in inputs.log)
    closing_balance = inputs.log[-1].balance_after if inputs.log else inputs.opening_balance
    pct = round(total_dispensed / inputs.opening_balance * 100, 1) if inputs.opening_balance else 0.0

    peak = max(inputs.log, key=lambda e: e.dispensed)
    peak_hour_int = int(peak.hour.split(":")[0])
    recommend_hour = (peak_hour_int - CIT_LEAD_HOURS) % 24

    return TrendSummaryResult(
        total_dispensed=total_dispensed, closing_balance=closing_balance,
        pct_of_load_dispensed=pct, peak_hour=peak.hour, peak_amount=peak.dispensed,
        cit_recommendation_hour=f"{recommend_hour:02d}:00",
    )


def render_trend_summary_facts(inputs: TrendSummaryInputs, r: TrendSummaryResult) -> str:
    return "\n".join([
        "TREND SUMMARY (authoritative — do not contradict)",
        f"- {inputs.atm.atm_id} dispensed {_rs(r.total_dispensed)}, running the balance from "
        f"{_rs(inputs.opening_balance)} to {_rs(r.closing_balance)} - {r.pct_of_load_dispensed}% "
        "of the load gone",
        f"- Peaks at {r.peak_hour} ({_rs(r.peak_amount)} in that hour)",
        f"- Schedule CIT visits before {r.cit_recommendation_hour} so the machine is full ahead of "
        "the peak",
    ])


# ════════════════════════════════════════════════════════════════════════
# uptime SLA check
# ════════════════════════════════════════════════════════════════════════

MINUTES_PER_DAY = 1440


@dataclass
class DowntimeIncident:
    code: str
    description: str
    minutes: int


@dataclass
class UptimeSlaInputs:
    atm: AtmMachine
    period_days: int
    sla_pct: float            # e.g. 98.0 for 98%
    incidents: list[DowntimeIncident]


@dataclass
class UptimeSlaCheckResult:
    period_minutes: int
    total_downtime_minutes: int
    uptime_pct: float
    allowed_downtime_minutes: int
    breach: bool
    breach_minutes: int       # positive if in breach, 0 otherwise
    biggest_contributor: DowntimeIncident | None


def check_uptime_sla(inputs: UptimeSlaInputs) -> UptimeSlaCheckResult:
    period_minutes = inputs.period_days * MINUTES_PER_DAY
    total_downtime = sum(i.minutes for i in inputs.incidents)
    uptime_pct = round((period_minutes - total_downtime) / period_minutes * 100, 2) if period_minutes else 0.0
    allowed_downtime = round(period_minutes * (1 - inputs.sla_pct / 100))
    breach = total_downtime > allowed_downtime
    breach_minutes = max(0, total_downtime - allowed_downtime)
    biggest = max(inputs.incidents, key=lambda i: i.minutes) if inputs.incidents else None

    return UptimeSlaCheckResult(
        period_minutes=period_minutes, total_downtime_minutes=total_downtime,
        uptime_pct=uptime_pct, allowed_downtime_minutes=allowed_downtime,
        breach=breach, breach_minutes=breach_minutes, biggest_contributor=biggest,
    )


def render_uptime_sla_facts(inputs: UptimeSlaInputs, r: UptimeSlaCheckResult) -> str:
    lines = [
        "UPTIME SLA CHECK (authoritative — do not contradict)",
        f"- Uptime for {inputs.atm.atm_id} over {inputs.period_days} days: {r.uptime_pct}% "
        f"(SLA {inputs.sla_pct}%)",
        f"- Total downtime: {r.total_downtime_minutes:,} minutes across {len(inputs.incidents)} incident(s)",
        f"- Downtime allowed under SLA: {r.allowed_downtime_minutes:,} minutes",
        f"- Result: {'BREACH by ' + format(r.breach_minutes, ',') + ' minutes' if r.breach else 'within SLA'}",
    ]
    if r.biggest_contributor:
        b = r.biggest_contributor
        lines.append(f"- Single biggest contributor: {b.code} ({b.description}) at {b.minutes:,} minutes")
    return "\n".join(lines)
