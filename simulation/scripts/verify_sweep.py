"""Verification sweep: run every one of the 51 task types against the live
model and check whether the narration contradicts the computed facts.

The deterministic half is already pinned by 159 unit tests. What is NOT pinned
is whether the model, handed an authoritative facts block, then writes
something that disagrees with it. This sweep answers that per task type, so
you know which ones are safe to demo live.

Grading
  GREEN  narration agrees with the facts on every checked dimension
  AMBER  narration introduces a figure not present in the facts. Often a
         harmless derivation (a+b, a percentage) — needs a human read
  RED    narration asserts the opposite of a computed verdict, or the model
         failed to produce anything

Usage:
    python scripts/verify_sweep.py              # all 51
    python scripts/verify_sweep.py --only loans # or: --only atm
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.atm import TASK_REGISTRY as ATM_TASKS
from agents.atm import cash_mechanics, customer_money, judgment, network_ops
from agents.atm.agent import AtmAgent
from agents.atm.entities import AtmMachine, Cassette, CustomerAtmProfile, Transaction
from dispatch import build_args, call_render, to_jsonable

from app.agents.loans.agent import LoanAgent
from app.agents.loans.models import (
    BureauFacility, GoldOffer, LoanProductOption, LoanQuote, RunningLoan, SalaryAdvanceProduct,
)

ATM_MODULES = {"cash_mechanics": cash_mechanics, "network_ops": network_ops,
               "customer_money": customer_money, "judgment": judgment}

# ---------------------------------------------------------------- fixtures

MACHINE = dict(atm_id="HBL-LHR-6620", bank="HBL", location="Gulberg III, Lahore",
               area_type="commercial district", model="NCR SelfServ 26", status="ONLINE",
               cassettes=[dict(denom=5000, notes=900, capacity=2000, status="OK", note=""),
                          dict(denom=1000, notes=1600, capacity=2500, status="OK", note=""),
                          dict(denom=500, notes=700, capacity=2000, status="LOW", note="below low-cash threshold")])
FAULTED = dict(MACHINE, atm_id="MEBL-KDU-6892",
               cassettes=[dict(denom=5000, notes=780, capacity=3000, status="FAULT", note="pick failure, cassette locked out"),
                          dict(denom=1000, notes=2250, capacity=3000, status="OK", note=""),
                          dict(denom=500, notes=60, capacity=2000, status="LOW", note="below low-cash threshold")])
PROFILE = dict(name="Zainab Malik", bank="Habib Bank Limited (HBL)", account_type="Salary Account",
               account_number="PK42HBL7719340025", available_balance=214_600, card_network="Mastercard",
               card_tier="Mastercard Platinum Debit", card_last4="4417", per_txn_limit=50_000,
               daily_limit=150_000, free_offus_left=4, credit_line=300_000, credit_utilised=41_200,
               cash_advance_sub_limit_pct=0.40, cash_advance_drawn=0)
FLEET_ROWS = [("HBL-LHR-6620", "Gulberg III", 3_180_000, 210_000),
              ("UBL-LHR-3382", "Liberty Market", 780_000, 165_000),
              ("MCB-LHR-9014", "Model Town", 6_400_000, 96_000),
              ("MEZN-LHR-4415", "Saddar", 210_000, 140_000)]
TXNS = [dict(date="2026-07-30", time="02:04", description="ATM withdrawal Lahore - Ferozepur Road", channel="ATM", amount=-50_000),
        dict(date="2026-07-30", time="02:09", description="ATM withdrawal Lahore - Kalma Chowk", channel="ATM", amount=-50_000),
        dict(date="2026-07-30", time="11:20", description="Al-Fatah Store", channel="POS", amount=-8_400),
        dict(date="2026-07-29", time="18:02", description="ATM withdrawal Lahore - Garden Town", channel="ATM", amount=-20_000)]

ATM_FIXTURES = {
    "cash_runout_forecast": {"machine": MACHINE, "inp": dict(current_time="14:00", dispense_rate_per_hour=210_000,
                              low_cash_threshold=150_000, cit_time="18:00", cit_hours_from_now=4)},
    "cassette_status_triage": {"machine": FAULTED},
    "cash_reconciliation": {"opening_cash": 1_967_000, "loaded_this_cycle": 11_400_000,
                            "dispensed_ej": 10_030_000, "physical_count": 3_337_000},
    "denomination_mix_planning": {"inp": dict(denominations=[5000, 1000],
                                   histogram=[[1000, 175], [2000, 47], [5000, 156], [10000, 203], [20000, 241]],
                                   cassette_capacity_notes={5000: 3000, 1000: 5000})},
    "denomination_dispensability": {"machine": MACHINE, "requested_amount": 4_500},
    "cash_load_planning": {"inp": dict(avg_daily_dispense=4_000_000, days_until_cit=6, policy_buffer_pct=0.15,
                            cassette_specs=[[5000, 1500], [1000, 2500]],
                            withdrawal_mix={5000: 0.30, 1000: 0.55, 500: 0.15})},
    "withdrawal_feasibility": {"profile": PROFILE, "machine": MACHINE, "requested_amount": 50_000,
                               "transactions": TXNS},
    "daily_limit_remaining": {"profile": PROFILE, "transactions": TXNS, "requested_amount": 10_000},
    "alert_triage": {"alerts": [dict(atm_id="FWBL-MUX-9514", location="Shahi Bazaar", alert_code="CASH_LOW", description="cash below threshold", open_minutes=111),
                                dict(atm_id="NBP-MUX-8535", location="Ring Road", alert_code="DISPENSER_FAULT", description="dispenser hardware fault", open_minutes=72),
                                dict(atm_id="HBL-MUX-4499", location="GT Road", alert_code="DOOR_OPEN", description="safe door open unscheduled", open_minutes=194)]},
    "replenishment_priority": {"candidates": [dict(atm_id=a, location=l, cash_on_hand=c, dispense_rate_per_hr=r) for a, l, c, r in FLEET_ROWS],
                               "window_hours": 12},
    "cit_route_planning": {"candidates": [dict(atm_id=a, distance_km=8.4 + i * 4, cash_left=c, hours_to_empty=round(c / r, 1), requested_load=3_000_000)
                                          for i, (a, l, c, r) in enumerate(FLEET_ROWS)], "cap": 15_000_000},
    "demand_forecast": {"inp": dict(machine=MACHINE, week_dispensed=[["Monday", 1_130_000], ["Tuesday", 950_000], ["Wednesday", 880_000],
                                     ["Thursday", 930_000], ["Friday", 1_320_000], ["Saturday", 1_000_000], ["Sunday", 730_000]],
                                     forecast_day="Thursday", seasonal_label="pre Eid-ul-Fitr", seasonal_multiplier=2.1,
                                     machine_capacity=7_000_000)},
    "eod_position_report": {"sites": [dict(atm_id=a, location=l, status="ONLINE" if i % 3 else "COMMS_DOWN",
                                           cash_on_hand=c, dispensed_today=r * 12) for i, (a, l, c, r) in enumerate(FLEET_ROWS)],
                            "low_cash_threshold": 500_000},
    "growth_analysis": {"inp": dict(atm_id="KMBL-ATD-6018", metric_label="Off-us withdrawals", old_period="June",
                                    old_value=168_210_000, new_period="July", new_value=176_250_000, is_money=True)},
    "share_analysis": {"rows": [dict(atm_id=a, location=l, value=c) for a, l, c, _ in FLEET_ROWS],
                       "target_atm_id": "UBL-LHR-3382"},
    "ranking": {"rows": [dict(atm_id=a, location=l, value=float(c)) for a, l, c, _ in FLEET_ROWS],
                "metric_label": "downtime minutes", "higher_is_worse": True},
    "cash_carrying_cost": {"p": dict(atm=MACHINE, idle_cash=7_150_000, daily_dispense=3_550_000,
                                     annual_rate=0.175, cit_trip_cost=12_000, window_days=90)},
    "surge_capacity_planning": {"p": dict(atm=MACHINE, normal_daily_dispense=4_790_000, season_name="Ramzan",
                                          demand_multiplier=1.35, closed_days=3, max_capacity=20_000_000)},
    "atm_fee_calculation": {"p": dict(free_offus_per_month=4, offus_already_made=7, planned_offus_count=6,
                                      fee_per_txn=23.59, fed_pct=0.0, per_txn_limit=30_000, amount_needed=80_000)},
    "atm_recommendation": {"p": dict(customer=PROFILE, amount_needed=40_000, off_us_charge=18.5,
                            candidates=[dict(atm_id="HBL-LHR-6620", bank="HBL", location="Gulberg III", distance_km=1.9,
                                             status="ONLINE", cash_on_hand=3_180_000, denominations=[5000, 1000], queue=7),
                                        dict(atm_id="SCB-LHR-6090", bank="SCB", location="Main Blvd", distance_km=1.1,
                                             status="OFFLINE", cash_on_hand=10_876_000, denominations=[5000, 1000], queue=3)])},
    "cash_advance_assessment": {"p": dict(customer=PROFILE, atm=MACHINE, requested_amount=20_000, fee_pct=0.025,
                                          fee_min=750, monthly_markup_pct=0.035, days_held=30)},
    "cash_affordability_check": {"p": dict(customer=PROFILE, withdrawal_amount=75_000,
                                  bills=[dict(label="house rent", amount=16_400, due_date="2026-08-04"),
                                         dict(label="LESCO bill", amount=28_400, due_date="2026-08-09")],
                                  salary_date="2026-08-01")},
    "monthly_atm_cost_summary": {"a": dict(on_us_count=2, off_us_count=11, inquiries_count=15,
                                           avg_withdrawal=18_300, free_offus_allowance=3,
                                           offus_fee=18.5, inquiry_fee=5.0)},
    "spend_pattern_summary": {"profile": PROFILE, "transactions": TXNS},
    "travel_cash_planning": {"p": dict(customer=PROFILE, days=10, daily_spend=5_500,
                                       working_atms_at_destination=0, off_us_charge=23.59)},
    "fault_root_cause": {"inputs": dict(atm=MACHINE, fault_code="R0005", down_minutes=285, failed_transactions=50,
                          vendor_table={"R0005": "receipt printer paper out", "P0022": "power failure - UPS drained",
                                        "H0013": "cash-out sensor mismatch"})},
    "security_anomaly_assessment": {"obs": dict(atm=MACHINE, event_count=37, window_start="04:00", window_end="04:20",
                                     total_amount=740_000, unmatched_authorisation=True, safe_door_opened_time="03:24",
                                     cit_visit_scheduled=False)},
    "card_fraud_assessment": {"inputs": dict(profile=PROFILE, transactions=TXNS)},
    "card_retention_guidance": {"inputs": dict(profile=PROFILE, atm=dict(MACHINE, atm_id="UBL-LHR-3382", bank="UBL"),
                                 atm_bank_full_name="United Bank Limited", hours_since_capture=14,
                                 capture_reason="card left in slot, timed out and retracted", hold_days=5)},
    "failed_transaction_dispute": {"inputs": dict(atm=MACHINE, date="2026-07-29", debited=25_000, received=0,
                                    journal_entry="DISPENSE FAILED - notes retracted to purge bin",
                                    reconciliation_excess=25_000, reversal_window_days=7, days_since_transaction=1)},
    "interbank_settlement": {"inputs": dict(bank_code="HBL", on_us_count=46_407, acquired_offus_count=6_244,
                                            issued_offus_count=13_057, interchange_rate=25.0)},
    "trend_summary": {"inputs": dict(atm=MACHINE, opening_balance=6_100_000,
                       log=[dict(hour=f"{h:02d}:00", balance_after=6_100_000 - (i + 1) * 480_000, dispensed=480_000)
                            for i, h in enumerate(range(8, 20))])},
    "uptime_sla_check": {"inputs": dict(atm=MACHINE, period_days=31, sla_pct=98.0,
                          incidents=[dict(code="H0013", description="cash-out sensor mismatch", minutes=852),
                                     dict(code="N0009", description="note quality reject rate high", minutes=721),
                                     dict(code="D0011", description="dispenser fault - note pick failure", minutes=254)])},
}

RUNNING_LOAN = RunningLoan(950_000, 0.318, "reducing", 24, 14, 768_432, 54_001)
SHELF = [
    LoanProductOption("Personal Instalment Loan", 50_000, 3_000_000, [12, 24, 36, 48], 0.31, "reducing", 0.02, min_income=50_000),
    LoanProductOption("Salary Advance", 20_000, 500_000, [1, 3, 6], 0.27, "flat", 0.01, min_income=40_000, requires_salary_account=True),
    LoanProductOption("Deposit-backed Finance", 25_000, 10_000_000, [12, 24, 36], 0.16, "reducing", 0.005, requires_deposit=True),
    LoanProductOption("Gold-backed Finance", 30_000, 5_000_000, [12, 24, 36], 0.24, "reducing", 0.015, requires_gold=True),
]

def loan_calls(agent: LoanAgent):
    return {
        "relationship_scoring": lambda: agent.relationship_scoring("C007"),
        "low_cash_detection": lambda: agent.low_cash_detection("C007"),
        "proactive_offer_decision": lambda: agent.assess("C007"),
        "decline_with_alternatives": lambda: agent.assess("C038"),
        "portfolio_triage": lambda: agent.portfolio_triage(),
        "dbr_calculation": lambda: agent.dbr_calculation(90_000, 5_500, 11_224),
        "max_affordable_loan": lambda: agent.max_affordable_loan("C007"),
        "emi_calculation": lambda: agent.emi_calculation(2_340_000, 0.182, 120, "reducing"),
        "ecib_report_reading": lambda: agent.ecib_report([
            BureauFacility("Consumer Durable", "KMBL", 2_050_000, "90+ DPD"),
            BureauFacility("Microfinance Loan", "BAFL", 910_000, "90+ DPD"),
            BureauFacility("Microfinance Loan", "MCB", 1_910_000, "Written off")]),
        "delinquency_risk_grading": lambda: agent.delinquency_risk("C007", RUNNING_LOAN, [2, 2, 14, 14]),
        "restructuring_assessment": lambda: agent.restructuring("C007", RUNNING_LOAN, 133_000, 24),
        "risk_based_pricing": lambda: agent.risk_pricing("C007", 27.3, 2.5, 5.5, 396_000, 12),
        "topup_eligibility": lambda: agent.topup("C007", RUNNING_LOAN, 0, 540_000),
        "loan_offer_comparison": lambda: agent.offer_comparison(1_230_000, 36, [
            LoanQuote("Offer A", 0.269, "reducing", 0.015),
            LoanQuote("Offer B", 0.185, "flat", 0.015),
            LoanQuote("Offer C", 0.161, "flat", 0.005)]),
        "early_settlement_analysis": lambda: agent.early_settlement(RUNNING_LOAN, 0.02),
        "gold_loan_sizing": lambda: agent.gold_loan(GoldOffer(12.8, 21, 244_000, 0.70), 1_710_000, 0.231, 24),
        "salary_advance_assessment": lambda: agent.salary_advance("C007", SalaryAdvanceProduct(1.0, 0.02, 0.273, 6), 80_000),
        "product_recommendation": lambda: agent.product_recommendation("C007", 1_180_000, SHELF, deposit_amount=1_500_000),
    }

# ---------------------------------------------------------------- checks

NUM = re.compile(r"(?<![\w.])(\d[\d,]{2,})(?![\w])")

def numbers(text: str) -> set[str]:
    """Digit-groups, normalised. Indian and Western grouping both collapse to
    the same key so 13,50,000 and 1,350,000 compare equal."""
    return {m.replace(",", "").lstrip("0") or "0" for m in NUM.findall(text or "")}

# verdict words that must not be inverted, as (facts_marker, forbidden_in_narrative)
FLIPS = [
    ("action': 'OFFER", ["decline the", "we decline", "no unsecured offer", "not eligible"]),
    ("action': 'DECLINE", ["approve the loan", "we can offer", "proceed with the offer"]),
    ("survives_to_cit': True", ["will not survive", "runs dry before", "empties before the"]),
    ("survives_to_cit': False", ["survives to", "no emergency run", "comfortably covers"]),
    ("breach': True", ["within sla", "no breach", "meets the sla"]),
    ("breach': False", ["breach", "in breach"]),
    ("is_attack': True", ["no attack", "benign", "nothing suspicious", "normal activity"]),
    ("is_attack': False", ["jackpotting", "under attack", "confirmed attack"]),
    ("approved': True", ["decline", "not approvable", "reject"]),
    ("approved': False", ["approved", "we can approve"]),
    ("worth_it': True", ["not worth", "keep the current"]),
    ("worth_it': False", ["worth doing", "switch to smaller"]),
    ("can_lend_unsecured': True", ["cannot lend unsecured", "secured products only"]),
    ("can_lend_unsecured': False", ["can lend unsecured", "unsecured is fine"]),
]

def grade(facts_repr: str, narrative: str):
    issues, level = [], "GREEN"
    if not narrative or not narrative.strip():
        return "RED", ["model produced nothing"]

    low = narrative.lower()
    for marker, forbidden in FLIPS:
        if marker.lower() in facts_repr.lower():
            for bad in forbidden:
                if bad in low:
                    issues.append(f"contradicts facts ({marker.split(chr(39))[0].strip()}): says {bad!r}")
                    level = "RED"

    extra = numbers(narrative) - numbers(facts_repr)
    # ignore small integers — counts, tenors, days, percentages restated in prose
    extra = {n for n in extra if len(n) >= 4}
    if extra:
        issues.append(f"figures not in facts: {sorted(extra)[:6]}")
        if level != "RED":
            level = "AMBER"
    return level, issues

# ---------------------------------------------------------------- run

def run_atm(name, agent):
    mod_name, fn_name, render_name = ATM_TASKS[name]
    module = ATM_MODULES[mod_name]
    compute, render = getattr(module, fn_name), getattr(module, render_name)
    body = ATM_FIXTURES.get(name)
    if body is None:
        return None, None, ["no fixture"]
    kwargs = build_args(compute, dict(body))
    facts = compute(**kwargs)
    prompt = call_render(render, kwargs, facts)
    narrative, _src = agent.narrate(prompt, name, use_llm=True)
    return json.dumps(to_jsonable(facts), default=str), narrative, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["loans", "atm"], default=None)
    ap.add_argument("--out", default=str(ROOT / "verification_report.json"))
    args = ap.parse_args()

    loan_agent, atm_agent = LoanAgent(), AtmAgent()
    results = []

    if args.only in (None, "loans"):
        for name, call in loan_calls(loan_agent).items():
            t0 = time.time()
            try:
                res = call()
                facts_repr = json.dumps(to_jsonable(res.facts), default=str)
                level, issues = grade(facts_repr, res.narrative)
                if res.narrative_source != "llm":
                    level, issues = "RED", ["model unreachable — fell back to template"]
            except Exception as exc:
                level, issues, res = "RED", [f"{type(exc).__name__}: {exc}"], None
            results.append(dict(domain="loans", task=name, level=level, issues=issues,
                                narrative=(res.narrative if res else None),
                                secs=round(time.time() - t0, 1)))
            print(f"  {level:<5} loans/{name}  {issues or ''}")

    if args.only in (None, "atm"):
        for name in ATM_TASKS:
            t0 = time.time()
            try:
                facts_repr, narrative, err = run_atm(name, atm_agent)
                if err:
                    level, issues = "RED", err
                else:
                    level, issues = grade(facts_repr, narrative)
            except Exception as exc:
                level, issues, narrative = "RED", [f"{type(exc).__name__}: {exc}"], None
            results.append(dict(domain="atm", task=name, level=level, issues=issues,
                                narrative=narrative, secs=round(time.time() - t0, 1)))
            print(f"  {level:<5} atm/{name}  {issues or ''}")

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    tally = {k: sum(1 for r in results if r["level"] == k) for k in ("GREEN", "AMBER", "RED")}
    print(f"\n{tally['GREEN']} green · {tally['AMBER']} amber · {tally['RED']} red   ({len(results)} tasks)")
    print(f"report: {args.out}")

if __name__ == "__main__":
    main()
