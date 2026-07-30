"""Tests for the 10 loan task types added on top of the original 8
(relationship scoring / stress / offer-decline-triage / EMI-DBR-max-loan).

Worked examples are copied verbatim from merged_finetune_train.jsonl /
merged_finetune_val.jsonl (synthetic_loan_ops + synthetic_loan_customer).
Money figures use pytest.approx with a small tolerance since the dataset's
own reference numbers carry their own independent rounding.
"""
from __future__ import annotations

import re

import pytest

from app.agents.loans.models import (
    BureauFacility,
    CustomerProfile,
    GoldOffer,
    LoanProductOption,
    LoanQuote,
    RunningLoan,
    SalaryAdvanceProduct,
)
from app.agents.loans.agent import LoanAgent
from app.agents.loans.policy import (
    analyze_early_settlement,
    assess_restructuring,
    assess_salary_advance,
    assess_topup,
    compare_loan_offers,
    grade_delinquency_risk,
    price_by_risk,
    read_ecib_report,
    recommend_product,
    size_gold_loan,
)


def make_profile(**overrides) -> CustomerProfile:
    base = dict(
        customer_id="T001", name="Test Customer", age=35,
        bank="Habib Bank Limited", bank_code="HBL", city="Lahore",
        employment="Careem captain (gig)", net_monthly_income=90_000,
        account_age_years=5.0, salary_months_12=12, avg_balance_6m=120_000,
        previous_loan="clean", cheque_bounce_12m=False,
        ecib_dpd90_24m=False, ecib_writeoff=False, obligations=[],
        eom_balances=[], days_below_5k_30d=0, salary_to_low_gap_days=None,
    )
    base.update(overrides)
    return CustomerProfile(**base)


# ── eCIB report reading ──

def test_ecib_writeoff_blocks_unsecured():
    lines = [
        BureauFacility("Consumer Durable", "KMBL", 2_050_000, "90+ DPD"),
        BureauFacility("Microfinance Loan", "BAFL", 910_000, "90+ DPD"),
        BureauFacility("Microfinance Loan", "MCB", 1_910_000, "Written off"),
    ]
    a = read_ecib_report(lines)
    assert a.active_count == 2
    assert a.has_writeoff is True
    assert a.can_lend_unsecured is False
    assert a.verdict == "writeoff_block"


def test_ecib_active_delinquency_without_writeoff():
    lines = [
        BureauFacility("Personal Loan", "HBL", 530_000, "30 DPD"),
        BureauFacility("Auto Finance", "UBL", 940_000, "30 DPD"),
    ]
    a = read_ecib_report(lines)
    assert a.active_count == 2
    assert a.has_writeoff is False
    assert a.has_active_delinquency is True
    assert a.can_lend_unsecured is False
    assert a.verdict == "active_delinquency"


def test_ecib_clean_file_can_lend():
    lines = [BureauFacility("Auto Finance", "MCB", 1_120_000, "Closed - repaid")]
    a = read_ecib_report(lines)
    assert a.can_lend_unsecured is True
    assert a.verdict == "clean"


# ── delinquency risk grading ──

def test_delinquency_high_risk_lengthening_delays():
    p = make_profile(net_monthly_income=99_000, days_below_5k_30d=22, cheque_bounce_12m=False)
    loan = RunningLoan(950_000, 0.318, "reducing", 24, 11, 800_000, 54_001)
    g = grade_delinquency_risk(p, loan, [2, 2, 14, 14])
    assert g.score == 7
    assert g.grade == "HIGH"


def test_delinquency_medium_risk_no_lengthening():
    p = make_profile(net_monthly_income=68_000, days_below_5k_30d=20, cheque_bounce_12m=False)
    loan = RunningLoan(300_000, 0.282, "reducing", 24, 6, 280_000, 16_497)
    g = grade_delinquency_risk(p, loan, [9, 0, 0, 9])
    assert g.score == 4
    assert g.grade == "MEDIUM"
    assert any("2 of last 4" in r for r in g.reasons)
    assert any("20 days" in r for r in g.reasons)


def test_delinquency_low_risk_clean_file():
    p = make_profile(net_monthly_income=200_000, days_below_5k_30d=1)
    loan = RunningLoan(500_000, 0.28, "reducing", 24, 20, 100_000, 25_000)
    g = grade_delinquency_risk(p, loan, [0, 0, 0, 0])
    assert g.score == 0
    assert g.grade == "LOW"


# ── restructuring assessment ──

def test_restructuring_fixes_affordability():
    p = make_profile(net_monthly_income=190_000, obligations=[])
    loan = RunningLoan(950_000, 0.319, "reducing", 48, 15, 768_432, 35_263)
    r = assess_restructuring(p, loan, new_income=133_000, extension_months=24)
    assert r.new_instalment == pytest.approx(26_329, rel=0.01)
    assert r.fixes_affordability is True
    assert r.extra_markup_cost == pytest.approx(337_060, rel=0.05)


def test_restructuring_does_not_fix_severe_income_shock():
    p = make_profile(net_monthly_income=35_000, obligations=[])
    from app.agents.loans.models import Obligation
    p.obligations = [Obligation("existing personal loan EMI", 1_500)]
    loan = RunningLoan(2_340_000, 0.291, "reducing", 48, 23, 1_543_048, 83_034)
    r = assess_restructuring(p, loan, new_income=17_500, extension_months=18)
    assert r.new_instalment == pytest.approx(58_185, rel=0.01)
    assert r.fixes_affordability is False


# ── top-up eligibility ──

def test_topup_fails_seasoning_gate():
    p = make_profile(net_monthly_income=183_000)
    loan = RunningLoan(1_170_000, 0.261, "reducing", 36, 11, 902_898, 47_202)
    t = assess_topup(p, loan, late_in_12m=0, request_amount=580_000)
    assert t.approved is False
    assert "11" in t.reason and "12" in t.reason


def test_topup_fails_late_instalment_gate():
    p = make_profile(net_monthly_income=18_000)
    loan = RunningLoan(1_980_000, 0.316, "reducing", 36, 20, 1_108_567, 85_799)
    t = assess_topup(p, loan, late_in_12m=1, request_amount=510_000)
    assert t.approved is False
    assert "zero-late" in t.reason


def test_topup_approved_within_dbr_cap():
    p = make_profile(net_monthly_income=204_000)
    loan = RunningLoan(470_000, 0.271, "reducing", 36, 15, 318_489, 19_213)
    t = assess_topup(p, loan, late_in_12m=0, request_amount=540_000)
    assert t.approved is True
    assert t.new_principal == 858_489
    assert t.new_instalment == pytest.approx(51_789, rel=0.02)
    assert t.dbr_pct == pytest.approx(25.4, abs=0.5)


def test_topup_amount_breaches_dbr_cap():
    p = make_profile(net_monthly_income=113_000)
    from app.agents.loans.models import Obligation
    p.obligations = [
        Obligation("committee (BC) contribution", 9_500),
        Obligation("existing personal loan EMI", 9_500),
    ]
    loan = RunningLoan(310_000, 0.317, "reducing", 24, 12, 179_053, 17_605)
    t = assess_topup(p, loan, late_in_12m=0, request_amount=570_000)
    assert t.approved is False
    assert t.dbr_pct == pytest.approx(82.0, abs=1.0)


# ── loan offer comparison ──

def test_loan_offer_comparison_picks_cheapest():
    quotes = [
        LoanQuote("Offer A", 0.269, "reducing", 0.015),
        LoanQuote("Offer B", 0.185, "flat", 0.015),
        LoanQuote("Offer C", 0.161, "flat", 0.005),
    ]
    costs = compare_loan_offers(1_230_000, 36, quotes)
    assert costs[0].label == "Offer A"
    assert costs[0].total_cost == pytest.approx(1_823_816, rel=0.005)


# ── early settlement ──

def test_early_settlement_worth_it():
    loan = RunningLoan(1_820_000, 0.327, "reducing", 36, 11, 1_436_319, 79_978)
    r = analyze_early_settlement(loan, 0.02)
    assert r.cost_to_continue == 1_999_450
    assert r.worth_it is True
    assert r.savings == pytest.approx(534_400, rel=0.01)


def test_early_settlement_not_worth_it_late_in_loan():
    loan = RunningLoan(1_340_000, 0.28, "reducing", 36, 35, 54_163, 55_427)
    r = analyze_early_settlement(loan, 0.05)
    assert r.worth_it is False


# ── gold loan sizing ──

def test_gold_loan_fully_covers_request():
    gold = GoldOffer(weight_tola=12.8, purity_k=21, rate_per_tola_24k=244_000, ltv_pct=0.70)
    r = size_gold_loan(gold, requested_amount=1_710_000, annual_rate=0.231, tenor_months=24)
    assert r.assessed_value == 2_732_800
    assert r.max_loan == 1_912_960
    assert r.granted_amount == 1_710_000
    assert r.shortfall == 0
    assert r.instalment == pytest.approx(89_643, rel=0.01)


def test_gold_loan_short_of_request():
    gold = GoldOffer(weight_tola=2.1, purity_k=24, rate_per_tola_24k=272_000, ltv_pct=0.70)
    r = size_gold_loan(gold, requested_amount=500_000, annual_rate=0.238, tenor_months=36)
    assert r.max_loan == 399_840
    assert r.granted_amount == 399_840
    assert r.shortfall == pytest.approx(100_160, abs=1)


# ── salary advance ──

def test_salary_advance_approved():
    p = make_profile(net_monthly_income=214_000, salary_months_12=10, days_below_5k_30d=25)
    product = SalaryAdvanceProduct(limit_multiple=1.0, fee_pct=0.02, annual_rate=0.273, tenor_months=6)
    r = assess_salary_advance(p, product, requested_amount=80_000)
    assert r.approved is True
    assert r.granted_amount == 80_000
    assert r.instalment == pytest.approx(15_153, rel=0.01)
    assert r.total_cost == pytest.approx(12_520, rel=0.01)


def test_salary_advance_declined_irregular_salary():
    p = make_profile(net_monthly_income=55_000, salary_months_12=4)
    product = SalaryAdvanceProduct(limit_multiple=2.0, fee_pct=0.02, annual_rate=0.255, tenor_months=1)
    r = assess_salary_advance(p, product, requested_amount=33_000)
    assert r.approved is False


# ── product recommendation ──

SHELF = [
    LoanProductOption("Personal Instalment Loan", 50_000, 3_000_000, [12, 24, 36, 48], 0.31, "reducing", 0.02, min_income=50_000),
    LoanProductOption("Salary Advance", 20_000, 500_000, [1, 3, 6], 0.27, "flat", 0.01, min_income=40_000, requires_salary_account=True),
    LoanProductOption("Deposit-backed Finance", 25_000, 10_000_000, [12, 24, 36], 0.16, "reducing", 0.005, requires_deposit=True),
    LoanProductOption("Gold-backed Finance", 30_000, 5_000_000, [12, 24, 36], 0.24, "reducing", 0.015, requires_gold=True),
]


def test_product_recommendation_picks_deposit_backed():
    from app.agents.loans.models import Obligation
    p = make_profile(net_monthly_income=231_000, obligations=[])
    rec = recommend_product(p, need_amount=1_180_000, shelf=SHELF, deposit_amount=1_500_000)
    assert rec.best is not None
    assert rec.best.name == "Deposit-backed Finance"


def test_product_recommendation_eliminates_unmet_requirements():
    p = make_profile(net_monthly_income=380_000)
    rec = recommend_product(p, need_amount=900_000, shelf=SHELF, has_gold=False, deposit_amount=0)
    names_eliminated = {name for name, _ in rec.eliminated}
    assert "Gold-backed Finance" in names_eliminated
    assert "Deposit-backed Finance" in names_eliminated


# ── portfolio triage prompt (regression: the live model fabricated counts
#    and invented customer names when the facts block was under-specified) ──

def _triage_facts_block(agent):
    """The deterministic facts the narrator is handed — i.e. the template
    narrative, which is the facts block verbatim."""
    return agent.portfolio_triage(use_llm=False).narrative


def test_portfolio_triage_states_exact_counts():
    agent = LoanAgent()
    queues = agent.scan()
    text = _triage_facts_block(agent)
    total = sum(len(v) for v in queues.values())
    for action in ("OFFER", "MONITOR", "DECLINE"):
        # counts must be stated explicitly, as "<n> of <total>"
        assert f"{action}: {len(queues[action])} of {total}" in text


def test_portfolio_triage_avoids_count_percentage_collision():
    """A bare percentage next to a count made the model read the percentage
    as the count. The facts block should carry no '%' at all."""
    assert "%" not in _triage_facts_block(LoanAgent())


def test_portfolio_triage_names_only_supplied_customers():
    """Every customer id the narrator is shown must be a real one, and the
    MONITOR queue must not be enumerated (that gap is what it filled with
    invented names)."""
    agent = LoanAgent()
    queues = agent.scan()
    text = _triage_facts_block(agent)
    real_ids = {c.customer_id for c in agent.customers}
    shown = set(re.findall(r"\((C\d{3})\)", text))
    assert shown, "expected at least the priority offers to be named"
    assert shown <= real_ids
    monitor_ids = {c.customer_id for c, _ in queues["MONITOR"]}
    assert not (shown & monitor_ids), "MONITOR queue should not be enumerated by name"


# ── risk-based pricing ──

def test_risk_pricing_strong_band_base_rate():
    p = make_profile(
        account_age_years=1.4, salary_months_12=11, avg_balance_6m=440_401,
        previous_loan="late_1_2", cheque_bounce_12m=False,
    )
    r = price_by_risk(p, base_rate=27.3, acceptable_markup_pts=2.5, thin_markup_pts=5.5,
                       requested_amount=396_000, tenor_months=12)
    assert r.score.band == "STRONG"
    assert r.annual_rate == 27.3
    assert r.instalment == pytest.approx(38_081, rel=0.01)


def test_risk_pricing_thin_band_gets_markup():
    p = make_profile(
        account_age_years=4.3, salary_months_12=7, avg_balance_6m=7_954,
        previous_loan="clean", cheque_bounce_12m=True,
    )
    r = price_by_risk(p, base_rate=29.1, acceptable_markup_pts=2.5, thin_markup_pts=5.5,
                       requested_amount=1_639_000, tenor_months=12)
    assert r.score.band == "THIN"
    assert r.annual_rate == pytest.approx(34.6, abs=0.01)
    assert r.instalment == pytest.approx(163_513, rel=0.01)
