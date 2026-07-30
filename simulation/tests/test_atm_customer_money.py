"""Tests for the 9 ATM customer-money task types.

Worked examples are copied verbatim from merged_finetune_train.jsonl /
merged_finetune_val.jsonl (synthetic_atm_customer + synthetic_atm_ops).
Money figures use pytest.approx with a small tolerance since the dataset's
own reference numbers carry their own independent rounding; booleans,
counts, and classifications use exact equality.
"""
from __future__ import annotations

import pytest

from agents.atm.customer_money import (
    AtmFeeParams,
    AtmRecommendationParams,
    Bill,
    CashAdvanceParams,
    CashAffordabilityParams,
    CashCarryingCostParams,
    MonthlyAtmActivity,
    NearbyAtm,
    SurgeCapacityParams,
    TravelCashParams,
    calculate_atm_fee,
    calculate_cash_carrying_cost,
    check_cash_affordability,
    assess_cash_advance,
    plan_surge_capacity,
    plan_travel_cash,
    recommend_atm,
    render_atm_fee_facts,
    render_atm_recommendation_facts,
    render_cash_advance_facts,
    render_cash_affordability_facts,
    render_cash_carrying_cost_facts,
    render_monthly_atm_cost_facts,
    render_spend_pattern_facts,
    render_surge_capacity_facts,
    render_travel_cash_facts,
    summarize_monthly_atm_cost,
    summarize_spend_pattern,
)
from agents.atm.entities import AtmMachine, Cassette, CustomerAtmProfile, Transaction


def make_customer(**overrides) -> CustomerAtmProfile:
    base = dict(
        name="Test Customer", bank="Test Bank", account_type="Savings",
        account_number="PK00TEST0000000000", available_balance=100_000,
        card_network="Visa", card_tier="Classic", card_last4="0000",
        per_txn_limit=30_000, daily_limit=100_000, free_offus_left=4,
        credit_line=0, credit_utilised=0, cash_advance_sub_limit_pct=0.0,
        cash_advance_drawn=0,
    )
    base.update(overrides)
    return CustomerAtmProfile(**base)


# ── ATM fee calculation ──

def test_atm_fee_all_free_allowance_used_no_fed():
    p = AtmFeeParams(
        free_offus_per_month=4, offus_already_made=7, planned_offus_count=6,
        fee_per_txn=23.59, fed_pct=0.0, per_txn_limit=30_000, amount_needed=80_000,
    )
    r = calculate_atm_fee(p)
    assert r.free_left == 0
    assert r.chargeable == 6
    assert r.total == pytest.approx(142, rel=0.02)
    assert r.alt_count == 3
    assert r.alt_total == pytest.approx(71, rel=0.02)
    assert r.cheaper_route_available is True
    assert r.savings == pytest.approx(71, rel=0.02)
    assert "ATM FEE" in render_atm_fee_facts(r)


def test_atm_fee_with_fed_alternative_does_not_help():
    p = AtmFeeParams(
        free_offus_per_month=3, offus_already_made=4, planned_offus_count=3,
        fee_per_txn=20.0, fed_pct=0.16, per_txn_limit=25_000, amount_needed=60_000,
    )
    r = calculate_atm_fee(p)
    assert r.free_left == 0
    assert r.chargeable == 3
    assert r.fee == 60
    assert r.fed == pytest.approx(10, abs=1)
    assert r.total == pytest.approx(70, rel=0.02)
    assert r.cheaper_route_available is False
    assert r.savings == 0


# ── ATM recommendation ──

def test_atm_recommendation_prefers_on_us_over_closer_off_us():
    customer = make_customer(bank="BAFL")
    candidates = [
        NearbyAtm("BAFL-VHR-6372", "BAFL", "Cantt Area, Vehari", 1.9, "OFFLINE", 5_154_500, [1000, 500], 9),
        NearbyAtm("SCB-VHR-6090", "SCB", "Model Town, Vehari", 1.1, "ONLINE", 10_876_000, [5000, 1000], 7),
        NearbyAtm("BAFL-VHR-6734", "BAFL", "Airport Road, Vehari", 6.1, "ONLINE", 769_500, [1000, 500], 9),
        NearbyAtm("FABL-VHR-3119", "FABL", "Railway Road, Vehari", 1.4, "ONLINE", 5_457_500, [1000, 500], 4),
        NearbyAtm("HMB-VHR-7117", "HMB", "Satellite Town, Vehari", 1.8, "ONLINE", 2_187_500, [1000, 500], 3),
    ]
    p = AtmRecommendationParams(customer=customer, amount_needed=40_000, off_us_charge=18.5, candidates=candidates)
    r = recommend_atm(p)
    assert r.recommended.atm_id == "BAFL-VHR-6734"
    assert r.backup.atm_id == "SCB-VHR-6090"
    assert r.cost_to_customer == 0
    assert r.dispense_breakdown == [(1000, 40)]
    assert r.survivors_count == 4
    assert "BAFL-VHR-6734" in render_atm_recommendation_facts(r)


def test_atm_recommendation_drops_offline_and_out_of_cash():
    customer = make_customer(bank="BOK")
    candidates = [
        NearbyAtm("BOK-NWS-1516", "BOK", "Airport", 3.1, "OUT_OF_CASH", 0, [5000, 1000, 500], 2),
        NearbyAtm("MCB-NWS-2398", "MCB", "City Center", 5.3, "ONLINE", 5_685_000, [5000, 1000], 6),
        NearbyAtm("BOK-NWS-7517", "BOK", "Model Bazaar", 6.8, "ONLINE", 210_000, [5000], 2),
        NearbyAtm("FINCA-NWS-4956", "FINCA", "Cantt", 0.6, "OFFLINE", 2_571_000, [1000], 6),
    ]
    p = AtmRecommendationParams(customer=customer, amount_needed=40_000, off_us_charge=23.59, candidates=candidates)
    r = recommend_atm(p)
    assert r.recommended.atm_id == "BOK-NWS-7517"
    assert r.dispense_breakdown == [(5000, 8)]
    assert r.survivors_count == 2


# ── cash advance assessment ──

def test_cash_advance_bound_by_per_txn_limit():
    customer = make_customer(
        per_txn_limit=75_000, daily_limit=250_000, credit_line=800_000, credit_utilised=435_900,
        cash_advance_sub_limit_pct=0.50, cash_advance_drawn=1_200,
    )
    atm = AtmMachine("MEBL-LHE-4968", "MEBL", "Bahria Town Lahore", cassettes=[
        Cassette(5000, 181, 1200), Cassette(1000, 1594, 2500), Cassette(500, 1151, 2000),
    ])
    p = CashAdvanceParams(
        customer=customer, atm=atm, requested_amount=100_000,
        fee_pct=0.035, fee_min=600, monthly_markup_pct=0.03, days_held=10,
    )
    r = assess_cash_advance(p)
    assert r.granted_amount == 75_000
    assert r.binding_constraint == "per-transaction ATM limit"
    assert r.fee == pytest.approx(2_625, rel=0.02)
    assert r.markup == pytest.approx(750, rel=0.02)
    assert r.total_cost == pytest.approx(3_375, rel=0.02)
    assert r.pct_of_cash == pytest.approx(4.5, abs=0.2)


def test_cash_advance_bound_by_requested_amount():
    customer = make_customer(
        per_txn_limit=30_000, daily_limit=75_000, credit_line=400_000, credit_utilised=269_700,
        cash_advance_sub_limit_pct=0.30, cash_advance_drawn=52_000,
    )
    atm = AtmMachine("SBL-SGI-3238", "SBL", "Housing Colony, Sargodha", cassettes=[
        Cassette(1000, 1481, 3000), Cassette(500, 425, 2500),
    ])
    p = CashAdvanceParams(
        customer=customer, atm=atm, requested_amount=20_000,
        fee_pct=0.025, fee_min=750, monthly_markup_pct=0.035, days_held=30,
    )
    r = assess_cash_advance(p)
    assert r.granted_amount == 20_000
    assert r.binding_constraint == "requested amount"
    assert r.fee == pytest.approx(750, rel=0.02)
    assert r.markup == pytest.approx(700, rel=0.02)
    assert r.total_cost == pytest.approx(1_450, rel=0.02)
    assert r.pct_of_cash == pytest.approx(7.2, abs=0.2)


# ── cash affordability check ──

def test_cash_affordability_unsafe_withdrawal():
    customer = make_customer(available_balance=13_974)
    p = CashAffordabilityParams(
        customer=customer, withdrawal_amount=75_000,
        bills=[
            Bill("house rent", 16_400, "2025-01-14"),
            Bill("LESCO bill", 28_400, "2025-01-25"),
            Bill("SNGPL gas bill", 77_100, "2025-01-26"),
        ],
    )
    r = check_cash_affordability(p)
    assert r.projected_balance == -61_026
    assert r.total_bills_due == 121_900
    assert r.safe is False
    assert r.max_affordable == 0
    assert r.shortfall == pytest.approx(182_926, rel=0.01)
    assert r.smallest_bill_label == "house rent"


def test_cash_affordability_safe_withdrawal():
    customer = make_customer(available_balance=505_256)
    p = CashAffordabilityParams(
        customer=customer, withdrawal_amount=15_000,
        bills=[
            Bill("society maintenance", 48_100, "2025-12-15"),
            Bill("society maintenance", 33_600, "2025-12-10"),
            Bill("internet bill", 81_600, "2025-12-11"),
        ],
    )
    r = check_cash_affordability(p)
    assert r.projected_balance == 490_256
    assert r.total_bills_due == 163_300
    assert r.headroom == 326_956
    assert r.safe is True


# ── monthly ATM cost summary ──

def test_monthly_atm_cost_with_inquiries():
    a = MonthlyAtmActivity(
        on_us_count=2, off_us_count=11, inquiries_count=15, avg_withdrawal=18_300,
        free_offus_allowance=3, offus_fee=18.5, inquiry_fee=5.0,
    )
    r = summarize_monthly_atm_cost(a)
    assert r.chargeable_offus == 8
    assert r.offus_fee_total == pytest.approx(148, rel=0.02)
    assert r.inquiry_fee_total == 75
    assert r.total_fee == pytest.approx(223, rel=0.02)
    assert r.total_cash_accessed == 237_900
    assert r.pct_of_cash == pytest.approx(0.09, abs=0.01)


def test_monthly_atm_cost_second_example():
    a = MonthlyAtmActivity(
        on_us_count=3, off_us_count=16, inquiries_count=15, avg_withdrawal=26_600,
        free_offus_allowance=3, offus_fee=25.0, inquiry_fee=5.0,
    )
    r = summarize_monthly_atm_cost(a)
    assert r.chargeable_offus == 13
    assert r.total_fee == pytest.approx(400, rel=0.02)
    assert r.total_cash_accessed == 505_400
    assert r.pct_of_cash == pytest.approx(0.08, abs=0.01)


# ── spend pattern summary ──

def test_spend_pattern_summary_umair():
    txns = [
        Transaction("2025-06-16", "12:08", "Total Parco fuel", "POS", -19_030),
        Transaction("2025-06-17", "15:05", "Al-Fatah Store", "e-commerce", -1_170),
        Transaction("2025-06-17", "20:28", "ATM withdrawal HMB-DDU-2955 (Railway Road, Dadu)", "ATM", -20_000),
        Transaction("2025-06-18", "14:57", "ATM withdrawal SMBL-CJL-8816 (Main Bazaar, Chitral)", "ATM", -7_000),
        Transaction("2025-06-18", "22:09", "Imtiaz Super Market", "POS", -26_260),
        Transaction("2025-06-19", "06:49", "OPTP restaurant", "e-commerce", -6_070),
        Transaction("2025-06-20", "12:07", "OPTP restaurant", "app", -43_550),
        Transaction("2025-06-25", "11:17", "ATM withdrawal SMBL-CJL-8816 (Main Bazaar, Chitral)", "ATM", -34_000),
        Transaction("2025-06-25", "15:26", "ATM withdrawal NRSP-CJL-9928 (Industrial Estate, Chitral)", "ATM", -14_000),
        Transaction("2025-06-27", "12:53", "ATM withdrawal SMBL-CJL-8816 (Main Bazaar, Chitral)", "ATM", -35_000),
        Transaction("2025-06-28", "19:21", "ATM withdrawal SBL-BDN-1761 (Housing Colony, Badin)", "ATM", -15_000),
        Transaction("2025-07-06", "11:24", "ATM withdrawal NRSP-CJL-9928 (Industrial Estate, Chitral)", "ATM", -35_000),
        Transaction("2025-07-06", "22:35", "ATM withdrawal NRSP-CJL-9928 (Industrial Estate, Chitral)", "ATM", -50_000),
        Transaction("2025-07-07", "06:16", "ATM withdrawal SBL-BDN-1761 (Housing Colony, Badin)", "ATM", -13_000),
        Transaction("2025-07-07", "15:19", "ATM withdrawal SBL-BDN-1761 (Housing Colony, Badin)", "ATM", -12_000),
        Transaction("2025-07-09", "10:48", "ATM withdrawal NRSP-CJL-9928 (Industrial Estate, Chitral)", "ATM", -39_000),
    ]
    r = summarize_spend_pattern(txns)
    assert r.total_withdrawn == 274_000
    assert r.withdrawal_count == 11
    assert r.avg_withdrawal == pytest.approx(24_909, rel=0.01)
    assert r.largest_withdrawal == 50_000
    assert r.most_used_atm == "NRSP-CJL-9928"
    assert r.most_used_atm_visits == 4
    assert r.cities == ["Dadu", "Chitral", "Badin"]
    assert r.card_digital_spend == 96_080
    assert r.cash_pct_of_outflow == 74
    assert "SPEND PATTERN" in render_spend_pattern_facts(r)


def test_spend_pattern_summary_ahmed_low_cash_share():
    txns = [
        Transaction("2025-06-11", "21:46", "ATM withdrawal SCB-RYK-5443 (Cantt Area, Rahim Yar Khan)", "ATM", -10_000),
        Transaction("2025-06-15", "07:25", "Foodpanda order", "POS", -39_960),
        Transaction("2025-06-15", "08:47", "Raast transfer", "e-commerce", -37_200),
        Transaction("2025-06-16", "06:34", "Servis Shoes", "POS", -18_950),
        Transaction("2025-06-16", "11:38", "ATM withdrawal BOK-JCB-2465 (Industrial Estate, Jacobabad)", "ATM", -7_000),
        Transaction("2025-06-19", "22:13", "ATM withdrawal BOK-JCB-2465 (Industrial Estate, Jacobabad)", "ATM", -23_500),
        Transaction("2025-06-22", "12:43", "ATM withdrawal BOK-JCB-2465 (Industrial Estate, Jacobabad)", "ATM", -3_000),
        Transaction("2025-06-23", "10:14", "ATM withdrawal BOK-JCB-2465 (Industrial Estate, Jacobabad)", "ATM", -8_000),
        Transaction("2025-06-23", "23:46", "Sapphire outlet", "app", -42_710),
        Transaction("2025-06-26", "17:54", "Jazz load", "e-commerce", -31_400),
        Transaction("2025-06-28", "13:01", "Sapphire outlet", "POS", -38_520),
        Transaction("2025-07-04", "12:13", "inDrive ride", "app", -43_690),
        Transaction("2025-07-05", "22:41", "Sindbad Amusement", "POS", -14_440),
        Transaction("2025-07-08", "11:42", "PSO fuel", "POS", -7_880),
        Transaction("2025-07-08", "14:53", "Sindbad Amusement", "POS", -6_060),
        Transaction("2025-07-08", "16:19", "ATM withdrawal SCB-RYK-5443 (Cantt Area, Rahim Yar Khan)", "ATM", -6_000),
        Transaction("2025-07-09", "07:33", "Freelance remittance (USD)", "IBFT", 184_900),
    ]
    r = summarize_spend_pattern(txns)
    assert r.total_withdrawn == 57_500
    assert r.withdrawal_count == 6
    assert r.most_used_atm == "BOK-JCB-2465"
    assert r.most_used_atm_visits == 4
    assert r.card_digital_spend == 280_810
    assert r.cash_pct_of_outflow == 17


# ── travel cash planning ──

def test_travel_cash_planning_thin_coverage():
    customer = make_customer(per_txn_limit=25_000, daily_limit=60_000)
    p = TravelCashParams(customer=customer, days=4, daily_spend=13_000, working_atms_at_destination=1, off_us_charge=25.0)
    r = plan_travel_cash(p)
    assert r.total_cash_needed == 52_000
    assert r.withdrawal_count == 3
    assert r.days_to_withdraw == 1
    assert r.cost_if_drawn_at_destination == pytest.approx(75, rel=0.02)
    assert r.coverage_classification == "thin"


def test_travel_cash_planning_reasonable_coverage():
    customer = make_customer(per_txn_limit=75_000, daily_limit=250_000)
    p = TravelCashParams(customer=customer, days=3, daily_spend=9_500, working_atms_at_destination=4, off_us_charge=30.0)
    r = plan_travel_cash(p)
    assert r.total_cash_needed == 28_500
    assert r.withdrawal_count == 1
    assert r.cost_if_drawn_at_destination == pytest.approx(30, rel=0.02)
    assert r.coverage_classification == "reasonable"
    assert "TRAVEL CASH" in render_travel_cash_facts(r)


# ── cash carrying cost (ATM ops) ──

def test_cash_carrying_cost_matches_dataset_formula():
    atm = AtmMachine("MCB-UET-9316", "MCB", "Sadar Bazaar, Quetta")
    p = CashCarryingCostParams(atm=atm, idle_cash=8_840_000, daily_dispense=3_000_000, annual_rate=0.1325, cit_trip_cost=4_500, window_days=90)
    r = calculate_cash_carrying_cost(p)
    assert r.carrying_cost == pytest.approx(288_814, rel=0.01)
    assert r.worth_it is True
    assert "CASH CARRYING COST" in render_cash_carrying_cost_facts(r)


def test_cash_carrying_cost_second_site():
    atm = AtmMachine("KMBL-BDN-9252", "KMBL", "Cantt Area, Badin")
    p = CashCarryingCostParams(atm=atm, idle_cash=3_940_000, daily_dispense=2_650_000, annual_rate=0.15, cit_trip_cost=9_000, window_days=90)
    r = calculate_cash_carrying_cost(p)
    assert r.carrying_cost == pytest.approx(145_726, rel=0.01)
    assert r.worth_it is True


# ── surge capacity planning (ATM ops) ──

def test_surge_capacity_single_load_covers_holiday():
    atm = AtmMachine("ABPL-KDU-1030", "ABPL", "Sadar Bazaar, Skardu")
    p = SurgeCapacityParams(atm=atm, normal_daily_dispense=4_790_000, season_name="Ramzan", demand_multiplier=1.35, closed_days=3, max_capacity=20_000_000)
    r = plan_surge_capacity(p)
    assert r.peak_daily_demand == pytest.approx(6_466_500, rel=0.01)
    assert r.total_cash_needed == pytest.approx(19_399_500, rel=0.01)
    assert r.trips_required == 1
    assert r.slack == pytest.approx(600_500, rel=0.01)
    assert "SURGE CAPACITY" in render_surge_capacity_facts(r)


def test_surge_capacity_needs_multiple_trips():
    atm = AtmMachine("KMBL-CWL-5314", "KMBL", "Katchery Chowk, Chakwal")
    p = SurgeCapacityParams(atm=atm, normal_daily_dispense=4_210_000, season_name="pre Eid-ul-Azha", demand_multiplier=2.35, closed_days=3, max_capacity=13_500_000)
    r = plan_surge_capacity(p)
    assert r.peak_daily_demand == pytest.approx(9_893_500, rel=0.01)
    assert r.total_cash_needed == pytest.approx(29_680_500, rel=0.01)
    assert r.trips_required == 3
    assert r.topup_count == 2
    assert r.topup_interval_days == pytest.approx(1.4, abs=0.05)
