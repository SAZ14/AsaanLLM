"""Tests for the 8 ATM "cash mechanics" task types.

Worked examples are copied verbatim from merged_finetune_train.jsonl /
merged_finetune_val.jsonl (synthetic_atm_ops + synthetic_atm_customer).
Money figures use pytest.approx with a small tolerance since the dataset's
own reference numbers carry their own independent rounding; classifications,
booleans, and note counts use exact equality.
"""
from __future__ import annotations

import pytest

from agents.atm.cash_mechanics import (
    CashLoadPlanningInput,
    CashRunoutForecastInput,
    DenominationMixPlanningInput,
    check_denomination_dispensable,
    check_withdrawal_feasibility,
    forecast_cash_runout,
    plan_cash_load,
    plan_denomination_mix,
    reconcile_cash,
    remaining_daily_limit,
    triage_cassette_status,
)
from agents.atm.entities import AtmMachine, Cassette, CustomerAtmProfile, Transaction


# ── 1. cash run-out forecast ──

def test_cash_runout_survives_to_cit():
    # EXAMPLE 1: BIPL-SWL-5079
    machine = AtmMachine(
        atm_id="BIPL-SWL-5079", bank="BIPL", location="Katchery Chowk, Sahiwal",
        cassettes=[
            Cassette(1000, 1155, 2500), Cassette(1000, 1715, 2500),
            Cassette(500, 1074, 2000), Cassette(500, 823, 2000),
        ],
    )
    inp = CashRunoutForecastInput(
        current_time="07:10", dispense_rate_per_hour=210_000, low_cash_threshold=150_000,
        cit_time="10:10", cit_hours_from_now=3,
    )
    r = forecast_cash_runout(machine, inp)
    assert r.total_cash == 3_818_500
    assert r.low_cash_time == "00:38"
    assert r.empty_time == "01:21"
    assert r.survives_to_cit is True
    assert r.headroom_hours == pytest.approx(15.2, rel=0.02)


def test_cash_runout_needs_emergency_run():
    # EXAMPLE 2: BAFL-LRK-8003
    machine = AtmMachine(
        atm_id="BAFL-LRK-8003", bank="BAFL", location="Ring Road, Larkana",
        cassettes=[
            Cassette(5000, 188, 1200), Cassette(1000, 171, 2500), Cassette(500, 825, 2000),
        ],
    )
    inp = CashRunoutForecastInput(
        current_time="17:29", dispense_rate_per_hour=261_000, low_cash_threshold=250_000,
        cit_time="03:29", cit_hours_from_now=10,
    )
    r = forecast_cash_runout(machine, inp)
    assert r.total_cash == 1_523_500
    assert r.empty_time == "23:19"
    assert r.survives_to_cit is False
    assert r.headroom_hours == pytest.approx(-4.2, rel=0.05)


# ── 2. cassette status triage ──

def test_cassette_triage_faults_and_low():
    # EXAMPLE 1: MEBL-KDU-6892
    machine = AtmMachine(
        atm_id="MEBL-KDU-6892", bank="MEBL", location="Katchery Chowk, Skardu",
        cassettes=[
            Cassette(5000, 783, 1500, status="FAULT", note="pick failure, cassette locked out"),
            Cassette(1000, 2253, 2500, status="OK"),
            Cassette(1000, 1158, 2500, status="FAULT", note="pick failure, cassette locked out"),
            Cassette(500, 62, 2000, status="LOW", note="below low-cash threshold"),
        ],
    )
    r = triage_cassette_status(machine)
    assert r.usable_cash == 2_284_000
    assert r.working_cassette_count == 2
    assert r.max_single_withdrawal == 200_000
    assert r.max_withdrawal_breakdown == [(1000, 200)]
    assert r.keep_in_service is True
    assert any("Cassette 1 (Rs 5000)" in a and "locked out" in a for a in r.actions)
    assert any("Cassette 3 (Rs 1000)" in a and "locked out" in a for a in r.actions)
    assert any("Cassette 4 (Rs 500)" in a and "62 notes left" in a for a in r.actions)


def test_cassette_triage_all_healthy_no_action():
    # EXAMPLE 4: JSBL-SWT-4177
    machine = AtmMachine(
        atm_id="JSBL-SWT-4177", bank="JSBL", location="GT Road, Mingora Swat",
        cassettes=[Cassette(5000, 33, 1500, status="OK"), Cassette(1000, 0, 2500, status="OK")],
    )
    r = triage_cassette_status(machine)
    assert r.usable_cash == 165_000
    assert r.working_cassette_count == 2
    assert r.max_single_withdrawal == 165_000
    assert r.max_withdrawal_breakdown == [(5000, 33)]
    assert r.actions == ["No action needed"]


# ── 3. cash reconciliation ──

def test_reconciliation_shortage():
    # EXAMPLE 3: HBLMFB-CJL-4608
    r = reconcile_cash(opening_cash=800_000, loaded_this_cycle=11_200_000, dispensed_ej=8_594_000, physical_count=3_405_000)
    assert r.expected_closing == 3_406_000
    assert r.variance == -1_000
    assert r.status == "shortage"


def test_reconciliation_balanced():
    # EXAMPLE 1: FABL-MDN-9791
    r = reconcile_cash(opening_cash=1_967_000, loaded_this_cycle=11_400_000, dispensed_ej=10_030_000, physical_count=3_337_000)
    assert r.expected_closing == 3_337_000
    assert r.variance == 0
    assert r.status == "balanced"


# ── 4. denomination mix planning ──

def test_denomination_mix_fits_capacity():
    # EXAMPLE 1: BOP-ZHB-3896
    inp = DenominationMixPlanningInput(
        histogram=[(1000, 175), (2000, 47), (5000, 156), (10000, 203), (20000, 241), (25000, 127)],
        denominations=[5000, 1000],
        cassette_capacity_notes={5000: 3000, 1000: 5000},
    )
    r = plan_denomination_mix(inp)
    assert r.total_withdrawals == 949
    assert r.total_value == 11_074_000
    assert r.notes_per_denom[5000] == 2_161
    assert r.notes_per_denom[1000] == 269
    assert r.all_fit_capacity is True


def test_denomination_mix_exceeds_capacity():
    # EXAMPLE 2: FINCA-BDN-1299
    inp = DenominationMixPlanningInput(
        histogram=[(1000, 189), (2000, 217), (5000, 168), (10000, 272), (20000, 81), (25000, 60)],
        denominations=[1000, 500],
        cassette_capacity_notes={1000: 5000, 500: 2500},
    )
    r = plan_denomination_mix(inp)
    assert r.total_withdrawals == 987
    assert r.total_value == 7_303_000
    assert r.notes_per_denom[1000] == 7_303
    assert r.over_capacity_denoms == [1000]
    assert r.all_fit_capacity is False


# ── 5. denomination dispensability ──

def test_denomination_dispensable_exact():
    # EXAMPLE 1: NBP-VHR-1815, Rs 4,500
    machine = AtmMachine(
        atm_id="NBP-VHR-1815", bank="NBP", location="College Road, Vehari",
        cassettes=[Cassette(1000, 1818, 3000), Cassette(500, 291, 2500)],
    )
    r = check_denomination_dispensable(machine, 4_500)
    assert r.dispensable is True
    assert r.breakdown == [(1000, 4), (500, 1)]
    assert r.nearest_amount == 4_500


def test_denomination_not_dispensable_nearest():
    # EXAMPLE 3: HBL-MUX-9527, Rs 17,500 requested
    machine = AtmMachine(
        atm_id="HBL-MUX-9527", bank="HBL", location="Shahi Bazaar, Multan",
        cassettes=[
            Cassette(5000, 27, 1200), Cassette(1000, 0, 2500), Cassette(500, 0, 2000),
        ],
    )
    r = check_denomination_dispensable(machine, 17_500)
    assert r.dispensable is False
    assert r.nearest_amount == 15_000
    assert r.nearest_breakdown == [(5000, 3)]


# ── 6. cash load planning ──

def test_cash_load_planning_capacity_bound():
    # EXAMPLE 2: UBL-DDU-2556
    inp = CashLoadPlanningInput(
        avg_daily_dispense=3_270_000, days_until_cit=3, policy_buffer_pct=0.25,
        cassette_specs=[(5000, 1200), (1000, 2500), (500, 2000)],
        withdrawal_mix={5000: 0.30, 1000: 0.55, 500: 0.15},
    )
    r = plan_cash_load(inp)
    assert r.demand == 9_810_000
    assert r.required_with_buffer == pytest.approx(12_262_500, rel=0.01)
    assert r.total_capacity == 9_500_000
    assert r.load_amount == 9_500_000
    assert r.capacity_binding is True
    alloc = {d: (n, v) for d, n, v in r.allocations}
    assert alloc[5000] == (570, 2_850_000)
    assert alloc[1000] == (2_500, 2_500_000)
    assert alloc[500] == (2_000, 1_000_000)
    assert r.total_loaded == 6_350_000
    assert r.coverage_days == pytest.approx(1.9, rel=0.02)


def test_cash_load_planning_demand_bound():
    # EXAMPLE 4: UMBL-KHT-7267 (mix includes denoms with no cassette on this machine)
    inp = CashLoadPlanningInput(
        avg_daily_dispense=1_900_000, days_until_cit=3, policy_buffer_pct=0.10,
        cassette_specs=[(5000, 1500), (5000, 1500)],
        withdrawal_mix={5000: 0.30, 1000: 0.55, 500: 0.15},
    )
    r = plan_cash_load(inp)
    assert r.demand == 5_700_000
    assert r.required_with_buffer == pytest.approx(6_270_000, rel=0.01)
    assert r.total_capacity == 15_000_000
    assert r.load_amount == 6_270_000
    assert r.capacity_binding is False
    alloc = {d: (n, v) for d, n, v in r.allocations}
    assert alloc[5000] == (1_254, 6_270_000)
    assert r.total_loaded == 6_270_000
    assert r.coverage_days == pytest.approx(3.3, rel=0.02)


# ── 7. withdrawal feasibility ──

def test_withdrawal_infeasible_balance_binds():
    # EXAMPLE 1: Danish Khan at TMBL-SGI-1549
    profile = CustomerAtmProfile(
        name="Danish Khan", bank="Telenor Microfinance Bank", available_balance=15_344,
        per_txn_limit=50_000, daily_limit=150_000,
    )
    machine = AtmMachine(
        atm_id="TMBL-SGI-1549", bank="TMBL", location="Housing Colony, Sargodha",
        cassettes=[
            Cassette(5000, 115, 1500), Cassette(1000, 791, 2500),
            Cassette(1000, 1328, 2500), Cassette(500, 1229, 2000),
        ],
    )
    transactions = [Transaction("2025-01-12", "16:30", "ATM withdrawal", "ATM", -15_000)]
    r = check_withdrawal_feasibility(profile, machine, 50_000, transactions)
    assert r.binding_constraint == "account balance"
    assert r.binding_value == 15_344
    assert r.feasible is False
    assert r.dispense_amount == 15_000
    assert r.dispense_breakdown == [(5000, 3)]


def test_withdrawal_feasible_per_txn_binds():
    # EXAMPLE 3: Iqra Yousafzai at DIBPL-JHG-8655
    profile = CustomerAtmProfile(
        name="Iqra Yousafzai", bank="Dubai Islamic Bank Pakistan", available_balance=87_817,
        per_txn_limit=50_000, daily_limit=150_000,
    )
    machine = AtmMachine(
        atm_id="DIBPL-JHG-8655", bank="DIBPL", location="Bypass Road, Jhang",
        cassettes=[Cassette(5000, 917, 1200), Cassette(1000, 1972, 2500), Cassette(500, 1125, 2000)],
    )
    transactions = [
        Transaction("2025-11-17", "18:29", "ATM withdrawal", "ATM", -20_000),
        Transaction("2025-11-17", "16:42", "ATM withdrawal", "ATM", -2_000),
    ]
    r = check_withdrawal_feasibility(profile, machine, 30_000, transactions)
    assert r.feasible is True
    assert r.dispense_amount == 30_000
    assert r.dispense_breakdown == [(5000, 6)]
    assert r.balance_after == 57_817
    assert r.daily_limit_remaining_after == 98_000


# ── 8. daily limit remaining ──

def test_daily_limit_remaining_example1():
    profile = CustomerAtmProfile(name="Mudassir Rizvi", bank="Zarai Taraqiati Bank", daily_limit=100_000)
    transactions = [
        Transaction("2025-12-09", "08:18", "ATM withdrawal BIPL-ATK-2959", "ATM", -2_000),
        Transaction("2025-12-09", "09:31", "Nishat Linen", "POS", -27_920),
        Transaction("2025-12-09", "15:44", "ATM withdrawal FINCA-SGI-4477", "ATM", -6_000),
        Transaction("2025-12-09", "16:05", "Ufone load", "e-commerce", -9_590),
    ]
    r = remaining_daily_limit(profile, transactions, requested_amount=10_000)
    assert r.atm_used_today == 8_000
    assert r.remaining == 92_000
    assert r.feasible is True


def test_daily_limit_remaining_example3():
    profile = CustomerAtmProfile(name="Basit Abbasi", bank="Standard Chartered Bank Pakistan", daily_limit=100_000)
    transactions = [
        Transaction("2025-10-23", "09:37", "ATM withdrawal BAHL-SKZ-7401", "ATM", -1_000),
        Transaction("2025-10-23", "10:15", "ATM withdrawal KMBL-CJL-2363", "ATM", -10_000),
        Transaction("2025-10-23", "16:00", "ATM withdrawal ABPL-ATD-1669", "ATM", -6_000),
        Transaction("2025-10-23", "17:00", "NayaPay top-up", "POS", -27_850),
        Transaction("2025-10-23", "18:59", "ATM withdrawal NRSP-ATK-6272", "ATM", -6_000),
        Transaction("2025-10-23", "20:07", "Khaadi outlet", "e-commerce", -4_910),
    ]
    r = remaining_daily_limit(profile, transactions, requested_amount=10_000)
    assert r.atm_used_today == 23_000
    assert r.remaining == 77_000
    assert r.feasible is True
