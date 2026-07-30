"""Tests for the 8 ATM judgment/classification task types.

Worked examples are copied verbatim from merged_finetune_train.jsonl /
merged_finetune_val.jsonl (synthetic_atm_ops / synthetic_atm_customer).
Money/percentage figures use pytest.approx with a small tolerance since the
dataset's own reference numbers carry their own independent rounding;
verdicts/classifications/root-causes use exact equality.
"""
from __future__ import annotations

import pytest

from agents.atm.entities import AtmMachine, CustomerAtmProfile, Transaction
from agents.atm.judgment import (
    CardFraudInputs,
    CardRetentionInputs,
    DisputeInputs,
    DowntimeIncident,
    FaultDiagnosisInputs,
    HourlyLogEntry,
    InterbankSettlementInputs,
    SecurityObservation,
    TrendSummaryInputs,
    UptimeSlaInputs,
    assess_card_fraud,
    assess_failed_transaction_dispute,
    assess_security_anomaly,
    calculate_interbank_settlement,
    check_uptime_sla,
    diagnose_fault,
    guide_card_retention,
    summarize_trend,
)


def make_profile(**overrides) -> CustomerAtmProfile:
    base = dict(
        name="Test Customer", bank="Test Bank (TSTB)", account_type="Current",
        account_number="PK00TSTB0000000000", available_balance=100_000,
        card_network="Visa", card_tier="Classic", card_last4="0000",
        per_txn_limit=25_000, daily_limit=50_000, free_offus_left=3,
        credit_line=0, credit_utilised=0, cash_advance_sub_limit_pct=0.0, cash_advance_drawn=0,
    )
    base.update(overrides)
    return CustomerAtmProfile(**base)


# ── fault root cause ──

VENDOR_TABLE_EX1 = {
    "R0005": "receipt printer paper out",
    "P0022": "power failure - UPS drained",
    "H0013": "cash-out sensor mismatch",
    "E0004": "electronic journal full",
    "N0009": "note quality reject rate high",
}


def test_fault_root_cause_receipt_printer():
    atm = AtmMachine("JSBL-JCB-5449", "JSBL", "Cantt Area, Jacobabad")
    inputs = FaultDiagnosisInputs(
        atm=atm, fault_code="R0005", down_minutes=285, failed_transactions=50,
        vendor_table=VENDOR_TABLE_EX1,
    )
    r = diagnose_fault(inputs)
    assert r.meaning == "receipt printer paper out"
    assert r.likely_cause == "Paper roll not replaced on last visit"
    assert r.action == "Replace roll during next custodian visit"
    assert r.past_notice_threshold is True


def test_fault_root_cause_communication_link_down():
    atm = AtmMachine("AKBL-MUX-8579", "AKBL", "Cantt Area, Multan")
    vendor_table = {
        "S0015": "supervisor mode left open",
        "L0002": "communication link down",
        "X0001": "out of cash",
        "D0087": "cassette jam in feed path",
        "C0031": "card reader read failure",
    }
    inputs = FaultDiagnosisInputs(
        atm=atm, fault_code="L0002", down_minutes=137, failed_transactions=22,
        vendor_table=vendor_table,
    )
    r = diagnose_fault(inputs)
    assert r.meaning == "communication link down"
    assert r.likely_cause == "ISP or router failure at site"
    assert r.action == "Failover to backup 4G link, raise ISP ticket"
    assert r.past_notice_threshold is True


def test_fault_root_cause_power_failure_short_downtime():
    atm = AtmMachine("UMBL-OKR-1355", "UMBL", "College Road, Okara")
    vendor_table = {
        "R0005": "receipt printer paper out",
        "T0007": "PIN pad tamper alarm",
        "P0022": "power failure - UPS drained",
        "D0087": "cassette jam in feed path",
        "C0031": "card reader read failure",
    }
    inputs = FaultDiagnosisInputs(
        atm=atm, fault_code="P0022", down_minutes=24, failed_transactions=27,
        vendor_table=vendor_table,
    )
    r = diagnose_fault(inputs)
    assert r.meaning == "power failure - UPS drained"
    assert r.likely_cause == "Extended load-shedding beyond UPS backup"
    assert r.past_notice_threshold is False


# ── security anomaly assessment ──

def test_security_jackpotting_signature():
    atm = AtmMachine("BIPL-KSR-1111", "BIPL", "Civil Lines, Kasur")
    obs = SecurityObservation(
        atm=atm, event_count=37, window_start="04:00", window_end="04:20",
        total_amount=740_000, unmatched_authorisation=True,
        safe_door_opened_time="03:24", cit_visit_scheduled=False,
        usb_port_cover_open=True,
    )
    r = assess_security_anomaly(obs)
    assert r.signature == "jackpotting_black_box"
    assert r.is_attack is True
    assert r.headline == "jackpotting / black-box attack"


def test_security_genuine_demand_surge():
    atm = AtmMachine("BAFL-LYH-7298", "BAFL", "Civil Lines, Layyah")
    obs = SecurityObservation(
        atm=atm, event_count=313, window_start="09:00", window_end="21:00",
        baseline_avg_count=172, avg_ticket_size=12_000,
        unmatched_authorisation=False, payout_event_nearby=True,
    )
    r = assess_security_anomaly(obs)
    assert r.signature == "genuine_demand_surge"
    assert r.is_attack is False
    assert r.headline == "genuine demand surge, not an attack"


def test_security_second_jackpotting_example():
    atm = AtmMachine("ABL-SWT-3943", "ABL", "Civil Lines, Mingora Swat")
    obs = SecurityObservation(
        atm=atm, event_count=66, window_start="01:00", window_end="01:26",
        total_amount=1_320_000, unmatched_authorisation=True,
        safe_door_opened_time="00:10", cit_visit_scheduled=False,
        usb_port_cover_open=True,
    )
    r = assess_security_anomaly(obs)
    assert r.signature == "jackpotting_black_box"
    assert r.is_attack is True


# ── card fraud assessment ──

def test_card_fraud_high_risk_velocity():
    profile = make_profile(bank="Standard Chartered Bank Pakistan (SCB)", per_txn_limit=25_000)
    txns = [
        Transaction("2025-05-16", "01:59", "ATM withdrawal Skardu - Ring Road", "ATM", -25_000),
        Transaction("2025-05-16", "02:01", "ATM withdrawal Skardu - Bypass Road", "ATM", -25_000),
        Transaction("2025-05-16", "02:03", "ATM withdrawal Skardu - Bypass Road", "ATM", -25_000),
        Transaction("2025-05-16", "02:08", "ATM withdrawal Skardu - College Road", "ATM", -25_000),
    ]
    r = assess_card_fraud(CardFraudInputs(profile=profile, transactions=txns))
    assert r.verdict == "high_risk_velocity"
    assert r.is_high_risk is True
    assert r.total_atm_outflow == 100_000


def test_card_fraud_impossible_travel():
    profile = make_profile(bank="Bank Alfalah (BAFL)", per_txn_limit=25_000)
    txns = [
        Transaction("2025-11-14", "12:36", "ATM withdrawal Larkana - Cantt Area", "ATM", -25_000),
        Transaction("2025-11-14", "13:20", "ATM withdrawal Gujranwala - College Road", "ATM", -30_000),
        Transaction("2025-11-14", "13:55", "ATM withdrawal Gujranwala - Bypass Road", "ATM", -25_000),
    ]
    r = assess_card_fraud(CardFraudInputs(profile=profile, transactions=txns))
    assert r.verdict == "high_risk_impossible_travel"
    assert r.is_high_risk is True
    assert r.total_atm_outflow == 80_000


def test_card_fraud_normal_usage():
    profile = make_profile(bank="MCB Bank Limited (MCB)", per_txn_limit=25_000, daily_limit=60_000)
    txns = [
        Transaction("2025-10-18", "09:31", "ATM withdrawal Mingora Swat - Civil Lines", "ATM", -10_000),
        Transaction("2025-10-18", "12:32", "Cinepax ticket", "POS", -8_800),
        Transaction("2025-10-19", "11:43", "ATM withdrawal Mingora Swat - Industrial Estate", "ATM", -15_000),
    ]
    r = assess_card_fraud(CardFraudInputs(profile=profile, transactions=txns))
    assert r.verdict == "normal"
    assert r.is_high_risk is False
    assert r.total_atm_outflow == 25_000


# ── card retention guidance ──

def test_card_retention_hotlist_different_bank():
    profile = make_profile(bank="Sindh Bank (SNDB)")
    atm = AtmMachine("HMB-ISB-7254", "HMB", "Blue Area, Islamabad")
    inputs = CardRetentionInputs(
        profile=profile, atm=atm, atm_bank_full_name="Habib Metropolitan Bank",
        hours_since_capture=32, capture_reason="card reported lost/stolen and hot-listed",
        hold_days=5,
    )
    r = guide_card_retention(inputs)
    assert r.category == "hotlist"
    assert r.category_label == "hot-list capture"
    assert r.own_bank is False
    assert r.uncertain_handling is True


def test_card_retention_fault_different_bank():
    profile = make_profile(bank="Zarai Taraqiati Bank (ZTBL)")
    atm = AtmMachine("SBL-UET-6276", "SBL", "University Road, Quetta")
    inputs = CardRetentionInputs(
        profile=profile, atm=atm, atm_bank_full_name="Soneri Bank",
        hours_since_capture=23, capture_reason="power failure mid-transaction",
        hold_days=7,
    )
    r = guide_card_retention(inputs)
    assert r.category == "fault"
    assert r.category_label == "fault retention"
    assert r.own_bank is False
    assert r.uncertain_handling is False


def test_card_retention_timeout_different_bank():
    profile = make_profile(bank="MCB Bank Limited (MCB)")
    atm = AtmMachine("FINCA-MUX-5683", "FINCA", "Sadar Bazaar, Multan")
    inputs = CardRetentionInputs(
        profile=profile, atm=atm, atm_bank_full_name="FINCA Microfinance Bank",
        hours_since_capture=27,
        capture_reason="card left in the slot for more than 30 seconds after the transaction",
        hold_days=3,
    )
    r = guide_card_retention(inputs)
    assert r.category == "timeout"
    assert r.category_label == "time-out retention"
    assert r.uncertain_handling is False


def test_card_retention_own_bank():
    profile = make_profile(bank="Habib Metropolitan Bank (HMB)")
    atm = AtmMachine("HMB-ISB-7254", "HMB", "Blue Area, Islamabad")
    inputs = CardRetentionInputs(
        profile=profile, atm=atm, atm_bank_full_name="Habib Metropolitan Bank",
        hours_since_capture=5, capture_reason="power failure mid-transaction", hold_days=5,
    )
    r = guide_card_retention(inputs)
    assert r.own_bank is True


# ── failed transaction dispute ──

def test_dispute_nothing_to_claim():
    atm = AtmMachine("AKBL-KZD-6540", "AKBL", "Shahi Bazaar, Khuzdar")
    inputs = DisputeInputs(
        atm=atm, date="2025-01-14", debited=20_000, received=20_000,
        journal_entry="DISPENSE OK - notes taken by customer",
        reconciliation_excess=0, reversal_window_days=7, days_since_transaction=1,
    )
    r = assess_failed_transaction_dispute(inputs)
    assert r.shortfall == 0
    assert r.verdict == "no_claim"


def test_dispute_owed_within_window():
    atm = AtmMachine("ZTBL-UET-4730", "ZTBL", "Housing Colony, Quetta")
    inputs = DisputeInputs(
        atm=atm, date="2025-04-20", debited=10_000, received=0,
        journal_entry="DISPENSE FAILED - notes retracted to purge bin",
        reconciliation_excess=10_000, reversal_window_days=7, days_since_transaction=7,
    )
    r = assess_failed_transaction_dispute(inputs)
    assert r.shortfall == 10_000
    assert r.verdict == "owed"
    assert r.reconciliation_corroborates is True
    assert r.within_window is True
    assert r.days_over == 0


def test_dispute_owed_past_window():
    atm = AtmMachine("BAHL-HDD-9051", "BAHL", "GT Road, Hyderabad")
    inputs = DisputeInputs(
        atm=atm, date="2025-12-11", debited=50_000, received=0,
        journal_entry="DISPENSE FAILED - notes retracted to purge bin",
        reconciliation_excess=50_000, reversal_window_days=3, days_since_transaction=7,
    )
    r = assess_failed_transaction_dispute(inputs)
    assert r.shortfall == 50_000
    assert r.verdict == "owed"
    assert r.within_window is False
    assert r.days_over == 4


# ── interbank settlement ──

def test_interbank_settlement_net_payer_jsbl():
    inputs = InterbankSettlementInputs(
        bank_code="JSBL", on_us_count=46_407, acquired_offus_count=6_244,
        issued_offus_count=13_057, interchange_rate=25.0,
    )
    r = calculate_interbank_settlement(inputs)
    assert r.acquiring_income == 156_100
    assert r.issuing_cost == 326_425
    assert r.net_position == -170_325
    assert r.net_payer is True


def test_interbank_settlement_net_payer_mebl_half_up_rounding():
    inputs = InterbankSettlementInputs(
        bank_code="MEBL", on_us_count=38_444, acquired_offus_count=5_543,
        issued_offus_count=77_034, interchange_rate=18.5,
    )
    r = calculate_interbank_settlement(inputs)
    assert r.acquiring_income == 102_546
    assert r.issuing_cost == 1_425_129
    assert r.net_position == -1_322_584


def test_interbank_settlement_scb():
    inputs = InterbankSettlementInputs(
        bank_code="SCB", on_us_count=42_479, acquired_offus_count=44_647,
        issued_offus_count=75_314, interchange_rate=18.5,
    )
    r = calculate_interbank_settlement(inputs)
    assert r.acquiring_income == 825_970
    assert r.issuing_cost == 1_393_309
    assert r.net_position == -567_340


# ── trend summary ──

def test_trend_summary_full_drawdown():
    atm = AtmMachine("SILK-RYK-9480", "SILK", "GT Road, Rahim Yar Khan")
    log = [
        HourlyLogEntry("08:00", 6_057_000, 43_000),
        HourlyLogEntry("09:00", 5_863_000, 194_000),
        HourlyLogEntry("10:00", 5_518_000, 345_000),
        HourlyLogEntry("11:00", 5_021_000, 497_000),
        HourlyLogEntry("12:00", 4_373_000, 648_000),
        HourlyLogEntry("13:00", 3_574_000, 799_000),
        HourlyLogEntry("14:00", 2_623_000, 951_000),
        HourlyLogEntry("15:00", 1_521_000, 1_102_000),
        HourlyLogEntry("16:00", 268_000, 1_253_000),
        HourlyLogEntry("17:00", 0, 268_000),
        HourlyLogEntry("18:00", 0, 0),
        HourlyLogEntry("19:00", 0, 0),
        HourlyLogEntry("20:00", 0, 0),
    ]
    r = summarize_trend(TrendSummaryInputs(atm=atm, opening_balance=6_100_000, log=log))
    assert r.total_dispensed == 6_100_000
    assert r.closing_balance == 0
    assert r.pct_of_load_dispensed == pytest.approx(100.0)
    assert r.peak_hour == "16:00"
    assert r.peak_amount == 1_253_000
    assert r.cit_recommendation_hour == "14:00"


def test_trend_summary_partial_drawdown():
    atm = AtmMachine("UMBL-KHP-9845", "UMBL", "College Road, Khairpur")
    log = [
        HourlyLogEntry("08:00", 11_600_000, 0),
        HourlyLogEntry("09:00", 11_588_000, 12_000),
        HourlyLogEntry("10:00", 11_531_000, 57_000),
        HourlyLogEntry("11:00", 11_429_000, 102_000),
        HourlyLogEntry("12:00", 11_281_000, 148_000),
        HourlyLogEntry("13:00", 11_088_000, 193_000),
        HourlyLogEntry("14:00", 10_850_000, 238_000),
        HourlyLogEntry("15:00", 10_567_000, 283_000),
        HourlyLogEntry("16:00", 10_239_000, 328_000),
        HourlyLogEntry("17:00", 9_866_000, 373_000),
        HourlyLogEntry("18:00", 9_448_000, 418_000),
        HourlyLogEntry("19:00", 8_985_000, 463_000),
        HourlyLogEntry("20:00", 8_567_000, 418_000),
    ]
    r = summarize_trend(TrendSummaryInputs(atm=atm, opening_balance=11_600_000, log=log))
    assert r.total_dispensed == 3_033_000
    assert r.closing_balance == 8_567_000
    assert r.pct_of_load_dispensed == pytest.approx(26.1, abs=0.05)
    assert r.peak_hour == "19:00"
    assert r.peak_amount == 463_000
    assert r.cit_recommendation_hour == "17:00"


# ── uptime SLA check ──

def test_uptime_sla_breach_abl_bnu():
    atm = AtmMachine("ABL-BNU-9563", "ABL", "College Road, Bannu")
    incidents = [
        DowntimeIncident("H0013", "cash-out sensor mismatch", 852),
        DowntimeIncident("N0009", "note quality reject rate high", 721),
        DowntimeIncident("D0011", "dispenser fault - note pick failure", 254),
    ]
    r = check_uptime_sla(UptimeSlaInputs(atm=atm, period_days=31, sla_pct=98.0, incidents=incidents))
    assert r.uptime_pct == pytest.approx(95.91, abs=0.01)
    assert r.total_downtime_minutes == 1_827
    assert r.allowed_downtime_minutes == 893
    assert r.breach is True
    assert r.breach_minutes == 934
    assert r.biggest_contributor.code == "H0013"


def test_uptime_sla_breach_fwbl():
    atm = AtmMachine("FWBL-CJL-8452", "FWBL", "Sadar Bazaar, Chitral")
    incidents = [
        DowntimeIncident("E0004", "electronic journal full", 803),
        DowntimeIncident("S0015", "supervisor mode left open", 368),
        DowntimeIncident("D0011", "dispenser fault - note pick failure", 427),
    ]
    r = check_uptime_sla(UptimeSlaInputs(atm=atm, period_days=31, sla_pct=97.0, incidents=incidents))
    assert r.uptime_pct == pytest.approx(96.42, abs=0.01)
    assert r.allowed_downtime_minutes == 1_339
    assert r.breach is True
    assert r.breach_minutes == 259


def test_uptime_sla_breach_smbl_short_period():
    atm = AtmMachine("SMBL-SKZ-4800", "SMBL", "Railway Road, Sukkur")
    incidents = [
        DowntimeIncident("C0031", "card reader read failure", 63),
        DowntimeIncident("R0005", "receipt printer paper out", 180),
        DowntimeIncident("X0001", "out of cash", 429),
        DowntimeIncident("R0005", "receipt printer paper out", 877),
    ]
    r = check_uptime_sla(UptimeSlaInputs(atm=atm, period_days=7, sla_pct=98.0, incidents=incidents))
    assert r.uptime_pct == pytest.approx(84.63, abs=0.01)
    assert r.total_downtime_minutes == 1_549
    assert r.allowed_downtime_minutes == 202
    assert r.breach is True
    assert r.breach_minutes == 1_347
    assert r.biggest_contributor.minutes == 877
