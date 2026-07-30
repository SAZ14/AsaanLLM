"""Tests for the 8 ATM network-ops task types.

Worked examples are copied verbatim from merged_finetune_train.jsonl /
merged_finetune_val.jsonl (synthetic_atm_ops). Money/hour figures use
pytest.approx with a small relative tolerance since the dataset's own
reference numbers carry their own independent rounding; orderings,
classifications, and counts use exact equality.
"""
from __future__ import annotations

import pytest

from agents.atm.entities import AtmMachine
from agents.atm.network_ops import (
    AlertRow,
    CitCandidate,
    DemandForecastInput,
    EodSiteRow,
    GrowthInput,
    RankingRow,
    ReplenishmentCandidate,
    ShareRow,
    analyze_growth,
    analyze_share,
    plan_cit_route,
    prioritize_replenishment,
    rank_sites,
    report_eod_position,
    forecast_demand,
    triage_alerts,
)


# ── alert triage ──

def test_alert_triage_example1_order_and_dispatch():
    alerts = [
        AlertRow("FWBL-MUX-9514", "Shahi Bazaar", "CASH_LOW", "cash below threshold", 111),
        AlertRow("NBP-MUX-8535", "Ring Road", "DISPENSER_FAULT", "dispenser hardware fault", 72),
        AlertRow("HMB-MUX-9222", "Shahi Bazaar", "UPS_ON_BATTERY", "running on UPS battery", 127),
        AlertRow("BIPL-MUX-7174", "College Road", "UPS_ON_BATTERY", "running on UPS battery", 43),
        AlertRow("HBL-MUX-4499", "GT Road", "DOOR_OPEN", "safe door open unscheduled", 194),
        AlertRow("AKBL-MUX-4493", "College Road", "PRINTER_PAPER_OUT", "receipt paper out", 224),
    ]
    r = triage_alerts(alerts)
    order = [t.atm_id for t in r.triaged]
    assert order == [
        "NBP-MUX-8535", "HBL-MUX-4499", "HMB-MUX-9222",
        "FWBL-MUX-9514", "BIPL-MUX-7174", "AKBL-MUX-4493",
    ]
    assert r.triaged[0].severity == 1
    assert r.dispatch_tonight == ["NBP-MUX-8535"]


def test_alert_triage_example3_severities():
    alerts = [
        AlertRow("AKBL-NWB-8760", "Railway Road", "DOOR_OPEN", "safe door open unscheduled", 54),
        AlertRow("HBLMFB-NWB-5216", "GT Road", "DISPENSER_FAULT", "dispenser hardware fault", 4),
        AlertRow("BIPL-NWB-8849", "Airport Road", "COMM_DOWN", "communication link down", 92),
        AlertRow("UBL-NWB-5072", "Bypass Road", "HIGH_REJECT_RATE", "note reject rate high", 85),
    ]
    r = triage_alerts(alerts)
    order = [t.atm_id for t in r.triaged]
    assert order == ["BIPL-NWB-8849", "HBLMFB-NWB-5216", "AKBL-NWB-8760", "UBL-NWB-5072"]
    assert set(r.dispatch_tonight) == {"BIPL-NWB-8849", "HBLMFB-NWB-5216"}
    by_id = {t.atm_id: t.severity for t in r.triaged}
    assert by_id["AKBL-NWB-8760"] == 2
    assert by_id["UBL-NWB-5072"] == 3


# ── replenishment priority ──

def test_replenishment_example1_hours_and_no_risk():
    candidates = [
        ReplenishmentCandidate("UMBL-GWD-5939", "Katchery Chowk", 7_254_000, 86_000),
        ReplenishmentCandidate("UMBL-GWD-8779", "Shahi Bazaar", 4_344_000, 89_000),
        ReplenishmentCandidate("UMBL-GWD-8911", "Liaquat Road", 16_294_000, 12_000),
        ReplenishmentCandidate("UMBL-GWD-6641", "Industrial Estate", 6_095_000, 121_000),
    ]
    r = prioritize_replenishment(candidates, window_hours=12)
    order = [x.atm_id for x in r.ranked]
    assert order == ["UMBL-GWD-8779", "UMBL-GWD-6641", "UMBL-GWD-5939", "UMBL-GWD-8911"]
    assert r.ranked[0].hours_to_empty == pytest.approx(48.8, rel=0.01)
    assert r.ranked[3].hours_to_empty == pytest.approx(1357.8, rel=0.01)
    assert r.at_risk_count == 0
    assert r.first_site == "UMBL-GWD-8779"


def test_replenishment_example2_at_risk_sites():
    candidates = [
        ReplenishmentCandidate("UBL-GWD-3443", "Katchery Chowk", 4_450_000, 60_000),
        ReplenishmentCandidate("UBL-GWD-2011", "Cantt Area", 2_879_500, 117_000),
        ReplenishmentCandidate("UBL-GWD-9704", "Main Bazaar", 1_393_000, 268_000),
        ReplenishmentCandidate("UBL-GWD-6791", "Cantt Area", 1_626_000, 148_000),
        ReplenishmentCandidate("UBL-GWD-3713", "Housing Colony", 4_375_000, 138_000),
    ]
    r = prioritize_replenishment(candidates, window_hours=12)
    order = [x.atm_id for x in r.ranked]
    assert order == [
        "UBL-GWD-9704", "UBL-GWD-6791", "UBL-GWD-2011", "UBL-GWD-3713", "UBL-GWD-3443",
    ]
    assert r.at_risk_count == 2
    assert r.first_site == "UBL-GWD-9704"
    by_id = {x.atm_id: x.hours_to_empty for x in r.ranked}
    assert by_id["UBL-GWD-9704"] == pytest.approx(5.2, rel=0.02)
    assert by_id["UBL-GWD-6791"] == pytest.approx(11.0, rel=0.02)


# ── CIT route planning ──

def test_cit_route_example1_greedy_fill():
    candidates = [
        CitCandidate("BAFL-LYH-3737", 18.9, 5_358_000, 107.2, 7_300_000),
        CitCandidate("BAFL-LYH-1752", 18.8, 8_804_000, 106.1, 3_600_000),
        CitCandidate("BAFL-LYH-3945", 28.5, 5_198_500, 45.2, 4_200_000),
        CitCandidate("BAFL-LYH-5907", 28.7, 836_000, 7.8, 5_800_000),
        CitCandidate("BAFL-LYH-7668", 20.2, 10_391_000, 207.8, 8_000_000),
    ]
    r = plan_cit_route(candidates, cap=15_000_000)
    assert [c.atm_id for c in r.included] == ["BAFL-LYH-5907", "BAFL-LYH-3945", "BAFL-LYH-1752"]
    assert [c.atm_id for c in r.deferred] == ["BAFL-LYH-3737", "BAFL-LYH-7668"]
    assert r.total_loaded == 13_600_000
    assert r.spare_capacity == 1_400_000
    assert r.sites_included == 3
    assert r.sites_total == 5


def test_cit_route_example4_all_fit_with_tiebreak():
    candidates = [
        CitCandidate("MCB-MUX-1590", 10.3, 26_500, 0.0, 5_500_000),
        CitCandidate("MCB-MUX-6754", 16.3, 0, 0.0, 8_500_000),
        CitCandidate("MCB-MUX-6192", 22.1, 1_217_000, 1.4, 7_800_000),
        CitCandidate("MCB-MUX-3844", 22.5, 1_017_500, 7.2, 3_100_000),
        CitCandidate("MCB-MUX-8223", 9.1, 1_209_500, 1.9, 3_900_000),
        CitCandidate("MCB-MUX-3986", 15.9, 220_000, 0.1, 2_700_000),
        CitCandidate("MCB-MUX-5689", 9.1, 0, 0.0, 6_900_000),
    ]
    r = plan_cit_route(candidates, cap=40_000_000)
    assert r.sites_included == 7
    assert r.deferred == []
    assert r.total_loaded == 38_400_000
    assert r.spare_capacity == 1_600_000
    assert [c.atm_id for c in r.included] == [
        "MCB-MUX-6754", "MCB-MUX-5689", "MCB-MUX-1590",
        "MCB-MUX-3986", "MCB-MUX-6192", "MCB-MUX-8223", "MCB-MUX-3844",
    ]


# ── demand forecast ──

def test_demand_forecast_example1_within_capacity():
    machine = AtmMachine("BOP-BHV-7338", "BOP", "Civil Lines, Bahawalpur", model="NCR 6622")
    inp = DemandForecastInput(
        machine=machine,
        week_dispensed=[
            ("Monday", 1_130_000), ("Tuesday", 950_000), ("Wednesday", 880_000),
            ("Thursday", 930_000), ("Friday", 1_320_000), ("Saturday", 1_000_000),
            ("Sunday", 730_000),
        ],
        forecast_day="Thursday", seasonal_label="pre Eid-ul-Fitr", seasonal_multiplier=2.10,
        machine_capacity=7_000_000,
    )
    r = forecast_demand(inp)
    assert r.forecast == 1_953_000
    assert r.weekly_average == pytest.approx(991_429, rel=0.001)
    assert r.day_of_week_factor == pytest.approx(0.94, abs=0.01)
    assert r.within_capacity is True
    assert r.topup_needed == 0


def test_demand_forecast_example3_exceeds_capacity():
    machine = AtmMachine("UBL-HDD-4985", "UBL", "Housing Colony, Hyderabad", model="NCR 6622")
    inp = DemandForecastInput(
        machine=machine,
        week_dispensed=[
            ("Monday", 14_340_000), ("Tuesday", 12_330_000), ("Wednesday", 12_590_000),
            ("Thursday", 13_050_000), ("Friday", 16_440_000), ("Saturday", 15_320_000),
            ("Sunday", 9_320_000),
        ],
        forecast_day="Friday", seasonal_label="pension disbursement day", seasonal_multiplier=1.45,
        machine_capacity=9_500_000,
    )
    r = forecast_demand(inp)
    assert r.forecast == 23_838_000
    assert r.weekly_average == pytest.approx(13_341_429, rel=0.001)
    assert r.day_of_week_factor == pytest.approx(1.23, abs=0.01)
    assert r.within_capacity is False
    assert r.topup_needed == 14_338_000


# ── end-of-day position report ──

def test_eod_position_example1():
    sites = [
        EodSiteRow("SMBL-RYK-6577", "University Road", "ONLINE", 290_000, 8_380_000),
        EodSiteRow("SMBL-RYK-4546", "Katchery Chowk", "OUT_OF_CASH", 0, 840_000),
        EodSiteRow("SMBL-RYK-8911", "Model Bazaar", "COMMS_DOWN", 38_000, 4_880_000),
        EodSiteRow("SMBL-RYK-2735", "Main Bazaar", "ONLINE", 12_500, 500_000),
        EodSiteRow("SMBL-RYK-4758", "Civil Lines", "ONLINE", 1_062_500, 2_480_000),
        EodSiteRow("SMBL-RYK-1879", "Shahi Bazaar", "COMMS_DOWN", 2_461_000, 3_270_000),
    ]
    r = report_eod_position(sites)
    assert r.availability_pct == pytest.approx(50.0)
    assert r.online_count == 3
    assert r.cash_on_hand_total == 3_864_000
    assert r.dispensed_total == 20_350_000
    assert len(r.action_items) == 5
    assert [a.atm_id for a in r.action_items] == [
        "SMBL-RYK-4546", "SMBL-RYK-8911", "SMBL-RYK-1879", "SMBL-RYK-6577", "SMBL-RYK-2735",
    ]


def test_eod_position_example4_mixed_fault_types():
    sites = [
        EodSiteRow("JSBL-DGK-9443", "Sadar Bazaar", "OFFLINE_FAULT", 605_500, 5_890_000),
        EodSiteRow("JSBL-DGK-3163", "Airport Road", "ONLINE", 0, 4_990_000),
        EodSiteRow("JSBL-DGK-9712", "Ring Road", "COMMS_DOWN", 3_150_000, 3_240_000),
        EodSiteRow("JSBL-DGK-1074", "Cantt Area", "ONLINE", 1_485_500, 5_800_000),
        EodSiteRow("JSBL-DGK-9019", "Liaquat Road", "OUT_OF_CASH", 0, 5_870_000),
        EodSiteRow("JSBL-DGK-2938", "Ring Road", "ONLINE", 175_000, 6_200_000),
    ]
    r = report_eod_position(sites)
    assert r.availability_pct == pytest.approx(50.0)
    assert r.cash_on_hand_total == 5_416_000
    assert r.dispensed_total == 31_990_000
    assert [a.atm_id for a in r.action_items] == [
        "JSBL-DGK-9443", "JSBL-DGK-9712", "JSBL-DGK-9019", "JSBL-DGK-3163", "JSBL-DGK-2938",
    ]


# ── growth analysis ──

def test_growth_analysis_money_metric():
    inp = GrowthInput(
        atm_id="KMBL-ATD-6018", metric_label="Off-us withdrawals",
        old_period="June", old_value=168_210_000, new_period="July", new_value=176_250_000,
        is_money=True,
    )
    r = analyze_growth(inp)
    assert r.change == 8_040_000
    assert r.pct_change == pytest.approx(4.8, abs=0.05)


def test_growth_analysis_count_metric():
    inp = GrowthInput(
        atm_id="BOP-ISB-1322", metric_label="Transaction count",
        old_period="January", old_value=46_100, new_period="February", new_value=60_780,
        is_money=False,
    )
    r = analyze_growth(inp)
    assert r.change == 14_680
    assert r.pct_change == pytest.approx(31.8, abs=0.05)


# ── share analysis ──

def test_share_analysis_example1():
    rows = [
        ShareRow("FINCA-GIL-5043", "Ring Road", 11_090_000),
        ShareRow("FINCA-GIL-3371", "Cantt Area", 5_480_000),
        ShareRow("FINCA-GIL-7638", "Industrial Estate", 2_360_000),
        ShareRow("FINCA-GIL-6114", "Katchery Chowk", 8_320_000),
        ShareRow("FINCA-GIL-5636", "Main Bazaar", 13_650_000),
    ]
    r = analyze_share(rows, "FINCA-GIL-7638")
    assert r.total == 40_900_000
    assert r.share_pct == pytest.approx(5.8, abs=0.05)
    assert r.rank == 5
    assert r.total_sites == 5


def test_share_analysis_example2():
    rows = [
        ShareRow("SBL-SWL-4343", "Main Bazaar", 4_220_000),
        ShareRow("SBL-SWL-9344", "Main Bazaar", 13_990_000),
        ShareRow("SBL-SWL-8911", "College Road", 8_950_000),
        ShareRow("SBL-SWL-4538", "Airport Road", 10_440_000),
        ShareRow("SBL-SWL-9738", "Bypass Road", 8_500_000),
    ]
    r = analyze_share(rows, "SBL-SWL-4538")
    assert r.total == 46_100_000
    assert r.share_pct == pytest.approx(22.6, abs=0.05)
    assert r.rank == 2


# ── site ranking ──

def test_ranking_downtime_higher_is_worse():
    rows = [
        RankingRow("SILK-ATD-4504", "University Road", 314),
        RankingRow("FWBL-ATD-9147", "GT Road", 1_040),
        RankingRow("BAFL-ATD-2958", "Katchery Chowk", 1_175),
        RankingRow("MEBL-ATD-2957", "Model Bazaar", 2_402),
    ]
    r = rank_sites(rows, metric_label="downtime minutes", higher_is_worse=True)
    order = [x.atm_id for x in r.ranked]
    assert order == ["MEBL-ATD-2957", "BAFL-ATD-2958", "FWBL-ATD-9147", "SILK-ATD-4504"]
    assert r.ranked[0].rank == 1
    assert "remediation queue" in r.top_comment
    assert "MEBL-ATD-2957" in r.top_comment


def test_ranking_cash_dispensed_higher_is_better():
    rows = [
        RankingRow("HBLMFB-BHV-2293", "Main Bazaar", 3_690_000),
        RankingRow("UBL-BHV-5690", "Liaquat Road", 6_840_000),
        RankingRow("FINCA-BHV-3086", "Cantt Area", 3_110_000),
        RankingRow("SBL-BHV-7108", "Bypass Road", 7_630_000),
        RankingRow("SMBL-BHV-7260", "Sadar Bazaar", 7_530_000),
    ]
    r = rank_sites(rows, metric_label="cash dispensed", higher_is_worse=False)
    order = [x.atm_id for x in r.ranked]
    assert order == [
        "SBL-BHV-7108", "SMBL-BHV-7260", "UBL-BHV-5690", "HBLMFB-BHV-2293", "FINCA-BHV-3086",
    ]
    assert "protect on cash availability" in r.top_comment
    assert "SBL-BHV-7108" in r.top_comment
