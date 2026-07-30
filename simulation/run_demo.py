#!/usr/bin/env python
"""AsaanBank full-engine demo — a day in the life across loans and ATM ops.

Walks through a representative slice of the 51 task types (18 loan + 33 ATM)
trained into merged_finetune_train.jsonl, using real scenarios transcribed
from that dataset. Every number here is computed by the deterministic policy
engines in loan-agent / agents.atm — the LLM only narrates (falls back to
readable template narration if Ollama isn't running).

Usage:
    python run_demo.py                # full tour, LLM narration if available
    python run_demo.py --no-llm       # deterministic narratives only, fully offline
    python run_demo.py --chapter atm  # only the ATM half (or --chapter loans)

Point at your fine-tuned model with:
    LOAN_LLM_MODEL=<your-ollama-tag> ATM_LLM_MODEL=<your-ollama-tag> python run_demo.py
"""
from __future__ import annotations

import argparse

from app.agents.loans.agent import LoanAgent
from app.agents.loans.models import (
    BureauFacility, GoldOffer, LoanProductOption, LoanQuote, RunningLoan,
)

from agents.atm import AtmAgent, AtmMachine, Cassette, CustomerAtmProfile, Transaction
from agents.atm.cash_mechanics import (
    CashRunoutForecastInput, forecast_cash_runout, render_cash_runout_facts,
    triage_cassette_status, render_cassette_triage_facts,
    reconcile_cash, render_cash_reconciliation_facts,
)
from agents.atm.network_ops import (
    AlertRow, triage_alerts, render_alert_triage_facts,
    ReplenishmentCandidate, prioritize_replenishment, render_replenishment_facts,
    EodSiteRow, report_eod_position, render_eod_position_facts,
)
from agents.atm.customer_money import (
    NearbyAtm, AtmRecommendationParams, recommend_atm, render_atm_recommendation_facts,
    TravelCashParams, plan_travel_cash, render_travel_cash_facts,
)
from agents.atm.judgment import (
    SecurityObservation, assess_security_anomaly, render_security_assessment_facts,
    CardFraudInputs, assess_card_fraud, render_card_fraud_facts,
    UptimeSlaInputs, DowntimeIncident, check_uptime_sla, render_uptime_sla_facts,
)

W = 100


def hr(ch: str = "─") -> str:
    return ch * W


def chapter(title: str) -> None:
    print("\n\n" + hr("═"))
    print(title.center(W))
    print(hr("═"))


def scene(title: str) -> None:
    print(f"\n{hr()}\n{title}\n{hr()}")


def show(narrative: str, source: str) -> None:
    print(f"[{source}]  {narrative}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-llm", action="store_true", help="deterministic template narratives only")
    ap.add_argument("--chapter", choices=["loans", "atm", "all"], default="all")
    args = ap.parse_args()
    use_llm = not args.no_llm

    loan_agent = LoanAgent()
    atm_agent = AtmAgent()

    print(hr("═"))
    print("ASAANBANK — FULL ENGINE DEMO".center(W))
    print(f"18 loan task types + 33 ATM task types — {'live LLM narration' if use_llm else 'offline / template narration'}".center(W))
    print(hr("═"))

    # ================================================================ LOANS
    if args.chapter in ("loans", "all"):
        chapter("CHAPTER 1 — THE LOANS DESK")

        scene("Morning scan: triage the whole book (portfolio_triage)")
        result = loan_agent.portfolio_triage(use_llm=use_llm)
        queues = result.facts
        print(f"  OFFER: {len(queues['OFFER'])}   MONITOR: {len(queues['MONITOR'])}   DECLINE: {len(queues['DECLINE'])}")
        show(result.narrative, result.narrative_source)

        if queues["OFFER"]:
            cid = queues["OFFER"][0][0].customer_id
            scene(f"Deep dive: top OFFER file {cid} (proactive_offer_decision)")
            a = loan_agent.assess(cid, use_llm=use_llm)
            print(f"  {a.customer.name} — score {a.decision.score.score} ({a.decision.score.band}) — {a.decision.action}")
            show(a.narrative, a.narrative_source)

        if queues["DECLINE"]:
            cid = queues["DECLINE"][0][0].customer_id
            scene(f"Deep dive: worst DECLINE file {cid} (decline_with_alternatives)")
            a = loan_agent.assess(cid, use_llm=use_llm)
            show(a.narrative, a.narrative_source)

        scene("Credit-ops: reading a bureau file (ecib_report_reading)")
        r = loan_agent.ecib_report([
            BureauFacility("Consumer Durable", "KMBL", 2_050_000, "90+ DPD"),
            BureauFacility("Microfinance Loan", "BAFL", 910_000, "90+ DPD"),
            BureauFacility("Microfinance Loan", "MCB", 1_910_000, "Written off"),
        ], use_llm=use_llm)
        show(r.narrative, r.narrative_source)

        scene("Credit-ops: early-warning risk grade on a running facility (delinquency_risk_grading)")
        loan = RunningLoan(950_000, 0.318, "reducing", 24, 11, 800_000, 54_001)
        r = loan_agent.delinquency_risk("C001", loan, [2, 2, 14, 14], use_llm=use_llm)
        print(f"  Grade: {r.facts.grade} (score {r.facts.score})")
        show(r.narrative, r.narrative_source)

        scene("Customer-facing: sizing a gold-backed loan (gold_loan_sizing)")
        gold = GoldOffer(weight_tola=12.8, purity_k=21, rate_per_tola_24k=244_000, ltv_pct=0.70)
        r = loan_agent.gold_loan(gold, requested_amount=1_710_000, annual_rate=0.231, tenor_months=24, use_llm=use_llm)
        show(r.narrative, r.narrative_source)

        scene("Customer-facing: matching a need to the product shelf (product_recommendation)")
        shelf = [
            LoanProductOption("Personal Instalment Loan", 50_000, 3_000_000, [12, 24, 36, 48], 0.31, "reducing", 0.02, min_income=50_000),
            LoanProductOption("Salary Advance", 20_000, 500_000, [1, 3, 6], 0.27, "flat", 0.01, min_income=40_000, requires_salary_account=True),
            LoanProductOption("Deposit-backed Finance", 25_000, 10_000_000, [12, 24, 36], 0.16, "reducing", 0.005, requires_deposit=True),
            LoanProductOption("Gold-backed Finance", 30_000, 5_000_000, [12, 24, 36], 0.24, "reducing", 0.015, requires_gold=True),
        ]
        r = loan_agent.product_recommendation("C001", need_amount=1_180_000, shelf=shelf, deposit_amount=1_500_000, use_llm=use_llm)
        show(r.narrative, r.narrative_source)

        scene("Customer-facing: comparing three loan offers (loan_offer_comparison)")
        quotes = [
            LoanQuote("Offer A", 0.269, "reducing", 0.015),
            LoanQuote("Offer B", 0.185, "flat", 0.015),
            LoanQuote("Offer C", 0.161, "flat", 0.005),
        ]
        r = loan_agent.offer_comparison(1_230_000, 36, quotes, use_llm=use_llm)
        show(r.narrative, r.narrative_source)

    # ================================================================== ATM
    if args.chapter in ("atm", "all"):
        chapter("CHAPTER 2 — ATM OPERATIONS")

        scene("NOC alert queue — Multan zone, 10:37 (alert_triage)")
        alerts = [
            AlertRow("FWBL-MUX-9514", "Shahi Bazaar", "CASH_LOW", "cash below threshold", 111),
            AlertRow("NBP-MUX-8535", "Ring Road", "DISPENSER_FAULT", "dispenser hardware fault", 72),
            AlertRow("HMB-MUX-9222", "Shahi Bazaar", "UPS_ON_BATTERY", "running on UPS battery", 127),
            AlertRow("BIPL-MUX-7174", "College Road", "UPS_ON_BATTERY", "running on UPS battery", 43),
            AlertRow("HBL-MUX-4499", "GT Road", "DOOR_OPEN", "safe door open unscheduled", 194),
            AlertRow("AKBL-MUX-4493", "College Road", "PRINTER_PAPER_OUT", "receipt paper out", 224),
        ]
        facts = triage_alerts(alerts)
        narrative, source = atm_agent.narrate(render_alert_triage_facts(facts), "alert_triage", use_llm)
        show(narrative, source)

        scene("Cash run-out forecast (cash_runout_forecast)")
        machine = AtmMachine(
            atm_id="BIPL-SWL-5079", bank="BIPL", location="Katchery Chowk, Sahiwal",
            area_type="on-site (branch lobby)", model="Wincor Nixdorf Procash 2050xe", status="ONLINE",
            cassettes=[
                Cassette(1000, 1155, 2500), Cassette(1000, 1715, 2500),
                Cassette(500, 1074, 2000), Cassette(500, 823, 2000),
            ],
        )
        inp = CashRunoutForecastInput("07:10", 210_000, 150_000, "10:10", 3)
        facts = forecast_cash_runout(machine, inp)
        narrative, source = atm_agent.narrate(render_cash_runout_facts(facts), "cash_runout_forecast", use_llm)
        show(narrative, source)

        scene("Cassette status — mixed fault state (cassette_status_triage)")
        faulty = AtmMachine(
            atm_id="MEBL-KDU-6892", bank="MEBL", location="Katchery Chowk, Skardu", area_type="industrial estate",
            model="Hyosung MX 8200QT", status="ONLINE",
            cassettes=[
                Cassette(5000, 783, 3000, status="FAULT", note="pick failure, cassette locked out"),
                Cassette(1000, 2253, 3000, status="OK"),
                Cassette(1000, 1158, 3000, status="FAULT", note="pick failure, cassette locked out"),
                Cassette(500, 62, 2000, status="LOW", note="below low-cash threshold"),
            ],
        )
        facts = triage_cassette_status(faulty)
        narrative, source = atm_agent.narrate(render_cassette_triage_facts(facts), "cassette_status_triage", use_llm)
        show(narrative, source)

        scene("End-of-shift cash reconciliation (cash_reconciliation)")
        facts = reconcile_cash(opening_cash=1_967_000, loaded_this_cycle=11_400_000, dispensed_ej=10_030_000, physical_count=3_337_000)
        narrative, source = atm_agent.narrate(render_cash_reconciliation_facts(facts), "cash_reconciliation", use_llm)
        show(narrative, source)

        scene("CIT replenishment priority — Gwadar region (replenishment_priority)")
        candidates = [
            ReplenishmentCandidate("UMBL-GWD-5939", "Katchery Chowk", 7_254_000, 86_000),
            ReplenishmentCandidate("UMBL-GWD-8779", "Shahi Bazaar", 4_344_000, 89_000),
            ReplenishmentCandidate("UMBL-GWD-8911", "Liaquat Road", 16_294_000, 12_000),
            ReplenishmentCandidate("UMBL-GWD-6641", "Industrial Estate", 6_095_000, 121_000),
        ]
        facts = prioritize_replenishment(candidates, window_hours=12)
        narrative, source = atm_agent.narrate(render_replenishment_facts(facts), "replenishment_priority", use_llm)
        show(narrative, source)

        scene("Zone end-of-day position — Rahim Yar Khan (eod_position_report)")
        sites = [
            EodSiteRow("SMBL-RYK-6577", "University Road", "ONLINE", 290_000, 8_380_000),
            EodSiteRow("SMBL-RYK-4546", "Katchery Chowk", "OUT_OF_CASH", 0, 840_000),
            EodSiteRow("SMBL-RYK-8911", "Model Bazaar", "COMMS_DOWN", 38_000, 4_880_000),
            EodSiteRow("SMBL-RYK-2735", "Main Bazaar", "ONLINE", 12_500, 500_000),
            EodSiteRow("SMBL-RYK-4758", "Civil Lines", "ONLINE", 1_062_500, 2_480_000),
            EodSiteRow("SMBL-RYK-1879", "Shahi Bazaar", "COMMS_DOWN", 2_461_000, 3_270_000),
        ]
        facts = report_eod_position(sites)
        narrative, source = atm_agent.narrate(render_eod_position_facts(facts), "eod_position_report", use_llm)
        show(narrative, source)

        scene("Security review — overnight anomaly (security_anomaly_assessment)")
        suspect = AtmMachine("BIPL-KSR-1111", "BIPL", "Civil Lines, Kasur", area_type="cantonment area", model="Hyosung MX 5600", status="ONLINE")
        obs = SecurityObservation(
            atm=suspect, event_count=37, window_start="04:00", window_end="04:20", total_amount=740_000,
            unmatched_authorisation=True, safe_door_opened_time="03:24", cit_visit_scheduled=False,
        )
        facts = assess_security_anomaly(obs)
        narrative, source = atm_agent.narrate(render_security_assessment_facts(facts), "security_anomaly_assessment", use_llm)
        show(narrative, source)

        scene("Customer support: card fraud check on recent withdrawals (card_fraud_assessment)")
        customer = CustomerAtmProfile(
            name="Talha Yousafzai", bank="Standard Chartered Bank Pakistan (SCB)", account_type="Asaan Account",
            account_number="PK71SCB8183440622", available_balance=6_613, card_network="PayPak", card_tier="PayPak Classic",
            card_last4="4929", per_txn_limit=25_000, daily_limit=50_000, free_offus_left=3,
        )
        txns = [
            Transaction("2025-05-16", "01:59", "ATM withdrawal Skardu - Ring Road", "ATM", -25_000),
            Transaction("2025-05-16", "02:01", "ATM withdrawal Skardu - Bypass Road", "ATM", -25_000),
            Transaction("2025-05-16", "02:04", "ATM withdrawal Skardu - GT Road", "ATM", -25_000),
            Transaction("2025-05-16", "02:07", "ATM withdrawal Skardu - Airport Road", "ATM", -25_000),
        ]
        facts = assess_card_fraud(CardFraudInputs(profile=customer, transactions=txns))
        narrative, source = atm_agent.narrate(render_card_fraud_facts(facts), "card_fraud_assessment", use_llm)
        show(narrative, source)

        scene("Customer support: which ATM should I walk to? (atm_recommendation)")
        seeker = CustomerAtmProfile(
            name="Kinza Tarar", bank="Bank Alfalah (BAFL)", account_type="Business Current Account",
            account_number="PK63BAFL2538300999", available_balance=673_355, card_network="PayPak",
            card_tier="PayPak Gold", card_last4="5851", per_txn_limit=30_000, daily_limit=75_000, free_offus_left=4,
        )
        candidates = [
            NearbyAtm("BAFL-VHR-6372", "BAFL", "Airport Road", 1.9, "OFFLINE", 5_154_500, [1000, 500], 9),
            NearbyAtm("SCB-VHR-6090", "SCB", "Model Town", 1.1, "ONLINE", 10_876_000, [5000, 1000], 7),
            NearbyAtm("BAFL-VHR-6734", "BAFL", "Airport Road", 6.1, "ONLINE", 769_500, [1000], 9),
        ]
        facts = recommend_atm(AtmRecommendationParams(customer=seeker, amount_needed=40_000, off_us_charge=18.5, candidates=candidates))
        narrative, source = atm_agent.narrate(render_atm_recommendation_facts(facts), "atm_recommendation", use_llm)
        show(narrative, source)

        scene("Customer support: uptime SLA breach check (uptime_sla_check)")
        m = AtmMachine("ABL-BNU-9563", "ABL", "College Road, Bannu", area_type="cantonment area", model="NCR SelfServ 26", status="ONLINE")
        uptime_inputs = UptimeSlaInputs(
            atm=m, period_days=31, sla_pct=98.0,
            incidents=[
                DowntimeIncident("H0013", "cash-out sensor mismatch", 852),
                DowntimeIncident("N0009", "note quality reject rate high", 721),
                DowntimeIncident("D0011", "dispenser fault - note pick failure", 254),
            ],
        )
        facts = check_uptime_sla(uptime_inputs)
        narrative, source = atm_agent.narrate(render_uptime_sla_facts(uptime_inputs, facts), "uptime_sla_check", use_llm)
        show(narrative, source)

    chapter("SUMMARY")
    print(
        "This tour covered 16 of the 51 trained task types across both domains.\n"
        "Every task type — 18 loan + 33 ATM — is reachable via the full REST API:\n"
        "  GET  /tasks                     list every task + its input shape\n"
        "  POST /loans/{task_name}         any of the 18 loan tasks\n"
        "  POST /atm/tasks/{task_name}     any of the 33 ATM tasks\n"
        "See README.md for the full task list and `uvicorn service:app --port 8080`.\n"
    )


if __name__ == "__main__":
    main()
