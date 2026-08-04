"""Agentic orchestration layer over the 51 banking tasks.

Turns every loan + ATM task into a registered "tool" with a formal schema,
then runs a ReAct-style loop: the orchestrator reads the user's request,
decides which tool(s) to call, chains the outputs, and produces a
coherent multi-step answer -- all grounded in the deterministic policy
engine's computed facts.

Design principles:
  * The LLM never decides anything -- only the policy engine does.
  * Every tool call is deterministic and auditable (input -> facts -> output).
  * The same fine-tuned model handles narration; no new model introduced.
  * Fallback to template narration whenever the LLM is unreachable.
"""
from __future__ import annotations

import dataclasses
import json
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from app.agents.loans.agent import LoanAgent
from app.agents.loans.models import (
    BureauFacility, CustomerProfile, GoldOffer, LoanProductOption, LoanQuote, RunningLoan,
    SalaryAdvanceProduct, Obligation,
)

from agents.atm import AtmAgent, AtmMachine, Cassette, CustomerAtmProfile, Transaction
from agents.atm.cash_mechanics import CashRunoutForecastInput, DenominationMixPlanningInput
from agents.atm.network_ops import AlertRow, ReplenishmentCandidate, CitCandidate, EodSiteRow, GrowthInput, ShareRow, RankingRow
from agents.atm.customer_money import (
    AtmFeeParams, AtmRecommendationParams, CashAdvanceParams, CashAffordabilityParams,
    MonthlyAtmActivity, TravelCashParams, CashCarryingCostParams, SurgeCapacityParams, Bill,
)
from agents.atm.judgment import (
    FaultDiagnosisInputs, SecurityObservation, CardFraudInputs, CardRetentionInputs,
    DisputeInputs, HourlyLogEntry, UptimeSlaInputs, DowntimeIncident,
)


class Domain(str, Enum):
    LOANS = "loans"
    ATM = "atm"


@dataclass
class ToolDef:
    """Formal description of one callable banking task."""
    name: str
    domain: Domain
    description: str
    parameters: dict[str, Any]
    requires_customer: bool = False


LOAN_TOOLS: list[ToolDef] = [
    ToolDef("relationship_scoring", Domain.LOANS,
        "Score a customer's relationship with the bank (0-10 scale). Returns STRONG/ACCEPTABLE/THIN/POOR.",
        {"customer_id": {"type": "string", "description": "Customer ID in the book"}}, True),
    ToolDef("low_cash_detection", Domain.LOANS,
        "Detect whether a customer is genuinely short of cash based on balance trends.",
        {"customer_id": {"type": "string", "description": "Customer ID in the book"}}, True),
    ToolDef("proactive_offer_decision", Domain.LOANS,
        "Make a lending decision: OFFER / MONITOR / DECLINE with terms if approved.",
        {"customer_id": {"type": "string", "description": "Customer ID in the book"}}, True),
    ToolDef("decline_with_alternatives", Domain.LOANS,
        "Decision gate for declined customers plus secured alternatives.",
        {"customer_id": {"type": "string", "description": "Customer ID in the book"}}, True),
    ToolDef("portfolio_triage", Domain.LOANS,
        "Scan the entire customer book into OFFER / MONITOR / DECLINE queues.",
        {}, False),
    ToolDef("dbr_calculation", Domain.LOANS,
        "Calculate debt-burden ratio. DBR = (existing obligations + new EMI) / net income. Cap is 40%.",
        {"net_income": {"type": "integer"}, "existing_obligations": {"type": "integer"}, "new_emi": {"type": "number"}}, False),
    ToolDef("max_affordable_loan", Domain.LOANS,
        "Calculate the largest loan amount a customer can responsibly borrow under the 40% DBR cap.",
        {"customer_id": {"type": "string", "description": "Customer ID in the book"}}, True),
    ToolDef("emi_calculation", Domain.LOANS,
        "Calculate monthly instalment (EMI). Supports reducing and flat rates.",
        {"principal": {"type": "integer"}, "annual_rate": {"type": "number"}, "tenor_months": {"type": "integer"}, "rate_type": {"type": "string", "enum": ["reducing", "flat"]}}, False),
    ToolDef("ecib_report_reading", Domain.LOANS,
        "Read and interpret an eCIB credit bureau report. Identifies write-offs and delinquencies.",
        {"facilities": {"type": "array", "description": "List of bureau facilities"}}, False),
    ToolDef("delinquency_risk_grading", Domain.LOANS,
        "Grade the delinquency risk of a running loan: LOW, MEDIUM, or HIGH.",
        {"customer_id": {"type": "string"}, "loan": {"type": "object"}, "late_delays": {"type": "array"}}, True),
    ToolDef("restructuring_assessment", Domain.LOANS,
        "Assess whether restructuring a loan after an income drop fixes affordability.",
        {"customer_id": {"type": "string"}, "loan": {"type": "object"}, "new_income": {"type": "integer"}, "extension_months": {"type": "integer"}}, True),
    ToolDef("risk_based_pricing", Domain.LOANS,
        "Determine the appropriate interest rate based on the customer's score band.",
        {"customer_id": {"type": "string"}, "base_rate": {"type": "number"}, "acceptable_markup_pts": {"type": "number"}, "thin_markup_pts": {"type": "number"}, "requested_amount": {"type": "integer"}, "tenor_months": {"type": "integer"}}, True),
    ToolDef("topup_eligibility", Domain.LOANS,
        "Check eligibility for a top-up on an existing running loan.",
        {"customer_id": {"type": "string"}, "loan": {"type": "object"}, "late_in_12m": {"type": "integer"}, "request_amount": {"type": "integer"}}, True),
    ToolDef("loan_offer_comparison", Domain.LOANS,
        "Compare multiple loan quotes side by side. Ranks cheapest first.",
        {"principal": {"type": "integer"}, "tenor_months": {"type": "integer"}, "quotes": {"type": "array"}}, False),
    ToolDef("early_settlement_analysis", Domain.LOANS,
        "Compare cost of settling early vs continuing. Includes settlement fee.",
        {"loan": {"type": "object"}, "settlement_fee_pct": {"type": "number"}}, False),
    ToolDef("gold_loan_sizing", Domain.LOANS,
        "Size a gold-backed loan. Returns granted amount and shortfall.",
        {"weight_tola": {"type": "number"}, "purity_k": {"type": "integer"}, "rate_per_tola_24k": {"type": "integer"}, "ltv_pct": {"type": "number"}, "requested_amount": {"type": "integer"}, "annual_rate": {"type": "number"}, "tenor_months": {"type": "integer"}}, False),
    ToolDef("salary_advance_assessment", Domain.LOANS,
        "Assess eligibility for a salary advance product.",
        {"customer_id": {"type": "string"}, "limit_multiple": {"type": "number"}, "fee_pct": {"type": "number"}, "annual_rate": {"type": "number"}, "tenor_months": {"type": "integer"}, "requested_amount": {"type": "integer"}}, True),
    ToolDef("product_recommendation", Domain.LOANS,
        "Match a customer's loan need to the product shelf. Ranks survivors by total cost.",
        {"customer_id": {"type": "string"}, "need_amount": {"type": "integer"}, "shelf": {"type": "array"}, "has_gold": {"type": "boolean"}, "deposit_amount": {"type": "integer"}, "has_salary_account": {"type": "boolean"}}, True),
]

ATM_TOOLS: list[ToolDef] = [
    ToolDef("cash_runout_forecast", Domain.ATM,
        "Forecast when an ATM will run out of cash and check CIT survival.",
        {"atm_id": {"type": "string"}, "bank": {"type": "string"}, "location": {"type": "string"}, "cassettes": {"type": "array"}, "current_time": {"type": "string"}, "dispense_rate_per_hour": {"type": "number"}, "low_cash_threshold": {"type": "integer"}, "cit_time": {"type": "string"}, "cit_hours_from_now": {"type": "number"}}, False),
    ToolDef("cassette_status_triage", Domain.ATM,
        "Triage cassette health: FAULT, LOW, OK. Calculates max withdrawable.",
        {"atm_id": {"type": "string"}, "bank": {"type": "string"}, "location": {"type": "string"}, "cassettes": {"type": "array"}}, False),
    ToolDef("cash_reconciliation", Domain.ATM,
        "Reconcile expected vs physical cash count. Returns balanced/shortage/overage.",
        {"opening_cash": {"type": "integer"}, "loaded_this_cycle": {"type": "integer"}, "dispensed_ej": {"type": "integer"}, "physical_count": {"type": "integer"}}, False),
    ToolDef("denomination_mix_planning", Domain.ATM,
        "Plan optimal denomination mix from withdrawal histogram.",
        {"histogram": {"type": "array"}, "denominations": {"type": "array"}, "cassette_capacity_notes": {"type": "object"}}, False),
    ToolDef("denomination_dispensability", Domain.ATM,
        "Check if ATM can exactly dispense a requested amount.",
        {"atm_id": {"type": "string"}, "bank": {"type": "string"}, "location": {"type": "string"}, "cassettes": {"type": "array"}, "requested_amount": {"type": "integer"}}, False),
    ToolDef("cash_load_planning", Domain.ATM,
        "Plan cash load amount and denomination mix for an ATM.",
        {"avg_daily_dispense": {"type": "integer"}, "days_until_cit": {"type": "integer"}, "policy_buffer_pct": {"type": "number"}, "cassette_specs": {"type": "array"}, "withdrawal_mix": {"type": "object"}}, False),
    ToolDef("withdrawal_feasibility", Domain.ATM,
        "Check if a customer's ATM withdrawal is feasible.",
        {"customer": {"type": "object"}, "atm": {"type": "object"}, "requested_amount": {"type": "integer"}, "transactions": {"type": "array"}}, True),
    ToolDef("daily_limit_remaining", Domain.ATM,
        "Show remaining daily ATM withdrawal limit after today's transactions.",
        {"customer": {"type": "object"}, "transactions": {"type": "array"}, "requested_amount": {"type": "integer"}}, True),
    ToolDef("alert_triage", Domain.ATM,
        "Triage NOC alerts by severity P1-P4. Identifies dispatch targets.",
        {"alerts": {"type": "array", "description": "List of alerts"}}, False),
    ToolDef("replenishment_priority", Domain.ATM,
        "Rank ATMs by replenishment urgency within a time window.",
        {"candidates": {"type": "array"}, "window_hours": {"type": "number"}}, False),
    ToolDef("cit_route_planning", Domain.ATM,
        "Plan a CIT vehicle route within cash-carry capacity.",
        {"candidates": {"type": "array"}, "cap": {"type": "integer"}}, False),
    ToolDef("demand_forecast", Domain.ATM,
        "Forecast ATM demand using last week's data and seasonal multipliers.",
        {"atm": {"type": "object"}, "week_dispensed": {"type": "array"}, "forecast_day": {"type": "string"}, "seasonal_label": {"type": "string"}, "seasonal_multiplier": {"type": "number"}, "machine_capacity": {"type": "integer"}}, False),
    ToolDef("eod_position_report", Domain.ATM,
        "End-of-day position report for a zone of ATMs.",
        {"sites": {"type": "array"}, "low_cash_threshold": {"type": "integer"}}, False),
    ToolDef("growth_analysis", Domain.ATM,
        "Calculate month-on-month growth for an ATM metric.",
        {"atm_id": {"type": "string"}, "metric_label": {"type": "string"}, "old_period": {"type": "string"}, "old_value": {"type": "integer"}, "new_period": {"type": "string"}, "new_value": {"type": "integer"}, "is_money": {"type": "boolean"}}, False),
    ToolDef("share_analysis", Domain.ATM,
        "Calculate a site's share of total zone dispense. Returns rank.",
        {"sites": {"type": "array"}, "target_atm_id": {"type": "string"}}, False),
    ToolDef("ranking", Domain.ATM,
        "Rank ATM sites by any numeric metric.",
        {"sites": {"type": "array"}, "metric_label": {"type": "string"}, "higher_is_worse": {"type": "boolean"}}, False),
    ToolDef("atm_fee_calculation", Domain.ATM,
        "Calculate ATM fees including free allowance, FED, and cheaper route.",
        {"free_offus_per_month": {"type": "integer"}, "offus_already_made": {"type": "integer"}, "planned_offus_count": {"type": "integer"}, "fee_per_txn": {"type": "number"}, "fed_pct": {"type": "number"}, "per_txn_limit": {"type": "integer"}, "amount_needed": {"type": "integer"}}, False),
    ToolDef("atm_recommendation", Domain.ATM,
        "Recommend the best nearby ATM for a customer to withdraw from.",
        {"customer": {"type": "object"}, "amount_needed": {"type": "integer"}, "off_us_charge": {"type": "number"}, "candidates": {"type": "array"}}, True),
    ToolDef("cash_advance_assessment", Domain.ATM,
        "Assess a cash advance request against credit card limits.",
        {"customer": {"type": "object"}, "atm": {"type": "object"}, "requested_amount": {"type": "integer"}, "fee_pct": {"type": "number"}, "fee_min": {"type": "integer"}, "monthly_markup_pct": {"type": "number"}, "days_held": {"type": "integer"}}, True),
    ToolDef("cash_affordability_check", Domain.ATM,
        "Check if a customer can safely withdraw considering upcoming bills.",
        {"customer": {"type": "object"}, "withdrawal_amount": {"type": "integer"}, "bills": {"type": "array"}, "salary_date": {"type": "string"}}, True),
    ToolDef("monthly_atm_cost_summary", Domain.ATM,
        "Summarize a customer's total ATM costs for the month.",
        {"on_us_count": {"type": "integer"}, "off_us_count": {"type": "integer"}, "inquiries_count": {"type": "integer"}, "avg_withdrawal": {"type": "integer"}, "free_offus_allowance": {"type": "integer"}, "offus_fee": {"type": "number"}, "inquiry_fee": {"type": "number"}}, False),
    ToolDef("spend_pattern_summary", Domain.ATM,
        "Summarize cash spending patterns and cash vs card ratio.",
        {"transactions": {"type": "array"}}, False),
    ToolDef("travel_cash_planning", Domain.ATM,
        "Plan travel cash needs and withdrawal strategy.",
        {"customer": {"type": "object"}, "days": {"type": "integer"}, "daily_spend": {"type": "integer"}, "working_atms_at_destination": {"type": "integer"}, "off_us_charge": {"type": "number"}}, True),
    ToolDef("cash_carrying_cost", Domain.ATM,
        "Calculate carrying cost of idle ATM cash and evaluate load reduction.",
        {"atm": {"type": "object"}, "idle_cash": {"type": "integer"}, "daily_dispense": {"type": "integer"}, "annual_rate": {"type": "number"}, "cit_trip_cost": {"type": "integer"}, "window_days": {"type": "integer"}}, False),
    ToolDef("surge_capacity_planning", Domain.ATM,
        "Plan cash capacity for a surge period like Ramzan.",
        {"atm": {"type": "object"}, "normal_daily_dispense": {"type": "integer"}, "season_name": {"type": "string"}, "demand_multiplier": {"type": "number"}, "closed_days": {"type": "integer"}, "max_capacity": {"type": "integer"}}, False),
    ToolDef("fault_root_cause", Domain.ATM,
        "Diagnose ATM fault root cause from vendor fault code.",
        {"atm": {"type": "object"}, "fault_code": {"type": "string"}, "down_minutes": {"type": "integer"}, "failed_transactions": {"type": "integer"}, "vendor_table": {"type": "object"}}, False),
    ToolDef("security_anomaly_assessment", Domain.ATM,
        "Assess security anomaly: jackpotting, skimming, cash trapping, or genuine surge.",
        {"atm": {"type": "object"}, "event_count": {"type": "integer"}, "window_start": {"type": "string"}, "window_end": {"type": "string"}, "total_amount": {"type": "integer"}, "unmatched_authorisation": {"type": "boolean"}, "safe_door_opened_time": {"type": "string"}, "cit_visit_scheduled": {"type": "boolean"}, "usb_port_cover_open": {"type": "boolean"}}, False),
    ToolDef("card_fraud_assessment", Domain.ATM,
        "Assess card transactions for fraud: velocity or impossible travel.",
        {"customer": {"type": "object"}, "transactions": {"type": "array"}}, True),
    ToolDef("card_retention_guidance", Domain.ATM,
        "Guide a customer whose card was retained by an ATM.",
        {"customer": {"type": "object"}, "atm": {"type": "object"}, "atm_bank_full_name": {"type": "string"}, "hours_since_capture": {"type": "integer"}, "capture_reason": {"type": "string"}, "hold_days": {"type": "integer"}}, True),
    ToolDef("failed_transaction_dispute", Domain.ATM,
        "Assess failed ATM transaction dispute. Returns verdict.",
        {"atm": {"type": "object"}, "date": {"type": "string"}, "debited": {"type": "integer"}, "received": {"type": "integer"}, "journal_entry": {"type": "string"}, "reconciliation_excess": {"type": "integer"}, "reversal_window_days": {"type": "integer"}, "days_since_transaction": {"type": "integer"}}, False),
    ToolDef("interbank_settlement", Domain.ATM,
        "Calculate interbank interchange settlement position.",
        {"bank_code": {"type": "string"}, "on_us_count": {"type": "integer"}, "acquired_offus_count": {"type": "integer"}, "issued_offus_count": {"type": "integer"}, "interchange_rate": {"type": "number"}}, False),
    ToolDef("trend_summary", Domain.ATM,
        "Summarize hourly dispensing trend. Identifies peak hour and CIT timing.",
        {"atm": {"type": "object"}, "opening_balance": {"type": "integer"}, "log": {"type": "array"}}, False),
    ToolDef("uptime_sla_check", Domain.ATM,
        "Check ATM uptime against SLA target. Identifies biggest downtime contributor.",
        {"atm": {"type": "object"}, "period_days": {"type": "integer"}, "sla_pct": {"type": "number"}, "incidents": {"type": "array"}}, False),
]

ALL_TOOLS: list[ToolDef] = LOAN_TOOLS + ATM_TOOLS
TOOLS_BY_NAME: dict[str, ToolDef] = {t.name: t for t in ALL_TOOLS}


# ════════════════════════════════════════════════════════════════════════
# Helper: convert raw dicts to dataclass instances
# ════════════════════════════════════════════════════════════════════════

def _resolve_customer(customer_id: str | None, agent: LoanAgent) -> CustomerProfile | None:
    if customer_id is None:
        return None
    return agent.get(customer_id)


def _to_cassette(c: dict) -> Cassette:
    return Cassette(denom=int(c.get("denom", 0)), notes=int(c.get("notes", 0)),
                    capacity=int(c.get("capacity", 0)), status=str(c.get("status", "OK")),
                    note=str(c.get("note", "")))


def _to_atm_machine(d: dict) -> AtmMachine:
    return AtmMachine(atm_id=str(d.get("atm_id", "")), bank=str(d.get("bank", "")),
                      location=str(d.get("location", "")), area_type=str(d.get("area_type", "")),
                      model=str(d.get("model", "")), status=str(d.get("status", "ONLINE")),
                      cassettes=[_to_cassette(c) for c in d.get("cassettes", [])])


def _to_customer_profile(d: dict) -> CustomerProfile:
    return CustomerProfile(
        customer_id=str(d.get("customer_id", "")), name=str(d.get("name", "")),
        age=int(d.get("age", 0)), bank=str(d.get("bank", "")),
        bank_code=str(d.get("bank_code", "")), city=str(d.get("city", "")),
        employment=str(d.get("employment", "")),
        net_monthly_income=int(d.get("net_monthly_income", 0)),
        account_age_years=float(d.get("account_age_years", 0)),
        salary_months_12=int(d.get("salary_months_12", 0)),
        avg_balance_6m=int(d.get("avg_balance_6m", 0)),
        previous_loan=str(d.get("previous_loan", "none")),
        cheque_bounce_12m=bool(d.get("cheque_bounce_12m", False)),
        ecib_dpd90_24m=bool(d.get("ecib_dpd90_24m", False)),
        ecib_writeoff=bool(d.get("ecib_writeoff", False)),
        obligations=[Obligation(label=o["label"], monthly_amount=int(o["monthly_amount"]))
                     for o in d.get("obligations", [])],
        eom_balances=[int(b) for b in d.get("eom_balances", [])],
        days_below_5k_30d=int(d.get("days_below_5k_30d", 0)),
        salary_to_low_gap_days=int(d["salary_to_low_gap_days"]) if "salary_to_low_gap_days" in d else None)


def _to_atm_customer_profile(d: dict) -> CustomerAtmProfile:
    return CustomerAtmProfile(
        name=str(d.get("name", "")), bank=str(d.get("bank", "")),
        account_type=str(d.get("account_type", "")),
        account_number=str(d.get("account_number", "")),
        available_balance=int(d.get("available_balance", 0)),
        card_network=str(d.get("card_network", "")), card_tier=str(d.get("card_tier", "")),
        card_last4=str(d.get("card_last4", "")),
        per_txn_limit=int(d.get("per_txn_limit", 0)),
        daily_limit=int(d.get("daily_limit", 0)),
        free_offus_left=int(d.get("free_offus_left", 0)),
        credit_line=int(d.get("credit_line", 0)),
        credit_utilised=int(d.get("credit_utilised", 0)),
        cash_advance_sub_limit_pct=float(d.get("cash_advance_sub_limit_pct", 0)),
        cash_advance_drawn=int(d.get("cash_advance_drawn", 0)))


def _to_transaction(d: dict) -> Transaction:
    return Transaction(date=str(d.get("date", "")), time=str(d.get("time", "")),
                       description=str(d.get("description", "")),
                       channel=str(d.get("channel", "")), amount=int(d.get("amount", 0)))


def _to_bureau_facility(d: dict) -> BureauFacility:
    return BureauFacility(facility=str(d.get("facility", "")),
                          institution=str(d.get("institution", "")),
                          amount=int(d.get("amount", 0)), status=str(d.get("status", "")))


def _to_gold_offer(d: dict) -> GoldOffer:
    return GoldOffer(weight_tola=float(d.get("weight_tola", 0)),
                     purity_k=int(d.get("purity_k", 0)),
                     rate_per_tola_24k=int(d.get("rate_per_tola_24k", 0)),
                     ltv_pct=float(d.get("ltv_pct", 0)))


def _to_loan_product_option(d: dict) -> LoanProductOption:
    return LoanProductOption(name=str(d.get("name", "")),
                             min_amount=int(d.get("min_amount", 0)),
                             max_amount=int(d.get("max_amount", 0)),
                             tenors=[int(t) for t in d.get("tenors", [])],
                             annual_rate=float(d.get("annual_rate", 0)),
                             rate_type=str(d.get("rate_type", "reducing")),
                             processing_fee_pct=float(d.get("processing_fee_pct", 0)),
                             min_income=int(d.get("min_income", 0)))


def _to_loan_quote(d: dict) -> LoanQuote:
    return LoanQuote(label=str(d.get("label", "")),
                     annual_rate=float(d.get("annual_rate", 0)),
                     rate_type=str(d.get("rate_type", "reducing")),
                     processing_fee_pct=float(d.get("processing_fee_pct", 0)))


def _to_running_loan(d: dict) -> RunningLoan:
    return RunningLoan(original_amount=int(d.get("original_amount", 0)),
                       annual_rate=float(d.get("annual_rate", 0)),
                       rate_type=str(d.get("rate_type", "reducing")),
                       instalments_total=int(d.get("instalments_total", 0)),
                       instalments_paid=int(d.get("instalments_paid", 0)),
                       outstanding=int(d.get("outstanding", 0)),
                       instalment=int(d.get("instalment", 0)))


def _to_salary_advance_product(d: dict) -> SalaryAdvanceProduct:
    return SalaryAdvanceProduct(limit_multiple=float(d.get("limit_multiple", 1.0)),
                                fee_pct=float(d.get("fee_pct", 0)),
                                annual_rate=float(d.get("annual_rate", 0)),
                                tenor_months=int(d.get("tenor_months", 3)))


def _to_alert_row(d: dict) -> AlertRow:
    return AlertRow(atm_id=str(d.get("atm_id", "")), location=str(d.get("location", "")),
                    alert_code=str(d.get("alert_code", "")),
                    description=str(d.get("description", "")),
                    open_minutes=int(d.get("open_minutes", 0)))


def _to_replenishment_candidate(d: dict) -> ReplenishmentCandidate:
    return ReplenishmentCandidate(atm_id=str(d.get("atm_id", "")),
                                  location=str(d.get("location", "")),
                                  cash_on_hand=int(d.get("cash_on_hand", 0)),
                                  dispense_rate_per_hr=int(d.get("dispense_rate_per_hr", 0)))


def _to_cit_candidate(d: dict) -> CitCandidate:
    return CitCandidate(atm_id=str(d.get("atm_id", "")),
                        distance_km=float(d.get("distance_km", 0)),
                        cash_left=int(d.get("cash_left", 0)),
                        hours_to_empty=float(d.get("hours_to_empty", 0)),
                        requested_load=int(d.get("requested_load", 0)))


def _to_eod_site_row(d: dict) -> EodSiteRow:
    return EodSiteRow(atm_id=str(d.get("atm_id", "")),
                      location=str(d.get("location", "")),
                      status=str(d.get("status", "ONLINE")),
                      cash_on_hand=int(d.get("cash_on_hand", 0)),
                      dispensed_today=int(d.get("dispensed_today", 0)))


def _to_share_row(d: dict) -> ShareRow:
    return ShareRow(atm_id=str(d.get("atm_id", "")),
                    location=str(d.get("location", "")),
                    value=int(d.get("value", 0)))


def _to_ranking_row(d: dict) -> RankingRow:
    return RankingRow(atm_id=str(d.get("atm_id", "")),
                      location=str(d.get("location", "")),
                      value=float(d.get("value", 0)))


def _to_downtime_incident(d: dict) -> DowntimeIncident:
    return DowntimeIncident(code=str(d.get("code", "")),
                            description=str(d.get("description", "")),
                            minutes=int(d.get("minutes", 0)))


def _to_hourly_log_entry(d: dict) -> HourlyLogEntry:
    return HourlyLogEntry(hour=str(d.get("hour", "")),
                          balance_after=int(d.get("balance_after", 0)),
                          dispensed=int(d.get("dispensed", 0)))


def _to_bill(d: dict) -> Bill:
    return Bill(label=str(d.get("label", "")), amount=int(d.get("amount", 0)),
                due_date=str(d.get("due_date", "")))


# ════════════════════════════════════════════════════════════════════════
# Tool executors
# ════════════════════════════════════════════════════════════════════════

def _exec_loan_tool(tool_name: str, params: dict, loan_agent: LoanAgent) -> Any:
    """Execute a loan tool by name with raw params dict."""
    if tool_name == "relationship_scoring":
        c = _resolve_customer(params.get("customer_id"), loan_agent)
        if not c:
            raise ValueError(f"Customer {params.get('customer_id')} not found")
        from app.agents.loans.policy import relationship_score
        return relationship_score(c)

    if tool_name == "low_cash_detection":
        c = _resolve_customer(params.get("customer_id"), loan_agent)
        if not c:
            raise ValueError(f"Customer {params.get('customer_id')} not found")
        from app.agents.loans.policy import detect_stress
        return detect_stress(c)

    if tool_name == "proactive_offer_decision":
        c = _resolve_customer(params.get("customer_id"), loan_agent)
        if not c:
            raise ValueError(f"Customer {params.get('customer_id')} not found")
        from app.agents.loans.policy import decide
        return decide(c, loan_agent.product)

    if tool_name == "decline_with_alternatives":
        c = _resolve_customer(params.get("customer_id"), loan_agent)
        if not c:
            raise ValueError(f"Customer {params.get('customer_id')} not found")
        from app.agents.loans.policy import decide
        return decide(c, loan_agent.product)

    if tool_name == "portfolio_triage":
        from app.agents.loans.policy import triage
        return triage(loan_agent.customers, loan_agent.product)

    if tool_name == "dbr_calculation":
        from app.agents.loans.policy import dbr
        return dbr(int(params["net_income"]), int(params["existing_obligations"]),
                   float(params["new_emi"]))

    if tool_name == "max_affordable_loan":
        c = _resolve_customer(params.get("customer_id"), loan_agent)
        if not c:
            raise ValueError(f"Customer {params.get('customer_id')} not found")
        from app.agents.loans.policy import max_affordable_loan
        tenor = int(params.get("tenor", loan_agent.product.tenors[-1]))
        return max_affordable_loan(c, loan_agent.product, tenor)

    if tool_name == "emi_calculation":
        from app.agents.loans.policy import emi
        return round(emi(int(params["principal"]), float(params["annual_rate"]),
                         int(params["tenor_months"]),
                         str(params.get("rate_type", "reducing"))))

    if tool_name == "ecib_report_reading":
        from app.agents.loans.policy import read_ecib_report
        facilities = [_to_bureau_facility(f) for f in params["facilities"]]
        return read_ecib_report(facilities)

    if tool_name == "delinquency_risk_grading":
        c = _resolve_customer(params.get("customer_id"), loan_agent)
        if not c:
            raise ValueError(f"Customer {params.get('customer_id')} not found")
        from app.agents.loans.policy import grade_delinquency_risk
        loan = _to_running_loan(params["loan"])
        late = [int(d) for d in params["late_delays"]]
        return grade_delinquency_risk(c, loan, late)

    if tool_name == "restructuring_assessment":
        c = _resolve_customer(params.get("customer_id"), loan_agent)
        if not c:
            raise ValueError(f"Customer {params.get('customer_id')} not found")
        from app.agents.loans.policy import assess_restructuring
        loan = _to_running_loan(params["loan"])
        return assess_restructuring(c, loan, int(params["new_income"]),
                                     int(params["extension_months"]))

    if tool_name == "risk_based_pricing":
        c = _resolve_customer(params.get("customer_id"), loan_agent)
        if not c:
            raise ValueError(f"Customer {params.get('customer_id')} not found")
        from app.agents.loans.policy import price_by_risk
        return price_by_risk(c, float(params["base_rate"]),
                             float(params["acceptable_markup_pts"]),
                             float(params["thin_markup_pts"]),
                             int(params["requested_amount"]),
                             int(params["tenor_months"]))

    if tool_name == "topup_eligibility":
        c = _resolve_customer(params.get("customer_id"), loan_agent)
        if not c:
            raise ValueError(f"Customer {params.get('customer_id')} not found")
        from app.agents.loans.policy import assess_topup
        loan = _to_running_loan(params["loan"])
        return assess_topup(c, loan, int(params["late_in_12m"]),
                            int(params["request_amount"]))

    if tool_name == "loan_offer_comparison":
        from app.agents.loans.policy import compare_loan_offers
        quotes = [_to_loan_quote(q) for q in params["quotes"]]
        return compare_loan_offers(int(params["principal"]),
                                    int(params["tenor_months"]), quotes)

    if tool_name == "early_settlement_analysis":
        from app.agents.loans.policy import analyze_early_settlement
        loan = _to_running_loan(params["loan"])
        return analyze_early_settlement(loan, float(params["settlement_fee_pct"]))

    if tool_name == "gold_loan_sizing":
        from app.agents.loans.policy import size_gold_loan
        gold = _to_gold_offer(params)
        return size_gold_loan(gold, int(params["requested_amount"]),
                              float(params["annual_rate"]),
                              int(params["tenor_months"]))

    if tool_name == "salary_advance_assessment":
        c = _resolve_customer(params.get("customer_id"), loan_agent)
        if not c:
            raise ValueError(f"Customer {params.get('customer_id')} not found")
        from app.agents.loans.policy import assess_salary_advance
        product = _to_salary_advance_product(params)
        return assess_salary_advance(c, product, int(params["requested_amount"]))

    if tool_name == "product_recommendation":
        c = _resolve_customer(params.get("customer_id"), loan_agent)
        if not c:
            raise ValueError(f"Customer {params.get('customer_id')} not found")
        from app.agents.loans.policy import recommend_product
        shelf = [_to_loan_product_option(p) for p in params["shelf"]]
        return recommend_product(c, int(params["need_amount"]), shelf,
                                  has_gold=bool(params.get("has_gold", False)),
                                  deposit_amount=int(params.get("deposit_amount", 0)),
                                  has_salary_account=bool(params.get("has_salary_account", True)))

    raise ValueError(f"Unknown loan tool: {tool_name}")


def _exec_atm_tool(tool_name: str, params: dict) -> Any:
    """Execute an ATM tool by name with raw params dict."""
    from agents.atm import cash_mechanics, network_ops, customer_money, judgment

    if tool_name == "cash_runout_forecast":
        machine = _to_atm_machine(params)
        inp = cash_mechanics.CashRunoutForecastInput(
            current_time=str(params["current_time"]),
            dispense_rate_per_hour=float(params["dispense_rate_per_hour"]),
            low_cash_threshold=int(params["low_cash_threshold"]),
            cit_time=str(params["cit_time"]),
            cit_hours_from_now=float(params["cit_hours_from_now"]))
        return cash_mechanics.forecast_cash_runout(machine, inp)

    if tool_name == "cassette_status_triage":
        machine = _to_atm_machine(params)
        return cash_mechanics.triage_cassette_status(machine)

    if tool_name == "cash_reconciliation":
        return cash_mechanics.reconcile_cash(
            opening_cash=int(params["opening_cash"]),
            loaded_this_cycle=int(params["loaded_this_cycle"]),
            dispensed_ej=int(params["dispensed_ej"]),
            physical_count=int(params["physical_count"]))

    if tool_name == "denomination_mix_planning":
        inp = cash_mechanics.DenominationMixPlanningInput(
            histogram=[(int(h[0]), int(h[1])) for h in params["histogram"]],
            denominations=[int(d) for d in params["denominations"]],
            cassette_capacity_notes={int(k): int(v) for k, v in params["cassette_capacity_notes"].items()})
        return cash_mechanics.plan_denomination_mix(inp)

    if tool_name == "denomination_dispensability":
        machine = _to_atm_machine(params)
        return cash_mechanics.check_denomination_dispensable(machine, int(params["requested_amount"]))

    if tool_name == "cash_load_planning":
        inp = cash_mechanics.CashLoadPlanningInput(
            avg_daily_dispense=int(params["avg_daily_dispense"]),
            days_until_cit=int(params["days_until_cit"]),
            policy_buffer_pct=float(params["policy_buffer_pct"]),
            cassette_specs=[(int(d), int(c)) for d, c in params["cassette_specs"]],
            withdrawal_mix={int(k): float(v) for k, v in params["withdrawal_mix"].items()})
        return cash_mechanics.plan_cash_load(inp)

    if tool_name == "withdrawal_feasibility":
        profile = _to_atm_customer_profile(params["customer"])
        machine = _to_atm_machine(params["atm"])
        txns = [_to_transaction(t) for t in params["transactions"]]
        return cash_mechanics.check_withdrawal_feasibility(
            profile, machine, int(params["requested_amount"]), txns)

    if tool_name == "daily_limit_remaining":
        profile = _to_atm_customer_profile(params["customer"])
        txns = [_to_transaction(t) for t in params["transactions"]]
        req = int(params["requested_amount"]) if params.get("requested_amount") is not None else None
        return cash_mechanics.remaining_daily_limit(profile, txns, req)

    if tool_name == "alert_triage":
        alerts = [_to_alert_row(a) for a in params["alerts"]]
        return network_ops.triage_alerts(alerts)

    if tool_name == "replenishment_priority":
        candidates = [_to_replenishment_candidate(c) for c in params["candidates"]]
        return network_ops.prioritize_replenishment(candidates, float(params["window_hours"]))

    if tool_name == "cit_route_planning":
        candidates = [_to_cit_candidate(c) for c in params["candidates"]]
        return network_ops.plan_cit_route(candidates, int(params["cap"]))

    if tool_name == "demand_forecast":
        machine = _to_atm_machine(params["atm"])
        week = [(str(d), int(v)) for d, v in params["week_dispensed"]]
        return network_ops.forecast_demand(network_ops.DemandForecastInput(
            machine=machine, week_dispensed=week,
            forecast_day=str(params["forecast_day"]),
            seasonal_label=str(params["seasonal_label"]),
            seasonal_multiplier=float(params["seasonal_multiplier"]),
            machine_capacity=int(params["machine_capacity"])))

    if tool_name == "eod_position_report":
        sites = [_to_eod_site_row(s) for s in params["sites"]]
        threshold = int(params.get("low_cash_threshold", 500_000))
        return network_ops.report_eod_position(sites, threshold)

    if tool_name == "growth_analysis":
        inp = network_ops.GrowthInput(
            atm_id=str(params["atm_id"]), metric_label=str(params["metric_label"]),
            old_period=str(params["old_period"]), old_value=int(params["old_value"]),
            new_period=str(params["new_period"]), new_value=int(params["new_value"]),
            is_money=bool(params.get("is_money", True)))
        return network_ops.analyze_growth(inp)

    if tool_name == "share_analysis":
        rows = [_to_share_row(s) for s in params["sites"]]
        return network_ops.analyze_share(rows, str(params["target_atm_id"]))

    if tool_name == "ranking":
        rows = [_to_ranking_row(s) for s in params["sites"]]
        return network_ops.rank_sites(rows, str(params["metric_label"]),
                                       bool(params.get("higher_is_worse", False)))

    if tool_name == "atm_fee_calculation":
        p = customer_money.AtmFeeParams(
            free_offus_per_month=int(params["free_offus_per_month"]),
            offus_already_made=int(params["offus_already_made"]),
            planned_offus_count=int(params["planned_offus_count"]),
            fee_per_txn=float(params["fee_per_txn"]),
            fed_pct=float(params["fed_pct"]),
            per_txn_limit=int(params["per_txn_limit"]),
            amount_needed=int(params["amount_needed"]))
        return customer_money.calculate_atm_fee(p)

    if tool_name == "atm_recommendation":
        customer = _to_atm_customer_profile(params["customer"])
        candidates = [
            customer_money.NearbyAtm(atm_id=str(c["atm_id"]), bank=str(c["bank"]),
                location=str(c["location"]), distance_km=float(c["distance_km"]),
                status=str(c["status"]), cash_on_hand=int(c["cash_on_hand"]),
                denominations=[int(d) for d in c["denominations"]],
                queue=int(c["queue"]))
            for c in params["candidates"]]
        p = customer_money.AtmRecommendationParams(
            customer=customer, amount_needed=int(params["amount_needed"]),
            off_us_charge=float(params["off_us_charge"]), candidates=candidates)
        return customer_money.recommend_atm(p)

    if tool_name == "cash_advance_assessment":
        customer = _to_atm_customer_profile(params["customer"])
        machine = _to_atm_machine(params["atm"])
        p = customer_money.CashAdvanceParams(
            customer=customer, atm=machine,
            requested_amount=int(params["requested_amount"]),
            fee_pct=float(params["fee_pct"]), fee_min=int(params["fee_min"]),
            monthly_markup_pct=float(params["monthly_markup_pct"]),
            days_held=int(params["days_held"]))
        return customer_money.assess_cash_advance(p)

    if tool_name == "cash_affordability_check":
        customer = _to_atm_customer_profile(params["customer"])
        bills = [_to_bill(b) for b in params["bills"]]
        salary_date = str(params["salary_date"]) if params.get("salary_date") else None
        p = customer_money.CashAffordabilityParams(
            customer=customer, withdrawal_amount=int(params["withdrawal_amount"]),
            bills=bills, salary_date=salary_date)
        return customer_money.check_cash_affordability(p)

    if tool_name == "monthly_atm_cost_summary":
        a = customer_money.MonthlyAtmActivity(
            on_us_count=int(params["on_us_count"]),
            off_us_count=int(params["off_us_count"]),
            inquiries_count=int(params["inquiries_count"]),
            avg_withdrawal=int(params["avg_withdrawal"]),
            free_offus_allowance=int(params["free_offus_allowance"]),
            offus_fee=float(params["offus_fee"]),
            inquiry_fee=float(params["inquiry_fee"]))
        return customer_money.summarize_monthly_atm_cost(a)

    if tool_name == "spend_pattern_summary":
        txns = [_to_transaction(t) for t in params["transactions"]]
        return customer_money.summarize_spend_pattern(txns)

    if tool_name == "travel_cash_planning":
        customer = _to_atm_customer_profile(params["customer"])
        p = customer_money.TravelCashParams(
            customer=customer, days=int(params["days"]),
            daily_spend=int(params["daily_spend"]),
            working_atms_at_destination=int(params["working_atms_at_destination"]),
            off_us_charge=float(params["off_us_charge"]))
        return customer_money.plan_travel_cash(p)

    if tool_name == "cash_carrying_cost":
        machine = _to_atm_machine(params["atm"])
        p = customer_money.CashCarryingCostParams(
            atm=machine, idle_cash=int(params["idle_cash"]),
            daily_dispense=int(params["daily_dispense"]),
            annual_rate=float(params["annual_rate"]),
            cit_trip_cost=int(params["cit_trip_cost"]),
            window_days=int(params["window_days"]))
        return customer_money.calculate_cash_carrying_cost(p)

    if tool_name == "surge_capacity_planning":
        machine = _to_atm_machine(params["atm"])
        p = customer_money.SurgeCapacityParams(
            atm=machine, normal_daily_dispense=int(params["normal_daily_dispense"]),
            season_name=str(params["season_name"]),
            demand_multiplier=float(params["demand_multiplier"]),
            closed_days=int(params["closed_days"]),
            max_capacity=int(params["max_capacity"]))
        return customer_money.plan_surge_capacity(p)

    if tool_name == "fault_root_cause":
        machine = _to_atm_machine(params["atm"])
        vendor_table = {str(k): str(v) for k, v in params["vendor_table"].items()}
        inp = judgment.FaultDiagnosisInputs(
            atm=machine, fault_code=str(params["fault_code"]),
            down_minutes=int(params["down_minutes"]),
            failed_transactions=int(params["failed_transactions"]),
            vendor_table=vendor_table)
        return judgment.diagnose_fault(inp)

    if tool_name == "security_anomaly_assessment":
        machine = _to_atm_machine(params["atm"])
        obs = judgment.SecurityObservation(
            atm=machine, event_count=int(params["event_count"]),
            window_start=str(params["window_start"]),
            window_end=str(params["window_end"]),
            total_amount=int(params["total_amount"]),
            unmatched_authorisation=bool(params["unmatched_authorisation"]),
            safe_door_opened_time=str(params["safe_door_opened_time"]) if params.get("safe_door_opened_time") else None,
            cit_visit_scheduled=bool(params["cit_visit_scheduled"]),
            usb_port_cover_open=bool(params["usb_port_cover_open"]))
        return judgment.assess_security_anomaly(obs)

    if tool_name == "card_fraud_assessment":
        customer = _to_atm_customer_profile(params["customer"])
        txns = [_to_transaction(t) for t in params["transactions"]]
        inp = judgment.CardFraudInputs(profile=customer, transactions=txns)
        return judgment.assess_card_fraud(inp)

    if tool_name == "card_retention_guidance":
        customer = _to_atm_customer_profile(params["customer"])
        machine = _to_atm_machine(params["atm"])
        inp = judgment.CardRetentionInputs(
            profile=customer, atm=machine,
            atm_bank_full_name=str(params["atm_bank_full_name"]),
            hours_since_capture=int(params["hours_since_capture"]),
            capture_reason=str(params["capture_reason"]),
            hold_days=int(params["hold_days"]))
        return judgment.guide_card_retention(inp)

    if tool_name == "failed_transaction_dispute":
        machine = _to_atm_machine(params["atm"])
        inp = judgment.DisputeInputs(
            atm=machine, date=str(params["date"]),
            debited=int(params["debited"]), received=int(params["received"]),
            journal_entry=str(params["journal_entry"]),
            reconciliation_excess=int(params["reconciliation_excess"]),
            reversal_window_days=int(params["reversal_window_days"]),
            days_since_transaction=int(params["days_since_transaction"]))
        return judgment.assess_failed_transaction_dispute(inp)

    if tool_name == "interbank_settlement":
        inp = judgment.InterbankSettlementInputs(
            bank_code=str(params["bank_code"]),
            on_us_count=int(params["on_us_count"]),
            acquired_offus_count=int(params["acquired_offus_count"]),
            issued_offus_count=int(params["issued_offus_count"]),
            interchange_rate=float(params["interchange_rate"]))
        return judgment.calculate_interbank_settlement(inp)

    if tool_name == "trend_summary":
        machine = _to_atm_machine(params["atm"])
        log = [_to_hourly_log_entry(e) for e in params["log"]]
        inp = judgment.TrendSummaryInputs(
            atm=machine, opening_balance=int(params["opening_balance"]),
            log=log)
        return judgment.summarize_trend(inp)

    if tool_name == "uptime_sla_check":
        machine = _to_atm_machine(params["atm"])
        incidents = [_to_downtime_incident(i) for i in params["incidents"]]
        inp = judgment.UptimeSlaInputs(
            atm=machine, period_days=int(params["period_days"]),
            sla_pct=float(params["sla_pct"]), incidents=incidents)
        return judgment.check_uptime_sla(inp)

    raise ValueError(f"Unknown ATM tool: {tool_name}")


# ════════════════════════════════════════════════════════════════════════
# Tool call result
# ════════════════════════════════════════════════════════════════════════

@dataclass
class ToolCall:
    """A single tool execution with its result and metadata."""
    tool_name: str
    domain: Domain
    params: dict[str, Any]
    result: Any
    facts_repr: str
    elapsed_ms: float
    error: str | None = None


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses to dicts for JSON serialization."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def _facts_repr(result: Any) -> str:
    """Convert a tool result to a JSON string for feeding to the LLM."""
    return json.dumps(_to_jsonable(result), default=str, ensure_ascii=False)


# ════════════════════════════════════════════════════════════════════════
# Intent classification
# ════════════════════════════════════════════════════════════════════════

class BankingOrchestrator:
    """Multi-step reasoning agent that chains banking tasks based on natural
    language requests.

    The orchestrator:
    1. Parses the user's intent from their natural language request
    2. Selects and chains the appropriate tool(s) from the 51-task registry
    3. Chains outputs where needed (e.g. score -> stress -> decision)
    4. Hands the computed facts to the LLM for narration
    5. Falls back to deterministic template narration if the LLM is down

    The LLM never decides anything -- only the policy engine does.
    The orchestrator's job is routing and chaining.
    """

    def __init__(self, loan_agent: LoanAgent | None = None,
                 atm_agent: AtmAgent | None = None,
                 loan_client=None, atm_client=None):
        self.loan_agent = loan_agent or LoanAgent(client=loan_client)
        self.atm_agent = atm_agent or AtmAgent(client=atm_client)
        self._call_log: list[ToolCall] = []

    @property
    def tools(self) -> list[ToolDef]:
        return ALL_TOOLS

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in ALL_TOOLS]

    def _log_call(self, call: ToolCall):
        self._call_log.append(call)

    # -- intent classification --

    def _classify_intent(self, query: str) -> tuple[str, dict[str, Any]]:
        """Parse a natural language query and extract intent + parameters.

        Returns (intent_key, params_dict). This is a rule-based classifier --
        the fine-tuned model is NOT asked to classify intents.
        """
        q = query.lower().strip()

        # Customer ID extraction
        customer_id = None
        for token in q.split():
            if token.startswith("c") and len(token) > 1 and token[1:].isdigit():
                customer_id = token
                break

        # -- Loan intents --
        if "triage" in q or "whole book" in q or "portfolio" in q or "scan" in q or "all customers" in q:
            return "portfolio_triage", {}

        if ("score" in q or "grade" in q or "relationship") and "customer" in q:
            return "relationship_scoring", {"customer_id": customer_id}

        if "cash stress" in q or "short of cash" in q or "low cash" in q or "genuine stress" in q:
            return "low_cash_detection", {"customer_id": customer_id}

        if "offer" in q and ("approve" in q or "lend" in q or "credit" in q or "loan" in q):
            if "decline" in q or "declined" in q or "reject" in q or "no " in q:
                return "decline_with_alternatives", {"customer_id": customer_id}
            return "proactive_offer_decision", {"customer_id": customer_id}

        if "dbr" in q or "debt burden" in q or "debt-burden" in q:
            return "dbr_calculation", {"customer_id": customer_id}

        if "emi" in q or "instalment" in q or "monthly payment" in q:
            return "emi_calculation", {"customer_id": customer_id}

        if "max" in q and ("affordable" in q or "borrow" in q or "lend" in q or "how much" in q):
            return "max_affordable_loan", {"customer_id": customer_id}

        if "ecib" in q or "bureau" in q or "credit bureau" in q or "eCIB" in q:
            return "ecib_report_reading", {"customer_id": customer_id}

        if "delinquency" in q or "risk grade" in q or "delinquency risk" in q:
            return "delinquency_risk_grading", {"customer_id": customer_id}

        if "restructure" in q or "restructuring" in q or "extend" in q or "income drop" in q:
            return "restructuring_assessment", {"customer_id": customer_id}

        if "topup" in q or "top up" in q or "additional" in q and ("loan" in q or "amount" in q):
            return "topup_eligibility", {"customer_id": customer_id}

        if "compare" in q or "comparison" in q or "which offer" in q or "quotes" in q:
            return "loan_offer_comparison", {"customer_id": customer_id}

        if "settle" in q or "settlement" in q or "pay off early" in q or "early settlement" in q:
            return "early_settlement_analysis", {"customer_id": customer_id}

        if "gold" in q and ("loan" in q or "sizing" in q or "pledge" in q):
            return "gold_loan_sizing", {}

        if "salary advance" in q:
            return "salary_advance_assessment", {"customer_id": customer_id}

        if "recommend" in q and ("product" in q or "which product" in q or "best product" in q):
            return "product_recommendation", {"customer_id": customer_id}

        if "pricing" in q or ("rate" in q and ("band" in q or "markup" in q or "spread" in q)):
            return "risk_based_pricing", {"customer_id": customer_id}

        # -- ATM intents --
        if "run out" in q or "run-out" in q or "cash empty" in q or "hours to empty" in q or "survives to cit" in q:
            return "cash_runout_forecast", {}

        if "cassette" in q and ("triage" in q or "health" in q or "status" in q):
            return "cassette_status_triage", {}

        if "reconcil" in q or "physical count" in q or "variance" in q:
            return "cash_reconciliation", {}

        if "alert" in q and ("triage" in q or "noc" in q or "severity" in q or "queue" in q):
            return "alert_triage", {}

        if "replenish" in q or "replenishment" in q or ("priority" in q and "cash" in q):
            return "replenishment_priority", {}

        if "cit" in q and ("route" in q or "planning" in q or "vehicle" in q or "cash-in-transit" in q):
            return "cit_route_planning", {}

        if "demand forecast" in q or "forecast demand" in q:
            return "demand_forecast", {}

        if "eod" in q or "end of day" in q or "position report" in q:
            return "eod_position_report", {}

        if "fee" in q and ("atm" in q or "withdrawal" in q) and ("calculate" in q or "cost" in q):
            return "atm_fee_calculation", {}

        if "recommend" in q and ("atm" in q or "machine" in q or "which atm" in q):
            return "atm_recommendation", {}

        if "cash advance" in q:
            return "cash_advance_assessment", {}

        if "afford" in q and ("withdraw" in q or "safe" in q or "bill" in q):
            return "cash_affordability_check", {}

        if "monthly cost" in q or "atm cost" in q:
            return "monthly_atm_cost_summary", {}

        if "spend pattern" in q or "spending" in q and ("pattern" in q or "cash" in q):
            return "spend_pattern_summary", {}

        if "travel" in q and ("cash" in q or "planning" in q):
            return "travel_cash_planning", {}

        if "carry" in q and ("cost" in q or "idle" in q):
            return "cash_carrying_cost", {}

        if "surge" in q or ("ramzan" in q or "eid" in q) and ("capacity" in q or "cash" in q):
            return "surge_capacity_planning", {}

        if "fault" in q and ("root cause" in q or "diagnose" in q or "vendor" in q):
            return "fault_root_cause", {}

        if "security" in q and ("anomaly" in q or "jackpot" in q or "skimming" in q or "cash trap" in q):
            return "security_anomaly_assessment", {}

        if "fraud" in q or "unauthorised" in q or "unauthorized" in q or "stolen" in q:
            return "card_fraud_assessment", {}

        if "card" in q and ("retained" in q or "captured" in q or "stuck" in q or "kept" in q or "retracted" in q):
            return "card_retention_guidance", {}

        if "dispute" in q or "debited" in q and ("no cash" in q or "failed" in q or "not received" in q):
            return "failed_transaction_dispute", {}

        if "interbank" in q or "settlement" in q or "interchange" in q:
            return "interbank_settlement", {}

        if "trend" in q and ("hourly" in q or "dispensing" in q or "peak" in q):
            return "trend_summary", {}

        if "sla" in q or "uptime" in q or "availability" in q:
            return "uptime_sla_check", {}

        if "growth" in q or "month on month" in q or "mom" in q or "month-on-month" in q:
            return "growth_analysis", {}

        if "share" in q and ("of total" in q or "market share" in q or "zone" in q):
            return "share_analysis", {}

        if "rank" in q or "ranking" in q:
            return "ranking", {}

        if "limit" in q and ("daily" in q or "remaining" in q):
            return "daily_limit_remaining", {}

        if "feasible" in q or "feasibility" in q or "can withdraw" in q:
            return "withdrawal_feasibility", {}

        if "denomination" in q and "mix" in q:
            return "denomination_mix_planning", {}

        if "denomination" in q and ("dispens" in q or "exact" in q or "can give" in q):
            return "denomination_dispensability", {}

        if "load" in q and ("plan" in q or "how much" in q or "cash load" in q):
            return "cash_load_planning", {}

        # Default: free-form question
        return "freeform", {"query": query}


    # -- tool execution --

    def _execute_tool(self, tool_name: str, params: dict) -> ToolCall:
        """Execute a single tool and record the call."""
        tool_def = TOOLS_BY_NAME.get(tool_name)
        if not tool_def:
            raise ValueError(f"Unknown tool: {tool_name}. Available: {self.tool_names[:10]}...")

        t0 = time.time()
        error = None
        result = None

        try:
            if tool_def.domain == Domain.LOANS:
                result = _exec_loan_tool(tool_name, params, self.loan_agent)
            else:
                result = _exec_atm_tool(tool_name, params)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        elapsed_ms = (time.time() - t0) * 1000
        facts_repr = _facts_repr(result) if result is not None else ""

        call = ToolCall(
            tool_name=tool_name, domain=tool_def.domain,
            params=params, result=result,
            facts_repr=facts_repr, elapsed_ms=elapsed_ms, error=error)
        self._log_call(call)
        return call

    # -- orchestration --

    def _build_prompt(self, tool_calls: list[ToolCall], task_hint: str) -> str:
        """Build the prompt for the LLM from a sequence of tool calls.

        This assembles the facts from each tool call into a coherent
        narrative prompt that the fine-tuned model can explain.
        """
        parts = []

        for i, call in enumerate(tool_calls):
            if call.error:
                parts.append(f"[ERROR] {call.tool_name}: {call.error}")
                continue

            domain_label = "LOAN" if call.domain == Domain.LOANS else "ATM"
            parts.append(f"{domain_label} TASK: {call.tool_name}")
            parts.append("Facts (authoritative -- do not contradict):")
            parts.append(call.facts_repr)
            parts.append("")

        # Add the task hint to guide the model
        parts.append(f"Task: {task_hint}")
        return "\n\n".join(parts)

    def handle(self, query: str, use_llm: bool = True) -> dict:
        """Main entry point: handle a natural language banking request.

        Returns a dict with:
          - narrative: the LLM (or template) narrative
          - narrative_source: "llm" | "template"
          - tool_calls: list of ToolCall objects executed
          - facts: JSON-serialised facts from all tool calls
          - intent: the classified intent
          - error: any error message
        """
        try:
            intent, params = self._classify_intent(query)
        except Exception as exc:
            return {
                "narrative": f"Intent classification error: {exc}",
                "narrative_source": "error",
                "tool_calls": [],
                "facts": {},
                "intent": "error",
                "error": str(exc)}

        tool_calls: list[ToolCall] = []

        # -- Multi-step chains --
        if intent == "proactive_offer_decision":
            cid = params.get("customer_id")
            if cid:
                score_call = self._execute_tool("relationship_scoring", {"customer_id": cid})
                tool_calls.append(score_call)
                if not score_call.error:
                    stress_call = self._execute_tool("low_cash_detection", {"customer_id": cid})
                    tool_calls.append(stress_call)
                decision_call = self._execute_tool("proactive_offer_decision", {"customer_id": cid})
                tool_calls.append(decision_call)
            else:
                triage_call = self._execute_tool("portfolio_triage", {})
                tool_calls.append(triage_call)

        elif intent == "decline_with_alternatives":
            cid = params.get("customer_id")
            if cid:
                score_call = self._execute_tool("relationship_scoring", {"customer_id": cid})
                tool_calls.append(score_call)
                decision_call = self._execute_tool("decline_with_alternatives", {"customer_id": cid})
                tool_calls.append(decision_call)

        elif intent == "portfolio_triage":
            call = self._execute_tool("portfolio_triage", {})
            tool_calls.append(call)

        elif intent == "relationship_scoring":
            cid = params.get("customer_id")
            if cid:
                call = self._execute_tool("relationship_scoring", {"customer_id": cid})
                tool_calls.append(call)

        elif intent == "low_cash_detection":
            cid = params.get("customer_id")
            if cid:
                call = self._execute_tool("low_cash_detection", {"customer_id": cid})
                tool_calls.append(call)

        elif intent == "dbr_calculation":
            cid = params.get("customer_id")
            if cid:
                try:
                    c = self.loan_agent.get(cid)
                    dbr_call = self._execute_tool("dbr_calculation", {
                        "net_income": c.net_monthly_income,
                        "existing_obligations": c.total_obligations,
                        "new_emi": 50000})
                    tool_calls.append(dbr_call)
                except KeyError:
                    tool_calls.append(ToolCall("dbr_calculation", Domain.LOANS, {}, None, "", 0,
                                               "customer not found"))
            else:
                tool_calls.append(ToolCall("dbr_calculation", Domain.LOANS, {}, None, "", 0,
                                           "need net_income, existing_obligations, new_emi"))

        elif intent == "max_affordable_loan":
            cid = params.get("customer_id")
            if cid:
                call = self._execute_tool("max_affordable_loan", {"customer_id": cid})
                tool_calls.append(call)

        elif intent == "emi_calculation":
            call = self._execute_tool("emi_calculation", params)
            tool_calls.append(call)

        elif intent == "ecib_report_reading":
            call = self._execute_tool("ecib_report_reading", params)
            tool_calls.append(call)

        elif intent == "delinquency_risk_grading":
            cid = params.get("customer_id")
            if cid and "loan" in params:
                call = self._execute_tool("delinquency_risk_grading", {
                    "customer_id": cid, "loan": params["loan"],
                    "late_delays": params.get("late_delays", [])})
                tool_calls.append(call)

        elif intent == "restructuring_assessment":
            cid = params.get("customer_id")
            if cid and "loan" in params:
                call = self._execute_tool("restructuring_assessment", {
                    "customer_id": cid, "loan": params["loan"],
                    "new_income": params.get("new_income", 50000),
                    "extension_months": params.get("extension_months", 12)})
                tool_calls.append(call)

        elif intent == "risk_based_pricing":
            cid = params.get("customer_id")
            if cid:
                call = self._execute_tool("risk_based_pricing", {
                    "customer_id": cid, "base_rate": params.get("base_rate", 0.20),
                    "acceptable_markup_pts": params.get("acceptable_markup_pts", 2.5),
                    "thin_markup_pts": params.get("thin_markup_pts", 5.5),
                    "requested_amount": params.get("requested_amount", 500000),
                    "tenor_months": params.get("tenor_months", 24)})
                tool_calls.append(call)

        elif intent == "topup_eligibility":
            cid = params.get("customer_id")
            if cid and "loan" in params:
                call = self._execute_tool("topup_eligibility", {
                    "customer_id": cid, "loan": params["loan"],
                    "late_in_12m": params.get("late_in_12m", 0),
                    "request_amount": params.get("request_amount", 200000)})
                tool_calls.append(call)

        elif intent == "loan_offer_comparison":
            call = self._execute_tool("loan_offer_comparison", params)
            tool_calls.append(call)

        elif intent == "early_settlement_analysis":
            call = self._execute_tool("early_settlement_analysis", params)
            tool_calls.append(call)

        elif intent == "gold_loan_sizing":
            call = self._execute_tool("gold_loan_sizing", params)
            tool_calls.append(call)

        elif intent == "salary_advance_assessment":
            cid = params.get("customer_id")
            if cid:
                call = self._execute_tool("salary_advance_assessment", {
                    "customer_id": cid, "limit_multiple": params.get("limit_multiple", 1.0),
                    "fee_pct": params.get("fee_pct", 0.02),
                    "annual_rate": params.get("annual_rate", 0.27),
                    "tenor_months": params.get("tenor_months", 3),
                    "requested_amount": params.get("requested_amount", 80000)})
                tool_calls.append(call)

        elif intent == "product_recommendation":
            cid = params.get("customer_id")
            if cid and "shelf" in params:
                call = self._execute_tool("product_recommendation", {
                    "customer_id": cid, "need_amount": params.get("need_amount", 500000),
                    "shelf": params["shelf"],
                    "has_gold": params.get("has_gold", False),
                    "deposit_amount": params.get("deposit_amount", 0),
                    "has_salary_account": params.get("has_salary_account", True)})
                tool_calls.append(call)

        # -- ATM tool dispatch --
        elif intent in ["cash_runout_forecast", "cassette_status_triage", "cash_reconciliation",
                        "denomination_mix_planning", "denomination_dispensability",
                        "cash_load_planning", "alert_triage", "replenishment_priority",
                        "cit_route_planning", "demand_forecast", "eod_position_report",
                        "growth_analysis", "share_analysis", "ranking",
                        "atm_fee_calculation", "cash_carrying_cost", "surge_capacity_planning",
                        "fault_root_cause", "interbank_settlement", "trend_summary",
                        "uptime_sla_check", "daily_limit_remaining", "withdrawal_feasibility"]:
            call = self._execute_tool(intent, params)
            tool_calls.append(call)

        elif intent == "atm_recommendation":
            call = self._execute_tool("atm_recommendation", params)
            tool_calls.append(call)

        elif intent == "cash_advance_assessment":
            call = self._execute_tool("cash_advance_assessment", params)
            tool_calls.append(call)

        elif intent == "cash_affordability_check":
            call = self._execute_tool("cash_affordability_check", params)
            tool_calls.append(call)

        elif intent == "monthly_atm_cost_summary":
            call = self._execute_tool("monthly_atm_cost_summary", params)
            tool_calls.append(call)

        elif intent == "spend_pattern_summary":
            call = self._execute_tool("spend_pattern_summary", params)
            tool_calls.append(call)

        elif intent == "travel_cash_planning":
            call = self._execute_tool("travel_cash_planning", params)
            tool_calls.append(call)

        elif intent == "security_anomaly_assessment":
            call = self._execute_tool("security_anomaly_assessment", params)
            tool_calls.append(call)

        elif intent == "card_fraud_assessment":
            call = self._execute_tool("card_fraud_assessment", params)
            tool_calls.append(call)

        elif intent == "card_retention_guidance":
            call = self._execute_tool("card_retention_guidance", params)
            tool_calls.append(call)

        elif intent == "failed_transaction_dispute":
            call = self._execute_tool("failed_transaction_dispute", params)
            tool_calls.append(call)

        # -- Free-form: ask the agent directly --
        elif intent == "freeform":
            query = params.get("query", query)
            cid = None
            for token in query.split():
                if token.startswith("c") and len(token) > 1 and token[1:].isdigit():
                    cid = token
                    break
            if cid:
                narrative = self.loan_agent.ask(query, cid)
                if narrative:
                    tool_calls.append(ToolCall("freeform", Domain.LOANS, {}, narrative, narrative, 0, None))
                else:
                    tool_calls.append(ToolCall("freeform", Domain.LOANS, {}, None, "", 0,
                                               "LLM unreachable"))
            else:
                narrative = self.atm_agent.ask(query)
                if narrative:
                    tool_calls.append(ToolCall("freeform", Domain.ATM, {}, narrative, narrative, 0, None))
                else:
                    tool_calls.append(ToolCall("freeform", Domain.ATM, {}, None, "", 0,
                                               "LLM unreachable"))

        # -- Narration --
        # If there was an error in the first call, return early
        if tool_calls and tool_calls[0].error and not any(not tc.error for tc in tool_calls):
            return {
                "narrative": f"Error: {tool_calls[0].error}",
                "narrative_source": "error",
                "tool_calls": [tc.tool_name for tc in tool_calls],
                "facts": {},
                "intent": intent,
                "error": tool_calls[0].error}

        # Build the prompt from all successful tool calls
        successful = [tc for tc in tool_calls if not tc.error]
        if not successful:
            first = tool_calls[0] if tool_calls else None
            return {
                "narrative": f"No tools executed. Intent: {intent}. Params: {params}",
                "narrative_source": "error",
                "tool_calls": [tc.tool_name for tc in tool_calls],
                "facts": {},
                "intent": intent,
                "error": "No successful tool calls"}

        prompt = self._build_prompt(successful, intent)
        task_hints = [tc.tool_name for tc in successful]

        # Call the LLM for narration
        narrative, source = None, "template"
        if use_llm:
            try:
                from agents.atm import llm as atm_llm
                raw = atm_llm.chat(
                    prompt, system="",
                    fewshot_tasks=task_hints[:3],
                )
                if raw:
                    # Strip thinking tags
                    _THINK_RE = re.compile(r"\x3cthink.*?\x3c/think\x3e\s*", re.DOTALL)
                    stripped = _THINK_RE.sub("", raw)
                    if "\x3cthink" in stripped:
                        stripped = stripped.split("\x3cthink", 1)[0]
                    stripped = stripped.strip()
                    if stripped:
                        narrative, source = stripped, "llm"
            except Exception:
                pass  # LLM unavailable, fall through to template

        if narrative is None:
            # Deterministic fallback: use the facts block itself
            facts_block = prompt.rsplit("\n\n", 1)[-1]
            lines = facts_block.split("\n")
            if lines and "authoritative" in lines[0].lower():
                lines = lines[1:]
            narrative = "\n".join(lines).strip()
            source = "template"

        return {
            "narrative": narrative,
            "narrative_source": source,
            "tool_calls": [tc.tool_name for tc in tool_calls],
            "facts": {tc.tool_name: _to_jsonable(tc.result) for tc in successful if tc.result},
            "intent": intent,
            "error": None,
        }

    def handle_batch(self, queries: list[str], use_llm: bool = True) -> list[dict]:
        """Process multiple queries sequentially."""
        return [self.handle(q, use_llm=use_llm) for q in queries]

    def call_log(self) -> list[ToolCall]:
        """Return all tool calls made during this session."""
        return list(self._call_log)
