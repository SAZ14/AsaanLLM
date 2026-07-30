"""The ATM agent: local-LLM narration on top of the deterministic policy
functions in cash_mechanics.py / network_ops.py / customer_money.py /
judgment.py. Same split as the loan agent: the policy functions compute
every number and verdict; the LLM (the same fine-tuned model, see
llm.py) only narrates. A hallucinating model can't override a computed
verdict because it's never asked to produce one.
"""
from __future__ import annotations

import re

from . import llm as atm_llm
from .prompts import SYSTEM_PROMPT

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

# task_name -> (module, compute_fn_name, render_fn_name) — for introspection
# (listing supported tasks in the API/CLI) rather than dynamic dispatch, since
# each task's inputs are shaped too differently for one generic call surface.
TASK_REGISTRY: dict[str, tuple[str, str, str]] = {
    # cash_mechanics.py
    "cash_runout_forecast": ("cash_mechanics", "forecast_cash_runout", "render_cash_runout_facts"),
    "cassette_status_triage": ("cash_mechanics", "triage_cassette_status", "render_cassette_triage_facts"),
    "cash_reconciliation": ("cash_mechanics", "reconcile_cash", "render_cash_reconciliation_facts"),
    "denomination_mix_planning": ("cash_mechanics", "plan_denomination_mix", "render_denomination_mix_facts"),
    "denomination_dispensability": ("cash_mechanics", "check_denomination_dispensable", "render_denomination_dispensability_facts"),
    "cash_load_planning": ("cash_mechanics", "plan_cash_load", "render_cash_load_planning_facts"),
    "withdrawal_feasibility": ("cash_mechanics", "check_withdrawal_feasibility", "render_withdrawal_feasibility_facts"),
    "daily_limit_remaining": ("cash_mechanics", "remaining_daily_limit", "render_daily_limit_facts"),
    # network_ops.py
    "alert_triage": ("network_ops", "triage_alerts", "render_alert_triage_facts"),
    "replenishment_priority": ("network_ops", "prioritize_replenishment", "render_replenishment_facts"),
    "cit_route_planning": ("network_ops", "plan_cit_route", "render_cit_route_facts"),
    "demand_forecast": ("network_ops", "forecast_demand", "render_demand_forecast_facts"),
    "eod_position_report": ("network_ops", "report_eod_position", "render_eod_position_facts"),
    "growth_analysis": ("network_ops", "analyze_growth", "render_growth_facts"),
    "share_analysis": ("network_ops", "analyze_share", "render_share_facts"),
    "ranking": ("network_ops", "rank_sites", "render_ranking_facts"),
    # customer_money.py
    "atm_fee_calculation": ("customer_money", "calculate_atm_fee", "render_atm_fee_facts"),
    "atm_recommendation": ("customer_money", "recommend_atm", "render_atm_recommendation_facts"),
    "cash_advance_assessment": ("customer_money", "assess_cash_advance", "render_cash_advance_facts"),
    "cash_affordability_check": ("customer_money", "check_cash_affordability", "render_cash_affordability_facts"),
    "monthly_atm_cost_summary": ("customer_money", "summarize_monthly_atm_cost", "render_monthly_atm_cost_facts"),
    "spend_pattern_summary": ("customer_money", "summarize_spend_pattern", "render_spend_pattern_facts"),
    "travel_cash_planning": ("customer_money", "plan_travel_cash", "render_travel_cash_facts"),
    "cash_carrying_cost": ("customer_money", "calculate_cash_carrying_cost", "render_cash_carrying_cost_facts"),
    "surge_capacity_planning": ("customer_money", "plan_surge_capacity", "render_surge_capacity_facts"),
    # judgment.py
    "fault_root_cause": ("judgment", "diagnose_fault", "render_fault_diagnosis_facts"),
    "security_anomaly_assessment": ("judgment", "assess_security_anomaly", "render_security_assessment_facts"),
    "card_fraud_assessment": ("judgment", "assess_card_fraud", "render_card_fraud_facts"),
    "card_retention_guidance": ("judgment", "guide_card_retention", "render_card_retention_facts"),
    "failed_transaction_dispute": ("judgment", "assess_failed_transaction_dispute", "render_dispute_facts"),
    "interbank_settlement": ("judgment", "calculate_interbank_settlement", "render_interbank_settlement_facts"),
    "trend_summary": ("judgment", "summarize_trend", "render_trend_summary_facts"),
    "uptime_sla_check": ("judgment", "check_uptime_sla", "render_uptime_sla_facts"),
}


def _strip_think(text: str) -> str:
    """The fine-tuned model emits <think> scratchpads; hide them from output.
    An unclosed tag (truncated generation) drops everything from the tag on."""
    text = _THINK_RE.sub("", text)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


class AtmAgent:
    """Thin narration wrapper. Callers compute facts with the pure functions
    in cash_mechanics/network_ops/customer_money/judgment, render them with
    the matching render_*_facts function, then call narrate() to get a
    human-facing explanation grounded in those facts."""

    def __init__(self, client=None):
        self.client = client

    def narrate(self, prompt: str, task_hint: str, use_llm: bool = True) -> tuple[str, str]:
        """Returns (narrative, source) where source is "llm" or "template".
        On any LLM failure (or use_llm=False), falls back to returning the
        facts block itself — still correct, just not prose."""
        if use_llm:
            raw = atm_llm.chat(
                prompt, system=SYSTEM_PROMPT, fewshot_tasks=[task_hint], client=self.client,
            )
            stripped = _strip_think(raw) if raw else ""
            if stripped:
                return stripped, "llm"
        # deterministic fallback: the facts block is already human-readable —
        # just drop the ALL-CAPS "(authoritative — do not contradict)" header
        # line, which is an instruction for the LLM, not narration for a user.
        facts_block = prompt.rsplit("\n\n", 1)[-1]
        lines = facts_block.split("\n")
        if lines and "authoritative" in lines[0].lower():
            lines = lines[1:]
        return "\n".join(lines).strip(), "template"

    def ask(self, question: str, context: str = "") -> str | None:
        """Free-form ATM-ops question against the local model, optionally
        grounded in extra context text. Returns None if no model is up."""
        content = f"{context}\n\n{question}" if context else question
        raw = atm_llm.chat(content, system=SYSTEM_PROMPT, client=self.client)
        return _strip_think(raw) if raw else None
