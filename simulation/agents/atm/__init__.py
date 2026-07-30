"""ATM agent: instant-loan-agent's sibling for ATM cash-ops and customer
support. Deterministic policy (cash_mechanics / network_ops / customer_money
/ judgment) + local-LLM narration (agent.py, llm.py), covering all 33 ATM
task types trained into merged_finetune_train.jsonl.
"""
from .agent import AtmAgent, TASK_REGISTRY
from .entities import AtmMachine, Cassette, CustomerAtmProfile, Transaction

from .cash_mechanics import (
    CashRunoutForecastInput, CashRunoutForecastResult, forecast_cash_runout, render_cash_runout_facts,
    CassetteTriageResult, triage_cassette_status, render_cassette_triage_facts,
    CashReconciliationResult, reconcile_cash, render_cash_reconciliation_facts,
    DenominationMixPlanningInput, DenominationMixPlanningResult, plan_denomination_mix, render_denomination_mix_facts,
    DenominationDispensabilityResult, check_denomination_dispensable, render_denomination_dispensability_facts,
    CashLoadPlanningInput, CashLoadPlanningResult, plan_cash_load, render_cash_load_planning_facts,
    WithdrawalFeasibilityResult, check_withdrawal_feasibility, render_withdrawal_feasibility_facts,
    DailyLimitRemainingResult, remaining_daily_limit, render_daily_limit_facts,
)
from .network_ops import (
    AlertRow, TriagedAlert, AlertTriageResult, triage_alerts, render_alert_triage_facts,
    ReplenishmentCandidate, RankedReplenishment, ReplenishmentPriorityResult, prioritize_replenishment, render_replenishment_facts,
    CitCandidate, CitRoutePlanResult, plan_cit_route, render_cit_route_facts,
    DemandForecastInput, DemandForecastResult, forecast_demand, render_demand_forecast_facts,
    EodSiteRow, EodActionItem, EodPositionResult, report_eod_position, render_eod_position_facts,
    GrowthInput, GrowthResult, analyze_growth, render_growth_facts,
    ShareRow, ShareResult, analyze_share, render_share_facts,
    RankingRow, RankedSite, RankingResult, rank_sites, render_ranking_facts,
)
from .customer_money import (
    AtmFeeParams, AtmFeeResult, calculate_atm_fee, render_atm_fee_facts,
    NearbyAtm, AtmRecommendationParams, AtmRecommendationResult, recommend_atm, render_atm_recommendation_facts,
    CashAdvanceParams, CashAdvanceResult, assess_cash_advance, render_cash_advance_facts,
    Bill, CashAffordabilityParams, CashAffordabilityResult, check_cash_affordability, render_cash_affordability_facts,
    MonthlyAtmActivity, MonthlyAtmCostResult, summarize_monthly_atm_cost, render_monthly_atm_cost_facts,
    SpendPatternResult, summarize_spend_pattern, render_spend_pattern_facts,
    TravelCashParams, TravelCashResult, plan_travel_cash, render_travel_cash_facts,
    CashCarryingCostParams, CashCarryingCostResult, calculate_cash_carrying_cost, render_cash_carrying_cost_facts,
    SurgeCapacityParams, SurgeCapacityResult, plan_surge_capacity, render_surge_capacity_facts,
)
from .judgment import (
    FaultDiagnosisInputs, FaultRootCauseResult, diagnose_fault, render_fault_diagnosis_facts,
    SecurityObservation, SecurityAnomalyAssessmentResult, assess_security_anomaly, render_security_assessment_facts,
    CardFraudInputs, CardFraudAssessmentResult, assess_card_fraud, render_card_fraud_facts,
    CardRetentionInputs, CardRetentionGuidanceResult, guide_card_retention, render_card_retention_facts,
    DisputeInputs, FailedTransactionDisputeResult, assess_failed_transaction_dispute, render_dispute_facts,
    InterbankSettlementInputs, InterbankSettlementResult, calculate_interbank_settlement, render_interbank_settlement_facts,
    HourlyLogEntry, TrendSummaryInputs, TrendSummaryResult, summarize_trend, render_trend_summary_facts,
    DowntimeIncident, UptimeSlaInputs, UptimeSlaCheckResult, check_uptime_sla, render_uptime_sla_facts,
)

__all__ = [
    "AtmAgent", "TASK_REGISTRY",
    "AtmMachine", "Cassette", "CustomerAtmProfile", "Transaction",
]
