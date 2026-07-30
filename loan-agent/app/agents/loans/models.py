"""Data models for the loans agent.

Field names and the rendered profile format mirror the fine-tuning dataset
(``data/loans/loans_pakistan_finetune_10k.jsonl``) exactly, so a model
fine-tuned on that data can be dropped in without any prompt changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Obligation:
    label: str
    monthly_amount: int


@dataclass
class CustomerProfile:
    customer_id: str
    name: str
    age: int
    bank: str            # e.g. "Sindh Bank"
    bank_code: str       # e.g. "SNDB"
    city: str
    employment: str      # e.g. "Careem captain (gig)"
    net_monthly_income: int
    account_age_years: float
    salary_months_12: int        # salary/inflow months credited in last 12
    avg_balance_6m: int
    previous_loan: str           # "clean" | "late_1_2" | "none"
    cheque_bounce_12m: bool
    ecib_dpd90_24m: bool         # 90+ DPD on eCIB in last 24 months
    ecib_writeoff: bool          # write-off / litigation flag, ever
    obligations: list[Obligation] = field(default_factory=list)
    eom_balances: list[int] = field(default_factory=list)  # 6 months, oldest first
    days_below_5k_30d: int = 0
    salary_to_low_gap_days: int | None = None  # days before month-end money runs out

    @property
    def total_obligations(self) -> int:
        return sum(o.monthly_amount for o in self.obligations)


@dataclass
class LoanProduct:
    name: str
    min_amount: int
    max_amount: int
    tenors: list[int]            # months
    annual_rate: float           # e.g. 0.283
    rate_type: str               # "reducing" | "flat"
    processing_fee_pct: float    # e.g. 0.02
    min_income: int


@dataclass
class ScoreComponent:
    label: str
    points: int


@dataclass
class ScoreBreakdown:
    components: list[ScoreComponent]
    score: int
    band: str  # STRONG | ACCEPTABLE | THIN | POOR


@dataclass
class StressSignals:
    stressed: bool
    days_below_5k: int
    balance_trend_pct: float | None  # % change first→last of 6 EoM balances
    notes: list[str]


@dataclass
class OfferTerms:
    amount: int
    tenor_months: int
    annual_rate: float
    rate_type: str
    emi: int
    dbr_pct: float               # (existing + new EMI) / net income, in %
    processing_fee: int
    headroom_after: int          # income − obligations − EMI


@dataclass
class Decision:
    action: str                  # OFFER | MONITOR | DECLINE
    score: ScoreBreakdown
    stress: StressSignals
    offer: OfferTerms | None
    reasons: list[str]
    alternatives: list[str]      # for declines: what we can responsibly offer instead


# ── eCIB report reading ──

@dataclass
class BureauFacility:
    facility: str        # e.g. "Consumer Durable"
    institution: str      # e.g. "KMBL"
    amount: int
    status: str           # "Regular" | "30 DPD" | "90+ DPD" | "Written off" | "Closed - repaid" | ...


@dataclass
class ECIBAssessment:
    lines: list[BureauFacility]
    active_count: int
    has_writeoff: bool
    has_90dpd: bool
    has_active_delinquency: bool   # any DPD status (30/60/90+) that isn't closed/written-off/regular
    can_lend_unsecured: bool
    verdict: str


# ── running loan (shared by delinquency/restructuring/topup/early-settlement) ──

@dataclass
class RunningLoan:
    original_amount: int
    annual_rate: float
    rate_type: str                # "reducing" | "flat"
    instalments_total: int
    instalments_paid: int
    outstanding: int
    instalment: int

    @property
    def remaining_instalments(self) -> int:
        return self.instalments_total - self.instalments_paid


@dataclass
class DelinquencyGrade:
    score: int
    grade: str            # LOW | MEDIUM | HIGH
    reasons: list[str]
    action: str


@dataclass
class RestructuringResult:
    old_instalment: int
    new_instalment: int
    old_dbr_pct: float
    new_dbr_pct: float
    new_tenor_months: int
    extra_markup_cost: int
    fixes_affordability: bool
    target_instalment: int        # what 40% of the new income actually supports


@dataclass
class TopupDecision:
    approved: bool
    reason: str
    new_principal: int | None = None
    new_instalment: int | None = None
    dbr_pct: float | None = None
    max_approvable_topup: int | None = None


# ── loan offer comparison ──

@dataclass
class LoanQuote:
    label: str
    annual_rate: float
    rate_type: str          # "reducing" | "flat"
    processing_fee_pct: float


@dataclass
class QuoteCost:
    label: str
    instalment: int
    total_cost: int          # EMI * n + processing fee


# ── early settlement ──

@dataclass
class EarlySettlementResult:
    cost_to_settle: int
    cost_to_continue: int
    savings: int
    worth_it: bool


# ── gold loan sizing ──

@dataclass
class GoldOffer:
    weight_tola: float
    purity_k: int              # e.g. 21, 24
    rate_per_tola_24k: int
    ltv_pct: float              # e.g. 0.70


@dataclass
class GoldLoanResult:
    assessed_value: int
    max_loan: int
    granted_amount: int
    instalment: int
    shortfall: int              # requested - granted, 0 if fully covered


# ── salary advance ──

@dataclass
class SalaryAdvanceProduct:
    limit_multiple: float       # e.g. 1.0x or 2.0x net monthly salary
    fee_pct: float
    annual_rate: float           # flat
    tenor_months: int
    min_salary_months: int = 10  # of the last 12


@dataclass
class SalaryAdvanceResult:
    approved: bool
    reason: str
    granted_amount: int | None = None
    instalment: int | None = None
    fee: int | None = None
    total_cost: int | None = None


# ── product recommendation ──

@dataclass
class LoanProductOption:
    name: str
    min_amount: int
    max_amount: int
    tenors: list[int]
    annual_rate: float
    rate_type: str
    processing_fee_pct: float
    min_income: int = 0
    secured: bool = False
    requires_salary_account: bool = False
    requires_gold: bool = False
    requires_deposit: bool = False


@dataclass
class ProductFit:
    name: str
    tenor_months: int
    instalment: int
    dbr_pct: float
    total_cost: int


@dataclass
class ProductRecommendation:
    eliminated: list[tuple[str, str]]   # (product name, reason)
    survivors: list[ProductFit]         # ranked cheapest-first
    best: ProductFit | None


# ── risk-based pricing ──

@dataclass
class RiskPricingResult:
    score: ScoreBreakdown
    annual_rate: float
    markup_pts: float
    instalment: int
