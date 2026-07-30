"""Deterministic lending policy engine.

Implements the bank-internal relationship scoring policy, cash-stress
detection, EMI / DBR / affordability math, and the decision gate — all in
code. The LLM never decides; it only explains. Every formula matches the
worked examples in the fine-tuning dataset:

- Relationship score: +2/+1/0 tiers on account age, salary regularity,
  average balance, repayment history; −3 bounce, −4 eCIB 90+ DPD,
  −6 write-off/litigation. Bands: >=6 STRONG | 3-5 ACCEPTABLE |
  0-2 THIN | <0 POOR.
- Genuine cash stress marker: 10+ days under Rs 5,000 in the last 30.
- Proactive offers only to STRONG/ACCEPTABLE showing genuine stress;
  THIN monitored; POOR gets secured alternatives only. DBR cap 40%.
"""
from __future__ import annotations

from statistics import mean

from app.agents.loans.models import (
    BureauFacility,
    CustomerProfile,
    Decision,
    DelinquencyGrade,
    ECIBAssessment,
    EarlySettlementResult,
    GoldLoanResult,
    GoldOffer,
    LoanProduct,
    LoanProductOption,
    LoanQuote,
    OfferTerms,
    ProductFit,
    ProductRecommendation,
    QuoteCost,
    RestructuringResult,
    RiskPricingResult,
    RunningLoan,
    SalaryAdvanceProduct,
    SalaryAdvanceResult,
    ScoreBreakdown,
    ScoreComponent,
    StressSignals,
    TopupDecision,
)

DBR_CAP = 0.40
STRESS_DAYS_THRESHOLD = 10   # days under Rs 5,000 in last 30 = genuine stress
ROUND_DOWN_STEP = 10_000     # lendable amounts floor to nearest Rs 10,000

# The demo product shelf. Rates are typical Pakistani unsecured pricing
# (KIBOR + spread); the dataset uses the same shape.
PERSONAL_INSTALMENT_LOAN = LoanProduct(
    name="Personal Instalment Loan",
    min_amount=50_000,
    max_amount=3_000_000,
    tenors=[12, 24, 36, 48],
    annual_rate=0.283,
    rate_type="reducing",
    processing_fee_pct=0.02,
    min_income=50_000,
)

MICROFINANCE_ENTERPRISE_LOAN = LoanProduct(
    name="Microfinance Enterprise Loan",
    min_amount=20_000,
    max_amount=350_000,
    tenors=[6, 12, 18],
    annual_rate=0.383,
    rate_type="flat",
    processing_fee_pct=0.01,
    min_income=15_000,
)


# ── relationship scoring ──

def relationship_score(p: CustomerProfile) -> ScoreBreakdown:
    comps: list[ScoreComponent] = []

    if p.account_age_years >= 3:
        comps.append(ScoreComponent("account age 3 years or more", 2))
    elif p.account_age_years >= 1:
        comps.append(ScoreComponent("account age 1-3 years", 1))
    else:
        comps.append(ScoreComponent("account age under 1 year", 0))

    if p.salary_months_12 >= 11:
        comps.append(ScoreComponent("salary credited 11+ of last 12 months", 2))
    elif p.salary_months_12 >= 9:
        comps.append(ScoreComponent("salary credited 9-10 of last 12 months", 1))
    else:
        comps.append(ScoreComponent("salary credited under 9 of last 12 months", 0))

    if p.avg_balance_6m > 50_000:
        comps.append(ScoreComponent("average balance above Rs 50,000", 2))
    elif p.avg_balance_6m >= 15_000:
        comps.append(ScoreComponent("average balance Rs 15,000-50,000", 1))
    else:
        comps.append(ScoreComponent("average balance below Rs 15,000", 0))

    if p.previous_loan == "clean":
        comps.append(ScoreComponent("previous loan fully repaid, never 30+ days late", 2))
    elif p.previous_loan == "late_1_2":
        comps.append(ScoreComponent("previous loan repaid with 1-2 late months", 1))
    else:
        comps.append(ScoreComponent("no previous loan history", 0))

    if p.cheque_bounce_12m:
        comps.append(ScoreComponent("cheque/direct-debit bounce in last 12 months", -3))
    if p.ecib_dpd90_24m:
        comps.append(ScoreComponent("90+ day delinquency on eCIB in last 24 months", -4))
    if p.ecib_writeoff:
        comps.append(ScoreComponent("write-off/litigation flag on eCIB", -6))

    score = sum(c.points for c in comps)
    if score >= 6:
        band = "STRONG"
    elif score >= 3:
        band = "ACCEPTABLE"
    elif score >= 0:
        band = "THIN"
    else:
        band = "POOR"
    return ScoreBreakdown(components=comps, score=score, band=band)


# ── cash-stress detection ──

def detect_stress(p: CustomerProfile) -> StressSignals:
    notes: list[str] = []
    trend = None
    if len(p.eom_balances) >= 2 and p.eom_balances[0] > 0:
        trend = (p.eom_balances[-1] - p.eom_balances[0]) / p.eom_balances[0] * 100

    stressed = p.days_below_5k_30d >= STRESS_DAYS_THRESHOLD
    if stressed:
        notes.append(
            f"{p.days_below_5k_30d} of the last 30 days under Rs 5,000 "
            f"(stress marker is {STRESS_DAYS_THRESHOLD}+)"
        )
    else:
        notes.append(
            f"only {p.days_below_5k_30d} of the last 30 days under Rs 5,000 "
            f"(below the {STRESS_DAYS_THRESHOLD}-day stress marker)"
        )
    if trend is not None:
        if trend <= -50:
            notes.append(f"end-of-month balances down {abs(trend):.0f}% over 6 months")
        elif trend >= 0:
            notes.append(f"end-of-month balances up {trend:.0f}% over 6 months")
    if p.salary_to_low_gap_days is not None and stressed:
        notes.append(
            f"money typically runs out {p.salary_to_low_gap_days} days before month-end"
        )
    return StressSignals(
        stressed=stressed,
        days_below_5k=p.days_below_5k_30d,
        balance_trend_pct=trend,
        notes=notes,
    )


# ── instalment math ──

def emi(principal: float, annual_rate: float, months: int, rate_type: str) -> float:
    """Exact (unrounded) monthly instalment."""
    if months <= 0:
        raise ValueError(f"Tenor must be positive, got {months}")
    if rate_type == "flat":
        markup = principal * annual_rate * (months / 12)
        return (principal + markup) / months
    r = annual_rate / 12
    if r == 0:
        return principal / months
    growth = (1 + r) ** months
    return principal * r * growth / (growth - 1)


def invert_emi(instalment: float, annual_rate: float, months: int, rate_type: str) -> float:
    """Principal supportable by a given monthly instalment."""
    if months <= 0:
        raise ValueError(f"Tenor must be positive, got {months}")
    if rate_type == "flat":
        return instalment * months / (1 + annual_rate * months / 12)
    r = annual_rate / 12
    if r == 0:
        return instalment * months
    growth = (1 + r) ** months
    return instalment * (growth - 1) / (r * growth)


def dbr(net_income: int, existing_obligations: int, new_emi: float) -> float:
    """Debt burden ratio in percent: (existing + new instalment) / net income."""
    if net_income <= 0:
        return float("inf")
    return (existing_obligations + new_emi) / net_income * 100


def max_affordable_loan(
    p: CustomerProfile, product: LoanProduct, tenor: int, dbr_cap: float = DBR_CAP
) -> int:
    """Largest lendable principal under the DBR cap, floored to Rs 10,000
    and clamped to the product band. Returns 0 if nothing is lendable."""
    headroom = dbr_cap * p.net_monthly_income - p.total_obligations
    if headroom <= 0:
        return 0
    raw = invert_emi(headroom, product.annual_rate, tenor, product.rate_type)
    floored = int(raw // ROUND_DOWN_STEP) * ROUND_DOWN_STEP
    if floored < product.min_amount:
        return 0
    return min(floored, product.max_amount)


def build_offer(
    p: CustomerProfile, product: LoanProduct, tenor: int | None = None
) -> OfferTerms | None:
    """Size a responsible offer at the DBR cap for the given tenor
    (default: the product's longest tenor, which maximises headroom)."""
    tenor = tenor or product.tenors[-1]
    if tenor not in product.tenors:
        raise ValueError(
            f"Tenor {tenor} months not on the shelf for {product.name} "
            f"(available: {product.tenors})"
        )
    amount = max_affordable_loan(p, product, tenor)
    if amount <= 0:
        return None
    exact = emi(amount, product.annual_rate, tenor, product.rate_type)
    return OfferTerms(
        amount=amount,
        tenor_months=tenor,
        annual_rate=product.annual_rate,
        rate_type=product.rate_type,
        emi=round(exact),
        dbr_pct=round(dbr(p.net_monthly_income, p.total_obligations, exact), 1),
        processing_fee=round(amount * product.processing_fee_pct),
        headroom_after=round(p.net_monthly_income - p.total_obligations - exact),
    )


# ── decision gate ──

def _alternatives(p: CustomerProfile) -> list[str]:
    alts = [
        "Deposit-backed finance against any savings held with us",
        "Gold-backed finance — pledged gold typically supports 70-80% of assessed value",
        "Secured card or small nano facility repaid on time for 6-12 months to put "
        "fresh positive lines on eCIB",
    ]
    if p.ecib_writeoff:
        alts.append(
            "If the write-off is disputed, the eCIB correction process through the "
            "reporting bank — a cleaned record reopens the normal shelf"
        )
    return alts


def decide(
    p: CustomerProfile,
    product: LoanProduct = PERSONAL_INSTALMENT_LOAN,
    tenor: int | None = None,
) -> Decision:
    """The policy gate. OFFER only to STRONG/ACCEPTABLE files showing genuine
    cash stress; THIN monitored; POOR declined for unsecured with secured
    alternatives. Enforced here so no model output can override it."""
    score = relationship_score(p)
    stress = detect_stress(p)
    reasons: list[str] = []

    if score.band == "POOR":
        negatives = [c.label for c in score.components if c.points < 0]
        reasons.append(
            f"Score {score.score} (POOR) driven by: " + "; ".join(negatives) + "."
        )
        if stress.stressed:
            reasons.append(
                "The cash stress is real, which makes unsecured lending more "
                "dangerous, not less — it would likely become another delinquency."
            )
        return Decision(
            action="DECLINE", score=score, stress=stress, offer=None,
            reasons=reasons, alternatives=_alternatives(p),
        )

    if score.band == "THIN":
        reasons.append(
            f"Score {score.score} (THIN) — file too thin for a proactive unsecured "
            "offer; keep under monitoring while the relationship builds."
        )
        return Decision(
            action="MONITOR", score=score, stress=stress, offer=None,
            reasons=reasons, alternatives=[],
        )

    # STRONG or ACCEPTABLE from here.
    if not stress.stressed:
        reasons.append(
            f"Score {score.score} ({score.band}) but liquid — no genuine cash "
            "stress, so a proactive offer would land as marketing, not help. "
            "Keep pre-approved."
        )
        return Decision(
            action="MONITOR", score=score, stress=stress, offer=None,
            reasons=reasons, alternatives=[],
        )

    if p.net_monthly_income < product.min_income:
        reasons.append(
            f"Score {score.score} ({score.band}) with genuine stress, but verified "
            f"income Rs {p.net_monthly_income:,} is below the product minimum "
            f"Rs {product.min_income:,}."
        )
        return Decision(
            action="MONITOR", score=score, stress=stress, offer=None,
            reasons=reasons, alternatives=[],
        )

    offer = build_offer(p, product, tenor)
    if offer is None:
        reasons.append(
            f"Score {score.score} ({score.band}) with genuine stress, but existing "
            f"obligations of Rs {p.total_obligations:,}/month leave no instalment "
            f"headroom under the {DBR_CAP:.0%} DBR cap."
        )
        return Decision(
            action="MONITOR", score=score, stress=stress, offer=None,
            reasons=reasons, alternatives=[],
        )

    reasons.append(
        f"Score {score.score} ({score.band}) with genuine cash stress "
        f"({stress.days_below_5k} days under Rs 5,000) — a proactive offer lands "
        "as help. Sized under the 40% DBR cap; offer below the maximum where the "
        "purpose allows, since a customer at exactly the cap has no buffer for a "
        "bad month."
    )
    return Decision(
        action="OFFER", score=score, stress=stress, offer=offer,
        reasons=reasons, alternatives=[],
    )


def triage(
    customers: list[CustomerProfile],
    product: LoanProduct = PERSONAL_INSTALMENT_LOAN,
) -> dict[str, list[tuple[CustomerProfile, Decision]]]:
    """Portfolio triage: OFFER queue prioritised strongest-file-first
    (score desc, then depth of stress), plus MONITOR and DECLINE lists."""
    queues: dict[str, list[tuple[CustomerProfile, Decision]]] = {
        "OFFER": [], "MONITOR": [], "DECLINE": [],
    }
    for c in customers:
        d = decide(c, product)
        queues[d.action].append((c, d))
    queues["OFFER"].sort(key=lambda cd: (-cd[1].score.score, -cd[1].stress.days_below_5k))
    queues["MONITOR"].sort(key=lambda cd: -cd[1].score.score)
    queues["DECLINE"].sort(key=lambda cd: cd[1].score.score)
    return queues


# ── eCIB report reading ──

_CLOSED_STATUSES = {"Written off", "Closed - repaid"}


def read_ecib_report(lines: list[BureauFacility]) -> ECIBAssessment:
    """A write-off is the hardest flag (unsecured lending off the table until
    settled and reported as such); active delinquency (any DPD status) blocks
    fresh unsecured lending until the line runs current again."""
    active = [f for f in lines if f.status not in _CLOSED_STATUSES]
    has_writeoff = any(f.status == "Written off" for f in lines)
    has_90dpd = any(f.status == "90+ DPD" for f in lines)
    has_active_delinquency = any("DPD" in f.status for f in active)

    if has_writeoff:
        verdict = "writeoff_block"
    elif has_active_delinquency:
        verdict = "active_delinquency"
    else:
        verdict = "clean"

    return ECIBAssessment(
        lines=lines,
        active_count=len(active),
        has_writeoff=has_writeoff,
        has_90dpd=has_90dpd,
        has_active_delinquency=has_active_delinquency,
        can_lend_unsecured=not (has_writeoff or has_active_delinquency),
        verdict=verdict,
    )


# ── delinquency risk grading (early-warning, on a running facility) ──

def grade_delinquency_risk(
    p: CustomerProfile, loan: RunningLoan, late_delays: list[int]
) -> DelinquencyGrade:
    """RISK POLICY: +2 two or more late in last four; +2 delays lengthening;
    +2 ten or more days under Rs 5,000 last month; +1 instalment over 30% of
    income; +1 any bounce in 12 months. HIGH >= 5, MEDIUM 3-4, LOW 0-2."""
    score = 0
    reasons: list[str] = []

    late_count = sum(1 for d in late_delays if d > 0)
    if late_count >= 2:
        score += 2
        reasons.append(f"{late_count} of last {len(late_delays)} instalments paid late")

    if len(late_delays) >= 4:
        lengthening = mean(late_delays[2:]) > mean(late_delays[:2])
        if lengthening:
            score += 2
            reasons.append(
                "delays are lengthening (" + "->".join(str(d) for d in late_delays) + ")"
            )

    if p.days_below_5k_30d >= STRESS_DAYS_THRESHOLD:
        score += 2
        reasons.append(f"{p.days_below_5k_30d} days under Rs 5,000 last month")

    instalment_pct = loan.instalment / p.net_monthly_income * 100 if p.net_monthly_income else 0
    if instalment_pct > 30:
        score += 1
        reasons.append(f"instalment is {instalment_pct:.0f}% of income")

    if p.cheque_bounce_12m:
        score += 1
        reasons.append("a direct-debit bounce in the last 12 months")

    if score >= 5:
        grade = "HIGH"
        action = (
            "Call before the next due date, offer a restructuring conversation now - "
            "pre-delinquency restructuring is cheap; post-default recovery is not. "
            "Do not wait for the 30-DPD letter."
        )
    elif score >= 3:
        grade = "MEDIUM"
        action = (
            "Put the account on the watch list, send a soft reminder 3 days before "
            "due date, and check again after the next salary credit lands."
        )
    else:
        grade = "LOW"
        action = "No action needed beyond standard monitoring."

    return DelinquencyGrade(score=score, grade=grade, reasons=reasons, action=action)


# ── restructuring assessment ──

def assess_restructuring(
    p: CustomerProfile,
    loan: RunningLoan,
    new_income: int,
    extension_months: int,
    dbr_cap: float = DBR_CAP,
) -> RestructuringResult:
    new_tenor = loan.remaining_instalments + extension_months
    new_instalment = round(emi(loan.outstanding, loan.annual_rate, new_tenor, loan.rate_type))

    old_dbr = (p.total_obligations + loan.instalment) / p.net_monthly_income * 100
    new_dbr = (p.total_obligations + new_instalment) / new_income * 100 if new_income else float("inf")

    extra_markup_cost = round(
        new_instalment * new_tenor - loan.instalment * loan.remaining_instalments
    )
    target_instalment = max(0, round(dbr_cap * new_income - p.total_obligations))

    return RestructuringResult(
        old_instalment=loan.instalment,
        new_instalment=new_instalment,
        old_dbr_pct=round(old_dbr, 1),
        new_dbr_pct=round(new_dbr, 1),
        new_tenor_months=new_tenor,
        extra_markup_cost=extra_markup_cost,
        fixes_affordability=new_dbr <= dbr_cap * 100,
        target_instalment=target_instalment,
    )


# ── top-up eligibility ──

def assess_topup(
    p: CustomerProfile,
    loan: RunningLoan,
    late_in_12m: int,
    request_amount: int,
    min_instalments: int = 12,
    dbr_cap: float = DBR_CAP,
) -> TopupDecision:
    if loan.instalments_paid < min_instalments:
        return TopupDecision(
            approved=False,
            reason=(
                f"only {loan.instalments_paid} instalment(s) paid against a minimum "
                f"of {min_instalments}"
            ),
        )
    if late_in_12m > 0:
        return TopupDecision(
            approved=False,
            reason=(
                f"{late_in_12m} late instalment(s) in the last 12 months against a "
                "zero-late requirement"
            ),
        )

    new_principal = loan.outstanding + request_amount
    new_instalment = round(emi(new_principal, loan.annual_rate, loan.remaining_instalments, loan.rate_type))
    dbr_pct = round(dbr(p.net_monthly_income, p.total_obligations, new_instalment), 1)

    if dbr_pct <= dbr_cap * 100:
        return TopupDecision(
            approved=True, reason="passes policy gate and DBR cap",
            new_principal=new_principal, new_instalment=new_instalment, dbr_pct=dbr_pct,
        )

    headroom = dbr_cap * p.net_monthly_income - p.total_obligations
    max_new_principal = max(0.0, invert_emi(headroom, loan.annual_rate, loan.remaining_instalments, loan.rate_type))
    max_topup = max(0, int(max_new_principal // ROUND_DOWN_STEP) * ROUND_DOWN_STEP - loan.outstanding)

    return TopupDecision(
        approved=False,
        reason=f"merged instalment of Rs {new_instalment:,} pushes DBR to {dbr_pct}%, over the {dbr_cap:.0%} cap",
        new_principal=new_principal, new_instalment=new_instalment, dbr_pct=dbr_pct,
        max_approvable_topup=max(0, max_topup),
    )


# ── loan offer comparison ──

def compare_loan_offers(principal: int, tenor: int, quotes: list[LoanQuote]) -> list[QuoteCost]:
    costs = []
    for q in quotes:
        inst = emi(principal, q.annual_rate, tenor, q.rate_type)
        fee = round(principal * q.processing_fee_pct)
        total = round(inst * tenor) + fee
        costs.append(QuoteCost(label=q.label, instalment=round(inst), total_cost=total))
    costs.sort(key=lambda c: c.total_cost)
    return costs


# ── early settlement analysis ──

def analyze_early_settlement(loan: RunningLoan, settlement_fee_pct: float) -> EarlySettlementResult:
    cost_to_continue = loan.instalment * loan.remaining_instalments
    cost_to_settle = round(loan.outstanding * (1 + settlement_fee_pct))
    savings = cost_to_continue - cost_to_settle
    return EarlySettlementResult(
        cost_to_settle=cost_to_settle, cost_to_continue=cost_to_continue,
        savings=savings, worth_it=savings > 0,
    )


# ── gold loan sizing ──

def size_gold_loan(
    gold: GoldOffer, requested_amount: int, annual_rate: float, tenor_months: int,
) -> GoldLoanResult:
    assessed_value = round(gold.weight_tola * gold.rate_per_tola_24k * gold.purity_k / 24)
    max_loan = round(assessed_value * gold.ltv_pct)
    granted = min(requested_amount, max_loan)
    instalment = round(emi(granted, annual_rate, tenor_months, "reducing"))
    return GoldLoanResult(
        assessed_value=assessed_value, max_loan=max_loan, granted_amount=granted,
        instalment=instalment, shortfall=max(0, requested_amount - granted),
    )


# ── salary advance assessment ──

def assess_salary_advance(
    p: CustomerProfile, product: SalaryAdvanceProduct, requested_amount: int,
) -> SalaryAdvanceResult:
    if p.salary_months_12 < product.min_salary_months:
        return SalaryAdvanceResult(
            approved=False,
            reason=(
                f"salary landed in only {p.salary_months_12} of the last 12 months "
                f"against the {product.min_salary_months}-month requirement"
            ),
        )
    limit = p.net_monthly_income * product.limit_multiple
    granted = min(requested_amount, round(limit))
    fee = round(granted * product.fee_pct)
    markup_total = granted * product.annual_rate * (product.tenor_months / 12)
    instalment = round((granted + markup_total) / product.tenor_months)
    total_cost = round(markup_total) + fee
    return SalaryAdvanceResult(
        approved=True, reason="salary record supports it",
        granted_amount=granted, instalment=instalment, fee=fee, total_cost=total_cost,
    )


# ── product recommendation ──

def recommend_product(
    p: CustomerProfile,
    need_amount: int,
    shelf: list[LoanProductOption],
    has_gold: bool = False,
    deposit_amount: int = 0,
    has_salary_account: bool = True,
    dbr_cap: float = DBR_CAP,
) -> ProductRecommendation:
    eliminated: list[tuple[str, str]] = []
    survivors: list[ProductFit] = []

    for product in shelf:
        reasons: list[str] = []
        if need_amount > product.max_amount:
            reasons.append("amount above product maximum")
        elif need_amount < product.min_amount:
            if p.net_monthly_income < product.min_income:
                reasons.append(f"amount below product minimum, income under Rs {product.min_income:,}")
            else:
                reasons.append("amount below product minimum")
        elif p.net_monthly_income < product.min_income:
            reasons.append(f"income below minimum Rs {product.min_income:,}")

        if product.requires_gold and not has_gold:
            reasons.append("no gold offered")
        if product.requires_deposit:
            if deposit_amount <= 0:
                reasons.append("no deposit to lien")
            elif deposit_amount < need_amount:
                reasons.append("deposit smaller than need")
        if product.requires_salary_account and not has_salary_account:
            reasons.append("no salary account")

        if reasons:
            eliminated.append((product.name, "; ".join(reasons)))
            continue

        fit = None
        max_dbr_seen = 0.0
        for tenor in sorted(product.tenors):
            inst = emi(need_amount, product.annual_rate, tenor, product.rate_type)
            dbr_pct = (p.total_obligations + inst) / p.net_monthly_income * 100
            max_dbr_seen = max(max_dbr_seen, dbr_pct) if tenor == max(product.tenors) else max_dbr_seen
            if dbr_pct <= dbr_cap * 100:
                fee = round(need_amount * product.processing_fee_pct)
                total_cost = round(inst * tenor) + fee - need_amount
                fit = ProductFit(
                    name=product.name, tenor_months=tenor, instalment=round(inst),
                    dbr_pct=round(dbr_pct, 1), total_cost=total_cost,
                )
                break

        if fit is None:
            longest = max(product.tenors)
            inst = emi(need_amount, product.annual_rate, longest, product.rate_type)
            dbr_at_max = (p.total_obligations + inst) / p.net_monthly_income * 100
            eliminated.append((product.name, f"DBR {dbr_at_max:.0f}% even at max tenor"))
        else:
            survivors.append(fit)

    survivors.sort(key=lambda f: f.total_cost)
    return ProductRecommendation(
        eliminated=eliminated, survivors=survivors,
        best=survivors[0] if survivors else None,
    )


# ── risk-based pricing ──

def price_by_risk(
    p: CustomerProfile,
    base_rate: float,
    acceptable_markup_pts: float,
    thin_markup_pts: float,
    requested_amount: int,
    tenor_months: int,
) -> RiskPricingResult:
    score = relationship_score(p)
    markup = {
        "STRONG": 0.0, "ACCEPTABLE": acceptable_markup_pts, "THIN": thin_markup_pts,
    }.get(score.band)
    if markup is None:  # POOR
        rate = base_rate
        instalment = 0
    else:
        rate = round(base_rate + markup, 1)
        instalment = round(emi(requested_amount, rate / 100, tenor_months, "reducing"))
    return RiskPricingResult(score=score, annual_rate=rate, markup_pts=markup or 0.0, instalment=instalment)
