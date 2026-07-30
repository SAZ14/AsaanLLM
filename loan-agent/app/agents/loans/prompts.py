"""Prompt rendering for the loans agent.

Renders profiles, policy blocks, and questions in *exactly* the format of the
fine-tuning dataset, so the base model sees familiar few-shot prompts today
and the fine-tuned model sees its native training distribution tomorrow.
"""
from __future__ import annotations

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
    LoanQuote,
    ProductRecommendation,
    QuoteCost,
    RestructuringResult,
    RiskPricingResult,
    RunningLoan,
    SalaryAdvanceResult,
    TopupDecision,
)

SYSTEM_PROMPT = (
    "You are a senior credit officer at a Pakistani bank running the "
    "responsible instant-lending desk. You read customer relationship data, "
    "eCIB records, and cash-flow signals, then walk through lending decisions "
    "under the bank's internal policy. You never recommend unsecured lending "
    "to POOR-band files; you redirect them to secured alternatives. All "
    "amounts are in Pakistani rupees. Be concise, numerate, and direct — a "
    "declined customer should always leave with a workable path."
)

_PREVIOUS_LOAN_TEXT = {
    "clean": "fully repaid, never 30+ days late",
    "late_1_2": "repaid with 1-2 late months",
    "none": "no history",
}


def _rs(x: int | float) -> str:
    return f"Rs {x:,.0f}"


def render_profile(p: CustomerProfile) -> str:
    lines = [
        "CUSTOMER PROFILE",
        f"- Name: {p.name}, {p.age}",
        f"- Bank: {p.bank} ({p.bank_code}), {p.city}",
        f"- Employment: {p.employment}",
        f"- Verified net monthly income: {_rs(p.net_monthly_income)}",
        f"- Account age: {p.account_age_years} years",
        f"- Salary/inflow months (last 12): {p.salary_months_12}/12",
        f"- Average balance (6-month): {_rs(p.avg_balance_6m)}",
        f"- Previous loan: {_PREVIOUS_LOAN_TEXT.get(p.previous_loan, p.previous_loan)}",
        f"- Cheque/DD bounce in last 12 months: {'yes' if p.cheque_bounce_12m else 'no'}",
        f"- eCIB 90+ DPD in last 24 months: {'yes' if p.ecib_dpd90_24m else 'no'}",
        f"- eCIB write-off/litigation flag: {'yes' if p.ecib_writeoff else 'no'}",
    ]
    if p.obligations:
        lines.append("- Existing monthly obligations:")
        for o in p.obligations:
            lines.append(f"    {o.label}: {_rs(o.monthly_amount)}")
        lines.append(f"- Total existing obligations: {_rs(p.total_obligations)}/month")
    else:
        lines.append("- Existing monthly obligations: none")
    if p.eom_balances:
        lines.append(
            "- End-of-month balances (oldest first): "
            + ", ".join(_rs(b) for b in p.eom_balances)
        )
    lines.append(f"- Days below Rs 5,000 in the last 30 days: {p.days_below_5k_30d}")
    if p.salary_to_low_gap_days is not None:
        lines.append(
            "- Average gap between salary credit and balance falling under "
            f"Rs 5,000: {p.salary_to_low_gap_days} days before month-end"
        )
    return "\n".join(lines)


SCORING_POLICY_BLOCK = """RELATIONSHIP SCORING POLICY (bank internal)
+2  account age 3 years or more (+1 if 1-3 years, 0 if under 1 year)
+2  salary/inflow credited in at least 11 of the last 12 months (+1 if 9-10, 0 otherwise)
+2  average balance above Rs 50,000 (+1 if Rs 15,000-50,000, 0 below)
+2  previous loan fully repaid, never 30+ days late (+1 if repaid with 1-2 late months, 0 if no history)
-3  any cheque/direct-debit bounce in the last 12 months (per policy, applied once)
-4  any 90+ day delinquency on eCIB in the last 24 months
-6  any write-off or litigation flag on eCIB, ever
Decision bands: score >= 6 STRONG | 3-5 ACCEPTABLE | 0-2 THIN | below 0 POOR"""


def render_product(product: LoanProduct) -> str:
    rate_kind = "reducing rate" if product.rate_type == "reducing" else "flat rate"
    tenors = "/".join(str(t) for t in product.tenors)
    return (
        f"{product.name}: {_rs(product.min_amount)}-{_rs(product.max_amount)}, "
        f"tenors {tenors} months, {product.annual_rate * 100:.1f}% p.a. ({rate_kind}), "
        f"processing fee {product.processing_fee_pct * 100:.1f}%, "
        f"min income {_rs(product.min_income)}"
    )


def render_offer_question(p: CustomerProfile, product: LoanProduct) -> str:
    return (
        f"{render_profile(p)}\n\n{SCORING_POLICY_BLOCK}\n\n"
        f"Product on the shelf for proactive offers: {render_product(product)}\n"
        "Bank policy: proactive offers only to STRONG or ACCEPTABLE relationships "
        "showing genuine cash stress; THIN gets monitoring; POOR gets no unsecured "
        "offer (secured/deposit-backed alternatives only). DBR cap 40%.\n\n"
        "Should we proactively offer this customer a loan? Walk through the decision."
    )


def render_decision_facts(d: Decision) -> str:
    """The policy engine's verdict, given to the LLM as ground truth to
    narrate. The model explains the decision; it does not make it."""
    lines = [
        "POLICY ENGINE VERDICT (authoritative — do not contradict)",
        f"- Decision: {d.action}",
        f"- Relationship score: {d.score.score} ({d.score.band})",
        "- Score components: "
        + "; ".join(f"{c.label} ({c.points:+d})" for c in d.score.components),
        f"- Cash stress: {'yes' if d.stress.stressed else 'no'} — "
        + "; ".join(d.stress.notes),
    ]
    if d.offer:
        o = d.offer
        lines += [
            f"- Offer: {_rs(o.amount)} over {o.tenor_months} months at "
            f"{o.annual_rate * 100:.1f}% p.a. ({o.rate_type})",
            f"- Instalment: {_rs(o.emi)}/month, DBR {o.dbr_pct}% against the 40% cap",
            f"- Processing fee: {_rs(o.processing_fee)}; monthly headroom after "
            f"all obligations: {_rs(o.headroom_after)}",
        ]
    for r in d.reasons:
        lines.append(f"- Rationale: {r}")
    if d.alternatives:
        lines.append("- Responsible alternatives: " + "; ".join(d.alternatives))
    return "\n".join(lines)


def template_narrative(p: CustomerProfile, d: Decision) -> str:
    """Deterministic fallback narrative when no local LLM is reachable."""
    if d.action == "OFFER" and d.offer:
        o = d.offer
        return (
            f"Offer. {p.name} scores {d.score.score} ({d.score.band}) with genuine "
            f"cash stress ({d.stress.days_below_5k} days under Rs 5,000 in the last "
            f"30) — this is where a proactive offer lands as help rather than "
            f"marketing.\n\nProposed terms: {_rs(o.amount)} over {o.tenor_months} "
            f"months at {o.annual_rate * 100:.1f}% p.a. ({o.rate_type}), instalment "
            f"{_rs(o.emi)}/month, DBR {o.dbr_pct}% against the 40% cap, leaving "
            f"{_rs(o.headroom_after)}/month after all obligations. Offer below the "
            "maximum where the purpose allows — a customer at exactly the cap has "
            "no buffer for a bad month."
        )
    if d.action == "DECLINE":
        alts = "\n".join(f"- {a}" for a in d.alternatives)
        return (
            f"No unsecured offer. {' '.join(d.reasons)}\n\n"
            f"What we can responsibly put in front of them:\n{alts}\n\n"
            "Say it plainly: what blocked approval, that it is repairable, and "
            "which of the above we will open today. A declined customer who leaves "
            "with a workable path stays a customer."
        )
    return f"Monitor. {' '.join(d.reasons)}"


# ── eCIB report reading ──

def render_ecib_report(lines: list[BureauFacility]) -> str:
    rows = "\n".join(
        f"| {f.facility} | {f.institution} | {_rs(f.amount)} | {f.status} |" for f in lines
    )
    return (
        "eCIB REPORT EXTRACT\n\n"
        "| Facility | Institution | Amount | Current status |\n|---|---|---|---|\n"
        f"{rows}\n\nRead this bureau file - can we lend unsecured against it?"
    )


def render_ecib_facts(a: ECIBAssessment) -> str:
    lines = [
        "eCIB READING (authoritative)",
        f"- {len(a.lines)} lines on file, {a.active_count} active",
        f"- Write-off on file: {'yes' if a.has_writeoff else 'no'}",
        f"- Active delinquency (30/60/90+ DPD): {'yes' if a.has_active_delinquency else 'no'}",
        f"- Can lend unsecured: {'yes' if a.can_lend_unsecured else 'no'}",
    ]
    return "\n".join(lines)


# ── running loan block ──

def render_running_loan(loan: RunningLoan) -> str:
    return (
        f"- Rs {loan.original_amount:,} at {loan.annual_rate * 100:.1f}%, "
        f"{loan.instalments_paid}/{loan.instalments_total} instalments paid, "
        f"outstanding Rs {loan.outstanding:,}, instalment Rs {loan.instalment:,}"
    )


# ── delinquency risk grading ──

def render_delinquency_facts(g: DelinquencyGrade) -> str:
    lines = [f"RISK GRADE (authoritative): {g.grade} (score {g.score})"]
    lines += [f"- {r}" for r in g.reasons]
    lines.append(f"- Action: {g.action}")
    return "\n".join(lines)


# ── restructuring assessment ──

def render_restructuring_facts(r: RestructuringResult) -> str:
    return "\n".join([
        "RESTRUCTURING ANALYSIS (authoritative)",
        f"- Old instalment: Rs {r.old_instalment:,} (DBR {r.old_dbr_pct}%)",
        f"- New instalment over {r.new_tenor_months} months: Rs {r.new_instalment:,} "
        f"(DBR {r.new_dbr_pct}%)",
        f"- Extra markup cost of the extension: Rs {r.extra_markup_cost:,}",
        f"- Fixes affordability under the DBR cap: {'yes' if r.fixes_affordability else 'no'}",
        f"- Instalment that would fit under the cap: Rs {r.target_instalment:,}",
    ])


# ── top-up eligibility ──

def render_topup_facts(t: TopupDecision) -> str:
    if not t.approved and t.new_principal is None:
        return f"TOP-UP VERDICT (authoritative): DECLINED - {t.reason}"
    lines = [f"TOP-UP VERDICT (authoritative): {'APPROVED' if t.approved else 'AMOUNT TOO HIGH'}"]
    lines.append(f"- New principal: Rs {t.new_principal:,}, new instalment: Rs {t.new_instalment:,}")
    lines.append(f"- DBR: {t.dbr_pct}%")
    if not t.approved:
        lines.append(f"- Reason: {t.reason}")
        if t.max_approvable_topup is not None:
            lines.append(f"- Largest top-up that fits: about Rs {t.max_approvable_topup:,}")
    return "\n".join(lines)


# ── loan offer comparison ──

def render_offer_comparison_facts(costs: list[QuoteCost]) -> str:
    lines = ["OFFER COMPARISON (authoritative, cheapest first)"]
    for c in costs:
        lines.append(f"- {c.label}: instalment Rs {c.instalment:,}, all-in total Rs {c.total_cost:,}")
    return "\n".join(lines)


# ── early settlement ──

def render_early_settlement_facts(r: EarlySettlementResult) -> str:
    return "\n".join([
        "EARLY SETTLEMENT ANALYSIS (authoritative)",
        f"- Cost to settle today: Rs {r.cost_to_settle:,}",
        f"- Cost to continue (remaining instalments): Rs {r.cost_to_continue:,}",
        f"- Saving: Rs {r.savings:,}",
        f"- Worth settling: {'yes' if r.worth_it else 'no'}",
    ])


# ── gold loan sizing ──

def render_gold_loan_facts(g: GoldOffer, requested: int, r: GoldLoanResult) -> str:
    return "\n".join([
        "GOLD LOAN SIZING (authoritative)",
        f"- Assessed value: Rs {r.assessed_value:,} ({g.weight_tola} tola adjusted to {g.purity_k}K)",
        f"- Max loan at {g.ltv_pct:.0%} LTV: Rs {r.max_loan:,}",
        f"- Requested: Rs {requested:,}, granted: Rs {r.granted_amount:,}"
        + (f" (shortfall Rs {r.shortfall:,})" if r.shortfall else " (fully covered)"),
        f"- Instalment: Rs {r.instalment:,}/month",
    ])


# ── salary advance ──

def render_salary_advance_facts(r: SalaryAdvanceResult) -> str:
    if not r.approved:
        return f"SALARY ADVANCE VERDICT (authoritative): DECLINE - {r.reason}"
    return "\n".join([
        "SALARY ADVANCE VERDICT (authoritative): APPROVE",
        f"- Granted: Rs {r.granted_amount:,}",
        f"- Instalment: Rs {r.instalment:,}/month, fee: Rs {r.fee:,} upfront",
        f"- Total cost of the advance: Rs {r.total_cost:,}",
    ])


# ── product recommendation ──

def render_product_recommendation_facts(rec: ProductRecommendation) -> str:
    lines = ["PRODUCT MATCH (authoritative)"]
    if rec.best:
        b = rec.best
        lines.append(
            f"- Best fit: {b.name} over {b.tenor_months} months - instalment "
            f"Rs {b.instalment:,}, DBR {b.dbr_pct}%, total cost above principal Rs {b.total_cost:,}"
        )
        for s in rec.survivors[1:]:
            lines.append(f"- Next best: {s.name}, total cost Rs {s.total_cost:,}")
    else:
        lines.append("- No product on the shelf works for this request")
    for name, reason in rec.eliminated:
        lines.append(f"- Eliminated: {name} ({reason})")
    return "\n".join(lines)


# ── risk-based pricing ──

def render_risk_pricing_facts(r: RiskPricingResult) -> str:
    return "\n".join([
        "RISK-BASED PRICING (authoritative)",
        f"- Relationship score: {r.score.score} ({r.score.band})",
        f"- Rate: {r.annual_rate}% p.a. (reducing), markup {r.markup_pts} pts over base",
        f"- Instalment: Rs {r.instalment:,}/month",
    ])
