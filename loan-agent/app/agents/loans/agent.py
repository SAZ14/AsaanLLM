"""The Loans agent.

Scans the customer book, grades every relationship under the bank policy,
detects genuine cash stress, and decides who gets a proactive instant-loan
offer, who is monitored, and who is declined with secured alternatives.

Split of responsibilities:
  * :mod:`policy` computes every number and makes the decision — deterministic.
  * The local LLM (base model today, fine-tuned tomorrow) narrates the decision
    and answers free-form credit questions. It is handed the policy verdict as
    ground truth and cannot flip it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.agents.loans import llm as loan_llm
from app.agents.loans.book import load_book
from app.agents.loans.models import (
    BureauFacility,
    CustomerProfile,
    Decision,
    GoldOffer,
    LoanProduct,
    LoanProductOption,
    LoanQuote,
    RunningLoan,
    SalaryAdvanceProduct,
)
from app.agents.loans.policy import (
    DBR_CAP,
    PERSONAL_INSTALMENT_LOAN,
    STRESS_DAYS_THRESHOLD,
    assess_restructuring,
    assess_salary_advance,
    assess_topup,
    compare_loan_offers,
    analyze_early_settlement,
    decide,
    detect_stress,
    dbr,
    emi,
    grade_delinquency_risk,
    max_affordable_loan,
    price_by_risk,
    read_ecib_report,
    recommend_product,
    relationship_score,
    size_gold_loan,
    triage,
)
from app.agents.loans.prompts import (
    SYSTEM_PROMPT,
    render_decision_facts,
    render_delinquency_facts,
    render_early_settlement_facts,
    render_ecib_facts,
    render_ecib_report,
    render_gold_loan_facts,
    render_offer_comparison_facts,
    render_offer_question,
    render_product,
    render_product_recommendation_facts,
    render_profile,
    render_restructuring_facts,
    render_risk_pricing_facts,
    render_running_loan,
    render_salary_advance_facts,
    render_topup_facts,
    template_narrative,
)

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_think(text: str) -> str:
    """The fine-tuned model emits <think> scratchpads; hide them from output.
    An unclosed tag (truncated generation) drops everything from the tag on."""
    text = _THINK_RE.sub("", text)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


@dataclass
class Assessment:
    customer: CustomerProfile
    decision: Decision
    narrative: str
    narrative_source: str  # "llm" | "template"


@dataclass
class TaskResult:
    """Generic result for the non-OFFER/MONITOR/DECLINE task types: the
    deterministic facts (task-specific dataclass) plus a narrative."""
    facts: object
    narrative: str
    narrative_source: str  # "llm" | "template"


class LoanAgent:
    def __init__(
        self,
        customers: list[CustomerProfile] | None = None,
        product: LoanProduct | None = None,
        client=None,
        book_path=None,
    ) -> None:
        self.customers = customers if customers is not None else load_book(book_path)
        self.product = product or PERSONAL_INSTALMENT_LOAN
        self.client = client
        self._by_id: dict[str, CustomerProfile] = {}
        for c in self.customers:
            if c.customer_id in self._by_id:
                raise ValueError(f"Duplicate customer_id {c.customer_id!r} in book")
            self._by_id[c.customer_id] = c

    def get(self, customer_id: str) -> CustomerProfile:
        try:
            return self._by_id[customer_id]
        except KeyError:
            raise KeyError(f"No customer {customer_id!r} in the book") from None

    # ── single-customer decision ──

    def assess(self, customer_id: str, use_llm: bool = True) -> Assessment:
        customer = self.get(customer_id)
        decision = decide(customer, self.product)
        narrative, source = None, "template"
        if use_llm:
            prompt = (
                render_offer_question(customer, self.product)
                + "\n\n"
                + render_decision_facts(decision)
            )
            raw = loan_llm.chat(
                prompt,
                system=SYSTEM_PROMPT,
                fewshot_tasks=["proactive_offer_decision", "decline_with_alternatives"],
                client=self.client,
            )
            stripped = _strip_think(raw) if raw else ""
            if stripped:  # empty after stripping (e.g. truncated <think>) → template
                narrative, source = stripped, "llm"
        if narrative is None:
            narrative = template_narrative(customer, decision)
        return Assessment(customer=customer, decision=decision,
                          narrative=narrative, narrative_source=source)

    # ── whole-book scan ──

    def scan(self) -> dict[str, list[tuple[CustomerProfile, Decision]]]:
        """Triage the whole book into OFFER / MONITOR / DECLINE queues,
        offers prioritised strongest-file-deepest-stress first."""
        return triage(self.customers, self.product)

    # ── free-form questions (EMI maths, eCIB reading, restructuring…) ──

    def ask(self, question: str, customer_id: str | None = None) -> str | None:
        """Free-form credit-ops question against the local model, optionally
        grounded in one customer's profile. Returns None if no model is up."""
        content = question
        if customer_id:
            content = f"{render_profile(self.get(customer_id))}\n\n{question}"
        raw = loan_llm.chat(content, system=SYSTEM_PROMPT, client=self.client)
        return _strip_think(raw) if raw else None

    # ── shared narration helper for the task-specific methods below ──

    def _narrate(self, prompt: str, facts, task_hint: str, use_llm: bool) -> TaskResult:
        narrative, source = None, "template"
        if use_llm:
            raw = loan_llm.chat(
                prompt, system=SYSTEM_PROMPT, fewshot_tasks=[task_hint], client=self.client,
            )
            stripped = _strip_think(raw) if raw else ""
            if stripped:
                narrative, source = stripped, "llm"
        if narrative is None:
            # deterministic fallback: the "authoritative facts" block is
            # already human-readable — use it instead of the raw dataclass repr.
            # Drop the ALL-CAPS "(authoritative)" header line first — that's an
            # instruction for the LLM, not narration meant for a user to read.
            narrative = prompt.rsplit("\n\n", 1)[-1]
            lines = narrative.split("\n")
            if lines and "authoritative" in lines[0].lower():
                lines = lines[1:]
            narrative = "\n".join(lines).strip()
        return TaskResult(facts=facts, narrative=narrative, narrative_source=source)

    # ── the original 8 tasks, given the same narrated-method treatment as
    #    the 10 added later, so every one of the 18 loan task types has a
    #    consistent agent-level entry point ──

    def relationship_scoring(self, customer_id: str, use_llm: bool = True) -> TaskResult:
        """Explain the relationship grade — nothing more.

        The prompt states the task and scopes it explicitly. Handed only a
        profile and a bare score, the fine-tuned model guesses which of its 51
        trained tasks it is being asked for and frequently guesses wrong
        (observed: it invented a Rs 625,000 loan request, an instalment and a
        sanctioned amount, none of which exist, and never mentioned the score).
        """
        customer = self.get(customer_id)
        facts = relationship_score(customer)
        components = "; ".join(f"{c.label} ({c.points:+d})" for c in facts.components)
        prompt = (
            f"{render_profile(customer)}\n\n"
            "Grade this customer's relationship with the bank and explain the grade. "
            "Do NOT make a lending decision, quote a loan amount, or invent a request — "
            "this task is the score only.\n\n"
            "RELATIONSHIP SCORE (authoritative — do not contradict)\n"
            f"- Score: {facts.score} ({facts.band})\n"
            f"- Components: {components}"
        )
        return self._narrate(prompt, facts, "relationship_scoring", use_llm)

    def low_cash_detection(self, customer_id: str, use_llm: bool = True) -> TaskResult:
        """Read the liquidity picture — explicitly NOT a lending decision.

        Without the scope line the model volunteers a verdict, and it can be
        the wrong one (observed: "Decline unsecured lending" on a file the
        policy engine actually OFFERs).
        """
        customer = self.get(customer_id)
        facts = detect_stress(customer)
        notes = "\n".join(f"- {n}" for n in facts.notes)
        prompt = (
            f"{render_profile(customer)}\n\n"
            "Is this customer genuinely short of cash, or is the balance pattern normal? "
            "Report the liquidity read ONLY. Do not approve, decline, or recommend any "
            "product — whether this becomes an offer is decided elsewhere.\n\n"
            "CASH-STRESS SIGNALS (authoritative — do not contradict)\n"
            f"- Genuine cash stress: {'yes' if facts.stressed else 'no'}\n"
            f"- Days under Rs 5,000 in the last 30: {facts.days_below_5k} "
            f"(the policy marker is {STRESS_DAYS_THRESHOLD}+)\n"
            f"{notes}"
        )
        return self._narrate(prompt, facts, "low_cash_detection", use_llm)

    def portfolio_triage(self, use_llm: bool = True) -> TaskResult:
        """Whole-book triage. The facts block states the counts explicitly and
        marks them authoritative — without that the model counts the listed
        names itself and gets it wrong (observed: it reported 46 offers and an
        invented '45.7% of the queue' against a true count of 20)."""
        queues = self.scan()
        total = sum(len(rows) for rows in queues.values())
        # Naming rule matters: given a count but no names, the model invents
        # plausible ones (observed: six fictional MONITOR customers attributed
        # to real customer ids). Only names supplied below may be used.
        lines = [
            "Summarise this lending queue for the branch. Use the counts below "
            "exactly as given; do not recount. Name ONLY the customers listed "
            "below — do not enumerate the MONITOR queue, refer to it by count.",
            "",
            "PORTFOLIO TRIAGE (authoritative — do not contradict)",
            f"- Book size: {total} customers",
        ]
        # "36 of 60" rather than "36 customers (60.0% of the book)": with a
        # percentage sitting next to a count the model conflated the two
        # (observed: "Monitor list (60 customers)" for a true count of 36).
        for action in ("OFFER", "MONITOR", "DECLINE"):
            lines.append(f"- {action}: {len(queues[action])} of {total}")
        top = queues["OFFER"][:5]
        if top:
            lines.append(
                "- Highest-priority offers (strongest file, deepest stress first): "
                + "; ".join(
                    f"{c.name} ({c.customer_id}) score {d.score.score}, "
                    f"{d.stress.days_below_5k}d under Rs 5,000"
                    for c, d in top
                )
            )
        worst = queues["DECLINE"][:3]
        if worst:
            lines.append(
                "- Declines needing secured alternatives: "
                + "; ".join(f"{c.name} ({c.customer_id}) score {d.score.score}" for c, d in worst)
            )
        return self._narrate("\n".join(lines), queues, "portfolio_triage", use_llm)

    def dbr_calculation(
        self, net_income: int, existing_obligations: int, new_emi: float, use_llm: bool = True,
    ) -> TaskResult:
        pct = round(dbr(net_income, existing_obligations, new_emi), 1)
        cap = DBR_CAP * 100
        prompt = (
            "Work out the debt-burden ratio for this facility and say whether it passes. "
            "Show the arithmetic; do not decide anything beyond the DBR test.\n\n"
            "DBR CALCULATION (authoritative — do not contradict)\n"
            f"- Net monthly income: Rs {net_income:,}\n"
            f"- Existing obligations: Rs {existing_obligations:,}/month\n"
            f"- New instalment: Rs {new_emi:,.0f}/month\n"
            f"- DBR: {pct}% against the {cap:.0f}% cap — "
            f"{'passes' if pct <= cap else 'breaches the cap'}"
        )
        return self._narrate(prompt, pct, "dbr_calculation", use_llm)

    def max_affordable_loan(
        self, customer_id: str, product: LoanProduct | None = None, tenor: int | None = None,
        use_llm: bool = True,
    ) -> TaskResult:
        customer = self.get(customer_id)
        product = product or self.product
        tenor = tenor or product.tenors[-1]
        amount = max_affordable_loan(customer, product, tenor)
        headroom = round(DBR_CAP * customer.net_monthly_income - customer.total_obligations)
        prompt = (
            f"{render_profile(customer)}\n\n"
            f"How much can we responsibly lend this customer over {tenor} months? "
            "Report the sizing only — do not grade the relationship or approve the file.\n\n"
            "AFFORDABILITY (authoritative — do not contradict)\n"
            f"- Product: {render_product(product)}\n"
            f"- Instalment headroom under the {DBR_CAP:.0%} DBR cap: Rs {headroom:,}/month\n"
            f"- Largest responsible principal over {tenor} months: Rs {amount:,}"
            + ("" if amount else " (nothing is lendable at this tenor)")
        )
        return self._narrate(prompt, amount, "max_affordable_loan", use_llm)

    def emi_calculation(
        self, principal: int, annual_rate: float, tenor_months: int, rate_type: str = "reducing",
        use_llm: bool = True,
    ) -> TaskResult:
        instalment = round(emi(principal, annual_rate, tenor_months, rate_type))
        total_paid = instalment * tenor_months
        prompt = (
            "Calculate the instalment and the all-in cost of this quote. "
            "Do not assess eligibility or approve anything — this is the maths only.\n\n"
            "INSTALMENT (authoritative — do not contradict)\n"
            f"- Principal: Rs {principal:,} over {tenor_months} months at "
            f"{annual_rate * 100:.1f}% p.a. ({rate_type})\n"
            f"- Instalment: Rs {instalment:,}/month\n"
            f"- Total repaid: Rs {total_paid:,}\n"
            f"- Markup over the life: Rs {total_paid - principal:,}"
        )
        return self._narrate(prompt, instalment, "emi_calculation", use_llm)

    # ── eCIB report reading ──

    def ecib_report(self, lines: list[BureauFacility], use_llm: bool = True) -> TaskResult:
        facts = read_ecib_report(lines)
        prompt = render_ecib_report(lines) + "\n\n" + render_ecib_facts(facts)
        return self._narrate(prompt, facts, "ecib_report_reading", use_llm)

    # ── delinquency risk grading ──

    def delinquency_risk(
        self, customer_id: str, loan: RunningLoan, late_delays: list[int], use_llm: bool = True,
    ) -> TaskResult:
        customer = self.get(customer_id)
        facts = grade_delinquency_risk(customer, loan, late_delays)
        prompt = (
            f"{render_profile(customer)}\n\nRUNNING FACILITY\n{render_running_loan(loan)}\n"
            f"Payment delays on last {len(late_delays)} instalments: "
            f"{', '.join(str(d) for d in late_delays)} days\n\n"
            + render_delinquency_facts(facts)
        )
        return self._narrate(prompt, facts, "delinquency_risk_grading", use_llm)

    # ── restructuring assessment ──

    def restructuring(
        self, customer_id: str, loan: RunningLoan, new_income: int, extension_months: int,
        use_llm: bool = True,
    ) -> TaskResult:
        customer = self.get(customer_id)
        facts = assess_restructuring(customer, loan, new_income, extension_months)
        prompt = (
            f"{render_profile(customer)}\n\nSITUATION\nRunning loan: {render_running_loan(loan)}\n"
            f"Income has dropped to Rs {new_income:,}\n\n" + render_restructuring_facts(facts)
        )
        return self._narrate(prompt, facts, "restructuring_assessment", use_llm)

    # ── top-up eligibility ──

    def topup(
        self, customer_id: str, loan: RunningLoan, late_in_12m: int, request_amount: int,
        use_llm: bool = True,
    ) -> TaskResult:
        customer = self.get(customer_id)
        facts = assess_topup(customer, loan, late_in_12m, request_amount)
        prompt = (
            f"{render_profile(customer)}\n\nRUNNING LOAN\n{render_running_loan(loan)}\n"
            f"REQUEST: Rs {request_amount:,} top-up\n\n" + render_topup_facts(facts)
        )
        return self._narrate(prompt, facts, "topup_eligibility", use_llm)

    # ── loan offer comparison (no customer required) ──

    def offer_comparison(
        self, principal: int, tenor: int, quotes: list[LoanQuote], use_llm: bool = True,
    ) -> TaskResult:
        facts = compare_loan_offers(principal, tenor, quotes)
        prompt = (
            f"I need Rs {principal:,} for {tenor} months. Quotes on the table:\n\n"
            + render_offer_comparison_facts(facts)
        )
        return self._narrate(prompt, facts, "loan_offer_comparison", use_llm)

    # ── early settlement analysis (no customer required) ──

    def early_settlement(
        self, loan: RunningLoan, settlement_fee_pct: float, use_llm: bool = True,
    ) -> TaskResult:
        facts = analyze_early_settlement(loan, settlement_fee_pct)
        prompt = (
            f"MY RUNNING LOAN\n{render_running_loan(loan)}\n"
            f"Early settlement fee: {settlement_fee_pct:.0%} of outstanding principal\n\n"
            + render_early_settlement_facts(facts)
        )
        return self._narrate(prompt, facts, "early_settlement_analysis", use_llm)

    # ── gold loan sizing (no customer profile strictly required) ──

    def gold_loan(
        self, gold: GoldOffer, requested_amount: int, annual_rate: float, tenor_months: int,
        use_llm: bool = True,
    ) -> TaskResult:
        facts = size_gold_loan(gold, requested_amount, annual_rate, tenor_months)
        prompt = (
            f"GOLD OFFERED\n- Weight: {gold.weight_tola} tola, {gold.purity_k}K purity\n"
            f"- Assessed market rate: Rs {gold.rate_per_tola_24k:,} per tola (24K basis)\n"
            f"- Bank LTV policy: {gold.ltv_pct:.0%} of assessed value\n"
            f"- Customer wants: Rs {requested_amount:,}\n\n" + render_gold_loan_facts(gold, requested_amount, facts)
        )
        return self._narrate(prompt, facts, "gold_loan_sizing", use_llm)

    # ── salary advance assessment ──

    def salary_advance(
        self, customer_id: str, product: SalaryAdvanceProduct, requested_amount: int,
        use_llm: bool = True,
    ) -> TaskResult:
        customer = self.get(customer_id)
        facts = assess_salary_advance(customer, product, requested_amount)
        prompt = (
            f"{render_profile(customer)}\n\nSALARY ADVANCE PRODUCT\n"
            f"- Limit: up to {product.limit_multiple}x of net monthly salary\n"
            f"- Fee: {product.fee_pct:.1%} | Markup: {product.annual_rate:.1%} p.a. (flat) | "
            f"Tenor: {product.tenor_months} month(s)\n"
            f"- Customer requests: Rs {requested_amount:,}\n\n" + render_salary_advance_facts(facts)
        )
        return self._narrate(prompt, facts, "salary_advance_assessment", use_llm)

    # ── product recommendation ──

    def product_recommendation(
        self, customer_id: str, need_amount: int, shelf: list[LoanProductOption],
        has_gold: bool = False, deposit_amount: int = 0, has_salary_account: bool = True,
        use_llm: bool = True,
    ) -> TaskResult:
        customer = self.get(customer_id)
        facts = recommend_product(
            customer, need_amount, shelf, has_gold=has_gold,
            deposit_amount=deposit_amount, has_salary_account=has_salary_account,
        )
        prompt = (
            f"{render_profile(customer)}\n\nNEED\n- Amount: Rs {need_amount:,}\n\n"
            + render_product_recommendation_facts(facts)
        )
        return self._narrate(prompt, facts, "product_recommendation", use_llm)

    # ── risk-based pricing ──

    def risk_pricing(
        self, customer_id: str, base_rate: float, acceptable_markup_pts: float,
        thin_markup_pts: float, requested_amount: int, tenor_months: int, use_llm: bool = True,
    ) -> TaskResult:
        customer = self.get(customer_id)
        facts = price_by_risk(
            customer, base_rate, acceptable_markup_pts, thin_markup_pts,
            requested_amount, tenor_months,
        )
        prompt = (
            f"{render_profile(customer)}\n\nWhat band is this, and what should we price a "
            f"Rs {requested_amount:,}/{tenor_months}-month request at?\n\n"
            + render_risk_pricing_facts(facts)
        )
        return self._narrate(prompt, facts, "risk_based_pricing", use_llm)


# ── convenience one-shot runner (mirrors run_<name>_agent convention) ──

def run_loan_agent(
    customer_ids: list[str] | None = None,
    customers: list[CustomerProfile] | None = None,
    product: LoanProduct | None = None,
    client=None,
    use_llm: bool = True,
) -> list[Assessment]:
    """Assess a list of customers (default: everyone in the book)."""
    agent = LoanAgent(customers=customers, product=product, client=client)
    ids = customer_ids or [c.customer_id for c in agent.customers]
    return [agent.assess(cid, use_llm=use_llm) for cid in ids]
