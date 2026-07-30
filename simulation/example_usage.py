"""
Direct-import example — no server, no framework. Run: python example_usage.py
This is the 'integrate anywhere' path: import the agents, call functions, use results.
"""
from agents import CreditSentinel, CashPulse, Advisor, ollama_llm

# ----- 1. Credit Sentinel -----------------------------------------------------
credit = CreditSentinel()   # or CreditSentinel(CreditPolicy(markup_max=36, ...))

customer = {
    "name": "Bilal Ahmed", "area": "Johar Town",
    "income": 88000, "balance": 6200, "balance_trend": -1,
    "account_age_months": 31, "existing_exposure": 1,
    "repayment_pct": 91, "missed_payments": 0, "defaulted": False,
}
decision = credit.assess(customer)
print("CREDIT DECISION")
print(f"  {decision['decision']}  | risk {decision['risk']}"
      f"  | offer {decision['offer']:,}  | markup {decision['markup_pct']}%")
for r in decision["reasons"]:
    print(f"   - [{r['level']}] {r['text']}")

# batch + portfolio
book = [customer,
        {**customer, "name": "Kamran Malik", "repayment_pct": 34, "defaulted": True},
        {**customer, "name": "Hina Raza", "balance": 410000, "balance_trend": 1}]
print("\nPORTFOLIO", credit.portfolio_summary(book))

# ----- 2. Cash Pulse ----------------------------------------------------------
cash = CashPulse()
atms = [
    {"id": "LHR-01", "location": "Gulberg", "capacity": 9_000_000, "cash": 1_500_000, "daily_dispense": 2_900_000},
    {"id": "LHR-02", "location": "DHA Phase 5", "capacity": 7_000_000, "cash": 6_100_000, "daily_dispense": 1_000_000},
    {"id": "LHR-03", "location": "Wapda Town", "capacity": 4_000_000, "cash": 640_000, "daily_dispense": 1_500_000},
]
print("\nATM STATUS")
for e in cash.evaluate_all(atms):
    print(f"  {e['location']:<14} {e['status']:<12} {e['days_to_empty']:>4} days  -> {e['recommendation']}")
print("NETWORK", cash.network_summary(atms))
print("REBALANCE", cash.rebalance_plan(atms))

# ----- 3. Advisor chat (uses local LLM if reachable, else rule-based) ---------
advisor = Advisor(llm=ollama_llm(model="llama3.2"))   # pass llm=None to force rules
chat_customer = {**customer, **{k: decision[k] for k in ("decision", "offer")},
                 "markup_pct": decision["markup_pct"]}

# proactive opening the moment they're flagged 'instant':
if decision["instant"]:
    print("\nADVISOR opens chat:")
    for line in advisor.opening_offer(chat_customer):
        print("   bank>", line)

messages = [{"role": "assistant", "content": "You are pre-approved."},
            {"role": "user", "content": "kitna interest lagega?"}]
reply, source = advisor.reply(chat_customer, messages)
print(f"\nADVISOR reply [{source}]:\n   bank> {reply}")

# ----- 4. Full engine: all 51 loan + ATM task types, local-LLM narrated -----
# This is the deep engine behind /loans/{task} and /atm/tasks/{task} in
# service.py — deterministic policy from loan-agent + agents/atm, narrated by
# the fine-tuned model (see run_demo.py for the full "day in the life" tour).
from app.agents.loans.agent import LoanAgent as FullLoanAgent
from agents.atm import AtmAgent, AtmMachine, Cassette
from agents.atm.cash_mechanics import CashRunoutForecastInput, forecast_cash_runout, render_cash_runout_facts

full_loan_agent = FullLoanAgent()
scoring = full_loan_agent.relationship_scoring("C001", use_llm=False)
print(f"\nFULL LOAN ENGINE — relationship_scoring [{scoring.narrative_source}]:")
print(f"   {scoring.narrative}")

atm = AtmMachine(
    atm_id="LHR-01", bank="HBL", location="Gulberg", model="NCR SelfServ 26",
    cassettes=[Cassette(denom=5000, notes=200, capacity=2000), Cassette(denom=1000, notes=500, capacity=2500)],
)
runout = forecast_cash_runout(atm, CashRunoutForecastInput(
    current_time="09:00", dispense_rate_per_hour=180_000, low_cash_threshold=150_000,
    cit_time="17:00", cit_hours_from_now=8,
))
atm_agent = AtmAgent()
narrative, source = atm_agent.narrate(render_cash_runout_facts(runout), "cash_runout_forecast", use_llm=False)
print(f"\nFULL ATM ENGINE — cash_runout_forecast [{source}]:")
print(f"   {narrative}")
