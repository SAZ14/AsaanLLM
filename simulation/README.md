# Banking Agents — loans + ATM, quick-start and full engine

Two tiers, same directory:

1. **Quick-start** (`agents/credit_sentinel.py`, `agents/cash_pulse.py`,
   `agents/advisor.py`) — zero-dependency, hand-rolled scoring you can drop
   into any app in five minutes. Good for prototyping against your own data
   shape.
2. **Full engine** (`../loan-agent` + `agents/atm/`) — the deterministic
   policy engines behind all **51 task types** trained into
   `merged_finetune_train.jsonl` (18 loan + 33 ATM), narrated by your
   fine-tuned local model. This is what `run_demo.py` and the
   `/loans/{task}` / `/atm/tasks/{task}` API endpoints use.

Both tiers share one design: **the policy computes every number and verdict;
the LLM only narrates.** A hallucinating model can't override a decision it
was never asked to make.

```
simulation/
  agents/
    credit_sentinel.py, cash_pulse.py, advisor.py   — quick-start tier
    atm/                                              — full ATM engine (33 tasks)
      entities.py        — Cassette, AtmMachine, CustomerAtmProfile, Transaction
      cash_mechanics.py  — 8 tasks: runout forecast, cassette triage, reconciliation, ...
      network_ops.py     — 8 tasks: alert triage, CIT routing, EOD reports, ...
      customer_money.py  — 9 tasks: fees, ATM recommendation, travel cash, ...
      judgment.py         — 8 tasks: fault diagnosis, fraud, security, SLA, ...
      agent.py            — AtmAgent: LLM narration + TASK_REGISTRY
      llm.py               — local model client (Ollama by default)
      sample_data.py        — parsers pulling real scenarios out of the dataset
  service.py         — FastAPI: quick-start + full-engine endpoints + /dashboard
  dispatch.py         — generic dataclass<->JSON wiring for the /loans, /atm/tasks routes
  static/              — the visual demo (no build step, no framework, no CDN)
    dashboard.html     — shell: masthead, 4 tabs, drawer
    dashboard.css      — design tokens + every component
    dashboard.js       — loans desk, ATM ops, task ledger, what-if simulator
    story.js            — "How it works": pipeline diagram, guided tour, quiz
  run_demo.py          — narrative "day in the life" tour across both domains (terminal)
  example_usage.py      — direct-import examples, both tiers
  data/atm/fewshot.jsonl — one worked example per ATM task, for few-shot prompting
../loan-agent/          — full LOAN engine (18 tasks) — see its own README
```

---

## Quick-start tier

The core has **no dependencies**. Copy the `agents/` folder into your project.

```python
from agents import CreditSentinel, CashPulse, Advisor, ollama_llm

credit = CreditSentinel()
decision = credit.assess(customer_dict)     # -> dict with decision/risk/offer/markup/reasons

cash = CashPulse()
status = cash.evaluate(atm_dict)            # -> status/days_to_empty/recommendation
plan   = cash.rebalance_plan(atm_list)      # -> concrete "move X from A to B"

advisor = Advisor(llm=ollama_llm())         # or llm=None for rule-based only
reply, source = advisor.reply(customer_dict, messages)
```

### Input schemas (aliases are accepted, so messy data still maps)

**Customer** → Credit Sentinel / Advisor
```
name, area, income (monthly), balance, balance_trend (-1|0|1),
account_age_months, existing_exposure (# open loans), repayment_pct (0-100),
missed_payments, defaulted (bool)
```

**ATM** → Cash Pulse
```
id, location, capacity, cash (current), daily_dispense (avg per day)
```

### Dropping into an existing app

**Flask** — no HTTP hop, call the core directly:
```python
from flask import request, jsonify
from agents import CreditSentinel
credit = CreditSentinel()

@app.post("/api/assess")
def assess():
    return jsonify(credit.assess(request.get_json()["customer"]))
```

**Django** (DRF):
```python
from agents import CreditSentinel
credit = CreditSentinel()

class AssessView(APIView):
    def post(self, request):
        return Response(credit.assess(request.data["customer"]))
```

### Tuning to your real product

Everything policy-related lives in two dataclasses — no magic numbers scattered around:

```python
from agents import CreditSentinel, CreditPolicy, CashPulse, CashConfig

credit = CreditSentinel(CreditPolicy(
    min_repayment_pct=60, approve_risk_cutoff=40,
    markup_min=20, markup_max=34, offer_max=800_000,
))
cash = CashPulse(CashConfig(
    critical_days=1.0, overstock_days=10, idle_cost_annual=0.15,
))
```

---

## Full engine tier — all 51 task types

Install the loan engine as a local editable dependency (only needed once):

```bash
pip install -e ../loan-agent --no-deps
pip install -r requirements.txt   # fastapi, uvicorn, openai
```

### Run the visual demo (start here)

```bash
uvicorn service:app --port 8080
# then open http://localhost:8080/dashboard
```

Four tabs:

- **How it works** — the walkthrough for anyone non-technical. An animated diagram of
  the two-engine architecture, then a 7-step guided tour that follows one real customer
  from raw file → relationship score → cash-stress test → the decision gate → offer
  sizing → the handoff to the model → the final screen. Finishes with a
  **predict-the-decision quiz** on real customers from the book, including the
  counterintuitive one (a POOR file under stress is *declined*, and the walkthrough
  explains why that protects the customer).
- **Loans Desk** — live triage board over the 60-customer book, per-customer credit file
  with score breakdown and balance trend, plus a **what-if simulator** that recomputes the
  policy live in the browser.
- **ATM Ops** — triaged NOC alert queue, fleet grid coloured by cash runway, per-machine
  cassette illustration and live run-out forecast.
- **Task Ledger** — all 51 job types, searchable, each runnable live against the API with
  an editable request body.

### Run the terminal demo

```bash
python run_demo.py                # full tour — loans desk + ATM ops, LLM narration if available
python run_demo.py --no-llm       # fully offline, deterministic template narration
python run_demo.py --chapter atm  # just the ATM half (or --chapter loans)
```

Every scenario in the demo is transcribed from real `merged_finetune_train.jsonl`
/ `merged_finetune_val.jsonl` examples — the same jackpotting alert, the same
Multan NOC queue, the same Sahiwal cash run-out numbers the model was trained
on — so the computed facts and the dataset's reference answers match exactly.

### The model

Both agents are wired by default to the project's fine-tuned **Qwen3-14B**
(`atm-loans-qwen3-14b`, Q4_K_M, 40k context) served by Ollama at
`100.73.57.125:11434`. One model, two domains, two independent env-var sets:

```bash
# override either side independently; any OpenAI-compatible endpoint works
export LOAN_LLM_BASE_URL=http://<host>:11434/v1   LOAN_LLM_MODEL=<tag>
export ATM_LLM_BASE_URL=http://<host>:11434/v1    ATM_LLM_MODEL=<tag>
```

`GET /llm-status` reports live reachability (it checks `/v1/models`, so it
answers in ~0.1s and can tell "server down" apart from "wrong model tag").
The dashboard masthead shows this as a status pill.

If the model is unreachable every task falls back to a readable deterministic
narrative built from the computed facts — nothing breaks, nothing is silently
wrong, `narrative_source` just reads `"template"` instead of `"llm"`.

**Latency shape.** Generation runs 20–40s per narrative; the deterministic
verdict is always instant. Anything user-facing should therefore fetch facts
first (`use_llm: false`) and the prose second — the dashboard does this
everywhere, so ledgers render in ~100ms with the note filling in behind them.

### Use as a library

```python
from app.agents.loans.agent import LoanAgent
from agents.atm import AtmAgent, AtmMachine, Cassette
from agents.atm.cash_mechanics import CashRunoutForecastInput, forecast_cash_runout, render_cash_runout_facts

loan_agent = LoanAgent()
result = loan_agent.gold_loan(gold_offer, requested_amount=1_710_000, annual_rate=0.231, tenor_months=24)
print(result.narrative, result.narrative_source)

atm_agent = AtmAgent()
machine = AtmMachine(atm_id="LHR-01", bank="HBL", location="Gulberg",
                      cassettes=[Cassette(denom=5000, notes=200, capacity=2000)])
facts = forecast_cash_runout(machine, CashRunoutForecastInput("09:00", 180_000, 150_000, "17:00", 8))
narrative, source = atm_agent.narrate(render_cash_runout_facts(facts), "cash_runout_forecast")
```

### Run as an HTTP API

```bash
uvicorn service:app --host 0.0.0.0 --port 8080
# docs at http://localhost:8080/docs
```

```
GET  /tasks                     every loan + ATM task and its input shape
POST /loans/{task_name}         any of the 18 loan tasks — body: {..fields.., "use_llm": true}
POST /atm/tasks/{task_name}     any of the 33 ATM tasks  — body: {..fields.., "use_llm": true}

# quick-start tier, unchanged:
POST /credit/assess | /credit/batch | /credit/portfolio
POST /atm/evaluate | /atm/network | /atm/rebalance
POST /advisor/chat
```

Request bodies mirror the underlying dataclasses directly — call `GET /tasks`
to see each task's exact field names before calling it. Example:

```bash
curl -X POST localhost:8080/loans/gold_loan_sizing -d '{
  "gold": {"weight_tola": 12.8, "purity_k": 21, "rate_per_tola_24k": 244000, "ltv_pct": 0.70},
  "requested_amount": 1710000, "annual_rate": 0.231, "tenor_months": 24, "use_llm": false
}'
```

### The 51 task types

**Loans (18)** — see `../loan-agent/README.md` for the full table:
relationship_scoring, low_cash_detection, proactive_offer_decision,
decline_with_alternatives, portfolio_triage, dbr_calculation,
max_affordable_loan, emi_calculation, ecib_report_reading,
delinquency_risk_grading, restructuring_assessment, risk_based_pricing,
topup_eligibility, loan_offer_comparison, early_settlement_analysis,
gold_loan_sizing, salary_advance_assessment, product_recommendation.

**ATM — ops-facing (20)**: alert_triage, cash_carrying_cost,
cash_load_planning, cash_reconciliation, cash_runout_forecast,
cassette_status_triage, cit_route_planning, demand_forecast,
denomination_mix_planning, eod_position_report, fault_root_cause,
growth_analysis, interbank_settlement, ranking, replenishment_priority,
security_anomaly_assessment, share_analysis, surge_capacity_planning,
trend_summary, uptime_sla_check.

**ATM — customer-facing (13)**: atm_fee_calculation, atm_recommendation,
card_fraud_assessment, card_retention_guidance, cash_advance_assessment,
cash_affordability_check, daily_limit_remaining, denomination_dispensability,
failed_transaction_dispute, monthly_atm_cost_summary, spend_pattern_summary,
travel_cash_planning, withdrawal_feasibility.

### Guarantees (tested)

156 tests total: 82 in `../loan-agent/tests/` (18 loan tasks, pinned against
worked examples from the fine-tuning dataset) + 74 in `tests/` here (33 ATM
tasks, same discipline — every formula validated against real dataset
numbers, not just plausible-looking code).

```bash
cd ../loan-agent && pytest -q
cd ../simulation && pytest -q
```

---

## Note on responsible lending

The advisor is written to be transparent about markup and repayment and never to
pressure the customer, and Credit Sentinel / the full loan engine gate every
offer on repayment ability. Keep those guardrails when you connect real
customer data — it's what keeps the product defensible with regulators (SBP)
and auditors.
