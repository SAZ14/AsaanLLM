# Loan Agent

Scans a bank's customer book, grades every relationship, detects genuine
cash stress, and decides **who gets a proactive instant-loan offer, who is
monitored, and who is declined** — with secured alternatives so a declined
customer still leaves with a workable path. Covers all **18 loan task types**
trained into the fine-tuning dataset — from relationship scoring and EMI
math up through eCIB reading, restructuring, gold-loan sizing, and product
recommendation.

Pure-Python core (import it anywhere) plus a CLI demo. No server, no
database, no external services required.

This package is consumed as a local dependency by `../simulation/`, which
pairs it with a matching **ATM agent** (33 more task types, same fine-tuned
model, same policy-computes/LLM-narrates design) behind a FastAPI service
and a narrative demo — see `../simulation/README.md`.

---

## Design: the policy decides, the model narrates

```
customer_book.json ──► policy.py (deterministic)         llm.py (local model)
                        │ relationship score + band        │
                        │ cash-stress detection             │
                        │ EMI / DBR / affordability          │
                        │ DECISION GATE                     │
                        ▼                                  ▼
                     Decision ──────────────────────► narrative / free-form Q&A
```

Every number and every OFFER/MONITOR/DECLINE decision is computed in
`app/agents/loans/policy.py`. The LLM receives the policy verdict as ground
truth and only explains it — a hallucinating model can never approve a file
the policy declines.

### The policy (mirrors the fine-tuning dataset)

**Relationship score** — account age, salary regularity, average balance, and
repayment history earn up to +8; a cheque bounce (−3), eCIB 90+ DPD (−4), or
write-off/litigation flag (−6) pull it down.
Bands: `>=6 STRONG | 3-5 ACCEPTABLE | 0-2 THIN | <0 POOR`.

**Cash stress** — 10+ days under Rs 5,000 in the last 30 is the genuine-stress
marker (plus balance-trajectory context).

**Decision gate**
- **OFFER**: STRONG/ACCEPTABLE **and** genuinely stressed — sized under the
  40% DBR cap, floored to Rs 10,000, clamped to the product band.
- **MONITOR**: THIN files; liquid STRONG/ACCEPTABLE files (keep pre-approved);
  stressed files with no instalment headroom or income under the product floor.
- **DECLINE**: POOR files get no unsecured credit — stress makes unsecured
  lending to a bad file *more* dangerous, not less. They are redirected to
  deposit-backed / gold-backed finance, a secured card to rebuild eCIB, and
  the eCIB correction process where the flag is disputed.

---

## Install

```bash
pip install -e ".[test]"
```

## Run the demo

```bash
python scripts/loan_demo.py                  # full-book triage queues
python scripts/loan_demo.py --walkthrough    # + narrated deep-dives
python scripts/loan_demo.py --customer C007  # one file in detail
python scripts/loan_demo.py --no-llm         # fully offline (template narratives)
```

## Use as a library

```python
from app.agents.loans import LoanAgent

agent = LoanAgent()                 # loads data/loans/customer_book.json
assessment = agent.assess("C007")   # -> Assessment(customer, decision, narrative, narrative_source)
queues = agent.scan()               # -> {"OFFER": [...], "MONITOR": [...], "DECLINE": [...]}
answer = agent.ask("What's the DBR cap on this product?")
```

The deterministic core (`policy.py`, `models.py`, `book.py`) has zero
third-party dependencies — only the local-LLM narrator (`llm.py`) needs the
`openai` package, and only to talk to an OpenAI-compatible endpoint.

### All 18 task types

Every task has a narrated `LoanAgent` method (`agent.<method>(..., use_llm=True)`,
returning `TaskResult(facts, narrative, narrative_source)`):

| Task type | Method |
|---|---|
| relationship_scoring | `relationship_scoring(customer_id)` |
| low_cash_detection | `low_cash_detection(customer_id)` |
| proactive_offer_decision / decline_with_alternatives | `assess(customer_id)` |
| portfolio_triage | `portfolio_triage()` / `scan()` |
| dbr_calculation | `dbr_calculation(net_income, existing_obligations, new_emi)` |
| max_affordable_loan | `max_affordable_loan(customer_id, product?, tenor?)` |
| emi_calculation | `emi_calculation(principal, annual_rate, tenor_months, rate_type)` |
| ecib_report_reading | `ecib_report(lines)` |
| delinquency_risk_grading | `delinquency_risk(customer_id, loan, late_delays)` |
| restructuring_assessment | `restructuring(customer_id, loan, new_income, extension_months)` |
| risk_based_pricing | `risk_pricing(customer_id, base_rate, acceptable_markup_pts, thin_markup_pts, requested_amount, tenor_months)` |
| topup_eligibility | `topup(customer_id, loan, late_in_12m, request_amount)` |
| loan_offer_comparison | `offer_comparison(principal, tenor, quotes)` |
| early_settlement_analysis | `early_settlement(loan, settlement_fee_pct)` |
| gold_loan_sizing | `gold_loan(gold, requested_amount, annual_rate, tenor_months)` |
| salary_advance_assessment | `salary_advance(customer_id, product, requested_amount)` |
| product_recommendation | `product_recommendation(customer_id, need_amount, shelf, ...)` |

---

## The model

Wired by default to the project's fine-tuned **Qwen3-14B** (`atm-loans-qwen3-14b`,
Q4_K_M) served by Ollama on the GPU box. Any OpenAI-compatible endpoint works.
If the model is unreachable the agent degrades to readable deterministic
narratives — nothing breaks, `narrative_source` just reads `"template"`.

```
LOAN_LLM_BASE_URL   default http://100.73.57.125:11434/v1
LOAN_LLM_MODEL      default atm-loans-qwen3-14b
LOAN_LLM_TIMEOUT    default 120 (seconds)
LOAN_FEWSHOT        default 0
```

`LOAN_FEWSHOT` defaults to **0** because this model is already fine-tuned on
this exact prompt format — prepending examples from `data/loans/fewshot.jsonl`
measurably slows generation (~42s vs ~31s) with no gain. Set it to 4 only when
pointing at a base, non-fine-tuned model.

Typical generation is **20–40s** per narrative. The deterministic verdict is
always instant, so callers that need to stay responsive should request the
facts first (`use_llm=False`) and fetch the prose separately — that is what
the dashboard does.

## Fine-tuning

The production model backing this agent (and its sibling ATM agent in
`../simulation/`) was fine-tuned on `merged_finetune_train.jsonl` — 18,536
examples spanning all 18 loan + 33 ATM task types, half English half Roman
Urdu. `data/loans/loans_pakistan_finetune_10k.jsonl` in this repo is the
original loans-only precursor dataset (17 task types) used by
`notebooks/finetune_loans_qlora_colab.ipynb` (Unsloth QLoRA, free Colab T4,
~1–2 h) — the same notebook process applies to the merged dataset, just
point `DATA_PATH` at it. Export GGUF, `ollama create <your-tag>`, then:

```bash
export LOAN_LLM_MODEL=<your-tag> LOAN_FEWSHOT=0
```

The agent's prompts (`app/agents/loans/prompts.py`) byte-match the dataset's
format, so the fine-tuned model is a drop-in.

---

## Guarantees (tested)

82 tests across `tests/test_loan_agent.py`, `tests/test_loan_edge_cases.py`,
and `tests/test_loan_tasks_extended.py` (the 10 newer task types, pinned
against worked examples from `merged_finetune_train.jsonl`). The original
suite pins:
- every scoring threshold on both sides of its boundary (3.0 years, 11/12
  months, Rs 50,000, score exactly 0/3/6, 10 stress days)
- EMI/DBR/affordability math against worked examples from the dataset,
  including zero-rate and round-trip inversion; off-shelf tenors rejected
- a 500-profile randomized sweep: POOR is never offered unsecured credit,
  the 40% DBR cap is never breached, offers never leave the product band,
  every decline carries alternatives
- LLM output sanitisation: truncated `<think>` scratchpads never leak;
  an empty model reply falls back to the deterministic narrative
- corrupt book rows (15/12 salary months, negative stress days, unknown
  loan history, duplicate customer ids) are rejected at the loader boundary

Run with:

```bash
pytest
```

## Synthetic customer book

`scripts/generate_loan_book.py` generates a seeded, reproducible 60-customer
book covering every policy branch (strong+stressed, liquid, thin, poor).
All customers, banks' pairings, and balances are synthetic.

```bash
python scripts/generate_loan_book.py --n 100 --seed 47
```

---

## Project structure

```
app/agents/loans/
  agent.py      — LoanAgent: assess one customer, scan the whole book, free-form Q&A
  book.py       — load/save the customer book (JSON), row validation
  llm.py        — local OpenAI-compatible LLM client (Ollama by default)
  models.py     — CustomerProfile, LoanProduct, Decision, and related dataclasses
  policy.py     — the deterministic scoring/stress/EMI/decision engine
  prompts.py    — prompt rendering, byte-matched to the fine-tuning dataset format

data/loans/     — customer book, few-shot examples, fine-tuning dataset
notebooks/      — QLoRA fine-tuning notebook
scripts/
  generate_loan_book.py  — synthetic customer book generator
  loan_demo.py            — CLI demo (triage, walkthroughs, single-customer)
tests/          — policy math, edge cases, sanitisation, loader validation
```
