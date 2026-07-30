"""HTTP API layer over the agents — drop this behind your gateway / frontend.

    pip install fastapi uvicorn
    uvicorn service:app --host 0.0.0.0 --port 8080
    # docs at http://localhost:8080/docs

The full engine is wired to the fine-tuned atm-loans-qwen3-14b by default
(see app/agents/loans/llm.py and agents/atm/llm.py); it falls back to
deterministic narration whenever the model is unreachable. The quick-start
advisor has its own knobs:
    MODEL=<tag> OLLAMA_URL=http://<host>:11434/api/chat uvicorn service:app --port 8080

Two tiers of endpoints:

  Quick-start (zero deps beyond fastapi/uvicorn, hand-rolled scoring):
    POST /credit/assess | /credit/batch | /credit/portfolio
    POST /atm/evaluate | /atm/network | /atm/rebalance
    POST /advisor/chat

  Full engine — all 51 task types from merged_finetune_train.jsonl, backed by
  loan-agent's deterministic policy engine + the new ATM agent, narrated by
  the fine-tuned local model (see app/agents/loans/llm.py, agents/atm/llm.py):
    GET  /tasks                     list every loan/atm task + its input shape
    POST /loans/{task_name}         body: {..task-specific fields.., "use_llm": true}
    POST /atm/tasks/{task_name}     body: {..task-specific fields.., "use_llm": true}
"""
import os
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any

from agents import CreditSentinel, CashPulse, Advisor, ollama_llm
from agents.atm import AtmAgent, TASK_REGISTRY as ATM_TASK_REGISTRY
from agents.atm import cash_mechanics, network_ops, customer_money, judgment
from agents.atm import llm as atm_llm
from agents.atm.prompts import render_atm_status, render_customer_profile

from app.agents.loans.agent import LoanAgent
from app.agents.loans import llm as loan_llm
from app.agents.loans.policy import decide, detect_stress, relationship_score

from dispatch import MissingField, build_args, call_render, describe_params, to_jsonable

app = FastAPI(title="Agentic Banking API", version="2.0")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/dashboard")
def dashboard():
    return RedirectResponse(url="/static/dashboard.html")

credit = CreditSentinel()
cash = CashPulse()
advisor = Advisor(llm=ollama_llm(
    model=os.environ.get("MODEL", "atm-loans-qwen3-14b"),
    url=os.environ.get("OLLAMA_URL", "http://100.73.57.125:11434/api/chat"),
))

_loan_agent = LoanAgent()
_atm_agent = AtmAgent()

ATM_MODULES = {
    "cash_mechanics": cash_mechanics,
    "network_ops": network_ops,
    "customer_money": customer_money,
    "judgment": judgment,
}

# task_name -> LoanAgent method name. proactive_offer_decision and
# decline_with_alternatives are the same policy gate (decide()); the method
# just narrates whichever action it lands on.
LOAN_TASK_METHODS = {
    "relationship_scoring": "relationship_scoring",
    "low_cash_detection": "low_cash_detection",
    "proactive_offer_decision": "assess",
    "decline_with_alternatives": "assess",
    "portfolio_triage": "portfolio_triage",
    "dbr_calculation": "dbr_calculation",
    "max_affordable_loan": "max_affordable_loan",
    "emi_calculation": "emi_calculation",
    "ecib_report_reading": "ecib_report",
    "delinquency_risk_grading": "delinquency_risk",
    "restructuring_assessment": "restructuring",
    "risk_based_pricing": "risk_pricing",
    "topup_eligibility": "topup",
    "loan_offer_comparison": "offer_comparison",
    "early_settlement_analysis": "early_settlement",
    "gold_loan_sizing": "gold_loan",
    "salary_advance_assessment": "salary_advance",
    "product_recommendation": "product_recommendation",
}


class CustomerReq(BaseModel):
    customer: dict[str, Any]

class CustomersReq(BaseModel):
    customers: list[dict[str, Any]]

class AtmReq(BaseModel):
    atm: dict[str, Any]

class AtmsReq(BaseModel):
    atms: list[dict[str, Any]]

class ChatReq(BaseModel):
    customer: dict[str, Any]
    messages: list[dict[str, str]] = []


# ── quick-start tier (unchanged) ──

@app.post("/credit/assess")
def credit_assess(r: CustomerReq):
    return credit.assess(r.customer)

@app.post("/credit/batch")
def credit_batch(r: CustomersReq):
    return credit.assess_batch(r.customers)

@app.post("/credit/portfolio")
def credit_portfolio(r: CustomersReq):
    return credit.portfolio_summary(r.customers)

@app.post("/atm/evaluate")
def atm_evaluate(r: AtmReq):
    return cash.evaluate(r.atm)

@app.post("/atm/network")
def atm_network(r: AtmsReq):
    return {"summary": cash.network_summary(r.atms), "machines": cash.evaluate_all(r.atms)}

@app.post("/atm/rebalance")
def atm_rebalance(r: AtmsReq):
    return {"moves": cash.rebalance_plan(r.atms)}

@app.post("/advisor/chat")
def advisor_chat(r: ChatReq):
    reply, source = advisor.reply(r.customer, r.messages)
    return {"reply": reply, "source": source}


# ── full engine: every loan/ATM task type from the fine-tuning dataset ──

@app.get("/tasks")
def list_tasks():
    loan_tasks = {}
    for name, method_name in LOAN_TASK_METHODS.items():
        loan_tasks[name] = describe_params(getattr(_loan_agent, method_name))
    atm_tasks = {}
    for name, (mod_name, fn_name, _) in ATM_TASK_REGISTRY.items():
        atm_tasks[name] = describe_params(getattr(ATM_MODULES[mod_name], fn_name))
    return {"loans": loan_tasks, "atm": atm_tasks}


@app.post("/loans/{task_name}")
def loans_task(task_name: str, body: dict = Body(default={})):
    if task_name not in LOAN_TASK_METHODS:
        raise HTTPException(404, f"Unknown loan task: {task_name}. See GET /tasks.")
    method = getattr(_loan_agent, LOAN_TASK_METHODS[task_name])
    try:
        kwargs = build_args(method, body)
    except MissingField as exc:
        raise HTTPException(422, str(exc))
    except (TypeError, KeyError, AttributeError) as exc:
        raise HTTPException(422, f"malformed request body: {exc}")
    try:
        result = method(**kwargs)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:  # noqa: BLE001 — any bad-input failure -> clean 400, never a raw 500
        raise HTTPException(400, f"{type(exc).__name__}: {exc}")
    return to_jsonable(result)


@app.post("/atm/tasks/{task_name}")
def atm_task(task_name: str, body: dict = Body(default={})):
    if task_name not in ATM_TASK_REGISTRY:
        raise HTTPException(404, f"Unknown ATM task: {task_name}. See GET /tasks.")
    mod_name, fn_name, render_name = ATM_TASK_REGISTRY[task_name]
    module = ATM_MODULES[mod_name]
    compute_fn = getattr(module, fn_name)
    render_fn = getattr(module, render_name)
    use_llm = bool(body.get("use_llm", True))

    try:
        kwargs = build_args(compute_fn, body)
    except MissingField as exc:
        raise HTTPException(422, str(exc))
    except (TypeError, KeyError, AttributeError) as exc:
        raise HTTPException(422, f"malformed request body: {exc}")
    try:
        facts = compute_fn(**kwargs)
        prompt = call_render(render_fn, kwargs, facts)
    except Exception as exc:  # noqa: BLE001 — any bad-input failure -> clean 400, never a raw 500
        raise HTTPException(400, f"{type(exc).__name__}: {exc}")

    narrative, source = _atm_agent.narrate(prompt, task_name, use_llm)
    return {"facts": to_jsonable(facts), "narrative": narrative, "narrative_source": source}


@app.get("/loans/book")
def loans_book():
    """The full customer roster with score/band/decision — for the dashboard's
    triage board. Instant (no LLM call); narratives are fetched per-customer
    on demand via POST /loans/proactive_offer_decision."""
    rows = []
    for c in _loan_agent.customers:
        score = relationship_score(c)
        stress = detect_stress(c)
        d = decide(c, _loan_agent.product)
        rows.append({
            "customer_id": c.customer_id, "name": c.name, "bank": c.bank_code, "city": c.city,
            "net_monthly_income": c.net_monthly_income, "score": score.score, "band": score.band,
            "stressed": stress.stressed, "days_below_5k": stress.days_below_5k,
            "action": d.action, "offer_amount": d.offer.amount if d.offer else None,
        })
    return {"customers": rows}


@app.get("/loans/customers/{customer_id}")
def loans_customer_detail(customer_id: str):
    """Full profile + score breakdown + stress signals for one customer —
    everything the dashboard's detail drawer needs besides the narrative."""
    try:
        c = _loan_agent.get(customer_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    return {
        "profile": to_jsonable(c),
        "score": to_jsonable(relationship_score(c)),
        "stress": to_jsonable(detect_stress(c)),
    }


@app.get("/llm-status")
def llm_status():
    """Is the fine-tuned model reachable right now? Powers the dashboard's
    masthead status pill.

    Uses the OpenAI-compatible /v1/models listing rather than a generation
    call: it answers in ~0.1s instead of seconds, and it also tells us
    whether the specific tag we intend to call is actually loaded — a
    generation probe can't distinguish "server down" from "wrong tag"."""
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI

    def probe(base_url_fn, api_key_env, model_fn):
        model = model_fn()
        try:
            client = OpenAI(
                api_key=os.environ.get(api_key_env, "ollama"),
                base_url=base_url_fn(), timeout=4.0, max_retries=0,
            )
            available = [m.id for m in client.models.list().data]
            # Ollama reports tags as "name:latest"; accept a bare-name match.
            loaded = any(mid == model or mid.split(":")[0] == model.split(":")[0] for mid in available)
            return {"connected": loaded, "model": model,
                    "detail": None if loaded else f"server up, but '{model}' not among {available}"}
        except Exception as exc:
            return {"connected": False, "model": model, "detail": f"unreachable: {type(exc).__name__}"}

    with ThreadPoolExecutor(max_workers=2) as pool:
        loans_future = pool.submit(probe, loan_llm.get_base_url, "LOAN_LLM_API_KEY", loan_llm.get_model)
        atm_future = pool.submit(probe, atm_llm.get_base_url, "ATM_LLM_API_KEY", atm_llm.get_model)
        return {"loans": loans_future.result(), "atm": atm_future.result()}


@app.get("/health")
def health():
    return {"ok": True}
