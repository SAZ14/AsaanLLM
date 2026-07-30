"""Local LLM client for the loans agent.

Points at any OpenAI-compatible server — Ollama by default. This is the same
fine-tuned model used by the ATM agent (simulation/agents/atm/llm.py mirrors
this client): one model, two domains, trained on merged_finetune_train.jsonl.

Defaults point at the project's fine-tuned Qwen3-14B (atm-loans-qwen3-14b)
on the GPU box. Override any of these with env vars to point elsewhere —
no code changes needed.

Few-shot defaults to 0 because this model is *already* fine-tuned on this
exact prompt format; prepending examples measurably slows generation
(~42s vs ~31s) without improving output. Set LOAN_FEWSHOT=4 only when
pointing at a base (non-fine-tuned) model.

Env:
  LOAN_LLM_BASE_URL  default http://100.73.57.125:11434/v1
  LOAN_LLM_MODEL     default atm-loans-qwen3-14b
  LOAN_LLM_API_KEY   default "ollama" (Ollama ignores it)
  LOAN_LLM_TIMEOUT   seconds before giving up on a generation (default 120)
  LOAN_FEWSHOT       few-shot examples to prepend (default 0)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

FEWSHOT_PATH = Path(__file__).resolve().parents[3] / "data" / "loans" / "fewshot.jsonl"

_client = None


def get_base_url() -> str:
    return os.environ.get("LOAN_LLM_BASE_URL", "http://100.73.57.125:11434/v1")


def get_model() -> str:
    return os.environ.get("LOAN_LLM_MODEL", "atm-loans-qwen3-14b")


def get_timeout() -> float:
    try:
        return float(os.environ.get("LOAN_LLM_TIMEOUT", "120"))
    except ValueError:
        return 120.0


def fewshot_count() -> int:
    try:
        return int(os.environ.get("LOAN_FEWSHOT", "0"))
    except ValueError:
        return 0


def get_client():
    """Cached OpenAI-compatible client for the model."""
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(
            api_key=os.environ.get("LOAN_LLM_API_KEY", "ollama"),
            base_url=get_base_url(),
            timeout=get_timeout(),
            max_retries=1,
        )
    return _client


def load_fewshot(tasks: list[str] | None = None, limit: int | None = None) -> list[dict]:
    """Few-shot messages drawn from the curated dataset sample, optionally
    filtered to specific task types."""
    limit = fewshot_count() if limit is None else limit
    if limit <= 0 or not FEWSHOT_PATH.exists():
        return []
    messages: list[dict] = []
    taken = 0
    for line in FEWSHOT_PATH.read_text().splitlines():
        if taken >= limit:
            break
        row = json.loads(line)
        if tasks and row.get("meta", {}).get("task") not in tasks:
            continue
        messages.extend(row["messages"])
        taken += 1
    return messages


def chat(
    user_content: str,
    system: str,
    fewshot_tasks: list[str] | None = None,
    client=None,
    temperature: float = 0.3,
) -> str | None:
    """One chat completion against the local model. Returns None on any
    failure (server down, model missing) so callers can fall back to the
    deterministic narrative."""
    try:
        client = client or get_client()
        messages = [{"role": "system", "content": system}]
        messages += load_fewshot(tasks=fewshot_tasks)
        messages.append({"role": "user", "content": user_content})
        resp = client.chat.completions.create(
            model=get_model(), messages=messages, temperature=temperature,
        )
        return resp.choices[0].message.content
    except Exception as exc:  # noqa: BLE001 — any local-server failure → fallback
        logger.warning("Local LLM unavailable (%s) — using deterministic narrative", exc)
        return None
