"""
Advisor — the customer-facing chat agent.

Turns a Credit Sentinel decision into a conversation. You inject the LLM as a
callable, so this works with ANY local model (Ollama, LM Studio, llama.cpp,
vLLM, an OpenAI-compatible endpoint) or your own pipeline. If the LLM call
fails or you pass none, a deterministic rule-based advisor answers instead.

    from agents.advisor import Advisor, ollama_llm
    advisor = Advisor(llm=ollama_llm())      # defaults to the fine-tuned model
    reply, source = advisor.reply(customer, messages)

`customer` should carry the fields the decision produced, e.g.:
    {"name","area","income","balance","repayment_pct","decision","offer","markup_pct"}
`messages` is the running chat: [{"role":"assistant"|"user","content": "..."}]
"""
from __future__ import annotations
import re
import json
import urllib.request
from typing import Callable, Optional


def _money(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "the offered amount"
    if n >= 1e7:
        return f"PKR {n/1e7:.2f} Cr"
    if n >= 1e5:
        return f"PKR {n/1e5:.1f} L"
    return f"PKR {round(n):,}"


def build_system_prompt(c: dict) -> str:
    first = str(c.get("name", "the customer")).split(" ")[0]
    offer = c.get("offer") or 0
    markup = c.get("markup_pct", c.get("markup"))
    if offer:
        offer_line = (f"They are PRE-APPROVED for an instant loan of {_money(offer)} at a markup of "
                      f"{markup}% per year, repayable over 3-12 months, disbursed within minutes.")
    else:
        offer_line = ("They currently have NO active loan offer. Do NOT invent or promise a loan; "
                      "help only with general account questions.")
    return (
        f"You are the bank's friendly AI relationship assistant chatting with a customer named "
        f"{first} inside the mobile banking app in Lahore, Pakistan. "
        f"Profile: monthly income {_money(c.get('income'))}, current balance {_money(c.get('balance'))}, "
        f"repayment reliability {c.get('repayment_pct', c.get('repay'))}%, decision "
        f"'{c.get('decision')}'. {offer_line} "
        "Style: warm, concise (1-3 short sentences), WhatsApp-like; a little natural Lahore "
        "Urdu-English is fine. Always be transparent about markup and repayment. Never pressure "
        "the customer, never ask for their PIN or OTP, never make up numbers beyond the offer above."
    )


def fallback_reply(c: dict, messages: list[dict]) -> str:
    first = str(c.get("name", "there")).split(" ")[0]
    offer = c.get("offer") or 0
    markup = c.get("markup_pct", c.get("markup")) or 0
    q = next((m.get("content", "").lower() for m in reversed(messages) if m.get("role") == "user"), "")
    if not offer:
        return (f"Right now there's no active loan offer on your profile, {first}. "
                "I can still help with your balance, cards or account questions.")
    if re.search(r"(interest|markup|rate|sood|profit|percent|%)", q):
        return (f"The markup is {markup}% per year, {first} — charged only on your outstanding balance. "
                f"On {_money(offer)} over 6 months that's roughly {_money(offer*markup/100/2)} total. No hidden charges.")
    if re.search(r"(how much|amount|kitna|kitne|limit|maximum)", q):
        return (f"You're pre-approved for up to {_money(offer)}, {first}. Take the full amount or less — "
                "markup applies only to what you actually use.")
    if re.search(r"(time|tenure|month|repay|return|install|kist|wapas|duration)", q):
        return (f"Repayment is flexible, 3 to 12 months. For {_money(offer)} over 6 months your monthly "
                f"instalment is about {_money(offer/6*(1+markup/100/2))}. Prepay anytime, no penalty.")
    if re.search(r"(yes|haan|interested|okay|chahiye|want|apply|proceed|karo)", q):
        return (f"Great, {first}! Open your app -> tap 'Instant Loan' -> confirm {_money(offer)} -> "
                "funds arrive in ~2 minutes. Shall I send the confirmation link?")
    if re.search(r"(no|nahi|not|later|maybe|don'?t)", q):
        return f"No problem at all, {first}. The offer stays pre-approved for 30 days — reach out anytime."
    if re.search(r"(safe|secure|scam|real|genuine|trust|fraud)", q):
        return (f"Totally fair to ask, {first}. This is an official pre-approved offer inside your app "
                "under 'My Offers'. We'll never ask for your PIN or OTP.")
    return (f"Sure {first} — your pre-approved instant loan is {_money(offer)} at {markup}% p.a., repayable "
            "over 3-12 months. Ask me about the amount, markup, or monthly instalment.")


# ---- LLM adapters: each returns a callable(messages) -> str ----
def ollama_llm(model: str = "atm-loans-qwen3-14b",
               url: str = "http://100.73.57.125:11434/api/chat",
               temperature: float = 0.7, timeout: int = 120) -> Callable[[list[dict]], str]:
    def _call(messages: list[dict]) -> str:
        payload = {"model": model, "stream": False, "messages": messages,
                   "options": {"temperature": temperature}}
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode())
        reply = (out.get("message", {}) or {}).get("content", "")
        if not reply and out.get("choices"):                 # OpenAI-compatible shape
            reply = out["choices"][0].get("message", {}).get("content", "")
        return reply.strip()
    return _call


def openai_compatible_llm(model: str, url: str, api_key: str = "",
                          temperature: float = 0.7, timeout: int = 45) -> Callable[[list[dict]], str]:
    """For LM Studio / vLLM / any /v1/chat/completions endpoint."""
    def _call(messages: list[dict]) -> str:
        payload = {"model": model, "messages": messages, "temperature": temperature}
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode())
        return out["choices"][0]["message"]["content"].strip()
    return _call


class Advisor:
    def __init__(self, llm: Optional[Callable[[list[dict]], str]] = None):
        self.llm = llm

    def reply(self, customer: dict, messages: list[dict]) -> tuple[str, str]:
        """Return (reply_text, source). Falls back to rules if the LLM fails."""
        if self.llm is not None:
            try:
                full = [{"role": "system", "content": build_system_prompt(customer)}] + messages
                text = self.llm(full)
                if text:
                    return text, "local-llm"
            except Exception:
                pass
        return fallback_reply(customer, messages), "fallback"

    def opening_offer(self, customer: dict) -> list[str]:
        """The proactive messages to auto-send when a customer is flagged 'instant'."""
        first = str(customer.get("name", "there")).split(" ")[0]
        offer = customer.get("offer") or 0
        markup = customer.get("markup_pct", customer.get("markup"))
        if not offer:
            return []
        return [
            f"Assalam o Alaikum {first} \U0001F44B This is your bank's AI Instant Support.",
            (f"We noticed your balance is running low this month. As a valued customer with an "
             f"excellent payment record, you're pre-approved for an instant loan of {_money(offer)}."),
            (f"Markup is {markup}% p.a., flexible 3-12 month repayment, disbursed in minutes. "
             "Want the details?"),
        ]
