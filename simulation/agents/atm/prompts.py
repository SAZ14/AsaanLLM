"""Shared prompt scaffolding for the ATM agent: the system prompt and the two
recurring block renderers ("ATM STATUS" / "MY PROFILE") every task-module
file re-uses. Task-specific fact renderers live alongside their policy
functions in each task module (cash_mechanics.py, network_ops.py, etc.).
"""
from __future__ import annotations

from .entities import AtmMachine, CustomerAtmProfile


def _rs(x: int | float) -> str:
    return f"Rs {x:,.0f}"

SYSTEM_PROMPT = (
    "You are a senior ATM operations and customer-support officer at a "
    "Pakistani bank. You read machine telemetry, cassette states, network "
    "alerts, and customer account/transaction data, then work through "
    "cash-management and customer-support decisions under the bank's "
    "internal policy. All amounts are in Pakistani rupees. Be concise, "
    "numerate, and direct."
)


def render_atm_status(m: AtmMachine, extra: str = "") -> str:
    lines = [
        "ATM STATUS",
        f"- ATM ID: {m.atm_id} ({m.bank})",
        f"- Location: {m.location}" + (f" ({m.area_type})" if m.area_type else ""),
        f"- Machine: {m.model} | Status: {m.status}" if m.model else f"- Status: {m.status}",
    ]
    if m.cassettes:
        lines.append("- Cassettes:")
        for i, c in enumerate(m.cassettes, 1):
            lines.append(
                f"  Cassette {i}: Rs {c.denom} x {c.notes:,} notes = Rs {c.value:,} "
                f"(capacity {c.capacity:,} notes)" + (f" [{c.status}: {c.note}]" if c.status != "OK" else "")
            )
        lines.append(f"- Total cash in machine: Rs {m.total_cash:,}")
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def render_customer_profile(p: CustomerAtmProfile, extra: str = "") -> str:
    lines = [
        "MY PROFILE",
        f"- Name: {p.name}",
        f"- Bank: {p.bank}",
        f"- Account: {p.account_type} | A/C {p.account_number}",
        f"- Available balance: Rs {p.available_balance:,}",
        f"- Card: {p.card_tier} ({p.card_network}) ****{p.card_last4}",
        f"- Per-transaction ATM limit: Rs {p.per_txn_limit:,}",
        f"- Daily ATM withdrawal limit: Rs {p.daily_limit:,}",
        f"- Free off-us ATM transactions left this month: {p.free_offus_left}",
    ]
    if p.credit_line:
        lines.append(f"- Credit card line: Rs {p.credit_line:,} | utilised Rs {p.credit_utilised:,}")
        lines.append(
            f"- Cash advance sub-limit: {p.cash_advance_sub_limit_pct:.0%} of credit line | "
            f"already drawn Rs {p.cash_advance_drawn:,}"
        )
    if extra:
        lines.append(extra)
    return "\n".join(lines)
