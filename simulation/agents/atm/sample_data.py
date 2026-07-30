"""Parsers that pull real ATM/customer scenarios out of the fine-tuning
dataset's own prompt text (the "ATM STATUS" / "MY PROFILE" blocks and
markdown tables used throughout merged_finetune_train.jsonl /
merged_finetune_val.jsonl), so demo scenarios are drawn from real training
data rather than hand-invented fixtures.

These are text-scraping conveniences for building demo scenarios, not part
of the deterministic policy layer — keep them isolated here.
"""
from __future__ import annotations

import re

from .entities import AtmMachine, Cassette, CustomerAtmProfile

_ATM_ID_RE = re.compile(r"-\s*ATM ID:\s*([\w-]+)\s*\(([^)]+)\)")
_LOCATION_RE = re.compile(r"-\s*Location:\s*(.+)")
_MACHINE_RE = re.compile(r"-\s*Machine:\s*([^|]+?)\s*\|\s*Status:\s*(\w+)")
_CASSETTE_RE = re.compile(
    r"Cassette\s+\d+:\s*Rs\s*([\d,]+)\s*x\s*([\d,]+)\s*notes\s*=\s*Rs\s*[\d,]+\s*"
    r"\(capacity\s*([\d,]+)\s*notes\)(?:\s*\[(\w+):?\s*([^\]]*)\])?"
)


def _num(s: str) -> int:
    return int(s.replace(",", "").strip())


def parse_atm_status_block(text: str) -> AtmMachine:
    """Parse an "ATM STATUS" block (as emitted by prompts.render_atm_status,
    or as it appears verbatim in the training data) into an AtmMachine."""
    atm_id, bank = "UNKNOWN", ""
    m = _ATM_ID_RE.search(text)
    if m:
        atm_id, bank = m.group(1), m.group(2)

    location, area_type = "", ""
    m = _LOCATION_RE.search(text)
    if m:
        full = m.group(1).strip()
        if "(" in full:
            idx = full.index("(")
            location = full[:idx].strip().rstrip(",").strip()
            area_type = full[idx + 1:].strip()
            if area_type.endswith(")"):
                area_type = area_type[:-1]
        else:
            location = full

    model, status = "", "ONLINE"
    m = _MACHINE_RE.search(text)
    if m:
        model, status = m.group(1).strip(), m.group(2).strip()

    cassettes = []
    for cm in _CASSETTE_RE.finditer(text):
        denom = _num(cm.group(1))
        notes = _num(cm.group(2))
        capacity = _num(cm.group(3))
        c_status = (cm.group(4) or "OK").upper()
        note = cm.group(5) or ""
        cassettes.append(Cassette(denom=denom, notes=notes, capacity=capacity, status=c_status, note=note))

    return AtmMachine(
        atm_id=atm_id, bank=bank, location=location, area_type=area_type,
        model=model, status=status, cassettes=cassettes,
    )


_NAME_RE = re.compile(r"-\s*Name:\s*(.+)")
_BANK_RE = re.compile(r"-\s*Bank:\s*(.+)")
_ACCOUNT_RE = re.compile(r"-\s*Account:\s*(.+?)\s*\|\s*A/C\s*(\S+)")
_BALANCE_RE = re.compile(r"-\s*Available balance:\s*Rs\s*([\d,]+)")
_CARD_RE = re.compile(r"-\s*Card:\s*(.+?)\s*\(([^)]+)\)\s*\*+(\w+)")
_PER_TXN_RE = re.compile(r"-\s*Per-transaction ATM limit:\s*Rs\s*([\d,]+)")
_DAILY_RE = re.compile(r"-\s*Daily ATM withdrawal limit:\s*Rs\s*([\d,]+)")
_FREE_OFFUS_RE = re.compile(r"-\s*Free off-us ATM transactions left this month:\s*(\d+)")
_CREDIT_LINE_RE = re.compile(r"-\s*Credit card line:\s*Rs\s*([\d,]+)\s*\|\s*utili[sz]ed\s*Rs\s*([\d,]+)")
_CASH_ADV_RE = re.compile(
    r"-\s*Cash advance sub-limit:\s*([\d.]+)%\s*of credit line\s*\|\s*already drawn\s*Rs\s*([\d,]+)"
)


def parse_customer_profile_block(text: str) -> CustomerAtmProfile:
    """Parse a "MY PROFILE" block into a CustomerAtmProfile."""
    def g(rx, default=""):
        m = rx.search(text)
        return m.group(1).strip() if m else default

    name = g(_NAME_RE)
    bank = g(_BANK_RE)

    account_type, account_number = "", ""
    m = _ACCOUNT_RE.search(text)
    if m:
        account_type, account_number = m.group(1).strip(), m.group(2).strip()

    balance = _num(g(_BALANCE_RE, "0"))

    card_tier, card_network, card_last4 = "", "", ""
    m = _CARD_RE.search(text)
    if m:
        card_tier, card_network, card_last4 = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()

    per_txn = _num(g(_PER_TXN_RE, "0"))
    daily = _num(g(_DAILY_RE, "0"))
    free_offus = int(g(_FREE_OFFUS_RE, "0") or 0)

    credit_line, credit_utilised = 0, 0
    m = _CREDIT_LINE_RE.search(text)
    if m:
        credit_line, credit_utilised = _num(m.group(1)), _num(m.group(2))

    sub_limit_pct, drawn = 0.0, 0
    m = _CASH_ADV_RE.search(text)
    if m:
        sub_limit_pct, drawn = float(m.group(1)) / 100, _num(m.group(2))

    return CustomerAtmProfile(
        name=name, bank=bank, account_type=account_type, account_number=account_number,
        available_balance=balance, card_network=card_network, card_tier=card_tier,
        card_last4=card_last4, per_txn_limit=per_txn, daily_limit=daily,
        free_offus_left=free_offus, credit_line=credit_line, credit_utilised=credit_utilised,
        cash_advance_sub_limit_pct=sub_limit_pct, cash_advance_drawn=drawn,
    )


def parse_markdown_table(text: str) -> list[dict]:
    """Parse the first markdown pipe-table found in text into a list of
    {header: value} dicts, values left as raw strings (caller converts)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return []
    header = [h.strip() for h in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:  # skip the |---|---| separator row
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells)))
    return rows
