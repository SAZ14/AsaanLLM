"""Shared ATM-domain entities used across every task module in this package.

Mirrors loan-agent's models.py convention: plain dataclasses, no third-party
dependencies, field names chosen to match merged_finetune_train.jsonl's
"ATM STATUS" / "MY PROFILE" blocks so prompt renderers stay a near-literal
transcription of these objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Cassette:
    denom: int
    notes: int
    capacity: int
    status: str = "OK"     # OK | LOW | FAULT
    note: str = ""

    @property
    def value(self) -> int:
        return self.denom * self.notes


@dataclass
class AtmMachine:
    atm_id: str
    bank: str
    location: str
    area_type: str = ""
    model: str = ""
    status: str = "ONLINE"   # ONLINE | OFFLINE | COMMS_DOWN | OUT_OF_CASH
    cassettes: list[Cassette] = field(default_factory=list)

    @property
    def total_cash(self) -> int:
        return sum(c.value for c in self.cassettes)

    @property
    def usable_cash(self) -> int:
        """Cash in cassettes that aren't FAULT-locked out."""
        return sum(c.value for c in self.cassettes if c.status != "FAULT")

    @property
    def usable_denominations(self) -> list[int]:
        """Distinct denominations available for dispensing, largest first."""
        denoms = {c.denom for c in self.cassettes if c.status not in ("FAULT",) and c.notes > 0}
        return sorted(denoms, reverse=True)


@dataclass
class CustomerAtmProfile:
    name: str
    bank: str
    account_type: str = ""
    account_number: str = ""
    available_balance: int = 0
    card_network: str = ""
    card_tier: str = ""
    card_last4: str = ""
    per_txn_limit: int = 0
    daily_limit: int = 0
    free_offus_left: int = 0
    credit_line: int = 0
    credit_utilised: int = 0
    cash_advance_sub_limit_pct: float = 0.0
    cash_advance_drawn: int = 0


@dataclass
class Transaction:
    date: str
    time: str
    description: str
    channel: str          # "ATM" | "POS" | "e-commerce" | ...
    amount: int            # negative = debit/outflow, positive = credit/inflow
