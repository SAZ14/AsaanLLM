"""Turn verification_report.json into the list you actually need before a demo:
which task types are safe to run live in front of someone, and which are not.

    python scripts/demo_safety_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "verification_report.json"

# The tasks a banking audience is most likely to ask to see.
HEADLINE = {
    "proactive_offer_decision", "decline_with_alternatives", "portfolio_triage",
    "relationship_scoring", "gold_loan_sizing", "product_recommendation",
    "ecib_report_reading", "restructuring_assessment",
    "cash_runout_forecast", "replenishment_priority", "cit_route_planning",
    "cash_carrying_cost", "security_anomaly_assessment", "card_fraud_assessment",
    "uptime_sla_check", "eod_position_report",
}

def main():
    if not REPORT.exists():
        sys.exit("run scripts/verify_sweep.py first")
    rows = json.loads(REPORT.read_text(encoding="utf-8"))
    by = {lv: [r for r in rows if r["level"] == lv] for lv in ("GREEN", "AMBER", "RED")}

    print(f"\n{'='*72}\nDEMO SAFETY — {len(rows)} task types checked against the live model\n{'='*72}")
    print(f"  GREEN {len(by['GREEN']):>2}   narration agreed with the computed facts")
    print(f"  AMBER {len(by['AMBER']):>2}   introduced a figure not in the facts (read before showing)")
    print(f"  RED   {len(by['RED']):>2}   contradicted the facts, or produced nothing\n")

    safe = [r for r in by["GREEN"] if r["task"] in HEADLINE]
    print(f"SAFE TO DEMO LIVE — headline tasks that came back green ({len(safe)})")
    for r in sorted(safe, key=lambda r: (r["domain"], r["task"])):
        print(f"   + {r['domain']}/{r['task']}")

    risky = [r for r in rows if r["level"] != "GREEN" and r["task"] in HEADLINE]
    if risky:
        print(f"\nAVOID OR REHEARSE — headline tasks that did NOT come back clean ({len(risky)})")
        for r in sorted(risky, key=lambda r: (r["level"], r["task"])):
            print(f"   ! [{r['level']}] {r['domain']}/{r['task']}")
            for i in r["issues"][:2]:
                print(f"       {i}")

    if by["RED"]:
        print(f"\nALL RED ({len(by['RED'])}) — do not show these without a fix")
        for r in by["RED"]:
            print(f"   x {r['domain']}/{r['task']}: {'; '.join(r['issues'][:2])}")

    if by["AMBER"]:
        print(f"\nALL AMBER ({len(by['AMBER'])}) — usually a derived figure; verify before relying on it")
        for r in by["AMBER"]:
            print(f"   ~ {r['domain']}/{r['task']}: {'; '.join(r['issues'][:1])}")

    print()

if __name__ == "__main__":
    main()
