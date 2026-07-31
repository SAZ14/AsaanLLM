"""Regenerate the phone-demo replies with a customer-facing system prompt and
complete facts blocks, then verify them automatically.

Nothing the model writes is ever edited. When a reply fails a check we change
the PROMPT and generate again — the output on screen is always verbatim.

Checks applied to every reply:
  1. no credential coaching (PIN / CVV / OTP / password / full card number)
  2. no invented identifiers (claim refs, ticket ids)
  3. no past-tense "already done" claims — the assistant advises, it does not act
  4. every "Rs <amount>" in the reply also appears in the facts block
  5. no ops-desk voice leaking into a customer channel
"""
import json, re, sys, time
sys.path.insert(0, r"D:\Codes\AsaanLLM\simulation")

from agents.atm import AtmMachine, CustomerAtmProfile, Transaction
from agents.atm import judgment as J
from agents.atm.agent import _strip_think
from agents.atm import llm as atm_llm
from dispatch import to_jsonable

from app.agents.loans.agent import LoanAgent
from app.agents.loans.policy import decide, PERSONAL_INSTALMENT_LOAN

OUT = "phone_responses.json"

# ---------------------------------------------------------------- the prompt
# Written against the exact failures seen in v1: credential coaching, a
# fabricated claim reference, "Claim filed" for something not filed, an
# opening negation that contradicted the rest of the reply, and ops-desk
# phrasing ("position it as...") addressed to a customer.
CUSTOMER_SYS = (
    "You are AsaanBank's support assistant, replying to a customer inside their "
    "banking app in Pakistan.\n\n"
    "HARD RULES — breaking any of these is a serious error:\n"
    "1. Never ask the customer for, or tell them to have ready, their PIN, CVV, OTP, "
    "password, or full card number. The bank never needs these. Do not mention them at all.\n"
    "2. Never invent a reference number, case ID, ticket number, date, rate, fee or "
    "amount. If a reference number will be issued, say they will be given one — never "
    "write one out.\n"
    "3. Every number and every rupee figure you state must appear in the FACTS block "
    "below. Do not compute new ones.\n"
    "3b. The same applies to explanations. Do not offer a cause, a reason or a piece "
    "of background that is not in the FACTS block — not even a plausible-sounding one. "
    "If the facts say the card was retracted after being left in the slot, that is the "
    "reason; do not invent a different or additional one.\n"
    "4. You are advising, not acting. Never say a thing is done, filed, blocked or "
    "submitted — the customer or the bank still has to do it.\n"
    "5. Do not contradict or correct how the customer described their own problem, "
    "and never open with 'No'. Start with what they should do first.\n\n"
    "STYLE: speak to the customer as 'you'. Warm, calm, plain. Under 110 words. "
    "Short lines or a few bullets. You are talking TO the customer — never about them, "
    "and never coach a colleague on how to handle them.\n"
    "6. Stop as soon as the facts are delivered. Do not add a closing line of "
    "encouragement, reassurance or opinion — those are where mistakes creep in.\n"
    "7. The reply language is stated in the task line below. Follow it exactly."
)

# Only appended when the scenario is Roman Urdu, so it cannot bleed into the
# English replies (it did: the model started answering English messages in
# Urdu, and the Urdu itself degraded into nonsense).
URDU_RULES = (
    "\n\nLANGUAGE: reply in Roman Urdu, the way a Pakistani bank's own WhatsApp "
    "support writes — short, correct, everyday sentences. Keep every rupee figure "
    "and every banking term (ATM, reference number, instalment, processing fee, "
    "pre-approved, block, claim) in English. Use only common, certain Urdu words. "
    "Never invent a word or an English idiom inside an Urdu sentence. If you are "
    "unsure how to phrase something in Urdu, write that part in English. Two or "
    "three short sentences is the right length."
)
ENGLISH_RULES = "\n\nLANGUAGE: reply in English."

ZAINAB = CustomerAtmProfile(
    name="Zainab Malik", bank="Habib Bank Limited (HBL)", account_type="Salary Account",
    account_number="PK42HBL7719340025", available_balance=214_600,
    card_network="Mastercard", card_tier="Mastercard Platinum Debit", card_last4="4417",
    per_txn_limit=50_000, daily_limit=150_000, free_offus_left=4,
    credit_line=300_000, credit_utilised=41_200,
    cash_advance_sub_limit_pct=0.40, cash_advance_drawn=0,
)

def ask(user_msg, facts_block, task, lang):
    system = CUSTOMER_SYS + (URDU_RULES if lang == "ur" else ENGLISH_RULES)
    prompt = (f"The customer writes:\n\"{user_msg}\"\n\n"
              f"FACTS (authoritative — every figure you use must come from here)\n{facts_block}\n\n"
              f"Reply to the customer in {'Roman Urdu' if lang == 'ur' else 'English'}.")
    raw = atm_llm.chat(prompt, system=system, fewshot_tasks=[task], client=None)
    return _strip_think(raw).strip() if raw else None

# ---------------------------------------------------------------- scenarios
def scenarios():
    S = []

    # 1 — stolen card
    txns = [
        Transaction("2026-07-30", "02:04", "ATM withdrawal Lahore - Ferozepur Road", "ATM", -50_000),
        Transaction("2026-07-30", "02:09", "ATM withdrawal Lahore - Kalma Chowk", "ATM", -50_000),
        Transaction("2026-07-30", "02:15", "ATM withdrawal Lahore - Model Town Link", "ATM", -50_000),
        Transaction("2026-07-30", "02:21", "ATM withdrawal Lahore - Garden Town", "ATM", -14_600),
    ]
    f = J.assess_card_fraud(J.CardFraudInputs(profile=ZAINAB, transactions=txns))
    facts = (
        "- Verdict: high risk — unauthorised use is likely\n"
        "- 4 ATM withdrawals between 02:04 and 02:21 on 2026-07-30, total Rs 164,600\n"
        "- 3 of them were for the full Rs 50,000 per-transaction limit\n"
        "- Card affected: Mastercard Platinum Debit ending 4417\n"
        "- The card can be blocked instantly in the app under Cards, or on the 24-hour helpline\n"
        "- The bank issues a complaint reference number when the customer reports it\n"
        "- A replacement card with a new number can be requested at the same time"
    )
    S.append(dict(id="stolen_card", lang="en", task="card_fraud_assessment", domain="atm",
                  chip="My card is being used!", icon="alert", label="Card stolen / unauthorised withdrawals",
                  user="Someone is using my card! I just got 4 SMS alerts at 2am, I was asleep 😨",
                  facts=facts, obj=f))

    # 2 — ATM kept the card
    capturing = AtmMachine(atm_id="UBL-LHR-3382", bank="UBL", location="Liberty Market, Lahore",
                           area_type="shopping district", model="NCR SelfServ 26", status="ONLINE")
    f = J.guide_card_retention(J.CardRetentionInputs(
        profile=ZAINAB, atm=capturing, atm_bank_full_name="United Bank Limited",
        hours_since_capture=14, capture_reason="card left in slot, timed out and retracted", hold_days=5))
    facts = (
        "- What happened: the machine retracted the card after it was left in the slot\n"
        "- The machine belongs to United Bank Limited; the customer banks with HBL\n"
        "- Because it is another bank's machine, that branch cannot hand an HBL card back over the counter\n"
        "- Captured cards are held 5 working days, then destroyed\n"
        "- The practical route is to block the card with HBL and request a replacement\n"
        "- The account and its money are unaffected — app, internet banking and Raast keep working"
    )
    S.append(dict(id="captured_card", lang="en", task="card_retention_guidance", domain="atm",
                  chip="The ATM kept my card", icon="card", label="ATM captured the card",
                  user="The ATM took my card and didn't give it back. It was a UBL machine, not my bank.",
                  facts=facts, obj=f))

    # 3 — debited, no cash
    dispute_atm = AtmMachine(atm_id="HBL-LHR-6620", bank="HBL", location="Gulberg III, Lahore",
                             area_type="commercial", model="Wincor Nixdorf Procash 2050xe", status="ONLINE")
    di = J.DisputeInputs(atm=dispute_atm, date="2026-07-29", debited=25_000, received=0,
                         journal_entry="DISPENSE FAILED - notes retracted to purge bin",
                         reconciliation_excess=25_000, reversal_window_days=7, days_since_transaction=1)
    f = J.assess_failed_transaction_dispute(di)
    facts = (
        "- Outcome: the money is owed back to the customer\n"
        "- Rs 25,000 was debited on 2026-07-29 at ATM HBL-LHR-6620, Gulberg III; Rs 0 was dispensed\n"
        "- The ATM's own journal records a failed dispense — the notes were pulled into the purge bin\n"
        "- That machine was found Rs 25,000 over at the end of the day, matching exactly\n"
        "- Bank policy reverses a disputed ATM transaction within 7 working days of it being reported\n"
        "- 1 day has passed, so this is well inside the window\n"
        "- To report it the customer needs the date, the ATM ID and the amount; the bank then issues a complaint reference number"
    )
    S.append(dict(id="failed_dispense", lang="en", task="failed_transaction_dispute", domain="atm",
                  chip="Debited, no cash", icon="cash", label="Debited but no cash dispensed",
                  user="Rs 25,000 was taken from my account but the ATM gave me no cash. What do I do?",
                  facts=facts, obj=f))

    # 4 — loan, in the CUSTOMER's voice (v1 leaked credit-officer phrasing)
    loans = LoanAgent()
    c = loans.get("C007")
    d = decide(c, PERSONAL_INSTALMENT_LOAN)
    o = d.offer
    fee = round(o.amount * PERSONAL_INSTALMENT_LOAN.processing_fee_pct)
    facts = (
        f"- The customer is pre-approved — no new application needed\n"
        f"- Product: Personal Instalment Loan\n"
        f"- Amount available: Rs {o.amount:,}\n"
        f"- Term: {o.tenor_months} months\n"
        f"- Rate: {o.annual_rate*100:.1f}% per year, reducing balance\n"
        f"- Monthly instalment: Rs {o.emi:,}\n"
        f"- One-off processing fee: Rs {fee:,}\n"
        f"- After this instalment and their existing commitments they keep Rs {o.headroom_after:,} a month\n"
        f"- Total monthly repayments would reach {o.dbr_pct}% of take-home pay, inside the 40% regulatory cap\n"
        f"- They may take less than the full amount if they prefer"
    )
    S.append(dict(id="loan_offer", lang="en", task="proactive_offer_decision", domain="loans",
                  chip="Loan chahiye", icon="loan", label="Pre-approved loan enquiry",
                  user="Money is a bit tight this month. Can I get a loan?",
                  facts=facts, obj=d))
    return S

# ---------------------------------------------------------------- checks
# Coaching the customer to HAVE credentials ready is the failure (v1 said
# "with your card number and CVV handy"). Warning them NOT to share those is
# correct advice and must pass.
CRED = r"(?:cvv|otp|pin|password|card number)"
BANNED = re.compile(
    r"(?:have|keep|ready|handy|provide|give|share|tell|enter|quote)[^.\n]{0,40}\b" + CRED
    + r"|\b" + CRED + r"[^.\n]{0,20}(?:handy|ready)", re.I)
# ...unless the clause is a negation ("do not tell them your PIN")
NEGATED = re.compile(r"(?:never|not|don'?t|do not|mat|nahi)\b[^.\n]{0,40}\b" + CRED, re.I)
# Only flag something that actually looks like an issued identifier — a token
# with a digit or a slash in it. "you will be given a reference number" is the
# behaviour we asked for and must not trip this.
FAKE_REF = re.compile(
    r"\b(?:ref(?:erence)?|case|ticket|claim)\s*(?:no\.?|number|#|:)?\s*[:#]?\s*"
    r"((?=\S*\d)[A-Za-z0-9][A-Za-z0-9/_-]{3,})", re.I)
DONE_CLAIM = re.compile(r"(claim|dispute|complaint|card|reversal|refund)s?\s+(is|are|has been|have been|was|were)\s+(accepted|approved|filed|registered|blocked|submitted|reversed|settled|granted|done|processed)|i have (filed|blocked|submitted|raised)|claim filed", re.I)
OPS_VOICE = re.compile(r"\b(the customer|position it|sell it|approve|decline the request|this file)\b", re.I)

UR_MARKERS = "(?<![A-Za-z])(hai|hain|karein|karin|aapko|aapki|aap|mein|kar|karne|ke|ka|ki|se|nahi|lein|milega|milta|hoga|wapis|mahina|mahine|par|ye|is)(?![A-Za-z])"

def nums(text):
    return {n.replace(",", "") for n in re.findall(r"Rs\s*([\d,]+)", text)}

def verify(reply, facts, lang="en"):
    issues = []
    m = BANNED.search(reply)
    if m and not NEGATED.search(reply):
        issues.append(f"coaches sharing a credential: {m.group(0)!r}")
    if FAKE_REF.search(reply):
        issues.append(f"invented identifier: {FAKE_REF.search(reply).group(0)!r}")
    if DONE_CLAIM.search(reply):
        issues.append(f"claims action already done: {DONE_CLAIM.search(reply).group(0)!r}")
    if OPS_VOICE.search(reply):
        issues.append(f"ops-desk voice: {OPS_VOICE.search(reply).group(0)!r}")
    if reply.lstrip().lower().startswith("no "):
        issues.append("opens by negating the customer")
    # language check: Urdu function words are a reliable tell either way
    ur_hits = len(re.findall(UR_MARKERS, reply, re.I))
    if lang == "en" and ur_hits >= 3:
        issues.append(f"replied in Urdu when English was asked ({ur_hits} markers)")
    if lang == "ur" and ur_hits < 3:
        issues.append(f"replied in English when Roman Urdu was asked ({ur_hits} markers)")
    unsupported = nums(reply) - nums(facts)
    if unsupported:
        issues.append(f"figures not in FACTS: {sorted(unsupported)}")
    return issues

# ---------------------------------------------------------------- run
def main():
    """Best-of-N. Generate several candidates per scenario, keep only those that
    pass every check, and take the shortest survivor (shortest = least room for
    invention). Selecting among verbatim outputs is not editing them; no reply
    that reaches the screen has been touched."""
    N = 5
    out, allclean = [], True
    for s in scenarios():
        cands = []
        for k in range(N):
            t0 = time.time()
            reply = ask(s["user"], s["facts"], s["task"], s["lang"])
            iss = verify(reply or "", s["facts"], s["lang"]) if reply else ["no reply"]
            cands.append((reply, iss))
            print(f"    [{time.time()-t0:5.1f}s] {s['id']} #{k+1}: {'pass' if not iss else iss}")
            if not iss and len(reply) < 420:
                break                      # good enough, stop burning GPU time
        passing = [(r, i) for r, i in cands if not i and r]
        if passing:
            reply, iss = min(passing, key=lambda ri: len(ri[0]))
        else:
            reply, iss = min(cands, key=lambda ri: len(ri[1]))
            allclean = False
        print(f"[{s['id']}] -> {'CLEAN' if not iss else 'BEST AVAILABLE ' + str(iss)} "
              f"({len(passing)}/{len(cands)} candidates passed)" + chr(10))
        out.append({"id": s["id"], "lang": s["lang"], "task": s["task"], "domain": s["domain"],
                    "chip": s["chip"], "icon": s["icon"], "label": s["label"], "user": s["user"],
                    "ledger": s["facts"], "reply": reply, "issues": iss,
                    "candidates_passed": len(passing), "candidates_tried": len(cands)})
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("ALL CLEAN" if allclean else "SOME REPLIES STILL FAIL")

if __name__ == "__main__":
    main()
