// AUTO-GENERATED - do not hand-edit.
// Verbatim output of the fine-tuned atm-loans-qwen3-14b, captured 2026-07-31 by
// gen_phone_v2.py. Method: the deterministic engine produces the FACTS block, the
// model writes the reply, then every candidate is auto-checked for fabricated
// figures, invented reference numbers, credential coaching, false "already done"
// claims and ops-desk voice. Candidates that failed were discarded and another was
// generated; the text below is exactly what the model wrote, never edited.
const PHONE_SCRIPT = [
  {
    "id": "stolen_card",
    "task": "card_fraud_assessment",
    "domain": "atm",
    "chip": "My card is being used!",
    "icon": "alert",
    "label": "Card stolen / unauthorised withdrawals",
    "user": "Someone is using my card! I just got 4 SMS alerts at 2am, I was asleep 😨",
    "ledger": "- Verdict: high risk — unauthorised use is likely\n- 4 ATM withdrawals between 02:04 and 02:21 on 2026-07-30, total Rs 164,600\n- 3 of them were for the full Rs 50,000 per-transaction limit\n- Card affected: Mastercard Platinum Debit ending 4417\n- The card can be blocked instantly in the app under Cards, or on the 24-hour helpline\n- The bank issues a complaint reference number when the customer reports it\n- A replacement card with a new number can be requested at the same time",
    "reply": "This looks like unauthorised use — block the card first. Do it now: in the app go to Cards and tap Block next to Mastercard Platinum Debit ending 4417, or call the helpline at any hour.\n\nOnce it is blocked, report the incident with your full name, account number and what happened — the bank will issue a complaint reference number and start an investigation."
  },
  {
    "id": "captured_card",
    "task": "card_retention_guidance",
    "domain": "atm",
    "chip": "The ATM kept my card",
    "icon": "card",
    "label": "ATM captured the card",
    "user": "The ATM took my card and didn't give it back. It was a UBL machine, not my bank.",
    "ledger": "- What happened: the machine retracted the card after it was left in the slot\n- The machine belongs to United Bank Limited; the customer banks with HBL\n- Because it is another bank's machine, that branch cannot hand an HBL card back over the counter\n- Captured cards are held 5 working days, then destroyed\n- The practical route is to block the card with HBL and request a replacement\n- The account and its money are unaffected — app, internet banking and Raast keep working",
    "reply": "The machine retracted the card because it was left in the slot.\n\nThat branch cannot hand back an HBL card over the counter - captured cards are held 5 working days then destroyed. Block the card with your own bank and request a replacement. The account itself is unaffected, and you can still use internet banking and Raast."
  },
  {
    "id": "failed_dispense",
    "task": "failed_transaction_dispute",
    "domain": "atm",
    "chip": "Debited, no cash",
    "icon": "cash",
    "label": "Debited but no cash dispensed",
    "user": "Rs 25,000 was taken from my account but the ATM gave me no cash. What do I do?",
    "ledger": "- Outcome: the money is owed back to the customer\n- Rs 25,000 was debited on 2026-07-29 at ATM HBL-LHR-6620, Gulberg III; Rs 0 was dispensed\n- The ATM's own journal records a failed dispense — the notes were pulled into the purge bin\n- That machine was found Rs 25,000 over at the end of the day, matching exactly\n- Bank policy reverses a disputed ATM transaction within 7 working days of it being reported\n- 1 day has passed, so this is well inside the window\n- To report it the customer needs the date, the ATM ID and the amount; the bank then issues a complaint reference number",
    "reply": "The evidence matches: Rs 25,000 was debited and nothing came out. The ATM journal itself records this as a failed dispense.\n\n- Take a printed or digital copy of your SMS and screen capture showing the failed transaction\n- Go to any branch with your CNIC and the ATM ID HBL-LHR-6620, date 2026-07-29, and the amount Rs 25,000\n- The bank will issue a complaint reference number and start the reversal under policy"
  },
  {
    "id": "loan_offer",
    "task": "proactive_offer_decision",
    "domain": "loans",
    "chip": "Loan chahiye",
    "icon": "loan",
    "label": "Pre-approved loan enquiry",
    "user": "Money is a bit tight this month. Can I get a loan?",
    "ledger": "- The customer is pre-approved — no new application needed\n- Product: Personal Instalment Loan\n- Amount available: Rs 1,350,000\n- Term: 48 months\n- Rate: 28.3% per year, reducing balance\n- Monthly instalment: Rs 47,282\n- One-off processing fee: Rs 27,000\n- After this instalment and their existing commitments they keep Rs 80,418 a month\n- Total monthly repayments would reach 40.0% of take-home pay, inside the 40% regulatory cap\n- They may take less than the full amount if they prefer",
    "reply": "You are pre-approved — no application needed. Rs 1,350,000 at 28.3% over 48 months is Rs 47,282 a month, keeping Rs 80,418 of take-home pay after instalments.\n\nTake less than the full amount if you prefer — the approval stands either way."
  }
];
