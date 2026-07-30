// ============================================================
// "How it works" — the guided, gamified explanation of the whole flow.
// Written for someone who has never seen the codebase: no jargon
// without a plain-English gloss, and every number on screen is fetched
// live from the real engine rather than typed in here.
// ============================================================

const TOUR_CUSTOMER = 'C007';   // a STRONG file under genuine stress — the clearest teaching case
let tourData = null;            // {profile, score, stress, decision} fetched on demand
let tourIdx = 0;
let quizState = { round: 0, correct: 0, answered: false, roster: [] };

// ---------------------------------------------------------- pipeline diagram

function renderHeroDiagram() {
  const el = document.getElementById('hero-diagram');
  if (!el) return;
  el.innerHTML = `
  <svg viewBox="0 0 520 442" role="img" aria-label="Diagram: customer data flows into a deterministic policy engine, which produces a locked verdict; the AI model receives that verdict and writes the explanation, but cannot change the decision.">
    <defs>
      <marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
        <path d="M0 0 L10 5 L0 10 z" fill="var(--text-faint)"/>
      </marker>
    </defs>

    <!-- input -->
    <rect class="pipe-box" x="112" y="8" width="296" height="52" rx="6"/>
    <text class="pipe-title" x="260" y="30" text-anchor="middle">Customer file · ATM telemetry</text>
    <text class="pipe-sub" x="260" y="45" text-anchor="middle">balances, salary history, cassette levels, alarms</text>

    <path class="pipe-path pipe-path-live" d="M260 60 V80 H130 V100" marker-end="url(#ar)"/>

    <!-- policy engine -->
    <rect class="pipe-box pipe-box-ledger" x="10" y="100" width="240" height="152" rx="6"/>
    <text class="pipe-label" x="26" y="122" fill="var(--ledger)">STEP 1 — PLAIN CODE</text>
    <text class="pipe-title" x="26" y="142">The policy engine</text>
    <text class="pipe-sub" x="26" y="163">· scores the relationship</text>
    <text class="pipe-sub" x="26" y="178">· detects genuine cash stress</text>
    <text class="pipe-sub" x="26" y="193">· runs the instalment maths</text>
    <text class="pipe-sub" x="26" y="208">· applies the decision gate</text>
    <rect x="26" y="220" width="208" height="20" rx="3" fill="var(--ledger-bg)"/>
    <text class="pipe-sub" x="130" y="234" text-anchor="middle" fill="var(--ledger)">no AI involved · fully auditable</text>

    <!-- locked verdict handoff -->
    <g transform="translate(280 140)">
      <circle class="pipe-lock-ring" r="10"/>
      <path d="M-3.5 0 v-3.5 a3.5 3.5 0 0 1 7 0 v3.5" fill="none" stroke="var(--narrative)" stroke-width="1.4"/>
      <rect x="-5" y="0" width="10" height="7.5" rx="1.5" fill="var(--narrative)"/>
    </g>
    <path class="pipe-path pipe-path-live" d="M250 176 H298" marker-end="url(#ar)"/>
    <text class="pipe-sub" x="280" y="199" text-anchor="middle" fill="var(--narrative)">locked</text>

    <!-- model -->
    <rect class="pipe-box pipe-box-narr" x="310" y="100" width="198" height="152" rx="6"/>
    <text class="pipe-label" x="324" y="122" fill="var(--narrative)">STEP 2 — THE MODEL</text>
    <text class="pipe-title" x="324" y="142">Fine-tuned narrator</text>
    <text class="pipe-sub" x="324" y="163">receives the finished</text>
    <text class="pipe-sub" x="324" y="178">verdict as fixed fact,</text>
    <text class="pipe-sub" x="324" y="193">then writes it up in</text>
    <text class="pipe-sub" x="324" y="208">plain language</text>
    <rect x="324" y="220" width="170" height="20" rx="3" fill="var(--narrative-bg)"/>
    <text class="pipe-sub" x="409" y="234" text-anchor="middle" fill="var(--narrative)">explains · never decides</text>

    <!-- blocked back-path: the whole trust argument, drawn -->
    <path d="M320 274 H140" stroke="var(--status-bad)" stroke-width="1.4" fill="none" stroke-dasharray="4 4" opacity="0.5"/>
    <g transform="translate(230 274)">
      <circle r="10" fill="var(--bg)" stroke="var(--status-bad)" stroke-width="1.4"/>
      <path d="M-4 -4 L4 4 M4 -4 L-4 4" stroke="var(--status-bad)" stroke-width="1.6" stroke-linecap="round"/>
    </g>
    <text class="pipe-sub" x="230" y="297" text-anchor="middle" fill="var(--status-bad)">the model cannot change the decision</text>

    <!-- outputs -->
    <path class="pipe-path" d="M130 252 V320 H196 V340" marker-end="url(#ar)"/>
    <path class="pipe-path" d="M409 252 V320 H324 V340" marker-end="url(#ar)"/>

    <rect class="pipe-box" x="112" y="340" width="296" height="82" rx="6"/>
    <text class="pipe-label" x="260" y="362" text-anchor="middle" fill="var(--text-faint)">WHAT THE OFFICER SEES</text>
    <rect x="128" y="372" width="128" height="36" rx="4" fill="var(--ledger-bg)"/>
    <text class="pipe-sub" x="192" y="388" text-anchor="middle" fill="var(--ledger)">the numbers</text>
    <text class="pipe-sub" x="192" y="400" text-anchor="middle" fill="var(--ledger)">(the ledger)</text>
    <rect x="264" y="372" width="128" height="36" rx="4" fill="var(--narrative-bg)"/>
    <text class="pipe-sub" x="328" y="388" text-anchor="middle" fill="var(--narrative)">the explanation</text>
    <text class="pipe-sub" x="328" y="400" text-anchor="middle" fill="var(--narrative)">(the note)</text>
  </svg>`;
}

// ---------------------------------------------------------- the guided tour

const TOUR_STEPS = [
  {
    kicker: 'Step 1 — the file lands',
    title: 'Meet a real customer',
    render: (d) => {
      const p = d.profile;
      return `
      <div class="step-body">
        <strong>${esc(p.name)}</strong> banks with ${esc(p.bank)} in ${esc(p.city)} and works at
        ${esc(p.employment)}. The bank already holds everything below — no new paperwork, no
        application. That is the whole point: the bank goes looking for the customer,
        not the other way round.
      </div>
      <div class="step-grid">
        <div>
          <div class="kv-grid">
            <span class="k">Take-home pay</span><span class="v">${rs(p.net_monthly_income)}/mo</span>
            <span class="k">Banking with us</span><span class="v">${p.account_age_years} years</span>
            <span class="k">Salary arrived</span><span class="v">${p.salary_months_12} of last 12 months</span>
            <span class="k">Typical balance</span><span class="v">${rs(p.avg_balance_6m)}</span>
            <span class="k">Last loan</span><span class="v">${esc(p.previous_loan === 'clean' ? 'repaid, never late' : p.previous_loan)}</span>
            <span class="k">Already owes /mo</span><span class="v">${rs((p.obligations || []).reduce((s, o) => s + o.monthly_amount, 0))}</span>
          </div>
        </div>
        <div>
          <div class="section-note" style="margin-bottom:10px;">Her month-end balance, last 6 months</div>
          ${bigSparkline(p.eom_balances)}
          <div class="step-callout">
            Notice the shape. She is not poor — she was holding
            <b>${rs(p.eom_balances[0])}</b> six months ago. She is <b>draining</b>.
            That difference is the entire business case.
          </div>
        </div>
      </div>`;
    },
  },
  {
    kicker: 'Step 2 — grade the relationship',
    title: 'Four questions, then the red flags',
    render: (d) => {
      const rows = d.score.components.map((c, i) => `
        <div class="reveal-row" style="animation-delay:${i * 130}ms">
          <span class="rl">${esc(plainLabel(c.label))}</span>
          <span class="rp ${c.points < 0 ? 'neg' : ''}">${c.points > 0 ? '+' : ''}${c.points}</span>
        </div>`).join('');
      return `
      <div class="step-body">
        Before deciding anything, the bank asks how good this relationship has been.
        Four things earn points — <strong>how long</strong> they have banked here,
        <strong>how reliably</strong> the salary arrives, <strong>how much</strong> they usually
        hold, and <strong>how well</strong> they repaid last time. Then anything alarming on the
        credit bureau file takes points away.
      </div>
      <div class="step-grid">
        <div>${rows}
          <div class="reveal-total">
            <span class="tl">Relationship score</span>
            <span class="tv">${d.score.score} <span style="font-size:14px;color:var(--text-muted)">${d.score.band}</span></span>
          </div>
        </div>
        <div>
          <div class="step-callout">
            <b>Why bands, not a raw number?</b> Because the four bands each unlock a different
            shelf of products. <b>STRONG</b> and <b>ACCEPTABLE</b> can be offered credit without
            collateral. <b>THIN</b> means we simply do not know them well enough yet.
            <b>POOR</b> means the bureau is warning us — and that changes everything, as you will
            see in two steps.
          </div>
          <div class="step-callout" style="border-color:var(--ledger); background:var(--ledger-bg); color:#DCEFEA">
            Every one of these points is added by ordinary code you can read line by line.
            Run it twice on the same file and you get the same answer, forever.
          </div>
        </div>
      </div>`;
    },
  },
  {
    kicker: 'Step 3 — is the need real?',
    title: 'Short of cash, or just spending?',
    render: (d) => {
      const s = d.stress;
      return `
      <div class="step-body">
        A good customer who is comfortable does not need a loan — offering one is just
        marketing, and slightly insulting. So the engine looks for a specific, boring signal:
        <strong>how many days in the last 30 did the balance sit under Rs 5,000?</strong>
        Ten or more is the line.
      </div>
      <div class="step-grid">
        <div>
          ${stressMeter(s.days_below_5k)}
          <div class="reveal-total">
            <span class="tl">Verdict</span>
            <span class="tv" style="color:${s.stressed ? 'var(--status-warn)' : 'var(--status-good)'}">
              ${s.stressed ? 'Genuine stress' : 'Comfortable'}
            </span>
          </div>
        </div>
        <div>
          <div class="section-note" style="margin-bottom:8px;">What the engine noted</div>
          ${s.notes.map((n) => `<div class="reveal-row"><span class="rl">${esc(n)}</span><span></span></div>`).join('')}
          <div class="step-callout">
            This is deliberately a <b>dumb, countable</b> signal — not a mood, not a prediction.
            An auditor or a regulator can recount it from the statement themselves.
          </div>
        </div>
      </div>`;
    },
  },
  {
    kicker: 'Step 4 — the gate',
    title: 'The rule that surprises people',
    render: (d) => {
      const band = d.score.band, stressed = d.stress.stressed;
      const lit = (b, st) => (band === b && stressed === st) ? ' lit' : '';
      return `
      <div class="step-body">
        Now the two answers meet. Being <strong>short of cash is not enough</strong> to get an
        offer, and neither is <strong>being a good customer</strong>. You need both.
      </div>
      <div class="step-grid">
        <div class="gate">
          <div class="gate-row${lit('STRONG', true)}${lit('ACCEPTABLE', true)}">
            <span class="gate-cond">STRONG / ACCEPTABLE<br>+ short of cash</span>
            <span class="gate-arrow">→</span>
            <span class="gate-out OFFER">OFFER<span class="gate-why">a good customer, genuinely stuck — this is when help lands as help</span></span>
          </div>
          <div class="gate-row${lit('STRONG', false)}${lit('ACCEPTABLE', false)}">
            <span class="gate-cond">STRONG / ACCEPTABLE<br>+ comfortable</span>
            <span class="gate-arrow">→</span>
            <span class="gate-out MONITOR">MONITOR<span class="gate-why">keep them pre-approved, but do not pester them</span></span>
          </div>
          <div class="gate-row${lit('THIN', true)}${lit('THIN', false)}">
            <span class="gate-cond">THIN<br>(any situation)</span>
            <span class="gate-arrow">→</span>
            <span class="gate-out MONITOR">MONITOR<span class="gate-why">too little history to lend against safely</span></span>
          </div>
          <div class="gate-row${band === 'POOR' ? ' lit bad' : ''}">
            <span class="gate-cond">POOR<br>+ short of cash</span>
            <span class="gate-arrow">→</span>
            <span class="gate-out DECLINE">DECLINE<span class="gate-why">and this is the one people argue about</span></span>
          </div>
        </div>
        <div>
          <div class="step-callout">
            <b>Why refuse the person who most obviously needs money?</b> Because a POOR file
            means they are already behind on something. Handing them more unsecured debt does
            not rescue them — it buys a few weeks and then becomes the next default, with
            their credit record even worse than before.
          </div>
          <div class="step-callout" style="border-color:var(--ledger); background:var(--ledger-bg); color:#DCEFEA">
            So a decline is never just "no". The engine is required to return
            <b>secured alternatives</b> — lending against their own gold or deposit, or a small
            secured card to repair the bureau record. Nobody leaves with nothing.
          </div>
        </div>
      </div>`;
    },
  },
  {
    kicker: 'Step 5 — size it honestly',
    title: 'How much can she actually repay?',
    render: (d) => {
      const o = d.decision.offer;
      if (!o) return `<div class="step-body">This file did not reach an offer, so there is nothing to size.</div>`;
      const income = d.profile.net_monthly_income;
      const obl = (d.profile.obligations || []).reduce((s, x) => s + x.monthly_amount, 0);
      return `
      <div class="step-body">
        The regulator caps total monthly repayments at <strong>40% of take-home pay</strong>.
        The engine works backwards from that ceiling: what is the biggest loan whose instalment
        still fits underneath, once existing commitments are counted?
      </div>
      <div class="step-grid">
        <div>${dbrBar(income, obl, o.emi)}</div>
        <div>
          <div class="kv-grid">
            <span class="k">Loan offered</span><span class="v">${rs(o.amount)}</span>
            <span class="k">Repaid over</span><span class="v">${o.tenor_months} months</span>
            <span class="k">Rate</span><span class="v">${(o.annual_rate * 100).toFixed(1)}% a year</span>
            <span class="k">Monthly instalment</span><span class="v">${rs(o.emi)}</span>
            <span class="k">Left after everything</span><span class="v">${rs(o.headroom_after)}/mo</span>
          </div>
          <div class="step-callout">
            The cap is a ceiling, not a target. A customer sitting exactly on 40% has
            <b>no room for a bad month</b> — one delayed salary and they are late. Where the
            purpose allows, the engine offers below the maximum on purpose.
          </div>
        </div>
      </div>`;
    },
  },
  {
    kicker: 'Step 6 — the handoff',
    title: 'Now, and only now, the AI is allowed to speak',
    render: () => `
      <div class="step-body">
        Everything you have watched so far happened in ordinary code. The decision is already
        made and cannot move. <strong>Only at this point</strong> is the AI handed the file — and
        what it receives is a finished verdict labelled "authoritative, do not contradict".
      </div>
      <div class="handoff">
        <div class="handoff-card locked">
          <h4>What it is given</h4>
          <p>A sealed result:</p>
          <ul>
            <li>the decision (offer / monitor / decline)</li>
            <li>the score and every point behind it</li>
            <li>the exact amount, rate and instalment</li>
            <li>the reasoning the code used</li>
          </ul>
        </div>
        <div class="handoff-arrow"><span class="big">→</span>one direction only</div>
        <div class="handoff-card speaks">
          <h4>What it is asked for</h4>
          <p>One thing: write this up the way a senior colleague would explain it to the
             customer or the branch — warm, numerate, honest about the cost, in English or
             Roman Urdu.</p>
        </div>
      </div>
      <div class="step-callout">
        <b>So what happens if the model gets it wrong?</b> It cannot get the decision wrong,
        because it was never asked for one. Worst case it writes an awkward sentence — and if
        the model is switched off entirely, the screen simply shows the engine's own plain-text
        summary instead. The bank keeps working.
      </div>`,
  },
  {
    kicker: 'Step 7 — the result',
    title: 'What the loan officer actually sees',
    render: (d) => {
      const dec = d.decision;
      const lines = [
        `Decision: ${dec.action}`,
        `Score: ${dec.score.score} (${dec.score.band})`,
        `Cash stress: ${dec.stress.stressed ? 'yes' : 'no'} (${dec.stress.days_below_5k} days under Rs 5,000)`,
      ];
      if (dec.offer) lines.push(`Offer: ${rs(dec.offer.amount)} over ${dec.offer.tenor_months} months — instalment ${rs(dec.offer.emi)}, DBR ${dec.offer.dbr_pct}%`);
      return `
      <div class="step-body">
        Two halves, always. On the left the machine's arithmetic, which never varies. On the
        right the human-readable note. If they ever disagreed, the left half is the one that
        counts — and that is exactly why it is printed next to the prose instead of hidden
        behind it.
      </div>
      <div style="margin-top:20px">
        ${duoBlock('Policy verdict', lines.join('\n'), d.narrative, d.narrativeSource)}
      </div>
      <div class="step-callout">
        You have now seen the entire path a decision takes. The other
        <b>50 job types</b> — ATM cash forecasting, fraud checks, gold-loan sizing, van routing —
        all run on this same two-part shape.
      </div>`;
    },
  },
];

function plainLabel(label) {
  return label
    .replace('account age 3 years or more', 'banking with us 3+ years')
    .replace(/account age 1-3 years/, 'banking with us 1–3 years')
    .replace(/account age under 1 year/, 'new to the bank')
    .replace('salary credited 11+ of last 12 months', 'salary arrived almost every month')
    .replace('salary credited 9-10 of last 12 months', 'salary arrived most months')
    .replace('salary credited under 9 of last 12 months', 'salary often missing')
    .replace('average balance above Rs 50,000', 'keeps a healthy balance')
    .replace('average balance Rs 15,000-50,000', 'keeps a modest balance')
    .replace('average balance below Rs 15,000', 'runs a thin balance')
    .replace('previous loan fully repaid, never 30+ days late', 'repaid the last loan, never late')
    .replace('previous loan repaid with 1-2 late months', 'repaid the last loan, a little late')
    .replace('no previous loan history', 'never borrowed here before')
    .replace('cheque/direct-debit bounce in last 12 months', 'a payment bounced this year')
    .replace('90+ day delinquency on eCIB in last 24 months', '3 months behind on another lender')
    .replace('write-off/litigation flag on eCIB', 'a written-off debt on the credit bureau');
}

function bigSparkline(values) {
  if (!values || values.length < 2) return '';
  const w = 300, h = 92, pad = 6;
  const min = Math.min(...values), max = Math.max(...values), range = (max - min) || 1;
  const pt = (v, i) => [pad + (i / (values.length - 1)) * (w - pad * 2), h - pad - ((v - min) / range) * (h - pad * 2)];
  const pts = values.map(pt);
  const line = pts.map((p) => p.join(',')).join(' ');
  const area = `${pad},${h - pad} ${line} ${w - pad},${h - pad}`;
  const falling = values[values.length - 1] < values[0];
  const col = falling ? 'var(--status-bad)' : 'var(--status-good)';
  return `<svg viewBox="0 0 ${w} ${h}" style="width:100%;height:auto">
    <polygon points="${area}" fill="${col}" opacity="0.09"/>
    <polyline points="${line}" fill="none" stroke="${col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
    ${pts.map((p, i) => `<circle cx="${p[0]}" cy="${p[1]}" r="${i === pts.length - 1 ? 3.6 : 2.2}" fill="${col}"/>`).join('')}
  </svg>
  <div style="display:flex;justify-content:space-between;font-family:var(--f-mono);font-size:11px;color:var(--text-faint);margin-top:4px">
    <span>${rs(values[0])}</span><span style="color:${col}">${rs(values[values.length - 1])}</span>
  </div>`;
}

function stressMeter(days) {
  const cells = Array.from({ length: 30 }, (_, i) => {
    const on = i < days;
    return `<div style="flex:1;height:34px;border-radius:2px;background:${on ? 'var(--status-warn)' : 'var(--border-soft)'};opacity:${on ? 0.35 + (i / 30) * 0.65 : 1}"></div>`;
  }).join('');
  return `
    <div class="section-note" style="margin-bottom:8px;">Each bar is one day of the last 30</div>
    <div style="display:flex;gap:2px;align-items:flex-end">${cells}</div>
    <div style="display:flex;justify-content:space-between;font-family:var(--f-mono);font-size:11px;color:var(--text-faint);margin-top:7px">
      <span><b style="color:var(--status-warn)">${days} days</b> under Rs 5,000</span>
      <span>threshold: 10</span>
    </div>`;
}

function dbrBar(income, obligations, emi) {
  const cap = income * 0.4;
  const pctOf = (v) => Math.min(100, (v / income) * 100);
  return `
    <div class="section-note" style="margin-bottom:10px;">Every rupee of monthly pay</div>
    <div style="position:relative;height:42px;background:var(--border-soft);border-radius:4px;overflow:hidden">
      <div style="position:absolute;left:0;top:0;bottom:0;width:${pctOf(obligations)}%;background:var(--text-faint);opacity:.55"></div>
      <div style="position:absolute;left:${pctOf(obligations)}%;top:0;bottom:0;width:${pctOf(emi)}%;background:var(--ledger)"></div>
      <div style="position:absolute;left:40%;top:-4px;bottom:-4px;width:2px;background:var(--status-bad)"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-family:var(--f-mono);font-size:11px;color:var(--text-faint);margin-top:8px">
      <span>already committed ${rs(obligations)}</span><span style="color:var(--status-bad)">40% cap = ${rs(cap)}</span>
    </div>
    <div style="margin-top:14px;display:flex;gap:16px;font-size:12px;flex-wrap:wrap">
      <span><span style="display:inline-block;width:10px;height:10px;background:var(--text-faint);opacity:.55;border-radius:2px"></span> existing commitments</span>
      <span><span style="display:inline-block;width:10px;height:10px;background:var(--ledger);border-radius:2px"></span> the new instalment</span>
    </div>`;
}

// The profile and the policy verdict are instant (pure computation). The
// narrative needs the model and can take ~30s, so it is fetched separately
// in the background: the tour opens immediately on the fast data, and the
// prose has usually landed by the time the reader reaches step 7.
async function loadTourData() {
  const detail = await (await fetch(`/loans/customers/${TOUR_CUSTOMER}`)).json();
  const fast = await (await fetch('/loans/proactive_offer_decision', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ customer_id: TOUR_CUSTOMER, use_llm: false }),
  })).json();

  const data = {
    profile: detail.profile, score: detail.score, stress: detail.stress,
    decision: fast.decision, narrative: null, narrativeSource: null,
  };

  // background: upgrade the template narrative to the model's own words
  fetch('/loans/proactive_offer_decision', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ customer_id: TOUR_CUSTOMER, use_llm: true }),
  }).then((r) => r.json()).then((full) => {
    data.narrative = full.narrative;
    data.narrativeSource = full.narrative_source;
    // if the reader is already sitting on the final step, swap it in live
    if (tourIdx === TOUR_STEPS.length - 1) renderTourStep();
  }).catch(() => {});

  return data;
}

function renderTourStep() {
  const stage = document.getElementById('tour-stage');
  const step = TOUR_STEPS[tourIdx];
  stage.innerHTML = `<div class="step-in">
    <div class="step-kicker">${esc(step.kicker)}</div>
    <h2 class="step-title">${esc(step.title)}</h2>
    ${step.render(tourData)}
  </div>`;

  document.getElementById('tour-steps').innerHTML = TOUR_STEPS.map((_, i) =>
    `<div class="tour-dot ${i < tourIdx ? 'done' : ''}${i === tourIdx ? ' current' : ''}">${i < tourIdx ? '✓' : i + 1}</div>`).join('');
  document.getElementById('tour-bar-fill').style.width = `${((tourIdx + 1) / TOUR_STEPS.length) * 100}%`;
  document.getElementById('tour-counter').textContent = `${tourIdx + 1} of ${TOUR_STEPS.length}`;
  document.getElementById('tour-prev').disabled = tourIdx === 0;
  document.getElementById('tour-next').textContent = tourIdx === TOUR_STEPS.length - 1 ? 'Finish' : 'Continue';
}

async function startTour() {
  const tour = document.getElementById('tour');
  tour.hidden = false;
  document.getElementById('tour-stage').innerHTML = '<div class="hint"><span class="loading-spinner"></span> loading a real customer file…</div>';
  tour.scrollIntoView({ behavior: 'smooth', block: 'start' });
  if (!tourData) {
    try { tourData = await loadTourData(); }
    catch {
      document.getElementById('tour-stage').innerHTML = '<div class="hint" style="color:var(--status-bad)">Could not reach the engine. Is the server running?</div>';
      return;
    }
  }
  tourIdx = 0;
  renderTourStep();
}

// ---------------------------------------------------------- the quiz

async function buildQuiz() {
  const r = await fetch('/loans/book');
  const book = (await r.json()).customers;
  // Three cases that between them teach the whole gate: the clear offer,
  // the counterintuitive decline, and the good-but-comfortable monitor.
  const pick = (fn) => book.find(fn);
  quizState.roster = [
    pick((c) => c.action === 'OFFER' && c.score >= 6),
    pick((c) => c.action === 'DECLINE' && c.stressed),
    pick((c) => c.action === 'MONITOR' && c.score >= 6 && !c.stressed),
  ].filter(Boolean);
  quizState.round = 0; quizState.correct = 0;
  renderQuiz();
}

async function renderQuiz() {
  const host = document.getElementById('quiz');
  if (!quizState.roster.length) { host.innerHTML = '<div class="quiz-body hint">Quiz unavailable.</div>'; return; }

  if (quizState.round >= quizState.roster.length) {
    host.innerHTML = `
      <div class="quiz-head"><span class="quiz-score">Round complete</span></div>
      <div class="quiz-body" style="text-align:center;padding:38px 22px">
        <div style="font-family:var(--f-mono);font-size:38px;font-weight:600;color:var(--ledger)">${quizState.correct}/${quizState.roster.length}</div>
        <div style="margin:10px 0 20px;color:var(--text-muted);font-size:14px">
          ${quizState.correct === quizState.roster.length
            ? 'All three. You have the policy — including the awkward one.'
            : 'The declines are the counterintuitive ones. Worth a second run.'}
        </div>
        <button class="btn btn-primary" id="quiz-again">Play again</button>
      </div>`;
    document.getElementById('quiz-again').addEventListener('click', () => { quizState.round = 0; quizState.correct = 0; renderQuiz(); });
    return;
  }

  const c = quizState.roster[quizState.round];
  const detail = await (await fetch(`/loans/customers/${c.customer_id}`)).json();
  const p = detail.profile;
  const flags = [];
  if (p.cheque_bounce_12m) flags.push('a bounced payment');
  if (p.ecib_dpd90_24m) flags.push('3 months behind elsewhere');
  if (p.ecib_writeoff) flags.push('a written-off debt');

  host.innerHTML = `
    <div class="quiz-head">
      <span class="quiz-score">Case <b>${quizState.round + 1}</b> of ${quizState.roster.length}</span>
      <span class="quiz-score">Score <b>${quizState.correct}</b></span>
    </div>
    <div class="quiz-body">
      <div style="font-family:var(--f-display);font-size:19px;margin-bottom:4px">${esc(p.name)}</div>
      <div class="hint" style="margin-bottom:16px">${esc(p.employment)} · ${esc(p.city)}</div>
      <div class="quiz-file">
        <div class="quiz-fact"><div class="qk">With us</div><div class="qv">${p.account_age_years} yrs</div></div>
        <div class="quiz-fact"><div class="qk">Salary arrived</div><div class="qv">${p.salary_months_12}/12</div></div>
        <div class="quiz-fact"><div class="qk">Typical balance</div><div class="qv">${rs(p.avg_balance_6m)}</div></div>
        <div class="quiz-fact"><div class="qk">Days under Rs 5k</div><div class="qv">${p.days_below_5k_30d} of 30</div></div>
        <div class="quiz-fact ${flags.length ? 'flag' : ''}"><div class="qk">Bureau flags</div><div class="qv">${flags.length ? esc(flags.join(', ')) : 'none'}</div></div>
      </div>
      <div class="hint" style="margin-bottom:10px">What should the bank do?</div>
      <div class="quiz-choices" id="quiz-choices">
        <button class="quiz-choice" data-a="OFFER">Offer a loan</button>
        <button class="quiz-choice" data-a="MONITOR">Just monitor</button>
        <button class="quiz-choice" data-a="DECLINE">Decline</button>
      </div>
      <div id="quiz-verdict"></div>
    </div>`;

  document.querySelectorAll('#quiz-choices .quiz-choice').forEach((btn) => {
    btn.addEventListener('click', () => answerQuiz(btn.dataset.a, c, detail));
  });
}

function answerQuiz(answer, c, detail) {
  const truth = c.action;
  const right = answer === truth;
  if (right) quizState.correct++;

  document.querySelectorAll('#quiz-choices .quiz-choice').forEach((b) => {
    b.disabled = true;
    if (b.dataset.a === truth) b.classList.add('correct');
    else if (b.dataset.a === answer) b.classList.add('wrong');
  });

  const p = detail.profile, band = detail.score.band, stressed = detail.stress.stressed;
  let why;
  if (truth === 'OFFER') {
    why = `Score ${detail.score.score} puts this file in <b>${band}</b>, and the balance sat under Rs 5,000 on
           <b>${p.days_below_5k_30d} of the last 30 days</b> — a good relationship that is genuinely stuck.
           That is precisely the case where an offer is help rather than marketing.`;
  } else if (truth === 'DECLINE') {
    why = `The bureau flags drag the score to <b>${detail.score.score} (${band})</b>.
           ${stressed ? `They <b>are</b> short of cash — ${p.days_below_5k_30d} days under Rs 5,000 — and that is exactly why unsecured
           lending is refused. More debt on an already-broken file buys a few weeks and then becomes the next default.` : ''}
           They still leave with something: lending against their own gold or deposit, or a secured card to repair the record.`;
  } else {
    why = stressed
      ? `Score ${detail.score.score} (<b>${band}</b>) — not enough history yet to lend unsecured against, whatever the need looks like.`
      : `Score ${detail.score.score} (<b>${band}</b>) is a good file, but the balance never dips —
         only ${p.days_below_5k_30d} of 30 days under Rs 5,000. No real need, so no offer. Keep them pre-approved and leave them alone.`;
  }

  document.getElementById('quiz-verdict').innerHTML = `
    <div class="quiz-verdict">
      <div class="qv-head ${right ? 'right' : 'nope'}">${right ? '✓ Correct — the engine says ' + truth : '✗ The engine says ' + truth}</div>
      <p>${why}</p>
    </div>
    <div class="quiz-foot"><button class="btn btn-primary" id="quiz-next">${quizState.round === quizState.roster.length - 1 ? 'See result' : 'Next case'}</button></div>`;

  document.getElementById('quiz-next').addEventListener('click', () => { quizState.round++; renderQuiz(); });
}

// ---------------------------------------------------------- counters + wiring

function animateCounters() {
  const els = document.querySelectorAll('.trust-n[data-count]');
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      io.unobserve(e.target);
      const target = parseInt(e.target.dataset.count, 10);
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        e.target.textContent = target.toLocaleString(); return;
      }
      const dur = 1100, t0 = performance.now();
      const tick = (t) => {
        const k = Math.min(1, (t - t0) / dur);
        const eased = 1 - Math.pow(1 - k, 3);
        e.target.textContent = Math.round(target * eased).toLocaleString();
        if (k < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
  }, { threshold: 0.4 });
  els.forEach((el) => io.observe(el));
}

document.addEventListener('DOMContentLoaded', () => {
  renderHeroDiagram();
  animateCounters();
  buildQuiz().catch(() => {});

  document.getElementById('start-tour').addEventListener('click', startTour);
  document.getElementById('tour-exit').addEventListener('click', () => { document.getElementById('tour').hidden = true; });
  document.getElementById('tour-next').addEventListener('click', () => {
    if (tourIdx < TOUR_STEPS.length - 1) { tourIdx++; renderTourStep(); }
    else { document.getElementById('tour').hidden = true; document.getElementById('quiz-block').scrollIntoView({ behavior: 'smooth', block: 'center' }); }
  });
  document.getElementById('tour-prev').addEventListener('click', () => { if (tourIdx > 0) { tourIdx--; renderTourStep(); } });

  document.querySelectorAll('.domain-card').forEach((card) => {
    card.addEventListener('click', () => document.querySelector(`.tab[data-view="${card.dataset.goto}"]`).click());
  });
});
