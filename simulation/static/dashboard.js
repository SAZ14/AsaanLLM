// ============================================================
// AsaanBank Operations Ledger — dashboard logic
// Same-origin fetches against service.py. No build step, no framework.
// ============================================================

const $ = (sel, root = document) => root.querySelector(sel);
const $all = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const rs = (n) => 'Rs ' + Math.round(n).toLocaleString('en-IN');

let loanBook = [];
let tasksSchema = null;

// ---------------------------------------------------------- boot

document.addEventListener('DOMContentLoaded', () => {
  $('#today-date').textContent = new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
  setupTabs();
  setupDrawer();
  checkLlmStatus();
  loadLoansView();
  loadAtmView();
  loadTasksView();
});

function setupTabs() {
  $all('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      $all('.tab').forEach((t) => t.classList.remove('active'));
      $all('.view').forEach((v) => v.classList.remove('active'));
      tab.classList.add('active');
      $('#view-' + tab.dataset.view).classList.add('active');
    });
  });
}

function setupDrawer() {
  $('#drawer-close').addEventListener('click', closeDrawer);
  $('#drawer-overlay').addEventListener('click', closeDrawer);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeDrawer(); });
}
function openDrawer(title, subtitle, bodyHtml) {
  $('#drawer-title').textContent = title;
  $('#drawer-subtitle').textContent = subtitle;
  $('#drawer-body').innerHTML = bodyHtml;
  $('#drawer').classList.add('open');
  $('#drawer-overlay').classList.add('open');
}
function closeDrawer() {
  $('#drawer').classList.remove('open');
  $('#drawer-overlay').classList.remove('open');
}

// Re-checks while disconnected: a probe issued in the seconds after the
// server boots can time out on a cold connection to the GPU box and report
// "template mode" for a model that is actually up. Retrying a few times
// lets that false negative correct itself instead of misleading the reader.
async function checkLlmStatus(attempt = 0) {
  let d;
  try {
    d = await (await fetch('/llm-status', { cache: 'no-store' })).json();
  } catch {
    d = { loans: { connected: false, model: '?' }, atm: { connected: false, model: '?' } };
  }
  setPill('#loan-llm-pill', 'loans model', d.loans);
  setPill('#atm-llm-pill', 'atm model', d.atm);
  if (!(d.loans.connected && d.atm.connected) && attempt < 3) {
    setTimeout(() => checkLlmStatus(attempt + 1), 5000);
  }
}
function setPill(sel, label, status) {
  const pill = $(sel);
  pill.classList.toggle('live', status.connected);
  pill.classList.toggle('offline', !status.connected);
  $('span:last-child', pill).textContent = `${label}: ${status.connected ? status.model + ' live' : 'template mode'}`;
}

function duoBlock(ledgerLabel, ledgerText, narrativeText, source) {
  const narrHtml = narrativeText
    ? `<div class="narrative-body">${esc(narrativeText)}</div><div class="narrative-source">source: ${esc(source || 'template')}</div>`
    : `<div class="narrative-body loading"><span class="loading-spinner"></span> narrating…</div>`;
  return `
    <div class="duo">
      <div class="duo-half ledger-half"><span class="tag">${esc(ledgerLabel)}</span><div class="ledger-body">${esc(ledgerText)}</div></div>
      <div class="duo-seam"><span class="duo-seam-label">computed → narrated</span></div>
      <div class="duo-half narrative-half"><span class="tag">Officer's note</span>${narrHtml}</div>
    </div>`;
}

// ============================================================ LOANS DESK

async function loadLoansView() {
  const r = await fetch('/loans/book');
  const data = await r.json();
  loanBook = data.customers;

  const offer = loanBook.filter((c) => c.action === 'OFFER').sort((a, b) => b.score - a.score);
  const monitor = loanBook.filter((c) => c.action === 'MONITOR').sort((a, b) => b.score - a.score);
  const decline = loanBook.filter((c) => c.action === 'DECLINE').sort((a, b) => a.score - b.score);
  const avgScore = Math.round(loanBook.reduce((s, c) => s + c.score, 0) / loanBook.length);

  $('#loan-stats').innerHTML = [
    stat(loanBook.length, 'Book size', ''),
    stat(offer.length, 'Offer', 'good'),
    stat(monitor.length, 'Monitor', 'warn'),
    stat(decline.length, 'Decline', 'bad'),
    stat(avgScore, 'Avg. score', ''),
  ].join('');

  $('#count-offer').textContent = offer.length;
  $('#count-monitor').textContent = monitor.length;
  $('#count-decline').textContent = decline.length;

  $('#list-offer').innerHTML = offer.map(custCard).join('') || emptyNote('No offers queued right now.');
  $('#list-monitor').innerHTML = monitor.map(custCard).join('') || emptyNote('Nothing on watch.');
  $('#list-decline').innerHTML = decline.map(custCard).join('') || emptyNote('No declines on file.');

  $all('.cust-card').forEach((card) => card.addEventListener('click', () => openCustomer(card.dataset.id)));
}

function stat(n, label, tone) {
  return `<div class="stat-tile"><div class="n ${tone}">${n}</div><div class="lbl">${esc(label)}</div></div>`;
}
function emptyNote(text) { return `<div class="empty-note">${esc(text)}</div>`; }

// Generated monogram badges — not official bank marks, just a stable colour
// per institution code so the eye can group rows at a glance.
const BANK_HUES = {
  HBL: 6, UBL: 145, MCB: 42, NBP: 200, MEBL: 168, BAFL: 268, ABL: 24,
  SNDB: 190, BOP: 96, HBLMFB: 330, KMBL: 55, FINCA: 285, JSBL: 15, BOK: 210,
};
function bankHue(code) {
  if (BANK_HUES[code] !== undefined) return BANK_HUES[code];
  let h = 0;
  for (let i = 0; i < code.length; i++) h = (h * 31 + code.charCodeAt(i)) % 360;
  return h;
}
function bankBadge(code) {
  const h = bankHue(code);
  const short = code.length > 4 ? code.slice(0, 4) : code;
  return `<span class="bank-badge" style="background:hsl(${h} 52% 62%)" title="${esc(code)}">${esc(short)}</span>`;
}

function custCard(c) {
  const tone = c.score >= 6 ? 'good' : c.score >= 3 ? 'warn' : c.score >= 0 ? '' : 'bad';
  const stressTag = c.stressed ? ` · ${c.days_below_5k}d&lt;5k` : '';
  return `<div class="cust-card" tabindex="0" data-id="${esc(c.customer_id)}">
    <div class="who">
      ${bankBadge(c.bank)}
      <div class="who-text"><div class="name">${esc(c.name)}</div><div class="meta">${esc(c.city)}${stressTag}</div></div>
    </div>
    <div class="score ${tone}">${c.score}</div>
  </div>`;
}

// A small ATM drawn to scale: each cassette becomes a fill bar, faults go red.
function atmIllustration(machine) {
  const cass = machine.cassettes.slice(0, 4);
  const bw = 15, gap = 5;
  const startX = 62 - ((cass.length * bw + (cass.length - 1) * gap) / 2);
  const bars = cass.map((c, i) => {
    const pct = Math.max(0.04, Math.min(1, c.notes / c.capacity));
    const full = 30, hgt = full * pct;
    const x = startX + i * (bw + gap);
    const col = c.status === 'FAULT' ? 'var(--status-bad)' : c.status === 'LOW' ? 'var(--status-warn)' : 'var(--ledger)';
    return `<rect class="cass-track" x="${x}" y="74" width="${bw}" height="${full}" rx="1.5"/>
            <rect class="cass-fill" x="${x}" y="${74 + (full - hgt)}" width="${bw}" height="${hgt}" rx="1.5" fill="${col}"/>`;
  }).join('');
  return `<svg class="atm-illo" viewBox="0 0 124 128" role="img" aria-label="ATM with ${cass.length} cassettes">
    <rect class="body" x="14" y="6" width="96" height="116" rx="7"/>
    <rect class="screen" x="26" y="16" width="72" height="40" rx="3"/>
    <path d="M34 30h30M34 38h44M34 46h20" stroke="var(--text-faint)" stroke-width="2.5" stroke-linecap="round" opacity=".5"/>
    <rect x="26" y="62" width="72" height="3" rx="1.5" fill="var(--border)"/>
    ${bars}
    <rect x="38" y="110" width="48" height="5" rx="2.5" fill="var(--narrative)" opacity=".75"/>
  </svg>`;
}

async function openCustomer(id) {
  const summary = loanBook.find((c) => c.customer_id === id);
  openDrawer(summary.name, `${id} · ${summary.bank} · ${summary.city}`, '<div class="hint">Loading credit file…</div>');

  const [detailRes] = await Promise.all([fetch(`/loans/customers/${id}`)]);
  const detail = await detailRes.json();
  const p = detail.profile, score = detail.score, stress = detail.stress;

  const bars = score.components.map((c) => {
    const pct = Math.min(100, Math.abs(c.points) / 2 * 100);
    return `<div class="score-bar-row">
      <span class="score-bar-label">${esc(c.label)}</span><span class="score-bar-pts">${c.points > 0 ? '+' : ''}${c.points}</span>
      <div class="score-bar-track"><div class="score-bar-fill ${c.points < 0 ? 'neg' : ''}" style="width:${pct}%"></div></div>
    </div>`;
  }).join('');

  const spark = sparklineSvg(p.eom_balances);

  $('#drawer-body').innerHTML = `
    <div>
      <div class="kv-grid">
        <span class="k">Employment</span><span class="v">${esc(p.employment)}</span>
        <span class="k">Net monthly income</span><span class="v">${rs(p.net_monthly_income)}</span>
        <span class="k">Account age</span><span class="v">${p.account_age_years} yrs</span>
        <span class="k">Salary months (12)</span><span class="v">${p.salary_months_12}/12</span>
        <span class="k">Avg balance (6m)</span><span class="v">${rs(p.avg_balance_6m)}</span>
        <span class="k">Previous loan</span><span class="v">${esc(p.previous_loan)}</span>
      </div>
    </div>
    <div>
      <div class="section-note" style="margin-bottom:8px;">Relationship score — ${score.score} (${score.band})</div>
      <div class="score-bars">${bars}</div>
    </div>
    ${p.eom_balances.length ? `<div>
      <div class="section-note" style="margin-bottom:8px;">6-month balance trend — ${stress.stressed ? `stressed, ${stress.days_below_5k} days under Rs 5,000` : 'liquid'}</div>
      <div class="sparkline-wrap">${spark}<div class="sparkline-labels">${rs(p.eom_balances[0])} → <span class="hi">${rs(p.eom_balances[p.eom_balances.length - 1])}</span></div></div>
    </div>` : ''}
    <div id="decision-slot">${duoBlock('Policy verdict', 'computing…', null, null)}</div>
    <div>
      <div class="simulator">
        <div class="simulator-head"><span class="simulator-title">What-if simulator</span><span class="hint">live, mirrors policy.py in JS</span></div>
        <div class="field-row">
          <div class="field"><label>Account age (yrs)</label><input type="number" step="0.1" id="sim-age" value="${p.account_age_years}"></div>
          <div class="field"><label>Salary months /12</label><input type="number" min="0" max="12" id="sim-salmo" value="${p.salary_months_12}"></div>
          <div class="field"><label>Avg balance 6m</label><input type="number" id="sim-bal" value="${p.avg_balance_6m}"></div>
          <div class="field"><label>Previous loan</label><select id="sim-prevloan">
            <option value="clean" ${p.previous_loan === 'clean' ? 'selected' : ''}>clean</option>
            <option value="late_1_2" ${p.previous_loan === 'late_1_2' ? 'selected' : ''}>late 1-2</option>
            <option value="none" ${p.previous_loan === 'none' ? 'selected' : ''}>none</option>
          </select></div>
          <div class="field"><label>Net monthly income</label><input type="number" id="sim-income" value="${p.net_monthly_income}"></div>
          <div class="field"><label>Days &lt; Rs 5,000 (30d)</label><input type="number" min="0" max="30" id="sim-stress" value="${p.days_below_5k_30d}"></div>
          <div class="field field-full"><label>Total monthly obligations</label><input type="number" id="sim-obl" value="${(p.obligations || []).reduce((s, o) => s + o.monthly_amount, 0)}"></div>
        </div>
        <div class="field-row">
          <div class="field-check"><input type="checkbox" id="sim-bounce" ${p.cheque_bounce_12m ? 'checked' : ''}><label>Cheque/DD bounce (12m)</label></div>
          <div class="field-check"><input type="checkbox" id="sim-dpd" ${p.ecib_dpd90_24m ? 'checked' : ''}><label>eCIB 90+ DPD (24m)</label></div>
          <div class="field-check"><input type="checkbox" id="sim-writeoff" ${p.ecib_writeoff ? 'checked' : ''}><label>eCIB write-off</label></div>
        </div>
        <button class="btn btn-primary btn-block" id="sim-run">Recompute</button>
        <div class="sim-result" id="sim-result"></div>
        <div class="sim-caption">Deterministic — same formula as app/agents/loans/policy.py. No LLM involved.</div>
      </div>
    </div>
  `;

  $('#sim-run').addEventListener('click', () => runSimulator());
  runSimulator(); // seed with the customer's real numbers

  // narrated policy decision, live
  // Two phases: the policy verdict is pure computation and lands instantly,
  // so show it straight away rather than making the reader stare at a spinner
  // for the ~30s the model takes to write its note.
  const decisionLedger = (d) => {
    const lines = [
      `Decision: ${d.action}`,
      `Score: ${d.score.score} (${d.score.band})`,
      `Cash stress: ${d.stress.stressed ? 'yes' : 'no'} (${d.stress.days_below_5k} days under Rs 5,000)`,
    ];
    if (d.offer) lines.push(`Offer: ${rs(d.offer.amount)} / ${d.offer.tenor_months}mo @ ${(d.offer.annual_rate * 100).toFixed(1)}% — instalment ${rs(d.offer.emi)}, DBR ${d.offer.dbr_pct}%`);
    return lines.join('\n');
  };

  const fast = await (await fetch('/loans/proactive_offer_decision', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ customer_id: id, use_llm: false }),
  })).json();
  const slot = $('#decision-slot');
  if (!slot) return;               // drawer already closed
  slot.innerHTML = duoBlock('Policy verdict', decisionLedger(fast.decision), null, null);

  const full = await (await fetch('/loans/proactive_offer_decision', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ customer_id: id, use_llm: true }),
  })).json();
  // the reader may have moved on to another customer while the model wrote
  if ($('#drawer-title')?.textContent !== summary.name) return;
  $('#decision-slot').innerHTML = duoBlock('Policy verdict', decisionLedger(full.decision), full.narrative, full.narrative_source);
}

function runSimulator() {
  const p = {
    account_age_years: parseFloat($('#sim-age').value) || 0,
    salary_months_12: parseInt($('#sim-salmo').value) || 0,
    avg_balance_6m: parseFloat($('#sim-bal').value) || 0,
    previous_loan: $('#sim-prevloan').value,
    net_monthly_income: parseFloat($('#sim-income').value) || 1,
    days_below_5k_30d: parseInt($('#sim-stress').value) || 0,
    total_obligations: parseFloat($('#sim-obl').value) || 0,
    cheque_bounce_12m: $('#sim-bounce').checked,
    ecib_dpd90_24m: $('#sim-dpd').checked,
    ecib_writeoff: $('#sim-writeoff').checked,
  };
  const r = simulateDecision(p);
  $('#sim-result').innerHTML = `
    <div class="action ${r.action}">${r.action}</div>
    <div>score ${r.score} (${r.band}) · ${r.stressed ? 'genuinely stressed' : 'liquid'}</div>
    ${r.offer ? `<div style="margin-top:6px">offer ${rs(r.offer.amount)} / ${r.offer.tenor}mo, instalment ${rs(r.offer.instalment)}, DBR ${r.offer.dbr}%</div>` : ''}
    <div style="margin-top:8px; color: var(--text-muted); font-size: 11.5px;">${esc(r.reason)}</div>
  `;
}

function sparklineSvg(values, w = 160, h = 40) {
  if (!values || values.length < 2) return '';
  const min = Math.min(...values), max = Math.max(...values);
  const range = max - min || 1;
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * w},${h - ((v - min) / range) * h}`).join(' ');
  const last = values[values.length - 1] < values[0] ? 'var(--status-bad)' : 'var(--status-good)';
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><polyline points="${pts}" fill="none" stroke="${last}" stroke-width="1.75" stroke-linejoin="round" stroke-linecap="round"/></svg>`;
}

// -- JS mirror of app/agents/loans/policy.py, for the live what-if simulator --
function relationshipScore(p) {
  const comps = [];
  let score = 0;
  const add = (label, pts) => { comps.push({ label, points: pts }); score += pts; };
  if (p.account_age_years >= 3) add('account age 3+ years', 2);
  else if (p.account_age_years >= 1) add('account age 1-3 years', 1);
  else add('account age under 1 year', 0);
  if (p.salary_months_12 >= 11) add('salary 11+/12 months', 2);
  else if (p.salary_months_12 >= 9) add('salary 9-10/12 months', 1);
  else add('salary under 9/12 months', 0);
  if (p.avg_balance_6m > 50000) add('avg balance above Rs 50,000', 2);
  else if (p.avg_balance_6m >= 15000) add('avg balance Rs 15,000-50,000', 1);
  else add('avg balance below Rs 15,000', 0);
  if (p.previous_loan === 'clean') add('previous loan clean', 2);
  else if (p.previous_loan === 'late_1_2') add('previous loan late 1-2', 1);
  else add('no previous loan history', 0);
  if (p.cheque_bounce_12m) add('cheque/DD bounce (12m)', -3);
  if (p.ecib_dpd90_24m) add('eCIB 90+ DPD (24m)', -4);
  if (p.ecib_writeoff) add('eCIB write-off/litigation', -6);
  const band = score >= 6 ? 'STRONG' : score >= 3 ? 'ACCEPTABLE' : score >= 0 ? 'THIN' : 'POOR';
  return { score, band, comps };
}
function emiJs(principal, annualRate, months) {
  const r = annualRate / 12;
  if (r === 0) return principal / months;
  const g = Math.pow(1 + r, months);
  return principal * r * g / (g - 1);
}
function invertEmiJs(instalment, annualRate, months) {
  const r = annualRate / 12;
  if (r === 0) return instalment * months;
  const g = Math.pow(1 + r, months);
  return instalment * (g - 1) / (r * g);
}
function maxAffordableLoanJs(income, obligations, tenor) {
  const headroom = 0.40 * income - obligations;
  if (headroom <= 0) return 0;
  const raw = invertEmiJs(headroom, 0.283, tenor);
  const floored = Math.floor(raw / 10000) * 10000;
  if (floored < 50000) return 0;
  return Math.min(floored, 3000000);
}
function simulateDecision(p) {
  const { score, band, comps } = relationshipScore(p);
  const stressed = p.days_below_5k_30d >= 10;
  let action, offer = null, reason;
  if (band === 'POOR') { action = 'DECLINE'; reason = `Score ${score} (POOR) — no unsecured credit; stress makes it more dangerous, not less.`; }
  else if (band === 'THIN') { action = 'MONITOR'; reason = `Score ${score} (THIN) — file too thin for a proactive offer.`; }
  else if (!stressed) { action = 'MONITOR'; reason = `Score ${score} (${band}) but liquid — no genuine stress; offer would land as marketing.`; }
  else if (p.net_monthly_income < 50000) { action = 'MONITOR'; reason = `Score ${score} (${band}) with stress, but income is below the Rs 50,000 product floor.`; }
  else {
    const amount = maxAffordableLoanJs(p.net_monthly_income, p.total_obligations, 48);
    if (amount <= 0) { action = 'MONITOR'; reason = `Score ${score} (${band}) with stress, but no instalment headroom under the 40% DBR cap.`; }
    else {
      action = 'OFFER';
      const inst = emiJs(amount, 0.283, 48);
      offer = { amount, tenor: 48, instalment: Math.round(inst), dbr: ((p.total_obligations + inst) / p.net_monthly_income * 100).toFixed(1) };
      reason = `Score ${score} (${band}) with genuine stress (${p.days_below_5k_30d} days under Rs 5,000) — sized under the 40% DBR cap, longest tenor.`;
    }
  }
  return { score, band, comps, stressed, action, offer, reason };
}

// ============================================================ ATM OPS

const MULTAN_ALERTS = [
  { atm_id: 'FWBL-MUX-9514', location: 'Shahi Bazaar', alert_code: 'CASH_LOW', description: 'cash below threshold', open_minutes: 111 },
  { atm_id: 'NBP-MUX-8535', location: 'Ring Road', alert_code: 'DISPENSER_FAULT', description: 'dispenser hardware fault', open_minutes: 72 },
  { atm_id: 'HMB-MUX-9222', location: 'Shahi Bazaar', alert_code: 'UPS_ON_BATTERY', description: 'running on UPS battery', open_minutes: 127 },
  { atm_id: 'BIPL-MUX-7174', location: 'College Road', alert_code: 'UPS_ON_BATTERY', description: 'running on UPS battery', open_minutes: 43 },
  { atm_id: 'HBL-MUX-4499', location: 'GT Road', alert_code: 'DOOR_OPEN', description: 'safe door open unscheduled', open_minutes: 194 },
  { atm_id: 'AKBL-MUX-4493', location: 'College Road', alert_code: 'PRINTER_PAPER_OUT', description: 'receipt paper out', open_minutes: 224 },
];

// Two machines transcribed verbatim from the training data; four more in the
// same realistic style (bank/city/model conventions) for a fuller fleet.
const FLEET = [
  { atm_id: 'BIPL-SWL-5079', bank: 'BIPL', location: 'Katchery Chowk, Sahiwal', area_type: 'on-site (branch lobby)', model: 'Wincor Nixdorf Procash 2050xe', status: 'ONLINE',
    cassettes: [{ denom: 1000, notes: 1155, capacity: 2500, status: 'OK' }, { denom: 1000, notes: 1715, capacity: 2500, status: 'OK' }, { denom: 500, notes: 1074, capacity: 2000, status: 'OK' }, { denom: 500, notes: 823, capacity: 2000, status: 'OK' }],
    dispense_rate_per_hour: 210000, low_cash_threshold: 150000, cit_time: '10:10', cit_hours_from_now: 3 },
  { atm_id: 'MEBL-KDU-6892', bank: 'MEBL', location: 'Katchery Chowk, Skardu', area_type: 'industrial estate', model: 'Hyosung MX 8200QT', status: 'ONLINE',
    cassettes: [{ denom: 5000, notes: 783, capacity: 3000, status: 'FAULT', note: 'pick failure, cassette locked out' }, { denom: 1000, notes: 2253, capacity: 3000, status: 'OK' }, { denom: 1000, notes: 1158, capacity: 3000, status: 'FAULT', note: 'pick failure, cassette locked out' }, { denom: 500, notes: 62, capacity: 2000, status: 'LOW', note: 'below low-cash threshold' }],
    dispense_rate_per_hour: 95000, low_cash_threshold: 100000, cit_time: '14:00', cit_hours_from_now: 5 },
  { atm_id: 'HBL-GLB-1102', bank: 'HBL', location: 'Gulberg, Lahore', area_type: 'commercial district', model: 'NCR SelfServ 26', status: 'ONLINE',
    cassettes: [{ denom: 5000, notes: 1600, capacity: 2000, status: 'OK' }, { denom: 1000, notes: 2100, capacity: 2500, status: 'OK' }],
    dispense_rate_per_hour: 140000, low_cash_threshold: 300000, cit_time: '18:00', cit_hours_from_now: 9 },
  { atm_id: 'UBL-DHA-2207', bank: 'UBL', location: 'DHA Phase 5, Karachi', area_type: 'shopping mall', model: 'NCR 6622', status: 'ONLINE',
    cassettes: [{ denom: 5000, notes: 1850, capacity: 2000, status: 'OK' }, { denom: 1000, notes: 2400, capacity: 2500, status: 'OK' }],
    dispense_rate_per_hour: 165000, low_cash_threshold: 350000, cit_time: '20:00', cit_hours_from_now: 11 },
  { atm_id: 'MCB-BLU-3311', bank: 'MCB', location: 'Blue Area, Islamabad', area_type: 'on-site (branch lobby)', model: 'Wincor Nixdorf Procash 2050xe', status: 'ONLINE',
    cassettes: [{ denom: 5000, notes: 210, capacity: 2000, status: 'OK' }, { denom: 1000, notes: 480, capacity: 2500, status: 'LOW', note: 'below low-cash threshold' }],
    dispense_rate_per_hour: 120000, low_cash_threshold: 200000, cit_time: '11:30', cit_hours_from_now: 2 },
  { atm_id: 'MEZN-SDR-4415', bank: 'Meezan', location: 'Saddar, Faisalabad', area_type: 'bus terminal', model: 'Hyosung MX 5600', status: 'ONLINE',
    cassettes: [{ denom: 1000, notes: 45, capacity: 2500, status: 'OK' }, { denom: 500, notes: 60, capacity: 2000, status: 'OK' }],
    dispense_rate_per_hour: 55000, low_cash_threshold: 60000, cit_time: '09:00', cit_hours_from_now: 1 },
];

function fleetStatus(m) {
  const hasFault = m.cassettes.some((c) => c.status === 'FAULT');
  const total = m.cassettes.reduce((s, c) => s + c.denom * c.notes, 0);
  const usable = m.cassettes.filter((c) => c.status !== 'FAULT').reduce((s, c) => s + c.denom * c.notes, 0);
  const hoursLeft = usable / m.dispense_rate_per_hour;
  if (hasFault || hoursLeft < m.cit_hours_from_now) return 'bad';
  if (hoursLeft < m.cit_hours_from_now * 1.5) return 'warn';
  return 'good';
}

async function loadAtmView() {
  // Alerts, live-triaged. use_llm:false on purpose — this panel renders
  // facts.triaged only and never shows the narrative, so asking the model
  // to write one would just cost ~30s of page-load latency for nothing.
  const alertRes = await fetch('/atm/tasks/alert_triage', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ alerts: MULTAN_ALERTS, use_llm: false }),
  });
  const alertData = await alertRes.json();
  const triaged = alertData.facts.triaged;
  $('#alerts-list').innerHTML = triaged.map((a) => `
    <div class="alert-row">
      <span class="alert-sev p${a.severity}">P${a.severity}</span>
      <div class="alert-main"><div class="aid">${esc(a.atm_id)} — ${esc(a.location)}</div><div class="adesc">${esc(a.alert_code)}: ${esc(a.description)} → ${esc(a.action)}</div></div>
      <span class="alert-open">${a.open_minutes} min open</span>
    </div>`).join('');

  // fleet grid, computed client-side for coloring, drill-down calls the live API
  const goodN = FLEET.filter((m) => fleetStatus(m) === 'good').length;
  const totalCash = FLEET.reduce((s, m) => s + m.cassettes.reduce((a, c) => a + c.denom * c.notes, 0), 0);
  $('#atm-stats').innerHTML = [
    stat(FLEET.length, 'Machines', ''),
    stat(goodN, 'Healthy', 'good'),
    stat(FLEET.length - goodN, 'Needs attention', FLEET.length - goodN ? 'warn' : ''),
    stat(triaged.filter((a) => a.severity === 1).length, 'P1 alerts', triaged.some((a) => a.severity === 1) ? 'bad' : ''),
    { html: `<div class="stat-tile"><div class="n">${(totalCash / 1e6).toFixed(1)}M</div><div class="lbl">Cash in fleet</div></div>` }.html,
  ].join('');
  $('#fleet-note').textContent = `${FLEET.length} machines · click any card for a live cash run-out forecast`;

  $('#fleet-grid').innerHTML = FLEET.map((m) => {
    const status = fleetStatus(m);
    const total = m.cassettes.reduce((s, c) => s + c.denom * c.notes, 0);
    const cap = m.cassettes.reduce((s, c) => s + c.denom * c.capacity, 0);
    const pct = Math.min(100, Math.round((total / cap) * 100));
    return `<div class="machine-card status-${status}" tabindex="0" data-id="${esc(m.atm_id)}">
      <div class="mid">${esc(m.atm_id)}</div>
      <div class="mloc">${esc(m.location)}</div>
      <div class="mbar-track"><div class="mbar-fill" style="width:${pct}%; background: var(--status-${status === 'good' ? 'good' : status === 'warn' ? 'warn' : 'bad'})"></div></div>
      <div class="mcash"><span>${rs(total)}</span><span>${pct}% full</span></div>
    </div>`;
  }).join('');
  $all('.machine-card').forEach((card) => card.addEventListener('click', () => openMachine(card.dataset.id)));
}

async function openMachine(id) {
  const m = FLEET.find((x) => x.atm_id === id);
  const hasFault = m.cassettes.some((c) => c.status === 'FAULT');
  openDrawer(m.atm_id, `${m.bank} · ${m.location} · ${m.model}`, '<div class="hint">Loading telemetry…</div>');

  const cassetteRows = m.cassettes.map((c) => `
    <span class="k">Rs ${c.denom} cassette</span><span class="v">${c.notes.toLocaleString()} notes · ${c.status}${c.note ? ' — ' + esc(c.note) : ''}</span>`).join('');

  let html = `${atmIllustration(m)}<div class="kv-grid">${cassetteRows}</div>`;
  $('#drawer-body').innerHTML = html + `<div id="atm-decision-slot">${duoBlock('Computing…', '…', null, null)}</div>`;

  const task = hasFault ? 'cassette_status_triage' : 'cash_runout_forecast';
  const label = hasFault ? 'Cassette triage' : 'Cash run-out forecast';
  const payload = hasFault
    ? { machine: toMachinePayload(m) }
    : {
        machine: toMachinePayload(m),
        inp: { current_time: nowHHMM(), dispense_rate_per_hour: m.dispense_rate_per_hour, low_cash_threshold: m.low_cash_threshold, cit_time: m.cit_time, cit_hours_from_now: m.cit_hours_from_now },
      };
  const toLedger = (f) => hasFault
    ? `Dispensable now: ${rs(f.usable_cash)} from ${f.working_cassette_count} working cassette(s)\nLargest composable withdrawal: ${rs(f.max_single_withdrawal)}\nRecommendation: ${f.keep_in_service ? 'keep in service' : 'take offline'}\nActions:\n${f.actions.map((a) => '  - ' + a).join('\n')}`
    : `Total usable cash: ${rs(f.total_cash)}\nDispense rate: ${rs(f.dispense_rate_per_hour)}/hour\nLow-cash threshold hit: ${f.low_cash_time} (in ${f.hours_to_low.toFixed(1)}h)\nFully empty: ${f.empty_time} (in ${f.hours_to_empty.toFixed(1)}h)\nCIT scheduled: ${f.cit_time}\nSurvives to CIT: ${f.survives_to_cit ? 'yes' : 'NO'} (headroom ${f.headroom_hours}h)`;

  // Same two-phase pattern as the loans drawer: the telemetry maths is
  // instant, so render it immediately and let the prose catch up.
  const post = (useLlm) => fetch(`/atm/tasks/${task}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...payload, use_llm: useLlm }),
  }).then((r) => r.json());

  const fast = await post(false);
  const slot = $('#atm-decision-slot');
  if (!slot) return;                // drawer already closed
  slot.innerHTML = duoBlock(label, toLedger(fast.facts), null, null);

  const full = await post(true);
  if ($('#drawer-title')?.textContent !== m.atm_id) return;   // reader moved on
  $('#atm-decision-slot').innerHTML = duoBlock(label, toLedger(full.facts), full.narrative, full.narrative_source);
}
function toMachinePayload(m) {
  return { atm_id: m.atm_id, bank: m.bank, location: m.location, area_type: m.area_type, model: m.model, status: m.status, cassettes: m.cassettes };
}
function nowHHMM() { const d = new Date(); return String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0'); }

// ============================================================ TASK LEDGER

const LOAN_OPS = new Set(['relationship_scoring', 'low_cash_detection', 'proactive_offer_decision', 'decline_with_alternatives', 'portfolio_triage', 'dbr_calculation', 'max_affordable_loan', 'ecib_report_reading', 'delinquency_risk_grading', 'restructuring_assessment', 'risk_based_pricing', 'topup_eligibility']);
const ATM_OPS = new Set(['alert_triage', 'cash_carrying_cost', 'cash_load_planning', 'cash_reconciliation', 'cash_runout_forecast', 'cassette_status_triage', 'cit_route_planning', 'demand_forecast', 'denomination_mix_planning', 'eod_position_report', 'fault_root_cause', 'growth_analysis', 'interbank_settlement', 'ranking', 'replenishment_priority', 'security_anomaly_assessment', 'share_analysis', 'surge_capacity_planning', 'trend_summary', 'uptime_sla_check']);

async function loadTasksView() {
  const r = await fetch('/tasks');
  tasksSchema = await r.json();

  const groups = [
    { title: 'Loans — credit ops', domain: 'loans', tasks: Object.keys(tasksSchema.loans).filter((t) => LOAN_OPS.has(t)) },
    { title: 'Loans — customer facing', domain: 'loans', tasks: Object.keys(tasksSchema.loans).filter((t) => !LOAN_OPS.has(t)) },
    { title: 'ATM — network ops', domain: 'atm', tasks: Object.keys(tasksSchema.atm).filter((t) => ATM_OPS.has(t)) },
    { title: 'ATM — customer facing', domain: 'atm', tasks: Object.keys(tasksSchema.atm).filter((t) => !ATM_OPS.has(t)) },
  ];

  $('#task-groups').innerHTML = groups.map((g) => `
    <div class="task-group" data-group>
      <div class="task-group-title">${esc(g.title)} <span class="hint">(${g.tasks.length})</span></div>
      <div class="task-grid">${g.tasks.map((t) => `<button class="task-tile" data-task="${esc(t)}" data-domain="${g.domain}"><div class="tname">${esc(t)}</div><div class="tdomain">${g.domain}</div></button>`).join('')}</div>
    </div>`).join('');

  $all('.task-tile').forEach((btn) => btn.addEventListener('click', () => openTaskRunner(btn.dataset.task, btn.dataset.domain)));

  $('#task-search').addEventListener('input', (e) => {
    const q = e.target.value.trim().toLowerCase();
    $all('.task-tile').forEach((btn) => { btn.style.display = btn.dataset.task.includes(q) ? '' : 'none'; });
    $all('[data-group]').forEach((g) => {
      const anyVisible = $all('.task-tile', g).some((b) => b.style.display !== 'none');
      g.style.display = anyVisible ? '' : 'none';
    });
  });
}

function dummyForTypeString(typeStr, fieldName) {
  const t = (typeStr || '').replace(/^<class '/, '').replace(/'>$/, '');
  const n = (fieldName || '').toLowerCase();
  if (t.includes('bool')) return true;
  if (t.includes('float')) return 0.2;
  if (t.includes('int')) {
    if (/amount|income|balance|cash|value|capacity|notes/.test(n)) return 100000;
    if (/minutes|days/.test(n)) return 5;
    return 5;
  }
  if (t.startsWith('list[')) return [];
  if (n === 'customer_id') return 'C001';
  if (n.includes('time')) return '10:00';
  if (n === 'rate_type') return 'reducing';
  if (n === 'status') return 'ONLINE';
  if (n === 'channel') return 'ATM';
  if (n === 'date') return '2026-01-01';
  if (n === 'atm_id' || n === 'id') return 'ATM-001';
  return 'example';
}

// spec is either a plain type-name string (leaf) or a recursive object
// {type, fields: {...}} for a nested dataclass, {type, item_fields: {...}}
// for a list[dataclass] — mirrors dispatch.py's describe_type() on the server.
function dummyForType(spec, fieldName) {
  if (spec && typeof spec === 'object') {
    if (spec.fields) {
      const sub = {};
      for (const [fn, fspec] of Object.entries(spec.fields)) sub[fn] = dummyForType(fspec, fn);
      return sub;
    }
    if (spec.item_fields) {
      const sub = {};
      for (const [fn, fspec] of Object.entries(spec.item_fields)) sub[fn] = dummyForType(fspec, fn);
      return [sub];
    }
    return dummyForTypeString(spec.type, fieldName);
  }
  return dummyForTypeString(spec, fieldName);
}

function synthesizeExample(shape) {
  const out = {};
  for (const [name, spec] of Object.entries(shape)) out[name] = dummyForType(spec, name);
  return out;
}

function openTaskRunner(taskName, domain) {
  const shape = tasksSchema[domain][taskName];
  const example = synthesizeExample(shape);
  const endpoint = domain === 'loans' ? `/loans/${taskName}` : `/atm/tasks/${taskName}`;

  const fieldList = Object.entries(shape).map(([name, spec]) => {
    const t = spec.fields ? spec.type : spec.item_fields ? spec.type : spec.type.replace(/^<class '|'>$/g, '');
    return `<div><span class="k">${esc(name)}</span><span class="hint">${spec.required ? ' (required)' : ' (optional)'} — ${esc(t)}</span></div>`;
  }).join('');

  openDrawer(taskName, `${domain} task — edit the JSON below and run it live`, `
    <div>
      <div class="section-note" style="margin-bottom:8px;">Expected fields</div>
      <div class="kv-grid" style="grid-template-columns: 1fr;">${fieldList}</div>
    </div>
    <div>
      <div class="field"><label>Request body (editable)</label>
        <textarea id="runner-json" rows="14" style="font-size:12px;">${esc(JSON.stringify(example, null, 2))}</textarea>
      </div>
      <div class="field-check" style="margin: 10px 0;"><input type="checkbox" id="runner-use-llm" checked><label>Narrate with the fine-tuned model</label></div>
      <button class="btn btn-primary btn-block" id="runner-run">Run task</button>
    </div>
    <div id="runner-result"></div>
  `);

  $('#runner-run').addEventListener('click', async () => {
    let body;
    try { body = JSON.parse($('#runner-json').value); } catch (e) { $('#runner-result').innerHTML = `<div class="hint" style="color:var(--status-bad)">Invalid JSON: ${esc(e.message)}</div>`; return; }
    body.use_llm = $('#runner-use-llm').checked;
    $('#runner-result').innerHTML = duoBlock('Running…', '…', null, null);
    const btn = $('#runner-run'); btn.disabled = true;
    try {
      const r = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const d = await r.json();
      if (!r.ok) {
        $('#runner-result').innerHTML = `<div class="hint" style="color:var(--status-bad)">${r.status}: ${esc(d.detail || JSON.stringify(d))}</div>`;
      } else if (domain === 'loans') {
        $('#runner-result').innerHTML = duoBlock('Facts', JSON.stringify(d.facts ?? d, null, 2), d.narrative, d.narrative_source);
      } else {
        $('#runner-result').innerHTML = duoBlock('Facts', JSON.stringify(d.facts, null, 2), d.narrative, d.narrative_source);
      }
    } catch (e) {
      $('#runner-result').innerHTML = `<div class="hint" style="color:var(--status-bad)">Request failed: ${esc(e.message)}</div>`;
    } finally {
      btn.disabled = false;
    }
  });
}
