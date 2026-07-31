// ============================================================
// Cash Analytics — how fast the money is going and when it runs out.
//
// Every number on this page is computed by the engine: the charts are drawn
// from an hourly dispense model, and the insight cards below are eight of the
// 51 task types called live against this fleet (use_llm:false, so they are
// instant — the prose layer isn't needed to answer "when does it empty").
// ============================================================

// A day's shape at a Pakistani retail ATM: dead overnight, a mid-morning rise,
// a lunch dip, then the heaviest run in the early evening. Weights are relative
// and get normalised against each machine's own daily volume.
const HOUR_WEIGHTS = [
  0.5, 0.3, 0.2, 0.2, 0.3, 0.6, 1.2, 2.4, 4.0, 5.6, 6.4, 6.0,
  5.2, 5.8, 6.6, 7.2, 7.8, 8.4, 8.0, 6.4, 4.4, 2.8, 1.6, 0.9,
];
const WEIGHT_SUM = HOUR_WEIGHTS.reduce((a, b) => a + b, 0);

const ATM_TIP = (() => {
  const el = document.createElement('div');
  el.className = 'chart-tip';
  document.body.appendChild(el);
  return el;
})();
function showTip(evt, html) {
  ATM_TIP.innerHTML = html;
  ATM_TIP.classList.add('on');
  const pad = 14;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  const r = ATM_TIP.getBoundingClientRect();
  if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
  ATM_TIP.style.left = `${x}px`;
  ATM_TIP.style.top = `${y}px`;
}
const hideTip = () => ATM_TIP.classList.remove('on');

let anSelected = null;

function machineDaily(m) { return m.dispense_rate_per_hour * 24; }
function hourlySeries(m) {
  const daily = machineDaily(m);
  return HOUR_WEIGHTS.map((w) => Math.round((w / WEIGHT_SUM) * daily));
}
function usableCash(m) {
  return m.cassettes.filter((c) => c.status !== 'FAULT').reduce((s, c) => s + c.denom * c.notes, 0);
}

// The one rate every part of this page runs on. Using the trailing-3h average
// rather than the flat daily mean matters: at 17:00 a machine is burning far
// faster than its 24h average, and "when does it empty" has to answer for the
// rate it is actually running at. Charts and hero used to disagree because one
// used each — same machine, two different answers.
function currentRate(m) {
  const hourly = hourlySeries(m);
  const nowH = new Date().getHours() + new Date().getMinutes() / 60;
  const recent = hourly.slice(Math.max(0, Math.floor(nowH) - 3), Math.floor(nowH) + 1);
  return Math.max(1, recent.reduce((a, b) => a + b, 0) / Math.max(1, recent.length));
}
const hoursLeftFor = (m) => usableCash(m) / currentRate(m);
function statusOf(hours, citHours) {
  if (hours < citHours) return 'critical';
  if (hours < citHours * 1.5) return 'low';
  return 'healthy';
}
const STATUS_INK = { critical: 'var(--status-bad)', low: 'var(--status-warn)', healthy: 'var(--status-good)' };
const fmtHrs = (h) => (h >= 100 ? `${Math.round(h)}h` : `${h.toFixed(1)}h`);
const compact = (n) => (n >= 1e7 ? `${(n / 1e7).toFixed(2)} Cr` : n >= 1e5 ? `${(n / 1e5).toFixed(1)} L` : n.toLocaleString('en-IN'));

// ---------------------------------------------------------- burn-down

function drawBurn(m) {
  const W = 900, H = 260, L = 62, R = 16, T = 12, B = 30;
  const pw = W - L - R, ph = H - T - B;
  const hourly = hourlySeries(m);
  const now = new Date();
  const nowH = now.getHours() + now.getMinutes() / 60;

  // opening cash = what's in it now, plus everything already dispensed today
  const dispensedSoFar = hourly.slice(0, Math.floor(nowH)).reduce((a, b) => a + b, 0)
    + hourly[Math.floor(nowH)] * (nowH % 1);
  const opening = usableCash(m) + dispensedSoFar;

  const observed = [];
  for (let h = 0; h <= Math.floor(nowH); h++) {
    const spent = hourly.slice(0, h).reduce((a, b) => a + b, 0);
    observed.push({ h, cash: Math.max(0, opening - spent) });
  }
  observed.push({ h: nowH, cash: usableCash(m) });

  // projection forward at the trailing-3h rate — the shared basis (currentRate)
  const rate = currentRate(m);
  const hoursLeft = usableCash(m) / rate;
  const emptyAt = nowH + hoursLeft;
  const projected = [{ h: nowH, cash: usableCash(m) }];
  for (let h = Math.ceil(nowH); h <= Math.min(30, Math.ceil(emptyAt)); h++) {
    projected.push({ h, cash: Math.max(0, usableCash(m) - (h - nowH) * rate) });
  }
  if (emptyAt <= 30) projected.push({ h: emptyAt, cash: 0 });

  const maxH = Math.max(24, Math.min(30, Math.ceil(emptyAt)));
  const maxY = opening * 1.06 || 1;
  const X = (h) => L + (h / maxH) * pw;
  const Y = (v) => T + ph - (v / maxY) * ph;

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(maxY * f));
  const grid = ticks.map((v) => `<line class="ax-line" x1="${L}" y1="${Y(v)}" x2="${W - R}" y2="${Y(v)}"/>
     <text class="ax-text" x="${L - 8}" y="${Y(v) + 3}" text-anchor="end">${compact(v)}</text>`).join('');
  const xt = [0, 6, 12, 18, 24, 30].filter((h) => h <= maxH).map((h) =>
    `<text class="ax-text" x="${X(h)}" y="${H - B + 16}" text-anchor="middle">${String(h % 24).padStart(2, '0')}:00</text>`).join('');

  const line = (pts) => pts.map((p, i) => `${i ? 'L' : 'M'}${X(p.h).toFixed(1)} ${Y(p.cash).toFixed(1)}`).join(' ');
  const area = `${line(observed)} L${X(observed[observed.length - 1].h).toFixed(1)} ${Y(0)} L${X(0)} ${Y(0)} Z`;
  const st = statusOf(hoursLeft, m.cit_hours_from_now);
  const ink = STATUS_INK[st];

  const emptyMark = emptyAt <= maxH ? `
    <line class="ax-line" x1="${X(emptyAt)}" y1="${T}" x2="${X(emptyAt)}" y2="${Y(0)}" stroke="${ink}" opacity=".5"/>
    <circle cx="${X(emptyAt)}" cy="${Y(0)}" r="4.5" fill="${ink}" stroke="var(--surface)" stroke-width="2"/>
    <text class="mark-label" x="${X(emptyAt)}" y="${T - 1}" text-anchor="middle" fill="${ink}">empty ${String(Math.floor(emptyAt) % 24).padStart(2, '0')}:${String(Math.round((emptyAt % 1) * 60)).padStart(2, '0')}</text>` : '';

  document.getElementById('burn-chart').innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Cash remaining at ${m.atm_id} through the day, with the projection to empty. Table view below.">
      ${grid}
      <path d="${area}" fill="var(--ledger)" opacity="0.10"/>
      <path d="${line(observed)}" fill="none" stroke="var(--ledger)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
      <path d="${line(projected)}" class="proj-line" fill="none" stroke="${ink}" stroke-width="2" stroke-linecap="round"/>
      <circle cx="${X(nowH)}" cy="${Y(usableCash(m))}" r="4.5" fill="var(--ledger)" stroke="var(--surface)" stroke-width="2"/>
      <text class="mark-label" x="${X(nowH) + 8}" y="${Y(usableCash(m)) - 8}">now · Rs ${compact(usableCash(m))}</text>
      ${emptyMark}
      <line class="ax-line" x1="${L}" y1="${Y(0)}" x2="${W - R}" y2="${Y(0)}"/>
      ${xt}
      <rect id="burn-hit" x="${L}" y="${T}" width="${pw}" height="${ph}" fill="transparent"/>
    </svg>
    <div class="chart-legend">
      <span><i class="lg-key" style="background:var(--ledger)"></i>dispensed so far today</span>
      <span><i class="lg-key" style="background:${ink}"></i>projected at the last-3h rate</span>
    </div>`;

  document.getElementById('burn-sub').textContent =
    `${m.atm_id} · opening Rs ${compact(opening)} · now ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;

  // crosshair + tooltip
  const svg = document.querySelector('#burn-chart svg');
  const hit = document.getElementById('burn-hit');
  hit.addEventListener('mousemove', (e) => {
    const box = svg.getBoundingClientRect();
    const hx = ((e.clientX - box.left) / box.width) * W;
    const h = Math.max(0, Math.min(maxH, ((hx - L) / pw) * maxH));
    const isProj = h > nowH;
    const cash = isProj ? Math.max(0, usableCash(m) - (h - nowH) * rate)
                        : Math.max(0, opening - hourly.slice(0, Math.floor(h)).reduce((a, b) => a + b, 0));
    showTip(e, `<span class="tip-k">${String(Math.floor(h) % 24).padStart(2, '0')}:${String(Math.round((h % 1) * 60)).padStart(2, '0')}</span> &nbsp; Rs ${compact(Math.round(cash))}<br><span class="tip-k">${isProj ? 'projected' : 'observed'}</span>`);
  });
  hit.addEventListener('mouseleave', hideTip);

  document.getElementById('burn-table').innerHTML = tableOf(
    ['Hour', 'Dispensed', 'Cash left'],
    hourly.map((v, h) => [`${String(h).padStart(2, '0')}:00`, `Rs ${v.toLocaleString('en-IN')}`,
      `Rs ${Math.max(0, opening - hourly.slice(0, h).reduce((a, b) => a + b, 0)).toLocaleString('en-IN')}`]));

  return { opening, hoursLeft, emptyAt, rate, status: st, hourly };
}

// ---------------------------------------------------------- runway bars

function drawRunway(fleet) {
  const rows = fleet.map((m) => {
    const cash = usableCash(m);
    const h = hoursLeftFor(m);              // same basis as the hero and burn chart
    return { m, cash, h, st: statusOf(h, m.cit_hours_from_now) };
  }).sort((a, b) => a.h - b.h);

  const W = 440, rowH = 34, L = 108, R = 58;
  const H = rows.length * rowH + 12;
  const maxH = Math.max(...rows.map((r) => r.h)) || 1;
  const bw = 14;   // ≤24px, with the band's leftover left as air

  const bars = rows.map((r, i) => {
    const y = i * rowH + 8;
    const w = Math.max(3, (r.h / maxH) * (W - L - R));
    const ink = STATUS_INK[r.st];
    return `<g class="runway-row" data-id="${esc(r.m.atm_id)}">
      <text class="ax-text" x="${L - 10}" y="${y + bw / 2 + 3}" text-anchor="end">${esc(r.m.atm_id)}</text>
      <rect x="${L}" y="${y}" width="${w}" height="${bw}" rx="4" fill="${ink}"/>
      <rect x="${L}" y="${y}" width="4" height="${bw}" fill="${ink}"/>
      <text class="mark-label" x="${L + w + 8}" y="${y + bw / 2 + 3.5}">${fmtHrs(r.h)}</text>
      <rect x="0" y="${y - 6}" width="${W}" height="${bw + 12}" fill="transparent" class="runway-hit"/>
    </g>`;
  }).join('');

  document.getElementById('runway-chart').innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Hours of cash left per machine. Table view below.">${bars}</svg>
    <div class="chart-legend">
      <span class="st st-critical">empties before the van arrives</span>
      <span class="st st-low">tight</span>
      <span class="st st-healthy">comfortable</span>
    </div>`;

  document.querySelectorAll('.runway-hit').forEach((el, i) => {
    const r = rows[i];
    el.addEventListener('mousemove', (e) => showTip(e,
      `<b>${esc(r.m.atm_id)}</b><br><span class="tip-k">${esc(r.m.location)}</span><br>Rs ${compact(r.cash)} · ${fmtHrs(r.h)} left<br><span class="tip-k">status</span> ${r.st}`));
    el.addEventListener('mouseleave', hideTip);
    el.addEventListener('click', () => selectMachine(r.m.atm_id));
  });

  document.getElementById('runway-table').innerHTML = tableOf(
    ['ATM', 'Usable cash', 'Hours left', 'Status'],
    rows.map((r) => [r.m.atm_id, `Rs ${r.cash.toLocaleString('en-IN')}`, fmtHrs(r.h), r.st]));
  return rows;
}

// ---------------------------------------------------------- hourly columns

function drawHourly(m, hourly) {
  const W = 440, H = 190, L = 46, R = 10, T = 10, B = 26;
  const pw = W - L - R, ph = H - T - B;
  const max = Math.max(...hourly) || 1;
  const band = pw / 24;
  const bw = Math.min(16, band - 2);          // 2px surface gap between neighbours
  const nowH = new Date().getHours();

  const cols = hourly.map((v, h) => {
    const x = L + h * band + (band - bw) / 2;
    const hgt = Math.max(2, (v / max) * ph);
    const isNow = h === nowH;
    return `<g><rect x="${x}" y="${T + ph - hgt}" width="${bw}" height="${hgt}" rx="4"
        fill="${isNow ? 'var(--ledger)' : 'var(--ledger-dim)'}" opacity="${isNow ? 1 : .55}"/>
      <rect x="${x}" y="${T + ph - Math.min(hgt, 4)}" width="${bw}" height="${Math.min(hgt, 4)}"
        fill="${isNow ? 'var(--ledger)' : 'var(--ledger-dim)'}" opacity="${isNow ? 1 : .55}"/>
      <rect x="${L + h * band}" y="${T}" width="${band}" height="${ph}" fill="transparent" class="hr-hit" data-h="${h}"/></g>`;
  }).join('');

  const ticks = [0, 0.5, 1].map((f) => Math.round(max * f));
  const grid = ticks.map((v) => `<line class="ax-line" x1="${L}" y1="${T + ph - (v / max) * ph}" x2="${W - R}" y2="${T + ph - (v / max) * ph}"/>
     <text class="ax-text" x="${L - 7}" y="${T + ph - (v / max) * ph + 3}" text-anchor="end">${compact(v)}</text>`).join('');

  document.getElementById('hourly-chart').innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Cash dispensed per hour at ${m.atm_id}. Table view below.">
      ${grid}${cols}
      <line class="ax-line" x1="${L}" y1="${T + ph}" x2="${W - R}" y2="${T + ph}"/>
      ${[0, 6, 12, 18, 23].map((h) => `<text class="ax-text" x="${L + h * band + band / 2}" y="${H - B + 15}" text-anchor="middle">${String(h).padStart(2, '0')}</text>`).join('')}
    </svg>`;
  document.getElementById('hourly-sub').textContent = `${m.atm_id} · peak ${String(hourly.indexOf(max)).padStart(2, '0')}:00`;

  document.querySelectorAll('.hr-hit').forEach((el) => {
    const h = +el.dataset.h;
    el.addEventListener('mousemove', (e) => showTip(e,
      `<span class="tip-k">${String(h).padStart(2, '0')}:00</span> &nbsp; Rs ${hourly[h].toLocaleString('en-IN')}${h === nowH ? '<br><span class="tip-k">current hour</span>' : ''}`));
    el.addEventListener('mouseleave', hideTip);
  });

  document.getElementById('hourly-table').innerHTML = tableOf(['Hour', 'Dispensed'],
    hourly.map((v, h) => [`${String(h).padStart(2, '0')}:00`, `Rs ${v.toLocaleString('en-IN')}`]));
}

function tableOf(headers, rows) {
  return `<table><thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join('')}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}

// ---------------------------------------------------------- live task calls

const atmPost = (task, body) => fetch(`/atm/tasks/${task}`, {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ ...body, use_llm: false }),
}).then((r) => r.json()).catch(() => null);

async function loadInsights(m, burn, rows) {
  const fleet = FLEET;
  const payloadMachine = toMachinePayload(m);

  const [runout, replen, cit, carry, demand, uptime, settle, eod] = await Promise.all([
    atmPost('cash_runout_forecast', {
      machine: payloadMachine,
      inp: { current_time: nowHHMM(), dispense_rate_per_hour: Math.round(burn.rate),
             low_cash_threshold: m.low_cash_threshold, cit_time: m.cit_time, cit_hours_from_now: m.cit_hours_from_now },
    }),
    atmPost('replenishment_priority', {
      candidates: fleet.map((x) => ({ atm_id: x.atm_id, location: x.location, cash_on_hand: usableCash(x), dispense_rate_per_hr: Math.round(currentRate(x)) })),
      window_hours: 12,
    }),
    atmPost('cit_route_planning', {
      candidates: rows.map((r) => ({ atm_id: r.m.atm_id, distance_km: 6 + (r.m.atm_id.length % 5) * 3.4,
        cash_left: r.cash, hours_to_empty: +r.h.toFixed(1),
        requested_load: Math.max(500000, Math.round((r.m.dispense_rate_per_hour * 24) / 100000) * 100000) })),
      cap: 15000000,
    }),
    atmPost('cash_carrying_cost', {
      p: { atm: payloadMachine, idle_cash: Math.round(usableCash(m) * 0.55), daily_dispense: machineDaily(m),
           annual_rate: 0.175, cit_trip_cost: 12000, window_days: 90 },
    }),
    atmPost('demand_forecast', {
      inp: { machine: payloadMachine,
             week_dispensed: [['Monday', Math.round(machineDaily(m) * .95)], ['Tuesday', Math.round(machineDaily(m) * .88)],
                              ['Wednesday', Math.round(machineDaily(m) * .92)], ['Thursday', Math.round(machineDaily(m) * 1.02)],
                              ['Friday', Math.round(machineDaily(m) * 1.28)], ['Saturday', Math.round(machineDaily(m) * 1.05)],
                              ['Sunday', Math.round(machineDaily(m) * .74)]],
             forecast_day: 'Friday', seasonal_label: 'pre Eid-ul-Fitr', seasonal_multiplier: 2.1,
             machine_capacity: m.cassettes.reduce((s, c) => s + c.denom * c.capacity, 0) },
    }),
    atmPost('uptime_sla_check', {
      inputs: { atm: payloadMachine, period_days: 31, sla_pct: 98.0,
                incidents: [{ code: 'H0013', description: 'cash-out sensor mismatch', minutes: 852 },
                            { code: 'N0009', description: 'note quality reject rate high', minutes: 721 },
                            { code: 'D0011', description: 'dispenser fault - note pick failure', minutes: 254 }] },
    }),
    atmPost('interbank_settlement', {
      inputs: { bank_code: m.bank, on_us_count: 46407, acquired_offus_count: 6244, issued_offus_count: 13057, interchange_rate: 25.0 },
    }),
    atmPost('eod_position_report', {
      sites: fleet.map((x) => ({ atm_id: x.atm_id, location: x.location, status: x.status,
        cash_on_hand: usableCash(x), dispensed_today: Math.round(machineDaily(x) * 0.62) })),
      low_cash_threshold: 500000,
    }),
  ]);

  const cards = [];
  const push = (task, headline, body) => cards.push({ task, headline, body });

  if (runout?.facts) {
    const f = runout.facts;
    push('cash_runout_forecast',
      f.survives_to_cit ? `Survives to the ${f.cit_time} van` : `Empties before the ${f.cit_time} van`,
      `Low-cash alarm at <b>${f.low_cash_time}</b>, fully empty <b>${f.empty_time}</b>. Headroom against the scheduled visit: <b>${f.headroom_hours}h</b>.`);
  }
  if (replen?.facts) {
    const f = replen.facts;
    push('replenishment_priority', `Refill <b>${esc(f.first_site || '—')}</b> first`,
      `${f.at_risk_count} of ${f.ranked.length} ${f.at_risk_count === 1 ? 'machine empties' : 'machines empty'} inside the 12-hour window. Order is by hours-to-empty, not by how low the cash looks.`);
  }
  if (cit?.facts) {
    const f = cit.facts;
    push('cit_route_planning', `One run covers ${f.sites_included} of ${f.sites_total} sites`,
      `Rs ${compact(f.total_loaded)} of the Rs ${compact(f.cap)} vehicle insurance cap, Rs ${compact(f.spare_capacity)} spare. ${f.deferred.length ? `<b>${f.deferred.length}</b> slip to the next run.` : 'Everything fits in one run.'}`);
  }
  if (carry?.facts) {
    const f = carry.facts;
    push('cash_carrying_cost', `Idle cash costs Rs ${compact(f.carrying_cost)} a quarter`,
      `Trimming the load by Rs ${compact(f.reduced_load_amount)} saves Rs ${compact(f.funding_saved)} in funding but adds Rs ${compact(f.extra_cit_cost)} of CIT trips — net <b>${f.net_benefit >= 0 ? '+' : '−'}Rs ${compact(Math.abs(f.net_benefit))}</b>, so ${f.worth_it ? 'switch to smaller, more frequent loads' : 'keep the current load size'}.`);
  }
  if (demand?.facts) {
    const f = demand.facts;
    push('demand_forecast', `Friday before Eid: Rs ${compact(f.forecast)}`,
      `Against a Rs ${compact(f.weekly_average)} weekly average — a <b>${f.seasonal_multiplier}×</b> holiday multiplier. ${f.within_capacity ? 'One pre-load covers it.' : `Over capacity — needs a Rs ${compact(f.topup_needed)} mid-day top-up.`}`);
  }
  if (uptime?.facts) {
    const f = uptime.facts;
    const bc = f.biggest_contributor;   // an incident object, not a string
    const worst = bc ? `${bc.code} (${bc.description}) at ${bc.minutes.toLocaleString()} min` : '—';
    push('uptime_sla_check', `${f.uptime_pct}% uptime — ${f.breach ? 'SLA breach' : 'within SLA'}`,
      `${f.total_downtime_minutes.toLocaleString()} minutes down against an allowance of ${f.allowed_downtime_minutes.toLocaleString()}. ${f.breach ? `Over by <b>${f.breach_minutes.toLocaleString()}</b> minutes; worst offender ${esc(worst)}.` : ''}`);
  }
  if (settle?.facts) {
    const f = settle.facts;
    push('interbank_settlement', `Interchange: Rs ${compact(Math.abs(f.net_position))} ${f.net_payer ? 'payable' : 'receivable'}`,
      `Rs ${compact(f.acquiring_income)} earned on other banks' cards against Rs ${compact(f.issuing_cost)} paid away. ${f.net_payer ? 'Our cardholders use other banks\' machines more than the reverse — add ATMs where they actually transact, or push them on-us.' : 'The fleet earns more than it pays.'}`);
  }
  if (eod?.facts) {
    const f = eod.facts;
    const n = f.action_items.length;
    push('eod_position_report', `Fleet availability ${f.availability_pct}%`,
      `${f.online_count} of ${f.total_count} online. Rs ${compact(f.cash_on_hand_total)} on hand, Rs ${compact(f.dispensed_total)} dispensed today. ` +
      (n ? `<b>${n}</b> ${n === 1 ? 'site needs' : 'sites need'} attention in the morning.` : 'Nothing needs attention in the morning.'));
  }

  document.getElementById('insight-grid').innerHTML = cards.map((c) => `
    <article class="insight">
      <div class="insight-task">${esc(c.task)}</div>
      <div class="insight-headline">${c.headline}</div>
      <div class="insight-body">${c.body}</div>
    </article>`).join('');
  document.getElementById('an-task-count').textContent = `${cards.length}`;
}

// ---------------------------------------------------------- orchestration

function selectMachine(id) {
  anSelected = id;
  document.querySelectorAll('#an-filter button').forEach((b) => b.classList.toggle('on', b.dataset.id === id));
  renderAnalytics();
}

async function renderAnalytics() {
  const m = FLEET.find((x) => x.atm_id === anSelected) || FLEET[0];
  const burn = drawBurn(m);
  const rows = drawRunway(FLEET);
  drawHourly(m, burn.hourly);

  const st = burn.status;
  document.getElementById('hf-value').innerHTML =
    `<span style="color:${STATUS_INK[st]}">${fmtHrs(burn.hoursLeft)}</span>`;
  document.getElementById('hf-sub').textContent =
    `${m.atm_id} · ${m.location} · empties ${String(Math.floor(burn.emptyAt) % 24).padStart(2, '0')}:${String(Math.round((burn.emptyAt % 1) * 60)).padStart(2, '0')}`;

  const worst = rows[0];
  const totalCash = FLEET.reduce((s, x) => s + usableCash(x), 0);
  const atRisk = rows.filter((r) => r.st === 'critical').length;
  document.getElementById('an-stats').innerHTML = [
    { n: `Rs ${compact(totalCash)}`, l: 'Cash in the fleet', tone: '' },
    { n: `Rs ${compact(Math.round(burn.rate))}`, l: 'Burning per hour, here', tone: '' },
    { n: atRisk, l: 'Empty before their van', tone: atRisk ? 'bad' : 'good' },
    { n: worst.m.atm_id, l: 'Most urgent site', tone: 'warn', small: true },
  ].map((s) => `<div class="stat-tile"><div class="n ${s.tone}" ${s.small ? 'style="font-size:15px;line-height:1.7"' : ''}>${esc(String(s.n))}</div><div class="lbl">${esc(s.l)}</div></div>`).join('');

  await loadInsights(m, burn, rows);
}

function initAnalytics() {
  if (!document.getElementById('view-analytics')) return;
  anSelected = FLEET[0].atm_id;
  document.getElementById('an-filter').innerHTML = FLEET.map((m) =>
    `<button data-id="${esc(m.atm_id)}" class="${m.atm_id === anSelected ? 'on' : ''}">${esc(m.atm_id)}</button>`).join('');
  document.querySelectorAll('#an-filter button').forEach((b) =>
    b.addEventListener('click', () => selectMachine(b.dataset.id)));
  renderAnalytics();
}

document.addEventListener('DOMContentLoaded', () => {
  // FLEET lives in dashboard.js; wait a tick so load order can't bite
  setTimeout(initAnalytics, 0);
});
