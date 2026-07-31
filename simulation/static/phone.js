// ============================================================
// Customer app — the phone side of the same engine.
//
// The replies come from phone_data.js: verbatim, precomputed output of the
// fine-tuned model (see the header of that file). They are hardcoded so the
// phone answers instantly; the panel beside it shows which of the 51 task
// types fired and the deterministic ledger the model was handed, so the
// "computed → narrated" split stays visible here too.
// ============================================================

const CHIP_ICONS = {
  alert: '<svg viewBox="0 0 16 16" fill="none"><path d="M8 2.5 1.8 13h12.4L8 2.5Z" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><path d="M8 6.6v3M8 11.4v.01" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>',
  card:  '<svg viewBox="0 0 16 16" fill="none"><rect x="1.6" y="3.4" width="12.8" height="9.2" rx="1.6" stroke="currentColor" stroke-width="1.4"/><path d="M1.6 6.6h12.8" stroke="currentColor" stroke-width="1.4"/></svg>',
  cash:  '<svg viewBox="0 0 16 16" fill="none"><rect x="1.6" y="4" width="12.8" height="8" rx="1.4" stroke="currentColor" stroke-width="1.4"/><circle cx="8" cy="8" r="1.9" stroke="currentColor" stroke-width="1.4"/></svg>',
  loan:  '<svg viewBox="0 0 16 16" fill="none"><path d="M8 1.8v12.4M11.2 4.4H6.4a1.9 1.9 0 0 0 0 3.8h3.2a1.9 1.9 0 0 1 0 3.8H4.4" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>',
};

const phoneUsed = new Set();
let phoneBusy = false;

function phoneReset() {
  phoneUsed.clear();
  phoneBusy = false;
  const thread = document.getElementById('chat-thread');
  thread.innerHTML = '';
  addBubble('sys', 'Today');
  addBubble('them', 'Assalam o Alaikum Zainab 👋 This is AsaanBank support. Tap anything below and I\'ll take it from there.');
  document.getElementById('phone-behind').innerHTML =
    '<div class="behind-empty">Tap a message on the phone to see which of the 51 task types handles it — and the exact figures the model was given before it wrote a word.</div>';
  renderChips();
}

function addBubble(kind, text) {
  const thread = document.getElementById('chat-thread');
  const el = document.createElement('div');
  el.className = `bubble ${kind}`;
  el.textContent = text;
  thread.appendChild(el);
  thread.scrollTop = thread.scrollHeight;
  return el;
}

function renderChips() {
  document.getElementById('chat-chips').innerHTML = PHONE_SCRIPT.map((s) => `
    <button class="chip" data-id="${esc(s.id)}" ${phoneUsed.has(s.id) ? 'disabled' : ''}>
      ${CHIP_ICONS[s.icon] || ''}${esc(s.chip)}
    </button>`).join('');
  document.querySelectorAll('#chat-chips .chip').forEach((b) => {
    b.addEventListener('click', () => playScript(b.dataset.id));
  });
}

async function playScript(id) {
  if (phoneBusy) return;
  const s = PHONE_SCRIPT.find((x) => x.id === id);
  if (!s) return;
  phoneBusy = true;
  phoneUsed.add(id);
  renderChips();

  addBubble('me', s.user);

  // typing indicator — cosmetic only; the reply is already on disk
  const thread = document.getElementById('chat-thread');
  const typing = document.createElement('div');
  typing.className = 'bubble them bubble-typing';
  typing.innerHTML = '<i></i><i></i><i></i>';
  thread.appendChild(typing);
  thread.scrollTop = thread.scrollHeight;

  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  await new Promise((r) => setTimeout(r, reduced ? 0 : 900));
  typing.remove();

  addBubble('them', s.reply);
  showBehind(s);
  phoneBusy = false;
  renderChips();
}

function showBehind(s) {
  const host = document.getElementById('phone-behind');
  if (host.querySelector('.behind-empty')) host.innerHTML = '';
  const block = document.createElement('div');
  block.className = 'behind-step';
  block.innerHTML = `
    <div class="behind-head">
      <span class="behind-task">${esc(s.task)}</span>
      <span class="behind-domain">${esc(s.domain === 'atm' ? 'ATM · customer facing' : 'Loans · customer facing')}</span>
    </div>
    <div class="behind-body">${duoBlock('What the engine computed', s.ledger, s.reply, 'llm · precomputed')}</div>`;
  host.prepend(block);
}

document.addEventListener('DOMContentLoaded', () => {
  if (!document.getElementById('view-phone')) return;
  phoneReset();
  document.getElementById('phone-reset').addEventListener('click', phoneReset);
  document.querySelectorAll('[data-goto-tab]').forEach((b) => {
    b.addEventListener('click', () => document.querySelector(`.tab[data-view="${b.dataset.gotoTab}"]`)?.click());
  });
});
