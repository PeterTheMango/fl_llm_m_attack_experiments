/* CANARY Monitor — single-page dashboard.
 *
 * Views: Live / Results / Detail / Access / Launch. All data comes from the
 * FastAPI JSON API; nothing here invents a number. Where the backend cannot
 * know something (no run-state report, no GPU, no per-round loss), the view
 * says so rather than filling the gap.
 */
'use strict';

// ---------- themes ----------
const THEMES = {
  observatory: '--bg:#0d1014;--pn:#15191f;--p2:#1b212a;--bd:#252c36;--fg:#e6ebf0;--fd:#9aa6b2;--fm:#5f6b78;--gd:#222a34;--hv:#1f2630;--ac:#36c08f;--ok:#3fcf8e;--no:#f0606a;--wn:#e3b341;--rn:#4aa8ff;--sh:0 1px 2px rgba(0,0,0,.45),0 10px 30px rgba(0,0,0,.28)',
  lab: '--bg:#f4f5f3;--pn:#ffffff;--p2:#fafbf9;--bd:#e3e5e0;--fg:#1a1d1a;--fd:#5b615b;--fm:#969b95;--gd:#ebede9;--hv:#f0f1ee;--ac:#4f5bd5;--ok:#2e9e63;--no:#d24a4f;--wn:#bf8a18;--rn:#2f6fe0;--sh:0 1px 2px rgba(20,24,20,.06),0 8px 24px rgba(20,24,20,.06)',
  signal: '--bg:#efe9e1;--pn:#fbf8f4;--p2:#f4efe7;--bd:#e2d9cc;--fg:#241f18;--fd:#6b6155;--fm:#a89c8c;--gd:#ece4d8;--hv:#f1ebe2;--ac:#c2691d;--ok:#4f8a4a;--no:#c2451f;--wn:#b67d12;--rn:#3b72a8;--sh:0 1px 2px rgba(60,40,20,.07),0 8px 24px rgba(60,40,20,.07)',
};
const THEME_SWATCH = [['observatory', '#0d1014', '#36c08f'], ['lab', '#ffffff', '#4f5bd5'], ['signal', '#efe9e1', '#c2691d']];

const STAGES = ['fine-tune', 'attack', 'measure'];
const TIMELINE_WINDOW_S = 360;
const POLL_MS = 2500;
const TICK_MS = 1000;
// Only rounding noise counts as "at bottom". Once the user scrolls upward,
// even by a few pixels, live refreshes must stop following new log entries.
const LOG_BOTTOM_EPSILON_PX = 2;
const MONO = "'IBM Plex Mono',monospace";

// ---------- state ----------
const S = {
  view: 'live',
  theme: localStorage.getItem('canary.theme') || 'observatory',
  liveViz: 'cards',
  privacyView: false,
  xFactor: 'epsilon',
  sort: { key: 'adv', dir: 'desc' },
  filters: { attacks: null, status: 'all', model: 'all', mech: 'all', q: '' },
  selectedRunId: null,
  live: null, results: null, detail: null, detailError: '', tunnel: null, launch: null,
  // Settings edit credentials, so the server serves them to loopback callers
  // only. /api/session decides; this flag only keeps the nav honest.
  local: false,
  // null on a field means "whatever is saved in the .env" — the browser is
  // never sent a saved token, so an untouched field has nothing to send back.
  tunnelForm: { provider: null, apiKey: null, code: null, port: null },
  launchForm: { mode: 'manual', configFile: null, attacks: [], useFirestore: true },
  // Manual mode. A card is { name, values, sweeps, advanced }; `values` holds
  // only what the user typed, so an untouched field never reaches the payload
  // and a saved config stays as small as hand-written YAML. `dirty` greys the
  // last validation rather than clearing it — a stale run count must never
  // read as current.
  manual: { cards: [], validation: null, dirty: false, saveName: '', confirmOverwrite: false },
  attackFields: null,
  // Config editor: `name` is the file loaded, `saveName` is where Save writes.
  editor: { name: null, text: '', dirty: false, validation: null, saveName: '', confirmOverwrite: false },
  settings: null,
  settingsEdits: {}, settingsDeletes: [], settingsRevealed: {}, newVar: { key: '', value: '' },
  stopConfirm: false, deleteConfirmRunId: null,
  banner: '', copied: false,
  serverOffset: 0,
};

const charts = {};
const chartSig = {};
let root = null;

// ---------- helpers ----------
const esc = (v) => String(v === null || v === undefined ? '' : v)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const now = () => Date.now() / 1000 + S.serverOffset;
const fmt3 = (x) => (x === null || x === undefined || Number.isNaN(x)) ? '—' : Number(x).toFixed(3);
const shortModel = (m) => String(m || '').split('/').pop();
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

function cvar(name, fallback) {
  if (!root) return fallback;
  const v = getComputedStyle(root).getPropertyValue(name).trim();
  return v || fallback;
}
function advColor(a) {
  if (a === null || a === undefined) return cvar('--fm', '#5f6b78');
  if (a >= 0.72) return cvar('--no', '#f0606a');
  if (a >= 0.6) return cvar('--wn', '#e3b341');
  return cvar('--ok', '#3fcf8e');
}
function timeAgo(unix) {
  if (!unix) return '—';
  const d = Math.max(0, now() - unix);
  if (d < 60) return Math.round(d) + 's';
  if (d < 3600) return Math.round(d / 60) + 'm';
  if (d < 86400) return Math.round(d / 3600) + 'h';
  return Math.round(d / 86400) + 'd';
}
function elapsedFmt(s) {
  if (s === null || s === undefined) return '—';
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  const pad = (n) => String(n).padStart(2, '0');
  return h ? `${h}:${pad(m)}:${pad(s % 60)}` : `${pad(m)}:${pad(s % 60)}`;
}
function clockText() {
  const t = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(t.getHours())}:${pad(t.getMinutes())}:${pad(t.getSeconds())}`;
}
function meta(attack) {
  const attacks = (S.live && S.live.attacks) || (S.results && S.results.attacks) || {};
  return attacks[attack] || { label: String(attack || '?').toUpperCase(), color: '#5f6b78', title: '', methodology: {} };
}
function mechEps(cfg) {
  const mech = cfg.ldp_mechanism, eps = cfg.epsilon;
  if (mech === undefined && eps === undefined) return '—';
  if (!mech || mech === 'none' || eps === null || eps === undefined) return 'none';
  return `${mech} ε${eps}`;
}
function panel(title, note, inner, extraHeader) {
  return `<div style="background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:12px;box-shadow:var(--sh,0 1px 2px rgba(0,0,0,.4));overflow:hidden">
    <div style="display:flex;align-items:center;gap:9px;padding:13px 16px;border-bottom:1px solid var(--bd,#252c36)">
      <span style="font-size:10px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--fd,#9aa6b2)">${esc(title)}</span>
      ${note ? `<span style="font-size:10px;color:var(--fm,#5f6b78)">${esc(note)}</span>` : ''}
      ${extraHeader || ''}
    </div>${inner}</div>`;
}
function emptyNote(title, body) {
  return `<div style="padding:34px 20px;text-align:center;color:var(--fm,#5f6b78);font-size:12.5px;line-height:1.6">
    ${title ? `<div style="font-family:${MONO};font-size:11px;color:var(--wn,#e3b341);margin-bottom:8px">${esc(title)}</div>` : ''}
    ${body}</div>`;
}
const FIELD = `width:100%;font-size:12px;font-family:${MONO};padding:10px 13px;border-radius:8px;border:1px solid var(--bd,#252c36);background:var(--bg,#0d1014);color:var(--fg,#e6ebf0);outline:none;-webkit-appearance:none;appearance:none;box-shadow:inset 0 1px 3px rgba(0,0,0,.3);transition:border-color .15s,box-shadow .15s`;
const SEL = `font-size:11px;padding:5px 9px;border-radius:7px;border:1px solid var(--bd,#252c36);background:var(--p2,#1b212a);color:var(--fd,#9aa6b2);cursor:pointer`;
const segStyle = (active) => `padding:6px 14px;border-radius:7px;font-size:12px;font-weight:600;cursor:pointer;transition:.12s;${active ? 'background:var(--ac,#36c08f);color:#0d1014' : 'color:var(--fd,#9aa6b2)'}`;

// ---------- data ----------
async function getJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return res.json();
}
async function postJSON(url, body) {
  return getJSON(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
}
async function deleteJSON(url) {
  return getJSON(url, { method: 'DELETE' });
}

async function refresh() {
  try {
    if (S.view === 'live') {
      S.live = await getJSON('/api/live');
      if (!S.live.worker || !S.live.worker.running) S.stopConfirm = false;
      S.serverOffset = S.live.now_unix - Date.now() / 1000;
      S.tunnel = await getJSON('/api/tunnel');
    } else if (S.view === 'results') {
      S.results = await getJSON('/api/results');
      initFilters();
    } else if (S.view === 'detail' && S.selectedRunId) {
      try {
        S.detail = await getJSON('/api/runs/' + encodeURIComponent(S.selectedRunId));
        S.detailError = '';
      } catch (err) { S.detail = null; S.detailError = err.message; }
    } else if (S.view === 'access') {
      S.tunnel = await getJSON('/api/tunnel');
    } else if (S.view === 'launch') {
      S.launch = await getJSON('/api/launch');
      if (!S.launch.worker.running) S.stopConfirm = false;
      // The schema is fixed for the process's lifetime; fetch it once rather
      // than on every poll.
      if (!S.attackFields) S.attackFields = await getJSON('/api/attacks/fields');
      if (!S.launchForm.configFile && S.launch.configs.length) S.launchForm.configFile = S.launch.configs[0];
      if (S.editor.name === null && !S.editor.dirty && S.launchForm.configFile) await loadConfig(S.launchForm.configFile);
    } else if (S.view === 'settings' && S.local) {
      S.settings = await getJSON('/api/settings');
    }
    S.banner = '';
  } catch (err) {
    S.banner = 'API unreachable: ' + err.message;
  }
  render({ background: true });
}

function initFilters() {
  if (S.filters.attacks) return;
  S.filters.attacks = {};
  Object.keys((S.results && S.results.attacks) || {}).forEach((k) => { S.filters.attacks[k] = true; });
}

function stopSweepControls(worker, compact) {
  if (!S.local || !worker || !worker.running) return '';
  const pad = compact ? '5px 10px' : '8px 14px';
  if (!S.stopConfirm) {
    return `<div data-act="sweep-stop-arm" style="font-size:11px;font-weight:600;cursor:pointer;padding:${pad};border-radius:8px;color:var(--no,#f0606a);border:1px solid color-mix(in srgb,var(--no,#f0606a) 48%,var(--bd));background:color-mix(in srgb,var(--no,#f0606a) 7%,var(--p2,#1b212a))">Stop sweep</div>`;
  }
  return `<div style="display:flex;align-items:center;gap:6px">
    <div data-act="sweep-stop-confirm" style="font-size:11px;font-weight:700;cursor:pointer;padding:${pad};border-radius:8px;color:#fff;border:1px solid var(--no,#f0606a);background:var(--no,#f0606a)">Confirm stop</div>
    <div data-act="sweep-stop-cancel" style="font-size:10.5px;cursor:pointer;padding:${pad};border-radius:8px;color:var(--fd,#9aa6b2);border:1px solid var(--bd,#252c36);background:var(--p2,#1b212a)">Cancel</div>
  </div>`;
}

// ---------- routing ----------
function pathFor() {
  if (S.view === 'detail' && S.selectedRunId) return '/results/' + encodeURIComponent(S.selectedRunId);
  return S.view === 'live' ? '/' : '/' + S.view;
}
function readLocation() {
  const p = location.pathname;
  const m = p.match(/^\/results\/(.+)$/);
  if (m) { S.view = 'detail'; S.selectedRunId = decodeURIComponent(m[1]); return; }
  if (p === '/results') S.view = 'results';
  else if (p === '/access') S.view = 'access';
  else if (p === '/launch') S.view = 'launch';
  // A deep link to /settings from a remote browser never reaches here (the
  // server answers 403), but a client-side nav from a stale tab could.
  else if (p === '/settings') S.view = S.local ? 'settings' : 'live';
  else S.view = 'live';
}
function go(view, runId) {
  if (view === 'settings' && !S.local) return;
  S.view = view;
  if (runId !== undefined) S.selectedRunId = runId;
  if (view === 'detail') { S.detail = null; S.detailError = ''; }
  history.pushState({}, '', pathFor());
  render();
  refresh();
}

// ---------- top bar ----------
function topBar() {
  const runningCount = S.live ? (S.live.running_live ?? S.live.running.length) : 0;
  const staleCount = S.live ? (S.live.running_stale || 0) : 0;
  const gpuCount = S.live ? S.live.gpus.length : 0;
  const tl = !!(S.tunnel && S.tunnel.connected);
  const nav = [['live', 'Live'], ['results', 'Results'], ['access', 'Access'], ['launch', 'Launch'], ['settings', 'Settings']]
    .filter(([k]) => k !== 'settings' || S.local)
    .map(([k, label]) => `<div data-act="nav" data-arg="${k}" style="${segStyle(S.view === k || (k === 'results' && S.view === 'detail'))}">${label}</div>`)
    .join('');
  const themes = THEME_SWATCH.map(([k, bg, ac]) =>
    `<div data-act="theme" data-arg="${k}" title="${k}" style="width:17px;height:17px;border-radius:5px;cursor:pointer;background:${bg};border:1px solid ${S.theme === k ? ac : 'var(--bd,#252c36)'};box-shadow:${S.theme === k ? '0 0 0 2px ' + ac : 'none'}"></div>`).join('');

  return `<div style="display:flex;align-items:center;gap:18px;padding:0 18px;height:54px;flex:0 0 auto;border-bottom:1px solid var(--bd,#252c36);background:var(--pn,#15191f)">
    <div style="display:flex;align-items:center;gap:11px">
      <div style="width:16px;height:16px;background:var(--ac,#36c08f);transform:rotate(45deg);border-radius:3px"></div>
      <div style="line-height:1.05">
        <div style="font-size:14px;font-weight:700;letter-spacing:.14em">CANARY</div>
        <div style="font-size:9.5px;font-weight:500;letter-spacing:.18em;color:var(--fm,#5f6b78);text-transform:uppercase">fed-MIA monitor</div>
      </div>
    </div>
    <div style="display:flex;gap:3px;background:var(--p2,#1b212a);border:1px solid var(--bd,#252c36);border-radius:9px;padding:3px">${nav}</div>
    <div style="flex:1"></div>
    <div style="display:flex;align-items:center;gap:7px;font-family:${MONO};font-size:11px;color:var(--fd,#9aa6b2);padding:5px 10px;border:1px solid var(--bd,#252c36);border-radius:8px;background:var(--p2,#1b212a)">
      <span style="width:7px;height:7px;border-radius:50%;background:var(--rn,#4aa8ff);${runningCount ? 'animation:pulse 1.6s infinite' : 'opacity:.35'}"></span>
      ${runningCount} running · ${gpuCount} GPU${staleCount ? `<span style="color:var(--wn,#e3b341)"> · ${staleCount} stale</span>` : ''}
    </div>
    <div data-act="nav" data-arg="access" data-hover style="display:flex;align-items:center;gap:7px;font-size:11px;cursor:pointer;padding:5px 11px;border-radius:8px;border:1px solid ${tl ? 'color-mix(in srgb,var(--ok,#3fcf8e) 45%,var(--bd))' : 'var(--bd,#252c36)'};background:var(--p2,#1b212a);color:${tl ? 'var(--ok,#3fcf8e)' : 'var(--fd,#9aa6b2)'};font-family:${MONO}">
      <span style="width:7px;height:7px;border-radius:50%;background:${tl ? 'var(--ok,#3fcf8e)' : 'var(--fm,#5f6b78)'}"></span>${tl ? 'tunnel live' : 'tunnel off'}
    </div>
    <div style="font-family:${MONO};font-size:12px;color:var(--fd,#9aa6b2);letter-spacing:.04em;min-width:74px;text-align:right">${clockText()}</div>
    <div style="display:flex;gap:5px;align-items:center;padding-left:4px;border-left:1px solid var(--bd,#252c36)">${themes}</div>
  </div>`;
}

function bannerBar() {
  if (!S.banner) return '';
  return `<div style="padding:9px 18px;background:color-mix(in srgb,var(--no,#f0606a) 12%,var(--pn,#15191f));border-bottom:1px solid color-mix(in srgb,var(--no,#f0606a) 30%,var(--bd));font-size:11.5px;font-family:${MONO};color:var(--no,#f0606a)">${esc(S.banner)}</div>`;
}

// ---------- live view ----------
function stageBars(stages, index) {
  return stages.map((label, i) => {
    const known = index >= 0;
    const done = known && i < index, cur = known && i === index;
    const bar = done ? 'var(--ok,#3fcf8e)' : (cur ? 'var(--rn,#4aa8ff)' : 'var(--gd,#222a34)');
    const anim = cur ? 'animation:pulse 1.4s infinite' : '';
    const txt = done ? 'var(--ok,#3fcf8e)' : (cur ? 'var(--rn,#4aa8ff)' : 'var(--fm,#5f6b78)');
    return `<div style="flex:1;display:flex;flex-direction:column;gap:4px">
      <div style="height:4px;border-radius:3px;background:${bar};${anim}"></div>
      <span style="font-size:9.5px;letter-spacing:.04em;color:${txt}">${esc(label)}</span>
    </div>`;
  }).join('');
}

function staleNotice(r) {
  // The run-state report stopped being re-stamped. Say exactly that, and what
  // it implies, rather than showing a run that looks alive.
  const age = r.heartbeat_age_seconds;
  const ago = age === null || age === undefined ? 'an unknown time' : elapsedFmt(age);
  return `<div style="display:flex;align-items:flex-start;gap:9px;margin:-2px 0 11px;padding:9px 11px;border-radius:8px;background:color-mix(in srgb,var(--wn,#e3b341) 9%,transparent);border:1px solid color-mix(in srgb,var(--wn,#e3b341) 28%,var(--bd))">
    <span style="width:7px;height:7px;border-radius:50%;background:var(--wn,#e3b341);flex:0 0 auto;margin-top:4px"></span>
    <div style="font-size:11px;color:var(--fd,#9aa6b2);line-height:1.55">
      <b style="color:var(--wn,#e3b341)">Stale — no heartbeat for ${esc(ago)}.</b>
      The process that reported this run stopped updating it, so it is almost certainly gone.
      Nothing was written for this run_id, so re-running the sweep recomputes it from scratch.
    </div>
  </div>`;
}

function runningCards() {
  return S.live.running.map((r) => {
    const m = meta(r.attack);
    const chips = r.chips.map((c) => `<span style="font-family:${MONO};font-size:10.5px;color:var(--fd,#9aa6b2);background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);padding:2px 7px;border-radius:5px">${esc(c)}</span>`).join('');
    return `<div class="card-hover" style="border:1px solid ${r.stale ? 'color-mix(in srgb,var(--wn,#e3b341) 45%,var(--bd))' : 'var(--bd,#252c36)'};border-radius:10px;background:var(--p2,#1b212a);padding:14px;${r.stale ? 'opacity:.72' : ''}">
      ${r.stale ? staleNotice(r) : ''}
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:11px">
        <span style="font-size:10px;font-weight:600;font-family:${MONO};padding:3px 8px;border-radius:6px;color:#0d1014;background:${m.color}">${esc(m.label)}</span>
        <span style="font-family:${MONO};font-size:12px;color:var(--fd,#9aa6b2)">${esc(r.run_id)}</span>
        <div style="flex:1"></div>
        <span style="font-size:10.5px;color:var(--fm,#5f6b78)">GPU ${r.gpu === null || r.gpu === undefined ? '—' : esc(r.gpu)}</span>
        <span style="font-family:${MONO};font-size:12px;color:${r.stale ? 'var(--fm,#5f6b78)' : 'var(--fg,#e6ebf0)'}">${elapsedFmt(r.elapsed_seconds)}</span>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:12px">${chips || `<span style="font-size:10.5px;color:var(--fm,#5f6b78)">config not reported</span>`}</div>
      <div style="display:flex;align-items:center;gap:8px">${stageBars(r.stages, r.stale ? -1 : r.stage_index)}</div>
      ${r.stage_index < 0 ? `<div style="margin-top:8px;font-size:10px;color:var(--fm,#5f6b78);font-family:${MONO}">stage not reported by the run-state report</div>` : ''}
    </div>`;
  }).join('');
}

function runningTimeline() {
  const t0 = now() - TIMELINE_WINDOW_S;
  const rows = [];
  S.live.running.forEach((r) => {
    const m = meta(r.attack);
    const start = r.elapsed_seconds === null || r.elapsed_seconds === undefined ? t0 : now() - r.elapsed_seconds;
    const left = clamp(((start - t0) / TIMELINE_WINDOW_S) * 100, 0, 100);
    rows.push({
      label: m.label, color: m.color, runId: r.run_id, left, width: Math.max(2, 100 - left),
      fill: 'var(--rn,#4aa8ff)',
      anim: 'background-image:linear-gradient(45deg,rgba(255,255,255,.22) 25%,transparent 25%,transparent 50%,rgba(255,255,255,.22) 50%,rgba(255,255,255,.22) 75%,transparent 75%);background-size:12px 12px;animation:flow 1s linear infinite',
      metric: 'running', metricColor: 'var(--rn,#4aa8ff)',
    });
  });
  S.live.recent.forEach((r) => {
    if (!r.updated_at_unix || r.updated_at_unix < t0) return;
    const m = meta(r.attack);
    const left = clamp(((r.updated_at_unix - t0) / TIMELINE_WINDOW_S) * 100, 0, 97);
    const failed = r.status === 'failed';
    rows.push({
      label: m.label, color: m.color, runId: r.run_id, left, width: 3,
      fill: failed ? 'var(--no,#f0606a)' : m.color, anim: '',
      metric: failed ? 'failed' : fmt3(r.adv), metricColor: failed ? 'var(--no,#f0606a)' : advColor(r.adv),
    });
  });
  if (!rows.length) return emptyNote('', 'Nothing started or finished in the last 6 minutes.');

  const body = rows.map((t) => `<div style="display:flex;align-items:center;gap:10px;margin-bottom:9px">
      <span style="width:64px;font-size:10px;font-weight:600;font-family:${MONO};padding:2px 6px;border-radius:5px;color:#0d1014;background:${t.color};text-align:center">${esc(t.label)}</span>
      <span style="width:118px;font-family:${MONO};font-size:11px;color:var(--fd,#9aa6b2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(t.runId)}</span>
      <div style="flex:1;height:18px;border-radius:5px;background:var(--p2,#1b212a);position:relative;overflow:hidden">
        <div style="position:absolute;left:${t.left}%;width:${t.width}%;top:0;bottom:0;border-radius:5px;background:${t.fill};${t.anim}"></div>
      </div>
      <span style="width:62px;text-align:right;font-family:${MONO};font-size:11px;color:${t.metricColor}">${esc(t.metric)}</span>
    </div>`).join('');
  return `<div style="padding:4px 2px">${body}
    <div style="display:flex;justify-content:space-between;font-size:9.5px;color:var(--fm,#5f6b78);font-family:${MONO};margin-left:202px;margin-top:2px"><span>−6 min</span><span>now</span></div>
  </div>`;
}

function runningGauges() {
  const gpus = S.live.gpus;
  return `<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">` + S.live.running.map((r) => {
    const m = meta(r.attack);
    const gpu = gpus.find((g) => g.id === r.gpu) || (gpus.length === 1 ? gpus[0] : null);
    // util is null when the host does not expose it (MIG, vGPU). An empty ring
    // is the honest render: 0% and "not reported" are different claims.
    const util = gpu && gpu.util !== null ? Math.round(gpu.util) : null;
    const dash = `${((util || 0) / 100) * 314} 314`;
    return `<div class="card-hover" style="border:1px solid var(--bd,#252c36);border-radius:10px;background:var(--p2,#1b212a);padding:16px;display:flex;flex-direction:column;align-items:center;gap:10px">
      <div style="display:flex;align-items:center;gap:8px;align-self:stretch">
        <span style="font-size:10px;font-weight:600;font-family:${MONO};padding:2px 7px;border-radius:5px;color:#0d1014;background:${m.color}">${esc(m.label)}</span>
        <span style="font-size:10px;color:var(--fm,#5f6b78)">${gpu ? 'GPU ' + gpu.id : 'GPU not reported'}</span>
      </div>
      <div style="position:relative;width:118px;height:118px">
        <svg width="118" height="118" viewBox="0 0 118 118" style="transform:rotate(-90deg)">
          <circle cx="59" cy="59" r="50" fill="none" stroke="var(--gd,#222a34)" stroke-width="11"></circle>
          <circle cx="59" cy="59" r="50" fill="none" stroke="${m.color}" stroke-width="11" stroke-linecap="round" stroke-dasharray="${dash}"></circle>
        </svg>
        <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center">
          <div style="font-family:${MONO};font-size:23px;font-weight:600">${util === null ? '—' : util + '%'}</div>
          <div style="font-size:9px;color:var(--fm,#5f6b78);letter-spacing:.1em;text-transform:uppercase">GPU util</div>
        </div>
      </div>
      <div style="font-family:${MONO};font-size:11px;color:var(--fd,#9aa6b2)">${esc(r.stage_label)} · ${elapsedFmt(r.elapsed_seconds)}</div>
    </div>`;
  }).join('') + `</div>`;
}

function nowPanel() {
  const gpuCount = S.live.gpus.length;
  const note = gpuCount ? `· ${gpuCount} GPU${gpuCount > 1 ? 's' : ''} on this VM` : '· no GPU detected';
  let inner;
  if (!S.live.run_state_available) {
    inner = emptyNote('run-state report unavailable',
      'The central script is not publishing run-state.<br>Currently-running set can\'t be identified from Firestore alone — showing completed / failed results only.');
  } else if (!S.live.running.length) {
    inner = emptyNote('', 'No runs currently in progress.');
  } else {
    const body = S.liveViz === 'timeline' ? runningTimeline()
      : S.liveViz === 'gauges' ? runningGauges() : runningCards();
    inner = `<div style="padding:14px;display:flex;flex-direction:column;gap:12px">${body}</div>`;
  }
  return panel('Running now', note, inner);
}

function sweepsPanel() {
  const rows = S.live.sweeps;
  if (!rows.length) return panel('Active sweeps', '· grouped by attack', emptyNote('', 'No runs recorded yet.'));
  const body = rows.map((sw) => {
    const m = meta(sw.attack);
    const hasTotal = sw.total !== null && sw.total !== undefined;
    const denom = Math.max(1, hasTotal ? sw.total : sw.complete + sw.failed + sw.running);
    const pct = (n) => (n / denom) * 100;
    return `<div style="padding:13px 0;border-bottom:1px solid var(--gd,#222a34)">
      <div style="display:flex;align-items:center;gap:9px;margin-bottom:9px">
        <span style="font-size:10px;font-weight:600;font-family:${MONO};padding:2px 7px;border-radius:5px;color:#0d1014;background:${m.color}">${esc(m.label)}</span>
        <span style="font-size:12px;color:var(--fd,#9aa6b2);font-family:${MONO}">${esc(sw.attack)}</span>
        <div style="flex:1"></div>
        <span style="font-size:12px;font-family:${MONO};color:var(--fg,#e6ebf0)">${hasTotal ? `${sw.complete}/${sw.total} done` : `${sw.complete}/${sw.complete + sw.failed} resolved`}</span>
      </div>
      <div style="display:flex;height:9px;border-radius:5px;overflow:hidden;background:var(--p2,#1b212a)">
        <div style="width:${pct(sw.complete)}%;background:var(--ok,#3fcf8e)"></div>
        <div style="width:${pct(sw.running)}%;background:var(--rn,#4aa8ff);background-image:linear-gradient(45deg,rgba(255,255,255,.25) 25%,transparent 25%,transparent 50%,rgba(255,255,255,.25) 50%,rgba(255,255,255,.25) 75%,transparent 75%);background-size:14px 14px;animation:flow 1s linear infinite"></div>
        <div style="width:${pct(sw.failed)}%;background:var(--no,#f0606a)"></div>
      </div>
      <div style="display:flex;gap:14px;margin-top:7px;font-size:10.5px;color:var(--fm,#5f6b78);font-family:${MONO}">
        <span style="color:var(--ok,#3fcf8e)">● ${sw.complete} done</span>
        <span style="color:var(--rn,#4aa8ff)">● ${sw.running} running</span>
        <span style="color:var(--no,#f0606a)">● ${sw.failed} failed</span>
        <span>${hasTotal ? `${sw.pending} pending` : '(no manifest → no denominator)'}</span>
      </div>
    </div>`;
  }).join('');
  return panel('Active sweeps', '· grouped by attack', `<div style="padding:6px 16px 14px">${body}</div>`);
}

function recentPanel() {
  const source = (S.live.source && S.live.source.listener) ? 'firestore ⟳ live' : 'firestore ⟳ polling';
  const header = `<div style="flex:1"></div><span style="font-size:9px;color:var(--fm,#5f6b78);font-family:${MONO}">${source}</span>`;
  if (!S.live.recent.length) {
    return panel('Recently finished', '', emptyNote('', 'No finished runs yet.'), header);
  }
  const body = S.live.recent.map((r) => {
    const m = meta(r.attack);
    const failed = r.status === 'failed';
    return `<div class="row-hover" data-act="open" data-arg="${esc(r.run_id)}" style="display:flex;align-items:center;gap:10px;padding:11px 16px;border-bottom:1px solid var(--gd,#222a34);cursor:pointer">
      <span style="width:6px;height:6px;border-radius:50%;background:${failed ? 'var(--no,#f0606a)' : 'var(--ok,#3fcf8e)'};flex:0 0 auto"></span>
      <span style="font-size:9.5px;font-weight:600;font-family:${MONO};padding:1px 6px;border-radius:4px;color:#0d1014;background:${m.color};flex:0 0 auto">${esc(m.label)}</span>
      <span style="font-family:${MONO};font-size:11px;color:var(--fd,#9aa6b2);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(r.run_id)}</span>
      <span style="font-family:${MONO};font-size:12px;color:${failed ? 'var(--no,#f0606a)' : advColor(r.adv)};flex:0 0 auto">${failed ? 'FAILED' : 'Adv ' + fmt3(r.adv)}</span>
      <span style="font-size:10px;color:var(--fm,#5f6b78);flex:0 0 auto;width:40px;text-align:right">${timeAgo(r.updated_at_unix)}</span>
    </div>`;
  }).join('');
  return panel('Recently finished', '', `<div style="max-height:228px;overflow-y:auto">${body}</div>`, header);
}

function resourcesPanel() {
  const gpus = S.live.gpus.map((g) => {
    // Any field can be null: nvidia-smi answers "[N/A]" for what the host does
    // not expose. Under MIG there is no parent-level utilisation, and a vGPU
    // guest sees no temperature -- but memory is still real, so the bar falls
    // back to memory used rather than the card disappearing.
    const hasUtil = g.util !== null && g.util !== undefined;
    const hasMem = g.mem_used_gib !== null && g.mem_total_gib;
    const pct = hasUtil ? Math.round(g.util)
      : (hasMem ? Math.round((g.mem_used_gib / g.mem_total_gib) * 100) : null);
    const color = pct === null ? 'var(--gd,#222a34)'
      : (pct > 85 ? 'var(--no,#f0606a)' : (pct > 60 ? 'var(--wn,#e3b341)' : 'var(--ok,#3fcf8e)'));
    const mem = hasMem ? `${g.mem_used_gib.toFixed(1)}/${g.mem_total_gib.toFixed(0)} GiB` : 'memory n/r';
    return `<div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:11px">
        <span style="font-family:${MONO};color:var(--fg,#e6ebf0)">GPU${g.id}</span>
        <span style="color:var(--fm,#5f6b78);font-size:10px">${esc(g.name)}</span>
        <div style="flex:1"></div>
        <span style="font-family:${MONO};color:var(--fd,#9aa6b2)">${mem}</span>
        <span style="font-family:${MONO};color:${color};width:38px;text-align:right">${pct === null ? '—' : pct + '%'}</span>
      </div>
      <div style="height:7px;border-radius:4px;background:var(--p2,#1b212a);overflow:hidden">
        <div style="height:100%;width:${pct === null ? 0 : pct}%;background:${color};border-radius:4px"></div>
      </div>
      ${hasUtil ? '' : `<div style="margin-top:4px;font-size:9.5px;color:var(--fm,#5f6b78)">utilisation not reported by nvidia-smi on this host${hasMem ? ' — bar shows memory used' : ''}</div>`}
    </div>`;
  }).join('');

  const logs = S.live.logs.map((l) => `<div style="display:flex;gap:8px">
      <span style="color:#3b4654;flex:0 0 auto">${esc(l.t)}</span>
      <span style="color:${l.stream === 'err' ? 'var(--no,#f0606a)' : 'var(--fd,#9aa6b2)'};white-space:pre-wrap;word-break:break-word">${esc(l.text)}</span>
    </div>`).join('');

  const inner = `<div style="padding:14px 16px;display:flex;flex-direction:column;gap:12px">
      ${gpus || `<div style="font-size:11.5px;color:var(--fm,#5f6b78);line-height:1.6">nvidia-smi reported no GPU on this host.</div>`}
    </div>
    <div style="border-top:1px solid var(--bd,#252c36);background:#07090c">
      <div style="display:flex;gap:12px;padding:7px 14px;font-size:9.5px;font-family:${MONO};color:var(--fm,#5f6b78);border-bottom:1px solid #14191f">
        <span style="color:var(--fd,#9aa6b2)">session log</span><span>· live tail</span>
      </div>
      <div id="logpane" style="height:176px;overflow-y:auto;padding:8px 14px;font-family:${MONO};font-size:11px;line-height:1.55">
        ${logs || `<div style="color:#3b4654">no session log on disk yet</div>`}
      </div>
    </div>`;
  const badge = `<span style="font-size:9px;color:var(--fm,#5f6b78);padding:1px 6px;border:1px solid var(--bd,#252c36);border-radius:4px">nvidia-smi</span>`;
  return panel('VM resources', '', inner, badge);
}

function liveView() {
  if (!S.live) return loadingView('live monitoring');
  const subtitle = S.live.run_state_available
    ? 'Running set + sweep denominators from the central script’s run-state report'
    : 'Firestore-only mode · currently-running set unavailable';
  const vizBtns = [['cards', 'Cards'], ['timeline', 'Timeline'], ['gauges', 'Gauges']].map(([k, l]) =>
    `<div data-act="viz" data-arg="${k}" style="padding:5px 12px;border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;${S.liveViz === k ? 'background:var(--ac,#36c08f);color:#0d1014' : 'color:var(--fd,#9aa6b2)'}">${l}</div>`).join('');
  const reportChip = `<div style="display:flex;align-items:center;gap:7px;font-size:11px;color:var(--fd,#9aa6b2);font-family:${MONO};padding:5px 11px;border-radius:8px;border:1px solid var(--bd,#252c36);background:var(--p2,#1b212a)">
      <span style="width:7px;height:7px;border-radius:50%;background:${S.live.run_state_available ? 'var(--ac,#36c08f)' : 'var(--wn,#e3b341)'}"></span>
      run-state report ${S.live.run_state_available ? 'live' : 'unavailable'}
    </div>`;

  return `<div>
    <div style="display:flex;align-items:flex-end;gap:14px;margin-bottom:16px">
      <div>
        <div style="font-size:20px;font-weight:700;letter-spacing:-.01em">Live monitoring</div>
        <div style="font-size:12px;color:var(--fd,#9aa6b2);margin-top:2px">${esc(subtitle)}</div>
      </div>
      <div style="flex:1"></div>
      ${stopSweepControls(S.live.worker, true)}
      ${reportChip}
      <div style="display:flex;gap:3px;background:var(--p2,#1b212a);border:1px solid var(--bd,#252c36);border-radius:8px;padding:3px">${vizBtns}</div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 380px;gap:14px;align-items:start">
      <div style="display:flex;flex-direction:column;gap:14px;min-width:0">${nowPanel()}${sweepsPanel()}</div>
      <div style="display:flex;flex-direction:column;gap:14px">${recentPanel()}${resourcesPanel()}</div>
    </div>
  </div>`;
}

// ---------- results view ----------
function filteredRuns() {
  if (!S.results) return [];
  const f = S.filters, q = f.q.trim().toLowerCase();
  return S.results.runs.filter((r) => {
    if (f.attacks && f.attacks[r.attack] === false) return false;
    if (f.status !== 'all' && r.status !== f.status) return false;
    if (f.model !== 'all' && r.config.model_id !== f.model) return false;
    if (f.mech !== 'all' && String(r.config.ldp_mechanism === undefined ? 'none' : r.config.ldp_mechanism) !== f.mech) return false;
    if (q && !r.run_id.toLowerCase().includes(q)) return false;
    return true;
  });
}
function sortedRows() {
  const { key, dir } = S.sort, m = dir === 'asc' ? 1 : -1;
  const get = (r) => {
    switch (key) {
      case 'attack': return r.attack;
      case 'run_id': return r.run_id;
      case 'status': return r.status;
      case 'model': return r.config.model_id || '';
      case 'rounds': return r.config.federated_rounds ?? -1;
      case 'clients': return r.config.num_clients ?? -1;
      case 'eps': return r.config.epsilon === null || r.config.epsilon === undefined ? Infinity : r.config.epsilon;
      case 'seed': return r.config.seed ?? -1;
      case 'adv': return r.metrics && r.metrics.adv !== null ? r.metrics.adv : -1;
      case 'updated': return r.updated_at_unix || 0;
      default: return 0;
    }
  };
  return filteredRuns().slice().sort((a, b) => {
    const x = get(a), y = get(b);
    return x < y ? -m : x > y ? m : 0;
  });
}

const COLUMNS = [['attack', 'attack'], ['run_id', 'run_id'], ['status', 'status'], ['model', 'model'],
  ['rounds', 'R'], ['clients', 'C'], ['eps', 'privacy'], ['seed', 'seed'], ['adv', 'Adv'],
  ['tprtnr', 'TPR/TNR'], ['updated', 'updated'], ['actions', '']];
const GRID = '82px 160px 96px 140px 58px 58px 88px 54px 130px 80px 82px 92px';

function deleteRunControls(runId, compact) {
  if (!S.local) return '';
  const armed = S.deleteConfirmRunId === runId;
  const pad = compact ? '3px 7px' : '5px 10px';
  if (!armed) {
    return `<div data-act="run-delete-arm" data-arg="${esc(runId)}" style="font-size:10px;cursor:pointer;padding:${pad};border-radius:6px;color:var(--no,#f0606a);border:1px solid color-mix(in srgb,var(--no,#f0606a) 38%,var(--bd));text-align:center">delete</div>`;
  }
  return `<div style="display:flex;align-items:center;gap:4px">
    <div data-act="run-delete-confirm" data-arg="${esc(runId)}" style="font-size:9.5px;font-weight:700;cursor:pointer;padding:${pad};border-radius:6px;color:#fff;background:var(--no,#f0606a);border:1px solid var(--no,#f0606a)">confirm</div>
    <div data-act="run-delete-cancel" data-arg="${esc(runId)}" style="font-size:12px;cursor:pointer;padding:2px 5px;color:var(--fm,#5f6b78)" title="cancel">×</div>
  </div>`;
}

function resultsView() {
  if (!S.results) return loadingView('results');
  initFilters();
  const rows = sortedRows();
  const attacksMeta = S.results.attacks;

  const chips = Object.keys(attacksMeta).sort().map((k) => {
    const on = S.filters.attacks[k] !== false, m = attacksMeta[k];
    return `<div data-act="attackchip" data-arg="${k}" style="font-size:11px;font-weight:600;font-family:${MONO};padding:4px 10px;border-radius:6px;cursor:pointer;border:1px solid ${on ? m.color : 'var(--bd,#252c36)'};color:${on ? '#0d1014' : 'var(--fm,#5f6b78)'};background:${on ? m.color : 'transparent'}">${esc(m.label)}</div>`;
  }).join('');

  const opts = (list, current) => list.map(([v, label]) =>
    `<option value="${esc(v)}"${String(v) === String(current) ? ' selected' : ''}>${esc(label)}</option>`).join('');
  const modelOpts = [['all', 'all models']].concat(S.results.models.map((m) => [m, shortModel(m)]));
  const mechOpts = [['all', 'all mechanisms']].concat(S.results.mechanisms.map((m) => [m, m]));
  const xOpts = S.results.x_factors.map((f) => [f, f]);

  const columns = COLUMNS.map(([k, label]) => {
    if (k === 'actions') return `<div style="font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--fm,#5f6b78);padding:12px 0">actions</div>`;
    const sk = k === 'tprtnr' ? 'adv' : k, active = S.sort.key === sk;
    const arrow = active ? (S.sort.dir === 'asc' ? ' ↑' : ' ↓') : '';
    return `<div data-act="sort" data-arg="${sk}" style="font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:${active ? 'var(--ac,#36c08f)' : 'var(--fm,#5f6b78)'};padding:12px 0;cursor:pointer;user-select:none">${label}${arrow}</div>`;
  }).join('');

  const body = rows.map((r) => {
    const m = meta(r.attack), adv = r.metrics ? r.metrics.adv : null;
    const failed = r.status === 'failed';
    const statusColor = failed ? 'var(--no,#f0606a)' : 'var(--ok,#3fcf8e)';
    const cfg = r.config;
    const advCell = adv === null || adv === undefined
      ? `<span style="color:var(--fm,#5f6b78)">—</span>`
      : `<div style="display:flex;align-items:center;gap:7px">
           <div style="flex:1;height:5px;border-radius:3px;background:var(--gd,#222a34);overflow:hidden"><div style="height:100%;width:${clamp((adv - 0.45) / 0.55 * 100, 0, 100)}%;background:${advColor(adv)}"></div></div>
           <span style="font-family:${MONO};font-size:11.5px;color:${advColor(adv)};font-weight:600;width:34px;text-align:right">${fmt3(adv)}</span>
         </div>`;
    const noDp = !cfg.ldp_mechanism || cfg.ldp_mechanism === 'none' || cfg.epsilon === null || cfg.epsilon === undefined;
    return `<div class="row-hover" data-act="open" data-arg="${esc(r.run_id)}" style="display:grid;grid-template-columns:${GRID};gap:0 10px;padding:14px 16px;border-bottom:1px solid var(--gd,#222a34);cursor:pointer;align-items:center;font-size:12px;line-height:1.4">
      <div><span style="font-size:9.5px;font-weight:600;font-family:${MONO};padding:1px 6px;border-radius:4px;color:#0d1014;background:${m.color}">${esc(m.label)}</span></div>
      <div style="font-family:${MONO};color:var(--fd,#9aa6b2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(r.run_id)}</div>
      <div><span style="font-size:10px;color:${statusColor};display:inline-flex;align-items:center;gap:5px"><span style="width:6px;height:6px;border-radius:50%;background:${statusColor}"></span>${esc(r.status)}</span></div>
      <div style="font-family:${MONO};color:var(--fd,#9aa6b2);font-size:10.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(shortModel(cfg.model_id) || '—')}</div>
      <div style="font-family:${MONO};color:var(--fd,#9aa6b2)">${esc(cfg.federated_rounds ?? '—')}</div>
      <div style="font-family:${MONO};color:var(--fd,#9aa6b2)">${esc(cfg.num_clients ?? '—')}</div>
      <div style="font-family:${MONO};color:${noDp ? 'var(--fm,#5f6b78)' : 'var(--ac,#36c08f)'};font-size:10.5px">${esc(mechEps(cfg))}</div>
      <div style="font-family:${MONO};color:var(--fd,#9aa6b2)">${esc(cfg.seed ?? '—')}</div>
      <div>${advCell}</div>
      <div style="font-family:${MONO};color:var(--fd,#9aa6b2);font-size:10.5px">${r.metrics ? fmt3(r.metrics.tpr) + '/' + fmt3(r.metrics.tnr) : '—'}</div>
      <div style="font-family:${MONO};color:var(--fm,#5f6b78);font-size:10px">${timeAgo(r.updated_at_unix)} ago</div>
      <div style="display:flex;justify-content:flex-end">${deleteRunControls(r.run_id, true)}</div>
    </div>`;
  }).join('');

  const summary = Object.keys(attacksMeta).sort().filter((k) => S.filters.attacks[k] !== false).map((k) => {
    const m = attacksMeta[k];
    const vals = rows.filter((r) => r.attack === k && r.metrics && r.metrics.adv !== null)
      .map((r) => r.metrics.adv).sort((a, b) => a - b);
    const mean = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    const median = vals.length ? vals[Math.floor(vals.length / 2)] : null;
    return `<div style="flex:1;min-width:110px;border:1px solid var(--bd,#252c36);border-radius:9px;padding:11px 12px;background:var(--p2,#1b212a)">
      <div style="display:flex;align-items:center;gap:6px;margin-bottom:7px">
        <span style="width:8px;height:8px;border-radius:2px;background:${m.color}"></span>
        <span style="font-size:10.5px;color:var(--fd,#9aa6b2);font-weight:600">${esc(m.label)}</span>
      </div>
      <div style="font-family:${MONO};font-size:22px;font-weight:600;line-height:1">${fmt3(mean)}</div>
      <div style="font-size:9.5px;color:var(--fm,#5f6b78);margin-top:4px">n=${vals.length} · med ${fmt3(median)}</div>
    </div>`;
  }).join('');

  const caption = S.privacyView
    ? 'Privacy reading: lower Adv = privacy improved within a matched environment. Reading Adv against epsilon / mechanism shows the privacy–utility direction. The dashboard reports the direction the runs imply; it makes no privacy claim of its own.'
    : `Each point is one run’s Adv vs ${esc(S.xFactor)}. Series coloured by attack. Higher Adv = stronger attack = weaker privacy.`;

  return `<div>
    <div style="display:flex;align-items:flex-end;gap:14px;margin-bottom:14px">
      <div>
        <div style="font-size:20px;font-weight:700;letter-spacing:-.01em">Results</div>
        <div style="font-size:12px;color:var(--fd,#9aa6b2);margin-top:2px">${S.results.runs.length} runs · ${Object.keys(attacksMeta).length} attacks · Firestore-driven (authoritative results layer)</div>
      </div>
      <div style="flex:1"></div>
      <div data-act="privacy" data-hover style="font-size:11px;cursor:pointer;padding:6px 12px;border-radius:8px;border:1px solid ${S.privacyView ? 'var(--ac,#36c08f)' : 'var(--bd,#252c36)'};color:${S.privacyView ? 'var(--ac,#36c08f)' : 'var(--fd,#9aa6b2)'};background:var(--p2,#1b212a)">privacy reading ${S.privacyView ? 'on' : 'off'}</div>
    </div>

    <div style="display:flex;flex-wrap:wrap;align-items:center;gap:9px;padding:11px 13px;background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:10px;margin-bottom:14px">
      <span style="font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--fm,#5f6b78)">attack</span>
      ${chips}
      <div style="width:1px;height:20px;background:var(--bd,#252c36)"></div>
      <select data-inp="status" style="${SEL}">${opts([['all', 'all status'], ['complete', 'complete'], ['failed', 'failed']], S.filters.status)}</select>
      <select data-inp="model" style="${SEL}">${opts(modelOpts, S.filters.model)}</select>
      <select data-inp="mech" style="${SEL}">${opts(mechOpts, S.filters.mech)}</select>
      <div style="flex:1"></div>
      <input data-inp="q" value="${esc(S.filters.q)}" placeholder="search run_id…" style="font-size:11px;padding:6px 11px;border-radius:7px;border:1px solid var(--bd,#252c36);background:var(--p2,#1b212a);color:var(--fg,#e6ebf0);width:160px;font-family:${MONO};outline:none">
      <span style="font-size:11px;color:var(--fm,#5f6b78);font-family:${MONO}">${rows.length} / ${S.results.runs.length}</span>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px">
      <div style="background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:12px;padding:15px 16px">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
          <span style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--fd,#9aa6b2)">Adv across runs</span>
          <div style="flex:1"></div>
          <span style="font-size:10px;color:var(--fm,#5f6b78)">x-axis</span>
          <select data-inp="xfactor" style="${SEL}">${opts(xOpts, S.xFactor)}</select>
        </div>
        <div style="height:230px"><canvas data-chart="compare"></canvas></div>
        <div style="font-size:10.5px;color:var(--fm,#5f6b78);margin-top:8px;line-height:1.5">${caption}</div>
      </div>
      <div style="background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:12px;padding:15px 16px">
        <div style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--fd,#9aa6b2);margin-bottom:6px">Mean Adv by attack</div>
        <div style="display:flex;gap:10px;margin:10px 0 14px;flex-wrap:wrap">${summary}</div>
        <div style="height:118px"><canvas data-chart="agg"></canvas></div>
      </div>
    </div>

    <div style="background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:12px;overflow:hidden">
      <div style="overflow-x:auto"><div style="min-width:1190px">
        <div style="display:grid;grid-template-columns:${GRID};gap:0 10px;padding:0 16px;border-bottom:2px solid var(--bd,#252c36);background:var(--p2,#1b212a);position:sticky;top:0">${columns}</div>
        <div data-scroll="table" style="max-height:420px;overflow-y:auto">${body || emptyNote('', 'No runs match the active filter.')}</div>
      </div></div>
    </div>
  </div>`;
}

// ---------- detail view ----------
function detailView() {
  const back = `<div data-act="back" style="display:inline-flex;align-items:center;gap:7px;font-size:12px;color:var(--fd,#9aa6b2);cursor:pointer;margin-bottom:14px">← back to results</div>`;
  if (S.detailError) {
    return `<div style="max-width:1080px;margin:0 auto">${back}${emptyNote('run not found', esc(S.detailError))}</div>`;
  }
  if (!S.detail) return `<div style="max-width:1080px;margin:0 auto">${back}${loadingView('run detail')}</div>`;

  const d = S.detail, m = d.meta, mt = d.metrics || {};
  const failed = d.status === 'failed';
  const statusColor = failed ? 'var(--no,#f0606a)' : 'var(--ok,#3fcf8e)';
  const metrics = [
    { label: 'Adv', value: fmt3(mt.adv), color: advColor(mt.adv ?? null), sub: '0.5·TPR + 0.5·TNR',
      accent: `border-color:color-mix(in srgb,${advColor(mt.adv ?? null)} 50%,var(--bd))` },
    { label: 'TPR', value: fmt3(mt.tpr), color: 'var(--fg,#e6ebf0)', sub: 'members detected', accent: '' },
    { label: 'TNR', value: fmt3(mt.tnr), color: 'var(--fg,#e6ebf0)', sub: 'non-members detected', accent: '' },
    { label: 'trials', value: mt.num_trials ?? '—', color: 'var(--fg,#e6ebf0)', sub: 'attack trials run', accent: '' },
  ].map((x) => `<div style="background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:11px;padding:15px 16px;${x.accent}">
      <div style="font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--fm,#5f6b78);margin-bottom:8px">${x.label}</div>
      <div style="font-family:${MONO};font-size:30px;font-weight:600;line-height:1;color:${x.color}">${esc(x.value)}</div>
      <div style="font-size:10px;color:var(--fm,#5f6b78);margin-top:6px">${x.sub}</div>
    </div>`).join('');

  const methodology = Object.entries(d.methodology || {}).map(([k, v]) =>
    `<div style="margin-bottom:11px">
       <div style="font-size:10px;color:var(--ac,#36c08f);font-family:${MONO};margin-bottom:3px">${esc(k)}</div>
       <div style="font-size:12px;color:var(--fd,#9aa6b2);line-height:1.5">${esc(typeof v === 'object' ? JSON.stringify(v) : v)}</div>
     </div>`).join('') || `<div style="font-size:12px;color:var(--fm,#5f6b78)">No methodology recorded on this document.</div>`;

  const config = d.config.map((c) => `<div style="display:flex;justify-content:space-between;gap:8px;font-family:${MONO};font-size:11px;border-bottom:1px solid var(--gd,#222a34);padding-bottom:5px">
      <span style="color:var(--fm,#5f6b78)">${esc(c.k)}</span><span style="color:var(--fg,#e6ebf0);text-align:right;word-break:break-all">${esc(c.v === null ? '∞' : (typeof c.v === 'object' ? JSON.stringify(c.v) : c.v))}</span>
    </div>`).join('');

  const trials = d.trials.slice(0, 200).map((t) => {
    const hit = t.pred_member === t.truth_member;
    return `<div style="display:grid;grid-template-columns:60px 1fr 1fr 1fr;padding:6px 16px;border-bottom:1px solid var(--gd,#222a34);font-family:${MONO};font-size:11px;align-items:center">
      <span style="color:var(--fm,#5f6b78)">${esc(t.trial_id)}</span>
      <span style="color:${t.truth_member ? 'var(--ac,#36c08f)' : 'var(--fm,#5f6b78)'}">${t.truth_member}</span>
      <span style="color:var(--fg,#e6ebf0)">${Number(t.score).toPrecision(5)}</span>
      <span style="color:${t.pred_member ? 'var(--fg,#e6ebf0)' : 'var(--fm,#5f6b78)'}">${t.pred_member} <span style="color:${hit ? 'var(--ok,#3fcf8e)' : 'var(--no,#f0606a)'}">${hit ? '✓' : '✗'}</span></span>
    </div>`;
  }).join('');

  const artifacts = d.artifacts.map((a) => `<div style="display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid var(--gd,#222a34)">
      <span style="font-size:10px;color:var(--fm,#5f6b78);width:170px;flex:0 0 auto">${esc(a.k)}</span>
      <span style="font-family:${MONO};font-size:11px;color:var(--fd,#9aa6b2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(a.v)}</span>
      <div style="flex:1"></div>
      <span style="font-size:9.5px;color:${a.authoritative ? 'var(--ok,#3fcf8e)' : 'var(--fm,#5f6b78)'};border:1px solid var(--bd,#252c36);padding:1px 7px;border-radius:5px;flex:0 0 auto">${a.authoritative ? 'authoritative' : 'may be cleaned up'}</span>
    </div>`).join('');

  const chartCard = (title, id, empty) => `<div style="background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:12px;padding:16px">
      <div style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--fd,#9aa6b2);margin-bottom:10px">${title}</div>
      ${empty ? `<div style="height:208px;display:flex;align-items:center;justify-content:center;text-align:center;font-size:11.5px;color:var(--fm,#5f6b78);line-height:1.6">${empty}</div>`
              : `<div style="height:208px"><canvas data-chart="${id}"></canvas></div>`}
    </div>`;

  return `<div style="max-width:1080px;margin:0 auto">
    ${back}
    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:6px">
      <span style="font-size:12px;font-weight:600;font-family:${MONO};padding:4px 10px;border-radius:7px;color:#0d1014;background:${m.color}">${esc(m.label)}</span>
      <span style="font-size:21px;font-weight:700;font-family:${MONO}">${esc(d.run_id)}</span>
      <span style="font-size:11px;color:${statusColor};display:inline-flex;align-items:center;gap:6px;border:1px solid var(--bd,#252c36);padding:3px 9px;border-radius:7px"><span style="width:7px;height:7px;border-radius:50%;background:${statusColor}"></span>${esc(d.status)}</span>
      <div style="flex:1"></div>
      <span style="font-size:11px;color:var(--fm,#5f6b78);font-family:${MONO}">updated_at ${timeAgo(d.updated_at_unix)} ago</span>
      ${deleteRunControls(d.run_id, false)}
    </div>
    <div style="font-size:12px;color:var(--fd,#9aa6b2);margin-bottom:18px">${esc(m.title)}</div>
    ${d.error ? `<div style="background:color-mix(in srgb,var(--no,#f0606a) 8%,var(--pn,#15191f));border:1px solid color-mix(in srgb,var(--no,#f0606a) 30%,var(--bd));border-radius:10px;padding:13px 16px;margin-bottom:16px;font-family:${MONO};font-size:11.5px;color:var(--no,#f0606a);white-space:pre-wrap">${esc(d.error)}</div>` : ''}
    ${d.prior_error ? `<div style="background:var(--p2,#1b212a);border:1px solid var(--bd,#252c36);border-radius:10px;padding:11px 16px;margin-bottom:16px;font-size:11px;color:var(--fm,#5f6b78);line-height:1.6">
      <b style="color:var(--fd,#9aa6b2)">An earlier attempt at this run_id failed and was later recovered.</b>
      This run completed; the message below is history from the previous attempt.
      <div style="font-family:${MONO};margin-top:6px;white-space:pre-wrap">${esc(d.prior_error)}</div>
    </div>` : ''}

    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px">${metrics}</div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px">
      <div style="background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:12px;padding:16px">
        <div style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--fd,#9aa6b2);margin-bottom:12px">Methodology</div>${methodology}
      </div>
      <div style="background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:12px;padding:16px">
        <div style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--fd,#9aa6b2);margin-bottom:12px">Config</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px 16px">${config}</div>
      </div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px">
      ${chartCard('Federated training history', 'fl', d.federated_history.length ? '' : 'This attack’s document records no per-round loss.')}
      ${chartCard('Attack-score distribution', 'dist', d.trials.length ? '' : 'No attack_trials recorded on this document.')}
    </div>

    <div style="display:grid;grid-template-columns:340px 1fr;gap:14px;margin-bottom:14px">
      ${chartCard('ROC · from trial scores', 'roc', d.trials.length ? '' : 'No attack_trials recorded.')}
      <div style="background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:12px;overflow:hidden">
        <div style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--fd,#9aa6b2);padding:16px 16px 10px">attack_trials[] · ${d.trials.length} records</div>
        <div style="display:grid;grid-template-columns:60px 1fr 1fr 1fr;padding:0 16px 7px;font-size:9.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--fm,#5f6b78);border-bottom:1px solid var(--bd,#252c36)">
          <span>trial</span><span>truth_member</span><span>score</span><span>pred_member</span>
        </div>
        <div data-scroll="trials" style="max-height:218px;overflow-y:auto">${trials || emptyNote('', 'No trials recorded.')}</div>
      </div>
    </div>

    <div style="background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:12px;padding:16px">
      <div style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--fd,#9aa6b2);margin-bottom:12px">Artifacts</div>${artifacts}
    </div>
  </div>`;
}

// ---------- access view ----------
/* What the form should show: whatever the user has typed this session, falling
 * back to what the server has saved in the .env. Saved credentials arrive
 * masked, so they are a placeholder — the field itself stays empty until the
 * user types a replacement. */
function tunnelFields() {
  const saved = (S.tunnel && S.tunnel.saved) || {};
  const f = S.tunnelForm;
  const typed = (v) => v !== null && v !== undefined;
  return {
    provider: f.provider || saved.provider || 'cloudflare',
    apiKey: typed(f.apiKey) ? f.apiKey : '',
    code: typed(f.code) ? f.code : '',
    port: typed(f.port) ? f.port : String(saved.port || 8080),
    apiKeyEdited: typed(f.apiKey),
    codeEdited: typed(f.code),
    apiKeySaved: !!saved.api_key_set,
    codeSaved: !!saved.code_set,
    apiKeyMask: saved.api_key_mask || '',
    codeMask: saved.code_mask || '',
    // A code, typed or saved, is what makes the public URL survive a restart.
    hasCode: typed(f.code) ? f.code.trim().length > 0 : !!saved.code_set,
  };
}

function accessView() {
  const t = S.tunnel || { connected: false };
  const live = !!t.connected;
  const form = tunnelFields();
  const prov = form.provider;
  const stable = form.hasCode;
  const savedNote = (isSaved, mask) => isSaved
    ? `<span style="font-size:9.5px;color:var(--ok,#3fcf8e);font-family:${MONO};margin-left:7px">saved · ${esc(mask)}</span>`
    : '';
  const dotFor = (k) => (k === 'cloudflare' ? '#f38020' : '#1f6feb');

  const providers = [['cloudflare', 'Cloudflare', 'cloudflared named tunnel'], ['ngrok', 'ngrok', 'authtoken + reserved domain']]
    .map(([k, name, sub]) => `<div data-act="provider" data-arg="${k}" style="padding:14px 16px;border-radius:10px;cursor:pointer;border:2px solid ${prov === k ? dotFor(k) : 'var(--bd,#252c36)'};background:${prov === k ? 'color-mix(in srgb,' + dotFor(k) + ' 8%,var(--p2,#1b212a))' : 'var(--p2,#1b212a)'}">
        <div style="display:flex;align-items:center;gap:10px">
          <div style="width:8px;height:8px;border-radius:50%;background:${dotFor(k)};flex:0 0 auto"></div>
          <div><div style="font-size:13px;font-weight:600">${name}</div>
               <div style="font-size:10.5px;color:var(--fm,#5f6b78);margin-top:2px">${sub}</div></div>
        </div>
      </div>`).join('');

  const stats = [
    { k: 'provider', v: t.provider || '—', color: 'var(--fg,#e6ebf0)' },
    { k: 'target port', v: t.port ? 'localhost:' + t.port : '—', color: 'var(--fg,#e6ebf0)' },
    { k: 'uptime', v: t.started_unix ? elapsedFmt(now() - t.started_unix) : '—', color: 'var(--fg,#e6ebf0)' },
    { k: 'url lifecycle', v: t.ephemeral ? 'ephemeral' : 'stable', color: t.ephemeral ? 'var(--wn,#e3b341)' : 'var(--ok,#3fcf8e)' },
  ].map((s) => `<div style="padding:0 16px;border-right:1px solid var(--bd,#252c36)">
      <div style="font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--fm,#5f6b78);margin-bottom:5px">${s.k}</div>
      <div style="font-family:${MONO};font-size:14px;font-weight:500;color:${s.color}">${esc(s.v)}</div>
    </div>`).join('');

  const urlStrip = !live ? '' : `<div style="margin-top:18px;padding-top:18px;border-top:1px solid color-mix(in srgb,var(--ok,#3fcf8e) 18%,var(--bd,#252c36))">
      <div style="display:flex;align-items:center;gap:10px;background:var(--p2,#1b212a);border:1px solid var(--bd,#252c36);border-radius:9px;padding:12px 14px;margin-bottom:16px">
        <span style="font-size:9.5px;color:var(--fm,#5f6b78);letter-spacing:.12em;text-transform:uppercase;flex:0 0 auto">public URL</span>
        <span style="font-family:${MONO};font-size:13px;color:var(--ac,#36c08f);flex:1;overflow:hidden;text-overflow:ellipsis">${esc(t.url || 'waiting for the agent to report a URL…')}</span>
        ${t.url ? `<div data-act="copy" style="font-size:11px;font-weight:500;color:var(--fd,#9aa6b2);border:1px solid var(--bd,#252c36);border-radius:6px;padding:5px 13px;cursor:pointer;flex:0 0 auto;white-space:nowrap">${S.copied ? 'copied' : 'copy'}</div>` : ''}
      </div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0">${stats}</div>
    </div>`;

  const warning = !live ? '' : `<div style="display:grid;grid-template-columns:auto 1fr;align-items:center;gap:14px;background:color-mix(in srgb,var(--wn,#e3b341) 8%,var(--pn,#15191f));border:1px solid color-mix(in srgb,var(--wn,#e3b341) 30%,var(--bd,#252c36));border-radius:10px;padding:14px 18px;margin-top:14px">
      <div style="width:30px;height:30px;border-radius:50%;border:2px solid var(--wn,#e3b341);color:var(--wn,#e3b341);display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;flex:0 0 auto">!</div>
      <div style="font-size:12px;color:var(--fd,#9aa6b2);line-height:1.6"><b style="color:var(--fg,#e6ebf0)">External access is currently live.</b> This monitor is reachable on the public internet for as long as the tunnel is up. The URL above is the access boundary — share it deliberately.</div>
    </div>`;

  const err = t.error ? `<div style="margin-top:14px;padding:12px 16px;border-radius:10px;border:1px solid color-mix(in srgb,var(--no,#f0606a) 30%,var(--bd));background:color-mix(in srgb,var(--no,#f0606a) 8%,var(--pn,#15191f));font-family:${MONO};font-size:11.5px;color:var(--no,#f0606a)">${esc(t.error)}</div>` : '';

  return `<div style="max-width:900px;margin:0 auto">
    <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin-bottom:20px">
      <div>
        <div style="font-size:20px;font-weight:700;letter-spacing:-.01em">Remote access</div>
        <div style="font-size:12px;color:var(--fd,#9aa6b2);margin-top:4px;max-width:480px;line-height:1.6">The VM can dial out but can't be reached inbound. A tunnel agent opens an outbound connection so a public URL proxies traffic back to the local dashboard.</div>
      </div>
    </div>

    <div style="background:${live ? 'color-mix(in srgb,var(--ok,#3fcf8e) 9%,var(--pn,#15191f))' : 'var(--pn,#15191f)'};border:1px solid ${live ? 'color-mix(in srgb,var(--ok,#3fcf8e) 35%,var(--bd))' : 'var(--bd,#252c36)'};border-radius:14px;padding:18px 20px;overflow:hidden">
      <div style="display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:20px">
        <div style="width:52px;height:52px;border-radius:50%;border:2px solid ${live ? 'var(--ok,#3fcf8e)' : 'var(--fm,#5f6b78)'};display:flex;align-items:center;justify-content:center;position:relative">
          <span style="width:20px;height:20px;border-radius:50%;background:${live ? 'var(--ok,#3fcf8e)' : 'var(--fm,#5f6b78)'};${live ? 'animation:pulse 1.6s infinite' : ''}"></span>
        </div>
        <div>
          <div style="font-size:15px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:${live ? 'var(--ok,#3fcf8e)' : 'var(--fg,#e6ebf0)'}">${live ? 'EXTERNAL ACCESS LIVE' : 'EXTERNAL ACCESS OFFLINE'}</div>
          <div style="font-size:12px;color:var(--fd,#9aa6b2);margin-top:5px;line-height:1.5">${live ? 'Outbound tunnel established · viewers off the VPN can reach the monitor' : 'Dashboard is reachable only on the VM / intranet VPN'}</div>
        </div>
        <div data-act="${live ? 'tunnel-stop' : 'tunnel-start'}" data-hover style="font-size:12px;font-weight:600;cursor:pointer;padding:9px 18px;border-radius:9px;${live ? 'background:transparent;border:1px solid var(--no,#f0606a);color:var(--no,#f0606a)' : 'background:var(--ac,#36c08f);border:1px solid var(--ac,#36c08f);color:#0d1014'}">${live ? 'Stop tunnel' : 'Start tunnel'}</div>
      </div>
      ${urlStrip}
    </div>
    ${warning}${err}

    <div style="background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:12px;overflow:hidden;margin-top:14px">
      <div style="padding:18px 20px;border-bottom:1px solid var(--bd,#252c36)">
        <div style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--fd,#9aa6b2);margin-bottom:12px">Tunnel provider</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">${providers}</div>
      </div>
      <div style="padding:18px 20px;border-bottom:1px solid var(--bd,#252c36)">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
          <div style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--fd,#9aa6b2)">Credentials</div>
          <div style="flex:1"></div>
          <div data-act="tunnel-save" data-hover style="font-size:10.5px;color:var(--fd,#9aa6b2);border:1px solid var(--bd,#252c36);border-radius:6px;padding:4px 12px;cursor:pointer">Save to .env</div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
          <div>
            <div style="font-size:10px;color:var(--fm,#5f6b78);letter-spacing:.08em;text-transform:uppercase;margin-bottom:7px">${prov === 'cloudflare' ? 'API token (cloudflared)' : 'authtoken (ngrok)'}${savedNote(form.apiKeySaved, form.apiKeyMask)}</div>
            <input class="field" data-inp="apiKey" type="password" value="${esc(form.apiKey)}" placeholder="${form.apiKeySaved ? 'unchanged · leave blank to reuse the saved token' : (prov === 'cloudflare' ? 'cf_api_token…' : '2abc…ngrok_authtoken')}" style="${FIELD}">
          </div>
          <div>
            <div style="font-size:10px;color:var(--fm,#5f6b78);letter-spacing:.08em;text-transform:uppercase;margin-bottom:7px">${prov === 'cloudflare' ? 'named-tunnel connector token' : 'reserved domain / tunnel code'}${savedNote(form.codeSaved, form.codeMask)}</div>
            <input class="field" data-inp="code" type="password" value="${esc(form.code)}" placeholder="${form.codeSaved ? 'unchanged · leave blank to reuse the saved code' : (prov === 'cloudflare' ? 'connector token (binds hostname)' : 'my-lab-monitor')}" style="${FIELD}">
          </div>
        </div>
        <div style="font-size:11px;color:var(--fm,#5f6b78);line-height:1.6;margin-top:12px">Starting the tunnel saves these to the <span style="font-family:${MONO}">.env</span> so they come back on the next start. They are stored in the clear there and are never sent back to this page — an already-saved field shows only its shape.${form.apiKeyEdited || form.codeEdited ? '' : ' Clear a field and save to remove the stored value.'}</div>
      </div>
      <div style="padding:18px 20px">
        <div style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--fd,#9aa6b2);margin-bottom:14px">Connection</div>
        <div style="display:grid;grid-template-columns:220px 1fr;gap:14px;align-items:start">
          <div>
            <div style="font-size:10px;color:var(--fm,#5f6b78);letter-spacing:.08em;text-transform:uppercase;margin-bottom:7px">target port (local dashboard)</div>
            <input class="field" data-inp="port" value="${esc(form.port)}" style="${FIELD}">
          </div>
          <div style="background:var(--p2,#1b212a);border:1px solid var(--bd,#252c36);border-radius:9px;padding:12px 14px;display:flex;align-items:flex-start;gap:10px">
            <div style="width:7px;height:7px;border-radius:50%;background:var(--wn,#e3b341);flex:0 0 auto;margin-top:4px"></div>
            <div style="font-size:11.5px;color:var(--fd,#9aa6b2);line-height:1.6">${stable ? 'A code/reserved domain is set → URL is stable across restarts.' : 'No code set → URL is ephemeral and regenerates each session.'}</div>
          </div>
        </div>
      </div>
      <div style="display:flex;align-items:flex-start;gap:10px;padding:13px 20px;background:var(--p2,#1b212a);border-top:1px solid var(--bd,#252c36)">
        <div style="width:6px;height:6px;border-radius:50%;background:var(--fm,#5f6b78);flex:0 0 auto;margin-top:5px"></div>
        <div style="font-size:11px;color:var(--fm,#5f6b78);line-height:1.6">Prerequisite: outbound egress must reach the provider's edge endpoints (${prov === 'cloudflare' ? 'cloudflared control + data endpoints' : 'ngrok control + tunnel endpoints'}), and the agent binary must be installed on the VM.</div>
      </div>
    </div>
  </div>`;
}

// ---------- config editor ----------
async function loadConfig(name) {
  const res = await getJSON('/api/configs/' + encodeURIComponent(name));
  S.editor = { name, text: res.text, dirty: false, validation: null, saveName: name, confirmOverwrite: false };
}

function editorPanel() {
  const e = S.editor;
  const options = S.launch.configs.map((c) =>
    `<option value="${esc(c)}"${c === e.name ? ' selected' : ''}>${esc(c)}</option>`).join('');
  const v = e.validation;
  const readout = !v
    ? `<span style="color:var(--fm,#5f6b78)">${e.dirty ? 'edited · not validated yet' : 'not validated yet'}</span>`
    : v.ok
      ? `<span style="color:var(--ok,#3fcf8e)">✓ ${esc(v.message)}</span>` +
        Object.entries(v.per_attack || {}).map(([a, n]) =>
          `<span style="margin-left:10px;color:var(--fd,#9aa6b2)">${esc(a)} ×${n}</span>`).join('')
      : `<span style="color:var(--no,#f0606a);white-space:pre-wrap">✗ ${esc(v.message)}</span>`;

  const btn = (act, label, primary) => `<div data-act="${act}" data-hover style="font-size:11.5px;font-weight:600;cursor:pointer;padding:7px 14px;border-radius:8px;${primary ? 'background:var(--ac,#36c08f);border:1px solid var(--ac,#36c08f);color:#0d1014' : 'background:var(--p2,#1b212a);border:1px solid var(--bd,#252c36);color:var(--fd,#9aa6b2)'}">${label}</div>`;

  return `<div style="background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:12px;overflow:hidden;margin-bottom:14px">
    <div style="display:flex;align-items:center;gap:10px;padding:14px 20px;border-bottom:1px solid var(--bd,#252c36)">
      <span style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--fd,#9aa6b2)">Config</span>
      <select data-inp="editorFile" style="${SEL};font-size:12px;padding:7px 11px;min-width:220px">
        <option value="">— select a config —</option>${options}
      </select>
      ${btn('config-new', 'New config')}
      <div style="flex:1"></div>
      ${e.dirty ? `<span style="font-size:10.5px;color:var(--wn,#e3b341);font-family:${MONO}">unsaved changes</span>` : ''}
    </div>

    <div style="padding:0">
      <textarea data-inp="configText" spellcheck="false" placeholder="Select a config above, or start a new one."
        style="width:100%;height:340px;resize:vertical;display:block;border:0;border-bottom:1px solid var(--bd,#252c36);background:var(--bg,#0d1014);color:var(--fg,#e6ebf0);font-family:${MONO};font-size:12px;line-height:1.65;padding:14px 18px;outline:none;tab-size:2">${esc(e.text)}</textarea>
    </div>

    <div style="display:flex;align-items:center;gap:10px;padding:13px 20px;border-bottom:1px solid var(--bd,#252c36);flex-wrap:wrap">
      ${btn('config-validate', 'Validate')}
      <span style="font-size:10px;color:var(--fm,#5f6b78);letter-spacing:.08em;text-transform:uppercase;margin-left:8px">save as</span>
      <input data-inp="saveName" value="${esc(e.saveName)}" placeholder="my-sweep.yaml"
        style="font-size:12px;font-family:${MONO};padding:7px 11px;border-radius:8px;border:1px solid var(--bd,#252c36);background:var(--p2,#1b212a);color:var(--fg,#e6ebf0);width:220px;outline:none">
      ${btn('config-save', e.confirmOverwrite ? 'Overwrite' : 'Save', true)}
      ${e.confirmOverwrite ? `<span style="font-size:11px;color:var(--wn,#e3b341)">file exists — Overwrite replaces it</span>` : ''}
    </div>

    <div style="padding:11px 20px;background:var(--p2,#1b212a);font-size:11.5px;font-family:${MONO}">${readout}</div>
  </div>`;
}

// ---------- manual launch mode ----------
function manualBody() {
  return {
    attacks: S.manual.cards.map((c) => ({ name: c.name, values: c.values, sweeps: c.sweeps })),
  };
}

function manualValue(card, field) {
  return Object.prototype.hasOwnProperty.call(card.values, field.name)
    ? card.values[field.name] : String(field.default);
}

function manualField(card, field) {
  const key = `manual:${card.name}:${field.name}`;
  const swept = card.sweeps.includes(field.name);
  const raw = manualValue(card, field);
  const label = `<div style="width:170px;flex:0 0 auto;font-size:11.5px;font-family:${MONO};color:var(--fd,#9aa6b2);overflow:hidden;text-overflow:ellipsis">${esc(field.name)}</div>`;

  // Read-only fields show their value and why it is fixed. Sending one changed
  // is refused by the server, so offering an input would only mislead.
  if (field.readonly) {
    return `<div style="display:flex;align-items:center;gap:10px;padding:5px 0">${label}
      <div style="flex:1;font-size:11.5px;font-family:${MONO};color:var(--fm,#5f6b78);overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(field.default)}">${esc(field.default)}</div>
      <div style="flex:0 0 auto;font-size:10px;color:var(--fm,#5f6b78);letter-spacing:.06em;text-transform:uppercase">${esc(field.reason || 'fixed')}</div>
    </div>`;
  }

  const input = (field.type === 'bool' && !swept)
    ? `<div data-act="manual-bool" data-arg="${esc(card.name)}:${esc(field.name)}" style="flex:1;display:flex;align-items:center;cursor:pointer">
         <div style="width:34px;height:19px;border-radius:11px;position:relative;transition:.15s;background:${raw === 'true' ? 'var(--ac,#36c08f)' : 'var(--gd,#222a34)'}">
           <div style="position:absolute;top:2px;left:${raw === 'true' ? '17px' : '2px'};width:15px;height:15px;border-radius:50%;background:#fff;transition:.15s"></div>
         </div>
       </div>`
    : `<input data-inp="${esc(key)}" value="${esc(raw)}" spellcheck="false"
         style="flex:1;min-width:0;font-size:11.5px;font-family:${MONO};padding:6px 10px;border-radius:7px;border:1px solid ${swept ? 'var(--rn,#4aa8ff)' : 'var(--bd,#252c36)'};background:var(--bg,#0d1014);color:var(--fg,#e6ebf0);outline:none">`;

  const count = swept ? String(raw).split(',').filter((p) => p.trim()).length : 0;
  return `<div style="display:flex;align-items:center;gap:10px;padding:5px 0">${label}${input}
    <div data-act="manual-sweep" data-arg="${esc(card.name)}:${esc(field.name)}" data-hover
      title="${swept ? 'back to a single value' : 'sweep this field over a comma-separated list'}"
      style="flex:0 0 auto;font-size:10px;font-weight:600;font-family:${MONO};padding:4px 8px;border-radius:6px;cursor:pointer;border:1px solid ${swept ? 'var(--rn,#4aa8ff)' : 'var(--bd,#252c36)'};color:${swept ? 'var(--rn,#4aa8ff)' : 'var(--fm,#5f6b78)'}">${swept ? `sweep ×${count}` : '⋯'}</div>
  </div>`;
}

function manualCard(card) {
  const info = S.attackFields.attacks[card.name];
  if (!info) return '';
  const m = meta(card.name);
  const groups = S.attackFields.group_order.map((group) => {
    const rows = info.fields.filter((f) => f.group === group);
    if (!rows.length) return '';
    const body = rows.map((f) => manualField(card, f)).join('');
    const head = (text) => `<div style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--fm,#5f6b78)">${text}</div>`;
    if (group !== 'Advanced') {
      return `<div style="padding:12px 18px;border-top:1px solid var(--bd,#252c36)">
        ${head(esc(group))}<div style="margin-top:6px">${body}</div></div>`;
    }
    // Plumbing: correct by default and rarely the thing being tuned.
    return `<div style="padding:12px 18px;border-top:1px solid var(--bd,#252c36)">
      <div data-act="manual-advanced" data-arg="${esc(card.name)}" style="cursor:pointer">${head(`${card.advanced ? '▾' : '▸'} Advanced`)}</div>
      ${card.advanced ? `<div style="margin-top:6px">${body}</div>` : ''}</div>`;
  }).join('');

  return `<div style="background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:12px;overflow:hidden;margin-bottom:12px">
    <div style="display:flex;align-items:center;gap:10px;padding:12px 18px">
      <div style="width:8px;height:8px;border-radius:2px;background:${m.color}"></div>
      <div style="font-size:12.5px;font-weight:700;font-family:${MONO}">${esc(card.name)}</div>
      <div style="flex:1"></div>
      <div data-act="manual-remove" data-arg="${esc(card.name)}" data-hover style="font-size:11px;cursor:pointer;color:var(--fm,#5f6b78);padding:3px 8px;border-radius:6px">✕</div>
    </div>
    ${groups}
  </div>`;
}

function manualPanel() {
  if (!S.attackFields) return loadingView('attack fields');
  const mn = S.manual;
  const taken = mn.cards.map((c) => c.name);
  const addable = Object.keys(S.attackFields.attacks).filter((a) => !taken.includes(a));
  const v = mn.validation;
  const readout = !v
    ? `<span style="color:var(--fm,#5f6b78)">not validated yet</span>`
    : v.ok
      ? `<span style="color:${mn.dirty ? 'var(--fm,#5f6b78)' : 'var(--ok,#3fcf8e)'}">${mn.dirty ? '·' : '✓'} ${esc(v.message)}${mn.dirty ? ' (edited since)' : ''}</span>` +
        Object.entries(v.per_attack || {}).map(([a, n]) =>
          `<span style="margin-left:10px;color:var(--fd,#9aa6b2)">${esc(a)} ×${n}</span>`).join('')
      : `<span style="color:var(--no,#f0606a);white-space:pre-wrap">✗ ${esc(v.message)}</span>`;

  const btn = (act, label) => `<div data-act="${act}" data-hover style="font-size:11.5px;font-weight:600;cursor:pointer;padding:7px 14px;border-radius:8px;background:var(--p2,#1b212a);border:1px solid var(--bd,#252c36);color:var(--fd,#9aa6b2)">${label}</div>`;

  return `<div>
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;flex-wrap:wrap">
      <select data-inp="manualAdd" style="${SEL};font-size:12px;padding:8px 12px;min-width:200px">
        <option value="">+ add attack</option>
        ${addable.map((a) => `<option value="${esc(a)}">${esc(a)}</option>`).join('')}
      </select>
      <div style="font-size:11.5px;color:var(--fm,#5f6b78)">Edit a field to change it; <span style="font-family:${MONO}">⋯</span> turns it into a comma-separated sweep.</div>
    </div>

    ${mn.cards.length ? mn.cards.map(manualCard).join('')
      : `<div style="background:var(--pn,#15191f);border:1px dashed var(--bd,#252c36);border-radius:12px;padding:34px;text-align:center;font-size:12px;color:var(--fm,#5f6b78);margin-bottom:12px">No attacks yet. Add one above.</div>`}

    <div style="background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:12px;overflow:hidden">
      <div style="display:flex;align-items:center;gap:10px;padding:13px 20px;flex-wrap:wrap">
        ${btn('manual-validate', 'Validate')}
        <span style="font-size:10px;color:var(--fm,#5f6b78);letter-spacing:.08em;text-transform:uppercase;margin-left:8px">save as</span>
        <input data-inp="manualSaveName" value="${esc(mn.saveName)}" placeholder="my-sweep.yaml"
          style="font-size:12px;font-family:${MONO};padding:7px 11px;border-radius:8px;border:1px solid var(--bd,#252c36);background:var(--p2,#1b212a);color:var(--fg,#e6ebf0);width:200px;outline:none">
        ${btn('manual-save', mn.confirmOverwrite ? 'Overwrite' : 'Save as config')}
        ${mn.confirmOverwrite ? `<span style="font-size:11px;color:var(--wn,#e3b341)">file exists — Overwrite replaces it</span>` : ''}
      </div>
      <div style="padding:11px 20px;background:var(--p2,#1b212a);border-top:1px solid var(--bd,#252c36);font-size:11.5px;font-family:${MONO}">${readout}</div>
    </div>
  </div>`;
}

// ---------- launch view ----------
function launchView() {
  if (!S.launch) return loadingView('launch options');
  const w = S.launch.worker;
  const f = S.launchForm;
  const manualMode = f.mode === 'manual';
  const configs = S.launch.configs.map((c) =>
    `<option value="${esc(c)}"${c === f.configFile ? ' selected' : ''}>${esc(c)}</option>`).join('');
  const chips = S.launch.attacks.map((a) => {
    const on = f.attacks.includes(a);
    const m = meta(a);
    return `<div data-act="launch-attack" data-arg="${esc(a)}" style="font-size:11px;font-weight:600;font-family:${MONO};padding:5px 11px;border-radius:6px;cursor:pointer;border:1px solid ${on ? m.color : 'var(--bd,#252c36)'};color:${on ? '#0d1014' : 'var(--fm,#5f6b78)'};background:${on ? m.color : 'transparent'}">${esc(a)}</div>`;
  }).join('');

  const status = w.running
    ? `<span style="color:var(--rn,#4aa8ff)">● running · ${w.finished}/${w.planned} finished · ${elapsedFmt(w.started_unix ? now() - w.started_unix : null)} elapsed</span>`
    : w.error ? `<span style="color:var(--no,#f0606a)">● last sweep failed: ${esc(w.error)}</span>`
    : w.stopped ? `<span style="color:var(--wn,#e3b341)">● idle · last sweep was stopped</span>`
    : w.planned ? `<span style="color:var(--ok,#3fcf8e)">● idle · last sweep finished ${w.finished}/${w.planned}</span>`
    : `<span style="color:var(--fm,#5f6b78)">● idle · no sweep started this session</span>`;

  const tab = (mode, label, hint) => `<div data-act="launch-mode" data-arg="${mode}" data-hover
    style="flex:1;padding:11px 16px;cursor:pointer;border-radius:9px;background:${f.mode === mode ? 'var(--p2,#1b212a)' : 'transparent'};border:1px solid ${f.mode === mode ? 'var(--ac,#36c08f)' : 'transparent'}">
    <div style="font-size:12.5px;font-weight:600;color:${f.mode === mode ? 'var(--fg,#e6ebf0)' : 'var(--fm,#5f6b78)'}">${label}</div>
    <div style="font-size:11px;color:var(--fm,#5f6b78);margin-top:2px">${hint}</div></div>`;

  const existingBody = `${editorPanel()}
    <div style="background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:12px;overflow:hidden">
      <div style="padding:18px 20px;border-bottom:1px solid var(--bd,#252c36)">
        <div style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--fd,#9aa6b2);margin-bottom:12px">Run this config</div>
        <select data-inp="configFile" style="${SEL};font-size:12px;padding:9px 12px;width:280px">${configs || '<option>no configs found</option>'}</select>
        ${S.editor.dirty ? `<div style="margin-top:10px;font-size:11px;color:var(--wn,#e3b341)">The editor has unsaved changes. A sweep runs the file on disk — save first to run what you're looking at.</div>` : ''}
      </div>
      <div style="padding:18px 20px">
        <div style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--fd,#9aa6b2);margin-bottom:12px">Attacks <span style="color:var(--fm,#5f6b78);font-weight:500;letter-spacing:0;text-transform:none">· none selected = every attack in the config</span></div>
        <div style="display:flex;flex-wrap:wrap;gap:7px">${chips}</div>
      </div>
    </div>`;

  return `<div style="max-width:900px;margin:0 auto">
    <div style="margin-bottom:16px">
      <div style="font-size:20px;font-weight:700;letter-spacing:-.01em">Launch a sweep</div>
      <div style="font-size:12px;color:var(--fd,#9aa6b2);margin-top:4px;max-width:560px;line-height:1.6">Runs go through <span style="font-family:${MONO}">core.runner.run_sweep</span> — the same entry point <span style="font-family:${MONO}">perform_experiments.py</span> uses, so the dashboard cannot drift from the CLI.</div>
    </div>

    <div style="display:flex;gap:8px;background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:12px;padding:6px;margin-bottom:14px">
      ${tab('manual', 'Manual', 'tune an attack field by field')}
      ${tab('existing', 'Existing config', 'run a saved .yaml')}
    </div>

    ${manualMode ? manualPanel() : existingBody}

    <div style="background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:12px;overflow:hidden;margin-top:12px">
      <div style="padding:18px 20px;display:flex;align-items:center;gap:14px">
        <div data-act="launch-firestore" style="display:flex;align-items:center;gap:9px;cursor:pointer;font-size:12px;color:var(--fd,#9aa6b2)">
          <div style="width:34px;height:19px;border-radius:11px;position:relative;transition:.15s;background:${f.useFirestore ? 'var(--ac,#36c08f)' : 'var(--gd,#222a34)'}">
            <div style="position:absolute;top:2px;left:${f.useFirestore ? '17px' : '2px'};width:15px;height:15px;border-radius:50%;background:#fff;transition:.15s"></div>
          </div>persist to Firestore
        </div>
        <div style="flex:1"></div>
        ${w.running
          ? stopSweepControls(w, false)
          : `<div data-act="${manualMode ? 'manual-start' : 'launch-start'}" data-hover style="font-size:12px;font-weight:600;cursor:pointer;padding:9px 18px;border-radius:9px;background:var(--ac,#36c08f);color:#0d1014;border:1px solid var(--ac,#36c08f)">Start sweep</div>`}
      </div>
      <div style="padding:13px 20px;background:var(--p2,#1b212a);border-top:1px solid var(--bd,#252c36);font-size:11.5px;font-family:${MONO}">${status}</div>
    </div>
  </div>`;
}

// ---------- settings view ----------
function settingsView() {
  if (!S.settings) return loadingView('settings');
  const st = S.settings;
  const pending = Object.keys(S.settingsEdits).length + S.settingsDeletes.length;

  const renderEntry = (entry) => {
    const key = entry.key;
    const deleted = S.settingsDeletes.includes(key);
    const edited = Object.prototype.hasOwnProperty.call(S.settingsEdits, key);
    const revealed = !!S.settingsRevealed[key];
    // Secrets show a shape, never a value, until explicitly revealed. An empty
    // box with a "unchanged" placeholder means "leave what's on disk alone".
    const shown = edited ? S.settingsEdits[key] : (entry.secret ? '' : entry.value);
    const placeholder = entry.secret && entry.set
      ? (revealed ? '' : `unchanged · ${entry.value}`)
      : (entry.set ? '' : (entry.placeholder || 'not set'));

    return `<div style="padding:15px 20px;border-bottom:1px solid var(--gd,#222a34);${deleted ? 'opacity:.45' : ''}">
      <div style="display:flex;align-items:center;gap:9px;margin-bottom:7px">
        <span style="font-family:${MONO};font-size:12px;font-weight:600;color:var(--fg,#e6ebf0)">${esc(key)}</span>
        <span style="font-size:9px;letter-spacing:.1em;text-transform:uppercase;color:${entry.known ? 'var(--ac,#36c08f)' : 'var(--fm,#5f6b78)'};border:1px solid ${entry.known ? 'color-mix(in srgb,var(--ac,#36c08f) 35%,var(--bd))' : 'var(--bd,#252c36)'};padding:1px 6px;border-radius:4px">${entry.known ? 'expected' : 'custom'}</span>
        ${entry.secret ? `<span style="font-size:9px;letter-spacing:.1em;text-transform:uppercase;color:var(--wn,#e3b341);border:1px solid color-mix(in srgb,var(--wn,#e3b341) 40%,var(--bd));padding:1px 6px;border-radius:4px">secret</span>` : ''}
        ${entry.set ? '' : `<span style="font-size:9px;letter-spacing:.1em;text-transform:uppercase;color:var(--fm,#5f6b78);border:1px solid var(--bd,#252c36);padding:1px 6px;border-radius:4px">not set</span>`}
        ${edited ? `<span style="font-size:10px;color:var(--ac,#36c08f);font-family:${MONO}">edited</span>` : ''}
        ${deleted ? `<span style="font-size:10px;color:var(--no,#f0606a);font-family:${MONO}">will be removed</span>` : ''}
        <div style="flex:1"></div>
        ${entry.secret && entry.set && !revealed ? `<div data-act="env-reveal" data-arg="${esc(key)}" style="font-size:10.5px;color:var(--fd,#9aa6b2);border:1px solid var(--bd,#252c36);border-radius:6px;padding:3px 10px;cursor:pointer">reveal</div>` : ''}
        ${entry.present ? `<div data-act="env-delete" data-arg="${esc(key)}" style="font-size:10.5px;color:${deleted ? 'var(--fd,#9aa6b2)' : 'var(--no,#f0606a)'};border:1px solid var(--bd,#252c36);border-radius:6px;padding:3px 10px;cursor:pointer">${deleted ? 'keep' : 'remove'}</div>` : ''}
      </div>
      ${entry.help ? `<div style="font-size:11px;color:var(--fm,#5f6b78);line-height:1.5;margin-bottom:8px">${esc(entry.help)}</div>` : ''}
      <input class="field" data-inp="env:${esc(key)}" value="${esc(shown)}" placeholder="${esc(placeholder)}" ${deleted ? 'disabled' : ''} style="${FIELD}">
    </div>`;
  };

  const grouped = [];
  st.entries.forEach((entry) => {
    const name = entry.category || 'Additional .env values';
    let group = grouped.find((item) => item.name === name);
    if (!group) { group = { name, entries: [] }; grouped.push(group); }
    group.entries.push(entry);
  });
  const rows = grouped.map((group) => {
    const configured = group.entries.filter((entry) => entry.present).length;
    return `<section>
      <div style="display:flex;align-items:center;gap:10px;padding:10px 20px;background:var(--p2,#1b212a);border-bottom:1px solid var(--bd,#252c36)">
        <span style="font-size:10px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--fd,#9aa6b2)">${esc(group.name)}</span>
        <div style="flex:1"></div>
        <span style="font-family:${MONO};font-size:9.5px;color:var(--fm,#5f6b78)">${configured}/${group.entries.length} in file</span>
      </div>
      ${group.entries.map(renderEntry).join('')}
    </section>`;
  }).join('');
  const expected = st.entries.filter((entry) => entry.known).length;
  const configured = st.entries.filter((entry) => entry.present).length;

  const warning = st.tunnel_live ? `<div style="display:grid;grid-template-columns:auto 1fr;align-items:center;gap:14px;background:color-mix(in srgb,var(--no,#f0606a) 8%,var(--pn,#15191f));border:1px solid color-mix(in srgb,var(--no,#f0606a) 35%,var(--bd));border-radius:10px;padding:14px 18px;margin-bottom:14px">
      <div style="width:30px;height:30px;border-radius:50%;border:2px solid var(--no,#f0606a);color:var(--no,#f0606a);display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;flex:0 0 auto">!</div>
      <div style="font-size:12px;color:var(--fd,#9aa6b2);line-height:1.6"><b style="color:var(--fg,#e6ebf0)">The tunnel is live while this page is open.</b> This page can read and write the service-account credentials, and it is currently reachable from the public internet. Stop the tunnel before revealing a secret on an untrusted network.</div>
    </div>` : '';

  return `<div style="max-width:820px;margin:0 auto">
    <div style="margin-bottom:20px">
      <div style="font-size:20px;font-weight:700;letter-spacing:-.01em">Settings</div>
      <div style="font-size:12px;color:var(--fd,#9aa6b2);margin-top:4px;line-height:1.6">Every environment variable CANARY expects is listed below, including variables missing from the current <span style="font-family:${MONO}">.env</span>. Saving reloads edits into the running process; the previous file is kept as <span style="font-family:${MONO}">.env.bak</span>.</div>
      <div style="font-size:11px;color:var(--fm,#5f6b78);margin-top:8px;font-family:${MONO}">${esc(st.path)}${st.exists ? '' : ' · will be created on save'}</div>
    </div>

    ${warning}

    <div style="background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:12px;overflow:hidden">
      <div style="display:flex;align-items:center;gap:10px;padding:13px 20px;border-bottom:1px solid var(--bd,#252c36)">
        <span style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--fd,#9aa6b2)">Environment catalog</span>
        <div style="flex:1"></div>
        <span style="font-family:${MONO};font-size:9.5px;color:var(--fm,#5f6b78)">${expected} expected · ${configured} present</span>
      </div>
      ${rows}
      <div style="padding:15px 20px;border-bottom:1px solid var(--bd,#252c36)">
        <div style="font-size:10px;color:var(--fm,#5f6b78);letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px">add a custom variable</div>
        <div style="display:grid;grid-template-columns:260px 1fr;gap:10px">
          <input class="field" data-inp="newVarKey" value="${esc(S.newVar.key)}" placeholder="KEY_NAME" style="${FIELD}">
          <input class="field" data-inp="newVarValue" value="${esc(S.newVar.value)}" placeholder="value" style="${FIELD}">
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:12px;padding:15px 20px;background:var(--p2,#1b212a)">
        <span style="font-size:11.5px;color:var(--fm,#5f6b78);font-family:${MONO}">${pending || S.newVar.key ? `${pending + (S.newVar.key ? 1 : 0)} pending change(s)` : 'no pending changes'}</span>
        <div style="flex:1"></div>
        ${pending || S.newVar.key ? `<div data-act="env-discard" style="font-size:11.5px;color:var(--fd,#9aa6b2);border:1px solid var(--bd,#252c36);border-radius:8px;padding:8px 14px;cursor:pointer">Discard</div>` : ''}
        <div data-act="env-save" data-hover style="font-size:12px;font-weight:600;cursor:pointer;padding:9px 18px;border-radius:9px;background:var(--ac,#36c08f);border:1px solid var(--ac,#36c08f);color:#0d1014">Save &amp; reload</div>
      </div>
    </div>
  </div>`;
}

function loadingView(what) {
  return `<div style="padding:60px 20px;text-align:center;color:var(--fm,#5f6b78);font-size:12.5px;font-family:${MONO}">loading ${esc(what)}…</div>`;
}

// ---------- charts ----------
function chartBase() {
  const fd = cvar('--fd', '#9aa6b2'), grid = cvar('--gd', '#222a34'), fm = cvar('--fm', '#5f6b78');
  return {
    fd, fm,
    base: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { labels: { color: fd, boxWidth: 10, boxHeight: 10, font: { size: 10 } } } },
      scales: {
        x: { grid: { color: grid }, ticks: { color: fm }, border: { color: grid } },
        y: { grid: { color: grid }, ticks: { color: fm }, border: { color: grid } },
      },
    },
  };
}

function cfgCompare() {
  const { base, fd } = chartBase();
  const xf = S.xFactor;
  const runs = filteredRuns().filter((r) => r.metrics && r.metrics.adv !== null && r.metrics.adv !== undefined);
  // epsilon = null means "no DP"; plot it past the largest finite epsilon so
  // the no-DP baseline is visible on the same axis instead of being dropped.
  const finite = runs.map((r) => r.config.epsilon).filter((e) => typeof e === 'number');
  const noDpX = finite.length ? Math.max(...finite) * 2 : 16;
  const attacks = S.results.attacks;
  const datasets = Object.keys(attacks).filter((k) => S.filters.attacks[k] !== false).map((k) => {
    const pts = runs.filter((r) => r.attack === k).map((r) => {
      let x = r.config[xf];
      if (xf === 'epsilon' && (x === null || x === undefined)) x = noDpX;
      return (typeof x === 'number') ? { x, y: Number(r.metrics.adv.toFixed(4)) } : null;
    }).filter(Boolean);
    return { label: attacks[k].label, data: pts, backgroundColor: attacks[k].color, borderColor: attacks[k].color, pointRadius: 4, pointHoverRadius: 6, showLine: false };
  }).filter((d) => d.data.length);

  const xTitle = xf === 'epsilon' ? `epsilon (${noDpX} = no DP)` : xf;
  return {
    sig: 'cmp' + xf + JSON.stringify(datasets.map((d) => [d.label, d.data.length])) + runs.length,
    spec: {
      type: 'scatter', data: { datasets },
      options: { ...base, scales: {
        x: { ...base.scales.x, title: { display: true, text: xTitle, color: fd } },
        y: { ...base.scales.y, min: 0.45, max: 1, title: { display: true, text: 'Adv', color: fd } },
      } },
    },
  };
}

function cfgAgg() {
  const { base } = chartBase();
  const runs = filteredRuns().filter((r) => r.metrics && r.metrics.adv !== null && r.metrics.adv !== undefined);
  const attacks = S.results.attacks;
  const labels = [], data = [], colors = [];
  Object.keys(attacks).sort().filter((k) => S.filters.attacks[k] !== false).forEach((k) => {
    const v = runs.filter((r) => r.attack === k).map((r) => r.metrics.adv);
    if (!v.length) return;
    labels.push(attacks[k].label); colors.push(attacks[k].color);
    data.push(Number((v.reduce((a, b) => a + b, 0) / v.length).toFixed(3)));
  });
  return {
    sig: 'agg' + JSON.stringify([labels, data]),
    spec: {
      type: 'bar', data: { labels, datasets: [{ data, backgroundColor: colors, borderRadius: 4, maxBarThickness: 38 }] },
      options: { ...base, plugins: { legend: { display: false } }, scales: { x: base.scales.x, y: { ...base.scales.y, min: 0.45, max: 1 } } },
    },
  };
}

function cfgFL() {
  const { base, fd } = chartBase();
  const d = S.detail, h = d.federated_history;
  const color = d.meta.color;
  return {
    sig: 'fl' + d.run_id + h.length,
    spec: {
      type: 'line',
      data: { labels: h.map((x) => 'r' + x.round), datasets: [{ label: 'mean client loss', data: h.map((x) => x.mean_loss), borderColor: color, backgroundColor: color + '22', tension: .3, fill: true, pointRadius: 4, borderWidth: 2 }] },
      options: { ...base, scales: { x: base.scales.x, y: { ...base.scales.y, title: { display: true, text: 'loss', color: fd } } } },
    },
  };
}

function cfgDist() {
  const { base, fd, fm } = chartBase();
  const d = S.detail, tr = d.trials, bins = 18;
  const scores = tr.map((t) => t.score);
  const lo = Math.min(...scores), hi = Math.max(...scores);
  const span = hi - lo || 1;
  const member = new Array(bins).fill(0), non = new Array(bins).fill(0);
  tr.forEach((t) => {
    const b = clamp(Math.floor(((t.score - lo) / span) * bins), 0, bins - 1);
    (t.truth_member ? member : non)[b]++;
  });
  const labels = Array.from({ length: bins }, (_, i) => (lo + ((i + 0.5) / bins) * span).toPrecision(3));
  const color = d.meta.color;
  return {
    sig: 'dist' + d.run_id + tr.length,
    spec: {
      type: 'bar',
      data: { labels, datasets: [{ label: 'member', data: member, backgroundColor: color + 'cc' }, { label: 'non-member', data: non, backgroundColor: fm + '99' }] },
      options: { ...base, scales: {
        x: { ...base.scales.x, title: { display: true, text: 'attack score', color: fd } },
        y: { ...base.scales.y, title: { display: true, text: 'trials', color: fd } },
      } },
    },
  };
}

function rocPoints(trials) {
  const arr = trials.map((t) => ({ s: t.score, p: t.truth_member })).sort((a, b) => b.s - a.s);
  const P = arr.filter((a) => a.p).length, N = arr.length - P;
  const pts = [{ x: 0, y: 0 }];
  let tp = 0, fp = 0;
  arr.forEach((a) => {
    if (a.p) tp++; else fp++;
    pts.push({ x: N ? fp / N : 0, y: P ? tp / P : 0 });
  });
  return pts;
}

function cfgROC() {
  const { base, fd, fm } = chartBase();
  const d = S.detail, color = d.meta.color;
  return {
    sig: 'roc' + d.run_id + d.trials.length,
    spec: {
      type: 'line',
      data: { datasets: [
        { label: 'ROC', data: rocPoints(d.trials), borderColor: color, backgroundColor: color + '22', fill: true, tension: .1, pointRadius: 0, borderWidth: 2 },
        { label: 'chance', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: fm, borderDash: [5, 4], pointRadius: 0, borderWidth: 1, fill: false },
      ] },
      options: { ...base, plugins: { legend: { display: false } }, scales: {
        x: { ...base.scales.x, type: 'linear', min: 0, max: 1, title: { display: true, text: 'FPR', color: fd } },
        y: { ...base.scales.y, min: 0, max: 1, title: { display: true, text: 'TPR', color: fd } },
      } },
    },
  };
}

function syncCharts() {
  if (!window.Chart || !root) return;
  Chart.defaults.font.family = "'IBM Plex Mono','IBM Plex Sans',monospace";
  Chart.defaults.font.size = 10;

  let need = [];
  if (S.view === 'results' && S.results) need = ['compare', 'agg'];
  else if (S.view === 'detail' && S.detail) {
    if (S.detail.federated_history.length) need.push('fl');
    if (S.detail.trials.length) need.push('dist', 'roc');
  }

  Object.keys(charts).forEach((id) => {
    const el = root.querySelector(`canvas[data-chart="${id}"]`);
    if (!need.includes(id) || !el || charts[id].canvas !== el) {
      try { charts[id].destroy(); } catch (e) { /* already gone */ }
      delete charts[id]; delete chartSig[id];
    }
  });

  const builders = { compare: cfgCompare, agg: cfgAgg, fl: cfgFL, dist: cfgDist, roc: cfgROC };
  need.forEach((id) => {
    const el = root.querySelector(`canvas[data-chart="${id}"]`);
    if (!el) return;
    const cfg = builders[id]();
    const sig = S.theme + '|' + cfg.sig;
    if (charts[id] && chartSig[id] === sig) return;
    if (charts[id]) { try { charts[id].destroy(); } catch (e) { /* already gone */ } }
    charts[id] = new Chart(el.getContext('2d'), cfg.spec);
    chartSig[id] = sig;
  });
}

// ---------- render ----------
function currentView() {
  switch (S.view) {
    case 'results': return resultsView();
    case 'detail': return detailView();
    case 'access': return accessView();
    case 'launch': return launchView();
    case 'settings': return settingsView();
    default: return liveView();
  }
}

function render(opts) {
  if (!root) return;
  // Preserve the caret across the full re-render: the whole page is rebuilt on
  // every state change, and typing in a filter is a state change.
  const active = document.activeElement;
  const focusKey = active && active.dataset ? active.dataset.inp : null;
  // A poll or clock tick must never rebuild the field being typed into --
  // caret restoration can't bring back undo history or a partial IME compose.
  if (opts && opts.background && focusKey) return;
  const caret = focusKey && active.selectionStart !== undefined ? active.selectionStart : null;
  const logPane = root.querySelector('#logpane');
  const logScrollTop = logPane ? logPane.scrollTop : null;
  const logAtBottom = !logPane ||
    logPane.scrollHeight - logPane.scrollTop - logPane.clientHeight <= LOG_BOTTOM_EPSILON_PX;
  // Scroll offsets survive the rebuild too, so a poll doesn't yank the table
  // back to the top while you're reading row 40.
  const scrolls = {};
  root.querySelectorAll('[data-scroll]').forEach((el) => { scrolls[el.dataset.scroll] = el.scrollTop; });

  root.setAttribute('style', `${THEMES[S.theme]};min-height:100vh;display:flex;flex-direction:column;background:var(--bg);color:var(--fg);font-family:'IBM Plex Sans',system-ui,sans-serif;font-size:13px;-webkit-font-smoothing:antialiased`);
  root.innerHTML = `${topBar()}${bannerBar()}<div style="flex:1;min-height:0;overflow-y:auto;padding:18px">${currentView()}</div>`;

  if (focusKey) {
    const el = root.querySelector(`[data-inp="${focusKey}"]`);
    if (el) {
      el.focus();
      if (caret !== null && el.setSelectionRange) { try { el.setSelectionRange(caret, caret); } catch (e) { /* not a text input */ } }
    }
  }
  root.querySelectorAll('[data-scroll]').forEach((el) => {
    const saved = scrolls[el.dataset.scroll];
    if (saved) el.scrollTop = saved;
  });
  const pane = root.querySelector('#logpane');
  if (pane) {
    if (logAtBottom) {
      // Initial render and an untouched live tail stay pinned to the newest
      // entry as lines arrive.
      pane.scrollTop = pane.scrollHeight;
    } else if (logScrollTop !== null) {
      // A deliberate upward scroll opts out of auto-follow. The pane is
      // recreated on every poll, so restore its exact previous viewport.
      pane.scrollTop = logScrollTop;
    }
  }
  syncCharts();
}

// ---------- events ----------
async function onAction(act, arg) {
  switch (act) {
    case 'nav': S.stopConfirm = false; S.deleteConfirmRunId = null; go(arg); return;
    case 'open': S.deleteConfirmRunId = null; go('detail', arg); return;
    case 'back': S.deleteConfirmRunId = null; go('results'); return;
    case 'theme': S.theme = arg; localStorage.setItem('canary.theme', arg); break;
    case 'viz': S.liveViz = arg; break;
    case 'privacy': S.privacyView = !S.privacyView; break;
    case 'attackchip': S.filters.attacks[arg] = S.filters.attacks[arg] === false; break;
    case 'sort':
      S.sort = { key: arg, dir: (S.sort.key === arg && S.sort.dir === 'desc') ? 'asc' : 'desc' };
      break;
    case 'sweep-stop-arm': S.stopConfirm = true; break;
    case 'sweep-stop-cancel': S.stopConfirm = false; break;
    case 'sweep-stop-confirm': {
      const res = await postJSON('/api/launch/stop', {});
      S.stopConfirm = false;
      if (S.view === 'launch') S.launch = await getJSON('/api/launch');
      if (S.view === 'live') {
        S.live = await getJSON('/api/live');
        S.serverOffset = S.live.now_unix - Date.now() / 1000;
      }
      S.banner = res.message;
      break;
    }
    case 'run-delete-arm': S.deleteConfirmRunId = arg; break;
    case 'run-delete-cancel':
      if (S.deleteConfirmRunId === arg) S.deleteConfirmRunId = null;
      break;
    case 'run-delete-confirm': {
      const res = await deleteJSON('/api/runs/' + encodeURIComponent(arg));
      S.deleteConfirmRunId = null;
      if (res.ok) {
        S.results = await getJSON('/api/results');
        initFilters();
        if (S.view === 'detail') {
          S.view = 'results';
          S.selectedRunId = null;
          S.detail = null;
          S.detailError = '';
          history.replaceState({}, '', '/results');
        }
      }
      S.banner = res.message;
      break;
    }
    case 'provider': S.tunnelForm.provider = arg; break;
    case 'copy':
      if (S.tunnel && S.tunnel.url) {
        try { await navigator.clipboard.writeText(S.tunnel.url); } catch (e) { /* clipboard blocked */ }
        S.copied = true;
        setTimeout(() => { S.copied = false; render(); }, 1200);
      }
      break;
    case 'tunnel-start': case 'tunnel-save': {
      const f = tunnelFields();
      // null for an untouched credential: the server keeps the saved one. A
      // typed empty string is a deliberate clear and travels as "".
      const res = await postJSON(act === 'tunnel-save' ? '/api/tunnel/save' : '/api/tunnel/start', {
        provider: f.provider,
        api_key: f.apiKeyEdited ? f.apiKey : null,
        code: f.codeEdited ? f.code : null,
        port: Number(f.port) || 8080,
      });
      S.tunnel = res.status;
      S.banner = res.ok ? '' : res.message;
      // Typed secrets go back to "saved" once the server has them; the field
      // must not keep holding a token the .env is now the home for.
      if (res.ok) S.tunnelForm = { ...S.tunnelForm, apiKey: null, code: null };
      if (res.ok && act === 'tunnel-save') S.banner = res.message;
      break;
    }
    case 'tunnel-stop': {
      const res = await postJSON('/api/tunnel/stop', {});
      S.tunnel = res.status;
      break;
    }
    case 'launch-attack': {
      const list = S.launchForm.attacks;
      const i = list.indexOf(arg);
      if (i >= 0) list.splice(i, 1); else list.push(arg);
      break;
    }
    case 'launch-firestore': S.launchForm.useFirestore = !S.launchForm.useFirestore; break;
    case 'launch-mode': S.launchForm.mode = arg; break;

    case 'manual-remove':
      S.manual.cards = S.manual.cards.filter((c) => c.name !== arg);
      S.manual.dirty = true;
      break;
    case 'manual-advanced': {
      const card = S.manual.cards.find((c) => c.name === arg);
      if (card) card.advanced = !card.advanced;
      break;
    }
    case 'manual-bool': {
      const [name, field] = arg.split(':');
      const card = S.manual.cards.find((c) => c.name === name);
      if (!card) break;
      const info = S.attackFields.attacks[name].fields.find((f) => f.name === field);
      card.values[field] = manualValue(card, info) === 'true' ? 'false' : 'true';
      S.manual.dirty = true;
      break;
    }
    case 'manual-sweep': {
      const [name, field] = arg.split(':');
      const card = S.manual.cards.find((c) => c.name === name);
      if (!card) break;
      const info = S.attackFields.attacks[name].fields.find((f) => f.name === field);
      const i = card.sweeps.indexOf(field);
      if (i >= 0) {
        // Leaving sweep mode keeps the first value, not the whole list.
        card.sweeps.splice(i, 1);
        const first = String(manualValue(card, info)).split(',')[0].trim();
        if (first) card.values[field] = first;
      } else {
        card.sweeps.push(field);
        card.values[field] = manualValue(card, info);
      }
      S.manual.dirty = true;
      break;
    }
    case 'manual-validate':
      S.manual.validation = await postJSON('/api/launch/manual/validate', manualBody());
      S.manual.dirty = false;
      break;
    case 'manual-save': {
      const mn = S.manual;
      if (!mn.saveName.trim()) { S.banner = 'Give the config a filename before saving.'; break; }
      // The YAML comes from the server's own build_doc, so the file saved is
      // the config the manual run would have expanded — not a second rendering
      // of the form that could disagree with it.
      const check = await postJSON('/api/launch/manual/validate', manualBody());
      mn.validation = check;
      mn.dirty = false;
      if (!check.ok) { S.banner = check.message; break; }
      const res = await postJSON('/api/configs/save', {
        name: mn.saveName, text: check.yaml, overwrite: mn.confirmOverwrite,
      });
      S.banner = res.message;
      if (res.ok) {
        mn.confirmOverwrite = false;
        mn.saveName = res.name;
        S.launch = await getJSON('/api/launch');
      } else {
        // Re-clicking Save after this confirms the overwrite.
        mn.confirmOverwrite = !!res.exists;
      }
      break;
    }
    case 'manual-start': {
      const res = await postJSON('/api/launch/manual', {
        ...manualBody(), use_firestore: S.launchForm.useFirestore,
      });
      S.banner = res.message;
      S.launch = await getJSON('/api/launch');
      break;
    }

    case 'config-new':
      S.editor = { name: null, text: S.launch.template || '', dirty: true, validation: null,
                   saveName: '', confirmOverwrite: false };
      break;
    case 'config-validate':
      S.editor.validation = await postJSON('/api/configs/validate', { text: S.editor.text });
      break;
    case 'config-save': {
      const e = S.editor;
      if (!e.saveName.trim()) { S.banner = 'Give the config a filename before saving.'; break; }
      const res = await postJSON('/api/configs/save', {
        name: e.saveName, text: e.text, overwrite: e.confirmOverwrite,
      });
      S.banner = res.message;
      if (res.ok) {
        S.editor = { name: res.name, text: e.text, dirty: false, saveName: res.name,
                     confirmOverwrite: false,
                     validation: { ok: true, message: res.message, per_attack: res.per_attack } };
        S.launch = await getJSON('/api/launch');
        S.launchForm.configFile = res.name;
      } else {
        // Re-clicking Save after this confirms the overwrite.
        S.editor.confirmOverwrite = !!res.exists;
      }
      break;
    }

    case 'env-reveal': {
      const res = await getJSON('/api/settings/reveal/' + encodeURIComponent(arg));
      S.settingsRevealed[arg] = true;
      S.settingsEdits[arg] = res.value;
      break;
    }
    case 'env-delete': {
      const i = S.settingsDeletes.indexOf(arg);
      if (i >= 0) S.settingsDeletes.splice(i, 1);
      else { S.settingsDeletes.push(arg); delete S.settingsEdits[arg]; }
      break;
    }
    case 'env-discard':
      S.settingsEdits = {}; S.settingsDeletes = []; S.settingsRevealed = {};
      S.newVar = { key: '', value: '' };
      break;
    case 'env-save': {
      const updates = { ...S.settingsEdits };
      if (S.newVar.key.trim()) updates[S.newVar.key.trim()] = S.newVar.value;
      const res = await postJSON('/api/settings', { updates, deletes: S.settingsDeletes });
      S.banner = res.message;
      if (res.ok) {
        S.settingsEdits = {}; S.settingsDeletes = []; S.settingsRevealed = {};
        S.newVar = { key: '', value: '' };
        S.settings = await getJSON('/api/settings');
      }
      break;
    }
    case 'launch-start': {
      const f = S.launchForm;
      const res = await postJSON('/api/launch', {
        config_file: f.configFile, attacks: f.attacks.length ? f.attacks : null, use_firestore: f.useFirestore,
      });
      S.banner = res.message;
      S.launch = await getJSON('/api/launch');
      break;
    }
    default: return;
  }
  render();
}

async function onInput(key, value) {
  if (key.startsWith('env:')) {
    S.settingsEdits[key.slice(4)] = value;
    return;  // typing in a field must not rebuild the field
  }
  if (key.startsWith('manual:')) {
    const [, name, field] = key.split(':');
    const card = S.manual.cards.find((c) => c.name === name);
    if (card) { card.values[field] = value; S.manual.dirty = true; }
    return;  // ditto
  }
  switch (key) {
    case 'status': case 'model': case 'mech': case 'q': S.filters[key] = value; break;
    case 'xfactor': S.xFactor = value; break;
    case 'apiKey': case 'code': case 'port': S.tunnelForm[key] = value; break;
    case 'configFile': S.launchForm.configFile = value; break;
    case 'configText':
      S.editor.text = value;
      S.editor.dirty = true;
      S.editor.validation = null;
      return;  // ditto: never re-render the textarea mid-keystroke
    case 'saveName':
      S.editor.saveName = value;
      S.editor.confirmOverwrite = false;
      return;
    case 'manualSaveName':
      S.manual.saveName = value;
      S.manual.confirmOverwrite = false;
      return;
    case 'manualAdd':
      if (!value || S.manual.cards.some((c) => c.name === value)) break;
      S.manual.cards.push({ name: value, values: {}, sweeps: [], advanced: false });
      S.manual.dirty = true;
      break;
    case 'newVarKey': S.newVar.key = value; return;
    case 'newVarValue': S.newVar.value = value; return;
    case 'editorFile':
      if (!value) { S.editor = { name: null, text: '', dirty: false, validation: null, saveName: '', confirmOverwrite: false }; break; }
      await loadConfig(value);
      S.launchForm.configFile = value;
      break;
    default: return;
  }
  render();
}

function attach() {
  root.addEventListener('click', (ev) => {
    const target = ev.target.closest('[data-act]');
    if (!target || !root.contains(target)) return;
    onAction(target.dataset.act, target.dataset.arg).catch((err) => {
      S.banner = err.message;
      render();
    });
  });
  const handler = (ev) => {
    const target = ev.target.closest('[data-inp]');
    if (!target) return;
    Promise.resolve(onInput(target.dataset.inp, target.value)).catch((err) => {
      S.banner = err.message;
      render();
    });
  };
  root.addEventListener('input', handler);
  root.addEventListener('change', handler);
  window.addEventListener('popstate', () => { readLocation(); render(); refresh(); });
}

// ---------- boot ----------
async function boot() {
  root = document.getElementById('root');
  // Before the first render: whether this browser may see Settings at all.
  // Guessing would either flash a tab that vanishes or hide one that belongs.
  try {
    S.local = !!(await getJSON('/api/session')).local;
  } catch (e) { S.local = false; }
  readLocation();
  attach();
  render();
  refresh();
  setInterval(refresh, POLL_MS);
  // Clock and elapsed counters advance between polls. Only the views that show
  // a moving number redraw on the tick; the rest wait for their poll.
  setInterval(() => {
    if (S.live) S.live.running.forEach((r) => {
      if (r.elapsed_seconds !== null && r.elapsed_seconds !== undefined) r.elapsed_seconds += TICK_MS / 1000;
    });
    if (S.view === 'live' || S.view === 'access' || S.view === 'launch') render({ background: true });
  }, TICK_MS);
}

boot();
