// ---- anonymous client id (keeps each visitor's history separate) ----
const CLIENT_ID = (() => {
  let id = localStorage.getItem("aa_client_id");
  if (!id) { id = crypto.randomUUID(); localStorage.setItem("aa_client_id", id); }
  return id;
})();

const $ = (id) => document.getElementById(id);
let lastRecord = null;       // serialized ApplicationRecord from /api/analyze
let lastAnalysis = null;     // full /api/analyze response (for the share card)
let lastTriage = null;       // full /api/triage response (for CSV / save-all)
let gaugeChart, stagesChart, outcomesChart;

const sevColor = (v) => (v >= 0.7 ? "#4ade80" : v >= 0.45 ? "#fbbf24" : "#f87171");

// ---------------------------------------------------------------- samples ----
async function loadSamples() {
  try {
    const res = await fetch("/api/samples");
    const samples = await res.json();
    const sel = $("sample-select");
    samples.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s.key;
      opt.textContent = `${s.role}  ·  (truth: ${s.truth})`;
      opt.dataset.truth = s.truth;
      sel.appendChild(opt);
    });
    window._samples = samples;
  } catch (e) { /* non-fatal */ }
}

async function loadSampleByKey(key, truth) {
  const res = await fetch(`/api/sample/${key}`);
  if (!res.ok) return;
  const data = await res.json();
  $("resume-text").value = data.resume_text;
  $("jd-text").value = data.jd_text;
  window._sampleTruth = truth || null;
}

// ---------------------------------------------------------------- uploads ----
async function extractFile(fileInput, targetTextarea) {
  const file = fileInput.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch("/api/extract", { method: "POST", body: fd });
  const data = await res.json();
  if (res.ok) targetTextarea.value = data.text;
  else showError(data.detail || "Could not read that file.");
}

// ---------------------------------------------------------------- analyze ----
function showError(msg) { const b = $("error-box"); b.textContent = msg; b.hidden = false; }
function clearError() { $("error-box").hidden = true; }

async function analyze() {
  clearError();
  const resume_text = $("resume-text").value.trim();
  const jd_text = $("jd-text").value.trim();
  if (!resume_text || !jd_text) { showError("Please provide both a résumé and a job description."); return; }

  $("loading").hidden = false;
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 60000);
    const res = await fetch("/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resume_text, jd_text }),
      signal: controller.signal,
    });
    clearTimeout(timer);
    const data = await res.json();
    if (!res.ok) { showError(data.detail || "Analysis failed."); return; }
    lastRecord = data.record;
    lastAnalysis = data;
    renderResults(data);
    renderReadyPanel(data);
    $("results").hidden = false;
    $("results").scrollIntoView({ behavior: "smooth" });
  } catch (e) {
    showError("Network error — please try again.");
  } finally {
    $("loading").hidden = true;
  }
}

function renderResults(d) {
  // gauge
  const pct = Math.round(d.overall * 100);
  $("gauge-num").style.color = sevColor(d.overall);
  countUp($("gauge-num"), pct, 850, "%");
  drawGauge(d.overall);

  // sample truth reveal
  const truthEl = $("sample-truth");
  if (window._sampleTruth) {
    truthEl.innerHTML = `Ground-truth outcome for this sample: <b>${window._sampleTruth}</b>`;
    truthEl.hidden = false;
  } else truthEl.hidden = true;

  // subscores
  const box = $("subscores"); box.innerHTML = "";
  d.subscores.forEach((s) => {
    const name = s.name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
    const div = document.createElement("div");
    div.className = "sub";
    div.innerHTML =
      `<div class="sub-top"><span>${name}</span><span style="color:${sevColor(s.score)}">${Math.round(s.score*100)}%</span></div>
       <div class="sub-bar"><div class="sub-fill" style="width:${s.score*100}%;background:${sevColor(s.score)}"></div></div>
       <div class="sub-rationale">${escapeHtml(s.rationale)}</div>`;
    box.appendChild(div);
  });

  // skills
  renderChips("matched-chips", d.matched_skills, "chip-match");
  renderChips("missing-chips", d.missing_skills, "chip-miss");
  $("matched-count").textContent = `(${d.matched_skills.length})`;
  $("missing-count").textContent = `(${d.missing_skills.length})`;

  // diagnosis
  const diag = d.diagnosis;
  const badge = $("stage-badge");
  badge.textContent = d.stage_label;
  badge.style.background = sevColor(d.overall);
  badge.style.color = "#07120b";
  $("diag-headline").textContent = diag.headline;
  $("diag-explanation").textContent = diag.explanation;
  const fixes = $("diag-fixes"); fixes.innerHTML = "";
  (diag.top_fixes || []).forEach((f) => { const li = document.createElement("li"); li.textContent = f; fixes.appendChild(li); });
  $("log-status").textContent = "";

  // reach-out: show contacts found in the posting, reset the outreach draft
  renderContacts(d.contacts || {});
  $("outreach-box").hidden = true;
  const ob = $("btn-outreach");
  ob.disabled = false;
  ob.textContent = "✨ Generate outreach message";
}

function renderContacts(c) {
  const box = $("contacts-box");
  box.innerHTML = "";
  const emails = c.emails || [], phones = c.phones || [], links = c.application_links || [];
  if (!emails.length && !phones.length && !links.length) {
    box.innerHTML = "<div class='no-contacts'>No contact details found in this posting — use the outreach draft below to reach out via LinkedIn or the company site.</div>";
    return;
  }
  const line = (label, html) => `<div class="contact-line"><span class="label">${label}</span>${html}</div>`;
  if (emails.length) box.insertAdjacentHTML("beforeend", line("Email", emails.map((e) => `<a href="mailto:${encodeURIComponent(e)}">${escapeHtml(e)}</a>`).join(", ")));
  if (phones.length) box.insertAdjacentHTML("beforeend", line("Phone", phones.map(escapeHtml).join(", ")));
  if (links.length) box.insertAdjacentHTML("beforeend", line("Apply", links.map((u) => `<a href="${escapeHtml(u)}" target="_blank" rel="noopener">${escapeHtml(u)}</a>`).join("<br>")));
}

async function generateOutreach() {
  if (!lastRecord) return;
  const btn = $("btn-outreach");
  btn.disabled = true; btn.textContent = "Generating…";
  try {
    const res = await fetch("/api/outreach", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ record: lastRecord }),
    });
    const d = await res.json();
    if (!res.ok) { showError(d.detail || "Could not generate outreach."); btn.disabled = false; btn.textContent = "✨ Generate outreach message"; return; }
    $("out-subject").textContent = d.email_subject;
    $("out-body").textContent = d.email_body;
    $("out-note").textContent = d.linkedin_note;
    $("outreach-box").hidden = false;
    btn.textContent = "🔄 Regenerate";
    btn.disabled = false;
  } catch (e) {
    showError("Network error generating outreach.");
    btn.disabled = false; btn.textContent = "✨ Generate outreach message";
  }
}

function renderChips(id, items, cls) {
  const el = $(id); el.innerHTML = "";
  if (!items || !items.length) { el.innerHTML = "<span class='sub-rationale'>None</span>"; return; }
  items.forEach((s) => { const span = document.createElement("span"); span.className = "chip " + cls; span.textContent = s; el.appendChild(span); });
}

function drawGauge(value) {
  const ctx = $("gauge").getContext("2d");
  if (gaugeChart) gaugeChart.destroy();
  gaugeChart = new Chart(ctx, {
    type: "doughnut",
    data: { datasets: [{ data: [value * 100, 100 - value * 100], backgroundColor: [sevColor(value), "rgba(255,255,255,0.06)"], borderWidth: 0 }] },
    options: { cutout: "76%", plugins: { legend: { display: false }, tooltip: { enabled: false } }, animation: { animateRotate: true } },
  });
}

// ------------------------------------------------------------------- log -----
async function logApplication() {
  if (!lastRecord) return;
  const outcome = $("log-outcome").value;
  const res = await fetch("/api/log", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ client_id: CLIENT_ID, record: lastRecord, outcome: outcome === "unknown" ? null : outcome }),
  });
  if (res.ok) { $("log-status").textContent = "Saved ✓"; loadPatterns(); }
  else { const d = await res.json(); $("log-status").textContent = d.detail || "Save failed"; }
}

// --------------------------------------------------------------- patterns ----
async function loadPatterns() {
  const res = await fetch(`/api/patterns?client_id=${encodeURIComponent(CLIENT_ID)}`);
  const data = await res.json();
  const rep = data.report;
  if (!rep || rep.total_applications === 0) {
    $("patterns-empty").hidden = false; $("patterns-body").hidden = true; return;
  }
  $("patterns-empty").hidden = true; $("patterns-body").hidden = false;
  $("m-total").textContent = rep.total_applications;
  $("m-fit").textContent = Math.round(rep.avg_overall_fit * 100) + "%";
  $("m-stage").textContent = rep.dominant_stage_label;
  $("insight").innerHTML = `💡 <b>Insight:</b> ${escapeHtml(rep.insight)}`;

  // charts
  const stageCounts = {}, outcomeCounts = {};
  data.history.forEach((h) => {
    stageCounts[h.stage] = (stageCounts[h.stage] || 0) + 1;
    outcomeCounts[h.outcome] = (outcomeCounts[h.outcome] || 0) + 1;
  });
  drawBar("chart-stages", stageCounts, "#22d3ee", (c) => (stagesChart = c), stagesChart);
  drawBar("chart-outcomes", outcomeCounts, "#4ade80", (c) => (outcomesChart = c), outcomesChart);

  // history table
  const tbody = document.querySelector("#history-table tbody"); tbody.innerHTML = "";
  data.history.forEach((h) => {
    const tr = document.createElement("tr");
    const opts = ["unknown", "rejected", "interview", "ghosted", "offer"]
      .map((o) => `<option ${o === h.outcome ? "selected" : ""}>${o}</option>`).join("");
    tr.innerHTML = `<td>${escapeHtml(h.role)}</td><td>${escapeHtml(h.company)}</td><td>${h.fit}</td><td>${escapeHtml(h.stage)}</td>
      <td><select data-id="${h.id}" class="outcome-edit">${opts}</select></td><td>${h.date}</td>`;
    tbody.appendChild(tr);
  });
  document.querySelectorAll(".outcome-edit").forEach((sel) => {
    sel.addEventListener("change", async (e) => {
      const val = e.target.value;
      if (val === "unknown") return;
      await fetch("/api/outcome", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: e.target.dataset.id, outcome: val }) });
      loadPatterns();
    });
  });
}

function drawBar(canvasId, counts, color, setRef, existing) {
  const ctx = $(canvasId).getContext("2d");
  if (existing) existing.destroy();
  const chart = new Chart(ctx, {
    type: "bar",
    data: { labels: Object.keys(counts), datasets: [{ data: Object.values(counts), backgroundColor: color, borderRadius: 6 }] },
    options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { ticks: { precision: 0, color: "#97a0b3" }, grid: { color: "rgba(255,255,255,0.05)" } }, y: { ticks: { color: "#97a0b3" }, grid: { display: false } } } },
  });
  setRef(chart);
}

function escapeHtml(s) { const d = document.createElement("div"); d.textContent = s == null ? "" : s; return d.innerHTML; }

// ------------------------------------------------------------------ wire ------
$("btn-analyze").addEventListener("click", analyze);
$("btn-log").addEventListener("click", logApplication);
$("btn-outreach").addEventListener("click", generateOutreach);
$("btn-refresh").addEventListener("click", loadPatterns);

// copy buttons (delegated)
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".copy-btn");
  if (!btn) return;
  const text = $(btn.dataset.target).textContent;
  navigator.clipboard.writeText(text).then(() => {
    const orig = btn.textContent; btn.textContent = "Copied ✓";
    setTimeout(() => { btn.textContent = orig; }, 1500);
  });
});
$("resume-file").addEventListener("change", (e) => extractFile(e.target, $("resume-text")));
$("jd-file").addEventListener("change", (e) => extractFile(e.target, $("jd-text")));
$("sample-select").addEventListener("change", (e) => {
  const opt = e.target.selectedOptions[0];
  if (e.target.value) loadSampleByKey(e.target.value, opt.dataset.truth);
  else window._sampleTruth = null;
});
$("btn-try-sample").addEventListener("click", async () => {
  if (!window._samples || !window._samples.length) await loadSamples();
  const s = window._samples[Math.floor(Math.random() * window._samples.length)];
  $("sample-select").value = s.key;
  await loadSampleByKey(s.key, s.truth);
  document.getElementById("analyze").scrollIntoView({ behavior: "smooth" });
});

loadSamples();
loadPatterns();

// ----------------------------------------------------------------- triage ----
const VERDICT_RANK = { apply_hard: 3, worth_a_shot: 2, skip: 1, error: 0 };

function addJdRow(focus) {
  const wrap = $("t-jds");
  const row = document.createElement("div");
  row.className = "card input-card t-jd-row";
  row.innerHTML =
    `<div class="input-head">
       <h3>💼 Job <span class="t-jd-n"></span></h3>
       <button class="btn btn-ghost btn-sm t-jd-remove" title="Remove this job">✕ Remove</button>
     </div>
     <textarea class="t-jd-text" placeholder="Paste a job description…"></textarea>`;
  wrap.appendChild(row);
  renumberJds();
  if (focus) row.querySelector(".t-jd-text").focus();
}

function renumberJds() {
  document.querySelectorAll("#t-jds .t-jd-row").forEach((row, i) => {
    row.querySelector(".t-jd-n").textContent = i + 1;
  });
}

function tShowError(msg) { const b = $("t-error"); b.textContent = msg; b.hidden = false; }
function tClearError() { $("t-error").hidden = true; }

async function triage() {
  tClearError();
  const resume_text = $("t-resume-text").value.trim();
  if (!resume_text) { tShowError("Please paste your résumé first."); return; }
  const jds = [...document.querySelectorAll("#t-jds .t-jd-text")]
    .map((t) => t.value.trim())
    .filter(Boolean)
    .map((text) => ({ text }));
  if (!jds.length) { tShowError("Add at least one job description."); return; }

  $("loading").hidden = false;
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 120000);
    const res = await fetch("/api/triage", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resume_text, jds }),
      signal: controller.signal,
    });
    clearTimeout(timer);
    const data = await res.json();
    if (!res.ok) { tShowError(data.detail || "Triage failed."); return; }
    lastTriage = data;
    renderTriage(data);
    $("t-results").hidden = false;
    $("t-results").scrollIntoView({ behavior: "smooth" });
  } catch (e) {
    tShowError("Network error — please try again.");
  } finally {
    $("loading").hidden = true;
  }
}

function renderTriage(data) {
  const s = data.summary;
  let summary = `<div class="t-summary-counts">
      <span class="t-pill v-apply_hard">🟢 Apply hard · <b class="t-count" data-to="${s.apply_hard}">0</b></span>
      <span class="t-pill v-worth_a_shot">🟡 Worth a shot · <b class="t-count" data-to="${s.worth_a_shot}">0</b></span>
      <span class="t-pill v-skip">🔴 Skip · <b class="t-count" data-to="${s.skip}">0</b></span>
    </div>`;
  if (s.best) summary += `<p class="t-summary-line">Your strongest match: <b>${escapeHtml(s.best)}</b>. Put your energy there.</p>`;
  if (s.dominant_blocker) summary += `<p class="t-summary-line">Across the jobs you should skip, your most common blocker is <b>${escapeHtml(s.dominant_blocker)}</b> — fix that and more roles open up.</p>`;
  if (s.rate_limited) summary += `<p class="t-summary-line warn">⚠️ Hit a rate limit partway through — some jobs weren't scored. Wait a minute and re-run the rest.</p>`;
  $("t-summary").innerHTML = summary;
  $("t-summary").querySelectorAll(".t-count").forEach((el) => countUp(el, +el.dataset.to));

  const tbody = document.querySelector("#t-table tbody");
  tbody.innerHTML = "";
  data.results.forEach((r, i) => {
    const tr = document.createElement("tr");
    tr.style.animationDelay = (i * 70) + "ms";   // staggered reveal
    if (r.error) {
      tr.innerHTML = `<td>${i + 1}</td><td colspan="6" class="t-err">⚠️ ${escapeHtml(r.label)} — could not score (${escapeHtml(r.error)})</td>`;
    } else {
      if (i === 0) tr.className = "top-row";       // highlight the best match
      const pct = Math.round(r.overall * 100);
      tr.innerHTML =
        `<td>${i + 1}</td>
         <td>${escapeHtml(r.jd_title)}</td>
         <td>${escapeHtml(r.jd_company)}</td>
         <td style="color:${sevColor(r.overall)};font-weight:700">${pct}%</td>
         <td><span class="t-pill v-${r.verdict}">${escapeHtml(r.verdict_label)}</span></td>
         <td>${escapeHtml(r.stage_label)}</td>
         <td class="t-fix">${escapeHtml(r.top_fix)}</td>`;
    }
    tbody.appendChild(tr);
  });
}

// count a number up from 0 -> target (requestAnimationFrame, ~0.65s)
function countUp(el, to, ms = 650, suffix = "") {
  if (!to) { el.textContent = "0" + suffix; return; }
  let startTs = null;
  function step(ts) {
    if (startTs === null) startTs = ts;
    const p = Math.min(1, (ts - startTs) / ms);
    el.textContent = Math.round(p * to) + suffix;
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// seed two empty JD rows, wire controls
addJdRow();
addJdRow();
$("t-add-jd").addEventListener("click", () => addJdRow(true));
$("t-analyze").addEventListener("click", triage);
$("t-resume-file").addEventListener("change", (e) => extractFile(e.target, $("t-resume-text")));
$("t-jds").addEventListener("click", (e) => {
  const btn = e.target.closest(".t-jd-remove");
  if (!btn) return;
  const rows = document.querySelectorAll("#t-jds .t-jd-row");
  if (rows.length <= 1) { btn.closest(".t-jd-row").querySelector(".t-jd-text").value = ""; return; }
  btn.closest(".t-jd-row").remove();
  renumberJds();
});

// ------------------------------------------------------------- mode toggle ----
function setMode(mode) {
  const single = mode === "single";
  document.querySelectorAll(".mode-btn").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  $("mode-toggle").classList.toggle("batch", !single);
  $("panel-single").hidden = !single;
  $("panel-batch").hidden = single;
  // carry the résumé across both modes so you only paste it once
  const from = single ? $("t-resume-text") : $("resume-text");
  const to = single ? $("resume-text") : $("t-resume-text");
  if (from.value.trim() && !to.value.trim()) to.value = from.value;
  // clear stale results from the mode we're leaving
  $("results").hidden = true;
  $("t-results").hidden = true;
}
document.querySelectorAll(".mode-btn").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
$("nav-triage").addEventListener("click", () => setMode("batch"));

// --------------------------------------------------------- share as image ----
async function shareCard() {
  if (!lastAnalysis) return;
  if (typeof html2canvas === "undefined") { showError("Image library didn't load — check your connection."); return; }
  const d = lastAnalysis;
  $("sc-role").textContent = d.jd_title || "This role";
  $("sc-company").textContent = d.jd_company ? "@ " + d.jd_company : "";
  $("sc-fit").textContent = Math.round(d.overall * 100) + "%";
  $("sc-fit").style.color = sevColor(d.overall);
  $("sc-stage").textContent = d.stage_label || "";
  $("sc-headline").textContent = (d.diagnosis && d.diagnosis.headline) || "";

  const btn = $("btn-share"); const orig = btn.textContent;
  btn.disabled = true; btn.textContent = "Rendering…";
  try {
    const canvas = await html2canvas($("share-card"), { backgroundColor: null, scale: 2 });
    const link = document.createElement("a");
    link.download = "hirelens-result.png";
    link.href = canvas.toDataURL("image/png");
    link.click();
  } catch (e) {
    showError("Could not generate the image.");
  } finally {
    btn.disabled = false; btn.textContent = orig;
  }
}
$("btn-share").addEventListener("click", shareCard);

// ----------------------------------------------------- triage save / export ----
async function saveAllTriage() {
  if (!lastTriage) return;
  const toSave = lastTriage.results.filter((r) => r.record);
  if (!toSave.length) return;
  const btn = $("t-save-all"); const orig = btn.textContent;
  btn.disabled = true; btn.textContent = "Saving…";
  let n = 0;
  for (const r of toSave) {
    try {
      const res = await fetch("/api/log", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ client_id: CLIENT_ID, record: r.record, outcome: null }),
      });
      if (res.ok) n++;
    } catch (e) { /* skip this one */ }
  }
  btn.textContent = `Saved ${n} ✓`;
  loadPatterns();
  setTimeout(() => { btn.textContent = orig; btn.disabled = false; }, 2200);
}
$("t-save-all").addEventListener("click", saveAllTriage);

function exportTriageCsv() {
  if (!lastTriage) return;
  const rows = [["Rank", "Role", "Company", "Fit %", "Verdict", "Kill-stage", "Top fix"]];
  lastTriage.results.forEach((r, i) => {
    if (r.error) { rows.push([i + 1, r.label, "", "", "Error", "", r.error]); return; }
    rows.push([i + 1, r.jd_title, r.jd_company, Math.round(r.overall * 100), r.verdict_label, r.stage_label, r.top_fix]);
  });
  const csv = rows.map((row) => row.map((c) => `"${String(c == null ? "" : c).replace(/"/g, '""')}"`).join(",")).join("\r\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "hirelens-triage.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}
$("t-export-csv").addEventListener("click", exportTriageCsv);

// ---------------------------------------------------------------- ready panel ----
const RP_ICONS = {
  keywords: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>`,
  trending: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>`,
  briefcase: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 7V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v2"/><line x1="12" y1="12" x2="12" y2="12"/></svg>`,
  shield: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg>`,
};

function renderReadyPanel(d) {
  const current = Math.round(d.overall * 100);
  const subMap = {};
  (d.subscores || []).forEach((s) => { subMap[s.name] = s.score; });

  // Estimate projected score boost from gaps
  const kGap = Math.max(0, 0.82 - (subMap.keywords_ats || 0));
  const sGap = Math.max(0, 0.82 - (subMap.skills || 0));
  const dGap = Math.max(0, 0.82 - (subMap.domain || 0));
  const snGap = Math.max(0, 0.82 - (subMap.seniority || 0));
  const rawBoost = Math.round(kGap * 28 + sGap * 18 + dGap * 14 + snGap * 10);
  const boost = Math.max(8, Math.min(32, rawBoost || 14));
  const lo = Math.min(97, current + boost - 2);
  const hi = Math.min(97, current + boost + 2);

  $("rp-from").textContent = current;
  $("rp-lo").textContent = lo;
  $("rp-hi").textContent = hi;
  $("rp-delta").textContent = `+${boost - 2}–${boost + 2} pts`;

  // Build action cards from subscores
  const cards = [];
  const kScore = Math.round((subMap.keywords_ats || 0) * 100);
  const sScore = Math.round((subMap.skills || 0) * 100);
  const snScore = Math.round((subMap.seniority || 0) * 100);
  const dScore = Math.round((subMap.domain || 0) * 100);

  if (kScore < 85) {
    cards.push({ icon: RP_ICONS.keywords, title: "Inject missing keywords", urgent: false,
      desc: `Keyword coverage is ${kScore}/100 — will embed all required tech terms contextually` });
  }
  if (sScore < 85 || snScore < 75) {
    cards.push({ icon: RP_ICONS.trending, title: "Reframe experience depth", urgent: false,
      desc: `Experience alignment is ${Math.min(sScore, snScore)}/100 — will add scale, impact metrics and ownership signals` });
  }
  if (dScore < 85) {
    cards.push({ icon: RP_ICONS.briefcase, title: "Weave in domain context", urgent: dScore < 55,
      desc: `Domain relevance is ${dScore}/100 — will embed industry-specific language throughout` });
  }
  // Always show the "preserve" card
  cards.push({ icon: RP_ICONS.shield, title: "Preserve dates, company names & links", urgent: false,
    desc: "Timeline integrity and hyperlinks are never altered" });

  const box = $("rp-cards");
  box.innerHTML = "";
  cards.forEach((c) => {
    const div = document.createElement("div");
    div.className = "rp-card" + (c.urgent ? " urgent" : "");
    div.innerHTML =
      `<div class="rp-card-icon">${c.icon}</div>
       <div class="rp-card-body"><h4>${escapeHtml(c.title)}</h4><p>${escapeHtml(c.desc)}</p></div>`;
    box.appendChild(div);
  });

  $("ready-panel").hidden = false;
  $("opt-result").hidden = true;
  $("optimize-error").hidden = true;
}

// ---------------------------------------------------------------- optimize ----
async function optimize() {
  if (!lastAnalysis) return;
  const btn = $("btn-optimize");
  const origText = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<svg class="spin-svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg> Optimizing…`;
  $("optimize-error").hidden = true;
  $("opt-result").hidden = true;

  const d = lastAnalysis;
  const missing = (d.missing_skills || []).slice(0, 14);
  const fixes = (d.diagnosis && d.diagnosis.top_fixes || []).slice(0, 5);

  try {
    const res = await fetch("/api/optimize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        resume_text: $("resume-text").value.trim(),
        jd_text: $("jd-text").value.trim(),
        missing_skills: missing,
        top_fixes: fixes,
      }),
    });
    const data = await res.json();
    if (!res.ok) { showOptError(data.detail || "Optimization failed."); return; }

    $("opt-body").textContent = data.optimized_resume;
    $("opt-result").hidden = false;
    $("compare-panel").hidden = true;
    $("opt-result").scrollIntoView({ behavior: "smooth" });
    btn.innerHTML = `🔄 Re-optimize`;

    // Auto-trigger before/after rescore
    rescoreAndCompare(data.optimized_resume);
  } catch (e) {
    showOptError("Network error — please try again.");
  } finally {
    btn.disabled = false;
    if (btn.innerHTML.includes("Optimizing")) btn.innerHTML = origText;
  }
}

// ─────────────────────────────── before / after rescore ──────────────────────
async function rescoreAndCompare(optimizedText) {
  if (!lastAnalysis) return;
  const jd_text = $("jd-text").value.trim();
  if (!optimizedText || !jd_text) return;

  const panel = $("compare-panel");
  panel.hidden = false;
  $("compare-rows").innerHTML = `<div style="color:var(--muted);font-size:.88rem;padding:12px 0">⏳ Re-scoring optimized résumé…</div>`;

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resume_text: optimizedText, jd_text }),
    });
    if (!res.ok) return;
    const data = await res.json();
    renderComparison(lastAnalysis, data);
  } catch (_) { panel.hidden = true; }
}

function renderComparison(before, after) {
  const bPct = Math.round(before.overall * 100);
  const aPct = Math.round(after.overall * 100);
  const delta = aPct - bPct;

  $("cmp-before").textContent = bPct + "%";
  $("cmp-after").textContent  = aPct + "%";
  const dpill = $("cmp-delta");
  dpill.textContent = (delta >= 0 ? "+" : "") + delta + " pts";
  dpill.style.background = delta >= 5 ? "rgba(74,222,128,0.15)" : delta >= 0 ? "rgba(251,191,36,0.12)" : "rgba(248,113,113,0.12)";
  dpill.style.color = delta >= 5 ? "var(--brand)" : delta >= 0 ? "var(--warn)" : "var(--danger)";
  dpill.style.borderColor = delta >= 5 ? "rgba(74,222,128,0.3)" : delta >= 0 ? "rgba(251,191,36,0.3)" : "rgba(248,113,113,0.3)";

  const rows = [
    { label: "Overall Fit", b: bPct, a: aPct },
    ...(before.subscores || []).map((s) => {
      const aS = (after.subscores || []).find((x) => x.name === s.name);
      return {
        label: s.name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
        b: Math.round(s.score * 100),
        a: aS ? Math.round(aS.score * 100) : Math.round(s.score * 100),
      };
    }),
  ];

  const rowsEl = $("compare-rows");
  rowsEl.innerHTML = rows.map((r) => {
    const d = r.a - r.b;
    const col = d > 0 ? "var(--brand)" : d < 0 ? "var(--danger)" : "var(--muted)";
    return `
      <div class="compare-row">
        <span class="compare-label">${escapeHtml(r.label)}</span>
        <div class="compare-bar-group">
          <div class="compare-bar-track"><div class="compare-bar-fill before" style="width:0%" data-w="${r.b}"></div></div>
          <div class="compare-bar-track" style="margin-top:4px"><div class="compare-bar-fill after" style="width:0%" data-w="${r.a}"></div></div>
          <div class="compare-pcts"><span>${r.b}% before</span><span style="color:${col}">${r.a}% after</span></div>
        </div>
        <span class="compare-delta" style="color:${col}">${d >= 0 ? "+" : ""}${d}pts</span>
      </div>`;
  }).join("");

  // Animate bars in
  requestAnimationFrame(() => {
    rowsEl.querySelectorAll(".compare-bar-fill").forEach((el) => {
      el.style.width = el.dataset.w + "%";
    });
  });
}

// ─────────────────────────────── cover letter ─────────────────────────────────
async function generateCoverLetter() {
  if (!lastAnalysis) return;
  const panel = $("cl-panel");
  const body  = $("cl-body");
  const spin  = $("cl-spinner");
  const btn   = $("btn-coverletter");

  panel.hidden = false;
  panel.scrollIntoView({ behavior: "smooth", block: "start" });
  spin.hidden = false;
  body.textContent = "";
  btn.disabled = true;
  const orig = btn.innerHTML;
  btn.innerHTML = `<svg class="spin-svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg> Writing…`;

  try {
    const res = await fetch("/api/cover-letter", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        resume_text: $("resume-text").value.trim(),
        jd_text:     $("jd-text").value.trim(),
        matched_skills: lastAnalysis.matched_skills || [],
        missing_skills: lastAnalysis.missing_skills || [],
      }),
    });
    const data = await res.json();
    if (!res.ok) { body.textContent = "Error: " + (data.detail || "Generation failed."); return; }
    body.textContent = data.cover_letter;
  } catch (_) {
    body.textContent = "Network error — please try again.";
  } finally {
    spin.hidden = true;
    btn.disabled = false;
    btn.innerHTML = orig;
  }
}

// ─────────────────────────────── format checker ───────────────────────────────
async function checkFormat(file) {
  const panel   = $("fmt-panel");
  const spinner = $("fmt-spinner");
  const list    = $("fmt-list");
  const verdict = $("fmt-verdict");

  panel.hidden   = false;
  spinner.hidden = false;
  list.innerHTML = "";
  verdict.textContent = "—";
  verdict.className = "fmt-verdict";
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });

  const fd = new FormData();
  fd.append("file", file);

  try {
    const res  = await fetch("/api/check-format", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) {
      list.innerHTML = `<div class="fmt-item error"><span class="fmt-icon">❌</span><div class="fmt-msg">${escapeHtml(data.detail || "Could not analyse file.")}</div></div>`;
      return;
    }

    verdict.textContent = data.verdict_label;
    verdict.classList.add(data.verdict);

    const items = [
      ...data.issues.map((i) => ({ ...i, kind: i.severity })),
      ...data.passes.map((p) => ({ kind: "pass", message: p })),
    ];

    list.innerHTML = items.map((it) => {
      const icon = it.kind === "pass" ? "✅" : it.kind === "warning" ? "⚠️" : "❌";
      const cls  = it.kind === "pass" ? "pass" : it.kind === "warning" ? "warning" : "error";
      return `<div class="fmt-item ${cls}">
        <span class="fmt-icon">${icon}</span>
        <div class="fmt-msg">${escapeHtml(it.message)}</div>
      </div>`;
    }).join("");
  } catch (_) {
    list.innerHTML = `<div class="fmt-item error"><span class="fmt-icon">❌</span><div class="fmt-msg">Network error — please try again.</div></div>`;
  } finally {
    spinner.hidden = true;
  }
}

function showOptError(msg) {
  const b = $("optimize-error");
  b.textContent = msg;
  b.hidden = false;
}

$("btn-optimize").addEventListener("click", optimize);
$("btn-copy-opt").addEventListener("click", () => {
  const text = $("opt-body").textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = $("btn-copy-opt"); const orig = btn.textContent;
    btn.textContent = "Copied ✓";
    setTimeout(() => { btn.textContent = orig; }, 1600);
  });
});

// ---- cover letter button ----
$("btn-coverletter").addEventListener("click", generateCoverLetter);
$("btn-copy-cl").addEventListener("click", () => {
  const text = $("cl-body").textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = $("btn-copy-cl"); const orig = btn.textContent;
    btn.textContent = "Copied ✓";
    setTimeout(() => { btn.textContent = orig; }, 1600);
  });
});

// ---- ATS format checker ----
$("fmt-file").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  e.target.value = "";   // reset so same file can be re-checked
  checkFormat(file);
});

// ---- rotating headline word ----
(function rotateWord() {
  const words = ["rejected", "ghosted", "ignored", "filtered out", "passed over"];
  const el = document.querySelector(".rotator");
  if (!el) return;
  let i = 0;
  setInterval(() => {
    el.style.opacity = "0";
    el.style.transform = "translateY(10px)";
    setTimeout(() => {
      i = (i + 1) % words.length;
      el.textContent = words[i];
      el.style.opacity = "1";
      el.style.transform = "none";
    }, 300);
  }, 2600);
})();

// ---- scroll reveal ----
(function reveal() {
  const els = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) { els.forEach((e) => e.classList.add("in")); return; }
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
  }, { threshold: 0.12 });
  els.forEach((e) => io.observe(e));
  // safety: never leave content invisible
  setTimeout(() => els.forEach((e) => e.classList.add("in")), 1600);
})();

// ================================================================ PARTICLES ===
(function initParticleNetwork() {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  const canvas = document.createElement("canvas");
  canvas.id = "bg-canvas";
  canvas.setAttribute("aria-hidden", "true");
  document.body.prepend(canvas);

  const ctx = canvas.getContext("2d");
  const COLORS = ["74,222,128", "34,211,238", "129,140,248"];
  const LINE   = 135;
  const mouse  = { x: -999, y: -999 };
  let W, H, pts;

  class Dot {
    reset(w, h) {
      this.x  = Math.random() * w;
      this.y  = Math.random() * h;
      this.vx = (Math.random() - 0.5) * 0.42;
      this.vy = (Math.random() - 0.5) * 0.42;
      this.r  = Math.random() * 1.2 + 0.6;
      this.c  = COLORS[Math.floor(Math.random() * COLORS.length)];
      return this;
    }
    tick(w, h) {
      const dx = mouse.x - this.x, dy = mouse.y - this.y;
      const md = Math.hypot(dx, dy);
      if (md < 160 && md > 0) { this.vx += (dx / md) * 0.016; this.vy += (dy / md) * 0.016; }
      this.vx *= 0.984; this.vy *= 0.984;
      const spd = Math.hypot(this.vx, this.vy);
      if (spd > 0.88) { this.vx = (this.vx / spd) * 0.88; this.vy = (this.vy / spd) * 0.88; }
      this.x = (this.x + this.vx + w) % w;
      this.y = (this.y + this.vy + h) % h;
    }
  }

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function setup() {
    resize();
    const n = window.innerWidth < 768 ? 30 : 65;
    pts = Array.from({ length: n }, () => new Dot().reset(W, H));
  }

  function frame() {
    ctx.clearRect(0, 0, W, H);
    const n = pts.length;
    for (let i = 0; i < n; i++) {
      pts[i].tick(W, H);
      // draw connections
      for (let j = i + 1; j < n; j++) {
        const d = Math.hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y);
        if (d < LINE) {
          ctx.beginPath();
          ctx.moveTo(pts[i].x, pts[i].y);
          ctx.lineTo(pts[j].x, pts[j].y);
          ctx.strokeStyle = `rgba(${pts[i].c},${(1 - d / LINE) * 0.18})`;
          ctx.lineWidth   = 0.85;
          ctx.stroke();
        }
      }
      // draw dot
      ctx.beginPath();
      ctx.arc(pts[i].x, pts[i].y, pts[i].r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${pts[i].c},0.55)`;
      ctx.fill();
    }
    requestAnimationFrame(frame);
  }

  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(setup, 120);
  });
  window.addEventListener("mousemove", (e) => { mouse.x = e.clientX; mouse.y = e.clientY; });
  window.addEventListener("mouseleave", () => { mouse.x = -999; mouse.y = -999; });

  setup();
  frame();
})();
