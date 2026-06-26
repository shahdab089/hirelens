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
