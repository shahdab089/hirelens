// ---- anonymous client id (keeps each visitor's history separate) ----
const CLIENT_ID = (() => {
  let id = localStorage.getItem("aa_client_id");
  if (!id) { id = crypto.randomUUID(); localStorage.setItem("aa_client_id", id); }
  return id;
})();

const $ = (id) => document.getElementById(id);
let lastRecord = null;       // serialized ApplicationRecord from /api/analyze
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
  $("gauge-num").textContent = pct + "%";
  $("gauge-num").style.color = sevColor(d.overall);
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
