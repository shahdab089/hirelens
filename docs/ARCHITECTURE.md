# Hirelens — Architecture & Interview Walkthrough

A complete, plain-English guide to how Hirelens works, why each decision was made,
and how to explain it confidently in an interview.

---

## 1. The 30-second and 2-minute pitch

**30 seconds (the elevator version):**
> "Hirelens is an AI tool for job seekers. You paste your résumé and a job
> description, and it gives you an honest fit score, the single most likely
> reason you'd get rejected, and concrete fixes — plus the pattern across all the
> roles you check. It's a deployed full-stack web app with an LLM-powered analysis
> core, a FastAPI backend, and a custom frontend."

**2 minutes (the structured version):** problem → solution → how it works → stack → results.
- **Problem:** Job seekers get rejected and ghosted with zero feedback. They don't know if it's skills, seniority, an ATS keyword screen, or just bad luck.
- **Solution:** An app that diagnoses *why* a specific application is weak and shows the bottleneck across many applications.
- **How:** A pipeline — parse → score → diagnose → store → aggregate → display.
- **Stack:** Python, Pydantic, an LLM via the Anthropic API (Claude Haiku 4.5 primary) with a Groq/Llama fallback, FastAPI, SQLite, a vanilla HTML/CSS/JS frontend, Docker, deployed on Render/Hugging Face.
- **Result:** A live, shareable product, plus an evaluation harness that measures how well the scorer agrees with real outcomes.

---

## 2. The core idea: a pipeline

Everything is one data pipeline. If you remember this diagram, you can explain the whole system:

```
 raw résumé text + raw job-description text
        │
        ▼   parse_resume / parse_jd        (core/parsing.py)
 ParsedResume / ParsedJD   ── structured: skills, seniority, requirements …
        │
        ▼   score(resume, jd)              (core/scoring.py)
 FitScore   ── overall + 4 sub-scores + matched/missing skills
        │
        ▼   diagnose(fit, resume, jd)      (core/diagnosis.py)
 Diagnosis  ── likely rejection stage + headline + explanation + fixes
        │
        ▼   save(record, client_id)        (storage.py, SQLite)
 ApplicationRecord (persisted)
        │
        ▼   build_report(records)          (analytics/patterns.py)
 PatternReport  ── dominant bottleneck + insight across all applications
        │
        ▼   shown in the browser           (app_web.py + web/static/*)
```

**Key sentence for the interview:** *"Every stage consumes and produces a typed
Pydantic model defined in one shared schema file, so the modules are decoupled and
talk only through data contracts."*

---

## 3. The shared contract (`core/schema.py`)

This is the **single source of truth**. Every model is a Pydantic `BaseModel`:

- `ParsedResume` — raw_text, skills[], years_experience, seniority, titles[]
- `ParsedJD` — raw_text, title, company, required_skills[], nice_to_have_skills[], seniority, hard_requirements[]
- `SubScore` — name, score (0–1), rationale
- `FitScore` — overall (0–1), subscores[], matched_skills[], missing_skills[]
- `RejectionStage` — an **Enum**: keyword_ats, seniority_mismatch, skills_gap, domain_mismatch, competitive, likely_fine
- `Diagnosis` — likely_stage, headline, explanation, top_fixes[]
- `ApplicationRecord` — id, created_at, jd, resume, fit, diagnosis, outcome
- `PatternReport` — total_applications, avg_overall_fit, dominant_stage, insight, recommended_focus[]

**Why this matters (talking point):** "Defining the data contract first let me build
the layers independently and even split the work across multiple AI coding agents —
each owned different files but everyone imported the same models. It's contract-first
design."

---

## 4. Component deep-dives (what + why)

### 4.1 Parsing — `core/parsing.py`
- **What:** `extract_text()` pulls text from PDF/DOCX/TXT (pypdf, python-docx). `parse_resume()` / `parse_jd()` send that text to the LLM and get back structured JSON matching the schema.
- **How:** The Pydantic model's JSON schema is injected into the prompt; the LLM returns JSON; we validate it into the model.
- **Why LLM instead of regex:** résumés/JDs are unstructured free text with infinite formats — rules break; an LLM generalizes.

### 4.2 Shared LLM helper — `core/llm.py`
- **What:** lazy clients for both providers plus `complete_json()` / `chat_json()` — calls the model with `temperature=0`, JSON output, and **retries**. Tries **Anthropic Claude Haiku 4.5** first and automatically falls back to **Groq (Llama 3.3 70B)** on a rate-limit.
- **Why lazy:** the app and tests can import everything without an API key present; the key is only needed at call time.
- **Why JSON mode + temp 0:** forces valid JSON and makes output deterministic/repeatable.
- **Why retries + fallback:** LLM/network calls are flaky; we retry with backoff, and if the primary provider is throttled we switch providers (independent token buckets) before failing.
- **Swappable models:** `ANTHROPIC_MODEL` / `GROQ_FALLBACK_MODEL` env vars — change models without touching code.

### 4.3 Scoring — `core/scoring.py`
- **What:** `score(resume, jd) -> FitScore`. The LLM rates four dimensions (skills, seniority, ATS keywords, domain), gives an overall score, and decides **semantically** which requirements are matched vs. missing.
- **Two design iterations (great story):**
  1. *First version:* matched/missing skills were computed **deterministically** (exact string match) for reliability.
  2. *Problem found:* on a real Mastercard JD, the candidate clearly fit but it showed "0 matched skills" — because the JD listed broad categories ("SQL & Data Manipulation") that didn't *literally* equal the résumé's atomic skills ("SQL"). The 80% skills score contradicted "0 matched."
  3. *Fix:* moved matched/missing to **semantic** LLM judgment (kept the deterministic version as a fallback).
- **Seniority calibration:** the model was over-flagging "seniority mismatch" for a 6-year candidate on a role that said "Senior" once in the body. Fixed with explicit **year-bands** (junior ~0-2, mid ~3-5, senior ~5-9, staff ~9+) so 6 years *satisfies* senior.
- **Talking point:** "I used deterministic logic where I could and the LLM only for judgment — then adjusted that boundary based on real failures. I also defensively validate and clamp every score."

### 4.4 Diagnosis — `core/diagnosis.py`
- **What:** `diagnose(fit, resume, jd) -> Diagnosis`. Picks the single most likely `RejectionStage` from the enum and writes a headline, explanation, and concrete fixes.
- **Guardrails:** a decision rule — only call it a seniority mismatch if there's a *real* seniority gap; if overall fit ≥ 0.8 and nothing is clearly weak, call it `likely_fine` (so a strong match doesn't read as "rejected"). A **fallback** maps the weakest sub-score to a stage if the LLM returns an invalid enum value.

### 4.5 Storage — `storage.py`
- **What:** SQLite via Python's stdlib `sqlite3`. `save()`, `load_all()`, `load_by_client()`, `set_outcome()`.
- **Design points:**
  - The full record is stored as a **JSON blob** (Pydantic `model_dump_json`) plus a few indexed columns (id, created_at, outcome, client_id) — flexible and simple.
  - `client_id` column enables **per-visitor isolation without login** (more below).
  - `init_db()` runs a lightweight **migration** (adds the client_id column if an older DB lacks it).
  - `DB_PATH` is **env-configurable** (`APP_DB_PATH`) so the container can write to a writable dir like `/tmp`.
  - `set_outcome()` **validates** the outcome against an allowed set.

### 4.6 Analytics — `analytics/patterns.py`
- **What:** `build_report(records) -> PatternReport`. Pure aggregation, **no LLM** — counts, averages, and finds the dominant bottleneck.
- **Subtle correctness fix:** the "dominant rejection stage" only counts **unsuccessful** outcomes (rejected/ghosted). Counting a successful application's diagnosis as a "rejection" stage would be wrong.

### 4.7 Evaluation harness — `evals/eval_harness.py`
- **What:** runs the scorer over a labeled dataset (`data/labels.csv`, 15 résumé/JD pairs with known outcomes) and reports **accuracy** plus the average fit of applications that advanced vs. those that didn't.
- **Why it matters (big talking point):** "I didn't just eyeball quality — I built an eval that measures whether the model's fit scores actually agree with real outcomes. That's the difference between a demo and an engineered system."

### 4.8 Web backend — `app_web.py` (FastAPI)
- **What:** a thin API over the core logic. Endpoints: `/api/analyze`, `/api/extract` (file upload), `/api/log`, `/api/outcome`, `/api/patterns`, `/api/samples`, `/api/health`. Serves the static frontend.
- **Why FastAPI:** async, fast, and it validates request/response bodies with Pydantic automatically (same models reused).
- **Key handling:** the API key is read from a server-side env var; the browser **never** sees it.

### 4.9 Frontend — `web/static/` (index.html, styles.css, app.js)
- **What:** a custom landing page + single-page app. A Chart.js gauge for fit, sub-score bars, skill chips, a diagnosis card, and a patterns dashboard with charts and an editable history table.
- **Multi-user without accounts:** the browser generates a random `client_id` (stored in `localStorage`) and sends it with each request, so each visitor sees only their own logged history.
- **No framework:** vanilla JS keeps it dependency-light and fast.

---

## 5. Tech stack & why (cheat sheet)

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Best ecosystem for AI/data work |
| Data models | Pydantic v2 | Validation, JSON (de)serialization, schema generation for prompts |
| LLM inference | Anthropic Claude Haiku 4.5 (primary) → Groq Llama 3.3 70B (fallback) | High-quality structured JSON; free/cheap fast fallback on rate-limit; models swappable via env var |
| Backend | FastAPI + Uvicorn | Async, fast, Pydantic-native request validation |
| Persistence | SQLite (stdlib) | Zero-config, file-based, fine for this scale |
| Frontend | Vanilla HTML/CSS/JS + Chart.js | Lightweight, no build step, full design control |
| File parsing | pypdf, python-docx | Extract text from uploaded résumés |
| Packaging | Docker | One image runs identically on any host |
| Hosting | Render / Hugging Face Spaces | Free tiers with Docker support |

---

## 6. Key engineering decisions & trade-offs (the meat of the interview)

1. **Contract-first / schema-driven design.** One Pydantic schema is the source of truth; layers are decoupled and testable in isolation.
2. **Deterministic where possible, LLM where necessary.** Aggregation is pure code; only genuine judgment goes to the LLM. I moved skill-matching from deterministic to semantic *after* real evidence showed exact-match failing.
3. **Defensive LLM handling.** JSON mode, `temperature=0`, retries with backoff, Pydantic validation, value clamping, and enum fallbacks — because LLMs are probabilistic and will occasionally misbehave.
4. **Evidence-based prompt iteration.** Seniority year-bands and the "likely_fine at high fit" rule came from observing concrete failures, not guessing.
5. **Objective evaluation.** A labeled eval set + harness measures agreement with real outcomes.
6. **12-factor config.** Everything host-specific is an env var (`ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_MODEL`, `GROQ_FALLBACK_MODEL`, `APP_DB_PATH`, `$PORT`) so the *same* Docker image runs on Render and Hugging Face unchanged.
7. **Multi-tenancy without auth.** Anonymous `client_id` in localStorage isolates each visitor's data — a pragmatic choice for a no-login public demo.
8. **Security.** The API key lives only on the server; secrets come from the platform's secret store; `.gitignore` keeps keys/DBs out of git.

---

## 7. Deployment & DevOps

- **Docker:** `python:3.11-slim`, install pinned `requirements.txt`, copy code, bind to `${PORT:-7860}` (Render injects `$PORT`; HF uses 7860).
- **Hugging Face Spaces:** Docker SDK; config via YAML front-matter in `README.md`; key stored as a Space **secret**.
- **Render:** `render.yaml` blueprint (infrastructure-as-code) defines a free Docker web service with a `/api/health` health check and both `ANTHROPIC_API_KEY` and `GROQ_API_KEY` as dashboard-entered secrets (`sync: false`).
- **Git remotes:** `origin` → GitHub (source), `space` → Hugging Face. Render auto-deploys from GitHub.
- **CI-ready:** `python -m pytest` runs the storage/analytics test; the eval harness is a separate quality check.

---

## 8. Limitations & future work (always have these ready)

- **Ephemeral storage** on free tiers — SQLite resets on rebuild. Fix: managed Postgres or a persistent volume.
- **No accounts/auth** — `client_id` is per-browser, not a real user. Fix: add auth + per-user data.
- **LLM rate limits** on the free tier under load. Fix: queueing, caching identical analyses, or a paid tier.
- **Provider dependency** — mitigated by a two-provider setup (Anthropic primary, Groq fallback with independent token buckets) behind a provider-agnostic `chat_json`/`complete_json` boundary, with models swappable via env vars.
- **LLM nondeterminism / hallucination** — mitigated by validation, fallbacks, and evals, but not eliminated.
- **Roadmap:** a recruiter-contact extractor (with attention to platform ToS and email/privacy law), persistent multi-user accounts, and richer analytics.

---

## 9. Interview Q&A bank (rehearse these)

**Q: Walk me through the architecture.**
> "It's a pipeline: parse → score → diagnose → store → aggregate → display. Each stage is a function that takes and returns a typed Pydantic model from one shared schema, so the layers are decoupled. A FastAPI backend exposes the pipeline as JSON endpoints and serves a custom frontend."

**Q: Why use an LLM for scoring instead of rules?**
> "Résumés and JDs are unstructured and infinitely varied — rules are brittle. The LLM generalizes. But I keep deterministic logic where it's reliable, like pure aggregation, and only use the LLM for genuine judgment."

**Q: How do you make sure the LLM returns something usable?**
> "JSON mode, temperature 0, a prompt that includes the exact target schema, retries with backoff, then Pydantic validation. I clamp numeric scores and fall back to a derived value if the model omits or returns an invalid field."

**Q: How do you know it's actually any good?**
> "I built an evaluation harness over a labeled dataset and measure whether higher fit scores correspond to applications that actually advanced. That turns 'looks right' into a number."

**Q: Tell me about a bug you fixed.**
> "On a real job description, a clearly-qualified candidate showed '0 matched skills' because I was doing exact string matching against JD skill *categories*. I switched matching to semantic LLM judgment, which fixed the contradiction. Another one: the loading spinner overlay was permanently visible because a CSS `display` rule overrode the `hidden` attribute — I added a global `[hidden]{display:none!important}` rule."

**Q: How would you scale to 10,000 users?**
> "Move SQLite to managed Postgres, add caching for identical résumé/JD pairs, put analyses on a queue to respect LLM rate limits, run multiple stateless backend replicas behind a load balancer, and add real auth."

**Q: Why SQLite — isn't that a toy?**
> "For a single-instance app at this scale it's perfect: zero-config, file-based, transactional. I isolated it behind a small storage module, so swapping to Postgres later is a localized change."

**Q: How do you handle multiple users without login?**
> "Each browser generates a UUID stored in localStorage and sends it with requests; the backend scopes stored records by that id. It's a lightweight way to give per-visitor history on a no-signup demo."

**Q: Security?**
> "The LLM key is server-side only, injected as an env var from the platform's secret store and never exposed to the client. Secrets and the database are gitignored."

**Q: Why this multi-model 'agent' build process?**
> "I defined the schema as a contract, then had different AI coding agents own different layers — one for the AI core, one for the frontend and evals — coordinating only through the shared models. It let me build fast while keeping clean separation, the same way a team would split work behind an API contract."

---

## 10. Glossary (so the vocabulary is automatic)

- **ATS** — Applicant Tracking System; software that keyword-screens résumés before a human sees them.
- **Structured output / JSON mode** — forcing the LLM to return valid JSON.
- **Pydantic model** — a typed, self-validating Python data class.
- **Eval harness** — code that scores the system against known-correct labels.
- **12-factor config** — keeping deploy-specific settings in environment variables.
- **Idempotent / stateless** — a request doesn't depend on server memory between calls (helps scaling).
- **Multi-tenancy** — serving many users from one app while keeping their data separate.
