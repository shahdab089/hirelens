# Application Autopsy — Multi-Agent Task Brief

**Goal:** An AI tool that tells a laid-off job seeker *why* their applications get rejected,
and (the wedge) shows the pattern across all their applications. Built to be a deployed,
demo-able interview piece that shows real AI engineering.

**Hard rule for all agents:** You own only your files (listed below). Do NOT edit files
owned by another agent. Everyone imports the data models from `core/schema.py` — never
redefine them. If you need a change to the shared schema, request it; do not edit it yourself.

**Stack:** Python 3.11+, Pydantic, Streamlit, SQLite (stdlib `sqlite3`), Claude API for
analysis. Prototype in notebooks, ship as a Streamlit app.

---

## The contract (already written): `core/schema.py`
The pipeline is: raw text → parse → score → diagnose → store → aggregate → display.
Read that file first. Every function below consumes/produces those models.

---

## Workstream A — CLAUDE (the AI core; hardest parts)
**Owns:** `core/schema.py`, `core/scoring.py`, `core/diagnosis.py`, `evals/`, `app.py`
- [ ] `scoring.py`: `score(resume: ParsedResume, jd: ParsedJD) -> FitScore` via Claude
      structured JSON output. Sub-scores for skills / seniority / keywords_ats / domain.
- [ ] `diagnosis.py`: `diagnose(fit: FitScore, resume, jd) -> Diagnosis` — the brutal,
      specific "here's why it failed + top fixes" narrative.
- [ ] `evals/eval_harness.py`: run scorer over a labeled set, report agreement with real
      outcomes (this is the interview-impressive part).
- [ ] `app.py`: Streamlit UI — paste resume + JD → show fit score + diagnosis; second tab
      shows the PatternReport across logged apps.

## Workstream B — GEMINI CLI (parsing, storage, analytics; agentic coding)
**Owns:** `core/parsing.py`, `storage.py`, `analytics/patterns.py`, `requirements.txt`
- [x] `parsing.py`:
      - `parse_resume(text: str) -> ParsedResume`
      - `parse_jd(text: str) -> ParsedJD`
      - Also a PDF/DOCX → text extractor helper (use `pypdf`, `python-docx`).
      LLM-assisted extraction is fine; output MUST match the schema models exactly.
- [x] `storage.py`: SQLite-backed `save(record: ApplicationRecord)`, `load_all() ->
      list[ApplicationRecord]`, `set_outcome(id, outcome)`. One file, stdlib sqlite3.
- [x] `analytics/patterns.py`: `build_report(records: list[ApplicationRecord]) ->
      PatternReport` — pure aggregation logic, no LLM needed (count stages, average fit,
      find the dominant rejection stage, write the insight string).
- [x] `requirements.txt`: pin every dependency the project uses.

## Workstream C — LOCAL GEMMA (test data + docs; low-stakes generation)
**Owns:** `data/samples/`, `README.md` (draft)
- [x] Generate 15–20 **synthetic sample resumes** (varied seniority/fields) as `.txt`.
- [x] Generate 15–20 **synthetic job descriptions** to pair with them as `.txt`.
- [x] Draft a `labels.csv`: `resume_file, jd_file, real_outcome` for the eval set
      (make plausible outcomes: rejected / interview / ghosted).
- [x] Draft `README.md` prose (what it does, who it's for) — Gemini/Claude will polish.
> Gemma is the weakest agent: keep its tasks to text generation, not code or tool use.

---

## Integration order (so nothing blocks)
1. Schema is done (Claude). 2. Gemma produces sample data + labels. 3. Gemini builds
parsing + storage against schema. 4. Claude builds scoring + diagnosis against schema +
sample data. 5. Gemini builds analytics. 6. Claude wires `app.py` + evals. 7. Deploy to
Streamlit Community Cloud.

## Anti-collision rules
- Branch per agent or commit only your owned files. Never touch `core/schema.py` unless Claude.
- All cross-module calls go through schema models — no agent reaches into another's internals.
- If a signature in this brief and `schema.py` disagree, **schema.py wins**.
