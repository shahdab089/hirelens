# Gemini Handoff — Workstream A (UI + Evals)

You (Gemini CLI) own **two files only**: `evals/eval_harness.py` and `app.py`.
Do NOT edit any other file. Everything you need already exists and is tested —
import it, don't reinvent it. All data flows through the models in
`core/schema.py` (the single source of truth — never redefine those models).

The project's LLM provider is **Groq** (free). It reads `GROQ_API_KEY` from the
environment. You do not call Groq directly — the functions below do.

---

## Functions you can call (already built, stable signatures)

```python
# Parsing — core/parsing.py
parse_resume(text: str) -> ParsedResume
parse_jd(text: str) -> ParsedJD
extract_text(file_path: str) -> str        # .pdf / .docx / .txt -> text

# Scoring + diagnosis — core/scoring.py, core/diagnosis.py
score(resume: ParsedResume, jd: ParsedJD) -> FitScore
diagnose(fit: FitScore, resume: ParsedResume, jd: ParsedJD) -> Diagnosis

# Storage — storage.py  (stdlib sqlite3, file applications.db)
save(record: ApplicationRecord) -> None
load_all() -> list[ApplicationRecord]
set_outcome(id: str, outcome: str) -> None   # outcome in {rejected,interview,ghosted,offer}

# Analytics — analytics/patterns.py
build_report(records: list[ApplicationRecord]) -> PatternReport
```

Read `core/schema.py` for the exact fields of every model (`ParsedResume`,
`ParsedJD`, `FitScore`, `SubScore`, `Diagnosis`, `RejectionStage`,
`ApplicationRecord`, `PatternReport`).

---

## File 1 — `evals/eval_harness.py`

**Goal:** measure how well the scorer agrees with real outcomes in
`data/labels.csv`. This is the interview-impressive part — make the output clear.

**Required function:**
```python
def run_eval(labels_csv: str = "data/labels.csv",
             samples_dir: str = "data/samples") -> dict
```

**What it must do, per row of `labels.csv` (`resume_file, jd_file, real_outcome`):**
1. `extract_text` both files from `samples_dir`, then `parse_resume` / `parse_jd`.
2. `score(...)` then `diagnose(...)`.
3. Map `real_outcome` to a binary "advanced" label: `interview`/`offer` -> 1,
   `rejected`/`ghosted` -> 0.
4. Map `fit.overall` to a predicted "advanced" label using threshold 0.5.
5. Collect: per-row fit, real_outcome, predicted_stage, agreement (pred == actual).

**Return** a dict with at least: `n`, `accuracy` (mean agreement), `avg_fit_advanced`
(mean fit of real interview/offer rows), `avg_fit_rejected` (mean fit of real
rejected/ghosted rows), and a `rows` list. The two avg-fit numbers should show the
scorer gives higher fit to applications that really advanced — print them.

**Make it runnable:** `if __name__ == "__main__":` calls `run_eval()` and prints a
readable summary table. Requires `GROQ_API_KEY` to be set (it makes real LLM calls).

**Acceptance:** `python -m evals.eval_harness` runs end to end and prints accuracy
plus the two average-fit numbers. Put an empty `evals/__init__.py` alongside it.

---

## File 2 — `app.py` (Streamlit)

**Goal:** the demo. Run with `streamlit run app.py`.

**Tab 1 — "Analyze an application":**
- Two text areas: resume text, JD text (and optionally a file uploader that calls
  `extract_text` on a saved temp file).
- A "Analyze" button that runs `parse_resume` -> `parse_jd` -> `score` -> `diagnose`.
- Display: overall fit (a metric / progress bar), the four sub-scores with
  rationales, matched vs missing skills, then the Diagnosis (headline big,
  explanation, top_fixes as a bullet list, likely_stage as a tag).
- A "Log this application" control: pick an outcome (rejected/interview/ghosted/
  offer or leave None), build an `ApplicationRecord` (`id=str(uuid4())`,
  `created_at=datetime.now()`), and `save(...)` it.

**Tab 2 — "Your patterns":**
- `records = load_all()`; if empty, show a friendly empty state.
- Otherwise call `build_report(records)` and show `total_applications`,
  `avg_overall_fit`, `dominant_stage`, the `insight` string, and
  `recommended_focus` as bullets.
- A small table of logged applications with a way to update an outcome via
  `set_outcome(id, outcome)`.

**Notes:**
- Handle the missing-key case gracefully: if a call raises `ValueError`
  mentioning `GROQ_API_KEY`, show a `st.error` telling the user to set it.
- Keep it to one file. Don't touch the schema or any core/ file.

**Acceptance:** `streamlit run app.py` launches, both tabs render, and analyzing a
pasted resume+JD shows a score and diagnosis (with `GROQ_API_KEY` set).

---

## Integration check (run after both files exist)
```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # existing tests still pass
python -m evals.eval_harness        # needs GROQ_API_KEY
streamlit run app.py                # needs GROQ_API_KEY
```
