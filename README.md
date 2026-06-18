---
title: Hirelens
emoji: 🔍
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Hirelens

See **why you're not getting hired.** Hirelens tells a job seeker why their
applications get rejected — and shows the **pattern across all of them**. Paste a
résumé and a job description, get a brutally honest fit score, the likely
rejection stage, and concrete fixes.

> Two interfaces ship in this repo: a **FastAPI web app** (`app_web.py`, the
> primary, deployed on Hugging Face Spaces) and a **Streamlit app** (`app.py`,
> an alternative). Both reuse the same `core/` logic.

## Key Features
- **Rejection analysis** — the single most likely reason an application failed.
- **Fit scoring** — overall + four sub-scores (skills, seniority, ATS keywords, domain).
- **Skill gaps** — matched vs. missing requirements, judged semantically.
- **Pattern recognition** — the dominant bottleneck across everything you log.

## How It Works
An LLM (via the free **Groq** API) parses the résumé and JD into structured data,
scores the fit, and diagnoses the likely rejection stage. No training — inference
only. The server reads the key from the `GROQ_API_KEY` environment variable;
end users never enter a key.

## Run the web app locally
```bash
pip install -r requirements.txt
export GROQ_API_KEY="gsk_your_free_key"     # Windows PowerShell: $env:GROQ_API_KEY="..."
python -m uvicorn app_web:app --reload --port 8000
# open http://localhost:8000
```
Click **Try a live sample** to demo it with no typing.

## Deploy free on Hugging Face Spaces (share on LinkedIn)
1. Create a new **Space** at https://huggingface.co/new-space → **SDK: Docker**.
2. Push this repo's contents to the Space (it auto-builds from the `Dockerfile`).
3. In **Settings → Variables and secrets**, add a secret:
   `GROQ_API_KEY = gsk_your_key_here`
4. The Space builds and serves on port 7860. Share the public `*.hf.space` URL.

Visitors use your shared key automatically (Groq's free tier is rate-limited
across all visitors). Each visitor's logged history is kept separate via an
anonymous browser id.

> Note: on the free tier the SQLite history is ephemeral (resets on rebuild).
> Enable persistent storage if you need it to survive restarts.

## Alternative: Streamlit interface
```bash
streamlit run app.py
```
Paste a key in the sidebar, or set `GROQ_API_KEY` / a `.streamlit/secrets.toml`.

## Evaluate the scorer
```bash
python -m evals.eval_harness     # needs GROQ_API_KEY
```
Reports accuracy and shows the scorer gives higher fit to applications that
really advanced (interview/offer) than to those that didn't.

## Data
- `data/samples/` — 15 varied résumé/JD pairs (strong matches, skill gaps,
  seniority/domain mismatches, ATS cases).
- `data/labels.csv` — ground-truth outcomes consistent with each pair's fit.
