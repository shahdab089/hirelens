"""
Application Autopsy — FastAPI backend.

Thin web layer over the existing core logic (parsing/scoring/diagnosis/storage/
analytics). Serves the custom frontend in web/static and exposes a small JSON API.
The Groq API key lives only on the server (env var GROQ_API_KEY) — clients never
send it. Per-visitor history is scoped by an anonymous client_id.
"""
import os
import tempfile

# Load .env file automatically for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed -- use real env vars (production)
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from analytics.patterns import build_report
from core.contacts import extract_contacts
from core.llm import GroqRateLimit
from core.outreach import draft_outreach
from core.parsing import extract_text, parse_both, parse_jd, parse_resume
from core.schema import ApplicationRecord
from core.scoring import score_and_diagnose
from storage import load_by_client, save, set_outcome

BASE = Path(__file__).parent
STATIC_DIR = BASE / "web" / "static"
SAMPLES_DIR = BASE / "data" / "samples"
LABELS_CSV = BASE / "data" / "labels.csv"

STAGE_LABELS = {
    "keyword_ats": "Keyword / ATS filter",
    "seniority_mismatch": "Seniority mismatch",
    "skills_gap": "Skills gap",
    "domain_mismatch": "Domain mismatch",
    "competitive": "Out-competed",
    "likely_fine": "Looks fine",
}

app = FastAPI(title="Hirelens", docs_url="/api/docs")


# ------------------------------------------------------------- request models -
class AnalyzeReq(BaseModel):
    resume_text: str
    jd_text: str


class LogReq(BaseModel):
    client_id: str
    record: dict
    outcome: Optional[str] = None


class OutcomeReq(BaseModel):
    id: str
    outcome: str


class OutreachReq(BaseModel):
    record: dict


class TriageJD(BaseModel):
    text: str
    label: Optional[str] = None


class TriageReq(BaseModel):
    resume_text: str
    jds: list[TriageJD]


# --------------------------------------------------------------------- helpers -
MAX_TRIAGE_JDS = 10

VERDICT_LABELS = {
    "apply_hard": "Apply hard",
    "worth_a_shot": "Worth a shot",
    "skip": "Skip",
}


def _verdict(overall: float) -> str:
    """Bucket an overall fit score into an actionable recommendation."""
    if overall >= 0.75:
        return "apply_hard"
    if overall >= 0.50:
        return "worth_a_shot"
    return "skip"
def _require_key():
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_groq = bool(os.environ.get("GROQ_API_KEY"))
    if not has_anthropic and not has_groq:
        raise HTTPException(
            status_code=503,
            detail="The analysis service isn't configured yet. Please try again later.",
        )


# ------------------------------------------------------------------- API routes -
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "anthropic_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "groq_configured": bool(os.environ.get("GROQ_API_KEY")),
    }


@app.post("/api/analyze")
def api_analyze(req: AnalyzeReq):
    _require_key()
    if not req.resume_text.strip() or not req.jd_text.strip():
        raise HTTPException(400, "Please provide both a résumé and a job description.")
    try:
        resume, jd = parse_both(req.resume_text, req.jd_text)
        fit, diag = score_and_diagnose(resume, jd)
    except ValueError as err:
        raise HTTPException(400, f"Could not parse the inputs: {err}")
    except GroqRateLimit:
        raise HTTPException(429, "We're experiencing high demand right now — please wait a minute and try again.")
    except Exception as err:  # noqa: BLE001
        raise HTTPException(500, f"Analysis failed: {err}")

    record = ApplicationRecord(
        id=str(uuid4()),
        created_at=datetime.now(),
        jd=jd,
        resume=resume,
        fit=fit,
        diagnosis=diag,
    )
    return {
        "overall": fit.overall,
        "subscores": [s.model_dump() for s in fit.subscores],
        "matched_skills": fit.matched_skills,
        "missing_skills": fit.missing_skills,
        "diagnosis": diag.model_dump(),
        "stage_label": STAGE_LABELS.get(diag.likely_stage.value, diag.likely_stage.value),
        "jd_title": jd.title,
        "jd_company": jd.company,
        # contacts the recruiter published in the posting (regex over pasted text).
        "contacts": extract_contacts(req.jd_text).model_dump(),
        # full serialized record so the client can log it without re-running.
        "record": record.model_dump(mode="json"),
    }


@app.post("/api/triage")
def api_triage(req: TriageReq):
    """
    Batch mode: score ONE résumé against MANY job descriptions and rank them.

    Parses the résumé once (not per-JD) to save tokens, then runs each JD through
    the normal parse + score + diagnose pipeline. A single malformed JD is isolated
    as an error row rather than failing the whole batch; a provider rate-limit stops
    the batch early and returns whatever has been scored so far.
    """
    _require_key()
    if not req.resume_text.strip():
        raise HTTPException(400, "Please provide your résumé.")
    jds = [j for j in req.jds if j.text.strip()]
    if not jds:
        raise HTTPException(400, "Please add at least one job description.")
    if len(jds) > MAX_TRIAGE_JDS:
        raise HTTPException(400, f"Please limit to {MAX_TRIAGE_JDS} job descriptions per batch.")

    # Parse the résumé ONCE — avoids re-extracting it for every JD.
    try:
        resume = parse_resume(req.resume_text)
    except GroqRateLimit:
        raise HTTPException(429, "We're experiencing high demand right now — please wait a minute and try again.")
    except Exception as err:  # noqa: BLE001
        raise HTTPException(400, f"Could not parse your résumé: {err}")

    results = []
    rate_limited = False
    for i, item in enumerate(jds):
        label = (item.label or "").strip()
        try:
            jd = parse_jd(item.text)
            fit, diag = score_and_diagnose(resume, jd)
            verdict = _verdict(fit.overall)
            stage = diag.likely_stage.value
            record = ApplicationRecord(
                id=str(uuid4()),
                created_at=datetime.now(),
                jd=jd,
                resume=resume,
                fit=fit,
                diagnosis=diag,
            )
            results.append({
                "label": label or jd.title or f"Job {i + 1}",
                "jd_title": jd.title,
                "jd_company": jd.company or "—",
                "overall": fit.overall,
                "stage": stage,
                "stage_label": STAGE_LABELS.get(stage, stage),
                "verdict": verdict,
                "verdict_label": VERDICT_LABELS[verdict],
                "top_fix": diag.top_fixes[0] if diag.top_fixes else "",
                "headline": diag.headline,
                "matched_count": len(fit.matched_skills),
                "missing_count": len(fit.missing_skills),
                # full serialized record so the client can save it to patterns.
                "record": record.model_dump(mode="json"),
                "error": None,
            })
        except GroqRateLimit:
            rate_limited = True
            break  # stop the batch; return partial results
        except Exception as err:  # noqa: BLE001
            results.append({
                "label": label or f"Job {i + 1}",
                "jd_title": label or f"Job {i + 1}",
                "jd_company": "—",
                "overall": None,
                "stage": None,
                "stage_label": "—",
                "verdict": "error",
                "verdict_label": "Error",
                "top_fix": "",
                "headline": "",
                "matched_count": 0,
                "missing_count": 0,
                "record": None,
                "error": str(err),
            })

    # Rank by fit (highest first); error rows sink to the bottom.
    ranked = sorted(results, key=lambda r: (r["overall"] is not None, r["overall"] or -1), reverse=True)

    scored = [r for r in ranked if r["overall"] is not None]
    buckets = {"apply_hard": 0, "worth_a_shot": 0, "skip": 0}
    stage_counts: dict[str, int] = {}
    for r in scored:
        buckets[r["verdict"]] = buckets.get(r["verdict"], 0) + 1
        if r["verdict"] == "skip":
            stage_counts[r["stage_label"]] = stage_counts.get(r["stage_label"], 0) + 1
    dominant_blocker = max(stage_counts, key=stage_counts.get) if stage_counts else None

    return {
        "results": ranked,
        "summary": {
            "total": len(scored),
            "apply_hard": buckets["apply_hard"],
            "worth_a_shot": buckets["worth_a_shot"],
            "skip": buckets["skip"],
            "best": scored[0]["label"] if scored else None,
            "dominant_blocker": dominant_blocker,
            "rate_limited": rate_limited,
        },
    }


@app.post("/api/outreach")
def api_outreach(req: OutreachReq):
    _require_key()
    try:
        record = ApplicationRecord.model_validate(req.record)
    except Exception as err:  # noqa: BLE001
        raise HTTPException(400, f"Invalid record: {err}")
    try:
        draft = draft_outreach(record.resume, record.jd, record.fit)
    except GroqRateLimit:
        raise HTTPException(429, "We're experiencing high demand right now — please wait a minute and try again.")
    except Exception as err:  # noqa: BLE001
        raise HTTPException(500, f"Could not generate outreach: {err}")
    return draft.model_dump()


@app.post("/api/extract")
async def api_extract(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".txt"
    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        text = extract_text(tmp_path)
    except Exception as err:  # noqa: BLE001
        raise HTTPException(400, f"Could not read that file: {err}")
    finally:
        os.unlink(tmp_path)
    return {"text": text}


@app.post("/api/log")
def api_log(req: LogReq):
    try:
        record = ApplicationRecord.model_validate(req.record)
    except Exception as err:  # noqa: BLE001
        raise HTTPException(400, f"Invalid record: {err}")
    record.outcome = req.outcome or None
    save(record, client_id=req.client_id)
    return {"ok": True, "id": record.id}


@app.post("/api/outcome")
def api_outcome(req: OutcomeReq):
    try:
        set_outcome(req.id, req.outcome)
    except ValueError as err:
        raise HTTPException(400, str(err))
    return {"ok": True}


@app.get("/api/patterns")
def api_patterns(client_id: str):
    records = load_by_client(client_id)
    report = build_report(records)
    history = [
        {
            "id": r.id,
            "role": r.jd.title,
            "company": r.jd.company or "—",
            "fit": round(r.fit.overall, 2),
            "stage": STAGE_LABELS.get(r.diagnosis.likely_stage.value, r.diagnosis.likely_stage.value),
            "outcome": r.outcome or "unknown",
            "date": r.created_at.strftime("%Y-%m-%d"),
        }
        for r in records
    ]
    rep = report.model_dump()
    rep["dominant_stage_label"] = STAGE_LABELS.get(report.dominant_stage.value, report.dominant_stage.value)
    return {"report": rep, "history": history}


@app.get("/api/samples")
def api_samples():
    import csv

    out = []
    if LABELS_CSV.exists():
        with open(LABELS_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                jd_path = SAMPLES_DIR / row["jd_file"]
                role = (
                    jd_path.read_text(encoding="utf-8").splitlines()[0].strip()
                    if jd_path.exists()
                    else row["jd_file"]
                )
                out.append(
                    {
                        "key": row["resume_file"].replace("resume_", "").replace(".txt", ""),
                        "role": role,
                        "truth": row["real_outcome"],
                        "resume_file": row["resume_file"],
                        "jd_file": row["jd_file"],
                    }
                )
    return out


@app.get("/api/sample/{key}")
def api_sample(key: str):
    resume = SAMPLES_DIR / f"resume_{key}.txt"
    jd = SAMPLES_DIR / f"jd_{key}.txt"
    if not resume.exists() or not jd.exists():
        raise HTTPException(404, "Sample not found.")
    return {
        "resume_text": resume.read_text(encoding="utf-8"),
        "jd_text": jd.read_text(encoding="utf-8"),
    }


# ------------------------------------------------------------- static frontend -
@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app_web:app", host="0.0.0.0", port=port, reload=True)
