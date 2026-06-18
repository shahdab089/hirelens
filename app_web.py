"""
Application Autopsy — FastAPI backend.

Thin web layer over the existing core logic (parsing/scoring/diagnosis/storage/
analytics). Serves the custom frontend in web/static and exposes a small JSON API.
The Groq API key lives only on the server (env var GROQ_API_KEY) — clients never
send it. Per-visitor history is scoped by an anonymous client_id.
"""
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from analytics.patterns import build_report
from core.diagnosis import diagnose
from core.parsing import extract_text, parse_jd, parse_resume
from core.schema import ApplicationRecord
from core.scoring import score
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


# --------------------------------------------------------------------- helpers -
def _require_key():
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="The analysis service isn't configured yet. Please try again later.",
        )


# ------------------------------------------------------------------- API routes -
@app.get("/api/health")
def health():
    return {"ok": True, "key_configured": bool(os.environ.get("GROQ_API_KEY"))}


@app.post("/api/analyze")
def api_analyze(req: AnalyzeReq):
    _require_key()
    if not req.resume_text.strip() or not req.jd_text.strip():
        raise HTTPException(400, "Please provide both a résumé and a job description.")
    try:
        resume = parse_resume(req.resume_text)
        jd = parse_jd(req.jd_text)
        fit = score(resume, jd)
        diag = diagnose(fit, resume, jd)
    except ValueError as err:
        raise HTTPException(400, f"Could not parse the inputs: {err}")
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
        # full serialized record so the client can log it without re-running.
        "record": record.model_dump(mode="json"),
    }


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
