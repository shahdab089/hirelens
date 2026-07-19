"""
Application Autopsy — FastAPI backend.

Thin web layer over the existing core logic (parsing/scoring/diagnosis/storage/
analytics). Serves the custom frontend in web/static and exposes a small JSON API.
The Groq API key lives only on the server (env var GROQ_API_KEY) — clients never
send it. Per-visitor history is scoped by an anonymous client_id.
"""
import hashlib
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
from core.schema import ApplicationRecord, Diagnosis, FitScore, ParsedJD, ParsedResume
from core.scoring import score_and_diagnose
from storage import cache_get_analysis, cache_put_analysis, load_by_client, save, set_outcome

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

# Bump this whenever the parsing/scoring/diagnosis PROMPTS change, so the result
# cache (keyed on it) is invalidated and the new prompts take effect. The active
# model IDs are folded into the key automatically below.
ANALYSIS_PIPELINE_VERSION = "v2"  # bumped: 4-step ATS audit scoring


def _analysis_cache_key(resume_text: str, jd_text: str) -> str:
    """Content-addressed key for an analysis: identical inputs + same pipeline
    (prompt version + model IDs) → same key → reuse the stored result, no LLM call."""
    pipeline = "|".join((
        ANALYSIS_PIPELINE_VERSION,
        os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        os.environ.get("GROQ_FALLBACK_MODEL", "llama-3.3-70b-versatile"),
    ))
    h = hashlib.sha256()
    for part in (pipeline, resume_text.strip(), jd_text.strip()):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# ── In-memory cache for text-generation endpoints (optimize + cover letter) ──
# Keyed on SHA-256 of inputs; capped at 256 entries (oldest evicted first).
# Survives within a single server process — free on cache hit, no LLM call.
_TEXT_CACHE: dict[str, str] = {}
_TEXT_CACHE_MAX = 256


def _text_cache_get(key: str) -> str | None:
    return _TEXT_CACHE.get(key)


def _text_cache_put(key: str, value: str) -> None:
    if len(_TEXT_CACHE) >= _TEXT_CACHE_MAX:
        # evict oldest entry
        oldest = next(iter(_TEXT_CACHE))
        del _TEXT_CACHE[oldest]
    _TEXT_CACHE[key] = value


def _text_cache_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


SAMPLE_CACHE_FILE = BASE / "data" / "sample_cache.json"


def _seed_sample_cache() -> None:
    """Load precomputed sample analyses into the (ephemeral) result cache on boot so
    the built-in 'Try a sample' demos never cost an LLM call — even after a redeploy
    wipes the on-disk cache. No-op if the file is absent or unreadable; seeding must
    never break startup. Regenerate the file via scripts/precompute_sample_cache.py."""
    try:
        if not SAMPLE_CACHE_FILE.exists():
            return
        import json
        entries = json.loads(SAMPLE_CACHE_FILE.read_text(encoding="utf-8"))
        for cache_key, payload in entries.items():
            if cache_get_analysis(cache_key) is None:
                cache_put_analysis(cache_key, payload)
    except Exception:  # noqa: BLE001 — never let seeding break startup
        pass


app = FastAPI(title="Hirelens", docs_url="/api/docs")
_seed_sample_cache()


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


class OptimizeReq(BaseModel):
    resume_text: str
    jd_text: str
    missing_skills: list[str] = []
    top_fixes: list[str] = []


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
    if not req.resume_text.strip() or not req.jd_text.strip():
        raise HTTPException(400, "Please provide both a résumé and a job description.")

    # Reuse a prior result for an identical résumé/JD pair — no LLM call (and no key
    # required) on a cache hit, which also makes the pre-seeded samples free to serve.
    cache_key = _analysis_cache_key(req.resume_text, req.jd_text)
    cached = cache_get_analysis(cache_key)
    if cached is not None:
        resume = ParsedResume.model_validate(cached["resume"])
        jd = ParsedJD.model_validate(cached["jd"])
        fit = FitScore.model_validate(cached["fit"])
        diag = Diagnosis.model_validate(cached["diagnosis"])
    else:
        _require_key()
        try:
            resume, jd = parse_both(req.resume_text, req.jd_text)
            fit, diag = score_and_diagnose(resume, jd)
        except ValueError as err:
            raise HTTPException(400, f"Could not parse the inputs: {err}")
        except GroqRateLimit:
            raise HTTPException(429, "We're experiencing high demand right now — please wait a minute and try again.")
        except Exception as err:  # noqa: BLE001
            raise HTTPException(500, f"Analysis failed: {err}")
        cache_put_analysis(cache_key, {
            "resume": resume.model_dump(mode="json"),
            "jd": jd.model_dump(mode="json"),
            "fit": fit.model_dump(mode="json"),
            "diagnosis": diag.model_dump(mode="json"),
        })

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


_OPTIMIZE_SYSTEM = """\
You are the world's most effective ATS optimization specialist. You know the exact parsing \
logic of Greenhouse, Workday, Lever, iCIMS, Taleo, and SuccessFactors and you use that \
knowledge to rewrite résumés that clear every automated filter before a human ever reads them.

HOW ENTERPRISE ATS SYSTEMS ACTUALLY SCORE RÉSUMÉS:
1. VERBATIM KEYWORD MATCH — the ATS tokenizes the JD, then scans the résumé for those exact \
strings. Paraphrase = 0 points. Mirror = full points. This is the #1 lever.
2. KEYWORD PLACEMENT WEIGHT — ATS parsers weight: Job Title > Professional Summary > Skills \
Section > Job Titles in Experience > Bullets. Put every critical keyword in at least 2 zones.
3. BOTH FORMS REQUIRED — write "Natural Language Processing (NLP)" and later just "NLP". \
Same for "GA4" and "Google Analytics 4", "ML" and "Machine Learning", etc.
4. SECTION HEADER PARSING — ATS expects exact headers: "Professional Summary", \
"Work Experience", "Technical Skills", "Education", "Certifications". Non-standard headers \
get mis-parsed and their content ignored.
5. HARD REQUIREMENT VISIBILITY — if the JD says "5+ years" and the candidate has 6, that \
number must appear early and clearly. ATS reads years from dates; make the math obvious.
6. SKILLS MIRROR — create or expand a Technical Skills section that directly reflects the \
JD's required tools/platforms/methods. ATS scores this section with highest confidence.
7. CONTEXTUAL KEYWORD EMBEDDING — every missing keyword that has evidence in the original \
résumé must be woven into 1-2 specific bullets using the JD's exact phrasing.
8. JOB TITLE ALIGNMENT — if truthfully supported by the candidate's background, open the \
résumé with a title that matches the JD target role (adds weight in Workday/Greenhouse).
9. ATS-SAFE FORMAT — NO tables, NO text boxes, NO columns, NO graphics, NO headers/footers. \
Plain Markdown only. Multi-column formats cause the ATS parser to scramble content.
10. FACT INTEGRITY — NEVER invent a fact. Every metric, tool, company, date, and certification \
must be directly traceable to the original résumé. Only the PRESENTATION changes.

Output ONLY the fully rewritten résumé in clean Markdown. No preamble, no explanation.\
"""


@app.post("/api/optimize")
def api_optimize(req: OptimizeReq):
    _require_key()
    from core.llm import GroqRateLimit, chat_text

    if not req.resume_text.strip() or not req.jd_text.strip():
        raise HTTPException(400, "Please provide both a résumé and a job description.")

    missing_str = (
        ", ".join(req.missing_skills[:14]) if req.missing_skills else "none identified"
    )
    fixes_str = (
        "\n".join(f"- {f}" for f in req.top_fixes[:5])
        if req.top_fixes
        else "- Improve keyword coverage and quantify impact statements"
    )

    user_prompt = (
        f"ORIGINAL RÉSUMÉ:\n{req.resume_text.strip()[:4200]}\n\n"
        f"JOB DESCRIPTION:\n{req.jd_text.strip()[:3000]}\n\n"
        f"MISSING SKILLS/REQUIREMENTS (embed where evidence supports it):\n{missing_str}\n\n"
        f"TOP FIXES TO APPLY:\n{fixes_str}\n\n"
        "Rewrite the résumé now. Output ONLY the Markdown résumé."
    )

    # Cache key covers every input that affects the output
    ck = _text_cache_key(
        "optimize",
        req.resume_text.strip()[:4200],
        req.jd_text.strip()[:3000],
        missing_str,
        fixes_str,
    )
    cached_opt = _text_cache_get(ck)
    if cached_opt is not None:
        return {"optimized_resume": cached_opt, "cached": True}

    try:
        # prefer_groq=True: Groq (llama-3.3-70b) handles long text rewrites well
        # and output tokens are ~85% cheaper than Claude Haiku — no quality loss.
        # Anthropic is kept as automatic fallback if Groq is rate-limited.
        optimized = chat_text(
            _OPTIMIZE_SYSTEM, user_prompt, max_tokens=5000, prefer_groq=True
        )
    except GroqRateLimit:
        raise HTTPException(
            429, "High demand right now — please wait a minute and try again."
        )
    except Exception as err:  # noqa: BLE001
        raise HTTPException(500, f"Optimization failed: {err}")

    _text_cache_put(ck, optimized)
    return {"optimized_resume": optimized}


# ─────────────────────────────────────────── /api/cover-letter ───────────────

class CoverLetterReq(BaseModel):
    resume_text: str
    jd_text: str
    matched_skills: list[str] = []
    missing_skills: list[str] = []


_COVER_LETTER_SYSTEM = """\
You are a master cover letter writer who has helped thousands of candidates land interviews \
at FAANG, McKinsey, and Fortune 500 companies. You write cover letters that clear ATS filters \
AND captivate the hiring manager who reads them after.

RULES YOU NEVER BREAK:
1. NEVER open with "I am writing to apply", "I am excited to apply", or any cliché variant. \
   Open with the candidate's single strongest achievement relevant to this role.
2. Mirror the top 5-6 JD keywords VERBATIM in natural, flowing sentences — ATS scores keyword density.
3. Reference 2-3 specific achievements from the résumé with EXACT metrics (never round or inflate).
4. Exactly 3 paragraphs: (1) hook + why this specific role, (2) proof from 2-3 achievements, \
   (3) domain fit + confident, specific CTA.
5. Maximum 270 words. Every sentence earns its place. No filler.
6. Sound like a sharp, confident professional wrote it — not a generic template.
7. End with: "I'd welcome a conversation to discuss how I can [specific value from the role]."
8. Output ONLY the 3-paragraph letter body. No date, no address, no salutation, no subject line.\
"""


@app.post("/api/cover-letter")
def api_cover_letter(req: CoverLetterReq):
    _require_key()
    from core.llm import GroqRateLimit, chat_text

    if not req.resume_text.strip() or not req.jd_text.strip():
        raise HTTPException(400, "Please provide both a résumé and a job description.")

    matched_str = ", ".join(req.matched_skills[:12]) or "see résumé"
    missing_str = ", ".join(req.missing_skills[:8]) or "none identified"

    user_prompt = (
        f"JOB DESCRIPTION:\n{req.jd_text.strip()[:2500]}\n\n"
        f"CANDIDATE RÉSUMÉ:\n{req.resume_text.strip()[:3500]}\n\n"
        f"ATS KEYWORDS ALREADY MATCHED: {matched_str}\n"
        f"KEYWORDS TO WEAVE IN NATURALLY: {missing_str}\n\n"
        "Write the 3-paragraph cover letter body now. "
        "Open with the candidate's strongest measurable achievement relevant to this role."
    )

    ck = _text_cache_key(
        "cover-letter",
        req.resume_text.strip()[:3500],
        req.jd_text.strip()[:2500],
        matched_str,
        missing_str,
    )
    cached_cl = _text_cache_get(ck)
    if cached_cl is not None:
        return {"cover_letter": cached_cl, "cached": True}

    try:
        # prefer_groq=True: Groq handles creative writing well and is far cheaper
        # on output tokens. Anthropic auto-fallback kept for reliability.
        letter = chat_text(
            _COVER_LETTER_SYSTEM, user_prompt, max_tokens=900, prefer_groq=True
        )
    except GroqRateLimit:
        raise HTTPException(429, "High demand — please wait a moment and try again.")
    except Exception as err:
        raise HTTPException(500, f"Cover letter generation failed: {err}")

    _text_cache_put(ck, letter)
    return {"cover_letter": letter}


# ─────────────────────────────────────────── /api/check-format ───────────────

@app.post("/api/check-format")
async def api_check_format(file: UploadFile = File(...)):
    """Analyse a PDF résumé for ATS-hostile formatting issues using pypdf."""
    import io
    import pypdf

    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported for format checking.")

    content = await file.read()
    issues: list[dict] = []
    passes: list[str] = []

    try:
        reader = pypdf.PdfReader(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(422, f"Could not open PDF: {exc}")

    # ── 1. Encrypted ──────────────────────────────────────────────────────────
    if reader.is_encrypted:
        issues.append({"severity": "error", "code": "encrypted",
            "message": "PDF is password-protected — ATS systems cannot parse encrypted files at all. Export an unlocked PDF from Word or Google Docs."})

    # ── 2. Embedded images ────────────────────────────────────────────────────
    has_images = False
    try:
        res = reader.pages[0].get("/Resources", {})
        xobj = res.get("/XObject", {})
        if hasattr(xobj, "get_object"):
            xobj = xobj.get_object()
        for k in (xobj or {}):
            obj = xobj[k]
            if hasattr(obj, "get_object"):
                obj = obj.get_object()
            if obj.get("/Subtype") == "/Image":
                has_images = True
                break
    except Exception:
        pass

    if has_images:
        issues.append({"severity": "error", "code": "has_images",
            "message": "Embedded images detected — ATS cannot read image content. Remove logos, profile photos, and decorative graphics."})
    else:
        passes.append("No embedded images — all content is machine-readable text")

    # ── 3. Extract text + character positions ─────────────────────────────────
    all_text = ""
    positions: list[dict] = []

    def _record(text, cm, tm, font_dict, font_size):
        if text.strip():
            positions.append({"x": float(tm[4]), "y": float(tm[5])})

    for pg in reader.pages:
        try:
            pg.extract_text(visitor_text=_record)
        except Exception:
            pass
        all_text += (pg.extract_text() or "")

    # ── 4. Text density ───────────────────────────────────────────────────────
    if len(all_text.strip()) < 150:
        issues.append({"severity": "error", "code": "low_text",
            "message": "Very little extractable text found — this PDF may be image-based (scanned). ATS would see a blank document. Export from Word or Google Docs instead."})
    else:
        passes.append("Good text density — ATS can extract your content")

    # ── 5. Multi-column detection (x-coordinate clustering) ───────────────────
    if positions:
        x_vals = [p["x"] for p in positions if p["x"] > 20]
        if x_vals:
            page_w = max(x_vals)
            mid = page_w / 2
            left_n  = sum(1 for x in x_vals if x < mid * 0.55)
            right_n = sum(1 for x in x_vals if x > mid * 1.45)
            if left_n > 8 and right_n > 8 and right_n > left_n * 0.25:
                issues.append({"severity": "error", "code": "multi_column",
                    "message": "Multi-column layout detected — Workday and Taleo parse columns left-to-right then top-to-bottom, scrambling your work history with your skills section. Switch to a single-column format."})
            else:
                passes.append("Single-column layout — ATS will read sections in the correct order")

    # ── 6. Standard section headers ───────────────────────────────────────────
    tl = all_text.lower()
    found_hdrs = [h for h in ["experience", "education", "skills", "summary"] if h in tl]
    if len(found_hdrs) >= 2:
        passes.append(f"Standard section headers found — ATS section parser will categorise correctly")
    else:
        issues.append({"severity": "warning", "code": "missing_headers",
            "message": "Standard section headers not detected. Use exactly: 'Work Experience', 'Technical Skills', 'Education', 'Professional Summary' — ATS parsers depend on these labels."})

    # ── 7. Page count ─────────────────────────────────────────────────────────
    pg_count = len(reader.pages)
    if pg_count > 2:
        issues.append({"severity": "warning", "code": "too_long",
            "message": f"{pg_count} pages — most ATS systems only score page 1 in detail. Keep to 1–2 pages for senior roles."})
    else:
        passes.append(f"{pg_count} page{'s' if pg_count > 1 else ''} — within optimal ATS length")

    error_count = sum(1 for i in issues if i["severity"] == "error")
    warn_count  = sum(1 for i in issues if i["severity"] == "warning")

    if   error_count == 0 and warn_count == 0: verdict, label = "clean",    "ATS-Ready ✓"
    elif error_count == 0:                     verdict, label = "warnings", "Mostly Safe"
    elif error_count <= 2:                     verdict, label = "issues",   "Needs Fixes"
    else:                                      verdict, label = "critical", "High Risk"

    return {"verdict": verdict, "verdict_label": label,
            "issues": issues, "passes": passes,
            "error_count": error_count, "warn_count": warn_count}


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
