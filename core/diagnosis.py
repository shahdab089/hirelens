"""
Diagnosis layer — owned by Claude.

diagnose(fit, resume, jd) -> Diagnosis

Turns a FitScore into the product's payload: the single brutal, specific reason
this application most likely failed, plus concrete fixes. The LLM picks the
likely_stage from the RejectionStage enum and writes the narrative.
"""
import json

from .llm import chat_json
from .schema import Diagnosis, FitScore, ParsedJD, ParsedResume, RejectionStage

_STAGES = [s.value for s in RejectionStage]


def diagnose(fit: FitScore, resume: ParsedResume, jd: ParsedJD) -> Diagnosis:
    """Diagnose the most likely rejection stage and write the why + fixes."""
    context = {
        "job_title": jd.title,
        "overall_fit": fit.overall,
        "subscores": [
            {"name": s.name, "score": s.score, "rationale": s.rationale}
            for s in fit.subscores
        ],
        "matched_skills": fit.matched_skills,
        "missing_skills": fit.missing_skills,
        "resume_seniority": resume.seniority,
        "resume_years": resume.years_experience,
        "jd_seniority": jd.seniority,
        "jd_hard_requirements": jd.hard_requirements,
    }

    system = (
        "You are a brutally honest career coach for laid-off job seekers. You "
        "diagnose why one application most likely failed. Output only JSON."
    )
    user = (
        "Pick the single most likely rejection stage for this application from "
        f"exactly this list: {_STAGES}.\n"
        "- keyword_ats: filtered by an ATS keyword screen before a human saw it.\n"
        "- seniority_mismatch: wrong level. Use this approximate calibration by "
        "years of experience: junior ~0-2, mid ~3-5, senior ~5-9, staff/lead ~9+. "
        "A candidate satisfies a level if their years are at or above that band's "
        "start — so 6 years SATISFIES a 'senior' role and is NOT a mismatch. ONLY "
        "choose seniority_mismatch when the candidate falls clearly short of the "
        "role's minimum (e.g. role needs 8+/lead and candidate has ~3), or is "
        "drastically overqualified. If years roughly fit the level, do not pick "
        "this — even if the title contains 'Senior'.\n"
        "- skills_gap: missing required hard skills.\n"
        "- domain_mismatch: wrong industry/role domain.\n"
        "- competitive: qualified but likely out-competed.\n"
        "- likely_fine: no obvious flaw — probably volume/luck.\n"
        "Decision rule: if overall_fit is high (>= 0.8) and no sub-score is "
        "clearly low (< 0.5), prefer 'likely_fine' — a strong match was most "
        "likely fine and lost to volume/luck, not a specific flaw.\n\n"
        "Return JSON with this exact shape:\n"
        '{"likely_stage": <one value from the list>, '
        '"headline": <one brutal, specific sentence>, '
        '"explanation": <2-4 sentences naming the concrete reason>, '
        '"top_fixes": [<2-4 concrete, actionable fixes>]}\n\n'
        f"Application analysis:\n{json.dumps(context, indent=2)}"
    )

    data = chat_json(system, user)

    raw_stage = str(data.get("likely_stage", "")).strip()
    try:
        likely_stage = RejectionStage(raw_stage)
    except ValueError:
        # Fall back to the weakest sub-score's stage if the model returned junk.
        likely_stage = _fallback_stage(fit)

    return Diagnosis(
        likely_stage=likely_stage,
        headline=str(data.get("headline", "")).strip() or "Application likely filtered out.",
        explanation=str(data.get("explanation", "")).strip(),
        top_fixes=[str(f) for f in data.get("top_fixes", []) if str(f).strip()],
    )


def _fallback_stage(fit: FitScore) -> RejectionStage:
    """Map the lowest sub-score to a stage when the LLM gives an invalid one."""
    by_name = {s.name: s.score for s in fit.subscores}
    if not by_name:
        return RejectionStage.likely_fine
    weakest = min(by_name, key=by_name.get)
    mapping = {
        "skills": RejectionStage.skills_gap,
        "seniority": RejectionStage.seniority_mismatch,
        "keywords_ats": RejectionStage.keyword_ats,
        "domain": RejectionStage.domain_mismatch,
    }
    # If nothing is clearly weak, call it competitive/fine.
    if by_name[weakest] >= 0.6:
        return RejectionStage.likely_fine
    return mapping.get(weakest, RejectionStage.competitive)
