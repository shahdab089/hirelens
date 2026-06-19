"""
Scoring layer — owned by Claude.

score(resume, jd) -> FitScore

Strategy: the LLM judges the four sub-scores, the overall fit, and which JD
requirements the candidate matches vs. misses — done *semantically*, so e.g.
"Python + scikit-learn" counts toward a "Machine Learning" requirement and a
broad category like "SQL & Data Manipulation" is satisfied by "SQL". A
deterministic keyword-rescue pass then fixes obvious LLM false negatives
(e.g. SQL listed but flagged missing; degree abbreviation not recognised).
"""
import json
import re

from .llm import chat_json
from .schema import Diagnosis, FitScore, ParsedJD, ParsedResume, RejectionStage, SubScore

SUBSCORE_NAMES = ["skills", "seniority", "keywords_ats", "domain"]

# Words too generic to serve as evidence that a requirement is satisfied.
_STOP = frozenset({
    "and", "the", "with", "for", "has", "are", "was", "that", "this",
    "will", "have", "from", "they", "your", "you", "can", "not", "all",
    "its", "our", "but", "any", "may", "also", "been", "both", "each",
    "able", "must", "more", "into", "used", "use", "using", "such", "via",
    "experience", "skills", "skill", "ability", "proven", "strong",
    "knowledge", "equivalent", "related", "including", "tools", "tool",
    "methods", "method", "practices", "practice", "applying", "applied",
    "background", "relevant", "environments", "environment",
})

# Recognises degree abbreviations (B.Tech, B.S., M.S., Ph.D, …)
_DEGREE_PAT = re.compile(
    r"\b(B\.?Tech|B\.?E\b|B\.?S\.?\b|B\.?Sc\b|B\.?A\b|"
    r"M\.?S\.?\b|M\.?Sc\b|Ph\.?D|M\.?B\.?A|bachelor|master|degree)\b",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    return s.strip().lower()


def _skill_overlap(resume: ParsedResume, jd: ParsedJD) -> tuple[list[str], list[str]]:
    """(matched, missing) from the extracted skills lists — deterministic fallback."""
    resume_skills = {_norm(s) for s in resume.skills}
    matched = [s for s in jd.required_skills if _norm(s) in resume_skills]
    missing = [s for s in jd.required_skills if _norm(s) not in resume_skills]
    return matched, missing


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _cap(text: str, n: int = 2200) -> str:
    """Trim very long text so prompts stay bounded (8B model has a 6k tokens/min cap)."""
    text = text or ""
    return text if len(text) <= n else text[:n] + " …[truncated]"


def _keyword_rescue(
    missing: list[str],
    matched: list[str],
    resume_raw: str,
) -> tuple[list[str], list[str]]:
    """
    Move items from missing→matched when their specific keywords appear verbatim
    (whole-word) in the full résumé text.  Fixes the most common LLM false
    negatives: SQL listed but called 'Advanced SQL', Snowflake/BigQuery in bullets
    but not in extracted skills, B.Tech/M.S. not recognised as a bachelor's degree.
    """
    resume_lower = (resume_raw or "").lower()
    has_degree = bool(_DEGREE_PAT.search(resume_raw or ""))

    new_missing: list[str] = []
    new_matched: list[str] = list(matched)

    for req in missing:
        req_lower = req.lower()

        # Special case: any degree abbreviation in the résumé satisfies a degree req.
        if re.search(r"\b(bachelor|degree|b\.tech|b\.s\b|b\.e\b)", req_lower) and has_degree:
            new_matched.append(req)
            continue

        # Extract specific tokens (≥3 chars, not generic stop-words).
        tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9+#\.]{2,}\b", req)
        keywords = [t.lower() for t in tokens if t.lower() not in _STOP]

        if not keywords:
            new_missing.append(req)
            continue

        # If ANY specific keyword appears as a whole word in the résumé, it's matched.
        if any(re.search(r"\b" + re.escape(kw) + r"\b", resume_lower) for kw in keywords):
            new_matched.append(req)
        else:
            new_missing.append(req)

    # Deduplicate matched list (rescue may have added duplicates).
    seen: set[str] = set()
    deduped: list[str] = []
    for item in new_matched:
        key = item.lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return new_missing, deduped


def score(resume: ParsedResume, jd: ParsedJD) -> FitScore:
    """Score how well a resume fits a job description, via Groq + keyword rescue."""
    matched_skills, missing_skills = _skill_overlap(resume, jd)

    context = {
        "resume": {
            "full_text": _cap(resume.raw_text),
            "extracted_skills": resume.skills,
            "years_experience": resume.years_experience,
            "seniority": resume.seniority,
            "titles": resume.titles,
        },
        "job": {
            "title": jd.title,
            "full_text": _cap(jd.raw_text),
            "required_skills": jd.required_skills,
            "nice_to_have_skills": jd.nice_to_have_skills,
            "seniority": jd.seniority,
            "hard_requirements": jd.hard_requirements,
        },
    }

    system = (
        "You are a brutally honest technical recruiter. You score how well a "
        "candidate fits a role on four dimensions and output only JSON."
    )
    user = (
        "Score this application on a 0.0-1.0 scale for each of exactly these "
        f"dimensions: {SUBSCORE_NAMES}.\n\n"
        "CRITICAL — judge everything against the candidate's FULL résumé text "
        "(skills, work experience, education, AND project descriptions), not just "
        "the extracted skills list. A requirement is MATCHED if it appears "
        "anywhere in the résumé, by meaning. Concrete matching rules:\n"
        "  • 'Bachelor's degree' is satisfied by B.Tech, B.E., B.S., B.Sc., B.A.,\n"
        "    or any master's degree (M.S./M.Sc. implies a prior bachelor's was held).\n"
        "  • 'Bachelor's degree in CS, Economics, Statistics, or a related quantitative\n"
        "    field' is satisfied by any engineering or data-science degree.\n"
        "  • SQL listed as a skill with professional experience satisfies 'Advanced SQL'.\n"
        "  • Snowflake or BigQuery mentioned ANYWHERE in skills or job bullets satisfies\n"
        "    'experience with modern data warehouses (Snowflake, BigQuery, Redshift)'.\n"
        "  • Python + scikit-learn satisfies 'Machine Learning'; SQL satisfies 'SQL & Data Manipulation'.\n"
        "  • A tool or skill in a project or work bullet counts even if not in the skills list.\n"
        "Put a requirement in missing_skills ONLY if it is genuinely absent from "
        "the entire résumé. When unsure, lean toward matched.\n\n"
        "Dimensions:\n"
        "- skills: do they have the required hard skills (judged against the full résumé, by meaning)?\n"
        "- seniority: calibrate by years of experience — junior ~0-2, mid ~3-5, "
        "senior ~5-9, staff/lead ~9+. A candidate meets a level if their years "
        "are at or above that band's start (so 6 years satisfies a 'senior' "
        "role). Score 0.8+ when years fit the role's level or the stated 'X+ "
        "years' bar; only score low when there is a clear gap.\n"
        "- keywords_ats: would an ATS keyword screen pass their resume for this JD?\n"
        "- domain: is their industry/role domain a match?\n\n"
        "Also list which JD requirements the candidate clearly DOES satisfy "
        "(matched_skills) vs. clearly LACKS (missing_skills), judged against the full résumé.\n\n"
        "Return JSON with this exact shape:\n"
        '{"overall": <float 0-1>, '
        '"subscores": [{"name": <one of the four>, "score": <float 0-1>, '
        '"rationale": <one specific sentence>}, ...], '
        '"matched_skills": [<JD requirements the candidate satisfies>], '
        '"missing_skills": [<JD requirements the candidate lacks>]}\n'
        "Include all four subscores. 'overall' should reflect a weighted view "
        "where skills and seniority matter most. Be specific in rationales.\n\n"
        f"Application data:\n{json.dumps(context, indent=2)}"
    )

    data = chat_json(system, user)

    subscores: list[SubScore] = []
    seen = set()
    for item in data.get("subscores", []):
        name = item.get("name")
        if name in SUBSCORE_NAMES and name not in seen:
            seen.add(name)
            subscores.append(
                SubScore(
                    name=name,
                    score=_clamp01(item.get("score", 0.0)),
                    rationale=str(item.get("rationale", "")),
                )
            )
    for name in SUBSCORE_NAMES:
        if name not in seen:
            subscores.append(SubScore(name=name, score=0.0, rationale="Not assessed."))

    overall = data.get("overall")
    if overall is None:
        overall = sum(s.score for s in subscores) / len(subscores)

    # Prefer the model's semantic lists; fall back to deterministic exact-match only if absent.
    raw_matched = [str(s) for s in data.get("matched_skills", []) if str(s).strip()]
    raw_missing = [str(s) for s in data.get("missing_skills", []) if str(s).strip()]
    raw_matched = raw_matched or matched_skills
    raw_missing = raw_missing or missing_skills

    # Post-correction: move items the LLM wrongly flagged as missing back to matched.
    final_missing, final_matched = _keyword_rescue(raw_missing, raw_matched, resume.raw_text or "")

    return FitScore(
        overall=_clamp01(overall),
        subscores=subscores,
        matched_skills=final_matched,
        missing_skills=final_missing,
    )


_ACQUIRE_VERBS = (
    "gain experience", "gain hands-on", "learn", "acquire", "take a course",
    "take courses", "get experience", "build experience", "develop experience",
    "familiarize", "upskill", "improve your skills in", "improve skills in",
    "get certified", "certification in", "pursue training", "study",
)


def _clean_fixes(fixes: list[str], resume: ParsedResume) -> list[str]:
    """Drop fixes that tell the candidate to acquire a skill they already list."""
    skills = [s.lower() for s in resume.skills if len(s) >= 2]
    out: list[str] = []
    for fix in fixes:
        low = fix.lower()
        if any(v in low for v in _ACQUIRE_VERBS) and any(
            re.search(r"\b" + re.escape(sk) + r"\b", low) for sk in skills
        ):
            continue  # recommends acquiring something already on the résumé
        out.append(fix)
    if not out:  # never leave the user with zero guidance
        out = ["Re-order and quantify your résumé so the role's key requirements appear first."]
    return out


def _build_fit(data: dict, resume: ParsedResume, fallback_missing: list[str], fallback_matched: list[str]) -> FitScore:
    """Build a FitScore from a model response dict (shared by score + combined)."""
    subscores: list[SubScore] = []
    seen = set()
    for item in data.get("subscores", []):
        name = item.get("name")
        if name in SUBSCORE_NAMES and name not in seen:
            seen.add(name)
            subscores.append(
                SubScore(
                    name=name,
                    score=_clamp01(item.get("score", 0.0)),
                    rationale=str(item.get("rationale", "")),
                )
            )
    for name in SUBSCORE_NAMES:
        if name not in seen:
            subscores.append(SubScore(name=name, score=0.0, rationale="Not assessed."))

    overall = data.get("overall")
    if overall is None:
        overall = sum(s.score for s in subscores) / len(subscores)

    raw_matched = [str(s) for s in data.get("matched_skills", []) if str(s).strip()] or fallback_matched
    raw_missing = [str(s) for s in data.get("missing_skills", []) if str(s).strip()] or fallback_missing
    final_missing, final_matched = _keyword_rescue(raw_missing, raw_matched, resume.raw_text or "")

    return FitScore(
        overall=_clamp01(overall),
        subscores=subscores,
        matched_skills=final_matched,
        missing_skills=final_missing,
    )


def score_and_diagnose(resume: ParsedResume, jd: ParsedJD) -> tuple[FitScore, Diagnosis]:
    """
    Score fit AND diagnose the likely rejection in a SINGLE Groq call.

    Merges what used to be two separate LLM round-trips (score + diagnose) into
    one, halving token use and latency so the pipeline stays under the free-tier
    per-minute limit. Used by the web app; the standalone score()/diagnose()
    remain for the eval harness.
    """
    from .diagnosis import _STAGES, _fallback_stage

    matched_skills, missing_skills = _skill_overlap(resume, jd)

    context = {
        "resume": {
            "full_text": _cap(resume.raw_text),
            "extracted_skills": resume.skills,
            "years_experience": resume.years_experience,
            "seniority": resume.seniority,
            "titles": resume.titles,
        },
        "job": {
            "title": jd.title,
            "full_text": _cap(jd.raw_text),
            "required_skills": jd.required_skills,
            "nice_to_have_skills": jd.nice_to_have_skills,
            "seniority": jd.seniority,
            "hard_requirements": jd.hard_requirements,
        },
    }

    system = (
        "You are a brutally honest technical recruiter AND career coach. You score "
        "how well a candidate fits a role, then diagnose why the application most "
        "likely failed. Output only JSON."
    )
    user = (
        "Do TWO things for this application and return both in one JSON object.\n\n"
        "PART 1 — SCORE each of exactly these dimensions 0.0-1.0: "
        f"{SUBSCORE_NAMES}.\n"
        "CRITICAL — judge everything against the candidate's FULL résumé text "
        "(skills, work experience, education, AND project descriptions), not just "
        "the extracted skills list. A requirement is MATCHED if it appears anywhere "
        "in the résumé, by meaning. Concrete rules:\n"
        "  • 'Bachelor's degree' is satisfied by B.Tech, B.E., B.S., B.Sc., B.A.,\n"
        "    or any master's degree (a master's implies a prior bachelor's).\n"
        "  • 'Bachelor's in CS, Economics, Statistics or related quantitative field'\n"
        "    is satisfied by any engineering or data-science degree.\n"
        "  • SQL listed as a skill satisfies 'Advanced SQL'.\n"
        "  • Snowflake or BigQuery mentioned anywhere satisfies 'modern data warehouses'.\n"
        "  • A tool named in a project or job bullet counts even if not in the skills list.\n"
        "Put a requirement in missing_skills ONLY if genuinely absent. When unsure, lean matched.\n"
        "Seniority calibration: junior ~0-2, mid ~3-5, senior ~5-9, staff/lead ~9+; a "
        "candidate meets a level if their years are at/above that band's start (6 years "
        "satisfies 'senior'). Score 0.8+ when years fit; only score low on a clear gap.\n\n"
        "PART 2 — DIAGNOSE the single most likely rejection stage from exactly this "
        f"list: {_STAGES}.\n"
        "- keyword_ats: filtered by an ATS keyword screen.\n"
        "- seniority_mismatch: ONLY when years fall clearly short of the role's minimum "
        "(or drastically overqualified). 6 years is NOT a mismatch for a senior role.\n"
        "- skills_gap: missing required hard skills.\n"
        "- domain_mismatch: wrong industry/role domain.\n"
        "- competitive: qualified but likely out-competed.\n"
        "- likely_fine: no obvious flaw — probably volume/luck.\n"
        "Decision rule: if overall_fit >= 0.8 and no sub-score < 0.5, prefer 'likely_fine'.\n"
        "CRITICAL for top_fixes and explanation: NEVER tell the candidate to gain, "
        "learn, or get certified in any skill or tool that already appears in their "
        "résumé or matched_skills (e.g. if Power BI and Tableau are listed, do not "
        "suggest learning them). Base fixes only on genuine gaps (missing_skills) and "
        "on positioning/presentation. Do NOT state or assume the candidate's industry "
        "or employer unless it is explicitly in their résumé.\n\n"
        "Return JSON with this EXACT shape:\n"
        '{"overall": <float 0-1>, '
        '"subscores": [{"name": <one of the four>, "score": <float 0-1>, "rationale": <one sentence>}, ...], '
        '"matched_skills": [<JD requirements satisfied>], '
        '"missing_skills": [<JD requirements lacked>], '
        '"likely_stage": <one value from the stage list>, '
        '"headline": <one brutal, specific sentence>, '
        '"explanation": <2-4 sentences naming the concrete reason>, '
        '"top_fixes": [<2-4 concrete, actionable fixes>]}\n\n'
        f"Application data:\n{json.dumps(context, indent=2)}"
    )

    data = chat_json(system, user)

    fit = _build_fit(data, resume, missing_skills, matched_skills)

    raw_stage = str(data.get("likely_stage", "")).strip()
    try:
        likely_stage = RejectionStage(raw_stage)
    except ValueError:
        likely_stage = _fallback_stage(fit)

    diagnosis = Diagnosis(
        likely_stage=likely_stage,
        headline=str(data.get("headline", "")).strip() or "Application likely filtered out.",
        explanation=str(data.get("explanation", "")).strip(),
        top_fixes=_clean_fixes([str(f) for f in data.get("top_fixes", []) if str(f).strip()], resume),
    )

    return fit, diagnosis
