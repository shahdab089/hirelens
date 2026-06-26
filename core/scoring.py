"""
Scoring layer.

score_and_diagnose(resume, jd) -> (FitScore, Diagnosis)

Strategy: Claude Haiku 3.5 judges the four sub-scores, overall fit, and which
JD requirements are matched vs missing -- done semantically. Two deterministic
post-correction passes then fix common LLM false-negatives before the user
ever sees the result:

  _keyword_rescue  -- moves items from missing->matched when their keywords
                      appear verbatim (whole-word) in the full resume text.
  _clean_fixes     -- drops fixes that tell the candidate to learn a skill
                      they already list.

Missing skills are returned ranked by JD relevance (most critical first).
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

# ── BUG FIX 2: Expanded degree pattern ───────────────────────────────────────
# Now catches: "Bachelor of Science", "B.S. in CS", "4-year degree",
# "undergraduate degree", "university degree", "associate degree", etc.
_DEGREE_PAT = re.compile(
    r"\b("
    r"B\.?Tech|B\.?E\b|B\.?S\.?\b|B\.?Sc\b|B\.?A\b|"
    r"M\.?S\.?\b|M\.?Sc\b|Ph\.?D|M\.?B\.?A|"
    r"bachelor(?:\'?s)?|master(?:\'?s)?|"
    r"undergraduate|postgraduate|"
    r"4[\-\s]year\s+degree|four[\-\s]year\s+degree|"
    r"university\s+degree|college\s+degree|"
    r"associate(?:\'?s)?\s+degree|"
    r"degree\s+in|honours\s+degree"
    r")\b",
    re.IGNORECASE,
)

# Pattern to detect a JD requirement is asking for a degree.
_DEGREE_REQ_PAT = re.compile(
    r"\b(bachelor|degree|b\.tech|b\.s\b|b\.e\b|undergraduate|4.year|college|university)\b",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    return s.strip().lower()


def _skill_overlap(resume: ParsedResume, jd: ParsedJD) -> tuple[list[str], list[str]]:
    """(matched, missing) from extracted skill lists -- deterministic fallback only."""
    resume_skills = {_norm(s) for s in resume.skills}
    matched = [s for s in jd.required_skills if _norm(s) in resume_skills]
    missing = [s for s in jd.required_skills if _norm(s) not in resume_skills]
    return matched, missing


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _cap(text: str, n: int = 4000) -> str:
    """Trim very long text so prompts stay bounded.
    BUG FIX 1: Default raised from 2200 -> 4000 chars so education sections
    near the bottom of a resume are not silently truncated before the LLM sees them.
    """
    text = text or ""
    return text if len(text) <= n else text[:n] + " ...[truncated]"


def _keyword_rescue(
    missing: list[str],
    matched: list[str],
    resume_raw: str,
) -> tuple[list[str], list[str]]:
    """
    Move items from missing->matched when their specific keywords appear verbatim
    (whole-word) in the full resume text.

    BUG FIX 2: Expanded degree pattern now catches "Bachelor of Science",
    "4-year degree", "undergraduate degree", "university degree", etc.
    """
    resume_lower = (resume_raw or "").lower()
    has_degree = bool(_DEGREE_PAT.search(resume_raw or ""))

    new_missing: list[str] = []
    new_matched: list[str] = list(matched)

    for req in missing:
        req_lower = req.lower()

        # Special case: any recognised degree in the resume satisfies a degree req.
        if _DEGREE_REQ_PAT.search(req_lower) and has_degree:
            new_matched.append(req)
            continue

        # Extract specific tokens (>=3 chars, not generic stop-words).
        tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9+#\.]{2,}\b", req)
        keywords = [t.lower() for t in tokens if t.lower() not in _STOP]

        if not keywords:
            new_missing.append(req)
            continue

        # If ANY specific keyword appears as a whole word in the resume -> matched.
        if any(re.search(r"\b" + re.escape(kw) + r"\b", resume_lower) for kw in keywords):
            new_matched.append(req)
        else:
            new_missing.append(req)

    # Deduplicate matched list.
    seen: set[str] = set()
    deduped: list[str] = []
    for item in new_matched:
        key = item.lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return new_missing, deduped


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
            continue
        out.append(fix)
    if not out:
        out = ["Re-order and quantify your resume so the role key requirements appear first."]
    return out


def _build_fit(
    data: dict,
    resume: ParsedResume,
    fallback_missing: list[str],
    fallback_matched: list[str],
) -> FitScore:
    """Build a FitScore from a model response dict."""
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
    Score fit AND diagnose the likely rejection in a SINGLE LLM call.
    Uses Claude Haiku 3.5 (primary) with Groq 70B fallback.

    Key improvements over the original:
    - Resume text cap raised 2200->4000 chars (BUG FIX 1)
    - Projects field included as dedicated context (BUG FIX 3)
    - Expanded degree rescue patterns (BUG FIX 2)
    - missing_skills returned ranked by JD relevance (FEATURE)
    """
    from .diagnosis import _STAGES, _fallback_stage

    matched_skills, missing_skills = _skill_overlap(resume, jd)

    # BUG FIX 1: cap raised to 4000 so education (often at bottom) is not cut off.
    # BUG FIX 3: projects included as a dedicated field so skills listed only in
    #            project bullets are explicitly surfaced to the LLM.
    projects_text = ""
    if hasattr(resume, "projects") and resume.projects:
        projects_text = "; ".join(resume.projects)

    context = {
        "resume": {
            "full_text": _cap(resume.raw_text, n=4000),
            "extracted_skills": resume.skills,
            "projects": projects_text or "(see full_text)",
            "years_experience": resume.years_experience,
            "seniority": resume.seniority,
            "titles": resume.titles,
        },
        "job": {
            "title": jd.title,
            "full_text": _cap(jd.raw_text, n=4000),
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
        "PART 1 -- SCORE each of exactly these dimensions 0.0-1.0: "
        f"{SUBSCORE_NAMES}.\n"
        "CRITICAL -- judge everything against the candidate FULL resume text "
        "(skills, work experience, education, AND project descriptions), not just "
        "the extracted skills list. A requirement is MATCHED if it appears anywhere "
        "in the resume, by meaning. Concrete rules:\n"
        "  * 'Bachelor degree' is satisfied by B.Tech, B.E., B.S., B.Sc., B.A.,\n"
        "    Bachelor of Science/Arts/Engineering, or any master degree.\n"
        "  * '4-year degree' / 'university degree' / 'undergraduate degree' is\n"
        "    satisfied by any bachelor or master degree.\n"
        "  * 'Bachelor in CS, Economics, Statistics or related quantitative field'\n"
        "    is satisfied by any engineering or data-science degree.\n"
        "  * SQL listed as a skill satisfies 'Advanced SQL'.\n"
        "  * Snowflake or BigQuery mentioned anywhere satisfies 'modern data warehouses'.\n"
        "  * A tool or skill mentioned in a PROJECT bullet counts, even if absent\n"
        "    from the top-level skills list. Check the 'projects' field too.\n"
        "  * Skills built specifically for this role (mentioned in a project built\n"
        "    to match the JD) count as relevant domain experience.\n"
        "Put a requirement in missing_skills ONLY if genuinely absent from the\n"
        "entire resume (including projects). When unsure, lean toward matched.\n\n"
        "Dimensions:\n"
        "- skills: do they have the required hard skills (judged against full resume)?\n"
        "- seniority: junior ~0-2 yrs, mid ~3-5, senior ~5-9, staff/lead ~9+.\n"
        "  Score 0.8+ when years fit the role level or stated 'X+ years' bar.\n"
        "  Only score low when there is a clear gap.\n"
        "- keywords_ats: would an ATS keyword screen pass their resume for this JD?\n"
        "- domain: is their industry/role domain a match?\n\n"
        "FEATURE -- RANK missing_skills by how critical each gap is to THIS role\n"
        "(most critical / hardest-to-ignore gap first, nice-to-have last).\n\n"
        "PART 2 -- DIAGNOSE the single most likely rejection stage from exactly:\n"
        f"{_STAGES}.\n"
        "- keyword_ats: filtered by an ATS keyword screen.\n"
        "- seniority_mismatch: ONLY when years fall clearly short of role minimum.\n"
        "- skills_gap: missing required hard skills.\n"
        "- domain_mismatch: wrong industry/role domain.\n"
        "- competitive: qualified but likely out-competed.\n"
        "- likely_fine: no obvious flaw -- probably volume/luck.\n"
        "Decision rule: if overall_fit >= 0.8 and no sub-score < 0.5, prefer 'likely_fine'.\n"
        "CRITICAL for top_fixes: NEVER tell the candidate to gain, learn, or get\n"
        "certified in any skill already in their resume or matched_skills.\n"
        "Base fixes only on genuine gaps and positioning/presentation.\n\n"
        "Return JSON with this EXACT shape:\n"
        '{"overall": <float 0-1>, '
        '"subscores": [{"name": <one of the four>, "score": <float 0-1>, "rationale": <one sentence>}, ...], '
        '"matched_skills": [<JD requirements satisfied>], '
        '"missing_skills": [<JD requirements lacked, ranked most-critical first>], '
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


def score(resume: ParsedResume, jd: ParsedJD) -> FitScore:
    """Standalone scorer (used by eval harness). Calls score_and_diagnose internally."""
    fit, _ = score_and_diagnose(resume, jd)
    return fit
