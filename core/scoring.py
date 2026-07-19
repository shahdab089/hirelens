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

    # Cap inputs tightly so the combined prompt + JSON response stays within
    # max_tokens. Resume full text is ~3 000 chars; JD full text ~2 500 chars;
    # the structured prompt adds ~1 800 chars. Together the OUTPUT JSON is
    # ~800–1 200 tokens, so 2 800 max_tokens is plenty with headroom.
    context = {
        "resume": {
            "full_text": _cap(resume.raw_text, n=3000),
            "extracted_skills": resume.skills[:40],        # cap list length
            "projects": _cap(projects_text, n=600) or "(see full_text)",
            "years_experience": resume.years_experience,
            "seniority": resume.seniority,
            "titles": resume.titles[:6],
        },
        "job": {
            "title": jd.title,
            "full_text": _cap(jd.raw_text, n=2500),
            "required_skills": jd.required_skills[:30],
            "nice_to_have_skills": jd.nice_to_have_skills[:15],
            "seniority": jd.seniority,
            "hard_requirements": jd.hard_requirements[:20],
        },
    }

    system = (
        "You are an enterprise ATS simulation engine that models exactly how "
        "Greenhouse, Workday, Lever, iCIMS, Taleo, and SuccessFactors parse and "
        "rank résumés. You score with clinical precision — not to encourage the "
        "candidate, but to reflect the actual algorithmic outcome. Output only JSON."
    )
    user = (
        "Run the 4-STEP ATS AUDIT below, then diagnose the rejection stage.\n\n"

        "STEP A — EXTRACT JD REQUIREMENTS into buckets:\n"
        "  hard_tools    : specific platforms/languages/tools required (e.g. 'Snowflake', 'GA4', 'dbt')\n"
        "  hard_skills   : methods/competencies required (e.g. 'A/B testing', 'attribution modeling')\n"
        "  certifications: degrees, certs, licences required or strongly preferred\n"
        "  years_req     : minimum experience duration (e.g. '5+ years')\n"
        "  industry_terms: domain/sector keywords ATS weights heavily (e.g. 'financial institutions')\n"
        "  nice_to_have  : explicitly preferred but not required\n\n"

        "STEP B — FOR EVERY item in hard_tools + hard_skills + certifications, "
        "classify its presence in the FULL résumé text (work history, projects, "
        "skills section, education — everywhere):\n"
        "  EXACT    : the exact term or standard abbreviation appears verbatim\n"
        "  SYNONYM  : a universally accepted equivalent is present\n"
        "             (BigQuery/Redshift → 'cloud data warehouse'; B.Tech → 'Bachelor';\n"
        "              Pandas/NumPy → 'Python data manipulation'; PySpark → 'Spark')\n"
        "  CONTEXT  : the skill is clearly demonstrated in work bullets without naming it\n"
        "  ABSENT   : genuinely not found anywhere — this ALONE goes into missing_skills\n\n"
        "MATCH RULES (get these exactly right):\n"
        "  • Any mention in projects section counts equally to the skills section.\n"
        "  • B.Tech / B.E. / B.S. / B.Sc. / M.S. / M.Sc. satisfy 'Bachelor degree'.\n"
        "  • 'Advanced SQL' is satisfied if SQL appears and the résumé shows complex queries.\n"
        "  • 'Google Analytics (GA4)' is ABSENT unless GA or GA4 appears explicitly — do NOT infer.\n"
        "  • Candidate years_experience ≥ years_req → seniority satisfied; partial credit within 1 yr.\n\n"

        "STEP C — COMPUTE SCORES from match rates (use these exact formulas):\n"
        "  Let N = total items in hard_tools + hard_skills (exclude nice_to_have).\n"
        "  keywords_ats = (exact_count + 0.6 × synonym_count) / N\n"
        "     Greenhouse/Workday weight verbatim matches highest; synonyms score 60%.\n"
        "  skills       = (exact_count + synonym_count + 0.7 × context_count) / N\n"
        "  seniority    : 1.0 if years ≥ required; 0.85 if 1 yr short; "
        "0.65 if 2 yrs short; 0.40 if 3+ yrs short\n"
        "  domain       = (industry_terms matched) / (total industry_terms in JD); min 0.10\n"
        "  overall      = 0.35×keywords_ats + 0.30×skills + 0.20×seniority + 0.15×domain\n"
        "  (clamp all to [0.0, 1.0])\n\n"

        "STEP D — GAP ANALYSIS:\n"
        "  missing_skills = ONLY ABSENT items, ranked most-critical-to-this-role first.\n"
        "  matched_skills = EXACT + SYNONYM + CONTEXT items with brief evidence.\n"
        "  top_fixes: address ABSENT gaps or poor keyword positioning ONLY.\n"
        "  NEVER suggest gaining a skill the résumé already demonstrates.\n\n"

        "DIAGNOSE the single most likely ATS/recruiter rejection stage:\n"
        f"  Choose from: {_STAGES}\n"
        "  keyword_ats        : keywords_ats < 0.60 — ATS auto-rejects before human review.\n"
        "  seniority_mismatch : years clearly below the stated minimum.\n"
        "  skills_gap         : hard skill absences a recruiter would catch.\n"
        "  domain_mismatch    : wrong sector/industry for this role.\n"
        "  competitive        : qualified but likely out-competed.\n"
        "  likely_fine        : overall ≥ 0.80, no sub-score < 0.55 — volume/luck issue.\n\n"

        "Return JSON with EXACTLY this shape:\n"
        '{"overall": <computed float 0-1>, '
        '"subscores": [{"name": <one of ' + str(SUBSCORE_NAMES) + '>, '
        '"score": <float 0-1>, "rationale": <cite specific counts/evidence>}, ...], '
        '"matched_skills": [<requirement — brief evidence>], '
        '"missing_skills": [<absent requirement, most critical first>], '
        '"likely_stage": <one value from stage list>, '
        '"headline": <one blunt sentence stating the primary ATS blocker>, '
        '"explanation": <2-4 sentences with exact evidence — tool names, counts, gaps>, '
        '"top_fixes": [<2-4 ATS-specific actionable fixes tied to real ABSENT items>]}\n\n'
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
