"""
Scoring layer — owned by Claude.

score(resume, jd) -> FitScore

Strategy: the LLM judges the four sub-scores, the overall fit, and which JD
requirements the candidate matches vs. misses — done *semantically*, so e.g.
"Python + scikit-learn" counts toward a "Machine Learning" requirement and a
broad category like "SQL & Data Manipulation" is satisfied by "SQL". A
deterministic exact-match overlap is kept only as a fallback if the model omits
those lists.
"""
import json

from .llm import chat_json
from .schema import FitScore, ParsedJD, ParsedResume, SubScore

SUBSCORE_NAMES = ["skills", "seniority", "keywords_ats", "domain"]


def _norm(s: str) -> str:
    return s.strip().lower()


def _skill_overlap(resume: ParsedResume, jd: ParsedJD) -> tuple[list[str], list[str]]:
    """Return (matched, missing) required skills, comparing case-insensitively."""
    resume_skills = {_norm(s) for s in resume.skills}
    matched = [s for s in jd.required_skills if _norm(s) in resume_skills]
    missing = [s for s in jd.required_skills if _norm(s) not in resume_skills]
    return matched, missing


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def score(resume: ParsedResume, jd: ParsedJD) -> FitScore:
    """Score how well a resume fits a job description, via Groq + deterministic skill diff."""
    matched_skills, missing_skills = _skill_overlap(resume, jd)

    context = {
        "resume": {
            "skills": resume.skills,
            "years_experience": resume.years_experience,
            "seniority": resume.seniority,
            "titles": resume.titles,
        },
        "job": {
            "title": jd.title,
            "required_skills": jd.required_skills,
            "nice_to_have_skills": jd.nice_to_have_skills,
            "seniority": jd.seniority,
            "hard_requirements": jd.hard_requirements,
        },
        "computed_matched_skills": matched_skills,
        "computed_missing_required_skills": missing_skills,
    }

    system = (
        "You are a brutally honest technical recruiter. You score how well a "
        "candidate fits a role on four dimensions and output only JSON."
    )
    user = (
        "Score this application on a 0.0-1.0 scale for each of exactly these "
        f"dimensions: {SUBSCORE_NAMES}.\n"
        "- skills: do they have the required hard skills? Judge by MEANING, not "
        "exact wording — e.g. Python + scikit-learn satisfies 'Machine Learning', "
        "and 'SQL' satisfies a broad 'SQL & Data Manipulation' requirement.\n"
        "- seniority: calibrate by years of experience — junior ~0-2, mid ~3-5, "
        "senior ~5-9, staff/lead ~9+. A candidate meets a level if their years "
        "are at or above that band's start (so 6 years satisfies a 'senior' "
        "role). Score 0.8+ when years fit the role's level or the stated 'X+ "
        "years' bar; only score low when there is a clear gap (e.g. role needs "
        "8+ and candidate has ~3). Do not penalize for an incidental 'senior' in "
        "the body if the years fit.\n"
        "- keywords_ats: would an ATS keyword screen pass their resume for this JD?\n"
        "- domain: is their industry/role domain a match?\n\n"
        "Also decide, semantically, which JD requirements the candidate clearly "
        "DOES satisfy (matched_skills) versus clearly LACKS (missing_skills).\n\n"
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
    # Guarantee all four dimensions exist even if the model dropped one.
    for name in SUBSCORE_NAMES:
        if name not in seen:
            subscores.append(SubScore(name=name, score=0.0, rationale="Not assessed."))

    overall = data.get("overall")
    if overall is None:
        overall = sum(s.score for s in subscores) / len(subscores)

    # Prefer the model's semantic match/miss lists; fall back to the
    # deterministic exact-match overlap only if the model didn't provide them.
    llm_matched = [str(s) for s in data.get("matched_skills", []) if str(s).strip()]
    llm_missing = [str(s) for s in data.get("missing_skills", []) if str(s).strip()]

    return FitScore(
        overall=_clamp01(overall),
        subscores=subscores,
        matched_skills=llm_matched or matched_skills,
        missing_skills=llm_missing or missing_skills,
    )
