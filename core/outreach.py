"""
Outreach drafting — owned by Claude.

draft_outreach(resume, jd, fit) -> OutreachDraft

Generates a concise, personalized cold-outreach email + a LinkedIn connection
note, grounded in the candidate's actual matched strengths and the target role.
"""
import json

from .llm import chat_json
from .schema import FitScore, OutreachDraft, ParsedJD, ParsedResume


def draft_outreach(resume: ParsedResume, jd: ParsedJD, fit: FitScore) -> OutreachDraft:
    context = {
        "role": jd.title,
        "company": jd.company,
        "candidate_seniority": resume.seniority,
        "candidate_years": resume.years_experience,
        "top_matched_skills": fit.matched_skills[:5],
        "overall_fit": fit.overall,
    }

    system = (
        "You write concise, professional cold outreach for job seekers reaching "
        "out to recruiters/hiring managers. Confident but not arrogant, specific, "
        "no fluff. Output only JSON."
    )
    user = (
        "Using the data below, write outreach that references 1-2 concrete "
        "matched strengths and the specific role. Do not invent facts not present "
        "in the data. Sign the email with the placeholder '[Your Name]'.\n\n"
        "Return JSON with exactly:\n"
        '{"email_subject": <short, specific, <70 chars>, '
        '"email_body": <90-140 words, warm, references the role + 1-2 strengths, '
        'ends with a soft ask for a brief chat, then a newline and "[Your Name]">, '
        '"linkedin_note": <under 300 characters, friendly connection request '
        "mentioning the role>}\n\n"
        f"Data:\n{json.dumps(context, indent=2)}"
    )

    data = chat_json(system, user)
    return OutreachDraft(
        email_subject=str(data.get("email_subject", "")).strip() or f"Interest in the {jd.title} role",
        email_body=str(data.get("email_body", "")).strip(),
        linkedin_note=str(data.get("linkedin_note", "")).strip()[:300],
    )
