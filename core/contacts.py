"""
Contact extraction — owned by Claude.

extract_contacts(text) -> ExtractedContacts

Pulls contact details that a recruiter VOLUNTARILY published in the job-posting
text the user pasted. This is deterministic regex over text the user already
provided — no scraping, no external lookups, no personal-data harvesting.
"""
import re

from .schema import ExtractedContacts

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# Phone: optional +country, then 7-14 digits with common separators. Conservative
# to avoid matching salaries/years.
_PHONE_RE = re.compile(r"(?<!\d)(\+?\d[\d\s().\-]{7,}\d)(?!\d)")
_URL_RE = re.compile(r"https?://[^\s)\"'<>]+", re.IGNORECASE)

# URLs that are clearly "apply here" / contact links worth surfacing.
_APPLY_HINTS = ("apply", "career", "job", "greenhouse", "lever", "workday", "ashby", "smartrecruiters")


def _dedupe(seq: list[str]) -> list[str]:
    seen, out = set(), []
    for s in seq:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _looks_like_phone(candidate: str) -> bool:
    s = candidate.strip()
    digits = re.sub(r"\D", "", s)
    if not (7 <= len(digits) <= 15):
        return False
    # Reject numeric ranges like "138000-221000" (salaries, not phones).
    if re.fullmatch(r"\d{3,}-\d{3,}", s):
        return False
    # Require genuine phone formatting (+, parens, spaces, or 2+ dashes) so we
    # don't grab bare ID-like numbers.
    return s.startswith("+") or "(" in s or " " in s or s.count("-") >= 2


def extract_contacts(text: str) -> ExtractedContacts:
    if not text:
        return ExtractedContacts()

    emails = _dedupe(_EMAIL_RE.findall(text))

    phones = _dedupe([m.strip() for m in _PHONE_RE.findall(text) if _looks_like_phone(m)])

    # Strip trailing sentence punctuation the regex may have swallowed.
    links = _dedupe([u.rstrip(".,;:)]}’\"'") for u in _URL_RE.findall(text)])
    # Keep apply/career links; drop noise like generic company homepages where possible.
    application_links = [u for u in links if any(h in u.lower() for h in _APPLY_HINTS)]
    # If nothing matched the hints but there are links, keep up to 3 as fallback.
    if not application_links and links:
        application_links = links[:3]

    return ExtractedContacts(
        emails=emails,
        phones=phones,
        application_links=application_links,
    )
