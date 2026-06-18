"""
SHARED CONTRACT — this file is the single source of truth for all three agents.

Owned by: Claude. Do NOT edit unless you are Claude.
Everyone else: import these models, code against them, do not redefine them.

The whole app is a pipeline:
    raw resume text + raw JD text
        -> ParsedResume / ParsedJD        (parsing.py     — Gemini)
        -> FitScore                        (scoring.py     — Claude)
        -> Diagnosis                       (diagnosis.py   — Claude)
        -> stored as ApplicationRecord     (storage.py     — Gemini)
        -> aggregated into PatternReport   (patterns.py    — Gemini)
        -> shown in Streamlit              (app.py         — Claude)
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------- Parsing layer (Gemini owns the producers) ----------

class ParsedResume(BaseModel):
    raw_text: str
    skills: list[str] = Field(default_factory=list)
    years_experience: Optional[float] = None
    seniority: Optional[str] = None          # "junior" | "mid" | "senior" | "staff"
    titles: list[str] = Field(default_factory=list)


class ParsedJD(BaseModel):
    raw_text: str
    title: str
    company: Optional[str] = None
    required_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    seniority: Optional[str] = None
    hard_requirements: list[str] = Field(default_factory=list)  # e.g. "5+ yrs", "must have AWS"


# ---------- Scoring layer (Claude owns the producer) ----------

class SubScore(BaseModel):
    name: str          # "skills" | "seniority" | "keywords_ats" | "domain"
    score: float       # 0.0 - 1.0
    rationale: str


class FitScore(BaseModel):
    overall: float                     # 0.0 - 1.0
    subscores: list[SubScore]
    matched_skills: list[str]
    missing_skills: list[str]          # in JD, absent in resume


# ---------- Diagnosis layer (Claude owns the producer) ----------

class RejectionStage(str, Enum):
    keyword_ats = "keyword_ats"        # filtered before a human saw it
    seniority_mismatch = "seniority_mismatch"
    skills_gap = "skills_gap"
    domain_mismatch = "domain_mismatch"
    competitive = "competitive"        # qualified but out-competed
    likely_fine = "likely_fine"        # no obvious flaw -> volume/luck problem


class Diagnosis(BaseModel):
    likely_stage: RejectionStage
    headline: str                      # one brutal, specific sentence
    explanation: str
    top_fixes: list[str]               # concrete, actionable


# ---------- Storage layer (Gemini owns) ----------

class ApplicationRecord(BaseModel):
    id: str
    created_at: datetime
    jd: ParsedJD
    resume: ParsedResume
    fit: FitScore
    diagnosis: Diagnosis
    outcome: Optional[str] = None      # "rejected" | "interview" | "ghosted" | "offer" | None


# ---------- Analytics layer (Gemini owns) ----------

class PatternReport(BaseModel):
    total_applications: int
    avg_overall_fit: float
    dominant_stage: RejectionStage     # the bottleneck across all apps
    insight: str                       # "You get filtered at keyword stage on 80% of infra roles"
    recommended_focus: list[str]


# ---------- Outreach layer (contacts the recruiter published + AI draft) ----------

class ExtractedContacts(BaseModel):
    """Contact details the recruiter voluntarily put in the job posting text."""
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    application_links: list[str] = Field(default_factory=list)


class OutreachDraft(BaseModel):
    email_subject: str
    email_body: str
    linkedin_note: str
