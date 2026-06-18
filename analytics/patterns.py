from collections import Counter
from typing import List

from core.schema import ApplicationRecord, PatternReport, RejectionStage

# Outcomes that represent an application that did NOT progress. The whole point
# of the pattern report is to find the rejection bottleneck, so successful
# applications (interview/offer) must not count toward the dominant stage.
UNSUCCESSFUL_OUTCOMES = {"rejected", "ghosted"}

_FOCUS_BY_STAGE = {
    RejectionStage.keyword_ats: [
        "Optimize resume keywords to mirror the JD",
        "Use standard, ATS-friendly job titles",
        "Ensure the PDF is machine-readable (no images/columns)",
    ],
    RejectionStage.skills_gap: [
        "Identify and close the missing core skills",
        "Target roles closer to your current tech stack",
        "Up-skill in the most frequently-required tools",
    ],
    RejectionStage.seniority_mismatch: [
        "Apply to roles matching your real seniority band",
        "Reframe how years of experience are presented",
        "Highlight scope/impact rather than just tenure",
    ],
    RejectionStage.domain_mismatch: [
        "Target your proven industry/domain",
        "Translate transferable experience explicitly",
        "Build a small project in the target domain",
    ],
    RejectionStage.competitive: [
        "Strengthen differentiators (portfolio, metrics)",
        "Leverage referrals to bypass the resume pile",
        "Apply earlier and to less-saturated roles",
    ],
}

_DEFAULT_FOCUS = [
    "Increase application volume",
    "Network with recruiters",
    "Polish your portfolio",
]


def build_report(records: List[ApplicationRecord]) -> PatternReport:
    """Aggregates ApplicationRecords into a PatternReport (pure logic, no LLM)."""
    if not records:
        return PatternReport(
            total_applications=0,
            avg_overall_fit=0.0,
            dominant_stage=RejectionStage.likely_fine,
            insight="No applications logged yet.",
            recommended_focus=["Start applying and logging your results!"],
        )

    total_apps = len(records)
    avg_fit = sum(r.fit.overall for r in records) / total_apps

    # Only unsuccessful applications inform the rejection bottleneck. If outcomes
    # haven't been recorded yet, fall back to every record's diagnosis.
    failed = [r for r in records if (r.outcome or "").lower() in UNSUCCESSFUL_OUTCOMES]
    pool = failed if failed else records
    denom = len(pool)

    stage_counts = Counter(r.diagnosis.likely_stage for r in pool)
    dominant_stage, dominant_count = stage_counts.most_common(1)[0]
    percentage = (dominant_count / denom) * 100

    scope = "rejected/ghosted" if failed else "all logged"
    insight = (
        f"Across your {denom} {scope} applications, the most common failure point "
        f"is '{dominant_stage.value}', appearing in {percentage:.0f}% of them."
    )
    focus = _FOCUS_BY_STAGE.get(dominant_stage, _DEFAULT_FOCUS)

    return PatternReport(
        total_applications=total_apps,
        avg_overall_fit=avg_fit,
        dominant_stage=dominant_stage,
        insight=insight,
        recommended_focus=focus,
    )
