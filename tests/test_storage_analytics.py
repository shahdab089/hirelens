import os
import unittest
from datetime import datetime
from core.schema import ApplicationRecord, ParsedJD, ParsedResume, FitScore, Diagnosis, RejectionStage, SubScore
from storage import save, load_all, set_outcome, DB_PATH
from analytics.patterns import build_report

class TestStorageAnalytics(unittest.TestCase):
    def setUp(self):
        # Use a temporary database for testing
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)

    def tearDown(self):
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)

    def create_mock_record(self, record_id: str, stage: RejectionStage, fit_score: float, created_at: datetime = None):
        jd = ParsedJD(raw_text="JD text", title="Engineer", required_skills=["Python"])
        resume = ParsedResume(raw_text="Resume text", skills=["Python"])
        fit = FitScore(
            overall=fit_score,
            subscores=[SubScore(name="skills", score=fit_score, rationale="Good match")],
            matched_skills=["Python"],
            missing_skills=[]
        )
        diag = Diagnosis(
            likely_stage=stage,
            headline="Match",
            explanation="Good fit",
            top_fixes=[]
        )
        return ApplicationRecord(
            id=record_id,
            created_at=created_at or datetime.now(),
            jd=jd,
            resume=resume,
            fit=fit,
            diagnosis=diag,
            outcome=None
        )

    def test_storage_and_analytics(self):
        # 1. Create and save records with explicit timestamps to ensure ordering
        now = datetime.now()
        from datetime import timedelta
        r1 = self.create_mock_record("1", RejectionStage.keyword_ats, 0.4, now - timedelta(minutes=10))
        r2 = self.create_mock_record("2", RejectionStage.keyword_ats, 0.5, now - timedelta(minutes=5))
        r3 = self.create_mock_record("3", RejectionStage.skills_gap, 0.8, now)
        
        save(r1)
        save(r2)
        save(r3)
        
        # 2. Load and verify
        loaded = load_all()
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded[0].id, "3") # Ordered by created_at DESC
        self.assertEqual(loaded[2].id, "1")
        
        # 3. Build report
        report = build_report(loaded)
        self.assertEqual(report.total_applications, 3)
        self.assertEqual(report.dominant_stage, RejectionStage.keyword_ats)
        self.assertAlmostEqual(report.avg_overall_fit, (0.4 + 0.5 + 0.8) / 3)
        self.assertIn("keyword_ats", report.insight)
        
        # 4. Set outcome
        set_outcome("1", "interview")
        loaded_updated = load_all()
        r1_updated = next(r for r in loaded_updated if r.id == "1")
        self.assertEqual(r1_updated.outcome, "interview")

if __name__ == "__main__":
    unittest.main()
