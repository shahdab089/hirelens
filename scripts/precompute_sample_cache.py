"""Precompute analysis results for the built-in sample résumé/JD pairs and write
them to data/sample_cache.json, so the deployed app serves the "Try a sample"
demos with zero LLM calls (even after an ephemeral-storage redeploy).

Run once locally whenever the samples, the parsing/scoring prompts, or the model
change (needs ANTHROPIC_API_KEY or GROQ_API_KEY in the environment):

    python scripts/precompute_sample_cache.py

The cache key folds in ANALYSIS_PIPELINE_VERSION + the model IDs, so the keys this
writes match what the server computes only when the same env defaults are in effect
— don't set ANTHROPIC_MODEL/GROQ_FALLBACK_MODEL differently here than on the host.
"""
import csv
import json
import sys
from pathlib import Path

# Make the repo root importable when run as a plain script from anywhere.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app_web import _analysis_cache_key  # noqa: E402
from core.parsing import parse_both  # noqa: E402
from core.scoring import score_and_diagnose  # noqa: E402

SAMPLES_DIR = ROOT / "data" / "samples"
LABELS_CSV = ROOT / "data" / "labels.csv"
OUT = ROOT / "data" / "sample_cache.json"


def main() -> None:
    with open(LABELS_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    entries: dict[str, dict] = {}
    for row in rows:
        resume_path = SAMPLES_DIR / row["resume_file"]
        jd_path = SAMPLES_DIR / row["jd_file"]
        if not resume_path.exists() or not jd_path.exists():
            print(f"  skip (missing files): {row['resume_file']} / {row['jd_file']}")
            continue

        resume_text = resume_path.read_text(encoding="utf-8")
        jd_text = jd_path.read_text(encoding="utf-8")
        key = _analysis_cache_key(resume_text, jd_text)
        if key in entries:
            continue

        print(f"Analyzing {row['resume_file']} × {row['jd_file']} ...")
        resume, jd = parse_both(resume_text, jd_text)
        fit, diag = score_and_diagnose(resume, jd)
        entries[key] = {
            "resume": resume.model_dump(mode="json"),
            "jd": jd.model_dump(mode="json"),
            "fit": fit.model_dump(mode="json"),
            "diagnosis": diag.model_dump(mode="json"),
        }

    OUT.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"\nWrote {len(entries)} cached analyses to {OUT}")


if __name__ == "__main__":
    main()
