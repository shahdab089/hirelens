import os
import csv
from typing import List, Dict
from core.parsing import extract_text, parse_resume, parse_jd
from core.scoring import score
from core.diagnosis import diagnose

def run_eval(labels_csv: str = "data/labels.csv",
             samples_dir: str = "data/samples") -> dict:
    """Measures how well the scorer agrees with real outcomes in labels_csv."""
    if not os.path.isabs(labels_csv):
        labels_csv = os.path.join(os.getcwd(), labels_csv)
    if not os.path.isabs(samples_dir):
        samples_dir = os.path.join(os.getcwd(), samples_dir)

    rows_data = []
    with open(labels_csv, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            resume_path = os.path.join(samples_dir, row['resume_file'])
            jd_path = os.path.join(samples_dir, row['jd_file'])
            
            resume_text = extract_text(resume_path)
            jd_text = extract_text(jd_path)
            
            parsed_resume = parse_resume(resume_text)
            parsed_jd = parse_jd(jd_text)
            
            fit = score(parsed_resume, parsed_jd)
            diag = diagnose(fit, parsed_resume, parsed_jd)
            
            real_outcome = row['real_outcome'].lower()
            # interview/offer -> 1, rejected/ghosted -> 0
            real_advanced = 1 if real_outcome in ['interview', 'offer'] else 0
            # threshold 0.5
            pred_advanced = 1 if fit.overall >= 0.5 else 0
            
            rows_data.append({
                "resume_file": row['resume_file'],
                "jd_file": row['jd_file'],
                "fit_overall": fit.overall,
                "real_outcome": real_outcome,
                "real_advanced": real_advanced,
                "pred_advanced": pred_advanced,
                "likely_stage": diag.likely_stage.value,
                "agreement": int(real_advanced == pred_advanced)
            })

    n = len(rows_data)
    accuracy = sum(r['agreement'] for r in rows_data) / n if n > 0 else 0
    
    advanced_fits = [r['fit_overall'] for r in rows_data if r['real_advanced'] == 1]
    rejected_fits = [r['fit_overall'] for r in rows_data if r['real_advanced'] == 0]
    
    avg_fit_advanced = sum(advanced_fits) / len(advanced_fits) if advanced_fits else 0
    avg_fit_rejected = sum(rejected_fits) / len(rejected_fits) if rejected_fits else 0
    
    results = {
        "n": n,
        "accuracy": accuracy,
        "avg_fit_advanced": avg_fit_advanced,
        "avg_fit_rejected": avg_fit_rejected,
        "rows": rows_data
    }
    return results

if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("Warning: Neither GROQ_API_KEY nor GOOGLE_API_KEY found. LLM calls will fail.")
    
    print("Running Evaluation Harness...")
    # Adjust paths if running from app_autopsy or project root
    base_dir = os.getcwd()
    if not os.path.exists("data") and os.path.exists("app_autopsy/data"):
        os.chdir("app_autopsy")
        
    results = run_eval()
    
    print("\n" + "="*80)
    print(f"{'Resume':<15} | {'JD':<15} | {'Fit':<5} | {'Real Outcome':<12} | {'Agreement'}")
    print("-" * 80)
    for row in results['rows']:
        print(f"{row['resume_file']:<15} | {row['jd_file']:<15} | {row['fit_overall']:.2f} | {row['real_outcome']:<12} | {row['agreement']}")
    print("="*80)
    
    print(f"\nTotal Samples (n): {results['n']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Avg Fit (Real Advanced): {results['avg_fit_advanced']:.2f}")
    print(f"Avg Fit (Real Rejected): {results['avg_fit_rejected']:.2f}")
    
    if results['avg_fit_advanced'] > results['avg_fit_rejected']:
        print("\nSuccess: The scorer correctly identifies advanced applications with higher scores.")
    else:
        print("\nNote: Scorer separation between advanced and rejected applications is low.")
