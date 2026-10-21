"""Quick CLI test: python run_demo.py [optional_training_csv]"""
import sys

from data import load
from llm import explain
from model import score, train

df = load(sys.argv[1] if len(sys.argv) > 1 else None)
model, baseline, m = train(df)
print(f"ROC-AUC: {m['auc']:.3f}   default rate: {m['default_rate']:.1%}")
print("Top features:", list(m["importance"])[:4])

applicant = dict(revolving_utilization=0.95, age=29, late_30_59=2, debt_ratio=0.8,
                 monthly_income=3200, open_credit_lines=9, late_90_plus=1,
                 real_estate_loans=0, late_60_89=1, dependents=2)
res = score(model, baseline, applicant)
print(res)
text, src = explain(res)
print(f"\n--- explanation ({src}) ---\n{text}")
