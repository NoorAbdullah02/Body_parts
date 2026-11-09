"""Score the returned radiologist answers. Usage: python score_review.py clinician_review/review_answers_filled.csv"""
import sys, math, pandas as pd
ans = pd.read_csv(sys.argv[1]); key = pd.read_csv("clinician_review/answer_key_DO_NOT_SHARE.csv")
d = key.merge(ans[["code", "answer_1_2_3", "correct_body_part_if_2", "comment"]], on="code")
d["a"] = pd.to_numeric(d["answer_1_2_3"], errors="coerce")
n_ok, n_bad, n_unsure = int((d.a == 1).sum()), int((d.a == 2).sum()), int((d.a == 3).sum())
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n); return c - h, c + h
dec = n_ok + n_bad; lo, hi = wilson(n_bad, dec)
agree = int(((d.a == 2) & (d.correct_body_part_if_2.fillna("").str.strip().str.lower() == d.model_says.str.lower())).sum())
print(f"cases {len(d)}: given label correct {n_ok}, wrong {n_bad}, unsure {n_unsure}")
print(f"audit precision (wrong / decided) = {n_bad}/{dec} = {n_bad / dec:.2f}  [Wilson 95 % CI {lo:.2f}, {hi:.2f}]" if dec else "no decided cases")
print(f"of the {n_bad} confirmed errors, the reviewer's correction equals the model's label in {agree}")
print("Paper wording: 'A radiologist reviewed the 39 suspects blind to the model label; X were judged wrong (precision Y [CI]); this is a precision estimate, not an error rate.'")
