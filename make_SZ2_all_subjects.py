import os
import pandas as pd

DATA_PATH = "/Users/rosalouisemarker/Desktop/Digital Media Project/dataset"  # <- adjust

out_tsv = "net/datasets/SZ2_all_subjects.tsv"
os.makedirs(os.path.dirname(out_tsv), exist_ok=True)

subjects = [
    d for d in os.listdir(DATA_PATH)
    if d.startswith("sub-") and os.path.isdir(os.path.join(DATA_PATH, d))
]

subjects = sorted(subjects)

df = pd.DataFrame({"subject": subjects})
df.to_csv(out_tsv, sep="\t", index=False, header=False)

print(f"Wrote {len(subjects)} subjects to {out_tsv}")