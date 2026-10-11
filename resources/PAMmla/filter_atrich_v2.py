import glob
import pandas as pd

nts = "ACGT"
AT_RICH = ["N"+a+b+c for a in nts for b in nts for c in nts
           if (a+b+c).count("A")+(a+b+c).count("T") >= 2]   # 32 种, >=2/3 AT
NGG_PAMS = ["NGGA","NGGC","NGGG","NGGT"]

ACT_CUTOFF = -2.5
SEL_CUTOFF = 0.5

hits_all = []
for f in sorted(glob.glob("./predictions_64m/shard_*.parquet")):
    df = pd.read_parquet(f)
    at_max  = df[AT_RICH].max(axis=1)
    ngg_max = df[NGG_PAMS].max(axis=1)
    mask = (at_max > ACT_CUTOFF) & ((at_max - ngg_max) > SEL_CUTOFF)
    if mask.sum() > 0:
        hits = df[mask].copy()
        hits["AT_max"] = at_max[mask]
        hits["NGG_max"] = ngg_max[mask]
        hits["selectivity"] = (at_max - ngg_max)[mask]
        hits["best_AT_PAM"] = df.loc[mask, AT_RICH].idxmax(axis=1)
        hits_all.append(hits)

if not hits_all:
    print("no hits, try ACT_CUTOFF=-3.0 or SEL_CUTOFF=0.3")
else:
    res = pd.concat(hits_all).sort_values("AT_max", ascending=False)
    res.to_csv("atrich_candidates_v2.csv")
    print(f"total {len(res)} candidates -> atrich_candidates_v2.csv")
    print(res[["AT_max","NGG_max","selectivity","best_AT_PAM"]].head(20))
