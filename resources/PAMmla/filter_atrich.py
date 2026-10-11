import glob
import pandas as pd

AT_PAMS = ["NAAA","NAAT","NATA","NATT","NTAA","NTAT","NTTA","NTTT"]  # 3/3 AT
NGG_PAMS = ["NGGA","NGGC","NGGG","NGGT"]

ACT_CUTOFF = -2.5   # 绝对活性阈值（WT 的 NGG ≈ -1.7，-2.5 约为其 1/6 活性）
SEL_CUTOFF = 1.0    # AT_max 要比 NGG_max 高 1 个 log（10 倍），确保是 AT 偏好而非泛活性

hits_all = []
for f in sorted(glob.glob("./predictions_64m/shard_*.parquet")):
    df = pd.read_parquet(f)
    at_max  = df[AT_PAMS].max(axis=1)
    ngg_max = df[NGG_PAMS].max(axis=1)
    mask = (at_max > ACT_CUTOFF) & ((at_max - ngg_max) > SEL_CUTOFF)
    if mask.sum() > 0:
        hits = df[mask].copy()
        hits["AT_max"] = at_max[mask]
        hits["NGG_max"] = ngg_max[mask]
        hits["selectivity"] = (at_max - ngg_max)[mask]
        hits["best_AT_PAM"] = df.loc[mask, AT_PAMS].idxmax(axis=1)
        hits_all.append(hits)

res = pd.concat(hits_all).sort_values("AT_max", ascending=False)
res.to_csv("atrich_candidates.csv")
print(f"总计 {len(res)} 个候选 -> atrich_candidates.csv")
print(res[["AT_max","NGG_max","selectivity","best_AT_PAM"]].head(20))
