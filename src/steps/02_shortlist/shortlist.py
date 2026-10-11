import glob
import pandas as pd

ACT_CUTOFF = -3.0
SEL_CUTOFF = 0.3

nts = "ACGT"
AT_RICH = ["N"+a+b+c for a in nts for b in nts for c in nts
           if (a+b+c).count("A")+(a+b+c).count("T") >= 2]
NGG_PAMS = ["NGGA","NGGC","NGGG","NGGT"]

# ---- 1. 全空间扫描(放宽阈值) ----
hits = []
for f in sorted(glob.glob("./predictions_64m/shard_*.parquet")):
    df = pd.read_parquet(f)
    at_max  = df[AT_RICH].max(axis=1)
    ngg_max = df[NGG_PAMS].max(axis=1)
    mask = (at_max > ACT_CUTOFF) & ((at_max - ngg_max) > SEL_CUTOFF)
    if mask.sum() > 0:
        h = df[mask].copy()
        h["AT_max"] = at_max[mask]
        h["NGG_max"] = ngg_max[mask]
        h["selectivity"] = (at_max - ngg_max)[mask]
        h["best_AT_PAM"] = df.loc[mask, AT_RICH].idxmax(axis=1)
        hits.append(h)
cand = pd.concat(hits).sort_values("AT_max", ascending=False)
print(f"候选池: {len(cand)}")

# ---- 2. HD 可信度分层 + 突变注释 ----
with open("train_variants.txt") as f:
    train = [l.strip() for l in f if l.strip()]
def hd(a, b): return sum(x != y for x, y in zip(a, b))
cand["HD_to_train"] = [min(hd(v, t) for t in train) for v in cand.index]
cand["tier"] = cand["HD_to_train"].map(
    lambda h: "A_reliable(HD<=2)" if h <= 2
    else ("B_extrapolated(HD3-4)" if h <= 4 else "C_highrisk(HD>=5)"))
WT = "DSGERT"
POS = [1135, 1136, 1218, 1219, 1335, 1337]
cand["mutations"] = [";".join(f"{w}{p}{v}" for v, w, p in zip(var, WT, POS) if v != w)
                     for var in cand.index]

# ---- 3. 分层家族去重 ----
def dedup(df):
    sel = []
    for v in df.index:
        if all(hd(v, s) > 1 for s in sel):
            sel.append(v)
    return df.loc[sel]

parts = []
for t in ["A_reliable(HD<=2)", "B_extrapolated(HD3-4)", "C_highrisk(HD>=5)"]:
    sub = cand[cand["tier"] == t]
    if len(sub):
        parts.append(dedup(sub))
        print(f"{t}: 去重前 {len(sub)} -> 家族代表 {len(parts[-1])}")
short = pd.concat(parts)

cand.to_csv("atrich_candidates_v3_all_scored.csv")
short.to_csv("final_shortlist_v2.csv")
print(f"\n最终名单 {len(short)} 个 -> final_shortlist_v2.csv")
print(short[["AT_max","NGG_max","selectivity","best_AT_PAM","HD_to_train","tier","mutations"]].head(30).to_string())
