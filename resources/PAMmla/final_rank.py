import pandas as pd

# ---- 输入 ----
cand = pd.read_csv("atrich_candidates_v2.csv", index_col=0)
with open("train_variants.txt") as f:
    train = [l.strip() for l in f if l.strip()]

def hd(a, b):
    return sum(x != y for x, y in zip(a, b))

# ---- 1. 每个候选到训练集的最近 Hamming 距离 ----
cand["HD_to_train"] = [min(hd(v, t) for t in train) for v in cand.index]
def tier(h):
    return "A(HD<=2 可信)" if h <= 2 else ("B(HD3-4 外推)" if h <= 4 else "C(HD>=5 高风险)")
cand["tier"] = cand["HD_to_train"].map(tier)
cand = cand.sort_values("AT_max", ascending=False)
cand.to_csv("atrich_candidates_scored.csv")

# ---- 2. 家族去重: 与已入选者 HD<=1 视为同一家族, 只留最强代表 ----
selected = []
for v in cand.index:
    if all(hd(v, s) > 1 for s in selected):
        selected.append(v)
short = cand.loc[selected]
short.to_csv("final_shortlist_diverse.csv")

# ---- 3. 汇总 ----
print("=== 可信度分层 ===")
print(cand["tier"].value_counts().to_string())
print(f"\n去重前 {len(cand)} -> 去重后 {len(short)} 个家族代表")
print("\n=== 最终名单 top 20 ===")
print(short[["AT_max","NGG_max","selectivity","best_AT_PAM","HD_to_train","tier"]].head(20).to_string())
