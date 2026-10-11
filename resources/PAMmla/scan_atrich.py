import glob
import pandas as pd

AT_PAMS = ["NAAA","NAAT","NATA","NATT","NTAA","NTAT","NTTA","NTTT"]
NGG_PAMS = ["NGGA","NGGC","NGGG","NGGT"]

tops = []
n25 = n30 = n35 = 0
for f in sorted(glob.glob("./predictions_64m/shard_*.parquet")):
    df = pd.read_parquet(f)
    at_max = df[AT_PAMS].max(axis=1)
    ngg_max = df[NGG_PAMS].max(axis=1)
    n25 += (at_max > -2.5).sum()
    n30 += (at_max > -3.0).sum()
    n35 += (at_max > -3.5).sum()
    tmp = pd.DataFrame({"AT_max": at_max, "NGG_max": ngg_max})
    tmp["sel"] = tmp["AT_max"] - tmp["NGG_max"]
    tmp["best_AT_PAM"] = df[AT_PAMS].idxmax(axis=1)
    tops.append(tmp.nlargest(50, "AT_max"))
    tops.append(tmp.nlargest(50, "sel"))

top = pd.concat(tops)
print(f"AT_max > -2.5: {n25}   > -3.0: {n30}   > -3.5: {n35}")
print("\n== 按 AT 绝对活性 top 20 ==")
print(top.nlargest(20, "AT_max"))
print("\n== 按选择性(AT-NGG) top 20 ==")
print(top.nlargest(20, "sel"))
