import os, sys, pickle as pkl
import numpy as np
import pandas as pd
from itertools import product
from tensorflow import keras

AAs = list("GPAVLIMCFYWHKRQNEDST")
MODEL_DIRS = ["./220924_NN_rand_seed0_ROS",
              "./220924_NN_rand_seed2_ROS",
              "./220924_NN_rand_seed3_ROS"]
OUT_DIR = "./predictions_64m"
BATCH_SIZE = 8192

shard_id = int(sys.argv[1])
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"shard_{shard_id:03d}.parquet")
if os.path.exists(out_path):
    print(f"shard {shard_id} exists, skip")
    sys.exit(0)

# 加载 3 个模型及各自的 encoder / mean_std / 参数
models = []
for d in MODEL_DIRS:
    model = keras.models.load_model(d + "/saved_model")
    with open(d + "/encoder.pkl", "rb") as f:
        enc = pkl.load(f)
    with open(d + "/mean_std.pkl", "rb") as f:
        mean, std = pkl.load(f)
    with open(d + "/data_parameters.pkl", "rb") as f:
        v = pkl.load(f)
    all_positions, pams = v[9], v[10]
    models.append((model, enc, mean, std))

all_positions, pams = v[9], v[10]

# 本分片：前两位固定，后四位全组合（20^4 = 16 万）
p1, p2 = AAs[shard_id // 20], AAs[shard_id % 20]
variants = [p1 + p2 + "".join(r) for r in product(AAs, repeat=4)]
df = pd.DataFrame([list(x) for x in variants], columns=all_positions)

# 3 个模型预测取平均（反归一化后）
preds_sum = None
for model, enc, mean, std in models:
    X = enc.transform(df).toarray()
    preds = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
    preds = preds * std + mean
    preds_sum = preds if preds_sum is None else preds_sum + preds
preds_avg = preds_sum / len(models)

out = pd.DataFrame(preds_avg, columns=pams, index=variants).round(2)
out.index.name = "variant"
out.to_parquet(out_path + ".tmp")
os.replace(out_path + ".tmp", out_path)
print(f"shard {shard_id} done: {len(out)} variants -> {out_path}")
