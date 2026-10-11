# 输入数据与溯源

- `inputs/reference_candidates.fasta`：候选全长序列及 WT 参考，用于编号和序列核对。
- `inputs/train_variants.txt`：六位点训练变体列表，用于计算候选与训练集的 Hamming 距离。
- `inputs/reference_front38.tsv`：第 5 步固定结构预测队列所用的 38 条参考名单。
- `inputs/06_colabfold_msa/`：运行时必需、但未随仓库分发的候选 A3M MSA。每个 query 必须与候选 FASTA 序列完全匹配。请按项目 README 的说明自行准备；当前代码不会自动联网搜索或替换 MSA。

派生 FASTA、MSA 副本、模型输出和运行日志均在运行时产生，不是原始输入。历史数据的公开版本、获取日期和完整许可信息尚未全部核实；使用者应回查原始来源并遵循其许可，不应将预测结果解释为实验活性。

