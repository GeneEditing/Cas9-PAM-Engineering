import os
from pathlib import Path
PACKAGE_ROOT = Path(os.environ["CAS9_ROOT"])
PACKAGE_RUN = Path(os.environ["CAS9_RUN"])
import math
from pathlib import Path

import pandas as pd

# ============================================================
# Paths
# ============================================================

AF2RUN = PACKAGE_RUN / "results/06_colabfold"

INPUT_TSV = PACKAGE_RUN / "inputs/07_final_ranking/final_ranking_input.tsv"

INPUT_FASTA = PACKAGE_RUN / "inputs/07_final_ranking/all38_candidates.fasta"

OUTPUT_DIR = PACKAGE_RUN / "results/07_final_ranking"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_TSV = OUTPUT_DIR / "final_ranking_all38.tsv"
TOP10_TSV = OUTPUT_DIR / "final_top10.tsv"
TOP10_SEQUENCE_TSV = OUTPUT_DIR / "final_top10_sequences.tsv"
TOP10_FASTA = OUTPUT_DIR / "final_top10.fasta"
RANKED_FASTA = OUTPUT_DIR / "final_all38_ranked.fasta"
REPORT = OUTPUT_DIR / "final_ranking_report.txt"

# ============================================================
# Read data
# ============================================================

df = pd.read_csv(INPUT_TSV, sep="\t")

if len(df) != 38:
    raise RuntimeError(
        f"输入候选数量为 {len(df)}，不是预期的38条"
    )

required_columns = [
    "candidate_id",
    "genotype",
    "pareto_front",
    "AT_max",
    "selectivity",
    "HD_to_train",
    "progen2_delta_vs_wt",
    "esmc_mutctx_mean_delta",
    "esmc_mutctx_min_delta",
    "mean_plddt",
    "ptm",
    "six_site_min_plddt",
    "pass_structural_gate",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise RuntimeError(
        f"缺少必要列：{missing}"
    )

numeric_columns = [
    "pareto_front",
    "AT_max",
    "selectivity",
    "HD_to_train",
    "progen2_delta_vs_wt",
    "esmc_mutctx_mean_delta",
    "esmc_mutctx_min_delta",
    "mean_plddt",
    "ptm",
    "six_site_min_plddt",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

if df[numeric_columns].isna().any().any():
    bad_columns = df[numeric_columns].columns[
        df[numeric_columns].isna().any()
    ].tolist()

    raise RuntimeError(
        f"以下指标包含缺失或非数值：{bad_columns}"
    )

df["pareto_front"] = (
    df["pareto_front"].astype(int)
)

# ============================================================
# Validate structural gate
# ============================================================

def parse_bool(value):
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true", "1", "yes", "y"
    }

df["pass_structural_gate"] = (
    df["pass_structural_gate"].map(parse_bool)
)

failed_structure = df[
    ~df["pass_structural_gate"]
]

if not failed_structure.empty:
    print("以下候选未通过结构门控，将被排除：")
    print(
        failed_structure[
            ["candidate_id", "mean_plddt",
             "ptm", "six_site_min_plddt"]
        ].to_string(index=False)
    )

df = df[
    df["pass_structural_gate"]
].copy()

if len(df) != 38:
    raise RuntimeError(
        f"结构门控后剩余 {len(df)} 条，不是预期的38条"
    )

# ============================================================
# Direction-corrected percentile
#
# Best = 1.0
# Worst = 0.0
# Ties use average rank
# ============================================================

def percentile_score(series, higher_is_better=True):
    n = len(series)

    if n <= 1:
        return pd.Series(
            1.0,
            index=series.index
        )

    ranks = series.rank(
        method="average",
        ascending=(
            True if higher_is_better else False
        )
    )

    return (ranks - 1.0) / (n - 1.0)

# PAM performance: higher is better
df["P_AT_max"] = percentile_score(
    df["AT_max"],
    higher_is_better=True
)

df["P_selectivity"] = percentile_score(
    df["selectivity"],
    higher_is_better=True
)

# Language-model plausibility: less negative / higher is better
df["P_progen2_delta"] = percentile_score(
    df["progen2_delta_vs_wt"],
    higher_is_better=True
)

df["P_esmc_mean"] = percentile_score(
    df["esmc_mutctx_mean_delta"],
    higher_is_better=True
)

# Worst-site robustness: ESM-C higher is better; HD lower is better
df["P_esmc_worst"] = percentile_score(
    df["esmc_mutctx_min_delta"],
    higher_is_better=True
)

df["P_neg_HD"] = percentile_score(
    df["HD_to_train"],
    higher_is_better=False
)

# AlphaFold2: higher is better
df["P_mean_plddt"] = percentile_score(
    df["mean_plddt"],
    higher_is_better=True
)

df["P_six_site_min_plddt"] = percentile_score(
    df["six_site_min_plddt"],
    higher_is_better=True
)

# ============================================================
# Component scores
# ============================================================

df["S_PAM"] = (
    0.50 * df["P_AT_max"]
    + 0.50 * df["P_selectivity"]
)

df["S_LM"] = (
    0.50 * df["P_progen2_delta"]
    + 0.50 * df["P_esmc_mean"]
)

df["S_R"] = (
    0.50 * df["P_esmc_worst"]
    + 0.50 * df["P_neg_HD"]
)

df["S_AF2"] = (
    0.30 * df["P_mean_plddt"]
    + 0.70 * df["P_six_site_min_plddt"]
)

df["S_final"] = (
    0.45 * df["S_PAM"]
    + 0.27 * df["S_LM"]
    + 0.18 * df["S_R"]
    + 0.10 * df["S_AF2"]
)

df["final_score_100"] = (
    100.0 * df["S_final"]
)

# ============================================================
# Final hierarchical sorting
# ============================================================

df = df.sort_values(
    by=[
        "pareto_front",
        "S_final",
        "S_PAM",
        "S_R",
        "six_site_min_plddt",
        "candidate_id",
    ],
    ascending=[
        True,
        False,
        False,
        False,
        False,
        True,
    ],
    kind="mergesort"
).reset_index(drop=True)

df.insert(
    0,
    "final_rank",
    range(1, len(df) + 1)
)

df.insert(
    2,
    "rank_within_front",
    df.groupby("pareto_front").cumcount() + 1
)

# ============================================================
# Arrange output columns
# ============================================================

leading_columns = [
    "final_rank",
    "pareto_front",
    "rank_within_front",
    "candidate_id",
    "genotype",
    "final_score_100",
    "S_final",
    "S_PAM",
    "S_LM",
    "S_R",
    "S_AF2",
    "AT_max",
    "selectivity",
    "best_AT_PAM",
    "HD_to_train",
    "progen2_delta_vs_wt",
    "progen2_ll_bi_mean",
    "esmc_mutctx_mean_delta",
    "esmc_mutctx_min_delta",
    "esmc_worst_mutation",
    "mean_plddt",
    "delta_plddt_vs_wt",
    "ptm",
    "delta_ptm_vs_wt",
    "six_site_mean_plddt",
    "six_site_min_plddt",
    "pass_structural_gate",
    "P_AT_max",
    "P_selectivity",
    "P_progen2_delta",
    "P_esmc_mean",
    "P_esmc_worst",
    "P_neg_HD",
    "P_mean_plddt",
    "P_six_site_min_plddt",
]

leading_columns = [
    column
    for column in leading_columns
    if column in df.columns
]

remaining_columns = [
    column
    for column in df.columns
    if column not in leading_columns
]

df = df[
    leading_columns + remaining_columns
]

df.to_csv(
    FINAL_TSV,
    sep="\t",
    index=False,
    float_format="%.8f"
)

top10 = df.head(10).copy()

top10.to_csv(
    TOP10_TSV,
    sep="\t",
    index=False,
    float_format="%.8f"
)

# ============================================================
# Read candidate sequences
# ============================================================

def read_fasta(path):
    records = {}
    header = None
    sequence_parts = []

    with path.open() as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(
                        sequence_parts
                    )

                header = line[1:].split()[0]
                sequence_parts = []
            else:
                sequence_parts.append(line)

    if header is not None:
        records[header] = "".join(sequence_parts)

    return records

sequences = read_fasta(INPUT_FASTA)

missing_sequences = [
    candidate_id
    for candidate_id in df["candidate_id"]
    if candidate_id not in sequences
]

if missing_sequences:
    raise RuntimeError(
        "以下候选未在FASTA中找到："
        + ", ".join(missing_sequences)
    )

# Ranked FASTA for all 38
with RANKED_FASTA.open("w") as handle:
    for _, row in df.iterrows():
        candidate_id = row["candidate_id"]
        sequence = sequences[candidate_id]

        handle.write(
            f">rank_{int(row['final_rank']):02d}|"
            f"front_{int(row['pareto_front'])}|"
            f"{candidate_id}\n"
        )

        for start in range(0, len(sequence), 80):
            handle.write(
                sequence[start:start + 80] + "\n"
            )

# Top-10 FASTA
with TOP10_FASTA.open("w") as handle:
    for _, row in top10.iterrows():
        candidate_id = row["candidate_id"]
        sequence = sequences[candidate_id]

        handle.write(
            f">rank_{int(row['final_rank']):02d}|"
            f"front_{int(row['pareto_front'])}|"
            f"{candidate_id}\n"
        )

        for start in range(0, len(sequence), 80):
            handle.write(
                sequence[start:start + 80] + "\n"
            )

top10_sequence_rows = []

for _, row in top10.iterrows():
    top10_sequence_rows.append({
        "final_rank": int(row["final_rank"]),
        "pareto_front": int(row["pareto_front"]),
        "rank_within_front": int(
            row["rank_within_front"]
        ),
        "candidate_id": row["candidate_id"],
        "genotype": row["genotype"],
        "final_score_100": row["final_score_100"],
        "sequence": sequences[row["candidate_id"]],
    })

pd.DataFrame(
    top10_sequence_rows
).to_csv(
    TOP10_SEQUENCE_TSV,
    sep="\t",
    index=False,
    float_format="%.8f"
)

# ============================================================
# Report
# ============================================================

front_counts = (
    df["pareto_front"]
    .value_counts()
    .sort_index()
)

with REPORT.open("w") as handle:
    handle.write(
        "Final ranking: PAM / ProGen2 / ESM-C / AlphaFold2\n"
    )
    handle.write("=" * 72 + "\n\n")

    handle.write(
        "Hierarchy:\n"
        "  1. Pareto Front ascending\n"
        "  2. S_final descending within each Front\n\n"
    )

    handle.write(
        "S_PAM = 0.50 P(AT_max) "
        "+ 0.50 P(Selectivity)\n"
    )
    handle.write(
        "S_LM = 0.50 P(ProGen2 delta) "
        "+ 0.50 P(ESM-C mean delta)\n"
    )
    handle.write(
        "S_R = 0.50 P(ESM-C worst delta) "
        "+ 0.50 P(-HD_to_train)\n"
    )
    handle.write(
        "S_AF2 = 0.30 P(mean pLDDT) "
        "+ 0.70 P(six-site minimum pLDDT)\n"
    )
    handle.write(
        "S_final = 0.45 S_PAM + 0.27 S_LM "
        "+ 0.18 S_R + 0.10 S_AF2\n\n"
    )

    handle.write(
        "Direction:\n"
        "  Higher is better: AT_max, Selectivity, "
        "ProGen2 delta, ESM-C mean, ESM-C worst, "
        "mean pLDDT, six-site minimum pLDDT\n"
        "  Lower is better: HD_to_train\n\n"
    )

    handle.write(
        f"Final eligible candidates: {len(df)}\n"
    )
    handle.write(
        "Structural gate passed: "
        f"{int(df['pass_structural_gate'].sum())}/"
        f"{len(df)}\n\n"
    )

    handle.write("Candidates by Front:\n")

    for front, count in front_counts.items():
        handle.write(
            f"  Front {front}: {count}\n"
        )

    handle.write("\nTop 10:\n")
    handle.write(
        top10[[
            "final_rank",
            "pareto_front",
            "rank_within_front",
            "candidate_id",
            "final_score_100",
            "S_PAM",
            "S_LM",
            "S_R",
            "S_AF2",
        ]].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )
    handle.write("\n")

print("=" * 80)
print("FINAL RANKING COMPLETED")
print("=" * 80)

print(
    df[[
        "final_rank",
        "pareto_front",
        "rank_within_front",
        "candidate_id",
        "final_score_100",
        "AT_max",
        "selectivity",
        "progen2_delta_vs_wt",
        "esmc_mutctx_mean_delta",
        "esmc_mutctx_min_delta",
        "HD_to_train",
        "mean_plddt",
        "ptm",
        "six_site_min_plddt",
    ]]
    .head(10)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()
print("Output files:")
print(" ", FINAL_TSV)
print(" ", TOP10_TSV)
print(" ", TOP10_SEQUENCE_TSV)
print(" ", TOP10_FASTA)
print(" ", RANKED_FASTA)
print(" ", REPORT)
