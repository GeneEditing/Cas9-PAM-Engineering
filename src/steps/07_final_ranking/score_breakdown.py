import os
from pathlib import Path
PACKAGE_ROOT = Path(os.environ["CAS9_ROOT"])
PACKAGE_RUN = Path(os.environ["CAS9_RUN"])
from pathlib import Path
import pandas as pd

RESULTS = PACKAGE_RUN / "results/07_final_ranking"

INPUT = RESULTS / "final_ranking_all38.tsv"

OUTPUT_TSV = RESULTS / "final_score_breakdown_all38.tsv"
OUTPUT_CSV = RESULTS / "final_score_breakdown_all38.csv"
OUTPUT_REPORT = RESULTS / "final_score_breakdown_report.txt"

if not INPUT.exists():
    raise FileNotFoundError(
        f"请先运行最终排序脚本，缺少：{INPUT}"
    )

df = pd.read_csv(INPUT, sep="\t")

if len(df) != 38:
    raise RuntimeError(
        f"最终排序表有 {len(df)} 条，不是38条"
    )

# ============================================================
# Eight direct contributions to the final 100-point score
# ============================================================

df["C_AT_max_100"] = (
    22.5 * df["P_AT_max"]
)

df["C_selectivity_100"] = (
    22.5 * df["P_selectivity"]
)

df["C_progen2_100"] = (
    13.5 * df["P_progen2_delta"]
)

df["C_esmc_mean_100"] = (
    13.5 * df["P_esmc_mean"]
)

df["C_esmc_worst_100"] = (
    9.0 * df["P_esmc_worst"]
)

df["C_HD_100"] = (
    9.0 * df["P_neg_HD"]
)

df["C_mean_plddt_100"] = (
    3.0 * df["P_mean_plddt"]
)

df["C_six_site_min_plddt_100"] = (
    7.0 * df["P_six_site_min_plddt"]
)

# ============================================================
# Four major score contributions
# ============================================================

df["C_PAM_100"] = (
    df["C_AT_max_100"]
    + df["C_selectivity_100"]
)

df["C_LM_100"] = (
    df["C_progen2_100"]
    + df["C_esmc_mean_100"]
)

df["C_R_100"] = (
    df["C_esmc_worst_100"]
    + df["C_HD_100"]
)

df["C_AF2_100"] = (
    df["C_mean_plddt_100"]
    + df["C_six_site_min_plddt_100"]
)

df["contribution_sum_100"] = (
    df["C_PAM_100"]
    + df["C_LM_100"]
    + df["C_R_100"]
    + df["C_AF2_100"]
)

df["score_check_difference"] = (
    df["contribution_sum_100"]
    - df["final_score_100"]
)

maximum_error = (
    df["score_check_difference"]
    .abs()
    .max()
)

if maximum_error > 1e-5:
    raise RuntimeError(
        f"贡献值加和与最终分不一致，最大误差={maximum_error}"
    )

# ============================================================
# Human-readable direction labels
# ============================================================

df["HD_direction"] = "lower_is_better"
df["other_metric_direction"] = "higher_is_better"

# ============================================================
# Arrange columns
# ============================================================

identity_columns = [
    "final_rank",
    "pareto_front",
    "rank_within_front",
    "candidate_id",
    "genotype",
    "best_AT_PAM",
]

final_columns = [
    "final_score_100",
    "contribution_sum_100",
    "score_check_difference",
]

major_score_columns = [
    "S_PAM",
    "S_LM",
    "S_R",
    "S_AF2",
    "S_final",
]

major_contribution_columns = [
    "C_PAM_100",
    "C_LM_100",
    "C_R_100",
    "C_AF2_100",
]

raw_metric_columns = [
    "AT_max",
    "selectivity",
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
]

percentile_columns = [
    "P_AT_max",
    "P_selectivity",
    "P_progen2_delta",
    "P_esmc_mean",
    "P_esmc_worst",
    "P_neg_HD",
    "P_mean_plddt",
    "P_six_site_min_plddt",
]

direct_contribution_columns = [
    "C_AT_max_100",
    "C_selectivity_100",
    "C_progen2_100",
    "C_esmc_mean_100",
    "C_esmc_worst_100",
    "C_HD_100",
    "C_mean_plddt_100",
    "C_six_site_min_plddt_100",
]

audit_columns = [
    "pass_structural_gate",
    "pass_both_models",
    "HD_direction",
    "other_metric_direction",
]

ordered_columns = (
    identity_columns
    + final_columns
    + major_score_columns
    + major_contribution_columns
    + raw_metric_columns
    + percentile_columns
    + direct_contribution_columns
    + audit_columns
)

ordered_columns = [
    column
    for column in ordered_columns
    if column in df.columns
]

remaining_columns = [
    column
    for column in df.columns
    if column not in ordered_columns
]

output = df[
    ordered_columns + remaining_columns
].copy()

output.to_csv(
    OUTPUT_TSV,
    sep="\t",
    index=False,
    float_format="%.8f"
)

output.to_csv(
    OUTPUT_CSV,
    index=False,
    float_format="%.8f"
)

# ============================================================
# Complete human-readable report for all 38 candidates
# ============================================================

report_columns = [
    "final_rank",
    "pareto_front",
    "rank_within_front",
    "candidate_id",
    "final_score_100",
    "C_PAM_100",
    "C_LM_100",
    "C_R_100",
    "C_AF2_100",
    "C_AT_max_100",
    "C_selectivity_100",
    "C_progen2_100",
    "C_esmc_mean_100",
    "C_esmc_worst_100",
    "C_HD_100",
    "C_mean_plddt_100",
    "C_six_site_min_plddt_100",
]

with OUTPUT_REPORT.open("w") as handle:
    handle.write(
        "Final score breakdown for all 38 candidates\n"
    )
    handle.write("=" * 100 + "\n\n")

    handle.write(
        "Final score on a 100-point scale:\n"
        "  22.5*P(AT_max)\n"
        "  + 22.5*P(Selectivity)\n"
        "  + 13.5*P(ProGen2 delta)\n"
        "  + 13.5*P(ESM-C mean delta)\n"
        "  + 9.0*P(ESM-C worst delta)\n"
        "  + 9.0*P(-HD_to_train)\n"
        "  + 3.0*P(mean pLDDT)\n"
        "  + 7.0*P(six-site minimum pLDDT)\n\n"
    )

    handle.write(
        "Ranking hierarchy:\n"
        "  Pareto Front ascending first;\n"
        "  final score descending within each Front.\n\n"
    )

    handle.write(
        output[report_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    handle.write("\n\n")
    handle.write(
        f"Maximum score check error: {maximum_error:.12g}\n"
    )

print("=" * 80)
print("SCORE BREAKDOWN COMPLETED")
print("=" * 80)
print("候选数量：", len(output))
print("最大加和误差：", maximum_error)
print()
print("输出文件：")
print(" ", OUTPUT_TSV)
print(" ", OUTPUT_CSV)
print(" ", OUTPUT_REPORT)

print()
print("===== SCORE BREAKDOWN PREVIEW =====")

print(
    output[[
        "final_rank",
        "pareto_front",
        "candidate_id",
        "final_score_100",
        "C_PAM_100",
        "C_LM_100",
        "C_R_100",
        "C_AF2_100",
    ]]
    .head(10)
    .to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)
