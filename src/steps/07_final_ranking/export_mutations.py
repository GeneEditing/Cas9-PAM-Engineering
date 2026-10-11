import os
from pathlib import Path
PACKAGE_ROOT = Path(os.environ["CAS9_ROOT"])
PACKAGE_RUN = Path(os.environ["CAS9_RUN"])
from pathlib import Path
import pandas as pd

RESULTS = PACKAGE_RUN / "results/07_final_ranking"

TOP10_TSV = RESULTS / "final_top10.tsv"

CANDIDATE_FASTA = PACKAGE_RUN / "inputs/07_final_ranking/all38_candidates.fasta"

OUTPUT_TSV = RESULTS / "final_top10_six_site_mutations.tsv"
OUTPUT_CSV = RESULTS / "final_top10_six_site_mutations.csv"
OUTPUT_TXT = RESULTS / "final_top10_six_site_mutations.txt"

# SpCas9 numbering, 1-based
POSITIONS = [1135, 1136, 1218, 1219, 1335, 1337]
WT_RESIDUES = ["D", "S", "G", "E", "R", "T"]
WT_GENOTYPE = "".join(WT_RESIDUES)

if not TOP10_TSV.exists():
    raise FileNotFoundError(
        f"缺少最终前10文件：{TOP10_TSV}"
    )

top10 = pd.read_csv(TOP10_TSV, sep="\t")

if len(top10) != 10:
    raise RuntimeError(
        f"final_top10.tsv包含 {len(top10)} 条，不是10条"
    )

# ============================================================
# Read reconstructed candidate sequences for validation only
# These sequences will not be written to the output.
# ============================================================

def read_fasta(path):
    records = {}
    header = None
    parts = []

    with path.open() as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(parts)

                header = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)

    if header is not None:
        records[header] = "".join(parts)

    return records

sequences = read_fasta(CANDIDATE_FASTA)

rows = []

for _, row in top10.iterrows():
    candidate_id = str(row["candidate_id"]).strip()
    genotype = str(row["genotype"]).strip().upper()

    if len(genotype) != 6:
        raise RuntimeError(
            f"{candidate_id}的genotype不是6个氨基酸：{genotype}"
        )

    if candidate_id not in sequences:
        raise RuntimeError(
            f"{candidate_id}未在候选FASTA中找到"
        )

    sequence = sequences[candidate_id]

    observed_genotype = "".join(
        sequence[position - 1]
        for position in POSITIONS
    )

    if observed_genotype != genotype:
        raise RuntimeError(
            f"{candidate_id}序列验证失败："
            f"TSV={genotype}，FASTA={observed_genotype}"
        )

    mutations = []
    mutation_by_site = {}

    for position, wt_aa, target_aa in zip(
        POSITIONS,
        WT_RESIDUES,
        genotype
    ):
        if target_aa == wt_aa:
            notation = "WT"
        else:
            notation = f"{wt_aa}{position}{target_aa}"
            mutations.append(notation)

        mutation_by_site[position] = notation

    rows.append({
        "final_rank": int(row["final_rank"]),
        "pareto_front": int(row["pareto_front"]),
        "rank_within_front": int(
            row["rank_within_front"]
        ),
        "candidate_id": candidate_id,
        "six_site_genotype": genotype,
        "AA_1135": genotype[0],
        "AA_1136": genotype[1],
        "AA_1218": genotype[2],
        "AA_1219": genotype[3],
        "AA_1335": genotype[4],
        "AA_1337": genotype[5],
        "mutation_1135": mutation_by_site[1135],
        "mutation_1136": mutation_by_site[1136],
        "mutation_1218": mutation_by_site[1218],
        "mutation_1219": mutation_by_site[1219],
        "mutation_1335": mutation_by_site[1335],
        "mutation_1337": mutation_by_site[1337],
        "n_actual_mutations": len(mutations),
        "mutations_only": ";".join(mutations),
        "final_score_100": float(
            row["final_score_100"]
        ),
    })

output = pd.DataFrame(rows)

output.to_csv(
    OUTPUT_TSV,
    sep="\t",
    index=False,
    float_format="%.4f"
)

output.to_csv(
    OUTPUT_CSV,
    index=False,
    float_format="%.4f"
)

# ============================================================
# Human-readable wet-lab sheet
# ============================================================

with OUTPUT_TXT.open("w") as handle:
    handle.write(
        "Top 10 SpCas9 six-site mutation sheet\n"
    )
    handle.write("=" * 72 + "\n\n")

    handle.write(
        "Numbering: SpCas9 protein sequence, 1-based\n"
    )
    handle.write(
        "WT residues: "
        "D1135 / S1136 / G1218 / "
        "E1219 / R1335 / T1337\n"
    )
    handle.write(
        f"WT six-site genotype: {WT_GENOTYPE}\n\n"
    )

    for record in rows:
        handle.write(
            f"Rank {record['final_rank']} | "
            f"Front {record['pareto_front']} | "
            f"{record['candidate_id']}\n"
        )

        handle.write(
            f"Target genotype: "
            f"{record['six_site_genotype']}\n"
        )

        handle.write(
            f"Point mutations ({record['n_actual_mutations']}): "
            f"{record['mutations_only']}\n"
        )

        handle.write(
            f"Final score: "
            f"{record['final_score_100']:.4f}\n"
        )

        handle.write("-" * 72 + "\n")

print("=" * 80)
print("TOP-10 MUTATION SHEET COMPLETED")
print("=" * 80)

print("WT genotype:", WT_GENOTYPE)
print("Candidate count:", len(output))
print()

print(
    output[[
        "final_rank",
        "pareto_front",
        "candidate_id",
        "six_site_genotype",
        "mutations_only",
        "final_score_100",
    ]].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()
print("Output files:")
print(" ", OUTPUT_TSV)
print(" ", OUTPUT_CSV)
print(" ", OUTPUT_TXT)
