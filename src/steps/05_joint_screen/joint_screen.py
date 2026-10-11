import os
from pathlib import Path
PACKAGE_ROOT = Path(os.environ["CAS9_ROOT"])
PACKAGE_RUN = Path(os.environ["CAS9_RUN"])
#!/usr/bin/env python3
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median

ROOT = PACKAGE_ROOT / "PAMmla"

PROGEN_FILES = [PACKAGE_RUN / "results/03_progen2/progen2_full_raw.tsv"]

ESMC_FILE = PACKAGE_RUN / "results/04_esmc/esmc_summary.csv"

PAM_FILE = PACKAGE_RUN / "results/02_shortlist/final_shortlist_v2.csv"

RUN_DIR = PACKAGE_RUN

RESULTS = PACKAGE_RUN / "results/05_joint_screen"

EXPECTED = 3420
KEEP_MAX_FRONT = 4


def read_table(path, delimiter):
    with path.open(newline="") as handle:
        return list(
            csv.DictReader(
                handle,
                delimiter=delimiter,
            )
        )


def require_columns(
    rows,
    columns,
    label,
):
    if not rows:
        raise ValueError(
            label + " is empty"
        )

    missing = [
        column
        for column in columns
        if column not in rows[0]
    ]

    if missing:
        raise ValueError(
            "{} missing columns: {}".format(
                label,
                missing,
            )
        )


def unique_by(
    rows,
    key_function,
    label,
):
    result = {}

    for row in rows:
        key = key_function(row)

        if not key:
            raise ValueError(
                label + " contains an empty key"
            )

        if key in result:
            raise ValueError(
                "{} duplicate key: {}".format(
                    label,
                    key,
                )
            )

        result[key] = row

    return result


def f12(value):
    return "{:.12f}".format(
        float(value)
    )


def write_tsv(
    path,
    rows,
    fields,
):
    with path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)


def pareto_fronts(rows):
    number = len(rows)

    dominates = [
        []
        for _ in range(number)
    ]

    dominated_count = [
        0
        for _ in range(number)
    ]

    for i in range(number):
        at_i = float(
            rows[i]["AT_max"]
        )

        selectivity_i = float(
            rows[i]["selectivity"]
        )

        for j in range(
            i + 1,
            number,
        ):
            at_j = float(
                rows[j]["AT_max"]
            )

            selectivity_j = float(
                rows[j]["selectivity"]
            )

            i_dominates_j = (
                at_i >= at_j
                and selectivity_i
                >= selectivity_j
                and (
                    at_i > at_j
                    or selectivity_i
                    > selectivity_j
                )
            )

            j_dominates_i = (
                at_j >= at_i
                and selectivity_j
                >= selectivity_i
                and (
                    at_j > at_i
                    or selectivity_j
                    > selectivity_i
                )
            )

            if i_dominates_j:
                dominates[i].append(j)
                dominated_count[j] += 1

            elif j_dominates_i:
                dominates[j].append(i)
                dominated_count[i] += 1

    current_front = [
        index
        for index, count
        in enumerate(dominated_count)
        if count == 0
    ]

    front_number = 1
    assigned = 0

    while current_front:
        next_front = []

        for i in current_front:
            rows[i][
                "pareto_front"
            ] = str(front_number)

            assigned += 1

            for j in dominates[i]:
                dominated_count[j] -= 1

                if (
                    dominated_count[j]
                    == 0
                ):
                    next_front.append(j)

        current_front = next_front
        front_number += 1

    if assigned != number:
        raise RuntimeError(
            "Pareto assignment incomplete: "
            "{}/{}".format(
                assigned,
                number,
            )
        )


def main():
    input_paths = (
        PROGEN_FILES
        + [
            ESMC_FILE,
            PAM_FILE,
        ]
    )

    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(
                str(path)
            )

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 合并GPU 5和GPU 7的ProGen2结果
    progen_candidates = []
    wt_rows = []

    for path in PROGEN_FILES:
        rows = read_table(
            path,
            "\t",
        )

        require_columns(
            rows,
            [
                "sequence_id",
                "sequence_sha256",
                "is_wt",
                "status",
                "ll_lr_mean",
                "ll_rl_mean",
                "ll_bi_mean",
            ],
            str(path),
        )

        bad_rows = [
            row
            for row in rows
            if row["status"] != "ok"
        ]

        if bad_rows:
            raise ValueError(
                "{} has {} non-ok rows".format(
                    path,
                    len(bad_rows),
                )
            )

        for row in rows:
            if (
                row["sequence_id"]
                == "SpCas9_WT"
                or row["is_wt"] == "1"
            ):
                wt_rows.append(row)
            else:
                progen_candidates.append(
                    row
                )

    # 两张卡都计算了WT
    if len(wt_rows) != 1:
        raise ValueError(
            "expected one WT row, "
            "found {}".format(
                len(wt_rows)
            )
        )

    wt_hashes = {
        row["sequence_sha256"]
        for row in wt_rows
    }

    if len(wt_hashes) != 1:
        raise ValueError(
            "WT sequence hashes differ "
            "between GPU shards"
        )

    wt_scores = {
        row["ll_bi_mean"]
        for row in wt_rows
    }

    if len(wt_scores) != 1:
        raise ValueError(
            "WT ProGen2 scores differ "
            "between GPU shards"
        )

    wt_score = float(
        wt_rows[0]["ll_bi_mean"]
    )

    progen = unique_by(
        progen_candidates,
        lambda row: (
            row["sequence_id"]
            .split("|")[-1]
        ),
        "ProGen2",
    )

    if len(progen) != EXPECTED:
        raise ValueError(
            "ProGen2 candidates: {}; "
            "expected {}".format(
                len(progen),
                EXPECTED,
            )
        )

    # 读取已有ESM-C结果
    esmc_rows = read_table(
        ESMC_FILE,
        ",",
    )

    require_columns(
        esmc_rows,
        [
            "candidate_id",
            "status",
            "mutctx_mean_delta",
            "mutctx_min_delta",
            "worst_mutctx_mutation",
        ],
        "ESM-C",
    )

    esmc_candidates = [
        row
        for row in esmc_rows
        if row["candidate_id"]
        != "SpCas9_WT"
    ]

    bad_esmc = [
        row
        for row in esmc_candidates
        if row["status"] != "ok"
    ]

    if bad_esmc:
        raise ValueError(
            "ESM-C has {} non-ok "
            "candidate rows".format(
                len(bad_esmc)
            )
        )

    esmc = unique_by(
        esmc_candidates,
        lambda row: (
            row["candidate_id"]
            .split("|")[-1]
        ),
        "ESM-C",
    )

    # 读取PAMmla指标
    pam_rows = read_table(
        PAM_FILE,
        ",",
    )

    require_columns(
        pam_rows,
        [
            "variant",
            "AT_max",
            "NGG_max",
            "selectivity",
            "best_AT_PAM",
            "HD_to_train",
            "tier",
            "mutations",
        ],
        "PAMmla",
    )

    pam = unique_by(
        pam_rows,
        lambda row: row["variant"],
        "PAMmla",
    )

    common = (
        set(progen)
        & set(esmc)
        & set(pam)
    )

    if (
        len(esmc) != EXPECTED
        or len(pam) != EXPECTED
        or len(common) != EXPECTED
    ):
        raise ValueError(
            (
                "key counts: "
                "ProGen2={} "
                "ESM-C={} "
                "PAMmla={} "
                "common={}"
            ).format(
                len(progen),
                len(esmc),
                len(pam),
                len(common),
            )
        )

    # 两个模型分别计算完整3420条候选的中位数
    progen_threshold = median(
        float(
            progen[genotype][
                "ll_bi_mean"
            ]
        )
        for genotype in common
    )

    esmc_threshold = median(
        float(
            esmc[genotype][
                "mutctx_mean_delta"
            ]
        )
        for genotype in common
    )

    merged = []

    for genotype in sorted(common):
        progen_row = progen[genotype]
        esmc_row = esmc[genotype]
        pam_row = pam[genotype]

        if (
            progen_row["sequence_id"]
            != esmc_row["candidate_id"]
        ):
            raise ValueError(
                (
                    "candidate ID mismatch "
                    "for {}: {} != {}"
                ).format(
                    genotype,
                    progen_row[
                        "sequence_id"
                    ],
                    esmc_row[
                        "candidate_id"
                    ],
                )
            )

        progen_score = float(
            progen_row["ll_bi_mean"]
        )

        esmc_score = float(
            esmc_row[
                "mutctx_mean_delta"
            ]
        )

        pass_progen = (
            progen_score
            >= progen_threshold
        )

        pass_esmc = (
            esmc_score
            >= esmc_threshold
        )

        merged.append({
            "candidate_id":
                progen_row["sequence_id"],
            "genotype":
                genotype,
            "AT_max":
                pam_row["AT_max"],
            "NGG_max":
                pam_row["NGG_max"],
            "selectivity":
                pam_row["selectivity"],
            "best_AT_PAM":
                pam_row["best_AT_PAM"],
            "HD_to_train":
                pam_row["HD_to_train"],
            "tier":
                pam_row["tier"],
            "mutations":
                pam_row["mutations"],
            "progen2_ll_lr_mean":
                progen_row[
                    "ll_lr_mean"
                ],
            "progen2_ll_rl_mean":
                progen_row[
                    "ll_rl_mean"
                ],
            "progen2_ll_bi_mean":
                progen_row[
                    "ll_bi_mean"
                ],
            "progen2_delta_vs_wt":
                f12(
                    progen_score
                    - wt_score
                ),
            "progen2_median_threshold":
                f12(
                    progen_threshold
                ),
            "pass_progen2_median":
                str(
                    int(pass_progen)
                ),
            "esmc_mutctx_mean_delta":
                esmc_row[
                    "mutctx_mean_delta"
                ],
            "esmc_mutctx_min_delta":
                esmc_row[
                    "mutctx_min_delta"
                ],
            "esmc_worst_mutation":
                esmc_row[
                    "worst_mutctx_mutation"
                ],
            "esmc_median_threshold":
                f12(
                    esmc_threshold
                ),
            "pass_esmc_median":
                str(
                    int(pass_esmc)
                ),
            "pass_both_models":
                str(
                    int(
                        pass_progen
                        and pass_esmc
                    )
                ),
            "pareto_front":
                "",
        })

    fields = list(
        merged[0].keys()
    )

    dual_pass = [
        row.copy()
        for row in merged
        if row[
            "pass_both_models"
        ] == "1"
    ]

    # 只有通过双模型门槛者才进入Pareto
    pareto_fronts(
        dual_pass
    )

    dual_pass.sort(
        key=lambda row: (
            int(
                row["pareto_front"]
            ),
            -float(
                row["AT_max"]
            ),
            -float(
                row["selectivity"]
            ),
            row["candidate_id"],
        )
    )

    fronts_1_to_4 = [
        row
        for row in dual_pass
        if int(
            row["pareto_front"]
        ) <= KEEP_MAX_FRONT
    ]

    write_tsv(
        RESULTS
        / "00_merged_all_3420.tsv",
        merged,
        fields,
    )

    write_tsv(
        RESULTS
        / "01_dual_model_median_pass.tsv",
        dual_pass,
        fields,
    )

    write_tsv(
        RESULTS
        / "02_dual_pass_with_pareto.tsv",
        dual_pass,
        fields,
    )

    write_tsv(
        RESULTS
        / "03_fronts_1_to_4.tsv",
        fronts_1_to_4,
        fields,
    )

    (
        RESULTS
        / "03_fronts_1_to_4_ids.txt"
    ).write_text(
        "\n".join(
            row["candidate_id"]
            for row in fronts_1_to_4
        )
        + "\n"
    )

    front_counts = Counter(
        int(row["pareto_front"])
        for row in dual_pass
    )

    audit = {
        "expected_candidates":
            EXPECTED,
        "progen2_candidates":
            len(progen),
        "esmc_candidates":
            len(esmc),
        "pammla_candidates":
            len(pam),
        "common_candidates":
            len(common),
        "progen2_wt_score":
            wt_score,
        "progen2_median_threshold":
            progen_threshold,
        "esmc_median_threshold":
            esmc_threshold,
        "pass_progen2":
            sum(
                row[
                    "pass_progen2_median"
                ] == "1"
                for row in merged
            ),
        "pass_esmc":
            sum(
                row[
                    "pass_esmc_median"
                ] == "1"
                for row in merged
            ),
        "pass_both_models":
            len(dual_pass),
        "front_counts":
            dict(
                sorted(
                    front_counts.items()
                )
            ),
        "fronts_1_to_4_count":
            len(fronts_1_to_4),
    }

    (
        RESULTS
        / "screening_audit.json"
    ).write_text(
        json.dumps(
            audit,
            indent=2,
        )
        + "\n"
    )

    report = [
        (
            "ProGen2 full-length / "
            "ESM-C / PAMmla joint screening"
        ),
        "=" * 65,
        "Input candidates: {}".format(
            len(common)
        ),
        (
            "ProGen2 median threshold: {}"
        ).format(
            f12(progen_threshold)
        ),
        (
            "ESM-C median threshold: {}"
        ).format(
            f12(esmc_threshold)
        ),
        "Passed ProGen2 median: {}".format(
            audit["pass_progen2"]
        ),
        "Passed ESM-C median: {}".format(
            audit["pass_esmc"]
        ),
        (
            "Passed both model medians: {}"
        ).format(
            len(dual_pass)
        ),
        (
            "Pareto objectives: "
            "AT_max up, selectivity up"
        ),
    ]

    for front in sorted(front_counts):
        report.append(
            "Front {}: {}".format(
                front,
                front_counts[front],
            )
        )

    report.extend([
        "Selected fronts: 1-{}".format(
            KEEP_MAX_FRONT
        ),
        "Selected candidates: {}".format(
            len(fronts_1_to_4)
        ),
        "Results directory: {}".format(
            RESULTS
        ),
    ])

    (
        RESULTS
        / "screening_report.txt"
    ).write_text(
        "\n".join(report)
        + "\n"
    )

    print(
        "\n".join(report)
    )


if __name__ == "__main__":
    main()
