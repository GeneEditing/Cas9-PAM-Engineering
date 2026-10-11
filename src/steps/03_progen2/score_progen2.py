import os
from pathlib import Path
PACKAGE_ROOT = Path(os.environ["CAS9_ROOT"])
PACKAGE_RUN = Path(os.environ["CAS9_RUN"])
#!/usr/bin/env python3
import csv
import hashlib
import math
import os
import re
import statistics
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO = PACKAGE_ROOT / "progen2"
MODEL = REPO / "checkpoints/progen2-base"
TOKENIZER = REPO / "tokenizer.json"

INPUT = PACKAGE_RUN / "inputs/03_progen2/candidates_with_wt.fasta"

RUN = PACKAGE_RUN / "results/03_progen2"

INPUT = Path(
    os.environ.get("PROGEN2_INPUT", str(INPUT))
)
RUN = Path(
    os.environ.get("PROGEN2_RUN", str(RUN))
)

RAW = RUN / "progen2_full_raw.tsv"
RANKED = RUN / "progen2_full_ranked.tsv"
SUMMARY = RUN / "progen2_full_summary.txt"

EXPECTED_N = int(
    os.environ.get("PROGEN2_EXPECTED_N", "3421")
)
MAX_TOKENS = 2048

# ProGen2官方25种氨基酸字符
ALPHABET = "ABCDEFGHIKLMNOPQRSTUVWXYZ"

FIELDS = [
    "sequence_id",
    "length",
    "sequence_sha256",
    "is_wt",
    "status",
    "error",
    "n_scored_residues",
    "ll_lr_sum",
    "ll_lr_mean",
    "ll_rl_sum",
    "ll_rl_mean",
    "ll_bi_sum",
    "ll_bi_mean",
    "nll_bi_mean",
    "perplexity_bi",
    "elapsed_seconds",
]


def read_fasta(path):
    records = []
    seen = set()
    header = None
    chunks = []

    def save():
        if header is None:
            return

        sequence_id = header.split()[0]
        sequence = "".join(chunks).replace(" ", "").upper()

        if not sequence_id or not sequence:
            raise ValueError("empty FASTA ID or sequence")

        if sequence_id in seen:
            raise ValueError(
                "duplicate FASTA ID: " + sequence_id
            )

        bad = sorted(set(sequence) - set(ALPHABET))
        if bad:
            raise ValueError(
                "{} has unsupported symbols: {}".format(
                    sequence_id,
                    bad,
                )
            )

        seen.add(sequence_id)
        records.append((sequence_id, sequence))

    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                save()
                header = line[1:].strip()
                chunks = []
            else:
                if header is None:
                    raise ValueError(
                        "sequence before header at line {}".format(
                            line_number
                        )
                    )
                chunks.append(line)

    save()
    return records


def find_wt_id(records):
    ids = [
        sequence_id
        for sequence_id, _ in records
    ]

    strong = [
        sequence_id
        for sequence_id in ids
        if sequence_id.lower()
        in {
            "wt",
            "spcas9_wt",
            "spcas9-wt",
            "spcas9.wt",
        }
    ]

    if len(strong) == 1:
        return strong[0]

    pattern = re.compile(
        r"(^|[|_.-])"
        r"(wt|wildtype|wild_type)"
        r"($|[|_.-])",
        re.I,
    )

    candidates = [
        sequence_id
        for sequence_id in ids
        if pattern.search(sequence_id)
    ]

    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        "cannot identify exactly one WT FASTA ID; "
        "possible matches: {}".format(candidates)
    )


def seq_hash(sequence):
    return hashlib.sha256(
        sequence.encode("ascii")
    ).hexdigest()


def f12(value):
    return "{:.12f}".format(float(value))


def load_existing():
    last = OrderedDict()
    ok_hashes = {}

    if not RAW.exists() or RAW.stat().st_size == 0:
        return last, ok_hashes

    with RAW.open(newline="") as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        if reader.fieldnames != FIELDS:
            raise ValueError(
                "existing output header does not "
                "match this script"
            )

        for row in reader:
            if (
                None in row.values()
                or not row.get("sequence_id")
            ):
                continue

            last[row["sequence_id"]] = row

            if row.get("status") == "ok":
                ok_hashes[
                    row["sequence_id"]
                ] = row["sequence_sha256"]

    return last, ok_hashes


def load_model():
    sys.path.insert(0, str(REPO))

    from models.progen.modeling_progen import (
        ProGenForCausalLM,
    )

    model = (
        ProGenForCausalLM.from_pretrained(
            str(MODEL),
            torch_dtype=torch.float16,
        )
        .to("cuda")
        .eval()
    )

    return model


def token_metadata(tokenizer):
    terminal_ids = set()

    for terminal in "12":
        ids = tokenizer.encode(terminal).ids

        if len(ids) != 1:
            raise ValueError(
                "terminal {} is not one token".format(
                    terminal
                )
            )

        terminal_ids.add(ids[0])

    aa_ids = []

    for amino_acid in ALPHABET:
        ids = tokenizer.encode(
            amino_acid
        ).ids

        if len(ids) != 1:
            raise ValueError(
                "amino acid {} is not one token".format(
                    amino_acid
                )
            )

        aa_ids.append(ids[0])

    if len(set(aa_ids)) != 25:
        raise ValueError(
            "unexpected ProGen2 amino-acid "
            "token mapping"
        )

    aa_tensor = torch.tensor(
        aa_ids,
        dtype=torch.long,
        device="cuda",
    )

    token_to_local = {
        token_id: index
        for index, token_id in enumerate(aa_ids)
    }

    return (
        terminal_ids,
        aa_tensor,
        token_to_local,
    )


def score_direction(
    wrapped,
    model,
    tokenizer,
    metadata,
):
    (
        terminal_ids,
        aa_tensor,
        token_to_local,
    ) = metadata

    encoded = tokenizer.encode(
        wrapped
    ).ids

    if len(encoded) > MAX_TOKENS:
        raise ValueError(
            "{} tokens exceeds MAX_TOKENS={}; "
            "refusing to truncate".format(
                len(encoded),
                MAX_TOKENS,
            )
        )

    input_ids = torch.tensor(
        encoded,
        dtype=torch.long,
        device="cuda",
    )

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            logits = model(
                input_ids,
                labels=input_ids,
            ).logits

    if logits.dim() == 3:
        logits = logits[0]

    logits = logits[:-1]
    target = input_ids[1:]

    # 去掉最后的终止符，不将其计入蛋白质分数
    if int(target[-1]) in terminal_ids:
        logits = logits[:-1]
        target = target[:-1]

    target_ids = (
        target.detach()
        .cpu()
        .tolist()
    )

    bad = sorted(
        set(target_ids)
        - set(token_to_local)
    )

    if bad:
        raise ValueError(
            "non-amino-acid target token IDs: "
            "{}".format(bad)
        )

    target_local = torch.tensor(
        [
            token_to_local[token_id]
            for token_id in target_ids
        ],
        dtype=torch.long,
        device="cuda",
    )

    # 与ProGen2官方likelihood.py一致：
    # 只在25种氨基酸token中归一化
    aa_logits = logits.index_select(
        dim=-1,
        index=aa_tensor,
    ).float()

    nll = F.cross_entropy(
        aa_logits,
        target_local,
        reduction="none",
    )

    ll_sum = -nll.sum().item()
    n_residues = target_local.numel()
    ll_mean = ll_sum / n_residues

    return (
        ll_sum,
        ll_mean,
        n_residues,
    )


def score_sequence(
    sequence,
    model,
    tokenizer,
    metadata,
):
    # 左到右：1 + 序列 + 2
    lr_sum, lr_mean, lr_n = score_direction(
        "1" + sequence + "2",
        model,
        tokenizer,
        metadata,
    )

    # 右到左：2 + 反向序列 + 1
    rl_sum, rl_mean, rl_n = score_direction(
        "2" + sequence[::-1] + "1",
        model,
        tokenizer,
        metadata,
    )

    if (
        lr_n != len(sequence)
        or rl_n != len(sequence)
    ):
        raise ValueError(
            "scored residue count does not "
            "equal sequence length"
        )

    ll_bi_sum = (
        lr_sum + rl_sum
    ) / 2.0

    ll_bi_mean = (
        lr_mean + rl_mean
    ) / 2.0

    return {
        "n_scored_residues": lr_n,
        "ll_lr_sum": lr_sum,
        "ll_lr_mean": lr_mean,
        "ll_rl_sum": rl_sum,
        "ll_rl_mean": rl_mean,
        "ll_bi_sum": ll_bi_sum,
        "ll_bi_mean": ll_bi_mean,
        "nll_bi_mean": -ll_bi_mean,
        "perplexity_bi": math.exp(
            -ll_bi_mean
        ),
    }


def write_ranked(records, wt_id):
    last, _ = load_existing()

    rows = [
        row
        for row in last.values()
        if row["status"] == "ok"
    ]

    rows.sort(
        key=lambda row: float(
            row["ll_bi_mean"]
        ),
        reverse=True,
    )

    wt_rows = [
        row
        for row in rows
        if row["sequence_id"] == wt_id
    ]

    wt_score = (
        float(
            wt_rows[0]["ll_bi_mean"]
        )
        if wt_rows
        else None
    )

    candidate_scores = [
        float(row["ll_bi_mean"])
        for row in rows
        if row["sequence_id"] != wt_id
    ]

    candidate_median = (
        statistics.median(
            candidate_scores
        )
        if candidate_scores
        else None
    )

    extra_fields = [
        "delta_ll_bi_mean_vs_wt",
        "candidate_median_ll_bi_mean",
        "pass_candidate_median",
        "rank_ll_bi_mean_desc",
    ]

    temporary = RANKED.with_suffix(
        ".tmp"
    )

    with temporary.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                FIELDS
                + extra_fields
            ),
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()

        for rank, row in enumerate(
            rows,
            1,
        ):
            output_row = dict(row)
            score = float(
                row["ll_bi_mean"]
            )

            output_row[
                "delta_ll_bi_mean_vs_wt"
            ] = (
                ""
                if wt_score is None
                else f12(
                    score - wt_score
                )
            )

            output_row[
                "candidate_median_ll_bi_mean"
            ] = (
                ""
                if candidate_median is None
                else f12(
                    candidate_median
                )
            )

            output_row[
                "pass_candidate_median"
            ] = (
                ""
                if (
                    row["sequence_id"]
                    == wt_id
                    or candidate_median
                    is None
                )
                else str(
                    int(
                        score
                        >= candidate_median
                    )
                )
            )

            output_row[
                "rank_ll_bi_mean_desc"
            ] = rank

            writer.writerow(
                output_row
            )

        handle.flush()
        os.fsync(handle.fileno())

    os.replace(
        str(temporary),
        str(RANKED),
    )

    passed = (
        sum(
            score >= candidate_median
            for score
            in candidate_scores
        )
        if candidate_median
        is not None
        else 0
    )

    summary_lines = [
        (
            "ProGen2-base full-length "
            "bidirectional scoring"
        ),
        "Input records: {}".format(
            len(records)
        ),
        "Successful records: {}".format(
            len(rows)
        ),
        "WT ID: {}".format(
            wt_id
        ),
        "WT ll_bi_mean: {}".format(
            "NA"
            if wt_score is None
            else f12(wt_score)
        ),
        "Candidate count: {}".format(
            len(candidate_scores)
        ),
        (
            "Candidate median "
            "ll_bi_mean: {}"
        ).format(
            "NA"
            if candidate_median is None
            else f12(
                candidate_median
            )
        ),
        (
            "Candidates passing "
            "median: {}"
        ).format(passed),
        (
            "Screening direction: "
            "ll_bi_mean higher is better"
        ),
    ]

    SUMMARY.write_text(
        "\n".join(
            summary_lines
        )
        + "\n"
    )


def main():
    os.environ[
        "TOKENIZERS_PARALLELISM"
    ] = "false"

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    for path in [
        REPO,
        MODEL,
        TOKENIZER,
        INPUT,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(path)
            )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available "
            "in the current environment"
        )

    RUN.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = read_fasta(
        INPUT
    )

    if len(records) != EXPECTED_N:
        raise ValueError(
            "FASTA has {} records; "
            "expected {}".format(
                len(records),
                EXPECTED_N,
            )
        )

    wt_id = find_wt_id(
        records
    )

    print(
        "records =",
        len(records),
        flush=True,
    )
    print(
        "WT ID =",
        wt_id,
        flush=True,
    )

    _, ok_hashes = load_existing()

    for sequence_id, sequence in records:
        if (
            sequence_id
            in ok_hashes
            and ok_hashes[
                sequence_id
            ]
            != seq_hash(sequence)
        ):
            raise ValueError(
                "completed sequence changed "
                "for ID "
                + sequence_id
            )

    pending = [
        (sequence_id, sequence)
        for sequence_id, sequence
        in records
        if sequence_id
        not in ok_hashes
    ]

    print(
        "completed =",
        len(records) - len(pending),
        flush=True,
    )
    print(
        "pending =",
        len(pending),
        flush=True,
    )

    if not pending:
        write_ranked(
            records,
            wt_id,
        )
        print(
            "nothing pending; "
            "ranked files refreshed",
            flush=True,
        )
        return

    print(
        "loading model:",
        MODEL,
        flush=True,
    )

    model = load_model()

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER)
    )

    metadata = token_metadata(
        tokenizer
    )

    print(
        "model loaded",
        flush=True,
    )

    new_file = (
        not RAW.exists()
        or RAW.stat().st_size == 0
    )

    with RAW.open(
        "a",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )

        if new_file:
            writer.writeheader()
            handle.flush()

        completed_before = (
            len(records)
            - len(pending)
        )

        for pending_index, (
            sequence_id,
            sequence,
        ) in enumerate(
            pending,
            1,
        ):
            index = (
                completed_before
                + pending_index
            )

            started = time.time()

            row = {
                field: ""
                for field in FIELDS
            }

            row.update({
                "sequence_id":
                    sequence_id,
                "length":
                    len(sequence),
                "sequence_sha256":
                    seq_hash(sequence),
                "is_wt":
                    str(
                        int(
                            sequence_id
                            == wt_id
                        )
                    ),
            })

            print(
                "[{}/{}] START {}".format(
                    index,
                    len(records),
                    sequence_id,
                ),
                flush=True,
            )

            try:
                scores = score_sequence(
                    sequence,
                    model,
                    tokenizer,
                    metadata,
                )

                row["status"] = "ok"

                for key, value in scores.items():
                    row[key] = (
                        value
                        if key
                        == "n_scored_residues"
                        else f12(value)
                    )

                print(
                    (
                        "[{}/{}] DONE {} "
                        "ll_bi_mean={:.8f}"
                    ).format(
                        index,
                        len(records),
                        sequence_id,
                        scores[
                            "ll_bi_mean"
                        ],
                    ),
                    flush=True,
                )

            except Exception as error:
                row["status"] = "error"

                row["error"] = (
                    " ".join(
                        str(error)
                        .replace("\t", " ")
                        .splitlines()
                    )[:2000]
                )

                print(
                    (
                        "[{}/{}] ERROR {}: {}"
                    ).format(
                        index,
                        len(records),
                        sequence_id,
                        row["error"],
                    ),
                    flush=True,
                )

                traceback.print_exc()
                torch.cuda.empty_cache()

            row[
                "elapsed_seconds"
            ] = f12(
                time.time()
                - started
            )

            writer.writerow(row)

            # 每条结果立即落盘
            handle.flush()

    write_ranked(
        records,
        wt_id,
    )

    print(
        "raw =",
        RAW,
        flush=True,
    )
    print(
        "ranked =",
        RANKED,
        flush=True,
    )
    print(
        "summary =",
        SUMMARY,
        flush=True,
    )


if __name__ == "__main__":
    main()
