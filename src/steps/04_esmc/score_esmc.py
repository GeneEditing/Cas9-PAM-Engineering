import os
from pathlib import Path
PACKAGE_ROOT = Path(os.environ["CAS9_ROOT"])
PACKAGE_RUN = Path(os.environ["CAS9_RUN"])
#!/usr/bin/env python3

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


MODEL_DIR = str(PACKAGE_ROOT / "esmc_models/ESMC-600M")
AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path):
    records = []
    name = None
    seq = []

    with open(path) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if name is not None:
                    records.append(
                        (name, "".join(seq).upper())
                    )

                name = line[1:].split()[0]
                seq = []

            else:
                seq.append(line)

    if name is not None:
        records.append(
            (name, "".join(seq).upper())
        )

    return records


def find_wt(records, wt_id):
    hits = [
        (name, seq)
        for name, seq in records
        if name == wt_id
    ]

    if len(hits) != 1:
        raise RuntimeError(
            f"WT ID {wt_id!r} matched {len(hits)} records"
        )

    return hits[0]


def find_mutations(wt, seq):
    if len(wt) != len(seq):
        raise RuntimeError(
            f"Length mismatch: WT={len(wt)}, candidate={len(seq)}"
        )

    muts = []

    for pos, (a, b) in enumerate(
        zip(wt, seq),
        start=1,
    ):
        if a != b:
            muts.append(
                (pos, a, b)
            )

    return muts


def encode_sequence(tokenizer, sequence):
    encoded = tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True,
        return_special_tokens_mask=True,
    )

    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]
    special_tokens_mask = encoded["special_tokens_mask"][0]

    residue_positions = torch.where(
        special_tokens_mask == 0
    )[0]

    if len(residue_positions) != len(sequence):
        raise RuntimeError(
            f"Token mismatch: "
            f"sequence={len(sequence)}, "
            f"residue_tokens={len(residue_positions)}"
        )

    return (
        input_ids,
        attention_mask,
        residue_positions,
    )


def load_completed(path):
    done = set()

    if not os.path.exists(path):
        return done

    if os.path.getsize(path) == 0:
        return done

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("status") in {
                "ok",
                "wt",
                "identical_to_wt",
            }:
                done.add(
                    row["candidate_id"]
                )

    return done


def load_detail_done(path):
    done = set()

    if not os.path.exists(path):
        return done

    if os.path.getsize(path) == 0:
        return done

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            cid = row.get("candidate_id")
            mut = row.get("mutation")

            if cid and mut:
                done.add(
                    (cid, mut)
                )

    return done


def chunked(items, size):
    for i in range(
        0,
        len(items),
        size,
    ):
        yield items[i:i + size]


def write_flush(writer, fh, row):
    writer.writerow(row)
    fh.flush()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--summary",
        required=True,
    )

    parser.add_argument(
        "--detail",
        required=True,
    )

    parser.add_argument(
        "--wt-id",
        required=True,
    )

    parser.add_argument(
        "--mask-batch-size",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    Path(args.summary).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = read_fasta(
        args.input
    )

    if not records:
        raise RuntimeError(
            "No FASTA records found"
        )

    wt_id, wt_seq = find_wt(
        records,
        args.wt_id,
    )

    if set(wt_seq) - AA20:
        raise RuntimeError(
            f"WT contains non-standard residues: "
            f"{sorted(set(wt_seq) - AA20)}"
        )

    print("=" * 80, flush=True)
    print("ESM-C local masked-marginal scoring", flush=True)
    print("Model:", MODEL_DIR, flush=True)
    print("Input:", args.input, flush=True)
    print("Sequences:", len(records), flush=True)
    print("WT ID:", wt_id, flush=True)
    print("WT length:", len(wt_seq), flush=True)
    print("Mask batch size:", args.mask_batch_size, flush=True)
    print("=" * 80, flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    device = torch.device(
        "cuda:0"
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True,
    )

    # ==================================================
    # LOCAL OFFLINE ESM-C LOADING
    # ==================================================

    print(
        "Loading tokenizer from local model...",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    print(
        "Loading ESM-C model from local model...",
        flush=True,
    )

    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        low_cpu_mem_usage=False,
    )

    model = model.to(
        device
    ).eval()

    print(
        "ESM-C loaded successfully.",
        flush=True,
    )

    print(
        "Mask token:",
        tokenizer.mask_token,
        flush=True,
    )

    print(
        "Mask token ID:",
        tokenizer.mask_token_id,
        flush=True,
    )

    # ==================================================
    # TOKENIZE WT ONCE
    # ==================================================

    (
        wt_ids,
        wt_attention,
        wt_residue_positions,
    ) = encode_sequence(
        tokenizer,
        wt_seq,
    )

    print(
        "WT token count:",
        len(wt_ids),
        flush=True,
    )

    print(
        "WT residue-token count:",
        len(wt_residue_positions),
        flush=True,
    )

    # ==================================================
    # OUTPUT / RESUME
    # ==================================================

    completed = load_completed(
        args.summary
    )

    detail_done = load_detail_done(
        args.detail
    )

    print(
        "Already completed:",
        len(completed),
        flush=True,
    )

    summary_fields = [
        "candidate_id",
        "length",
        "n_mutations",
        "mutations",

        "wtctx_sum_delta",
        "wtctx_mean_delta",
        "wtctx_min_delta",
        "wtctx_max_delta",

        "mutctx_sum_delta",
        "mutctx_mean_delta",
        "mutctx_min_delta",
        "mutctx_max_delta",

        "worst_wtctx_mutation",
        "worst_mutctx_mutation",

        "status",
        "error",
    ]

    detail_fields = [
        "candidate_id",
        "mutation",
        "position",
        "wt_aa",
        "mut_aa",
        "wtctx_delta",
        "mutctx_delta",
    ]

    summary_new = not (
        os.path.exists(args.summary)
        and os.path.getsize(args.summary) > 0
    )

    detail_new = not (
        os.path.exists(args.detail)
        and os.path.getsize(args.detail) > 0
    )

    summary_fh = open(
        args.summary,
        "a",
        newline="",
        buffering=1,
    )

    detail_fh = open(
        args.detail,
        "a",
        newline="",
        buffering=1,
    )

    summary_writer = csv.DictWriter(
        summary_fh,
        fieldnames=summary_fields,
    )

    detail_writer = csv.DictWriter(
        detail_fh,
        fieldnames=detail_fields,
    )

    if summary_new:
        summary_writer.writeheader()
        summary_fh.flush()

    if detail_new:
        detail_writer.writeheader()
        detail_fh.flush()

    # ==================================================
    # WT-CONTEXT CACHE
    #
    # 同一个 position 的 masked WT context
    # 对所有候选完全相同，因此只计算一次。
    # ==================================================

    wtctx_cache = {}

    @torch.inference_mode()
    def masked_forward(
        base_ids,
        base_attention,
        token_positions,
    ):
        """
        base_ids: [L]
        base_attention: [L]
        token_positions: list of token indices

        Returns:
            list of [vocab] CPU logits
        """

        batch_size = len(
            token_positions
        )

        ids = base_ids.unsqueeze(0).repeat(
            batch_size,
            1,
        )

        attention = base_attention.unsqueeze(0).repeat(
            batch_size,
            1,
        )

        rows = torch.arange(
            batch_size
        )

        for row, token_pos in enumerate(
            token_positions
        ):
            ids[
                row,
                token_pos
            ] = tokenizer.mask_token_id

        ids = ids.to(
            device
        )

        attention = attention.to(
            device
        )

        token_positions_gpu = torch.tensor(
            token_positions,
            dtype=torch.long,
            device=device,
        )

        rows_gpu = torch.arange(
            batch_size,
            device=device,
        )

        output = model(
            input_ids=ids,
            attention_mask=attention,
        )

        selected_logits = output.logits[
            rows_gpu,
            token_positions_gpu,
            :
        ]

        result = (
            selected_logits
            .float()
            .cpu()
        )

        del (
            ids,
            attention,
            output,
            selected_logits,
        )

        return result

    def populate_wtctx_cache(
        sequence_positions,
    ):
        missing = [
            pos
            for pos in sequence_positions
            if pos not in wtctx_cache
        ]

        for group in chunked(
            missing,
            args.mask_batch_size,
        ):
            token_positions = [
                int(
                    wt_residue_positions[
                        pos - 1
                    ].item()
                )
                for pos in group
            ]

            batch_logits = masked_forward(
                wt_ids,
                wt_attention,
                token_positions,
            )

            for row, pos in enumerate(
                group
            ):
                wtctx_cache[pos] = (
                    batch_logits[row].clone()
                )

    # ==================================================
    # SCORE CANDIDATES
    # ==================================================

    total_records = len(
        records
    )

    for record_index, (
        candidate_id,
        sequence,
    ) in enumerate(
        records,
        start=1,
    ):

        if candidate_id in completed:

            print(
                f"[{record_index}/{total_records}] "
                f"SKIP {candidate_id}",
                flush=True,
            )

            continue

        print(
            f"[{record_index}/{total_records}] "
            f"START {candidate_id}",
            flush=True,
        )

        try:
            bad = set(sequence) - AA20

            if bad:
                raise RuntimeError(
                    f"Non-standard residues: "
                    f"{sorted(bad)}"
                )

            mutations = find_mutations(
                wt_seq,
                sequence,
            )

            mutation_names = [
                f"{wt_aa}{pos}{mut_aa}"
                for pos, wt_aa, mut_aa
                in mutations
            ]

            # WT itself
            if len(mutations) == 0:

                status = (
                    "wt"
                    if candidate_id == wt_id
                    else "identical_to_wt"
                )

                write_flush(
                    summary_writer,
                    summary_fh,
                    {
                        "candidate_id": candidate_id,
                        "length": len(sequence),
                        "n_mutations": 0,
                        "mutations": "",

                        "wtctx_sum_delta": 0.0,
                        "wtctx_mean_delta": 0.0,
                        "wtctx_min_delta": 0.0,
                        "wtctx_max_delta": 0.0,

                        "mutctx_sum_delta": 0.0,
                        "mutctx_mean_delta": 0.0,
                        "mutctx_min_delta": 0.0,
                        "mutctx_max_delta": 0.0,

                        "worst_wtctx_mutation": "",
                        "worst_mutctx_mutation": "",

                        "status": status,
                        "error": "",
                    },
                )

                completed.add(
                    candidate_id
                )

                print(
                    f"[{record_index}/{total_records}] "
                    f"DONE {candidate_id} "
                    f"({status})",
                    flush=True,
                )

                continue

            # Candidate tokenization
            (
                candidate_ids,
                candidate_attention,
                candidate_residue_positions,
            ) = encode_sequence(
                tokenizer,
                sequence,
            )

            # ==========================================
            # WT-context
            #
            # 其它 mutation 全部不存在。
            # 当前位点放到 WT context 中评价。
            # ==========================================

            sequence_positions = [
                pos
                for pos, _, _
                in mutations
            ]

            populate_wtctx_cache(
                sequence_positions
            )

            wtctx_scores = {}

            for (
                pos,
                wt_aa,
                mut_aa,
            ) in mutations:

                wt_token_pos = int(
                    wt_residue_positions[
                        pos - 1
                    ].item()
                )

                candidate_token_pos = int(
                    candidate_residue_positions[
                        pos - 1
                    ].item()
                )

                wt_token_id = int(
                    wt_ids[
                        wt_token_pos
                    ].item()
                )

                mut_token_id = int(
                    candidate_ids[
                        candidate_token_pos
                    ].item()
                )

                logits = wtctx_cache[
                    pos
                ]

                # logP(mut) - logP(WT)
                #
                # 同一 context 下 log-softmax
                # normalization term 抵消，因此
                # 等于 logit(mut)-logit(WT)
                delta = float(
                    logits[
                        mut_token_id
                    ]
                    -
                    logits[
                        wt_token_id
                    ]
                )

                wtctx_scores[
                    pos
                ] = delta

            # ==========================================
            # MUTANT-context
            #
            # 其它 mutation 全部保留。
            # 每次只 mask 当前 mutation。
            # ==========================================

            mutctx_logits = {}

            for group in chunked(
                mutations,
                args.mask_batch_size,
            ):

                token_positions = [
                    int(
                        candidate_residue_positions[
                            pos - 1
                        ].item()
                    )
                    for pos, _, _
                    in group
                ]

                batch_logits = masked_forward(
                    candidate_ids,
                    candidate_attention,
                    token_positions,
                )

                for row, (
                    pos,
                    wt_aa,
                    mut_aa,
                ) in enumerate(group):

                    mutctx_logits[
                        pos
                    ] = batch_logits[
                        row
                    ].clone()

            mutctx_scores = {}

            for (
                pos,
                wt_aa,
                mut_aa,
            ) in mutations:

                wt_token_pos = int(
                    wt_residue_positions[
                        pos - 1
                    ].item()
                )

                candidate_token_pos = int(
                    candidate_residue_positions[
                        pos - 1
                    ].item()
                )

                wt_token_id = int(
                    wt_ids[
                        wt_token_pos
                    ].item()
                )

                mut_token_id = int(
                    candidate_ids[
                        candidate_token_pos
                    ].item()
                )

                logits = mutctx_logits[
                    pos
                ]

                delta = float(
                    logits[
                        mut_token_id
                    ]
                    -
                    logits[
                        wt_token_id
                    ]
                )

                mutctx_scores[
                    pos
                ] = delta

            # ==========================================
            # PER-MUTATION OUTPUT
            # ==========================================

            mutation_lookup = {}

            for (
                pos,
                wt_aa,
                mut_aa,
            ) in mutations:

                mutation_name = (
                    f"{wt_aa}{pos}{mut_aa}"
                )

                mutation_lookup[
                    pos
                ] = mutation_name

                key = (
                    candidate_id,
                    mutation_name,
                )

                if key in detail_done:
                    continue

                write_flush(
                    detail_writer,
                    detail_fh,
                    {
                        "candidate_id":
                            candidate_id,

                        "mutation":
                            mutation_name,

                        "position":
                            pos,

                        "wt_aa":
                            wt_aa,

                        "mut_aa":
                            mut_aa,

                        "wtctx_delta":
                            wtctx_scores[pos],

                        "mutctx_delta":
                            mutctx_scores[pos],
                    },
                )

                detail_done.add(
                    key
                )

            # ==========================================
            # CANDIDATE SUMMARY
            # ==========================================

            wt_values = list(
                wtctx_scores.values()
            )

            mut_values = list(
                mutctx_scores.values()
            )

            worst_wt_pos = min(
                wtctx_scores,
                key=wtctx_scores.get,
            )

            worst_mut_pos = min(
                mutctx_scores,
                key=mutctx_scores.get,
            )

            row = {
                "candidate_id":
                    candidate_id,

                "length":
                    len(sequence),

                "n_mutations":
                    len(mutations),

                "mutations":
                    ";".join(
                        mutation_names
                    ),

                "wtctx_sum_delta":
                    sum(wt_values),

                "wtctx_mean_delta":
                    sum(wt_values)
                    / len(wt_values),

                "wtctx_min_delta":
                    min(wt_values),

                "wtctx_max_delta":
                    max(wt_values),

                "mutctx_sum_delta":
                    sum(mut_values),

                "mutctx_mean_delta":
                    sum(mut_values)
                    / len(mut_values),

                "mutctx_min_delta":
                    min(mut_values),

                "mutctx_max_delta":
                    max(mut_values),

                "worst_wtctx_mutation":
                    mutation_lookup[
                        worst_wt_pos
                    ],

                "worst_mutctx_mutation":
                    mutation_lookup[
                        worst_mut_pos
                    ],

                "status":
                    "ok",

                "error":
                    "",
            }

            write_flush(
                summary_writer,
                summary_fh,
                row,
            )

            completed.add(
                candidate_id
            )

            print(
                f"[{record_index}/{total_records}] "
                f"DONE {candidate_id} "
                f"n_mut={len(mutations)} "
                f"WTctx_mean="
                f"{row['wtctx_mean_delta']:.4f} "
                f"WTctx_min="
                f"{row['wtctx_min_delta']:.4f} "
                f"MUTctx_mean="
                f"{row['mutctx_mean_delta']:.4f} "
                f"MUTctx_min="
                f"{row['mutctx_min_delta']:.4f}",
                flush=True,
            )

        except Exception as e:

            print(
                f"[{record_index}/{total_records}] "
                f"ERROR {candidate_id}: "
                f"{repr(e)}",
                file=sys.stderr,
                flush=True,
            )

            write_flush(
                summary_writer,
                summary_fh,
                {
                    "candidate_id": candidate_id,
                    "length": len(sequence),
                    "n_mutations": "",
                    "mutations": "",

                    "wtctx_sum_delta": "",
                    "wtctx_mean_delta": "",
                    "wtctx_min_delta": "",
                    "wtctx_max_delta": "",

                    "mutctx_sum_delta": "",
                    "mutctx_mean_delta": "",
                    "mutctx_min_delta": "",
                    "mutctx_max_delta": "",

                    "worst_wtctx_mutation": "",
                    "worst_mutctx_mutation": "",

                    "status": "error",
                    "error": repr(e),
                },
            )

            # error 不加入 completed，
            # 所以下次重启会重新计算
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary_fh.close()
    detail_fh.close()

    print("=" * 80, flush=True)
    print("ALL DONE", flush=True)
    print("Summary:", args.summary, flush=True)
    print("Detail:", args.detail, flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
