import argparse
import random
import os
import torch

from tokenizers import Tokenizer
from models.progen.modeling_progen import ProGenForCausalLM


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def truncate(sample):
    pos = []
    for terminal in ["1", "2"]:
        p = sample.find(terminal, 1)
        if p != -1:
            pos.append(p)

    if pos:
        return sample[:min(pos) + 1]
    return sample


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        default="/home/huangpeihuang/opencrispr_models/cas9_lm_model"
    )
    parser.add_argument(
        "--tokenizer",
        default="/home/huangpeihuang/progen/progen2/tokenizer.json"
    )
    parser.add_argument("--context", default="1")
    parser.add_argument("--max-length", type=int, default=1500)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--t", type=float, default=0.8)
    parser.add_argument("--p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)

    print("加载 Cas9 LM...")
    model = ProGenForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model = model.to(device)
    model.eval()

    print("加载 tokenizer...")
    tokenizer = Tokenizer.from_file(args.tokenizer)

    input_ids = torch.tensor(
        tokenizer.encode(args.context).ids,
        dtype=torch.long
    ).view(1, -1).to(device)

    pad_token_id = tokenizer.encode("<|pad|>").ids[0]

    print("开始生成...")
    print("prompt:", args.context)
    print("max_length:", args.max_length)
    print("temperature:", args.t)
    print("top_p:", args.p)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            do_sample=True,
            temperature=args.t,
            top_p=args.p,
            max_length=args.max_length,
            num_return_sequences=args.num_samples,
            pad_token_id=pad_token_id,
        )

    sequences = tokenizer.decode_batch(
        [x.detach().cpu().tolist() for x in output]
    )

    print("\n========== 生成结果 ==========")

    for i, seq in enumerate(sequences, 1):
        seq = truncate(seq)

        clean = seq
        if clean.startswith("1"):
            clean = clean[1:]
        if clean.endswith("2"):
            clean = clean[:-1]

        print(f"\nSample {i}")
        print("氨基酸长度:", len(clean))
        print(clean)


if __name__ == "__main__":
    main()
