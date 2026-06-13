"""
Compute loss_mask statistics for a Lightning OPD parquet.

Usage:
    python compute_loss_mask_mean.py <parquet_path> [--max-seq-len 5120]

Prints:
  - response_length mean/min/max (len of loss_mask list = total response tokens)
  - sum(loss_mask) mean/min/max  (= effective/unmasked response tokens)
  - loss_mask_mean per sample    (= sum(lm) / len(lm), "response-only" fraction)
  - full_seq_loss_mask_mean      (= sum(lm) / (prompt_len + len(lm)), like the fork's full_loss_masks)
  - padded_loss_mask_mean        (= sum(lm) / max_seq_len, matches our actor.py 2-D tensor mean)
"""

import argparse
import ast

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", help="Path to Lightning OPD parquet")
    parser.add_argument("--max-seq-len", type=int, default=5120)
    parser.add_argument(
        "--tokenizer",
        default="/scratch/ansh/models/lightning-sft-3000",
        help="Tokenizer path for computing prompt lengths",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process first N rows")
    args = parser.parse_args()

    print(f"Loading {args.parquet} ...")
    df = pd.read_parquet(args.parquet)
    if args.limit:
        df = df.iloc[: args.limit]
    print(f"Rows: {len(df)}")

    print(f"Loading tokenizer from {args.tokenizer} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    response_lens = []
    loss_sums = []
    prompt_lens = []

    for row in df.itertuples():
        meta = row.metadata
        if isinstance(meta, str):
            meta = ast.literal_eval(meta)

        lm = meta["loss_mask"]
        rt = meta["response_tokens"]
        prompt_str = row.prompt

        response_lens.append(len(lm))
        loss_sums.append(sum(lm))

        prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        prompt_lens.append(len(prompt_ids))

    response_lens = np.array(response_lens)
    loss_sums = np.array(loss_sums, dtype=float)
    prompt_lens = np.array(prompt_lens)
    total_lens = prompt_lens + response_lens

    # "response-only" fraction (what the fork's loss_mask_mean measures when masks are all-ones)
    response_only_mean = (loss_sums / response_lens).mean()

    # full-sequence fraction (sum(lm) / total_len), matches fork's full_loss_masks mean
    full_seq_mean = (loss_sums / total_lens).mean()

    # padded fraction (sum(lm) / max_seq_len), matches our actor.py 2-D tensor mean
    clamped_loss_sums = np.minimum(loss_sums, args.max_seq_len)
    padded_mean = (clamped_loss_sums / args.max_seq_len).mean()

    print()
    print(f"{'Metric':<40} {'mean':>10} {'min':>10} {'max':>10}")
    print("-" * 72)
    print(f"{'prompt_length (tokens)':<40} {prompt_lens.mean():>10.1f} {prompt_lens.min():>10} {prompt_lens.max():>10}")
    print(f"{'response_length = len(loss_mask)':<40} {response_lens.mean():>10.1f} {response_lens.min():>10} {response_lens.max():>10}")
    print(f"{'total_length':<40} {total_lens.mean():>10.1f} {total_lens.min():>10} {total_lens.max():>10}")
    print(f"{'sum(loss_mask) [effective tokens]':<40} {loss_sums.mean():>10.1f} {loss_sums.min():>10} {loss_sums.max():>10}")
    print()
    print(f"{'loss_mask_mean (response-only)':<40} {response_only_mean:>10.4f}  ← fork batch['loss_masks'] mean if all-ones")
    print(f"{'loss_mask_mean (full-sequence)':<40} {full_seq_mean:>10.4f}  ← sum(lm)/(prompt+response)")
    label = f"loss_mask_mean (padded, max_seq={args.max_seq_len})"
    print(f"{label:<40} {padded_mean:>10.4f}  ← our actor.py 2-D tensor mean")


if __name__ == "__main__":
    main()
