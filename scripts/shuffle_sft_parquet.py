# shuffle the darn parquet
import argparse
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(
        description="Shuffle a parquet with the generated rollouts at the SFT stage"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to a local parquet file. If not set, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output parquet file path (stem; seed suffix is appended automatically).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling (default: 42).",
    )
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="HuggingFace dataset repo id to download from when --input is not set.",
    )
    parser.add_argument(
        "--hf-split", type=str, default="train",
        help="HuggingFace dataset split to use (default: train).",
    )
    return parser.parse_args()


def build_output_path(output: str, seed: int) -> str:
    base, ext = os.path.splitext(output)
    if ext.lower() != ".parquet":
        base = output
    return f"{base}_seed_{seed}.parquet"


def shuffle_parquet(input_path: str, output_path: str, seed: int) -> None:
    """
    Shuffle a parquet file without ever materializing the full dataset as one
    contiguous Arrow array (which overflows 32-bit list offsets at ~300k rows
    of nested list<struct> columns).

    Strategy:
      1. Read the row-group manifest — just metadata, no data loaded.
      2. Build a global shuffle permutation over all rows.
      3. For each row group in shuffled order, read it and apply the local
         index permutation, then write it out.

    This keeps peak memory to ~1 row group at a time.
    """
    pf = pq.ParquetFile(input_path)
    n_rows = pf.metadata.num_rows
    n_rgs = pf.metadata.num_row_groups

    print(f"Shuffling {n_rows:,} rows across {n_rgs} row groups with seed={seed} …")

    rng = np.random.default_rng(seed)
    global_order = rng.permutation(n_rows).astype(np.int64)

    # Build a mapping: for each row, which row group does it live in?
    rg_sizes = [pf.metadata.row_group(i).num_rows for i in range(n_rgs)]
    rg_offsets = np.zeros(n_rgs + 1, dtype=np.int64)
    for i, s in enumerate(rg_sizes):
        rg_offsets[i + 1] = rg_offsets[i] + s

    # For each position in the output, record (source_rg, local_row_index)
    source_rg = np.searchsorted(rg_offsets[1:], global_order, side="right").astype(np.int32)
    local_idx = (global_order - rg_offsets[source_rg]).astype(np.int64)

    # Group output positions by source row group so we read each rg once
    order_by_rg = np.argsort(source_rg, kind="stable")

    writer = None
    pos = 0

    for rg_idx in range(n_rgs):
        # Slice out which output positions come from this row group
        start = pos
        while pos < n_rows and source_rg[order_by_rg[pos]] == rg_idx:
            pos += 1
        if start == pos:
            continue

        slot = order_by_rg[start:pos]          # positions in final output order
        rows_needed = local_idx[slot]           # local row indices within this rg

        # Read only the rows we need from this row group
        table = pf.read_row_group(rg_idx)
        # take() is safe here: each row group is small (~32 rows for 9376 rgs)
        chunk = table.take(rows_needed)
        chunk = chunk.replace_schema_metadata({})

        if writer is None:
            writer = pq.ParquetWriter(output_path, chunk.schema, compression="snappy")
        writer.write_table(chunk)

    if writer is not None:
        writer.close()


def main():
    args = parse_args()

    if args.input:
        input_path = args.input
    else:
        if args.hf_repo is None:
            raise ValueError("Provide --input or --hf-repo.")
        print(f"Downloading {args.hf_repo} (split={args.hf_split}) from HuggingFace …")
        from datasets import load_dataset  # type: ignore
        import tempfile, os
        ds = load_dataset(args.hf_repo, split=args.hf_split)
        tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        ds.to_parquet(tmp.name)
        input_path = tmp.name

    output_path = build_output_path(args.output, args.seed)
    print(f"Reading {input_path} …")
    shuffle_parquet(input_path, output_path, args.seed)

    import os
    size_mb = os.path.getsize(output_path) / 1024 ** 2
    print(f"Done. {output_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()