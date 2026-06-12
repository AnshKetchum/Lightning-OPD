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
    Globally shuffle a parquet file using DuckDB ORDER BY random().
    DuckDB handles external sorting so the full dataset never needs to fit in RAM.
    """
    import duckdb

    pf = pq.ParquetFile(input_path)
    n_rows = pf.metadata.num_rows
    print(f"Shuffling {n_rows:,} rows with seed={seed} …")

    con = duckdb.connect()
    con.execute(f"SELECT setseed({seed / (2**31 - 1)});")
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{input_path}') ORDER BY random()
        ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION snappy)
    """)


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