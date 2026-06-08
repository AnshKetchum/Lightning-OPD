# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Merge Arrow IPC files produced by data_curation/pipeline.py into a single parquet.
After multi-GPU data generation, each worker writes Arrow files into
rank-specific subdirectories. This script merges them into one parquet
file for downstream consumption (SFT training or Lightning OPD preparation).
Usage:
    python data_curation/merge.py \
        --input-dir data/sft_data \
        --output data/sft_data/merged.parquet
    # With filtering: only keep samples with token count <= 16384
    python data_curation/merge.py \
        --input-dir data/sft_data \
        --output data/sft_data/merged.parquet \
        --max-tokens 16384
"""
import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge Arrow IPC files into a single parquet file."
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory containing Arrow files (searched recursively).",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output parquet file path.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="If set, discard rows with tokens > this value.",
    )
    return parser.parse_args()


def merge_arrow_files(input_dir: str, output: str, max_tokens: int | None = None):
    input_path = Path(input_dir)
    arrow_files = sorted(input_path.rglob("*.arrow"))

    if not arrow_files:
        print(f"No Arrow files found in {input_dir}")
        return

    print(f"Found {len(arrow_files)} Arrow files in {input_dir}")

    # Stream directly to a ParquetWriter — never concatenate all tables into one
    # giant Arrow array. pa.concat_tables() + combine_chunks() on 9k+ files of
    # nested list columns causes a 32-bit offset overflow (2GB limit per array).
    # Writing each source file as its own row group sidesteps that entirely.
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    total_rows = 0
    written_rows = 0

    for f in tqdm(arrow_files, desc="Merging Arrow files"):
        with pa.OSFile(str(f), "rb") as source:
            table = ipc.open_file(source).read_all()

        total_rows += len(table)

        if max_tokens is not None and "tokens" in table.column_names:
            tokens_col = table.column("tokens")
            if pa.types.is_list(tokens_col.type):
                mask = pc.less_equal(pc.list_sum(tokens_col), max_tokens)
            else:
                mask = pc.less_equal(tokens_col, max_tokens)
            table = table.filter(mask)

        if len(table) == 0:
            continue

        table = table.replace_schema_metadata({})

        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")

        writer.write_table(table)
        written_rows += len(table)

    if writer is not None:
        writer.close()

    print(f"Total rows before filtering: {total_rows}")
    if max_tokens is not None:
        print(f"Filtered {total_rows - written_rows} rows with tokens > {max_tokens}")
    print(f"Merged {written_rows} rows -> {output}")


if __name__ == "__main__":
    args = parse_args()
    merge_arrow_files(args.input_dir, args.output, args.max_tokens)