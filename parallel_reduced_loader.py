#!/usr/bin/env python3
"""
Parallel loader for reduced CSV feature files.

- Parallelizes CSV I/O + parsing
- Uses fixed dtypes
- Injects typed metadata
- Concatenates safely in chunks
- Streams output to Parquet

Designed for ~1M tiny CSVs on a 16 GB machine.
"""

import gc
import multiprocessing as mp
from itertools import islice
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ============================================================
# USER CONFIGURATION
# ============================================================

N_WORKERS = 5                 # Safe on 16 GB RAM
CHUNK_SIZE = 10_000           # Files per chunk
OUTPUT_FILE = "reduced.parquet"

# Metadata dtypes (must be defined by you)
META_DTYPES = {
    "dimension": "int16",
    "seed_lhs": "int32",
    "n_samples": "int32",
    "function_idx": "int16",
    "instance_idx": "int32",
    "reduction_ratio": "float32",
    "embedding_seed": "int32",
    "round": "int16",
}
# ============================================================
# GLOBALS (initialized in main / workers)
# ============================================================

FEATURE_DTYPES = None

# ============================================================
# USER FUNCTIONS (you provide implementations)
# ============================================================

def extract_meta_data_from_reduced_feature_file_path(file_path: Path) -> dict:
    """
    Extracts meta data from the reduced feature file path.
    
    Parameters:
    file_path (Path): Path object of the reduced feature file.
    
    Returns:
    dict: A dictionary containing the extracted meta data.
    """
    """
    Extract key-value numeric metadata from path segments like key_value.
    """

    if not isinstance(file_path, (str, Path)):
        raise ValueError("path must be a string or Path object")

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"The path {file_path} does not exist.")

    metadata = {}

    round_txt = file_path.parts[-1]
    round_number = int(round_txt.split("_")[-1].replace(".csv", ""))
    metadata["round"] = round_number

    embedding_seed_txt = file_path.parts[-2]
    embedding_seed = int(embedding_seed_txt.split("_")[-1])
    metadata["embedding_seed"] = embedding_seed

    reduction_ratio_txt = file_path.parts[-4]
    reduction_ratio = float(reduction_ratio_txt.split("_")[-1])
    metadata["reduction_ratio"] = reduction_ratio

    instance_idx_txt = file_path.parts[-5]
    instance_idx = int(instance_idx_txt.split("_")[-1])
    metadata["instance_idx"] = instance_idx

    function_idx_txt = file_path.parts[-6]
    function_idx = int(function_idx_txt.split("_")[-1])
    metadata["function_idx"] = function_idx

    n_samples_txt = file_path.parts[-7]
    n_samples = int(n_samples_txt.split("_")[-1])
    metadata["n_samples"] = n_samples

    seed_txt = file_path.parts[-8]
    seed = int(seed_txt.split("_")[-1])
    metadata["seed_lhs"] = seed

    dimension_txt = file_path.parts[-9]
    dimension = int(dimension_txt.split("_")[-1])
    metadata["dimension"] = dimension
    
    return metadata


def load_reduced(fp):
    """
    Load one reduced CSV and inject typed metadata.
    Executed inside worker processes.
    """
    df = pd.read_csv(
        fp,
        dtype=FEATURE_DTYPES,
        engine="c",
        low_memory=False
    )

    meta = extract_meta_data_from_reduced_feature_file_path(fp)

    for k, v in meta.items():
        df[k] = pd.Series(v, dtype=META_DTYPES[k], index=df.index)

    return df

# ============================================================
# MULTIPROCESSING UTILITIES
# ============================================================

def init_worker(feature_dtypes):
    """
    Initialize worker globals (runs once per process).
    """
    global FEATURE_DTYPES
    FEATURE_DTYPES = feature_dtypes


def chunked(iterable, size):
    """
    Yield fixed-size chunks from iterable.
    """
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


def load_chunk_parallel(file_chunk, feature_dtypes):
    """
    Load a chunk of files in parallel and concat locally.
    """
    with mp.Pool(
        processes=N_WORKERS,
        initializer=init_worker,
        initargs=(feature_dtypes,)
    ) as pool:
        dfs = pool.map(load_reduced, file_chunk)

    return pd.concat(dfs, ignore_index=True, copy=False)

# ============================================================
# MAIN PIPELINE
# ============================================================

def build_feature_dtypes(example_file):
    """
    Read CSV header ONCE and build dtype dict.
    """
    cols = pd.read_csv(example_file, nrows=0).columns
    return {c: "float64" for c in cols}


def main(reduced_files):
    reduced_files = list(map(Path, reduced_files))

    if not reduced_files:
        raise ValueError("No input files provided.")

    # --------------------------------------------------------
    # Build feature dtypes ONCE
    # --------------------------------------------------------
    feature_dtypes = build_feature_dtypes(reduced_files[0])

    writer = None

    # --------------------------------------------------------
    # Process in chunks
    # --------------------------------------------------------
    for i, file_chunk in enumerate(chunked(reduced_files, CHUNK_SIZE), start=1):
        print(f"[Chunk {i}] Loading {len(file_chunk)} files...")

        chunk_df = load_chunk_parallel(file_chunk, feature_dtypes)

        table = pa.Table.from_pandas(chunk_df, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(OUTPUT_FILE, table.schema)

        writer.write_table(table)

        del chunk_df
        gc.collect()

    writer.close()
    print("✔ Finished successfully.")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Example usage:
    # reduced_files = sorted(Path("data").glob("*.csv"))
    #reduced_files = [...]

    ela_features_reduced_path = Path("ela_features_reduced_3/ELA_extraction/D_20").resolve()
    reduced_files = sorted(ela_features_reduced_path.rglob("*.csv"))

    main(reduced_files)
