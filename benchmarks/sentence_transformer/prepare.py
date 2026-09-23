# SPDX-License-Identifier: Apache-2.0
"""Prepare a pinned STS-B sentence-pair fixture without using the HF dataset cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq


MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DATASET = "sentence-transformers/stsb"
DATASET_REVISION = "ab7a5ac0e35aa22088bdcf23e7fd99b220e53308"
DATASET_MEMBER = "data/train-00000-of-00001.parquet"
DATASET_URL = (
    f"https://huggingface.co/datasets/{DATASET}/resolve/"
    f"{DATASET_REVISION}/{DATASET_MEMBER}?download=true"
)
DATASET_SHA256 = "eae324ff1eac2d0ba769851736eb7232eda64f370a16eb20e74a2c5f8f5fafe0"
DATASET_BYTES = 470_612
SOURCE_ROWS = 5_749
SELECTED_ROWS = 2_063
SCORE_THRESHOLD = 0.7
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "config_sentence_transformers.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "1_Pooling/config.json",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(model_path: Path) -> dict[str, object]:
    missing = [name for name in MODEL_FILES if not (model_path / name).is_file()]
    if missing:
        raise RuntimeError(f"pinned model is incomplete; missing: {missing}")
    metadata_root = model_path / ".cache" / "huggingface" / "download"
    revisions = set()
    for name in MODEL_FILES:
        metadata = metadata_root / f"{name}.metadata"
        if not metadata.is_file():
            raise RuntimeError(f"missing Hugging Face revision metadata: {metadata}")
        revisions.add(metadata.read_text(encoding = "utf-8").splitlines()[0])
    if revisions != {MODEL_REVISION}:
        raise RuntimeError(
            f"model revision mismatch: expected {MODEL_REVISION}, found {sorted(revisions)}"
        )
    weights = model_path / "model.safetensors"
    return {
        "path": str(model_path),
        "revision": MODEL_REVISION,
        "model_safetensors_bytes": weights.stat().st_size,
        "model_safetensors_sha256": sha256_path(weights),
    }


def valid_source(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size != DATASET_BYTES:
        return False
    with path.open("rb") as stream:
        if stream.read(4) != b"PAR1":
            return False
        stream.seek(-4, os.SEEK_END)
        return stream.read(4) == b"PAR1"


def download_source(destination: Path) -> None:
    if valid_source(destination) and sha256_path(destination) == DATASET_SHA256:
        return
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(
        DATASET_URL, headers = {"User-Agent": "unsloth-pr-4460-fixture/1"}
    )
    try:
        with (
            urllib.request.urlopen(request, timeout = 120) as response,
            temporary.open("wb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if not valid_source(temporary):
            raise RuntimeError("downloaded STS-B source is not a complete parquet file")
        actual = sha256_path(temporary)
        if actual != DATASET_SHA256:
            raise RuntimeError(
                f"downloaded STS-B SHA-256 mismatch: expected {DATASET_SHA256}, got {actual}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent = 2, ensure_ascii = False) + "\n").encode("utf-8")


def write_verified_json(path: Path, value: object) -> str:
    payload = json_bytes(value)
    temporary = path.with_suffix(path.suffix + ".write")
    temporary.write_bytes(payload)
    if json.loads(temporary.read_text(encoding = "utf-8")) != value:
        raise RuntimeError(f"JSON round-trip failed for {path}")
    os.replace(temporary, path)
    actual = sha256_path(path)
    expected = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise RuntimeError(f"post-write SHA-256 mismatch for {path}")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--root", type = Path, required = True)
    parser.add_argument("--download-model", action = "store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.download_model:
        from huggingface_hub import snapshot_download
        snapshot_download(
            MODEL,
            revision = MODEL_REVISION,
            local_dir = root / "models" / "all-MiniLM-L6-v2",
            allow_patterns = list(MODEL_FILES),
        )
    fixtures = root / "fixtures"
    fixtures.mkdir(parents = True, exist_ok = True)
    model_info = verify_model(root / "models" / "all-MiniLM-L6-v2")
    source = fixtures / f"stsb-train-{DATASET_REVISION}.parquet"
    download_source(source)

    table = pq.read_table(source, columns = ["sentence1", "sentence2", "score"])
    if table.num_rows != SOURCE_ROWS:
        raise RuntimeError(
            f"unexpected STS-B train row count: expected {SOURCE_ROWS}, got {table.num_rows}"
        )
    rows = table.to_pylist()
    pairs = [
        [row["sentence1"], row["sentence2"]]
        for row in rows
        if float(row["score"]) >= SCORE_THRESHOLD
    ]
    if len(pairs) != SELECTED_ROWS:
        raise RuntimeError(
            f"unexpected selected row count: expected {SELECTED_ROWS}, got {len(pairs)}"
        )
    if any(not isinstance(text, str) or not text.strip() for pair in pairs for text in pair):
        raise RuntimeError("selected STS-B rows contain an empty or non-string sentence")

    pairs_path = fixtures / "stsb-positive-pairs.json"
    fixture_sha256 = write_verified_json(pairs_path, pairs)
    provenance = {
        "schema": "unsloth-stsb-positive-pairs-v1",
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "model_path": model_info["path"],
        "model_safetensors_bytes": model_info["model_safetensors_bytes"],
        "model_safetensors_sha256": model_info["model_safetensors_sha256"],
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "dataset_member": DATASET_MEMBER,
        "source_url": DATASET_URL,
        "source_file": source.name,
        "source_bytes": source.stat().st_size,
        "source_sha256": sha256_path(source),
        "split": "train",
        "source_row_count": table.num_rows,
        "selection": f"score >= {SCORE_THRESHOLD}",
        "pair_count": len(pairs),
        "pairs_file": pairs_path.name,
        "pairs_sha256": fixture_sha256,
    }
    provenance_path = fixtures / "provenance.json"
    provenance_sha256 = write_verified_json(provenance_path, provenance)
    print(f"MODEL_READY {model_info['path']} {MODEL_REVISION}", flush = True)
    print(
        f"DATASET_READY {len(pairs)} {DATASET_REVISION} {provenance['source_sha256']}", flush = True
    )
    print(f"FIXTURE_READY {pairs_path} {fixture_sha256}", flush = True)
    print(f"PROVENANCE_READY {provenance_path} {provenance_sha256}", flush = True)


if __name__ == "__main__":
    main()
