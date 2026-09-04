# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Unpack the Playwright evidence the Unsloth payload smuggled home.

Kaggle's ``kernels output`` returns the whole of ``/kaggle/working``, and the
shared launcher deliberately does not take it: a previous incident lost two
passing notebooks because a multi-gigabyte saved model sorted ahead of them
and the stream broke partway through. So the launcher fetches executed
notebooks by direct URL and nothing else, and a screenshot sitting on the
Kaggle filesystem is a screenshot nobody will ever see.

The payload therefore tars its evidence, base64s it, and prints it in chunks
into its own cell output. This script puts it back together. Every chunk is
numbered ``i/n`` so a truncated log is detected rather than silently yielding
a corrupt archive.

If the shared launcher ever grows a way to collect arbitrary files from a
kernel's output, this whole path should go: it exists only because there is
no other channel.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import re
import tarfile
from pathlib import Path

EVIDENCE_PREFIX = "STUDIO_GPU_EVIDENCE_B64 "
_CHUNK_RE = re.compile(r"^(\d+)/(\d+)\s+(\S+)$")

# A tar member is trusted only as far as its name. Absolute paths and ``..``
# are refused rather than sanitised, because a bundle that contains one is not
# a bundle this payload wrote.
MAX_MEMBER_BYTES = 20_000_000


def iter_text(path: Path):
    """Every text stream in a collected evidence directory.

    Recursive, because the launcher collects each kernel into its own
    subdirectory (``kaggle_evidence/<kernel-slug>/``) so two kernels of one
    run cannot overwrite each other's ``kernel.log``, while the workflow
    hands this script the parent. A top-level ``glob`` therefore matched
    nothing and every real run reported "the payload emitted no evidence
    bundle". Same discovery ``launch.py::extract_reports`` uses.
    """
    if path.is_file():
        yield path.read_text(encoding = "utf-8", errors = "replace")
        return
    for nb_path in sorted(path.rglob("*_output.ipynb")):
        try:
            nb = json.loads(nb_path.read_text(encoding = "utf-8", errors = "replace"))
        except Exception:  # noqa: BLE001
            continue
        for cell in nb.get("cells", []):
            for output in cell.get("outputs", []):
                text = output.get("text") or ""
                if isinstance(text, list):
                    text = "".join(text)
                yield text
    for log_path in sorted(path.rglob("kernel.log")):
        raw = log_path.read_text(encoding = "utf-8", errors = "replace")
        try:
            records = json.loads(raw)
        except json.JSONDecodeError:
            yield raw
        else:
            if isinstance(records, list):
                yield "".join(r.get("data", "") for r in records if isinstance(r, dict))
            else:
                yield raw


def collect_chunks(streams) -> tuple[dict[int, str], int]:
    """{index: chunk} and the declared total, from any number of streams.

    The same chunk arrives twice, because the executed notebook and Kaggle's
    kernel log are two copies of one stdout. Either copy can be cut off mid
    line, and a cut only ever shortens: the survivor is a prefix of the whole
    chunk, and it still parses as ``i/n <payload>``. Overwriting on every
    sighting therefore let a truncated second copy replace a complete first
    one, and the reassembled bundle then failed base64 or tar validation with
    every chunk index present -- evidence lost with the complete source on
    disk. So a later sighting is taken only when it EXTENDS what is held.
    """
    chunks: dict[int, str] = {}
    total = 0
    for text in streams:
        for line in text.splitlines():
            if not line.startswith(EVIDENCE_PREFIX):
                continue
            match = _CHUNK_RE.match(line[len(EVIDENCE_PREFIX) :].strip())
            if not match:
                continue
            index, declared, payload = int(match.group(1)), int(match.group(2)), match.group(3)
            total = max(total, declared)
            held = chunks.get(index)
            if held is None or (len(payload) > len(held) and payload.startswith(held)):
                chunks[index] = payload
    return chunks, total


def is_safe_member(name: str) -> bool:
    if name.startswith("/") or name.startswith("\\"):
        return False
    parts = Path(name).parts
    return ".." not in parts and not any(p.startswith("/") for p in parts)


def extract(blob: bytes, outdir: Path) -> list[str]:
    outdir.mkdir(parents = True, exist_ok = True)
    written: list[str] = []
    with tarfile.open(fileobj = io.BytesIO(blob), mode = "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if not is_safe_member(member.name):
                print(f"[evidence] refusing member {member.name!r}", flush = True)
                continue
            if member.size > MAX_MEMBER_BYTES:
                print(f"[evidence] refusing oversized member {member.name!r}", flush = True)
                continue
            dest = outdir / member.name
            dest.parent.mkdir(parents = True, exist_ok = True)
            handle = tar.extractfile(member)
            if handle is None:
                continue
            dest.write_bytes(handle.read())
            written.append(member.name)
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required = True, help = "directory the launcher collected into")
    ap.add_argument("--outdir", required = True, help = "where to unpack the bundle")
    args = ap.parse_args()

    source = Path(args.evidence)
    if not source.exists():
        print("[evidence] nothing was collected, so there is nothing to unpack", flush = True)
        return 0

    chunks, total = collect_chunks(iter_text(source))
    if not chunks:
        print("[evidence] the payload emitted no evidence bundle", flush = True)
        return 0

    missing = [i for i in range(1, total + 1) if i not in chunks]
    if missing:
        # Report rather than guess. A bundle reassembled out of a truncated
        # log decodes to something, and that something is not the evidence.
        print(
            f"[evidence] {len(missing)} of {total} chunks are missing "
            f"(first: {missing[0]}), so the bundle is incomplete and is not "
            f"being unpacked",
            flush = True,
        )
        return 0

    encoded = "".join(chunks[i] for i in range(1, total + 1))
    try:
        blob = base64.b64decode(encoded, validate = True)
    except (binascii.Error, ValueError) as exc:
        print(f"[evidence] the bundle did not decode: {exc}", flush = True)
        return 0

    try:
        written = extract(blob, Path(args.outdir))
    except tarfile.TarError as exc:
        print(f"[evidence] the bundle is not a readable archive: {exc}", flush = True)
        return 0

    print(f"[evidence] unpacked {len(written)} file(s) into {args.outdir}", flush = True)
    for name in written:
        print(f"  {name}", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
