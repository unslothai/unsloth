# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""save_pretrained_gguf can put the .gguf in a sibling <dir>_gguf, not the directory passed in."""

from __future__ import annotations

import os
import subprocess
import time

# unsloth writes to the "_gguf" sibling; search both locations.
GGUF_SEARCH_SUFFIXES = ("", "_gguf")

PREBUILT_MARKER = "skipping compilation"

SOURCE_BUILD_MARKERS = ("cmake", "Building llama.cpp", "make -j")


def find_ggufs(save_dir: str) -> list:
    """Every .gguf reachable from `save_dir`, with the directory that held it."""
    found = []
    for suffix in GGUF_SEARCH_SUFFIXES:
        candidate = save_dir + suffix
        if not os.path.isdir(candidate):
            continue
        for root, _dirs, files in os.walk(candidate):
            for name in files:
                if not name.endswith(".gguf"):
                    continue
                path = os.path.join(root, name)
                found.append(
                    {
                        "path": path,
                        "mb": round(os.path.getsize(path) / 1024**2, 1),
                        "found_in": candidate,
                        "suffix": suffix,
                    }
                )
    return sorted(found, key = lambda f: -f["mb"])


def export_gguf(
    model,
    tokenizer,
    save_dir: str,
    *,
    quantization: str = "q8_0",
) -> dict:
    """Export and report. Never raises."""
    record = {"save_dir": save_dir, "requested_quantization": quantization}

    started = time.time()
    try:
        model.save_pretrained_gguf(save_dir, tokenizer, quantization_method = quantization)
        record["ok"] = True
    except BaseException as exc:  # noqa: BLE001
        # save.py wraps causes with empty messages, so record the exception type too.
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"[:4000]
    record["seconds"] = round(time.time() - started, 1)

    record["ggufs"] = find_ggufs(save_dir)
    return record


def run_gguf(
    gguf_path: str,
    llama_cpp_dir: str,
    *,
    max_tokens: int = 16,
    timeout: int = 240,
) -> dict:
    """Uses llama-bench and llama-completion, which cannot block on input; llama-cli hung on Kaggle."""
    record = {"gguf": gguf_path}
    for name, argv in (
        ("bench", ["llama-bench", "-m", gguf_path, "-p", "8", "-n", str(max_tokens), "-r", "1"]),
        (
            "completion",
            [
                "llama-completion",
                "-m",
                gguf_path,
                "-p",
                "The capital of France is",
                "-n",
                str(max_tokens),
                "--temp",
                "0",
            ],
        ),
    ):
        exe = os.path.join(llama_cpp_dir, argv[0])
        if not os.path.exists(exe):
            record[name] = {"skipped": f"no {argv[0]} in the bundle"}
            continue
        started = time.time()
        try:
            proc = subprocess.run(
                [exe] + argv[1:],
                capture_output = True,
                text = True,
                timeout = timeout,
                stdin = subprocess.DEVNULL,
            )
            record[name] = {
                "seconds": round(time.time() - started, 1),
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-2000:],
            }
        except BaseException as exc:  # noqa: BLE001
            record[name] = {
                "seconds": round(time.time() - started, 1),
                "error": f"{type(exc).__name__}: {exc}"[:1000],
            }
        if record[name].get("returncode") == 0:
            break
    return record


def export_failures(record: dict, *, accept_quantizations = None) -> list:
    """Callers list every quantization they accept, since gpt-oss overrides q8_0 to MXFP4 by design."""
    if not record:
        return ["GGUF export was never run"]

    failures = []
    if not record.get("ok"):
        failures.append("GGUF export raised: " + str(record.get("error", "no error recorded")))

    ggufs = record.get("ggufs") or []
    if not ggufs:
        # An export can "succeed" and leave no GGUF anywhere.
        failures.append(
            f"no .gguf under {record.get('save_dir')!r} or its _gguf sibling, "
            f"even though the export reported ok={record.get('ok')}"
        )
        return failures

    biggest = ggufs[0]
    if biggest["mb"] <= 1.0:
        failures.append(
            f"the largest .gguf is {biggest['mb']} MB ({biggest['path']}), which is "
            f"a header and no weights"
        )

    if accept_quantizations:
        names = [os.path.basename(g["path"]).lower() for g in ggufs]
        accepted = [q.lower() for q in accept_quantizations]
        if not any(q in n for n in names for q in accepted):
            failures.append(
                f"no exported file names any accepted quantization {sorted(accept_quantizations)}: "
                f"got {names}"
            )
    return failures


def run_failures(record: dict) -> list:
    """Did the exported file actually produce output?"""
    if not record:
        return ["the exported GGUF was never run"]

    attempts = {k: v for k, v in record.items() if isinstance(v, dict)}
    if not attempts:
        return ["no runner was attempted against the exported GGUF"]
    if all(a.get("skipped") for a in attempts.values()):
        return [
            "every GGUF runner was missing from the llama.cpp bundle: "
            + ", ".join(sorted(attempts))
        ]
    if any(a.get("returncode") == 0 for a in attempts.values()):
        return []

    detail = "; ".join(
        f"{name}: "
        + (
            a.get("error")
            or a.get("skipped")
            or f"rc={a.get('returncode')} {(a.get('stderr') or '')[-200:]}"
        )
        for name, a in sorted(attempts.items())
    )
    return [f"the exported GGUF produced no output from any runner ({detail})"]


def llama_cpp_facts(install_output: str, returned) -> dict:
    """install_llama_cpp returns a tuple of binary paths, not a directory, so the directory is derived."""
    paths = list(returned) if isinstance(returned, (tuple, list)) else [returned]
    paths = [str(p) for p in paths]
    return {
        "returned": paths,
        "all_exist": all(os.path.exists(p) for p in paths) if paths else False,
        "dir": os.path.dirname(paths[0]) if paths else None,
        # Tri-state: the prebuilt banner prints only on the install that downloads it, so None means unknown.
        "prebuilt": (
            True
            if PREBUILT_MARKER in (install_output or "")
            else (None if not (install_output or "").strip() else False)
        ),
        "source_build_markers": [m for m in SOURCE_BUILD_MARKERS if m in (install_output or "")],
    }
