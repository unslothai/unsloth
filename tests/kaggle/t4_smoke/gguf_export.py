# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Export a trained model to GGUF, and prove the file exists and is usable.

Shared by the payloads, in the shape the rest of this directory uses: nothing
here raises, everything returns a record, and a separate pure function turns
that record into failures. A diagnostic that kills the payload it diagnoses
leaves the leg reporting nothing at all.

Every constant below is a measurement, not a guess. The two that a natural
implementation gets WRONG, both learned by getting them wrong:

**1. The GGUF is not in the directory you passed.** On kernel
unsloth-probe-gguf-q8-peft-920e3e, `save_pretrained_gguf("/tmp/q8p", ...)` put
the merged `model.safetensors` (1136.9 MB) in `/tmp/q8p` and the actual
`qwen3-0.6b.Q8_0.gguf` (609.8 MB) in the SIBLING `/tmp/q8p_gguf`. Code that
globs the directory it passed finds no `.gguf`, raises nothing, and reports a
successful export. So this searches both and records which one held the file.

**2. `install_llama_cpp()` returns a TUPLE**, `(llama-quantize,
convert_hf_to_gguf.py)`, not a path to a bin directory. An earlier probe
treated it as a directory, found nothing in it, and reported "0 binaries" --
which was the probe being wrong rather than the bundle being empty.

Two more facts worth having here rather than rediscovering:

* llama.cpp installs PREBUILT on Kaggle, about 10-46s, logging "Installing
  prebuilt llama.cpp ... - skipping compilation". A source build is silent,
  correct, and costs many minutes on 4 vCPUs, so "it worked" is not the
  interesting assertion -- "it did not compile" is.
* The bundle is the `-cpu` one, so GGUF inference runs on CPU. Fine for a 0.6B
  or a 2B; a reason not to lean on GGUF inference for a 20B.

A model that REFUSES the requested quantization is not a failure. gpt-oss
answers q8_0 with "GPT-OSS does not support GGUF quantization (requested:
q8_0). Overriding to MXFP4 format", by design, so the caller says what it will
accept rather than this module assuming q8_0 is universal.
"""

from __future__ import annotations

import os
import subprocess
import time

# Where a GGUF may legitimately appear, relative to the directory passed to
# save_pretrained_gguf. The empty string is that directory itself; "_gguf" is
# the sibling unsloth actually writes to. Both are searched because relying on
# either alone has already produced a wrong answer.
GGUF_SEARCH_SUFFIXES = ("", "_gguf")

# Marks the prebuilt branch in unsloth's own install output.
PREBUILT_MARKER = "skipping compilation"

# Substrings that mean a source build happened instead. If any appears, the
# leg took the expensive path and should say so even though it succeeded.
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
        # unsloth/save.py:4777 wraps the real cause in
        # `RuntimeError(f"Failed to save model: {e}")` and interpolates str(e)
        # from exceptions whose args are empty, producing a message that ends
        # at the colon with no cause in it. Record the type too, so a report
        # from that path still names something.
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
    """Run the exported file, because an existing GGUF is not a working one.

    NOT via `llama-cli`. It hung for its entire budget twice on Kaggle -- 600s
    on kernel unsloth-probe-gguf-q8-peft-920e3e and 180s on -gguf-infer-50a98a,
    the second with stdin closed AND `-no-cnv` -- while 16 tokens from a 610 MB
    Q8_0 on CPU is seconds of work. Whatever it waits for is not stdin, so this
    uses binaries that cannot be interactive at all: `llama-bench` runs a fixed
    workload and exits, `llama-completion` is one-shot.
    """
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
        # One success is enough; the second is only run when the first is
        # unavailable or failed, so a healthy leg pays for one process.
        if record[name].get("returncode") == 0:
            break
    return record


def export_failures(record: dict, *, accept_quantizations = None) -> list:
    """The pass rule, as a pure function of the record.

    `accept_quantizations` is the set of quantization names the caller will
    accept in the filename. Callers pass more than one where the model is
    allowed to override the request -- gpt-oss answers q8_0 with MXFP4 by
    design, and failing on that would be failing on documented behaviour.
    """
    if not record:
        return ["GGUF export was never run"]

    failures = []
    if not record.get("ok"):
        failures.append("GGUF export raised: " + str(record.get("error", "no error recorded")))

    ggufs = record.get("ggufs") or []
    if not ggufs:
        # The trap, stated in the failure itself so the reader does not have to
        # know it: an export can "succeed" and leave no GGUF anywhere.
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
    """Was the PREBUILT branch taken, and what did the installer hand back?

    `returned` is whatever `install_llama_cpp()` returned -- a TUPLE of
    (llama-quantize, convert_hf_to_gguf.py) as measured, not a directory. The
    directory is derived from it rather than assumed, so a release that changes
    the layout shows up as a missing binary instead of a wrong path.
    """
    paths = list(returned) if isinstance(returned, (tuple, list)) else [returned]
    paths = [str(p) for p in paths]
    return {
        "returned": paths,
        "all_exist": all(os.path.exists(p) for p in paths) if paths else False,
        "dir": os.path.dirname(paths[0]) if paths else None,
        "prebuilt": PREBUILT_MARKER in (install_output or ""),
        "source_build_markers": [m for m in SOURCE_BUILD_MARKERS if m in (install_output or "")],
    }
