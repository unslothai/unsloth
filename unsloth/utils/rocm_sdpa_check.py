# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""ROCm: whether AOTriton's flash / memory-efficient SDPA kernels compute the right answer on this GPU.

Unsloth sets TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 on import, so on archs AOTriton only lists as
experimental (RDNA: gfx11xx, gfx12xx) torch offers those kernels to every SDPA call. On some of them
(gfx1200 RX 9060 XT, #18403) they fail with hipErrorInvalidValue or corrupt memory, and the fault
then surfaces elsewhere as an illegal memory access. A fault poisons the HIP context, so the check
runs in a child process: the GQA shapes Unsloth's attention sends (per-segment strided
views, expanded K/V, a boolean block mask), forward and backward, against an fp32 reference.
A backend that errors, faults, hangs or disagrees is not used by Unsloth's attention on that GPU;
one whose check ran out of memory (the model is often resident already) is kept, with a warning.
CDNA (gfx9xx) is AOTriton's release target and is trusted. gfx1200 is refused without a check: on
torch 2.14 + ROCm 7.2 both kernels pass it, yet padding-free training still faults on them (#18403).
UNSLOTH_ROCM_SDPA_CHECK=0 skips the check (and the gfx1200 refusal).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile

import torch

__all__ = ["rocm_sdpa_verdict", "ROCM_SDPA_CHECK_ENV"]

ROCM_SDPA_CHECK_ENV = "UNSLOTH_ROCM_SDPA_CHECK"
_CHECK_TIMEOUT_S = 240
_VERDICTS = {}
_REFUSED_ARCHS = ("gfx1200",)

# Runs in a fresh interpreter: argv = device index, backend ("flash" | "efficient").
_CHILD = r"""
import json, sys
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

index, name = int(sys.argv[1]), sys.argv[2]
backend = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION}[name]
torch.cuda.set_device(index)
device = torch.device("cuda", index)
# Long and short segments, none a multiple of 64: the reports failed on ~2k-token rows.
lengths = (1300, 748, 37)
total = sum(lengths)
heads, kv_heads = 8, 2
groups = heads // kv_heads
dtypes = [torch.float16] + ([torch.bfloat16] if torch.cuda.is_bf16_supported() else [])
gen = torch.Generator().manual_seed(0)


# fp32 output and Q / K / V grads, one segment and head at a time: the block mask is block-diagonal,
# so this is exact and needs (n, n) scores instead of (heads, T, T) plus their backward.
def reference(q, k, v, grad):
    q, k, v, grad = (x.detach().float() for x in (q, k, v, grad))
    out, dq, dk, dv = torch.empty_like(q), torch.empty_like(q), torch.zeros_like(k), torch.zeros_like(v)
    causal = torch.ones(max(lengths), max(lengths), dtype = torch.bool, device = device).tril_()
    start = 0
    for n in lengths:
        rows = slice(start, start + n)
        for h in range(heads):
            kv = h // groups
            qh, kh, vh = (x[0, i, rows].clone().requires_grad_() for x, i in ((q, h), (k, kv), (v, kv)))
            scores = (qh @ kh.T) * q.shape[-1] ** -0.5
            o = scores.masked_fill(~causal[:n, :n], float("-inf")).softmax(-1) @ vh
            gq, gk, gv = torch.autograd.grad(o, (qh, kh, vh), grad[0, h, rows])
            out[0, h, rows], dq[0, h, rows] = o.detach(), gq
            dk[0, kv, rows] += gk
            dv[0, kv, rows] += gv
        start += n
    return out, dq, dk, dv


def block_mask():
    seg = torch.cat([torch.full((n,), i) for i, n in enumerate(lengths)]).to(device)
    pos = torch.arange(total, device = device)
    return (seg[:, None] == seg[None, :]) & (pos[None, :] <= pos[:, None])


def segments(x):
    out, start = [], 0
    for n in lengths:
        out.append(x[0, :, start:start + n].unflatten(1, (1, n)).transpose(0, 1))
        start += n
    return out


def run(case, dtype, head_dim):
    # (1, T, H, D) transposed, as the attention layers hand Q / K / V over.
    q, k, v = (
        torch.randn(1, total, h, head_dim, generator = gen).to(device, dtype).transpose(1, 2).requires_grad_()
        for h in (heads, kv_heads, kv_heads)
    )
    grad = torch.randn(1, heads, total, head_dim, generator = gen).to(device, dtype)
    wants = reference(q, k, v, grad)
    with sdpa_kernel([backend]):
        if case == "masked":
            ke, ve = (x.repeat_interleave(groups, 1).contiguous() for x in (k, v))
            out = F.scaled_dot_product_attention(q.contiguous(), ke, ve, attn_mask = block_mask()[None, None])
        else:
            gqa = case == "segments_gqa"
            ke, ve = (k, v) if gqa else (x.repeat_interleave(groups, 1) for x in (k, v))
            kw = {"enable_gqa": True} if gqa else {}
            outs = [
                F.scaled_dot_product_attention(qs, ks, vs, is_causal = True, **kw)
                for qs, ks, vs in zip(segments(q), segments(ke), segments(ve))
            ]
            out = torch.cat([o.transpose(0, 1).flatten(1, 2) for o in outs], dim = 1).unsqueeze(0)
        gots = (out, *torch.autograd.grad(out, (q, k, v), grad))
    torch.cuda.synchronize(device)
    worst = 0.0
    for got, want in zip(gots, wants):
        if not torch.isfinite(got).all():
            return False, "non-finite"
        error = (got.float() - want).abs().max().item()
        bound = 0.02 * want.abs().max().item() + 1e-3
        worst = max(worst, error / bound)
    return worst <= 1.0, "worst error / bound %.3g" % worst


# Running out of GPU (or host) memory says nothing about the kernel: the model is often resident
# already. Such a case is inconclusive, never a refusal.
_OOM = tuple(
    e for e in (getattr(torch, "OutOfMemoryError", None), getattr(torch.cuda, "OutOfMemoryError", None)) if e
) + (MemoryError,)


def out_of_memory(error):
    text = str(error)
    return isinstance(error, _OOM) or "out of memory" in text.lower() or "hipErrorOutOfMemory" in text


ok, details, ran, short = True, [], 0, []
cases = ("segments", "segments_gqa") if name == "flash" else ("segments", "masked")
# torch < 2.5 has no enable_gqa (a TypeError, not a kernel fault): Unsloth never sends it there.
if "enable_gqa" not in (F.scaled_dot_product_attention.__doc__ or ""):
    cases = tuple(case for case in cases if case != "segments_gqa")
for dtype in dtypes:
    for head_dim in (64, 128):
        for case in cases:
            label = "%s %s hd%d" % (case, str(dtype).replace("torch.", ""), head_dim)
            try:
                passed, why = run(case, dtype, head_dim)
            except Exception as error:
                if "No available kernel" in str(error):
                    continue  # torch does not offer this backend here, so no call can reach it
                if out_of_memory(error):
                    short.append(label)
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue
                passed, why = False, (str(error).strip().splitlines() or [repr(error)])[0][:200]
            ran += 1
            if not passed:
                ok = False
                details.append("%s: %s" % (label, why))
print("UNSLOTH_ROCM_SDPA_CHECK " + json.dumps({"backend": name, "ok": ok, "ran": ran, "detail": details[:4], "out_of_memory": short}), flush = True)
"""


def _arch(index):
    try:
        name = str(getattr(torch.cuda.get_device_properties(index), "gcnArchName", ""))
    except Exception:
        return ""
    return name.split(":", 1)[0].strip().lower()


def _env_skips_check():
    return os.environ.get(ROCM_SDPA_CHECK_ENV, "1").strip().lower() in ("0", "false", "no", "off")


def _run_children(index):
    """{backend: (ok, detail)} from one child per backend, run side by side."""
    env = dict(os.environ)
    # Launch failures raise at the failing kernel instead of surfacing later as an illegal access.
    env["AMD_SERIALIZE_KERNEL"] = "3"
    env.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    # `python -c` puts its cwd first on sys.path, so a shared temp dir would let another user's
    # json.py / torch.py run in the child: give it a private (0700) directory instead.
    try:
        workdir = tempfile.mkdtemp(prefix = "unsloth_rocm_sdpa_")
    except Exception:
        workdir = None
    try:
        return _run_children_in(index, env, flags, workdir)
    finally:
        if workdir is not None:
            shutil.rmtree(workdir, ignore_errors = True)


def _run_children_in(index, env, flags, workdir):
    children = {}
    for backend in ("flash", "efficient"):
        if workdir is None:
            children[backend] = RuntimeError("no private working directory for the check")
            continue
        try:
            children[backend] = subprocess.Popen(
                [sys.executable, "-c", _CHILD, str(index), backend],
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE,
                stdin = subprocess.DEVNULL,
                cwd = workdir,
                env = env,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                creationflags = flags,
            )
        except Exception as error:
            children[backend] = error
    results = {}
    for backend, child in children.items():
        if isinstance(child, Exception):
            results[backend] = (None, f"could not start the check: {child!r}")
            continue
        try:
            stdout, stderr = child.communicate(timeout = _CHECK_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate()
            results[backend] = (False, f"timed out after {_CHECK_TIMEOUT_S}s")
            continue
        results[backend] = _child_result(child.returncode, stdout, stderr)
    return results


# Resource exhaustion, not a kernel fault: torch / HIP allocator errors and host MemoryError.
_OUT_OF_MEMORY = re.compile(
    r"out of memory|hipErrorOutOfMemory|MemoryError|Cannot allocate memory", re.I
)


def _child_result(returncode, stdout, stderr):
    """(ok, detail) of one finished child; ok is None (inconclusive, backend kept) on out-of-memory."""
    stdout, stderr = stdout or "", stderr or ""
    line = next(
        (l for l in stdout.splitlines() if l.startswith("UNSLOTH_ROCM_SDPA_CHECK ")),
        None,
    )
    if line is None:
        tail = (stderr or stdout).strip().splitlines()[-1:] or [""]
        detail = f"exit code {returncode}: {tail[0][:200]}"
        if sys.platform != "win32" and returncode == -signal.SIGKILL:
            # The kernel's OOM killer (timeouts are handled earlier).
            return None, f"killed (out of memory?), {detail}"
        if _OUT_OF_MEMORY.search(stderr) or _OUT_OF_MEMORY.search(stdout):
            return None, f"out of memory, {detail}"
        return False, detail
    report = json.loads(line.split(" ", 1)[1])
    failed = report.get("detail") or []
    if not report.get("ok"):
        return False, "; ".join(failed)
    short = report.get("out_of_memory") or []
    if short:
        return (
            None,
            f"out of memory in {len(short)} cases ({', '.join(short[:4])}), {report.get('ran', 0)} passed",
        )
    return True, f"{report.get('ran', 0)} cases passed"


def rocm_sdpa_verdict(device_index = 0) -> dict:
    """{"flash", "efficient": usable, "arch", "source", "detail"} for this device, computed once.
    source: "not_rocm" (both usable), "env" (check skipped), "cdna" (trusted), "arch" (both refused),
    "check" (child result)."""
    index = 0 if device_index is None else int(device_index)
    verdict = _VERDICTS.get(index)
    if verdict is not None:
        return verdict
    verdict = {"flash": True, "efficient": True, "arch": "", "source": "not_rocm", "detail": ""}
    if torch.version.hip is not None and torch.cuda.is_available():
        verdict["arch"] = arch = _arch(index)
        if _env_skips_check():
            verdict["source"] = "env"
        elif arch.startswith("gfx9"):
            verdict["source"] = "cdna"
        elif arch in _REFUSED_ARCHS:
            verdict.update(flash = False, efficient = False, source = "arch", detail = "arch refused")
            print(
                f"Unsloth: ROCm {arch} {index}: AOTriton flash and efficient attention fault in "
                f"training on this GPU even when they pass a self-check, so Unsloth's attention will "
                f"not use them. {ROCM_SDPA_CHECK_ENV}=0 uses them anyway."
            )
        else:
            verdict["source"] = "check"
            results = _run_children(index)
            details = []
            unsure = []
            for backend, (ok, detail) in results.items():
                # Could not start or ran out of memory: keep torch's choice.
                verdict[backend] = ok is not False
                if ok is None:
                    unsure.append(backend)
                details.append(f"{backend}: {detail}")
            verdict["detail"] = " | ".join(details)
            refused = [b for b in ("flash", "efficient") if not verdict[b]]
            if refused:
                print(
                    f"Unsloth: ROCm {arch or 'GPU'} {index}: AOTriton "
                    f"{' and '.join(refused)} attention failed a self-check, so Unsloth's attention "
                    f"will not use it ({verdict['detail']}). {ROCM_SDPA_CHECK_ENV}=0 skips the check."
                )
            if unsure:
                print(
                    f"Unsloth: ROCm {arch or 'GPU'} {index}: the AOTriton "
                    f"{' and '.join(unsure)} attention self-check could not finish, so Unsloth keeps "
                    f"using it unchecked ({verdict['detail']})."
                )
    _VERDICTS[index] = verdict
    return verdict
