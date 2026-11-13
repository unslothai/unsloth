#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Build the prebuilt int8 GEMM library, one nvcc unit per (family, tile), static CUDA runtime.

  python build.py --cutlass /path/to/cutlass --out build/ [--jobs 16] [--families sm80,sm75]

SASS only (no PTX): other archs keep Studio's other paths instead of a JIT build.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ABI_VERSION = 1

# (TBM, TBN, TBK, WM, WN, WK, stages, swizzle). The index is the config id the loader races; append only.
FAMILIES = {
    "sm80": {
        "arch_tag": "cutlass::arch::Sm80",
        "gencode": ("80", "86", "89"),
        "out": ("cutlass::bfloat16_t",),
        "xs": ("cutlass::bfloat16_t", "float"),
        "ws": ("cutlass::bfloat16_t", "float"),
        "tiles": (
            (128, 256, 64, 64, 64, 64, 3, 1),
            (256, 128, 64, 64, 64, 64, 3, 1),
            (128, 128, 64, 64, 64, 64, 4, 1),
            (128, 128, 128, 64, 64, 128, 3, 1),
            (64, 128, 64, 32, 64, 64, 4, 1),
            (128, 256, 64, 64, 64, 64, 3, 4),
            (128, 128, 64, 64, 64, 64, 4, 4),
            (64, 64, 64, 32, 32, 64, 5, 1),
            (256, 128, 64, 64, 64, 64, 3, 4),
            # 96 KB of stages fits sm86 / sm89's 99 KB
            (128, 256, 64, 64, 64, 64, 4, 1),
            (256, 128, 64, 64, 64, 64, 4, 1),
            (128, 128, 64, 64, 64, 64, 6, 1),
            (128, 256, 64, 64, 64, 64, 4, 4),
            (256, 128, 128, 64, 64, 128, 2, 1),
            (128, 256, 128, 64, 64, 128, 2, 1),
        ),
    },
    "sm75": {
        "arch_tag": "cutlass::arch::Sm75",
        "gencode": ("75",),
        "out": ("cutlass::half_t",),
        "xs": ("cutlass::half_t", "float"),
        "ws": ("cutlass::half_t", "cutlass::bfloat16_t", "float"),
        "native": True,
        "tiles": (
            (128, 256, 64, 64, 64, 64, 2, 1),
            (256, 128, 64, 64, 64, 64, 2, 1),
            (128, 128, 64, 64, 64, 64, 2, 1),
            (64, 128, 64, 32, 64, 64, 2, 1),
            (128, 256, 64, 64, 64, 64, 2, 4),
            (128, 128, 64, 64, 64, 64, 2, 4),
            (64, 64, 64, 32, 32, 64, 2, 1),
        ),
    },
}

KIND = {"float": 0, "cutlass::bfloat16_t": 1, "cutlass::half_t": 2}
FAMILY_ID = {"sm80": 80, "sm75": 75}


def variants(fam: dict) -> list:
    """(out, xs, ws, bias, epilogue); epilogue 0 = torchao's, 1 = the native W8A8 Linear's."""
    out = [
        (o, x, w, b, 0)
        for o in fam["out"]
        for x in fam["xs"]
        for w in fam["ws"]
        for b in (False, True)
    ]
    if fam.get("native"):
        out += [(o, "float", "float", False, 1) for o in fam["out"]]
    return out


def tile_tu(name: str, fam: dict, index: int) -> str:
    t = fam["tiles"][index]
    lines = [
        '#include "i8gemm_impl.cuh"',
        "namespace {",
        f"using T = ui8::Tile<{', '.join(str(v) for v in t)}>;",
        "}",
        f'extern "C" ui8::RunFn ui8_{name}_t{index}(int out, int xs, int ws, int bias, int epi) {{',
    ]
    for o, x, w, b, e in variants(fam):
        lines.append(
            f"  if (out == {KIND[o]} && xs == {KIND[x]} && ws == {KIND[w]} && bias == {int(b)} && epi == {e}) "
            f"return &ui8::FusedGemm<{fam['arch_tag']}, T, {o}, {x}, {w}, {str(b).lower()}, {e}>::run;"
        )
    lines += ["  return nullptr;", "}", ""]
    return "\n".join(lines)


def api_tu() -> str:
    decls, tables = [], []
    for name, fam in FAMILIES.items():
        n = len(fam["tiles"])
        for i in range(n):
            decls.append(f'extern "C" ui8::RunFn ui8_{name}_t{i}(int, int, int, int, int);')
        tables.append(
            f"static ui8::RunFn (*const k_{name}[])(int, int, int, int, int) = {{"
            + ", ".join(f"&ui8_{name}_t{i}" for i in range(n))
            + "};"
        )
        tables.append(
            f"static const int k_{name}_tiles[][8] = {{"
            + ", ".join("{" + ", ".join(str(v) for v in t) + "}" for t in fam["tiles"])
            + "};"
        )
    fam_cases = "\n".join(
        f"    case {FAMILY_ID[name]}: *table = k_{name}; *count = {len(fam['tiles'])}; *tiles = k_{name}_tiles; break;"
        for name, fam in FAMILIES.items()
    )
    return f"""#include "i8gemm_impl.cuh"
#if defined(_WIN32)
#define UI8_API __declspec(dllexport)
#else
#define UI8_API __attribute__((visibility("default")))
#endif
{chr(10).join(decls)}
{chr(10).join(tables)}

static bool family_table(int family, ui8::RunFn (*const** table)(int, int, int, int, int), int* count,
                         const int (**tiles)[8]) {{
  switch (family) {{
{fam_cases}
    default: return false;
  }}
  return true;
}}

extern "C" {{
UI8_API int ui8_abi_version() {{ return {ABI_VERSION}; }}

// Number of tiles for a family (80 or 75), 0 when absent.
UI8_API int ui8_num_tiles(int family) {{
  ui8::RunFn (*const* table)(int, int, int, int, int) = nullptr;
  int count = 0;
  const int (*tiles)[8] = nullptr;
  return family_table(family, &table, &count, &tiles) ? count : 0;
}}

// The 8 tile parameters (TBM, TBN, TBK, WM, WN, WK, stages, swizzle) into out8; 0 on success.
UI8_API int ui8_tile_info(int family, int tile, int* out8) {{
  ui8::RunFn (*const* table)(int, int, int, int, int) = nullptr;
  int count = 0;
  const int (*tiles)[8] = nullptr;
  if (!family_table(family, &table, &count, &tiles) || tile < 0 || tile >= count || out8 == nullptr) return 1;
  for (int i = 0; i < 8; ++i) out8[i] = tiles[tile][i];
  return 0;
}}

// 1 when (family, tile, out, xs, ws, bias, epilogue) is compiled in; epilogue 0 torchao, 1 native W8A8.
UI8_API int ui8_has_kernel(int family, int tile, int out_kind, int xs_kind, int ws_kind, int has_bias, int epilogue) {{
  ui8::RunFn (*const* table)(int, int, int, int, int) = nullptr;
  int count = 0;
  const int (*tiles)[8] = nullptr;
  if (!family_table(family, &table, &count, &tiles) || tile < 0 || tile >= count) return 0;
  return table[tile](out_kind, xs_kind, ws_kind, has_bias ? 1 : 0, epilogue) != nullptr;
}}

// out[M, N] = dequant(a[M, K] @ w[N, K]^T); kinds: 0 fp32, 1 bf16, 2 fp16. bias may be null. Status code returned.
UI8_API int ui8_gemm(int family, int tile, int out_kind, int xs_kind, int ws_kind, int epilogue, const void* a, const void* w,
                     const void* xs, const void* ws, const void* bias, void* out, int64_t M, int64_t N, int64_t K,
                     int64_t lda, int64_t ldw, int64_t ldc, void* stream) {{
  ui8::RunFn (*const* table)(int, int, int, int, int) = nullptr;
  int count = 0;
  const int (*tiles)[8] = nullptr;
  if (!family_table(family, &table, &count, &tiles) || tile < 0 || tile >= count) return ui8::kNoKernel;
  if (M <= 0 || N <= 0 || K <= 0 || M > 0x7fffffff || N > 0x7fffffff || K > 0x7fffffff) return ui8::kBadArgs;
  if (a == nullptr || w == nullptr || xs == nullptr || ws == nullptr || out == nullptr) return ui8::kBadArgs;
  if (lda < K || ldw < K || ldc < N) return ui8::kBadArgs;
  ui8::RunFn fn = table[tile](out_kind, xs_kind, ws_kind, bias != nullptr ? 1 : 0, epilogue);
  if (fn == nullptr) return ui8::kNoKernel;
  return fn(a, w, xs, ws, bias, out, int(M), int(N), int(K), lda, ldw, ldc, static_cast<cudaStream_t>(stream));
}}
}}
"""


def lib_name() -> str:
    return "unsloth_i8gemm.dll" if os.name == "nt" else "libunsloth_i8gemm.so"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutlass", required = True, help = "CUTLASS checkout (its include/ is used)")
    ap.add_argument("--out", required = True)
    ap.add_argument("--nvcc", default = shutil.which("nvcc") or "nvcc")
    ap.add_argument("--jobs", type = int, default = os.cpu_count() or 4)
    ap.add_argument("--families", default = ",".join(FAMILIES))
    ap.add_argument(
        "--tiles", default = "", help = "comma separated tile ids to keep (default all; testing only)"
    )
    ap.add_argument(
        "--sm",
        default = "",
        help = "comma separated SASS targets to keep, e.g. 89 (default all; testing only)",
    )
    args = ap.parse_args()

    out = Path(args.out).resolve()
    src = out / "src"
    obj = out / "obj"
    src.mkdir(parents = True, exist_ok = True)
    obj.mkdir(parents = True, exist_ok = True)
    shutil.copy2(HERE / "i8gemm_impl.cuh", src / "i8gemm_impl.cuh")
    keep_fams = [f for f in args.families.split(",") if f]
    for name in list(FAMILIES):
        if name not in keep_fams:
            del FAMILIES[name]
    if args.sm:
        want = set(args.sm.split(","))
        for name in list(FAMILIES):
            FAMILIES[name]["gencode"] = tuple(sm for sm in FAMILIES[name]["gencode"] if sm in want)
            if not FAMILIES[name]["gencode"]:
                del FAMILIES[name]
    if args.tiles:
        keep = [int(t) for t in args.tiles.split(",")]
        for fam in FAMILIES.values():
            fam["tiles"] = tuple(t for i, t in enumerate(fam["tiles"]) if i in keep)

    include = Path(args.cutlass).resolve() / "include"
    common = [
        args.nvcc,
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "--expt-relaxed-constexpr",
        f"-I{src}",
        f"-I{include}",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
    ]
    if os.name == "nt":
        # static MSVC runtime (no DLL dependency), the conforming preprocessor CCCL needs
        common += ["-Xcompiler", "/MT", "-Xcompiler", "/Zc:preprocessor", "--diag-suppress=177"]
    else:
        common += ["-Xcompiler", "-fPIC", "-Xcompiler", "-fvisibility=hidden"]
    units = []
    for name, fam in FAMILIES.items():
        gencode = []
        for sm in fam["gencode"]:
            gencode += ["-gencode", f"arch=compute_{sm},code=sm_{sm}"]
        for i in range(len(fam["tiles"])):
            path = src / f"{name}_t{i}.cu"
            path.write_text(tile_tu(name, fam, i), encoding = "utf-8")
            units.append((path, gencode))
    api = src / "i8gemm_api.cu"
    api.write_text(api_tu(), encoding = "utf-8")
    all_gencode = []
    for fam in FAMILIES.values():
        for sm in fam["gencode"]:
            all_gencode += ["-gencode", f"arch=compute_{sm},code=sm_{sm}"]
    units.append((api, all_gencode))

    def compile_one(unit):
        path, gencode = unit
        o = obj / (path.stem + (".obj" if os.name == "nt" else ".o"))
        cmd = common + gencode + ["-c", str(path), "-o", str(o)]
        res = subprocess.run(cmd, capture_output = True, text = True)
        return path.name, res.returncode, res.stdout + res.stderr, o

    objs = []
    failed = False
    with concurrent.futures.ThreadPoolExecutor(max_workers = max(1, args.jobs)) as pool:
        for name, rc, log, o in pool.map(compile_one, units):
            print(f"[{'ok' if rc == 0 else 'FAIL'}] {name}", flush = True)
            if rc != 0:
                failed = True
                print(log[-6000:], flush = True)
            objs.append(str(o))
    if failed:
        return 1
    lib = out / lib_name()
    link = [args.nvcc, "-shared", "-cudart", "static", "-o", str(lib)] + objs + all_gencode
    if os.name != "nt":
        # libstdc++ / libgcc static: the library then needs only glibc from the host
        link += ["-Xcompiler", "-static-libstdc++", "-Xcompiler", "-static-libgcc"]
        link += ["-Xlinker", "--exclude-libs,ALL", "-Xlinker", "-z,defs"]
    res = subprocess.run(link, capture_output = True, text = True)
    if res.returncode != 0:
        print(res.stdout + res.stderr)
        return 1
    print(f"built {lib} ({lib.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
