#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``nvfp4_budget_profile.py`` / ``nvfp4_budget_attention_ab.py`` JSON cells -> one Markdown report.

Reads only, and recomputes nothing: every number in the report is a field of a cell JSON, so the
report can be regenerated from the artifacts without a GPU. Cells are emitted in ``--order`` (a file
of tags, one per line) when given, otherwise in filename order; ``--notes FILE`` splices a prose
file in after the index table, which is where the run's caveats belong.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# The cell order of the image pass this harness was written for. A tag with no JSON is skipped, so
# this is a preference, not a requirement.
DEFAULT_ORDER = [
    "z-image_512_fp8_graphson_shipped",
    "z-image_512_fp8_graphsoff_shipped",
    "z-image_512_nvfp4_graphson_shipped",
    "z-image_512_nvfp4_graphsoff_shipped",
    "z-image_1024_fp8_graphson_shipped",
    "z-image_1024_fp8_graphsoff_shipped",
    "z-image_1024_nvfp4_graphson_shipped",
    "z-image_1024_nvfp4_graphsoff_shipped",
    "flux.1_1024_fp8_graphson_shipped",
    "flux.1_1024_fp8_graphsoff_shipped",
    "flux.1_1024_nvfp4_graphson_shipped",
    "flux.1_1024_nvfp4_graphsoff_shipped",
    "flux.1_1024_fp8_graphson_shipped_nodynamo",
    "flux.1_1024_fp8_graphsoff_shipped_nodynamo",
    "flux.1_1024_nvfp4_graphson_shipped_nodynamo",
    "flux.1_1024_nvfp4_graphsoff_shipped_nodynamo",
    "flux.1_1024_nvfp4_compileattempt_graphson_shipped",
    "qwen-image_1024_fp8_graphson_shipped",
    "qwen-image_1024_fp8_graphsoff_shipped",
    "qwen-image_1024_nvfp4_graphson_shipped_REFUSED",
    "wan2.2-ti2v-5b_1280x704x121_fp8_graphson_shipped",
    "wan2.2-ti2v-5b_1280x704x121_nvfp4_graphson_shipped",
    "z-image_1024_nvfp4_regionalref_graphson_shipped",
    "z-image_1024_nvfp4_regionalref_graphsoff_shipped",
    "z-image_1024_nvfp4_wholecompile_graphson_shipped",
    "z-image_1024_nvfp4_wholecompile_graphsoff_shipped",
]

DEFAULT_AB = ["attn_ab_z-image_1024_nvfp4_graphson", "attn_ab_flux.1_1024_nvfp4_graphson"]


def fmt(x, n = 4):
    return "n/a" if x is None else f"{x:.{n}f}"


def _tags(path: Path | None, default: list) -> list:
    """Tags from a file, one per line, ``#`` comments allowed; the built-in order otherwise."""
    if path is None:
        return list(default)
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def main(argv = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required = True, help = "directory of cell JSONs")
    ap.add_argument("--out", required = True, help = "Markdown report to write")
    ap.add_argument("--title", default = "NVFP4 image time budget")
    ap.add_argument(
        "--order", default = None, help = "file of cell tags, one per line; built-in order otherwise"
    )
    ap.add_argument("--ab-order", default = None, help = "file of attention A/B tags, one per line")
    ap.add_argument("--notes", default = None, help = "Markdown file spliced in after the index table")
    args = ap.parse_args(argv)

    out_dir = Path(args.results_dir)
    order = _tags(args.order, DEFAULT_ORDER)
    ab_order = _tags(args.ab_order, DEFAULT_AB)
    notes = Path(args.notes).read_text() if args.notes else ""

    lines = [f"# {args.title}", ""]
    lines += [
        "Per-render GPU time budget for the shipped Studio image path, one process per cell,",
        "`scripts/nvfp4_budget_profile.py`. Wall is the median of 7 unprofiled renders;",
        "GPU busy is the UNION of device-side intervals over the 2 profiled renders (summing",
        "durations double counts overlapping streams); host idle is wall minus busy. Bucket shares",
        "are of GPU busy, so they do not sum to 100 when kernels overlap.",
        "",
    ]
    index = [
        "| cell | arm | graphs | p50 wall s | GPU busy s | host idle s | busy % of wall | clean |",
        "|---|---|---|---|---|---|---|---|",
    ]
    bodies = []
    unmatched_all: dict = {}
    for tag in order:
        path = out_dir / f"{tag}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        if d.get("error"):
            index.append(
                f"| {tag} | {d.get('arm')} | {d.get('graphs')} | refused | refused | "
                f"refused | refused | {d.get('clean')} |"
            )
            bodies.append(f"\n## {tag}\n\nREFUSED at load: `{d['error']}`\n")
            continue
        index.append(
            f"| {tag} | {d['arm']} | {d['graphs']} | {fmt(d['p50_s'])} | "
            f"{fmt(d['gpu_busy_union_s'])} | {fmt(d['host_idle_s'])} | "
            f"{100 * d['gpu_busy_fraction_of_wall']:.1f}% | {d['clean']} |"
        )
        b = [
            f"\n## {tag}\n",
            f"model `{d['model']}` | steps {d['steps']} | {d['resolution']}px | "
            f"speed_optims {d.get('speed_optims')} | cuda_graph_reason "
            f"`{d.get('cuda_graph_reason')}` | graph stats {d.get('cuda_graph_stats')}",
            "",
            f"p50 wall **{fmt(d['p50_s'])} s** (min {fmt(d['min_s'])}) | GPU busy "
            f"**{fmt(d['gpu_busy_union_s'])} s** | host idle **{fmt(d['host_idle_s'])} s** | "
            f"profiler overhead x{d['profiler_overhead_ratio']:.2f} | phase-sync overhead "
            f"{fmt(d['phase_sync_overhead_s'])} s | contention pre/post "
            f"{d['contention']['pre']['verdict']}/{d['contention']['post']['verdict']}",
            "",
            "| bucket | calls / render | ms / render | % GPU busy |",
            "|---|---|---|---|",
        ]
        for name, row in d["buckets"].items():
            b.append(
                f"| {name} | {row['calls_per_render']:.0f} | {row['ms_per_render']:.2f} | "
                f"{row['pct_busy']:.1f}% |"
            )
        b += [
            "",
            "Attention, split by kernel:",
            "",
            "| kernel | calls / render | ms / render | us / call |",
            "|---|---|---|---|",
        ]
        for r in d.get("attention_by_kernel", []):
            b.append(
                f"| `{r['name']}` | {r['calls_per_render']:.0f} | {r['ms_per_render']:.2f} | "
                f"{r['us_per_call']:.1f} |"
            )
        b += [
            "",
            f"D2D memcpy {d['memcpy_d2d']['calls_per_render']:.0f} per render "
            f"({d['memcpy_d2d']['calls_per_step']:.1f} per step, "
            f"{d['memcpy_d2d']['ms_per_render']:.2f} ms); graph IO "
            f"{d.get('cuda_graph_io')}",
            "",
        ]
        bodies.append("\n".join(b))
        for r in d.get("unmatched_top", []):
            key = r["name"]
            cur = unmatched_all.setdefault(key, {"ms": 0.0, "calls": 0.0, "cells": []})
            cur["ms"] += r["ms_per_render"]
            cur["calls"] += r["calls_per_render"]
            cur["cells"].append(tag)
    lines += index + ["", notes]
    lines += ["## Unmatched kernels (bucket `other`, top 30 by summed ms across cells)", ""]
    if unmatched_all:
        lines += ["| kernel | summed ms/render | calls/render | cells |", "|---|---|---|---|"]
        for name, v in sorted(unmatched_all.items(), key = lambda kv: -kv[1]["ms"])[:30]:
            lines.append(
                f"| `{name[:140]}` | {v['ms']:.4f} | {v['calls']:.0f} | " f"{len(v['cells'])} |"
            )
    else:
        lines.append("None: every kernel in every cell matched a bucket.")
    # ---------------------------------------------------------------- attention backend A/B
    ab_lines = [
        "",
        "## Attention backend A/B (nvfp4 arm, graphs on, compiled, one process each)",
        "",
        "Backends forced with `set_attention_backend` on every denoiser, then",
        "`cuda_graph.reset_all` and one warm render (the capture baked the previous kernel),",
        "rotated round robin and paired by seed. Latent and LPIPS are against the cuDNN",
        "render at the same seed.",
        "",
    ]
    for tag in ab_order:
        path = out_dir / f"{tag}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        ab_lines += [
            f"### {tag}",
            "",
            f"speed_optims {d.get('speed_optims')} | rotations {d.get('rotations')} | "
            f"clean {d.get('clean')} | dropped {d.get('dropped')}",
            "",
            "| backend | p50 s | min s | sd s | GPU busy s | attention ms/render | "
            "speedup vs cuDNN | latent max-abs | latent mean-abs | LPIPS vs cuDNN |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for label, r in (d.get("per_backend") or {}).items():
            lat = r.get("latent_vs_cudnn") or {}
            pair = r.get("paired_vs_cudnn") or {}
            ab_lines.append(
                f"| {label} | {r['p50_s']:.4f} | {r['min_s']:.4f} | {r['sd_s']:.4f} | "
                f"{(r.get('gpu_busy_union_s') or 0):.4f} | {r['attention_ms_per_render']:.2f} | "
                f"{pair.get('speedup_p50', float('nan')):.3f} | "
                f"{lat.get('max_abs', 'n/a')} | {lat.get('mean_abs', 'n/a')} | "
                f"{r.get('lpips_vs_cudnn')} |"
            )
        ab_lines += ["", "Attention kernels per backend:", ""]
        for label, r in (d.get("per_backend") or {}).items():
            for a in r.get("attention_by_kernel", []):
                ab_lines.append(
                    f"- {label}: `{a['name'][:150]}` x{a['calls_per_render']:.0f} "
                    f"{a['ms_per_render']:.2f} ms {a['us_per_call']:.1f} us/call"
                )
        ab_lines.append("")
    lines += ab_lines
    lines += bodies
    report = Path(args.out)
    report.parent.mkdir(parents = True, exist_ok = True)
    report.write_text("\n".join(lines) + "\n")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
