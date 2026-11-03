// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LlamaBenchMeta, LlamaBenchRow } from "./llama-bench-api";

/** The table llama-bench prints with `-o md`, so a result pastes into an issue or a perf
 * thread next to everyone else's. */
export function toLlamaBenchMarkdown(
  rows: LlamaBenchRow[],
  meta: LlamaBenchMeta,
): string {
  const bytes = meta.model_size ?? 0;
  const size = !bytes
    ? ""
    : bytes < 1024 ** 3
      ? `${(bytes / 1024 ** 2).toFixed(2)} MiB`
      : `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
  const n = meta.model_n_params ?? 0;
  const params = !n
    ? ""
    : n < 1e9
      ? `${(n / 1e6).toFixed(2)} M`
      : `${(n / 1e9).toFixed(2)} B`;
  // llama-bench leaves fa out when it ran at its default, auto (-1).
  const fa = rows.some((r) => Number(r.flash_attn ?? -1) !== -1);
  const cols = (cells: string[], faCell: string) =>
    `| ${[...cells.slice(0, 5), ...(fa ? [faCell] : []), ...cells.slice(5)].join(" | ")} |`;
  const lines = [
    cols(["model", "size", "params", "backend", "ngl", "test", "t/s"], "fa"),
    cols(["-----", "---:", "-----:", "-------", "--:", "---:", "--:"], "-:"),
    ...rows.map((r) =>
      cols(
        [
          meta.model_type ?? "",
          size,
          params,
          meta.backends ?? "",
          String(r.n_gpu_layers ?? ""),
          r.test,
          `${r.avg_ts.toFixed(2)} ± ${(r.stddev_ts || 0).toFixed(2)}`,
        ],
        String(r.flash_attn ?? ""),
      ),
    ),
  ];
  const build = meta.build_commit
    ? `\nbuild: ${meta.build_commit} (${meta.build_number ?? "?"})`
    : "";
  return `${lines.join("\n")}\n${build}`.trimEnd();
}

/** How a run that didn't finish ended, or null for a full one. */
export function runEnding(meta: LlamaBenchMeta, done: number): string | null {
  const of = meta.total ? ` at ${done} of ${meta.total}` : "";
  if (meta.status === "cancelled") return `Stopped${of}`;
  if (meta.status === "error") return `Failed${of}`;
  return null;
}
