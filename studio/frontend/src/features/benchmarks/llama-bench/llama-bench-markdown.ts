// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LlamaBenchMeta, LlamaBenchRow } from "./llama-bench-api";

/** The table llama-bench prints with `-o md`, so a result pastes into an issue or a perf
 * thread next to everyone else's. */
export function toLlamaBenchMarkdown(
  rows: LlamaBenchRow[],
  meta: LlamaBenchMeta,
): string {
  const size = meta.model_size
    ? `${(meta.model_size / 1024 ** 3).toFixed(2)} GiB`
    : "";
  const params = meta.model_n_params
    ? `${(meta.model_n_params / 1e9).toFixed(2)} B`
    : "";
  const lines = [
    "| model | size | params | backend | ngl | fa | test | t/s |",
    "| ----- | ---: | -----: | ------- | --: | -: | ---: | --: |",
    ...rows.map(
      (r) =>
        `| ${meta.model_type ?? ""} | ${size} | ${params} | ${meta.backends ?? ""} | ${
          r.n_gpu_layers ?? ""
        } | ${r.flash_attn ?? ""} | ${r.test} | ${r.avg_ts.toFixed(2)} ± ${(
          r.stddev_ts || 0
        ).toFixed(2)} |`,
    ),
  ];
  const build = meta.build_commit
    ? `\nbuild: ${meta.build_commit} (${meta.build_number ?? "?"})`
    : "";
  return `${lines.join("\n")}\n${build}`.trimEnd();
}
