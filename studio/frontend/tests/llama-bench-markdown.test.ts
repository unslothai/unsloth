// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { toLlamaBenchMarkdown } from "../src/features/benchmarks/llama-bench/llama-bench-markdown.ts";

test("a run copies as llama-bench's own -o md table", () => {
  const row = (test: string, p: number, n: number, ts: number) => ({
    test,
    n_prompt: p,
    n_gen: n,
    n_depth: 0,
    avg_ts: ts,
    stddev_ts: 1.234,
    samples_ts: [ts],
    n_gpu_layers: 99,
    flash_attn: 1,
  });
  const md = toLlamaBenchMarkdown(
    [row("pp512", 512, 0, 1450.5), row("tg128", 0, 128, 61.25)],
    {
      model_type: "qwen3 4B Q4_K - Medium",
      model_size: 2.5 * 1024 ** 3,
      model_n_params: 4_020_000_000,
      backends: "ROCm",
      build_commit: "abc123",
      build_number: 9000,
    },
  );
  assert.equal(
    md,
    [
      "| model | size | params | backend | ngl | fa | test | t/s |",
      "| ----- | ---: | -----: | ------- | --: | -: | ---: | --: |",
      "| qwen3 4B Q4_K - Medium | 2.50 GiB | 4.02 B | ROCm | 99 | 1 | pp512 | 1450.50 ± 1.23 |",
      "| qwen3 4B Q4_K - Medium | 2.50 GiB | 4.02 B | ROCm | 99 | 1 | tg128 | 61.25 ± 1.23 |",
      "",
      "build: abc123 (9000)",
    ].join("\n"),
  );
});
