// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Only a same-model GGUF reload replays our own fitted context; the backend may re-fit only that (#9550).

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { resolveLoadMaxSeqLengthDetailed } = await import(
  "../src/features/chat/presets/preset-policy.ts"
);

const GGUF = {
  modelId: "unsloth/Qwen3-8B-GGUF",
  ggufVariant: "Q4_K_M",
  isGguf: true,
  customContextLength: null as number | null,
  loadedContextLength: null as number | null,
  currentCheckpoint: "",
  activeGgufVariant: null as string | null,
  pinnedMaxSeqLength: null as number | null,
  defaultMaxSeqLength: 4096,
  presetSource: "user" as const,
};
const RELOAD = {
  ...GGUF,
  loadedContextLength: 112896,
  currentCheckpoint: GGUF.modelId,
  activeGgufVariant: "Q4_K_M",
};

test("a same-model GGUF reload is marked as our own resolved context", () => {
  assert.deepEqual(resolveLoadMaxSeqLengthDetailed(RELOAD), {
    value: 112896,
    source: "resident-reload",
  });
});

test("contexts not replayed from the fitter carry other sources", () => {
  const cases: [
    Parameters<typeof resolveLoadMaxSeqLengthDetailed>[0],
    number,
    string,
  ][] = [
    [{ ...RELOAD, customContextLength: 65536 }, 65536, "user-pinned"],
    [
      { ...RELOAD, presetSource: "builtin-default" as const },
      0,
      "builtin-default",
    ],
    [GGUF, 0, "gguf-auto"],
    [
      {
        ...GGUF,
        isGguf: false,
        ggufVariant: null,
        modelId: "unsloth/Qwen3-8B",
        pinnedMaxSeqLength: 8192,
      },
      8192,
      "pinned",
    ],
  ];
  for (const [args, value, source] of cases) {
    assert.deepEqual(resolveLoadMaxSeqLengthDetailed(args), { value, source });
  }
});
