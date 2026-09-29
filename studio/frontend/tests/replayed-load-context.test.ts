// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const RUNTIME = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
const policy = await import("../src/features/chat/presets/preset-policy.ts");

const MODEL = "unsloth/Qwen3.6-27B-MTP-GGUF";
const VARIANT = "UD-Q8_K_XL";
const LOADED = 120064;

function replayed({
  isGguf = true,
  modelId = MODEL,
  customContextLength = null,
  presetSource = "custom",
  gpuMemoryMode = "auto",
  gpuLayers = -1,
}: {
  isGguf?: boolean;
  modelId?: string;
  customContextLength?: number | null;
  presetSource?: "builtin-default" | "custom" | "modified";
  gpuMemoryMode?: "auto" | "manual";
  gpuLayers?: number;
}) {
  const sent = policy.resolveFitMaxSeqLength(
    isGguf,
    gpuMemoryMode,
    gpuLayers,
    customContextLength,
    policy.resolveLoadMaxSeqLength({
      modelId,
      ggufVariant: isGguf ? VARIANT : null,
      isGguf,
      customContextLength,
      loadedContextLength: LOADED,
      currentCheckpoint: MODEL,
      activeGgufVariant: VARIANT,
      isMlx: false,
      pinnedMaxSeqLength: null,
      defaultMaxSeqLength: 4096,
      presetSource,
    }),
  );
  return {
    sent,
    replayed: policy.isReplayedLoadContext(isGguf, customContextLength, sent),
  };
}

test("a same-model reload on a saved preset replays the fitted context", () => {
  for (const presetSource of ["custom", "modified"] as const) {
    assert.deepEqual(replayed({ presetSource }), {
      sent: LOADED,
      replayed: true,
    });
  }
});

test("a typed context is the user's own", () => {
  assert.deepEqual(replayed({ customContextLength: 65536 }), {
    sent: 65536,
    replayed: false,
  });
  assert.deepEqual(replayed({ customContextLength: LOADED }), {
    sent: LOADED,
    replayed: false,
  });
});

test("loads that send no context are not replays", () => {
  assert.equal(replayed({ presetSource: "builtin-default" }).replayed, false);
  assert.equal(replayed({ modelId: "unsloth/Other-GGUF" }).replayed, false);
  assert.equal(replayed({ gpuMemoryMode: "manual" }).replayed, false);
});

test("non-GGUF loads never flag a replay", () => {
  assert.equal(replayed({ isGguf: false }).replayed, false);
});

test("the load request carries the flag", () => {
  assert.match(
    RUNTIME,
    /max_seq_length_auto_derived: isReplayedLoadContext\(\s*isGguf,\s*loadCustomContextLength,\s*loadMaxSeqLength,\s*\)/,
  );
});
