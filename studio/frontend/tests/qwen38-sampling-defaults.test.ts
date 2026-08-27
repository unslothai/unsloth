// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

installLocalStorageFake();
register("./store-settings-resolver.mjs", import.meta.url);

const { applyQwenThinkingParams, resolveQwenThinkingParams } =
  await import("../src/features/chat/utils/qwen-params.ts");
const { useChatRuntimeStore } =
  await import("../src/features/chat/stores/chat-runtime-store.ts");

test("the Qwen3.8 frontend table matches the backend recommendations", () => {
  const defaults = JSON.parse(
    readFileSync(
      new URL(
        "../../backend/assets/configs/inference_defaults.json",
        import.meta.url,
      ),
      "utf8",
    ),
  ) as {
    families: Record<
      string,
      { sampling_modes: Record<string, Record<string, number>> }
    >;
  };
  const modes = defaults.families["qwen3.8"].sampling_modes;

  for (const [thinkingOn, mode] of [
    [true, "thinking"],
    [false, "non_thinking"],
  ] as const) {
    const backend = modes[mode];
    assert.deepEqual(
      resolveQwenThinkingParams("unsloth/Qwen3.8-27B-GGUF", thinkingOn),
      {
        temperature: backend.temperature,
        topP: backend.top_p,
        topK: backend.top_k,
        minP: backend.min_p,
        presencePenalty: backend.presence_penalty,
      },
    );
  }

  assert.deepEqual(
    resolveQwenThinkingParams("unsloth/Qwen3.6-27B-GGUF", true),
    { temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0, presencePenalty: 1.5 },
  );
});

test("the Qwen3.8 Think toggle applies the matching live settings", () => {
  for (const [thinkingOn, temperature, presencePenalty] of [
    [true, 1.0, 0.0],
    [false, 0.7, 1.5],
  ] as const) {
    const store = useChatRuntimeStore.getState();
    useChatRuntimeStore.setState({
      params: {
        ...store.params,
        checkpoint: "unsloth/Qwen3.8-27B-GGUF",
        presencePenalty: 0.0,
      },
      activePresetSource: "builtin-default",
    });

    applyQwenThinkingParams(thinkingOn);

    const params = useChatRuntimeStore.getState().params;
    assert.equal(params.temperature, temperature);
    assert.equal(params.presencePenalty, presencePenalty);
  }
});

test("every status merge is wired to the active Qwen thinking table", () => {
  const source = readFileSync(
    new URL(
      "../src/features/chat/lib/apply-inference-status-to-store.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    source,
    /if \(status\.inference && supportsReasoning\) \{[\s\S]*?resolveQwenThinkingParams\(\s*checkpointId,\s*reasoningAlwaysOn \|\| current\.reasoningEnabled,\s*\)/,
  );
});

test("Qwen3.8 does not change the generic Qwen3 presence penalty", () => {
  assert.equal(
    resolveQwenThinkingParams("unsloth/Qwen3-8B-GGUF", true)?.presencePenalty,
    undefined,
  );
});
