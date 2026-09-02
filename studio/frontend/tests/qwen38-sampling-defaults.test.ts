// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

installLocalStorageFake();
register("./store-settings-resolver.mjs", import.meta.url);

const { applyQwenThinkingParams } = await import(
  "../src/features/chat/utils/qwen-params.ts"
);
const { resolveQwenThinkingParams } = await import(
  "../src/features/chat/utils/qwen-sampling-table.ts"
);
const { useChatRuntimeStore } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);

test("Qwen3.8 reuses the Qwen3.6 sampling table in both modes", () => {
  for (const thinkingOn of [true, false]) {
    const qwen36 = resolveQwenThinkingParams(
      "unsloth/Qwen3.6-27B-GGUF",
      thinkingOn,
    );
    const qwen38 = resolveQwenThinkingParams(
      "unsloth/Qwen3.8-27B-GGUF",
      thinkingOn,
    );

    assert.deepEqual(qwen38, qwen36);
    assert.equal(qwen38?.presencePenalty, 1.5);
  }
});

test("the Qwen3.8 Think toggle puts 1.5 in the live chat settings", () => {
  for (const thinkingOn of [true, false]) {
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

    assert.equal(useChatRuntimeStore.getState().params.presencePenalty, 1.5);
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
