// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_CONTEXT_POLICY,
  compactionStyleValue,
  ggufCompactionRequestFields,
  parseCompactionStyle,
  sanitizeCompactionHeadroomRatio,
} from "../src/features/chat/utils/auto-compaction.ts";

import { readSrc } from "./helpers/kit.ts";

test("the default preserves the server context policy", () => {
  assert.equal(DEFAULT_CONTEXT_POLICY, "inherit");
  assert.equal(compactionStyleValue("inherit", 0.25), "inherit");
  assert.deepEqual(parseCompactionStyle("inherit"), {
    contextPolicy: "inherit",
    compactionHeadroomRatio: 0.25,
  });
  assert.deepEqual(
    ggufCompactionRequestFields({
      isGguf: true,
      autoCompactEnabled: true,
      contextPolicy: "inherit",
      compactionHeadroomRatio: 0.25,
    }),
    { context_overflow: "truncate_oldest" },
  );
});

test("auto-compact off sends an explicit error overflow policy", () => {
  assert.deepEqual(
    ggufCompactionRequestFields({
      isGguf: true,
      autoCompactEnabled: false,
      contextPolicy: "checkpoint",
      compactionHeadroomRatio: 0.25,
    }),
    { context_overflow: "error" },
  );
});

test("checkpoint compaction sends truncate_oldest and the checkpoint policy", () => {
  assert.deepEqual(
    ggufCompactionRequestFields({
      isGguf: true,
      autoCompactEnabled: true,
      contextPolicy: "checkpoint",
      compactionHeadroomRatio: 0.25,
    }),
    { context_overflow: "truncate_oldest", context_policy: "checkpoint" },
  );
});

test("a sliding window sends rolling policy and the extra-trim ratio", () => {
  assert.deepEqual(
    ggufCompactionRequestFields({
      isGguf: true,
      autoCompactEnabled: true,
      contextPolicy: "rolling",
      compactionHeadroomRatio: 0.05,
    }),
    {
      context_overflow: "truncate_oldest",
      context_policy: "rolling",
      compaction_headroom_ratio: 0.05,
    },
  );
});

test("external models never opt into GGUF compaction", () => {
  assert.deepEqual(
    ggufCompactionRequestFields({
      isGguf: false,
      autoCompactEnabled: true,
      contextPolicy: "rolling",
      compactionHeadroomRatio: 0,
    }),
    {},
  );
});

test("the settings select round-trips style values", () => {
  assert.equal(compactionStyleValue("checkpoint", 0.25), "checkpoint");
  assert.equal(compactionStyleValue("rolling", 0), "rolling:0");
  assert.deepEqual(parseCompactionStyle("rolling:0.1"), {
    contextPolicy: "rolling",
    compactionHeadroomRatio: 0.1,
  });
});

test("unsupported headroom ratios snap to an exposed choice", () => {
  assert.equal(sanitizeCompactionHeadroomRatio(0.9), 0.25);
  assert.equal(sanitizeCompactionHeadroomRatio(0.07), 0.05);
  assert.equal(compactionStyleValue("rolling", 0.9), "rolling:0.25");
  assert.deepEqual(
    ggufCompactionRequestFields({
      isGguf: true,
      autoCompactEnabled: true,
      contextPolicy: "rolling",
      compactionHeadroomRatio: 0.9,
    }),
    {
      context_overflow: "truncate_oldest",
      context_policy: "rolling",
      compaction_headroom_ratio: 0.25,
    },
  );
});

test("the chat adapter sends compaction fields through the shared helper", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(adapter, /ggufCompactionRequestFields\(/);
  // Through isServedByLlamaCpp, not a catalog row: /api/models/list can replace the row a
  // load minted, and the panel that shows these settings asks the same owner.
  assert.match(adapter, /isGguf: isGgufForCompaction/);
  assert.match(adapter, /loadedIsGguf: runtime\.loadedIsGguf/);
  assert.match(adapter, /isServedByLlamaCpp\(/);
  // One request object for both streams, so a media turn cannot lose these fields.
  assert.match(adapter, /image_base64: imageBase64/);
});

test("a queued run keeps its own model's llama.cpp verdict after the picker moves on", async () => {
  const { registerBundlerResolver, installLocalStorageFake } = await import(
    "./helpers/kit.ts"
  );
  registerBundlerResolver();
  installLocalStorageFake();
  const { snapshotQueuedChatRunSettings } = await import(
    "../src/features/chat/utils/queued-chat-run-settings.ts"
  );
  const { isServedByLlamaCpp, loadedContextFields } = await import(
    "../src/features/model-picker/model-config/per-model-config.ts"
  );

  // An Ollama GGUF: the backend keeps the opaque inventory ref public, so the checkpoint
  // carries no .gguf suffix, and the load reports no quant. loadedIsGguf is the only
  // evidence llama.cpp serves it.
  const resident = {
    params: { checkpoint: "ollama-manifest:%2Fhome%2Fu%2F.ollama%2Fmanifests%2Fq" },
    activeGgufVariant: null,
    activeNativePathToken: null,
    loadedIsGguf: true,
    loadedContextLength: 8192,
    autoCompactEnabled: true,
    contextPolicy: "rolling" as const,
    compactionHeadroomRatio: 0.1,
  };
  const queued = snapshotQueuedChatRunSettings(
    resident as unknown as Parameters<typeof snapshotQueuedChatRunSettings>[0],
  );

  // Selecting an external provider clears the local residency fields without unloading
  // the model that the queued turn is still going to be served by.
  const live = {
    ...resident,
    params: { checkpoint: "external::openai::gpt-5" },
    activeGgufVariant: null,
    activeNativePathToken: null,
    ...loadedContextFields(null),
  };
  const runtime = { ...live, ...queued };

  const isGguf = isServedByLlamaCpp({
    loadedIsGguf: runtime.loadedIsGguf,
    activeGgufVariant: runtime.activeGgufVariant,
    activeNativePathToken: runtime.activeNativePathToken,
    checkpoint: runtime.params.checkpoint,
  });
  assert.equal(isGguf, true);
  assert.deepEqual(
    ggufCompactionRequestFields({
      isGguf,
      autoCompactEnabled: runtime.autoCompactEnabled,
      contextPolicy: runtime.contextPolicy,
      compactionHeadroomRatio: runtime.compactionHeadroomRatio,
    }),
    {
      context_overflow: "truncate_oldest",
      context_policy: "rolling",
      compaction_headroom_ratio: 0.1,
    },
  );
});
