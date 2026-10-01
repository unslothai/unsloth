// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { ggufCompactionRequestFields } from "../src/features/chat/utils/auto-compaction.ts";

import { readSrc } from "./helpers/kit.ts";

test("auto-compact on sends truncate_oldest and no policy of its own", () => {
  // No context_policy: the server applies UNSLOTH_CONTEXT_POLICY. Studio used to offer that
  // choice as a setting and no longer does, so this is the only shape an enabled request takes.
  assert.deepEqual(
    ggufCompactionRequestFields({ isGguf: true, autoCompactEnabled: true }),
    { context_overflow: "truncate_oldest" },
  );
});

test("auto-compact off sends an explicit error overflow policy", () => {
  assert.deepEqual(
    ggufCompactionRequestFields({ isGguf: true, autoCompactEnabled: false }),
    { context_overflow: "error" },
  );
});

test("external models never opt into GGUF compaction", () => {
  assert.deepEqual(
    ggufCompactionRequestFields({ isGguf: false, autoCompactEnabled: true }),
    {},
  );
});

test("nothing still offers the removed compaction style", () => {
  // The row is gone, so its state, its request fields and its strings should be too: a leftover
  // setter or key is a setting the user cannot reach but the code still carries.
  for (const file of [
    "features/settings/tabs/chat-tab.tsx",
    "features/chat/stores/chat-runtime-store.ts",
    "features/chat/utils/chat-settings-storage.ts",
    "features/chat/utils/queued-chat-run-settings.ts",
    "features/chat/api/chat-settings-api.ts",
    "features/chat/api/chat-adapter.ts",
    "features/chat/utils/auto-compaction.ts",
    "features/settings/settings-search.ts",
    "i18n/locales/en.ts",
  ]) {
    assert.doesNotMatch(
      readSrc(file),
      /contextPolicy|compactionHeadroomRatio|compactionStyle|compactionDescription/,
      `${file} still carries the removed compaction style`,
    );
  }
});

test("the chat adapter sends compaction fields through the shared helper", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(adapter, /ggufCompactionRequestFields\(/);
  // Through isServedByLlamaCpp, not a catalog row: /api/models/list can replace the row a
  // load minted, and the panel that shows these settings asks the same owner.
  assert.match(adapter, /isGguf: isGgufForCompaction/);
  assert.match(adapter, /loadedIsGguf: runtime\.loadedIsGguf/);
  assert.match(adapter, /isServedByLlamaCpp\(/);
// One request object for both streams, so a media turn cannot lose these fields; the value is
  // derived from THIS turn's messages only (durable-gate.ts), not a variable hoisted earlier.
  assert.match(adapter, /image_base64: findLatestUserImageBase64\(currentTurnMessages\)/);
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
    }),
    { context_overflow: "truncate_oldest" },
  );
});
