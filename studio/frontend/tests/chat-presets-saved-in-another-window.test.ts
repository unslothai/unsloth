// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./store-settings-resolver.mjs", import.meta.url);

const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { useChatRuntimeStore } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);

async function settled(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 600));
}

function preset(name: string, temperature = 0.5) {
  return { name, params: { temperature } };
}

test("a window saves and deletes only its own presets, not its whole stale list", async () => {
  settingsHttp.settings = { customPresets: [preset("Old"), preset("Spare")] };
  await useChatRuntimeStore.getState().hydratePersistedSettings();
  assert.deepEqual(
    useChatRuntimeStore.getState().customPresets.map((p) => p.name),
    ["Old", "Spare"],
  );

  const { customPresets, setCustomPresets } = useChatRuntimeStore.getState();
  const creative = {
    ...customPresets[0],
    name: "Creative",
    params: { ...customPresets[0].params, temperature: 1.1 },
  };
  setCustomPresets([
    ...customPresets.filter((p) => p.name !== "Spare"),
    creative,
  ]);
  await settled();

  const sent = settingsHttp.puts.filter(
    (put) => "customPresets" in put || "editCustomPresets" in put,
  );
  assert.equal(sent.length, 1);
  assert.equal("customPresets" in sent[0], false);
  const edits = sent[0].editCustomPresets as Record<
    string,
    { name: string; params: { temperature: number } } | null
  >;
  assert.deepEqual(Object.keys(edits).sort(), ["Creative", "Spare"]);
  assert.equal(edits.Spare, null);
  assert.equal(edits.Creative?.name, "Creative");
  assert.equal(edits.Creative?.params.temperature, 1.1);
});
