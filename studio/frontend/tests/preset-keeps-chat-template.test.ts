// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./store-settings-resolver.mjs", import.meta.url);

const { useChatRuntimeStore } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);
const { applyPresetLoadConfig, EMPTY_PRESET_LOAD_CONFIG } = await import(
  "../src/features/chat/presets/preset-load-config.ts"
);

const TEMPLATE = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}";

test("applying a preset keeps the custom chat template for the next load", () => {
  useChatRuntimeStore.setState({ chatTemplateOverride: TEMPLATE, nParallel: 2 } as never);
  applyPresetLoadConfig({ ...EMPTY_PRESET_LOAD_CONFIG, nParallel: 3 });
  const state = useChatRuntimeStore.getState();
  assert.equal(state.nParallel, 3);
  assert.equal(state.chatTemplateOverride, TEMPLATE);
});
