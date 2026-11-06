// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
const { store } = installLocalStorageFake();
const {
  DEFAULT_PER_MODEL_CONFIG,
  MAX_CHAT_TEMPLATE_BYTES,
  isChatTemplateWithinLimit,
  resolveInitialConfig,
  savePerModelConfig,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

const MODEL = "DavidAU/custom-template-GGUF";
const VARIANT = "Q4_K_M";

function templateOfBytes(bytes: number, unit: string): string {
  return unit.repeat(Math.ceil(bytes / unit.length)).slice(0, bytes);
}

test("a ~230 KB community chat template saves and reloads (#14841)", () => {
  store.clear();
  const template = templateOfBytes(
    230 * 1024,
    "{%- if message['role'] == 'user' -%}{{ message['content'] }}{%- endif -%}\n",
  );
  assert.equal(isChatTemplateWithinLimit(template), true);
  assert.equal(
    savePerModelConfig(MODEL, VARIANT, {
      ...DEFAULT_PER_MODEL_CONFIG,
      chatTemplateOverride: template,
    }),
    true,
  );
  assert.equal(
    resolveInitialConfig(MODEL, VARIANT).config.chatTemplateOverride,
    template,
  );
});

test("a template at the cap still fits the storage budget after JSON escaping", () => {
  store.clear();
  // Newlines and double quotes grow when serialized; single quotes do not.
  const template = templateOfBytes(
    MAX_CHAT_TEMPLATE_BYTES,
    '{{- "<|im_start|>" + message["role"] + "\\n" -}}\n',
  );
  assert.equal(
    savePerModelConfig(MODEL, VARIANT, {
      ...DEFAULT_PER_MODEL_CONFIG,
      chatTemplateOverride: template,
    }),
    true,
  );
  assert.equal(
    resolveInitialConfig(MODEL, VARIANT).config.chatTemplateOverride,
    template,
  );
});

test("a template over the cap is still refused", () => {
  const template = "a".repeat(MAX_CHAT_TEMPLATE_BYTES + 1);
  assert.equal(isChatTemplateWithinLimit(template), false);
  assert.equal(
    savePerModelConfig(MODEL, VARIANT, {
      ...DEFAULT_PER_MODEL_CONFIG,
      chatTemplateOverride: template,
    }),
    false,
  );
});
