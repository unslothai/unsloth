// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Run settings say whether they are saved, never forget a server row they did not read, and
// group the Advanced rows by purpose.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const CONFIG_PAGE = readFileSync(
  path.join(
    HERE,
    "..",
    "src/features/model-picker/components/model-config-page.tsx",
  ),
  "utf8",
);

function bodyOf(name: string): string {
  const start = CONFIG_PAGE.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `missing ${name}`);
  const next = CONFIG_PAGE.indexOf("\nfunction ", start + 1);
  return CONFIG_PAGE.slice(start, next === -1 ? undefined : next);
}

test("the save status covers every Remember state and is always rendered", () => {
  for (const label of [
    "Saved for this model",
    "Unsaved changes",
    "Saved settings will be forgotten",
    "Not saved yet",
    "Using default settings",
    "Not remembered for this model",
  ]) {
    assert.ok(CONFIG_PAGE.includes(`"${label}"`), `missing status: ${label}`);
  }
  // Mounted on demand it would push Load down under the cursor on the click that saves.
  assert.match(CONFIG_PAGE, /\{saveStatus\}/);
  assert.doesNotMatch(CONFIG_PAGE, /saveStatus &&/);
});

test("the status compares what Save would store against the stored record", () => {
  assert.match(
    CONFIG_PAGE,
    /normalizePerModelConfig\(storedSpeculativeAuto\(runtimeConfig, targetIsMlx\)\),\s*storedRecord\.config/,
  );
  assert.match(
    CONFIG_PAGE,
    /addEventListener\(PER_MODEL_CONFIG_UPDATED_EVENT, bump\)/,
  );
});

test("an unticked Load never forgets a server row the panel never read", () => {
  const persist = CONFIG_PAGE.slice(
    CONFIG_PAGE.indexOf("const persistConfig = "),
    CONFIG_PAGE.indexOf("const finishPersist = "),
  );
  assert.match(persist, /!isExtraArgsHydratedForDraft\(draftKey\)/);
  assert.match(persist, /saved &&\s*!forgetsUnreadRow &&/);
});

test("a fresh editor adopts the server row when nothing changed in flight", () => {
  // The first render's fallback and the primed draft are distinct objects, so an identity
  // check here refused every row and settings saved on another origin never came back.
  assert.match(
    CONFIG_PAGE,
    /!isModelConfigDraftEdited\(draftKey\) &&\s*perModelConfigsEqual\(configRef\.current, configAtStart\) &&/,
  );
  assert.doesNotMatch(CONFIG_PAGE, /configRef\.current === configAtStart/);
});

test("Reasoning Budget shows unlimited as a blank box", () => {
  assert.match(
    CONFIG_PAGE,
    /value=\{config\.reasoningBudget === -1 \? "" : config\.reasoningBudget\}/,
  );
});

test("GGUF Advanced rows sit in titled groups, the llama-server ones hidden for diffusion", () => {
  const gguf = bodyOf("GgufAdvancedSettings");
  const order = ["Memory", "Speed", "Output", "Server"].map((title) =>
    gguf.indexOf(`<AdvancedGroup title="${title}">`),
  );
  assert.ok(
    order.every((at) => at !== -1),
    "a group is missing",
  );
  assert.deepEqual(
    [...order].sort((a, b) => a - b),
    order,
  );
  const server = gguf.slice(order[3] - 40, order[3]);
  assert.match(server, /!isDiffusion && \(\s*$/);
  assert.ok(
    gguf.indexOf("<ExtraArgsRow") > order[3],
    "extra args left the Server group",
  );
});
