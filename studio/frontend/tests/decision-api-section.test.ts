// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { SETTINGS_SEARCH_INDEX } from "../src/features/settings/settings-search.ts";
import { en } from "../src/i18n/locales/en.ts";

function read(path: string): string {
  return readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf-8");
}

// The section reaches the hub and auth barrels, which cannot be imported here, so this asserts on source.
const SECTION = read(
  "../src/features/settings/components/decision-api-section.tsx",
);
const API = read("../src/features/settings/api/systemone.ts");
const API_TAB = read("../src/features/settings/tabs/api-keys-tab.tsx");

test("the API tab renders the section and search finds it", () => {
  assert.match(API_TAB, /<DecisionApiSection \/>/);
  assert.ok(
    SETTINGS_SEARCH_INDEX["api-keys"].includes(
      "settings.apiKeys.decisionApi.title",
    ),
  );
  assert.match(
    SECTION,
    /data-settings-label=\{t\("settings.apiKeys.decisionApi.title"\)\}/,
  );
});

test("only the owner sees it, below the chat usage examples", () => {
  assert.match(
    API_TAB,
    /<UsageExamples[\s\S]*?\/>\s*(?:\{\/\*[^*]*\*\/\}\s*)?\{isOwner \? <DecisionApiSection \/> : null\}/,
  );
  assert.doesNotMatch(SECTION, /useIsAccountOwner/);
});

test("turning it on or picking a model downloads, the device does not", () => {
  assert.match(SECTION, /apply\(\{ enabled: on \}, on\)/);
  assert.match(SECTION, /apply\(\{ model: name \}, true\)/);
  assert.match(
    SECTION,
    /apply\(\{ device: device as SystemOneDevice \}, false\)/,
  );
});

test("a download waits for confirmation before changing the setting", () => {
  const apply = SECTION.slice(
    SECTION.indexOf("const apply = async"),
    SECTION.indexOf("const acceptDownload = async"),
  );
  assert.doesNotMatch(apply, /startDownload\(/);
  const resolve = apply.indexOf(
    "resolveSystemOneDownload(nextModel, nextBackend)",
  );
  assert.ok(resolve >= 0);
  assert.ok(resolve < apply.indexOf("updateSystemOneSettings(settingsPatch)"));
  assert.match(apply, /expectedEnabled: settings\.enabled/);
  assert.match(apply, /expectedModel: settings\.model/);
  assert.match(apply, /expectedBackend: settings\.backend/);
  assert.match(
    apply,
    /setConfirm\(\{\s*plan: nextPlan,\s*patch: settingsPatch,\s*model: nextModel,?\s*\}\);\s*return;/,
  );
  assert.doesNotMatch(
    apply.slice(0, apply.indexOf("setConfirm")),
    /setPlanState/,
  );
  const accept = SECTION.slice(
    SECTION.indexOf("const acceptDownload = async"),
    SECTION.indexOf("const unload = async"),
  );
  assert.ok(
    accept.indexOf("validateSystemOneSettings(accepted.patch)") <
      accept.indexOf("startDownload(accepted.plan)"),
  );
  assert.ok(
    accept.indexOf("startDownload(accepted.plan)") <
      accept.indexOf("updateSystemOneSettings(accepted.patch)"),
  );
  assert.match(
    SECTION,
    /<AlertDialogAction[\s\S]*?acceptDownload\(accepted\)[\s\S]*?<\/AlertDialogAction>/,
  );
  assert.doesNotMatch(SECTION, /apply\(confirm\./);
  assert.match(en.settings.apiKeys.decisionApi.downloadConfirmBody, /\{size\}/);
});

test("the download goes through the manager with the exact files and one scope", () => {
  assert.match(SECTION, /const DOWNLOAD_SCOPE = "systemone";/);
  assert.match(SECTION, /scopeId: DOWNLOAD_SCOPE,/);
  assert.match(SECTION, /files: next\.files,/);
  assert.match(SECTION, /expectedBytes: next\.sizeBytes,/);
  assert.match(
    SECTION,
    /acceptedState === "running" \|\| acceptedState === "complete"/,
  );
  assert.match(SECTION, /resolveSystemOneDownload\(model, backend\)\.then/);
  assert.match(SECTION, /\[enabled, model, backend, downloadDone\]/);
  assert.match(SECTION, /resolveSystemOneDownload\(nextModel, nextBackend\)/);
  assert.match(SECTION, /await resyncSettingsAfterError/);
});

test("environment overrides lock their control and say why", () => {
  for (const [flag, env] of [
    ["enabledLocked", "ENV_DISABLE"],
    ["modelLocked", "ENV_MODEL"],
    ["deviceLocked", "ENV_DEVICE"],
  ]) {
    assert.match(
      SECTION,
      new RegExp(`disabled=\\{busy \\|\\| settings\\.${flag}\\}`),
    );
    assert.match(SECTION, new RegExp(`name: ${env}`));
  }
  assert.equal(en.settings.apiKeys.decisionApi.lockedByEnv, "Set by {name}.");
});

test("GPU is offered only where the backend found one", () => {
  assert.match(
    SECTION,
    /<SelectItem value="gpu" disabled=\{!settings\.gpuAvailable\}>/,
  );
});

test("the copy stays plain", () => {
  const copy = en.settings.apiKeys.decisionApi;
  assert.equal("experimental" in copy, false);
  assert.equal("tryIt" in copy, false);
  for (const value of Object.values(copy)) {
    assert.doesNotMatch(value, /[;\u2014]/);
  }
});

test("the client talks to the settings routes and maps the schema", () => {
  assert.match(API, /const SETTINGS_PATH = "\/api\/settings\/systemone";/);
  assert.match(API, /`\$\{SETTINGS_PATH\}\/unload`/);
  assert.match(API, /`\$\{SETTINGS_PATH\}\/validate`/);
  assert.match(API, /`\$\{SETTINGS_PATH\}\/resolve\$\{query\}`/);
  assert.match(API, /expected_enabled: expectedEnabled/);
  assert.match(API, /expected_model: expectedModel/);
  assert.match(API, /expected_backend: expectedBackend/);
  for (const [camel, snake] of [
    ["enabledLocked", "enabled_locked"],
    ["gpuAvailable", "gpu_available"],
    ["loadedModel", "loaded_model"],
    ["downloadBytes", "download_bytes"],
    ["sizeBytes", "size_bytes"],
  ]) {
    assert.match(API, new RegExp(`${camel}: \\w+\\.${snake}`));
  }
});

test("reports a resident model as loaded only when it is the selected one", () => {
  assert.match(SECTION, /settings\.loadedModel === settings\.model/);
  assert.match(SECTION, /settings\.loadingModel === settings\.model/);
});

test("a failed first load still renders the section with its error", () => {
  assert.match(SECTION, /if \(!settings\) \{\s*return error \?/);
  assert.match(
    SECTION,
    /translate\("settings.apiKeys.decisionApi.loadError"\)/,
  );
});
