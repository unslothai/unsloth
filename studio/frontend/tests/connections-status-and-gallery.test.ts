// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { type ViteDevServer, createServer } from "vite";

import {
  checkFailureDetail,
  connectionStatus,
  formatCheckedAgo,
  suggestedModelIds,
} from "../src/features/chat/connection-status.ts";
import { readSrc } from "./helpers/kit.ts";

const NOW = 1_800_000_000_000;
const keyed = { hasApiKey: true, models: ["m"] };
const on = { enabled: true, keyOptional: false, now: NOW };

test("a switched-off connection reads Off whatever else is true", () => {
  const check = {
    ok: false,
    message: "x",
    kind: "auth" as const,
    checkedAt: NOW,
  };
  assert.deepEqual(connectionStatus(keyed, { ...on, enabled: false, check }), {
    tone: "muted",
    label: "Off",
  });
});

test("a missing key outranks a stale check, and a self-hosted server needs none", () => {
  const ok = { ok: true, message: "", checkedAt: NOW };
  assert.equal(
    connectionStatus({ hasApiKey: false, models: ["m"] }, { ...on, check: ok })
      .label,
    "Needs API key",
  );
  assert.equal(
    connectionStatus(
      { hasApiKey: false, models: ["m"] },
      { ...on, keyOptional: true, check: ok },
    ).label,
    "Connected",
  );
});

test("a ChatGPT subscription reports its sign-in, not a key", () => {
  const codex = {
    authKind: "chatgpt_oauth" as const,
    hasApiKey: false,
    models: ["m"],
  };
  assert.equal(
    connectionStatus({ ...codex, authStatus: "reauthorization_required" }, on)
      .label,
    "Sign in again",
  );
  assert.equal(
    connectionStatus({ ...codex, authStatus: "disconnected" }, on).label,
    "Not signed in",
  );
  assert.equal(
    connectionStatus({ ...codex, authStatus: "connected" }, on).label,
    "Signed in",
  );
});

test("a failed check names its cause and drops the boilerplate prefix", () => {
  const failed = (
    kind: "auth" | "billing" | "rate_limit" | "unreachable" | "timeout" | null,
  ) =>
    connectionStatus(keyed, {
      ...on,
      check: {
        ok: false,
        message: "Connection failed: The server said no.",
        kind,
        checkedAt: NOW,
      },
    });
  assert.deepEqual(failed("auth"), {
    tone: "error",
    label: "Key rejected",
    detail: "Check the API key.",
    full: "The server said no.",
  });
  // No short hint for an unclassified failure: the server's words stand.
  assert.equal(failed(null).detail, "The server said no.");
  assert.equal(failed("billing").label, "No credit");
  assert.equal(failed("rate_limit").label, "Rate limited");
  assert.equal(failed("unreachable").label, "Unreachable");
  assert.equal(failed("timeout").label, "Timed out");
  assert.equal(failed(null).label, "Check failed");
  assert.equal(
    checkFailureDetail("Connection failed. Ask the owner."),
    "Ask the owner.",
  );
});

test("a passing check is Connected with its age, and an unchecked one only Ready", () => {
  assert.deepEqual(
    connectionStatus(keyed, {
      ...on,
      check: { ok: true, message: "", checkedAt: NOW - 120_000 },
    }),
    { tone: "ok", label: "Connected", detail: "Checked 2m ago" },
  );
  assert.deepEqual(connectionStatus(keyed, on), {
    tone: "muted",
    label: "Ready",
  });
  assert.equal(
    connectionStatus({ hasApiKey: true, models: [] }, on).label,
    "No models enabled",
  );
  assert.equal(
    connectionStatus(keyed, { ...on, checking: true }).label,
    "Checking",
  );
});

test("check ages read as just now, minutes, hours, then days", () => {
  assert.equal(formatCheckedAgo(NOW - 10_000, NOW), "just now");
  assert.equal(formatCheckedAgo(NOW - 5 * 60_000, NOW), "5m ago");
  assert.equal(formatCheckedAgo(NOW - 3 * 3_600_000, NOW), "3h ago");
  assert.equal(formatCheckedAgo(NOW - 2 * 86_400_000, NOW), "2d ago");
  assert.equal(formatCheckedAgo(NOW + 5_000, NOW), "just now");
});

test("suggestions are the newest listed seeds, capped, or a whole short self-hosted list", () => {
  assert.deepEqual(suggestedModelIds(["c", "b", "a"], ["a", "c", "x"]), [
    "c",
    "a",
  ]);
  assert.deepEqual(
    suggestedModelIds(
      ["1", "2", "3", "4", "5", "6", "7"],
      ["1", "2", "3", "4", "5", "6", "7"],
    ).length,
    6,
  );
  assert.deepEqual(suggestedModelIds([], ["qwen3:14b", "gemma4:12b"]), [
    "qwen3:14b",
    "gemma4:12b",
  ]);
  assert.deepEqual(
    suggestedModelIds([], ["1", "2", "3", "4", "5", "6", "7"]),
    [],
  );
  // Seeds the provider no longer lists are not replaced by a guess.
  assert.deepEqual(suggestedModelIds(["retired"], ["a", "b"]), []);
});

let vite: ViteDevServer;
type Group = { id: string; items: { providerType: string; name: string }[] };
let buildConnectionGallery: (registry: unknown[], query?: string) => Group[];
let formatContextWindow: (tokens: number) => string;

before(async () => {
  vite = await createServer({
    appType: "custom",
    server: { middlewareMode: true },
  });
  const gallery = await vite.ssrLoadModule(
    "/src/features/chat/connection-gallery.tsx",
  );
  buildConnectionGallery = gallery.buildConnectionGallery;
  const list = await vite.ssrLoadModule(
    "/src/features/chat/connection-model-list.tsx",
  );
  formatContextWindow = list.formatContextWindow;
});

after(async () => {
  await vite.close();
});

const entry = (providerType: string, displayName: string) => ({
  provider_type: providerType,
  display_name: displayName,
});
const REGISTRY = [
  entry("openai_codex", "ChatGPT / Codex subscription"),
  entry("openai", "OpenAI"),
  entry("anthropic", "Anthropic"),
  entry("deepseek", "DeepSeek"),
  entry("opencode", "OpenCode Console"),
  entry("typesafe", "TypeSafe"),
];

test("the gallery leads with popular providers and always offers the self-hosted presets", () => {
  const groups = buildConnectionGallery(REGISTRY);
  assert.deepEqual(
    groups.map((group) => group.id),
    ["popular", "local", "more", "even-more"],
  );
  assert.deepEqual(
    groups[0].items.map((item) => item.providerType),
    ["openai_codex", "openai", "anthropic", "deepseek", "custom"],
  );
  assert.deepEqual(
    groups[1].items.map((item) => item.providerType),
    ["llama_cpp", "vllm", "sglang", "ollama"],
  );
  assert.deepEqual(
    groups[2].items.map((item) => item.name),
    ["OpenCode", "Amazon Bedrock"],
  );
  // Folded behind a click in the UI.
  assert.deepEqual(
    groups[3].items.map((item) => item.name),
    ["Azure OpenAI", "Google Vertex AI", "TypeSafe"],
  );
});

test("gallery search ignores punctuation and drops empty groups", () => {
  assert.deepEqual(
    buildConnectionGallery(REGISTRY, "llamacpp").map((group) => [
      group.id,
      group.items.map((item) => item.providerType),
    ]),
    [["local", ["llama_cpp"]]],
  );
  assert.deepEqual(
    buildConnectionGallery(REGISTRY, "Open AI").flatMap((group) =>
      group.items.map((item) => item.providerType),
    ),
    ["openai_codex", "openai", "azure"],
  );
  assert.deepEqual(buildConnectionGallery(REGISTRY, "zzz"), []);
});

test("context windows read as K and M", () => {
  assert.equal(formatContextWindow(131_072), "131K");
  assert.equal(formatContextWindow(1_000_000), "1M");
  assert.equal(formatContextWindow(1_048_576), "1M");
  assert.equal(formatContextWindow(1_050_000), "1.1M");
  assert.equal(formatContextWindow(512), "512");
});

const dialog = readSrc("features/chat/chat-providers-dialog.tsx");
const sync = readSrc("features/chat/sync-external-providers.ts");

test("a switched-off connection stays listed in settings and out of every picker", () => {
  assert.match(
    sync,
    /const syncedProviders = syncedRows\.filter\(\(provider\) => enabledIds\.has\(provider\.id\)\);/,
  );
  assert.match(
    sync,
    /\.setDisabledProviders\(syncedRows\.filter\(\(provider\) => !enabledIds\.has\(provider\.id\)\)\);/,
  );
  assert.match(
    dialog,
    /await updateProviderConfig\(provider\.id, \{ isEnabled: enabled \}\);/,
  );
  assert.match(
    dialog,
    /\.\.\.disabledProviders\.map\(\(provider\) => \(\{ provider, enabled: false \}\)\),/,
  );
});

test("a typed key is checked once it stops changing, and a late answer cannot land in another form", () => {
  assert.match(
    dialog,
    /key\.length < MIN_AUTO_CHECK_KEY_LENGTH \|\|\s*key === checkedKeyRef\.current/,
  );
  assert.match(
    dialog,
    /window\.setTimeout\(\s*\(\) => void loadModels\(\{ keepSelection: true \}\),\s*AUTO_CHECK_DELAY_MS,\s*\)/,
  );
  // Only a load the user asked for may untick a model the fresh list lacks.
  assert.match(
    dialog,
    /const stillListed = keepSelection \? prev : prev\.filter\(\(id\) => modelIds\.includes\(id\)\);/,
  );
  assert.match(dialog, /checkedKeyRef\.current = legacyKey\.trim\(\);/);
  assert.match(
    dialog,
    /const owned = \(\) => silent \|\| request === formCheckRequestRef\.current;/,
  );
  assert.match(dialog, /else if \(owned\(\)\) setModelsLoading\(false\);/);
  // Every way out of a form retires the request it had in flight.
  for (const fn of ["resetForm", "editProvider"]) {
    const body = dialog.match(
      new RegExp(`function ${fn}\\([^)]*\\) \\{([\\s\\S]*?)\\n {2}\\}`),
    )?.[1];
    assert.ok(body, `${fn} not found`);
    assert.match(body, /formCheckRequestRef\.current \+= 1;/);
  }
  // Switching provider retires the load too, and the load it retired cannot clear its own spinner.
  const choose = dialog.match(/const chooseProviderType = \(value: string\) => \{([\s\S]*?)\n {2}\};/)?.[1];
  assert.ok(choose, "chooseProviderType not found");
  assert.match(
    choose,
    /formCheckRequestRef\.current \+= 1;[\s\S]*?setModelsLoading\(false\);/,
  );
});

test("a key the listing cannot vouch for is tested first", () => {
  assert.match(
    dialog,
    /const KEY_CHECKED_SEPARATELY = new Set\(\["openrouter", "huggingface", "vercel"\]\);/,
  );
  assert.match(
    dialog,
    /if \(!silent && typedKey && KEY_CHECKED_SEPARATELY\.has\(backendProviderType\)\) \{\s*const keyTest = await testProviderConnection\(/,
  );
});

test("cloud presets keep their identity across a save and a sync", async () => {
  const providers = await vite.ssrLoadModule(
    "/src/features/chat/external-providers.ts",
  );
  const sync = await vite.ssrLoadModule(
    "/src/features/chat/sync-external-providers.ts",
  );
  assert.equal(providers.toExternalBackendProviderType("azure"), "azure");
  assert.equal(providers.toExternalBackendProviderType("bedrock"), "bedrock");
  // Vertex is the native Gemini API at another host, so the backend row is a Gemini one.
  assert.equal(providers.toExternalBackendProviderType("vertex"), "gemini");
  const uiType = (type: string, baseUrl: string) =>
    sync.resolveUiProviderTypeFromConfig(type, "name", baseUrl, [], undefined);
  assert.equal(
    uiType("gemini", "https://aiplatform.googleapis.com/v1/publishers/google"),
    "vertex",
  );
  assert.equal(
    uiType("gemini", "https://us-central1-aiplatform.googleapis.com/v1/x"),
    "vertex",
  );
  assert.equal(
    uiType("gemini", "https://generativelanguage.googleapis.com/v1beta"),
    "gemini",
  );
  // Vertex's OpenAI-compatible endpoint stays a Gemini row on chat completions.
  assert.equal(
    uiType(
      "gemini",
      "https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi",
    ),
    "gemini",
  );
  assert.equal(
    uiType("azure", "https://res.openai.azure.com/openai/v1"),
    "azure",
  );
  // SGLang is saved as a plain custom server (chat completions), and reads back as SGLang.
  assert.equal(providers.toExternalBackendProviderType("sglang"), "custom");
  assert.equal(
    sync.resolveUiProviderTypeFromConfig("custom", "SGLang", "http://localhost:30000/v1", [], undefined),
    "sglang",
  );
  for (const type of ["azure", "bedrock", "vertex"]) {
    assert.equal(providers.isCustomProviderType(type), true);
    assert.equal(providers.isCloudPresetProviderType(type), true);
    assert.equal(providers.supportsRemoteModelCatalog(type), false);
    assert.equal(providers.connectionCheckNeedsModel(type), true);
  }
  assert.equal(providers.isCloudPresetProviderType("ollama"), false);
  assert.equal(providers.customProviderDefaultBaseUrl("azure"), "");
  assert.match(
    providers.customProviderDefaultBaseUrl("bedrock"),
    /bedrock-runtime\.us-east-1/,
  );
});
