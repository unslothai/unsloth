// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import type * as ClaudeDesktopApi from "../src/features/settings/api/claude-desktop.ts";
import type * as CodexDesktopApi from "../src/features/settings/api/codex-desktop.ts";
import type { CodexDesktopStatus } from "../src/features/settings/api/codex-desktop.ts";
import { claudeDesktopView } from "../src/features/settings/components/claude-desktop-state.ts";
import {
  codexPicks,
  codexTarget,
  sameModels,
  toggleModel,
} from "../src/features/settings/components/codex-desktop-state.ts";
import {
  CLAUDE_ROUTES,
  cleanMappings,
  FOLLOW_LOADED,
  mappingsEqual,
  pinnedModels,
  sharedMappingModel,
  updateAllMappings,
  updateMapping,
} from "../src/features/settings/components/desktop-app-models.ts";
import { en } from "../src/i18n/locales/en.ts";
import * as formatFastApiError from "../src/lib/format-fastapi-error.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const CODEX: CodexDesktopStatus = {
  supported: true,
  installed: true,
  app_path: "/Applications/Codex.app",
  download_url: "https://openai.com/codex/",
  connected: false,
  held_by: null,
  held_by_pid: null,
  running: false,
  model: null,
  models: [],
  loaded_model: "unsloth/Qwen3.6-27B-GGUF",
  selected_models: [],
  max_models: 5,
  local: true,
};

function stubbed<T>(file: string, response: () => Response) {
  const requests: { path: string; init?: RequestInit }[] = [];
  const api = loadWithStubs<T>(new URL(file, import.meta.url), {
    "@/features/auth": {
      authFetch: async (path: string, init?: RequestInit) => {
        requests.push({ path, init });
        return response();
      },
    },
    "@/lib/format-fastapi-error": formatFastApiError,
  });
  return { api, requests };
}

const sent = (requests: { path: string; init?: RequestInit }[]) =>
  requests.map((r) => [r.path, r.init?.method ?? "GET", r.init?.body ?? null]);

test("the All picker sets every Claude slot and reads Mixed when they differ (fork parity)", () => {
  const all = updateAllMappings("unsloth/gemma-4-E2B-it-GGUF");
  assert.deepEqual(
    Object.keys(all),
    CLAUDE_ROUTES.map((r) => r.id),
  );
  assert.equal(sharedMappingModel(all), "unsloth/gemma-4-E2B-it-GGUF");
  const mixed = updateMapping(all, "claude-haiku-4-5-20251001", "small/model");
  assert.equal(sharedMappingModel(mixed), null);
  // One empty slot is Mixed too: "All" only shows a model every slot shares.
  assert.equal(sharedMappingModel({ "claude-fable-5": "a" }), null);
  assert.deepEqual(
    CLAUDE_ROUTES.map((r) => r.name),
    ["Fable 5", "Opus 5", "Sonnet 5", "Haiku 4.5", "Sonnet 4.6"],
  );
});

test("mappings are sparse, ordered and compared without empty slots", () => {
  assert.deepEqual(
    cleanMappings({
      "claude-sonnet-4-6": " b ",
      "claude-fable-5": "a",
      "claude-opus-5": "",
      "not-a-slot": "x",
    }),
    { "claude-fable-5": "a", "claude-sonnet-4-6": "b" },
  );
  assert.ok(
    mappingsEqual(
      { "claude-fable-5": "a", "claude-opus-5": "" },
      {
        "claude-fable-5": "a",
      },
    ),
  );
  assert.ok(
    !mappingsEqual({ "claude-fable-5": "a" }, { "claude-fable-5": "b" }),
  );
  assert.deepEqual(
    pinnedModels({
      "claude-fable-5": FOLLOW_LOADED,
      "claude-opus-5": "a",
      "claude-sonnet-5": "a",
      "claude-sonnet-4-6": "b",
    }),
    ["a", "b"],
  );
});

test("Claude model choices load Studio's catalog and save with the restart choice", async () => {
  const c = stubbed<typeof ClaudeDesktopApi>(
    "../src/features/settings/api/claude-desktop.ts",
    () => Response.json([]),
  );
  await c.api.fetchClaudeDesktopModels();
  await c.api.setClaudeDesktopMappings({ "claude-fable-5": "a" });
  await c.api.setClaudeDesktopMappings({ "claude-fable-5": "a" }, true);
  assert.deepEqual(sent(c.requests), [
    ["/api/settings/claude-desktop/models", "GET", null],
    [
      "/api/settings/claude-desktop/mappings",
      "PUT",
      '{"mappings":{"claude-fable-5":"a"},"restart":null}',
    ],
    [
      "/api/settings/claude-desktop/mappings",
      "PUT",
      '{"mappings":{"claude-fable-5":"a"},"restart":true}',
    ],
  ]);
});

test("Codex Desktop hits its own backend-owned routes", async () => {
  const c = stubbed<typeof CodexDesktopApi>(
    "../src/features/settings/api/codex-desktop.ts",
    () => Response.json({ status: "connected", codex: CODEX }),
  );
  await c.api.connectCodexDesktop();
  await c.api.connectCodexDesktop(true, ["m", "n:Q4_K_M"]);
  await c.api.setCodexDesktopModels([], false);
  await c.api.setCodexDesktopModels(["a", "b"]);
  await c.api.disconnectCodexDesktop();
  await c.api.disconnectCodexDesktop(true);
  await c.api.fetchCodexDesktopModels();
  assert.deepEqual(sent(c.requests), [
    [
      "/api/settings/codex-desktop/connect",
      "POST",
      '{"restart":null,"models":null}',
    ],
    [
      "/api/settings/codex-desktop/connect",
      "POST",
      '{"restart":true,"models":["m","n:Q4_K_M"]}',
    ],
    [
      "/api/settings/codex-desktop/models",
      "PUT",
      '{"models":[],"restart":false}',
    ],
    [
      "/api/settings/codex-desktop/models",
      "PUT",
      '{"models":["a","b"],"restart":null}',
    ],
    ["/api/settings/codex-desktop/disconnect", "POST", '{"restart":null}'],
    ["/api/settings/codex-desktop/disconnect", "POST", '{"restart":true}'],
    ["/api/settings/codex-desktop/models", "GET", null],
  ]);
});

test("an older backend hides the Codex card; errors carry the backend's message", async () => {
  for (const status of [403, 404]) {
    const c = stubbed<typeof CodexDesktopApi>(
      "../src/features/settings/api/codex-desktop.ts",
      () => new Response("{}", { status }),
    );
    await assert.rejects(
      c.api.fetchCodexDesktopStatus(),
      (err: unknown) => err instanceof c.api.CodexDesktopUnavailable,
    );
  }
  const busy = stubbed<typeof CodexDesktopApi>(
    "../src/features/settings/api/codex-desktop.ts",
    () =>
      Response.json(
        { detail: "The Codex app is in use by `unsloth start codex --app`." },
        { status: 409 },
      ),
  );
  await assert.rejects(busy.api.connectCodexDesktop(), /unsloth start codex/);
});

test("the Codex card shares the Claude card's states and follows the loaded model by default", () => {
  assert.deepEqual(claudeDesktopView(CODEX), { kind: "idle", model: null });
  assert.deepEqual(claudeDesktopView({ ...CODEX, installed: false }), {
    kind: "not_installed",
    downloadUrl: "https://openai.com/codex/",
  });
  assert.deepEqual(
    claudeDesktopView({ ...CODEX, held_by: "cli", held_by_pid: 7 }),
    {
      kind: "held",
      pid: 7,
    },
  );
  assert.deepEqual(codexPicks(CODEX), []);
  assert.equal(codexTarget(CODEX), "unsloth/Qwen3.6-27B-GGUF");
  const picked = { ...CODEX, selected_models: ["m", "n"] };
  assert.deepEqual(codexPicks(picked), ["m", "n"]);
  assert.equal(codexTarget(picked), "m");
  // Connected while following: the card keeps showing "follow loaded", not the pinned id.
  assert.deepEqual(
    codexPicks({ ...CODEX, connected: true, model: "x", models: ["x"] }),
    [],
  );
  // Connected: the list Codex actually has, even if the saved one moved on.
  assert.deepEqual(
    codexPicks({ ...picked, connected: true, model: "m", models: ["m"] }),
    ["m"],
  );
});

test("the Codex list adds, removes and stops at its limit, like Ollama's", () => {
  assert.deepEqual(toggleModel([], "a", 2), ["a"]);
  assert.deepEqual(toggleModel(["a"], "b", 2), ["a", "b"]);
  assert.deepEqual(toggleModel(["a", "b"], "c", 2), ["a", "b"]);
  assert.deepEqual(toggleModel(["a", "b"], "a", 2), ["b"]);
  assert.ok(sameModels(["a", "b"], ["a", "b"]));
  assert.ok(!sameModels(["a", "b"], ["b", "a"]));
  const picker = read(
    "../src/features/settings/components/desktop-app-model-list-picker.tsx",
  );
  assert.ok(picker.includes("aria-multiselectable={true}"));
  assert.ok(picker.includes("settings.agents.desktopApps.codex.removeModel"));
  const c = en.settings.agents.desktopApps.codex;
  assert.match(c.maxReached, /\{count\}/);
  assert.match(c.connectedMany, /\{count\}.*\{model\}/);
  assert.match(c.removeModel, /\{model\}/);
});

test("every locale has the Codex list strings", async () => {
  const keys = [
    "modelLabel",
    "modelHint",
    "modelBody",
    "connectedMany",
    "addModel",
    "removeModel",
    "maxReached",
  ] as const;
  for (const locale of [
    "ar",
    "de",
    "en",
    "es",
    "fr",
    "he",
    "hi",
    "it",
    "ja",
    "ko",
    "pt-br",
    "ru",
    "sv",
    "zh-CN",
  ]) {
    const mod = await import(`../src/i18n/locales/${locale}.ts`);
    const table = Object.values(mod)[0] as typeof en;
    const codex = table.settings.agents.desktopApps.codex;
    for (const key of keys) {
      assert.ok(codex[key], `${locale}: codex.${key}`);
    }
    assert.ok(table.settings.agents.desktopApps.models.notDownloaded, locale);
    assert.match(codex.maxReached, /\{count\}/, locale);
    assert.match(codex.connectedMany, /\{count\}/, locale);
    assert.match(codex.connectedMany, /\{model\}/, locale);
    assert.match(codex.removeModel, /\{model\}/, locale);
  }
});

const read = (path: string) =>
  readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf-8");

test("both cards ask before restarting the app, and the strings match the fork", () => {
  const codex = read(
    "../src/features/settings/components/codex-desktop-card.tsx",
  );
  assert.ok(codex.includes('data-testid="codex-desktop-card"'));
  assert.ok(codex.includes("void connect(true)"));
  assert.ok(codex.includes("void disconnect(true)"));
  assert.ok(codex.includes("void disconnect(false)"));
  const models = read(
    "../src/features/settings/components/claude-desktop-models.tsx",
  );
  assert.ok(models.includes("setAskRestart(true)"));
  const section = read(
    "../src/features/settings/components/claude-desktop-section.tsx",
  );
  assert.ok(section.includes("<CodexDesktopCard"));
  const m = en.settings.agents.desktopApps.models;
  assert.equal(m.allRouteName, "all model options");
  assert.equal(
    m.allHint,
    "Use one model for all options. You can still change each option below.",
  );
  assert.equal(m.mixed, "Mixed");
  assert.equal(m.select, "Select a model");
  assert.equal(m.find, "Find model...");
  assert.equal(
    m.restartConfirm,
    "Restart Claude Desktop? Any running task will stop.",
  );
  assert.equal(
    m.restartGuidance,
    "Restarting Claude will stop any running task.",
  );
  const codexStrings = en.settings.agents.desktopApps.codex;
  assert.match(codexStrings.connected, /\{model\}/);
  assert.match(codexStrings.heldByCli, /\{pid\}/);
});
