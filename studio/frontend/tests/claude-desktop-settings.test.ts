// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import type * as ClaudeDesktopApi from "../src/features/settings/api/claude-desktop.ts";
import type { ClaudeDesktopStatus } from "../src/features/settings/api/claude-desktop.ts";
import { claudeDesktopView } from "../src/features/settings/components/claude-desktop-state.ts";
import { en } from "../src/i18n/locales/en.ts";
import * as formatFastApiError from "../src/lib/format-fastapi-error.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const STATUS: ClaudeDesktopStatus = {
  supported: true,
  installed: true,
  app_path: "/Applications/Claude.app",
  download_url: "https://claude.ai/download",
  connected: false,
  held_by: null,
  held_by_pid: null,
  running: false,
  model: "unsloth/gemma-4-E2B-it-GGUF",
  mappings: { "claude-fable-5": ":loaded" },
  routes: [
    { id: "claude-fable-5", name: "Fable 5" },
    { id: "claude-opus-5", name: "Opus 5" },
    { id: "claude-sonnet-5", name: "Sonnet 5" },
    { id: "claude-haiku-4-5-20251001", name: "Haiku 4.5" },
    { id: "claude-sonnet-4-6", name: "Sonnet 4.6" },
  ],
  local: true,
};

function client(response: () => Response) {
  const requests: { path: string; init?: RequestInit }[] = [];
  const api = loadWithStubs<typeof ClaudeDesktopApi>(
    new URL("../src/features/settings/api/claude-desktop.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: async (path: string, init?: RequestInit) => {
          requests.push({ path, init });
          return response();
        },
      },
      "@/lib/format-fastapi-error": formatFastApiError,
    },
  );
  return { api, requests };
}

test("status, connect and disconnect hit the backend-owned Claude Desktop routes", async () => {
  const c = client(() => Response.json(STATUS));
  assert.deepEqual(await c.api.fetchClaudeDesktopStatus(), STATUS);

  const connect = client(() =>
    Response.json({ status: "needs_restart", claude: STATUS }),
  );
  assert.equal(
    (await connect.api.connectClaudeDesktop()).status,
    "needs_restart",
  );
  await connect.api.connectClaudeDesktop(true);
  await connect.api.connectClaudeDesktop(false);
  assert.deepEqual(
    connect.requests.map((r) => [r.path, r.init?.method, r.init?.body]),
    [
      ["/api/settings/claude-desktop/connect", "POST", '{"restart":null}'],
      ["/api/settings/claude-desktop/connect", "POST", '{"restart":true}'],
      ["/api/settings/claude-desktop/connect", "POST", '{"restart":false}'],
    ],
  );

  const disconnect = client(() =>
    Response.json({ notes: ["Quit Claude Desktop"], claude: STATUS }),
  );
  assert.deepEqual((await disconnect.api.disconnectClaudeDesktop()).notes, [
    "Quit Claude Desktop",
  ]);
  assert.equal(
    disconnect.requests[0].path,
    "/api/settings/claude-desktop/disconnect",
  );
  assert.equal(disconnect.requests[0].init?.method, "POST");
});

test("an older backend or a non-owner hides the card instead of erroring", async () => {
  for (const status of [403, 404]) {
    const c = client(() => new Response("{}", { status }));
    await assert.rejects(
      c.api.fetchClaudeDesktopStatus(),
      (err: unknown) => err instanceof c.api.ClaudeDesktopUnavailable,
    );
  }
});

test("connect errors carry the backend's message (not installed, macOS only)", async () => {
  const c = client(() =>
    Response.json(
      {
        detail:
          "Claude Desktop isn't installed. Install it from https://claude.ai/download.",
      },
      { status: 409 },
    ),
  );
  await assert.rejects(c.api.connectClaudeDesktop(), /claude\.ai\/download/);
});

test("the card state puts what blocks Connect first", () => {
  assert.deepEqual(claudeDesktopView(null), { kind: "loading" });
  assert.deepEqual(claudeDesktopView(STATUS), {
    kind: "idle",
    model: "unsloth/gemma-4-E2B-it-GGUF",
  });
  assert.deepEqual(
    claudeDesktopView({ ...STATUS, supported: false, installed: false }),
    {
      kind: "unsupported",
    },
  );
  assert.deepEqual(claudeDesktopView({ ...STATUS, installed: false }), {
    kind: "not_installed",
    downloadUrl: "https://claude.ai/download",
  });
  assert.deepEqual(claudeDesktopView({ ...STATUS, local: false }), {
    kind: "remote",
  });
  assert.deepEqual(
    claudeDesktopView({ ...STATUS, held_by: "cli", held_by_pid: 4242 }),
    { kind: "held", pid: 4242 },
  );
  assert.deepEqual(claudeDesktopView({ ...STATUS, connected: true }, true), {
    kind: "connected",
    model: "unsloth/gemma-4-E2B-it-GGUF",
    pendingRestart: true,
  });
});

const read = (path: string) =>
  readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf-8");

test("the Agents tab shows Desktop apps right under the intro, before the command builder", () => {
  const tab = read("../src/features/settings/tabs/agents-tab.tsx");
  const section = tab.indexOf("<ClaudeDesktopSection />");
  assert.ok(section > tab.indexOf('t("settings.agents.intro")'));
  assert.ok(section < tab.indexOf("settings.agents.generatedCommand"));
});

test("the restart prompt is a dialog with restart, later and cancel", () => {
  const card = read(
    "../src/features/settings/components/claude-desktop-section.tsx",
  );
  assert.ok(card.includes("<AlertDialog open={askRestart}"));
  assert.ok(card.includes("void connect(true)"));
  assert.ok(card.includes("void connect(false)"));
  assert.ok(card.includes('t("common.cancel")'));
  const strings = en.settings.agents.desktopApps;
  assert.equal(strings.restartTitle, "Restart Claude Desktop?");
  assert.match(strings.connected, /\{model\}/);
  assert.match(strings.heldByCli, /\{pid\}/);
});
