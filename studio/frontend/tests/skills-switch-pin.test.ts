// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";

// The module opens a channel to other tabs when it can; an open one keeps the test process alive.
(globalThis as { BroadcastChannel?: unknown }).BroadcastChannel = undefined;

type Skill = {
  name: string;
  source: string;
  valid: boolean;
  shadowed: boolean;
  enabled: boolean;
};
type Api = {
  listSkills: (force?: boolean) => Promise<readonly Skill[]>;
  setSkillEnabled: (name: string, enabled: boolean) => Promise<Skill>;
  setAllSkillsEnabled: (enabled: boolean | null) => Promise<readonly Skill[]>;
};

const skill = (name: string, enabled: boolean): Skill => ({
  name,
  source: "user",
  valid: true,
  shadowed: false,
  enabled,
});

function harness(initial: Skill[]) {
  const sw = { on: null as boolean | null };
  let server = initial;
  const respond = (body: unknown) => ({
    ok: true,
    status: 200,
    json: async () => body,
  });
  const api = loadWithStubs<Api>(
    new URL("../src/features/chat/api/skills-api.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: async (
          url: string,
          init?: { method?: string; body?: string },
        ) => {
          const body = init?.body ? JSON.parse(init.body) : null;
          if (init?.method === "PUT" && url === "/api/skills") {
            // null restores defaults: user skills on.
            server = server.map((s) => ({
              ...s,
              enabled: body.enabled ?? true,
            }));
            return respond(server);
          }
          if (init?.method === "PUT") {
            const name = decodeURIComponent(url.split("/")[3]);
            server = server.map((s) =>
              s.name === name ? { ...s, enabled: body.enabled } : s,
            );
            return respond(server.find((s) => s.name === name));
          }
          return respond(server);
        },
      },
      "@/features/auth/session": { AUTH_SESSION_CLEARED_EVENT: "cleared" },
      react: { useEffect: () => undefined, useSyncExternalStore: () => null },
      "../stores/skills-switch-store": {
        useSkillsSwitchStore: {
          getState: () => ({
            on: sw.on,
            setOn: (on: boolean | null) => {
              sw.on = on;
            },
          }),
        },
      },
      "../utils/refresh-context-usage": {
        refreshContextUsage: async () => undefined,
      },
      "./skill-tools": { SKILL_MENTION_PATTERN: /@x/g },
    },
  );
  return { api, sw };
}

test("unticking the only ticked skill on an unset switch keeps Skills on", async () => {
  const { api, sw } = harness([skill("a", true), skill("b", false)]);
  await api.listSkills(true);
  await api.setSkillEnabled("a", false);
  assert.equal(sw.on, true);
});

test("Reset to defaults returns an unset switch to unset, so the restored ticks show again", async () => {
  const { api, sw } = harness([skill("a", false), skill("b", false)]);
  await api.listSkills(true);
  await api.setAllSkillsEnabled(null);
  assert.equal(sw.on, null);
});

test("Disable all keeps Skills on; Enable all turns it on", async () => {
  const { api, sw } = harness([skill("a", true)]);
  await api.listSkills(true);
  await api.setAllSkillsEnabled(false);
  assert.equal(sw.on, true);
  sw.on = false;
  await api.setAllSkillsEnabled(true);
  assert.equal(sw.on, true);
});
