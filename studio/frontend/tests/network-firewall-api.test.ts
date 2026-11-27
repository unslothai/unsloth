// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import * as formatFastApiError from "../src/lib/format-fastapi-error.ts";
import * as routeAbsent from "../src/features/settings/api/settings-route-absent.ts";
import type * as NetworkFirewallApi from "../src/features/settings/api/network-firewall.ts";
import {
  parseNetworkRequest,
  userNamedHost,
} from "../src/features/chat/utils/network-request.ts";

function client(response: () => Response) {
  const requests: { path: string; init?: RequestInit }[] = [];
  const api = loadWithStubs<typeof NetworkFirewallApi>(
    new URL("../src/features/settings/api/network-firewall.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: async (path: string, init?: RequestInit) => {
          requests.push({ path, init });
          return response();
        },
      },
      "@/lib/format-fastapi-error": formatFastApiError,
      "./settings-route-absent": routeAbsent,
    },
  );
  return { api, requests };
}

const stored = {
  mode: "ask",
  allowed_sites: ["pypi.org"],
  blocked_sites: ["example.com"],
  level: "locked",
};
const lists = {
  mode: "ask" as const,
  allowedSites: ["PyPI.org"],
  blockedSites: ["Example.com"],
};

test("one website per line or comma, blanks and padding dropped", () => {
  const { api } = client(() => new Response("{}"));
  assert.deepEqual(api.parseSites("  example.com \r\n\n a.org, b.org\n"), [
    "example.com",
    "a.org",
    "b.org",
  ]);
  assert.deepEqual(api.parseSites("\n , \n"), []);
});

test("saves the mode and every list and returns what the backend stored", async () => {
  const { api, requests } = client(() => new Response(JSON.stringify(stored), { status: 200 }));
  assert.deepEqual(await api.updateNetworkFirewall(lists, "fallback"), {
    mode: "ask",
    allowedSites: ["pypi.org"],
    blockedSites: ["example.com"],
    level: "locked",
  });
  assert.equal(requests[0].path, "/api/settings/network-firewall");
  assert.equal(requests[0].init?.method, "PUT");
  assert.deepEqual(JSON.parse(String(requests[0].init?.body)), {
    mode: "ask",
    allowed_sites: ["PyPI.org"],
    blocked_sites: ["Example.com"],
  });
});

test("a mode or level this build does not know reads as the weaker one", async () => {
  const { api } = client(
    () => new Response(JSON.stringify({ ...stored, mode: "strict", level: "sealed" }), { status: 200 }),
  );
  const read = await api.loadNetworkFirewall("fallback");
  assert.deepEqual([read.mode, read.level], ["open", "watched"]);
});

test("one website moves onto a list from the recent connections", async () => {
  const { api, requests } = client(() => new Response(JSON.stringify(stored), { status: 200 }));
  await api.addNetworkSite("blocked", "example.org", "fallback");
  assert.equal(requests[0].path, "/api/settings/network-firewall/site");
  assert.deepEqual(JSON.parse(String(requests[0].init?.body)), {
    kind: "blocked",
    host: "example.org",
  });
});

test("a refusal surfaces the backend's reason", async () => {
  const { api } = client(
    () =>
      new Response(JSON.stringify({ detail: "'not a site' is not a website. Use a name like example.com." }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }),
  );
  await assert.rejects(
    api.updateNetworkFirewall({ ...lists, blockedSites: ["not a site"] }, "fallback"),
    /not a website/,
  );
});

test("an older backend without the route reads as absent", async () => {
  const { api } = client(() => new Response("", { status: 404 }));
  await assert.rejects(api.loadNetworkFirewall("fallback"), {
    name: "SettingsRouteAbsentError",
  });
  assert.deepEqual(await api.loadNetworkConnections(), []);
});

test("Save stays disabled until the stored settings have loaded", () => {
  const section = readFileSync(
    new URL("../src/features/settings/components/network-firewall-section.tsx", import.meta.url),
    "utf8",
  );
  // A failed load leaves `saved` null; an enabled Save would send empty lists over the stored ones.
  assert.match(section, /const unsaveable =\s*saved === null \|\|/);
  assert.match(section, /disabled=\{saving \|\| unsaveable\}/);
});

test("a network request event becomes a question, a clear, or nothing", () => {
  assert.deepEqual(
    parseNetworkRequest({
      id: "r1",
      session_id: "s1",
      kind: "upload",
      host: "example.org",
      port: 8443,
      tool: "terminal",
      sent: 1200000,
      can_always: false,
    }),
    {
      id: "r1",
      sessionId: "s1",
      kind: "upload",
      host: "example.org",
      port: 8443,
      tool: "terminal",
      sent: 1200000,
      canAlways: false,
    },
  );
  const website = parseNetworkRequest({ id: "r1", host: "example.org", can_always: true });
  assert.deepEqual(
    [website?.kind, website?.tool, website?.port, website?.canAlways],
    ["website", "python", 443, true],
  );
  assert.equal(parseNetworkRequest({ id: "r1", host: "example.org", tool: "web" })?.tool, "web");
  assert.equal(parseNetworkRequest({ id: "r1", done: true }), null);
  assert.equal(parseNetworkRequest({ host: "example.org" }), undefined);
  assert.equal(parseNetworkRequest({ id: "r1" }), undefined);
  assert.equal(parseNetworkRequest("r1"), undefined);
});

test("a website the user wrote out needs no question, a lookalike still does", () => {
  for (const text of [
    "fetch https://www.iana.org/ for me",
    "get IANA.org.",
    "open (iana.org)",
    "iana.org",
  ]) {
    assert.equal(userNamedHost("www.iana.org", text), true, text);
    assert.equal(userNamedHost("iana.org", text), true, text);
  }
  for (const text of [
    "tell me about iana",
    "fetch https://iana.org.evil.test/",
    "fetch https://notiana.org/",
    "fetch https://data.iana.org/",
    "",
  ]) {
    assert.equal(userNamedHost("iana.org", text), false, text);
  }
});
