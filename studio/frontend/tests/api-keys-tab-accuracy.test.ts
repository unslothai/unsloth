// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { en } from "../src/i18n/locales/en.ts";
import { readSrc } from "./helpers/kit.ts";

const TAB = readSrc("features/settings/tabs/api-keys-tab.tsx");
const FORM = readSrc("features/settings/components/create-key-form.tsx");
const KEYS_API = readSrc("features/settings/api/api-keys.ts");
const ROW = readSrc("features/settings/components/api-key-row.tsx");
const EXAMPLES = readSrc("features/settings/components/usage-examples.tsx");
const AUDIO = readSrc("features/settings/components/audio-api-examples.tsx");
const MEDIA = readSrc("features/settings/components/media-api-examples.tsx");
const KEYLESS_API = readSrc("features/settings/api/keyless-api-access.ts");
const KEYLESS = readSrc(
  "features/settings/components/keyless-api-access-section.tsx",
);

test("a failed create or revoke keeps the token list and says why", () => {
  assert.match(TAB, /onError=\{setActionError\}/);
  assert.match(TAB, /\{actionError \? \([\s\S]*?\{error \? \(/);
  assert.match(
    KEYS_API,
    /typeof body\.detail === "string" \? body\.detail : ""/,
  );
  assert.equal(KEYS_API.match(/throw await actionError\(res\)/g)?.length, 2);
  assert.match(FORM, /onError\(null\);/);
});

test("token rows say Expired, count whole days and show minutes", () => {
  assert.match(ROW, /t\("settings\.apiKeys\.expiredStatus"\)/);
  assert.equal(en.settings.apiKeys.expiredStatus, "Expired");
  assert.match(ROW, /expired \? "bg-muted-foreground\/40" : "bg-emerald-500"/);
  assert.match(ROW, /Math\.ceil\(diff \/ 86400000\)/);
  assert.match(ROW, /formatRelativeTime\(locale, -minutes, "minute"\)/);
});

test("browser examples use the address the browser reached", () => {
  for (const source of [EXAMPLES, AUDIO, MEDIA]) {
    assert.match(
      source,
      /useTunnel && cloudflareUrl\s*\?\s*cloudflareUrl\s*:\s*isTauri\s*\?\s*\(serverUrl \?\? origin\)\s*:\s*origin/,
    );
  }
});

test("the raw-port warning is not shown for a loopback bind", () => {
  assert.match(EXAMPLES, /const loopbackBind = isLoopbackUrl\(serverUrl\);/);
  assert.match(EXAMPLES, /\{secure \|\| loopbackBind \? null : \(/);
});

test("keyless suspended by managed accounts reads as off for the examples", () => {
  assert.match(
    KEYLESS_API,
    /suspendedByAccounts: settings\.suspended_by_accounts === true/,
  );
  assert.match(
    KEYLESS,
    /scope: next\.suspendedByAccounts \? "off" : next\.scope/,
  );
  assert.match(KEYLESS, /Off while this install has other accounts/);
});
