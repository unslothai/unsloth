// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const LOCALE_DIR = new URL("../src/i18n/locales/", import.meta.url);

test("every locale names the API page and the API Monitor button", async () => {
  const files = readdirSync(LOCALE_DIR).filter((f) => f.endsWith(".ts"));
  assert.ok(files.length >= 25);
  for (const file of files) {
    const mod = await import(new URL(file, LOCALE_DIR).href);
    const catalog = Object.values(mod)[0] as {
      shell?: { navigation?: { api?: string; apiMonitor?: string } };
    };
    assert.equal(catalog.shell?.navigation?.api, "API", file);
    assert.ok(catalog.shell?.navigation?.apiMonitor, `${file}: missing apiMonitor`);
  }
});

test("the profile menu and Settings > API both open the API page", () => {
  assert.match(
    readSrc("components/app-sidebar.tsx"),
    /navigateFromRow\(\{ to: "\/api-monitor" \}\);[\s\S]{0,400}?t\("shell\.navigation\.api"\)/,
  );
  assert.match(
    readSrc("features/settings/tabs/api-keys-tab.tsx"),
    /navigate\(\{ to: "\/api-monitor" \}\)[\s\S]{0,300}?t\("shell\.navigation\.apiMonitor"\)/,
  );
});
