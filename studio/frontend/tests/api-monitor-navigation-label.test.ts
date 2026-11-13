// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const LOCALE_DIR = new URL("../src/i18n/locales/", import.meta.url);

test("the API Monitor page and API settings have different labels in every locale", async () => {
  const files = readdirSync(LOCALE_DIR).filter((f) => f.endsWith(".ts"));
  assert.ok(files.length >= 25);
  for (const file of files) {
    const mod = await import(new URL(file, LOCALE_DIR).href);
    const catalog = Object.values(mod)[0] as {
      shell?: { navigation?: { api?: string } };
      settings?: { tabs?: { apiKeys?: string } };
    };
    const monitor = catalog.shell?.navigation?.api;
    const settings = catalog.settings?.tabs?.apiKeys;
    assert.ok(monitor && settings, `${file}: missing label`);
    assert.notEqual(monitor, settings, `${file}: both read "${monitor}"`);
  }
});

test("routes to the monitor use the navigation label, the settings entry the settings label", () => {
  const sidebar = readSrc("components/app-sidebar.tsx");
  assert.match(
    sidebar,
    /label: t\("shell\.navigation\.api"\),\s*active: pathname === "\/api-monitor"/,
  );
  assert.match(
    sidebar,
    /openDialog\("api-keys"\)\}\s*>[\s\S]{0,200}?t\("settings\.tabs\.apiKeys"\)/,
  );
  assert.match(
    readSrc("components/command-palette.tsx"),
    /to: "\/api-monitor"[\s\S]{0,200}?t\("shell\.navigation\.api"\)/,
  );
  assert.match(
    readSrc("features/settings/components/sidebar-menu-customizer.tsx"),
    /api: \{ icon: ApiIcon, labelKey: "settings\.tabs\.apiKeys" \}/,
  );
});
