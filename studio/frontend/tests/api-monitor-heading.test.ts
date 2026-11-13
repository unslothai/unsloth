// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("the API tab's monitor card gets its own heading row", () => {
  const src = readSrc("features/settings/components/monitor-link.tsx");
  assert.match(
    src,
    /\{hideHeading \? \(\s*<SettingsRow\s*label=\{t\("settings\.resources\.liveMonitor\.apiTitle"\)\}\s*description=\{t\("settings\.resources\.liveMonitor\.apiDescription"\)\}/,
  );
  // The heading row sits above the card's button.
  assert.ok(src.indexOf("liveMonitor.apiDescription") < src.indexOf('void navigate({ to: "/api-monitor" })'));
  assert.match(readSrc("features/settings/tabs/api-keys-tab.tsx"), /<MonitorLink hideHeading=\{true\} \/>/);
});

test("every locale has the heading's description", () => {
  for (const locale of ["ar", "de", "en", "es", "fr", "he", "hi", "it", "ja", "ko", "pt-br", "ru", "sv", "zh-CN"]) {
    assert.match(readSrc(`i18n/locales/${locale}.ts`), /\n\s+apiTitle: "[^"]+",\n\s+apiDescription: "[^"]+",/, locale);
  }
});
