// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { readSrc } from "./helpers/kit.ts";

const PAGE = readSrc("features/browser/new-tab-page.tsx");

test("the new tab page's Browser use switch drives the composer's store", () => {
  assert.match(PAGE, /const enabled = useBrowserAgentStore\(\(state\) => state\.enabled\);/);
  assert.match(PAGE, /onCheckedChange=\{\(on\) => useBrowserAgentStore\.getState\(\)\.setEnabled\(on\)\}/);
  // Same rule as the composer: off and disabled where the composer's item is.
  assert.match(PAGE, /const disabled = useBrowserAgentDisabled\(\);/);
  assert.match(PAGE, /checked=\{enabled && !disabled\}\s*disabled=\{disabled\}/);
  assert.match(
    readSrc("features/browser/agent/composer-toggle.tsx"),
    /export function useBrowserAgentDisabled\(\): boolean \{/,
  );
});

test("the switch sits just under Suggested, outside its show setting", () => {
  assert.match(
    PAGE,
    /<SuggestedSite key=\{site\.url\} site=\{site\} tabId=\{tabId\} \/>\s*\)\)\}\s*<\/div>\s*<\/section>\s*\) : null\}\s*<BrowserUseToggle \/>\s*\{showRecents/,
  );
});

test("every locale has the switch's strings", () => {
  for (const locale of ["ar", "de", "en", "es", "fr", "he", "hi", "it", "ja", "ko", "pt-br", "ru", "sv", "zh-CN"]) {
    assert.match(
      readSrc(`i18n/locales/${locale}.ts`),
      /\n {4}browserUse: \{\n {6}label: "[^"]+",\n {6}description: "[^"]+",\n {6}unavailable: "[^"]+",\n {4}\},/,
      locale,
    );
  }
});
