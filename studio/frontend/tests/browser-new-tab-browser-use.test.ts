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

test("the switch sits under the leading section, outside both show settings", () => {
  assert.match(
    PAGE,
    /\{visited \? recentsSection : suggestedSection\}\s*<BrowserUseToggle \/>\s*\{visited \? suggestedSection : null\}/,
  );
  assert.match(PAGE, /const visited = recentsSection !== null;/);
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
