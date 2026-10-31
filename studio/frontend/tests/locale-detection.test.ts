// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();
Object.assign(globalThis, { document: { documentElement: { lang: "" } } });
const navigatorState = { language: "en-US", languages: ["en-US"] };
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: navigatorState,
});

const { initializeLocale } = await import("../src/i18n/locale-store.ts");

function detect(...languages: string[]): unknown {
  navigatorState.languages = languages;
  return initializeLocale({ loadMessages: () => undefined });
}

test("Traditional Chinese tags pick zh-TW, Simplified ones zh-CN", () => {
  for (const tag of ["zh-TW", "zh-HK", "zh-MO", "zh-Hant", "zh-Hant-HK"]) {
    assert.equal(detect(tag), "zh-TW", tag);
  }
  for (const tag of ["zh", "zh-CN", "zh-SG", "zh-Hans", "zh-Hans-HK"]) {
    assert.equal(detect(tag), "zh-CN", tag);
  }
  assert.equal(detect("zh-TW", "zh"), "zh-TW");
});

test("Latin American Spanish picks es-419, other Spanish es", () => {
  for (const tag of [
    "es-419",
    "es-MX",
    "es-AR",
    "es-CO",
    "es-CL",
    "es-PE",
    "es-US",
    "es-PR",
    "es-Latn-UY",
  ]) {
    assert.equal(detect(tag), "es-419", tag);
  }
  for (const tag of ["es", "es-ES", "es-GQ"])
    assert.equal(detect(tag), "es", tag);
});

test("other new languages match by their base subtag", () => {
  const cases: Record<string, string> = {
    "id-ID": "id",
    "vi-VN": "vi",
    "tr-TR": "tr",
    "pl-PL": "pl",
    "uk-UA": "uk",
    "th-TH": "th",
    "fa-IR": "fa",
    "cs-CZ": "cs",
    "nl-BE": "nl",
  };
  for (const [tag, locale] of Object.entries(cases)) {
    assert.equal(detect(tag), locale, tag);
  }
  assert.equal(detect("xx-YY", "de-AT"), "de");
  assert.equal(detect("xx-YY"), "en");
});
