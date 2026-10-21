// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The app is "Unsloth" or "Unsloth Studio", never a bare "Studio", in every
// language. Third-party names such as "LM Studio" stay as they are.

import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const BARE_STUDIO = /(?<!Unsloth[ -])(?<!LM )(?<!Recipe )Studio/;

function* strings(value: unknown, path: string): Generator<[string, string]> {
  if (typeof value === "string") yield [path, value];
  else if (value && typeof value === "object") {
    for (const [key, child] of Object.entries(value)) {
      yield* strings(child, `${path}.${key}`);
    }
  }
}

const localeDir = new URL("../src/i18n/locales/", import.meta.url);
const files = readdirSync(localeDir).filter((f) => f.endsWith(".ts"));

test("no locale calls the app a bare Studio", async () => {
  assert.ok(files.length >= 14);
  const offenders: string[] = [];
  for (const file of files) {
    const mod = await import(new URL(file, localeDir).href);
    for (const catalog of Object.values(mod)) {
      for (const [path, text] of strings(catalog, file)) {
        if (BARE_STUDIO.test(text)) offenders.push(`${path}: ${text}`);
      }
    }
  }
  assert.deepEqual(offenders, []);
});
