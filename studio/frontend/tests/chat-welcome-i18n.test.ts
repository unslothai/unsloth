// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The empty chat's greeting follows the UI language (#13391): every line is a
// welcome.* message key, not English baked into thread.tsx.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { en } = await import("../src/i18n/locales/en.ts");
const { de } = await import("../src/i18n/locales/de.ts");
const threadSrc = await readSrcAsync("components/assistant-ui/thread.tsx");

const welcomeSrc = threadSrc.slice(
  threadSrc.indexOf("type Welcome ="),
  threadSrc.indexOf("export const ProjectComposer"),
);

test("the greeting uses every welcome key and no English literals", () => {
  const used = new Set(
    [...welcomeSrc.matchAll(/"welcome\.(\w+)"/g)].map((m) => m[1]),
  );
  assert.deepEqual([...used].sort(), Object.keys(en.welcome).sort());
  for (const line of [
    "Good to see you",
    "Ready when you are",
    "Night owl mode?",
    "Temporary chat",
  ]) {
    assert.ok(!welcomeSrc.includes(`"${line}`), line);
  }
});

test("named lines keep {name} and translate", () => {
  for (const [key, text] of Object.entries(en.welcome)) {
    const german = de.welcome[key as keyof typeof de.welcome];
    assert.notEqual(german, text, key);
    assert.equal(key.endsWith("Name"), text.includes("{name}"), key);
    assert.equal(german.includes("{name}"), text.includes("{name}"), key);
  }
});
