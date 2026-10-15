// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings > About lists the installed audio.cpp release beside the llama.cpp build.

import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import test from "node:test";

const SRC = new URL("../src/", import.meta.url);
const read = (relative: string) => readFileSync(new URL(relative, SRC), "utf8");

test("the hardware details carry the audio.cpp version to the About tab", () => {
  assert.match(
    read("hooks/use-hardware-info.ts"),
    /audioCpp: data\?\.audio_cpp \?\? null,/,
  );
  assert.match(
    read("features/settings/tabs/about-tab.tsx"),
    /llamaCppVersion=\{hw\.llamaCpp\}\s*audioCppVersion=\{hw\.audioCpp\}/,
  );
});

test("the audio.cpp row shows only when a version is known", () => {
  assert.match(
    read("features/settings/components/studio-version-section.tsx"),
    /\{audioCppVersion \? \(\s*<SettingsRow label=\{t\("settings\.about\.audioCppVersion"\)\}>\s*<code className="font-mono text-xs text-muted-foreground">\s*\{audioCppVersion\}/,
  );
});

test("every locale labels the audio.cpp row", () => {
  const locales = readdirSync(new URL("i18n/locales/", SRC)).filter((name) =>
    name.endsWith(".ts"),
  );
  assert.ok(locales.length > 1);
  for (const name of locales) {
    assert.match(
      read(`i18n/locales/${name}`),
      /audioCppVersion: "[^"\n]*audio\.cpp[^"\n]*",/,
      name,
    );
  }
});
