// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings > About lists the installed llama.cpp, whisper.cpp and audio.cpp releases in its Engines section.

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
    /<EngineVersionsSection \/>/,
  );
});

test("the Engines section lists llama.cpp, whisper.cpp and audio.cpp", () => {
  const section = read(
    "features/settings/components/engine-versions-section.tsx",
  );
  assert.match(
    section,
    /installed: hw\.llamaCpp \?\? status\?\.llama\.installed_tag \?\? null,/,
  );
  assert.match(
    section,
    /installed: status\?\.whisper\?\.installed_tag \?\? null,/,
  );
  assert.match(
    section,
    /installed: hw\.audioCpp \?\? status\?\.audio\?\.installed_tag \?\? null,/,
  );
  assert.match(
    section,
    /row\.installed \?\? t\("settings\.about\.engines\.notInstalled"\)/,
  );
});

test("only the owner gets the engine Update button, which the update route requires", () => {
  const section = read(
    "features/settings/components/engine-versions-section.tsx",
  );
  assert.match(section, /const isOwner = useIsAccountOwner\(\);/);
  assert.match(section, /\{isOwner \? \(\s*<Button[^]*?data-testid="engine-update"/);
});

test("a self-managed llama.cpp still lets Update install an offered audio.cpp release", () => {
  const section = read(
    "features/settings/components/engine-versions-section.tsx",
  );
  assert.match(
    section,
    /const unmanaged = status\?\.supported === false && !offered;/,
  );
  assert.match(section, /disabled=\{!offered \|\| applying \|\| checking\}/);
  assert.doesNotMatch(section, /!supported/);
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
