// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings lists llama.cpp, whisper.cpp, audio.cpp and MLX in Engines, and mlx-lm / mlx-vlm in Environment.

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

test("the Engines section shows MLX only where it is installed", () => {
  assert.match(
    read("hooks/use-hardware-info.ts"),
    /mlx: data\?\.versions\?\.mlx \?\? null,/,
  );
  const section = read(
    "features/settings/components/engine-versions-section.tsx",
  );
  assert.match(section, /key: "mlx",[^}]*installed: hw\.mlx,\s*offer: null,/);
  assert.match(section, /row\.key !== "mlx" \|\| row\.installed != null/);
});

test("the Environment section lists mlx-lm and mlx-vlm only where installed", () => {
  const env = read("features/settings/tabs/resources-tab.tsx");
  for (const [pkg, key] of [
    ["mlx_lm", "mlxLm"],
    ["mlx_vlm", "mlxVlm"],
  ]) {
    assert.match(
      env,
      new RegExp(
        `\\{systemInfo\\.ml_packages\\.${pkg} \\? \\(\\s*<InfoRow\\s*` +
          `label=\\{t\\("settings\\.resources\\.environment\\.${key}"\\)\\}\\s*` +
          `value=\\{hostReading\\(systemInfo\\.ml_packages\\.${pkg}\\)\\}`,
      ),
      pkg,
    );
  }
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

test("every locale labels the audio.cpp, MLX, mlx-lm and mlx-vlm rows", () => {
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
    assert.match(
      read(`i18n/locales/${name}`),
      /mlxVersion: "[^"\n]*MLX[^"\n]*",/,
      name,
    );
    assert.match(
      read(`i18n/locales/${name}`),
      /mlxLm: "mlx-lm",\s*mlxVlm: "mlx-vlm",/,
      name,
    );
  }
});
