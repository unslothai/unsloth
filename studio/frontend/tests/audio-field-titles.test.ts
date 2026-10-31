// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { readSrc } from "./helpers/kit.ts";

const AUDIO_DIR = fileURLToPath(
  new URL("../src/features/audio/", import.meta.url),
);

function tsxFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return tsxFiles(path);
    return path.endsWith(".tsx") ? [path] : [];
  });
}

test("audio field titles go through FieldLabel, so clicking one does not focus or toggle its control", () => {
  const field = readSrc("features/audio/components/field.tsx");
  assert.match(
    field,
    /if \(!\(event\.target as Element\)\.closest\(CONTROL\)\) event\.preventDefault\(\);/,
  );
  assert.match(field, /"cursor-default!"/);
  for (const file of tsxFiles(AUDIO_DIR)) {
    // Windows paths use backslashes.
    if (file.replaceAll("\\", "/").endsWith("components/field.tsx")) continue;
    assert.doesNotMatch(readFileSync(file, "utf8"), /<label\b/, file);
  }
});

test("audio Optional and Required tags are grey, smaller and unbracketed", () => {
  const field = readSrc("features/audio/components/field.tsx");
  assert.match(
    field,
    /<span className="ml-1\.5 text-ui-11p5 font-normal text-muted-foreground\/70">/,
  );
  for (const file of tsxFiles(AUDIO_DIR)) {
    assert.doesNotMatch(
      readFileSync(file, "utf8"),
      /\((optional|Optional)\)/,
      file,
    );
  }
  const music = readSrc("features/audio/pages/music-page.tsx");
  assert.match(music, /optional=\{rule\.description !== "required"\}/);
  assert.match(music, /optional=\{rule\.lyrics !== "required"\}/);
});
