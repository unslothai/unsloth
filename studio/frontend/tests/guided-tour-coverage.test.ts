// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

// A tour breaks silently: a renamed anchor leaves the step spotlighting nothing. These read the
// shipped source so a rename has to update both ends.

const SRC_ROOT = new URL("../src/", import.meta.url);

/** Every .ts/.tsx under src. URLs throughout: a file: URL pathname is "/D:/..." on Windows. */
function sourceFiles(dir: URL): Array<{ path: string; text: string }> {
  const out: Array<{ path: string; text: string }> = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      out.push(...sourceFiles(new URL(`${entry.name}/`, dir)));
      continue;
    }
    if (!entry.name.endsWith(".ts") && !entry.name.endsWith(".tsx")) continue;
    const file = new URL(entry.name, dir);
    out.push({
      path: `src/${file.href.slice(SRC_ROOT.href.length)}`,
      text: readFileSync(file, "utf8"),
    });
  }
  return out;
}

const files = sourceFiles(SRC_ROOT);
const tourStepFiles = files.filter((file) => file.path.includes("/tour/"));

function matchAll(text: string, pattern: RegExp): string[] {
  return [...text.matchAll(pattern)].map((match) => match[1] as string);
}

/** Anchors reach the DOM either as a literal attribute or through a component's dataTour prop. */
const anchors = new Set<string>();
for (const file of files) {
  for (const value of matchAll(file.text, /data-tour="([\w-]+)"/g)) {
    anchors.add(value);
  }
  for (const value of matchAll(
    file.text,
    /(?:dataTour|triggerDataTour|contentDataTour)="([\w-]+)"/g,
  )) {
    anchors.add(value);
  }
}

test("every tour step points at an anchor that exists", () => {
  for (const file of tourStepFiles) {
    for (const target of matchAll(file.text, /target: "([\w-]+)"/g)) {
      assert.ok(
        anchors.has(target),
        `${file.path} targets "${target}", which no data-tour attribute or dataTour prop provides`,
      );
    }
  }
});

test("every routed tour is mounted by its page", () => {
  const routes = readSrc("features/tour/lib/tour-routes.ts");
  const routedIds = matchAll(routes, /id: "([\w-]+)"/g);
  assert.ok(routedIds.length > 0);

  const mountedIds = new Set<string>();
  for (const file of files) {
    for (const call of file.text.matchAll(
      /useGuidedTourController\(\{[\s\S]{0,240}?\}\)/g,
    )) {
      const id = /id: "([\w-]+)"/.exec(call[0])?.[1];
      if (id) mountedIds.add(id);
    }
  }

  for (const id of routedIds) {
    assert.ok(
      mountedIds.has(id),
      `the "${id}" tour is offered in the user menu but no page passes that id to useGuidedTourController`,
    );
  }
});

const EM_DASH = "\u2014";

test("tour copy avoids em dashes", () => {
  for (const file of tourStepFiles) {
    assert.ok(
      !file.text.includes(EM_DASH),
      `${file.path} contains an em dash; tour copy uses plain sentences`,
    );
  }
});
