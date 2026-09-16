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

const targets = new Set<string>();
for (const file of tourStepFiles) {
  for (const target of matchAll(file.text, /target: "([\w-]+)"/g)) {
    targets.add(target);
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

// The other direction: a dropped step must not leave a stale anchor that reads like a live one.
test("every anchor is used by a step", () => {
  for (const anchor of anchors) {
    assert.ok(
      targets.has(anchor),
      `"${anchor}" is marked as a tour anchor but no step targets it; drop the attribute or the step`,
    );
  }
});

// A prop-carried anchor fails silently if the component stops forwarding it: the source still
// spells the anchor out, so the test above stays green.
test("a prop-carried anchor is forwarded to the DOM", () => {
  const all = files.map((file) => file.text).join("\n");
  for (const prop of ["dataTour", "triggerDataTour", "contentDataTour"]) {
    if (!new RegExp(`${prop}="[\\w-]+"`).test(all)) continue;
    assert.match(
      all,
      // Rendered as the attribute itself, or handed to another prop that is.
      new RegExp(`(?:data-tour|dataTour)=\\{${prop}\\}`),
      `${prop} is used as a tour anchor but nothing forwards it to data-tour`,
    );
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

// Mounted by RootLayout, not by their route, so they outlive a navigation away.
const PERSISTENT_PAGES = [
  "features/chat/chat-page.tsx",
  "features/images/images-page.tsx",
  "features/video/video-page.tsx",
  "features/audio/audio-page.tsx",
];

test("a persistently mounted page hides its tour when its route goes", () => {
  for (const page of PERSISTENT_PAGES) {
    assert.match(
      readSrc(page),
      /\{active && <GuidedTour /,
      `${page} renders GuidedTour unconditionally; it portals to body, so the tour would stay modal over the next page`,
    );
  }
});

test("a step is left when the tour unmounts, not only when it advances", () => {
  assert.match(
    readSrc("features/tour/components/guided-tour.tsx"),
    /onEnter\?\.\(\);[\s\S]{0,240}?return \(\) => \{[\s\S]{0,160}?onExit\?\.\(\);/,
    "onEnter must be undone by the effect's cleanup; a separate close-only effect never runs when a page unmounts the tour mid-step, so onEnter's side effects leak onto the next page",
  );
});

test("the menu offers a tour only when one is listening", () => {
  assert.match(
    readSrc("components/app-sidebar.tsx"),
    /useTourAvailable\(/,
    "the Guided Tour entry must consult useTourAvailable; route alone offers it on pages that mount their tour behind a capability gate, where clicking it does nothing",
  );
  assert.match(
    readSrc("features/tour/hooks/use-guided-tour-controller.ts"),
    /registerTour\(id\)/,
    "the controller must register itself while it listens, or useTourAvailable cannot answer",
  );
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
