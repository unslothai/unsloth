// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Any site can send a signed-in tab to /audio?model=<repo> (or /images, /video). A model that arrived
// with the document itself waits for a click; an in-app pick still loads at once.
import assert from "node:assert/strict";
import test from "node:test";
import { register } from "node:module";

register("./helpers/toast-resolver.mjs", import.meta.url);

const { calls } = await import("./helpers/toast-stub.mjs");
const arrival = await import("../src/lib/linked-model-arrival.ts");
const { readSrc } = await import("./helpers/kit.ts");

const globals = globalThis as Record<string, unknown>;

function openedAt(documentUrl: string, currentUrl = documentUrl) {
  arrival.resetLinkedModelArrivalForTests();
  globals.window = { location: new URL(currentUrl) };
  globals.performance = { getEntriesByType: () => [{ name: documentUrl }] };
}

test("a model in the address the tab was opened at counts as linked from outside, once", () => {
  openedAt("http://127.0.0.1:8888/audio?model=mallory%2Ftts-demo");
  assert.equal(arrival.modelLinkedFromOutside("mallory/tts-demo"), true);
  // The pick the user confirms afterwards is an in-app navigation.
  assert.equal(arrival.modelLinkedFromOutside("mallory/tts-demo"), false);
});

test("a pick handed over inside the app loads as before", () => {
  openedAt("http://127.0.0.1:8888/chat", "http://127.0.0.1:8888/audio?model=unsloth%2Fcsm-1b");
  assert.equal(arrival.modelLinkedFromOutside("unsloth/csm-1b"), false);
  openedAt("http://127.0.0.1:8888/images");
  assert.equal(arrival.modelLinkedFromOutside("black-forest-labs/FLUX.1-dev"), false);
  // A Hub page opened with ?model= hands the same model to the media page with Run.
  openedAt("http://127.0.0.1:8888/hub?model=unsloth%2Fcsm-1b", "http://127.0.0.1:8888/audio?model=unsloth%2Fcsm-1b");
  assert.equal(arrival.modelLinkedFromOutside("unsloth/csm-1b"), false);
});

test("the confirm toast starts the pick only from its action", () => {
  calls.length = 0;
  let started = 0;
  arrival.confirmLinkedModel("mallory/tts-demo", () => started++);
  assert.equal(started, 0);
  const shown = calls.at(-1) as { options: { action: { onClick: () => void } } };
  shown.options.action.onClick();
  assert.equal(started, 1);
});

test("Audio, Images and Video ask before a linked model downloads or loads", () => {
  for (const file of [
    "features/audio/hooks/use-audio-handoff.ts",
    "features/images/images-page.tsx",
    "features/video/video-page.tsx",
  ]) {
    const source = readSrc(file);
    const guard = source.indexOf("modelLinkedFromOutside(wanted)");
    assert.ok(guard > 0, file);
    for (const load of ["handleModelSelect(wanted", "loadOrStage(", "loadGgufRepoPick(wanted"]) {
      const at = source.indexOf(load, source.indexOf("const wanted = routeSearch"));
      if (at > 0) assert.ok(guard < at, `${file}: ${load} before the linked-model check`);
    }
  }
});

test("a confirmed linked model replaces the stripped entry instead of stacking one", () => {
  // The page first replaces the linked URL with one that loads nothing; the confirmed pick puts the link
  // back in that same entry, so Back does not land on the stripped copy of the page.
  for (const file of [
    "features/audio/hooks/use-audio-handoff.ts",
    "features/images/images-page.tsx",
    "features/video/video-page.tsx",
  ]) {
    const source = readSrc(file);
    const confirm = source.match(/confirmLinkedModel\(wanted, \(\) => void navigateSelf\(\{[^}]*\}\)\)/);
    assert.ok(confirm, file);
    assert.match(confirm[0], /replace: true/, file);
  }
});
