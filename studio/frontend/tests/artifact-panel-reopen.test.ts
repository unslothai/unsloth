// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { useChatArtifactsStore } from "../src/features/chat/artifacts/store.ts";

const read = (path: string) =>
  readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf8");

const artifact = {
  id: "a1",
  title: "Board",
  code: "<p>hi</p>",
} as Parameters<ReturnType<typeof useChatArtifactsStore.getState>["openArtifact"]>[0];

test("every open bumps the sequence the panel reopens on, even for the artifact on screen", () => {
  const store = useChatArtifactsStore.getState();
  const before = store.openSequence;
  store.openArtifact(artifact, { surface: "panel", view: "preview" });
  store.openArtifact(artifact, { surface: "panel", view: "preview" });
  assert.equal(useChatArtifactsStore.getState().openSequence, before + 2);
});

test("switching tabs in the surface updates the view the card compares against", () => {
  const store = useChatArtifactsStore.getState();
  store.openArtifact(artifact, { surface: "panel", view: "preview" });
  store.setArtifactView("source");
  assert.equal(useChatArtifactsStore.getState().requestedView, "source");
});

test("a panel dragged shut is reported closed, not left selected at no width", () => {
  const page = read("../src/features/chat/chat-page.tsx");
  const remember = page.indexOf("const rememberArtifactPanelWidth");
  assert.ok(remember > 0);
  const closeAt = page.indexOf("onCloseArtifact();", remember);
  const widthAt = page.indexOf("artifactPanelWidthRef.current = `", remember);
  assert.ok(closeAt > 0 && closeAt < widthAt, "a shut panel still records a width");
});

test("the card hides the panel it is already showing and says so on aria-expanded", () => {
  const card = read("../src/features/chat/artifacts/artifact-card.tsx");
  assert.match(card, /if \(showing\) \{\s*closeArtifactSurface\(\);/);
  assert.match(card, /aria-expanded=\{showing\}/);
});

test("the panel's resize handle opts out of the double-click reset", () => {
  const page = read("../src/features/chat/chat-page.tsx");
  const handle = page.indexOf("<ResizableHandle", page.indexOf("const rememberArtifactPanelWidth"));
  assert.ok(handle > 0);
  assert.match(page.slice(handle, handle + 300), /disableDoubleClick/);
});
