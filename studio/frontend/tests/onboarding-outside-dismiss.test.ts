// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  OUTSIDE_DISMISS_WINDOW_MS,
  outsideClickCloses,
} from "../src/features/onboarding/lib/pages.ts";

const dialog = readFileSync(
  new URL(
    "../src/features/onboarding/components/onboarding-dialog.tsx",
    import.meta.url,
  ),
  "utf8",
);

test("the first outside click only arms the dismissal", () => {
  assert.equal(outsideClickCloses(null, 1_000), false);
});

test("a second outside click within the window closes", () => {
  assert.equal(outsideClickCloses(1_000, 1_000 + 400), true);
  assert.equal(
    outsideClickCloses(1_000, 1_000 + OUTSIDE_DISMISS_WINDOW_MS),
    true,
  );
});

test("a second outside click after the window arms again instead", () => {
  assert.equal(
    outsideClickCloses(1_000, 1_000 + OUTSIDE_DISMISS_WINDOW_MS + 1),
    false,
  );
});

test("the dialog blocks the first outside pointerdown and disarms on inside clicks", () => {
  const handler = dialog.match(
    /onPointerDownOutside=\{\(e\) => \{([\s\S]*?)\n {8}\}\}/,
  );
  assert.ok(handler, "onPointerDownOutside handler not found");
  const body = handler[1];
  assert.match(body, /outsideClickCloses\(armedAt\.current, Date\.now\(\)\)/);
  assert.ok(
    body.indexOf("outsideClickCloses") < body.indexOf("e.preventDefault()"),
    "the armed check must run before the click is swallowed",
  );
  assert.match(body, /armOutsideDismiss\(\)/);
  assert.match(dialog, /onPointerDownCapture=\{disarm\}/);
  assert.match(dialog, /setTimeout\(disarm, OUTSIDE_DISMISS_WINDOW_MS\)/);
});

test("Escape and the close controls still dismiss in one step", () => {
  assert.match(dialog, /onOpenChange=\{\(open\) => !open && skipAll\(\)\}/);
  assert.doesNotMatch(
    dialog.match(/onEscapeKeyDown=\{[\s\S]*?\}\}/)?.[0] ?? "",
    /armedAt|outsideClickCloses/,
  );
  assert.match(dialog, /<DialogClose asChild=\{true\}>/);
  assert.match(dialog, /onClick=\{skipAll\}/);
});

test("the first outside click shows no hint text, only the nudge", () => {
  assert.doesNotMatch(dialog, /outsideAgain/);
  assert.match(dialog, /animate\(NUDGE/);
});

test("the model picker card is not drawn under the open onboarding dialog", () => {
  const card = readFileSync(
    new URL(
      "../src/components/assistant-ui/model-picker-announcement.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(card, /useOnboardingStore\(\(s\) => s\.pages !== null\)/);
  assert.match(card, /\{!dismissed && !onboarding && \(/);
});
