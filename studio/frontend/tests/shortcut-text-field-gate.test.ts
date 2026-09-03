// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { COMPOSER_INPUT_SELECTOR, isTextEntryFocused } = await import(
  "../src/features/settings/hooks/use-shortcut.ts"
);

/** Put an element in the focus the gate reads, with no DOM to do it for us. */
function focus(el: { tagName?: string; isContentEditable?: boolean } | null) {
  (globalThis as { document?: unknown }).document = {
    activeElement: el
      ? { matches: (selector: string) => selector === el.tagName, ...el }
      : null,
  };
}

/** A focused field that answers `matches` for exactly these selectors. */
function field(tagName: string, ...selectors: string[]) {
  return {
    tagName,
    matches: (selector: string) => selectors.includes(selector),
  };
}

test("the gate holds for every text entry when no exception is named", () => {
  for (const tagName of ["INPUT", "TEXTAREA"]) {
    focus(field(tagName));
    assert.equal(isTextEntryFocused(), true);
  }
  focus({ tagName: "DIV", isContentEditable: true });
  assert.equal(isTextEntryFocused(), true);

  // Everything else is not typing, so the gate has nothing to hold back.
  focus(field("BUTTON"));
  assert.equal(isTextEntryFocused(), false);
  focus(null);
  assert.equal(isTextEntryFocused(), false);
});

test("the composer exception frees the composer and nothing else", () => {
  // The state a tool request arrives in: the prompt that caused it left focus
  // in the composer, and Escape types nothing there.
  focus(field("TEXTAREA", COMPOSER_INPUT_SELECTOR));
  assert.equal(isTextEntryFocused(), true, "the plain gate still holds it");
  assert.equal(isTextEntryFocused(COMPOSER_INPUT_SELECTOR), false);

  // Every other field keeps its own Escape: the queued-prompt editor, the
  // settings search, a rename pill.
  focus(field("TEXTAREA"));
  assert.equal(isTextEntryFocused(COMPOSER_INPUT_SELECTOR), true);
  focus(field("INPUT"));
  assert.equal(isTextEntryFocused(COMPOSER_INPUT_SELECTOR), true);
  focus({ tagName: "DIV", isContentEditable: true });
  assert.equal(isTextEntryFocused(COMPOSER_INPUT_SELECTOR), true);
});

test("the decline chord takes the exception and the approve chord does not", async () => {
  const controls = await readFile(
    new URL(
      "../src/components/assistant-ui/tool-confirmation-controls.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // Escape leaves the text alone. Enter sends, so it stays behind the gate.
  assert.match(
    controls,
    /useShortcut\(\n\s*"declineToolRequest",[\s\S]{0,800}?textFieldException: COMPOSER_INPUT_SELECTOR,/,
  );
  const approve = controls.slice(
    controls.indexOf('"approveToolRequest",'),
    controls.indexOf('"declineToolRequest",'),
  );
  assert.ok(approve.includes("skipInTextFields: true"));
  assert.ok(!approve.includes("textFieldException"));
});

test("the selector matches the class the composer actually carries", async () => {
  const thread = await readFile(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  assert.equal(COMPOSER_INPUT_SELECTOR, ".aui-composer-input");
  // A selector naming a class nothing carries would silently gate everything.
  assert.match(thread, /className="aui-composer-input /);
});
