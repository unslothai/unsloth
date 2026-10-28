// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  isStrayKeyText,
  watchMacStrayKeyText,
} from "../src/lib/mac-stray-key-text.ts";

const MAIN = await readFile(
  new URL("../src/main.tsx", import.meta.url),
  "utf8",
);

test("the control characters macOS types for unhandled keys are stray", () => {
  // Left, right, Home, End (seen on macOS 15 and 26), then up, down and DEL.
  for (const data of [
    "\u001c",
    "\u001d",
    "\u0001",
    "\u0004",
    "\u001e",
    "\u001f",
    "\u007f",
  ]) {
    assert.equal(
      isStrayKeyText("insertText", data),
      true,
      JSON.stringify(data),
    );
  }
  assert.equal(isStrayKeyText("insertText", "\u001c\u001c"), true);
});

test("typed text, tabs and line breaks are kept", () => {
  for (const data of [
    "a",
    " ",
    "é",
    "\uf8ff",
    "\t",
    "\n",
    "\r",
    "a\u001c",
    "日本",
  ]) {
    assert.equal(
      isStrayKeyText("insertText", data),
      false,
      JSON.stringify(data),
    );
  }
  assert.equal(isStrayKeyText("insertText", ""), false);
  assert.equal(isStrayKeyText("insertText", null), false);
});

test("only plain typing is filtered, never paste, drop or deletion", () => {
  for (const inputType of [
    "insertFromPaste",
    "insertFromDrop",
    "insertReplacementText",
    "insertCompositionText",
    "deleteContentBackward",
  ]) {
    assert.equal(isStrayKeyText(inputType, "\u001c"), false, inputType);
  }
});

test("the guard cancels a stray insertion before any other listener sees it", () => {
  type Registration = [string, (event: InputEvent) => void, unknown];
  const registered: Registration[] = [];
  const win = {
    addEventListener: (...args: Registration) => {
      registered.push(args);
    },
  } as unknown as Window;
  watchMacStrayKeyText(win);
  assert.equal(registered.length, 1);
  const [type, listener, capture] = registered[0];
  assert.equal(type, "beforeinput");
  assert.equal(capture, true);

  const send = (inputType: string, data: string | null) => {
    let prevented = false;
    listener({
      inputType,
      data,
      preventDefault: () => {
        prevented = true;
      },
    } as unknown as InputEvent);
    return prevented;
  };
  assert.equal(send("insertText", "\u001c"), true);
  assert.equal(send("insertText", "x"), false);
  assert.equal(send("insertFromPaste", "\u001c"), false);
});

test("the guard is installed only in the macOS desktop app", () => {
  assert.match(
    MAIN,
    /if \(isTauri && \/Mac\/\.test\(navigator\.platform\)\) \{\s*watchMacStrayKeyText\(window\);/,
  );
});
