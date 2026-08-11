// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  PASTED_TEXT_MIN_CHARS,
  PASTED_TEXT_MIN_LINES,
  createPastedTextFile,
  isPastedTextFile,
  pasteLongTextAsFile,
  pastedTextFileName,
  shouldAttachPastedText,
  unwrapAttachmentText,
} from "../src/features/chat/utils/pasted-text.ts";

type ClipboardStub = {
  readonly files: readonly File[];
  readonly items: readonly { kind: string }[];
  getData: (type: string) => string;
};

function clipboard(text: string, files: readonly File[] = []): ClipboardStub {
  return {
    files,
    items: files.map(() => ({ kind: "file" })),
    getData: (type) => (type === "text/plain" ? text : ""),
  };
}

function pasteEvent(clipboardData: ClipboardStub | null) {
  let defaultPrevented = false;
  return {
    clipboardData: clipboardData as unknown as DataTransfer | null,
    get defaultPrevented() {
      return defaultPrevented;
    },
    preventDefault: () => {
      defaultPrevented = true;
    },
  };
}

test("short pastes stay inline", () => {
  assert.equal(shouldAttachPastedText(""), false);
  assert.equal(shouldAttachPastedText("   \n  "), false);
  assert.equal(shouldAttachPastedText("what does this function do?"), false);
  assert.equal(
    shouldAttachPastedText("a".repeat(PASTED_TEXT_MIN_CHARS - 1)),
    false,
  );
  assert.equal(
    shouldAttachPastedText("line\n".repeat(PASTED_TEXT_MIN_LINES - 2)),
    false,
  );
});

test("bulk pastes become attachments by length or by line count", () => {
  assert.equal(shouldAttachPastedText("a".repeat(PASTED_TEXT_MIN_CHARS)), true);
  // Many short lines (a log tail, a stack trace) are just as unusable inline.
  assert.equal(
    shouldAttachPastedText("x\n".repeat(PASTED_TEXT_MIN_LINES)),
    true,
  );
});

test("pasted text files are named and recognised", () => {
  const name = pastedTextFileName(1_786_400_000_000);
  assert.equal(name, "Pasted_Text_1786400000.txt");

  const file = createPastedTextFile(
    "a".repeat(PASTED_TEXT_MIN_CHARS),
    1_786_400_000_000,
  );
  assert.equal(file.name, name);
  assert.equal(file.type, "text/plain");
  assert.equal(file.size, PASTED_TEXT_MIN_CHARS);
  assert.equal(isPastedTextFile(file), true);
  // A .txt the user actually attached keeps the normal file tile.
  assert.equal(
    isPastedTextFile(new File(["hi"], "notes.txt", { type: "text/plain" })),
    false,
  );
  assert.equal(isPastedTextFile(undefined, "Pasted_Text_1786400000.txt"), true);
});

test("a long text paste is swallowed and handed over as a file", async () => {
  const text = "a".repeat(PASTED_TEXT_MIN_CHARS);
  const added: File[] = [];
  const event = pasteEvent(clipboard(text));

  assert.equal(
    pasteLongTextAsFile(event, (file) => {
      added.push(file);
    }),
    true,
  );
  assert.equal(event.defaultPrevented, true);
  assert.equal(added.length, 1);
  assert.equal(await added[0]?.text(), text);
});

test("normal pastes and file pastes fall through untouched", () => {
  const shortPaste = pasteEvent(clipboard("hello"));
  assert.equal(
    pasteLongTextAsFile(shortPaste, () => {}),
    false,
  );
  assert.equal(shortPaste.defaultPrevented, false);

  // An image on the clipboard belongs to the file-paste path, even when the
  // app also offers a long text/plain rendering of it.
  const withFile = pasteEvent(
    clipboard("a".repeat(PASTED_TEXT_MIN_CHARS), [
      new File([new Uint8Array([1, 2, 3])], "shot.png", { type: "image/png" }),
    ]),
  );
  assert.equal(
    pasteLongTextAsFile(withFile, () => {}),
    false,
  );
  assert.equal(withFile.defaultPrevented, false);

  assert.equal(
    pasteLongTextAsFile(pasteEvent(null), () => {}),
    false,
  );
});

test("previews strip the wrapper the adapter sends to the model", () => {
  const text = "line one\nline two";
  assert.equal(
    unwrapAttachmentText(
      `<attachment name=Pasted_Text_1.txt>\n${text}\n</attachment>`,
    ),
    text,
  );
  assert.equal(unwrapAttachmentText(text), text);
});
