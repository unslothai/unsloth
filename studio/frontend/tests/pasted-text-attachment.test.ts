// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  PASTED_TEXT_MIN_CHARS,
  PASTED_TEXT_MIN_LINES,
  createPastedTextFile,
  isPastedTextAttachment,
  isPastedTextFile,
  pasteLongTextAsFile,
  pastedTextFileName,
  rememberPastedTextAttachment,
  shouldAttachPastedText,
  unwrapAttachmentText,
} from "../src/features/chat/utils/pasted-text.ts";

type ClipboardStub = {
  readonly files: readonly File[];
  readonly items: readonly { kind: string }[];
  readonly types: readonly string[];
  getData: (type: string) => string;
};

function clipboard(
  text: string,
  files: readonly File[] = [],
  extra: { types?: readonly string[]; data?: Record<string, string> } = {},
): ClipboardStub {
  return {
    files,
    items: files.map(() => ({ kind: "file" })),
    types: extra.types ?? ["text/plain"],
    getData: (type) =>
      type === "text/plain" ? text : (extra.data?.[type] ?? ""),
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

test("the file is named after the opening of the paste", () => {
  assert.equal(
    pastedTextFileName("Introducing Unsloth"),
    "Introducing Unsloth.txt",
  );
  // Leading blank lines are skipped.
  assert.equal(
    pastedTextFileName("\n\n  Release notes  \nbody"),
    "Release notes.txt",
  );
  // Long openings cut on a word boundary, short ones are not padded.
  assert.equal(
    pastedTextFileName(
      "Introducing Unsloth Studio, the fastest way to finetune",
    ),
    "Introducing Unsloth Studio, the.txt",
  );
  assert.equal(
    pastedTextFileName(`${"z".repeat(60)} tail`),
    `${"z".repeat(32)}.txt`,
  );
  // Path separators and control characters cannot reach the filename.
  assert.equal(
    pastedTextFileName("src/lib\\util:\tmain"),
    "src lib util main.txt",
  );
  assert.equal(pastedTextFileName("\n   \n"), "Pasted text.txt");
  assert.equal(pastedTextFileName("///"), "Pasted text.txt");
});

test("pasted text files are recognised by identity", () => {
  const file = createPastedTextFile(
    "Deploy log\n".repeat(PASTED_TEXT_MIN_LINES),
  );
  assert.equal(file.name, "Deploy log.txt");
  assert.equal(file.type, "text/plain");
  assert.equal(isPastedTextFile(file), true);
  // A .txt the user actually attached keeps the normal file tile.
  assert.equal(
    isPastedTextFile(
      new File(["hi"], "Deploy log.txt", { type: "text/plain" }),
    ),
    false,
  );
  assert.equal(isPastedTextFile(undefined), false);
});

test("the attachment id carries the chip through a send", () => {
  assert.equal(isPastedTextAttachment("attachment-1"), false);
  rememberPastedTextAttachment("attachment-1");
  assert.equal(isPastedTextAttachment("attachment-1"), true);
  // Re-registering the same id must not evict it.
  rememberPastedTextAttachment("attachment-1");
  for (let index = 0; index < 200; index += 1) {
    rememberPastedTextAttachment(`filler-${index}`);
  }
  assert.equal(isPastedTextAttachment("attachment-1"), false);
  assert.equal(isPastedTextAttachment("filler-199"), true);
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

test("a paste too big to hold inline still attaches", () => {
  // Anything the old cap rejected fell back to pasting inline, which is the
  // one case the input cannot survive.
  const huge = "a".repeat(20 * 1024 * 1024 + 1);
  assert.equal(shouldAttachPastedText(huge), true);
});

test("a .txt named like a pasted one keeps the normal tile", () => {
  const pasted = createPastedTextFile("Release notes\nbody");
  const lookalike = new File(["hi"], pasted.name, { type: "text/plain" });
  assert.equal(isPastedTextFile(lookalike), false);
});

test("native image and file payloads stay on the file paste path", () => {
  const text = "a".repeat(PASTED_TEXT_MIN_CHARS);
  // Tauri advertises native payloads by type only, with nothing in files.
  const svg = pasteEvent(
    clipboard(text, [], { types: ["text/plain", "image/svg+xml"] }),
  );
  assert.equal(
    pasteLongTextAsFile(svg, () => {}),
    false,
  );
  assert.equal(svg.defaultPrevented, false);

  const copiedFiles = pasteEvent(
    clipboard(text, [], {
      types: ["text/plain", "text/uri-list"],
      data: { "text/uri-list": "file:///tmp/notes.txt" },
    }),
  );
  assert.equal(
    pasteLongTextAsFile(copiedFiles, () => {}),
    false,
  );

  // A plain text/html copy is still text, not a file payload.
  const richText = pasteEvent(
    clipboard(text, [], { types: ["text/plain", "text/html"] }),
  );
  assert.equal(
    pasteLongTextAsFile(richText, () => {}),
    true,
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
