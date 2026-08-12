// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  PASTED_TEXT_MIN_CHARS,
  PASTED_TEXT_MIN_LINES,
  PASTED_TEXT_PREVIEW_MAX_CHARS,
  attachmentContentText,
  createPastedTextFile,
  isPastedTextContent,
  isPastedTextFile,
  pasteLongTextAsFile,
  pastedTextFileName,
  pastedTextPreview,
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
  // A single line of megabytes is read through a window, not copied whole.
  assert.equal(
    pastedTextFileName("b".repeat(4 * 1024 * 1024)),
    `${"b".repeat(32)}.txt`,
  );
  assert.equal(
    pastedTextFileName(`${" ".repeat(300)}${"b".repeat(1000)}`),
    "Pasted text.txt",
  );
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

test("the sent wrapper is what marks a paste after the File is gone", () => {
  const pasted = attachmentContentText("Deploy log.txt", "body", true);
  const attached = attachmentContentText("notes.txt", "body", false);

  assert.equal(
    pasted,
    "<pasted_text name=Deploy log.txt>\nbody\n</pasted_text>",
  );
  assert.equal(attached, "<attachment name=notes.txt>\nbody\n</attachment>");
  // This is the marker a reloaded message still has, so it decides the chip.
  assert.equal(isPastedTextContent(pasted), true);
  assert.equal(isPastedTextContent(attached), false);
  assert.equal(isPastedTextContent(undefined), false);
  assert.equal(isPastedTextContent("pasted_text elsewhere in the body"), false);
  // Both wrappers unwrap, and a mismatched pair is left alone.
  assert.equal(unwrapAttachmentText(pasted), "body");
  assert.equal(unwrapAttachmentText(attached), "body");
  assert.equal(
    unwrapAttachmentText("<attachment name=x.txt>\nbody\n</pasted_text>"),
    "<attachment name=x.txt>\nbody\n</pasted_text>",
  );
});

test("the preview is capped, and says how much it is holding back", () => {
  const short = pastedTextPreview("all of it");
  assert.equal(short.text, "all of it");
  assert.equal(short.remaining, 0);

  const long = pastedTextPreview(
    "a".repeat(PASTED_TEXT_PREVIEW_MAX_CHARS + 25),
  );
  assert.equal(long.text.length, PASTED_TEXT_PREVIEW_MAX_CHARS);
  assert.equal(long.remaining, 25);
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
  // Whitespace is no exemption either: it used to skip both thresholds.
  assert.equal(shouldAttachPastedText(" ".repeat(PASTED_TEXT_MIN_CHARS)), true);
  assert.equal(
    shouldAttachPastedText("\n".repeat(PASTED_TEXT_MIN_LINES)),
    true,
  );
  assert.equal(shouldAttachPastedText("   "), false);
});

test("an attachment that throws on the spot reports instead of vanishing", () => {
  let errors = 0;
  const event = pasteEvent(clipboard("a".repeat(PASTED_TEXT_MIN_CHARS)));

  const handled = pasteLongTextAsFile(
    event,
    () => {
      throw new Error("no room for another attachment");
    },
    () => {
      errors += 1;
    },
  );

  // The paste is already swallowed at this point, so the toast is the only
  // thing standing between the user and silently losing the clipboard.
  assert.equal(handled, true);
  assert.equal(event.defaultPrevented, true);
  assert.equal(errors, 1);
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

test("previews leave unwrapped text alone", () => {
  const text = "line one\nline two";
  assert.equal(unwrapAttachmentText(text), text);
  assert.equal(unwrapAttachmentText(""), "");
});
