// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { fallbackTitleFromUserText } from "../src/features/chat/utils/chat-title.ts";
import {
  PASTED_TEXT_MIN_CHARS,
  PASTED_TEXT_MIN_LINES,
  PASTED_TEXT_PREVIEW_MAX_CHARS,
  attachmentContentSample,
  attachmentContentText,
  attachmentsPastedText,
  attachmentsSample,
  createPastedTextFile,
  isPastedTextContent,
  isPastedTextFile,
  pasteLongTextAsFile,
  pastedTextContentBody,
  pastedTextContentBytes,
  pastedTextContentPreview,
  pastedTextFileName,
  pastedTextOf,
  pastedTextPreview,
  shouldAttachPastedText,
  unwrapPastedTextContent,
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
  // Blank lines are stepped over one at a time, so the walk is bounded too:
  // naming must not depend on how many of them there are.
  const started = process.hrtime.bigint();
  assert.equal(
    pastedTextFileName(`${"\n".repeat(4 * 1024 * 1024)}Deploy log`),
    "Pasted text.txt",
  );
  assert.ok(Number(process.hrtime.bigint() - started) / 1e6 < 250);
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

test("the pasted text is kept beside the file, not re-read from it", () => {
  // Draft autosave runs on a keystroke, so it cannot await File.text().
  const text = `Deploy log\n${"line\n".repeat(200)}`;
  assert.equal(pastedTextOf(createPastedTextFile(text)), text);
  assert.equal(
    pastedTextOf(new File(["hi"], "notes.txt", { type: "text/plain" })),
    undefined,
  );
  assert.equal(pastedTextOf(undefined), undefined);
});

test("the sent wrapper is what marks a paste after the File is gone", () => {
  const pasted = attachmentContentText("Deploy log.txt", "body", true, 4);
  const attached = attachmentContentText("notes.txt", "body", false, 4);

  assert.equal(
    pasted,
    "<pasted_text name=Deploy log.txt bytes=4>\nbody\n</pasted_text>",
  );
  // Only the paste is tagged and sized; a real attachment keeps its wrapper.
  assert.equal(attached, "<attachment name=notes.txt>\nbody\n</attachment>");
  assert.equal(isPastedTextContent(pasted), true);
  assert.equal(isPastedTextContent(attached), false);
  assert.equal(isPastedTextContent(undefined), false);
  assert.equal(isPastedTextContent("pasted_text elsewhere in the body"), false);
});

test("the chip size is read off the header, never measured", () => {
  const pasted = attachmentContentText("Deploy log.txt", "body", true, 12_483);
  assert.equal(pastedTextContentBytes(pasted), 12_483);
  // A name that itself looks like the size must not win over the real one.
  assert.equal(
    pastedTextContentBytes(
      attachmentContentText("x bytes=5.txt", "body", true, 99),
    ),
    99,
  );
  assert.equal(pastedTextContentBytes(undefined), undefined);
  assert.equal(
    pastedTextContentBytes(attachmentContentText("a.txt", "body", true)),
    undefined,
  );
  assert.equal(
    pastedTextContentBytes(attachmentContentText("a.txt", "body", false, 4)),
    undefined,
  );
});

test("previewing sent content never materialises the body", () => {
  const body = "line one\nline two";
  const pasted = attachmentContentText("a.txt", body, true, 17);
  assert.deepEqual(pastedTextContentPreview(pasted), {
    text: body,
    remaining: 0,
  });
  assert.deepEqual(
    pastedTextContentPreview(attachmentContentText("a.txt", body, false)),
    { text: body, remaining: 0 },
  );

  const huge = "z".repeat(PASTED_TEXT_PREVIEW_MAX_CHARS + 40);
  const cut = pastedTextContentPreview(
    attachmentContentText("a.txt", huge, true, huge.length),
  );
  assert.equal(cut.text.length, PASTED_TEXT_PREVIEW_MAX_CHARS);
  assert.equal(cut.remaining, 40);

  // Unwrapped or truncated content still previews rather than throwing.
  assert.deepEqual(pastedTextContentPreview("bare text"), {
    text: "bare text",
    remaining: 0,
  });
  assert.deepEqual(pastedTextContentPreview("<pasted_text name=a.txt>\nbody"), {
    text: "body",
    remaining: 0,
  });
});

test("a paste-only message still has something to name the thread with", () => {
  const paste = `Fix the retry backoff\n${"detail\n".repeat(500)}`;
  const sent = attachmentContentText(
    pastedTextFileName(paste),
    paste,
    true,
    paste.length,
  );

  // The composer sends no text at all in this case, so the title comes from
  // the attachment or the thread stays "New Chat".
  assert.equal(
    fallbackTitleFromUserText(attachmentsSample([{ content: [] }])),
    "New Chat",
  );
  assert.equal(
    fallbackTitleFromUserText(
      attachmentsSample([{ content: [{ type: "text", text: sent }] }]),
    ),
    "Fix the retry backoff",
  );
  // Only a bounded opening is read, never the whole body.
  assert.ok(
    attachmentsSample([{ content: [{ type: "text", text: sent }] }]).length <=
      512,
  );
  // Non-text parts are skipped, and the first text part wins.
  assert.equal(
    attachmentsSample([
      { content: [{ type: "image" }] },
      { content: [{ type: "text", text: sent }] },
    ]),
    attachmentContentSample(sent),
  );
  assert.equal(attachmentsSample(undefined), "");
});

test("chat search still finds what the paste moved out of the message", () => {
  const body = `Fix the retry backoff\n${"detail\n".repeat(500)}closing line`;
  const sent = attachmentContentText("Fix.txt", body, true, body.length);

  // The whole body, not the opening: a term late in a paste has to stay
  // findable or the paste is searchable only by its first line.
  assert.equal(pastedTextContentBody(sent), body);
  assert.equal(
    attachmentsPastedText([{ content: [{ type: "text", text: sent }] }]),
    body,
  );

  // Attachments that were never indexed stay that way.
  assert.equal(pastedTextContentBody("[PDF: paper.pdf]\nAbstract"), "");
  assert.equal(
    pastedTextContentBody(attachmentContentText("notes.txt", body, false)),
    "",
  );
  assert.equal(
    attachmentsPastedText([
      { content: [{ type: "text", text: "[PDF: paper.pdf]\nAbstract" }] },
      { content: [{ type: "image" }] },
    ]),
    "",
  );
  assert.equal(attachmentsPastedText(undefined), "");

  // Several pastes on one message all reach the index.
  const second = attachmentContentText("Log.txt", "second body", true, 11);
  assert.equal(
    attachmentsPastedText([
      { content: [{ type: "text", text: sent }] },
      { content: [{ type: "text", text: second }] },
    ]),
    `${body}\n\nsecond body`,
  );
});

test("copies and exports carry the paste, not its wrapper", () => {
  const body = "Deploy log\nline two";
  const sent = attachmentContentText("Deploy log.txt", body, true, 19);

  // The marker is an implementation detail; the same text pasted below the
  // threshold never had one.
  assert.equal(unwrapPastedTextContent(sent), body);
  assert.ok(!unwrapPastedTextContent(sent).includes("pasted_text"));

  // Everything else is handed back untouched, including the wrapper a real
  // attachment has always exported with.
  const attached = attachmentContentText("notes.txt", body, false);
  assert.equal(unwrapPastedTextContent(attached), attached);
  assert.equal(unwrapPastedTextContent("bare text"), "bare text");
  assert.equal(unwrapPastedTextContent(""), "");
});

test("a sample of unwrapped content is left as it is", () => {
  // PDF and DOCX adapters write their own prefix with no wrapper.
  assert.equal(
    attachmentContentSample("[PDF: paper.pdf]\nAbstract"),
    "[PDF: paper.pdf]\nAbstract",
  );
  assert.equal(attachmentContentSample("plain", 3), "pla");
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
