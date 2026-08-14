// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { strToU8, zipSync } from "fflate";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  attachmentAudioSrc,
  countAttachmentTextLines,
  getDocxAttachmentError,
  isAudioAttachment,
  parseAttachmentText,
  readAttachmentText,
  truncateAttachmentPreviewText,
} = await import("../src/features/chat/attachment-content.ts");

// The preview reads a sent attachment back out of the text the adapter built,
// so every wrapper the adapters write has to round-trip.
test("parseAttachmentText unwraps a labelled document header", () => {
  const parsed = parseAttachmentText("[PDF: report.pdf]\nline one\nline two");
  assert.deepEqual(parsed, {
    label: "PDF",
    text: "line one\nline two",
    truncated: false,
  });
});

test("parseAttachmentText unwraps the plain text attachment tag", () => {
  const parsed = parseAttachmentText(
    "<attachment name=notes.txt>\nline one\nline two\n</attachment>",
  );
  assert.deepEqual(parsed, {
    label: null,
    text: "line one\nline two",
    truncated: false,
  });
});

test("parseAttachmentText keeps text that carries no wrapper", () => {
  const parsed = parseAttachmentText("[not a label] still content");
  assert.deepEqual(parsed, {
    label: null,
    text: "[not a label] still content",
    truncated: false,
  });
});

test("parseAttachmentText keeps a header-like first line inside the body", () => {
  const parsed = parseAttachmentText("[PDF: a.pdf]\n[DOCX: b.docx]\nbody");
  assert.deepEqual(parsed, {
    label: "PDF",
    text: "[DOCX: b.docx]\nbody",
    truncated: false,
  });
});

test("truncateAttachmentPreviewText caps very long attachments", () => {
  const short = truncateAttachmentPreviewText("abc");
  assert.deepEqual(short, { text: "abc", truncated: false });

  const long = truncateAttachmentPreviewText("a".repeat(200_001));
  assert.equal(long.truncated, true);
  assert.equal(long.text.length, 200_000);
});

test("countAttachmentTextLines counts empty and single-line text", () => {
  assert.equal(countAttachmentTextLines(""), 0);
  assert.equal(countAttachmentTextLines("one line"), 1);
  assert.equal(countAttachmentTextLines("one\ntwo\n"), 3);
});

// The sent audio part only carries "mp3" or "wav", so an OGG or FLAC upload
// would be mislabelled without the attachment's own content type.
test("attachmentAudioSrc keeps the uploaded audio MIME", () => {
  const part = { data: "AAA", format: "wav" };
  assert.equal(
    attachmentAudioSrc(part, "audio/ogg", "clip.ogg"),
    "data:audio/ogg;base64,AAA",
  );
  assert.equal(
    attachmentAudioSrc({ data: "AAA", format: "mp3" }, undefined, "clip.mp3"),
    "data:audio/mpeg;base64,AAA",
  );
  assert.equal(
    attachmentAudioSrc(part, "", "clip.wav"),
    "data:audio/wav;base64,AAA",
  );
});

// An extension-only upload reaches the sent preview with an empty content type
// and format "wav", so the filename is what identifies the container.
test("attachmentAudioSrc falls back to the extension for untyped uploads", () => {
  const part = { data: "AAA", format: "wav" };
  assert.equal(
    attachmentAudioSrc(part, "", "clip.m4a"),
    "data:audio/mp4;base64,AAA",
  );
  assert.equal(
    attachmentAudioSrc(part, "application/octet-stream", "clip.flac"),
    "data:audio/flac;base64,AAA",
  );
  assert.equal(
    attachmentAudioSrc(part, undefined, "clip"),
    "data:audio/wav;base64,AAA",
  );
});

// The text and HTML adapters accept uploads with no size limit, so opening a
// preview must not materialize the whole file.
test("readAttachmentText reads a bounded slice of a large text file", async () => {
  const oversized = new File(["a".repeat(2_000_000)], "huge.txt", {
    type: "text/plain",
  });
  const { label, text, truncated } = await readAttachmentText(
    oversized,
    oversized.name,
    oversized.type,
  );
  assert.equal(label, null);
  assert.equal(truncated, true);
  assert.equal(text.length, 1_000_000);
  assert.equal(truncateAttachmentPreviewText(text).truncated, true);
});

test("readAttachmentText reads a bounded slice of a large html file", async () => {
  const parsed: number[] = [];
  const original = (globalThis as { DOMParser?: unknown }).DOMParser;
  (globalThis as { DOMParser?: unknown }).DOMParser = class {
    parseFromString(source: string) {
      parsed.push(source.length);
      return {
        querySelectorAll: () => [],
        body: { textContent: source },
      };
    }
  };

  try {
    const oversized = new File(
      [`<p>${"b".repeat(2_000_000)}</p>`],
      "huge.html",
      {
        type: "text/html",
      },
    );
    const { label, text, truncated } = await readAttachmentText(
      oversized,
      oversized.name,
      oversized.type,
    );
    assert.equal(label, "HTML");
    assert.equal(truncated, true);
    assert.deepEqual(parsed, [1_000_000]);
    assert.equal(text.length <= 1_000_000, true);
  } finally {
    (globalThis as { DOMParser?: unknown }).DOMParser = original;
  }
});

test("isAudioAttachment matches by MIME and by extension", () => {
  assert.equal(isAudioAttachment("clip.m4a", ""), true);
  assert.equal(isAudioAttachment("clip", "audio/webm"), true);
  assert.equal(isAudioAttachment("notes.txt", "text/plain"), false);
  assert.equal(isAudioAttachment(undefined, undefined), false);
});

// A bounded HTML read can extract almost nothing when the slice ends inside a
// script block, so the flag, not the text length, is what discloses the cut.
test("readAttachmentText reports truncation even when the slice extracts no text", async () => {
  const original = (globalThis as { DOMParser?: unknown }).DOMParser;
  (globalThis as { DOMParser?: unknown }).DOMParser = class {
    parseFromString() {
      return { querySelectorAll: () => [], body: { textContent: "" } };
    }
  };

  try {
    const oversized = new File(
      [`<script>${"c".repeat(2_000_000)}`],
      "big.html",
      {
        type: "text/html",
      },
    );
    const { text, truncated } = await readAttachmentText(
      oversized,
      oversized.name,
      oversized.type,
    );
    assert.equal(text, "");
    assert.equal(truncated, true);
    assert.equal(truncateAttachmentPreviewText(text).truncated, false);
  } finally {
    (globalThis as { DOMParser?: unknown }).DOMParser = original;
  }
});

// Stored payloads have no size limit, so unwrapping must copy at most the
// capped body rather than the whole attachment.
test("parseAttachmentText caps the body it copies out of a wrapper", () => {
  const body = "d".repeat(300_000);
  const tagged = parseAttachmentText(
    `<attachment name=huge.txt>\n${body}\n</attachment>`,
  );
  assert.equal(tagged.label, null);
  assert.equal(tagged.text.length, 200_000);
  assert.equal(tagged.truncated, true);

  const labelled = parseAttachmentText(`[PDF: huge.pdf]\n${body}`);
  assert.equal(labelled.label, "PDF");
  assert.equal(labelled.text.length, 200_000);
  assert.equal(labelled.truncated, true);

  const bare = parseAttachmentText(body);
  assert.equal(bare.text.length, 200_000);
  assert.equal(bare.truncated, true);
});

// A File the preview only ever asks for its size and its bytes, so the read can
// be observed without materializing a document-sized buffer.
function fakeDocumentFile(
  name: string,
  size: number,
  bytes: Uint8Array,
  reads: string[],
): File {
  return {
    name,
    size,
    arrayBuffer: () => {
      reads.push(name);
      return Promise.resolve(
        bytes.buffer.slice(
          bytes.byteOffset,
          bytes.byteOffset + bytes.byteLength,
        ) as ArrayBuffer,
      );
    },
  } as unknown as File;
}

function docxBytes(documentXml: string): Uint8Array {
  return zipSync({
    "[Content_Types].xml": strToU8("<Types/>"),
    "_rels/.rels": strToU8("<Relationships/>"),
    "word/document.xml": strToU8(documentXml),
  });
}

// unpdf and mammoth parse on the main thread, so an oversized document has to be
// refused before its bytes are read, not after.
test("readAttachmentText refuses an oversized pdf before reading it", async () => {
  const reads: string[] = [];
  const oversized = fakeDocumentFile(
    "huge.pdf",
    60 * 1024 * 1024,
    new Uint8Array(0),
    reads,
  );
  await assert.rejects(
    readAttachmentText(oversized, oversized.name, "application/pdf"),
    /PDF file is too large: huge\.pdf/,
  );
  assert.deepEqual(reads, []);
});

test("readAttachmentText refuses an oversized docx before reading it", async () => {
  const reads: string[] = [];
  const oversized = fakeDocumentFile(
    "huge.docx",
    60 * 1024 * 1024,
    new Uint8Array(0),
    reads,
  );
  await assert.rejects(
    readAttachmentText(oversized, oversized.name, undefined),
    /DOCX file is too large: huge\.docx/,
  );
  assert.deepEqual(reads, []);
});

// The bytes are requested synchronously, so the extractor is reached without
// waiting on unpdf, which the preview test does not exercise.
test("readAttachmentText reads a pdf under the ceiling", () => {
  const reads: string[] = [];
  const small = fakeDocumentFile(
    "small.pdf",
    64 * 1024,
    new Uint8Array([0x25, 0x50, 0x44, 0x46]),
    reads,
  );
  const pending = readAttachmentText(small, small.name, "application/pdf");
  pending.catch(() => undefined);
  assert.deepEqual(reads, ["small.pdf"]);
});

// mammoth's node build takes a buffer rather than an arrayBuffer, so the small
// case asserts the archive cleared both guards and reached mammoth itself.
test("readAttachmentText lets a normal docx through to the extractor", async () => {
  const reads: string[] = [];
  const bytes = docxBytes("<w:document><w:body/></w:document>");
  const small = fakeDocumentFile("notes.docx", bytes.length, bytes, reads);
  const error = await readAttachmentText(small, small.name, undefined).then(
    () => null,
    (thrown: Error) => thrown,
  );
  assert.deepEqual(reads, ["notes.docx"]);
  if (error) {
    assert.doesNotMatch(error.message, /too large/);
  }
});

// A DOCX is a zip, so a small upload can still declare a huge document.xml.
test("readAttachmentText refuses a docx that declares an oversized document.xml", async () => {
  const reads: string[] = [];
  const bytes = docxBytes("a".repeat(11 * 1024 * 1024));
  const bomb = fakeDocumentFile("bomb.docx", bytes.length, bytes, reads);
  assert.equal(bomb.size < 1024 * 1024, true);
  await assert.rejects(
    readAttachmentText(bomb, bomb.name, undefined),
    /DOCX XML file is too large: bomb\.docx:word\/document\.xml/,
  );
});

// mammoth reads "_rels/.rels" first and "[Content_Types].xml" next, and picks
// the body part out of "word/_rels/document.xml.rels", so a bomb parked in any
// of them never passes through word/*.xml.
test("readAttachmentText refuses an oversized docx part outside word/*.xml", async () => {
  const huge = "a".repeat(11 * 1024 * 1024);
  const parts = [
    "[Content_Types].xml",
    "_rels/.rels",
    "word/_rels/document.xml.rels",
  ];

  for (const part of parts) {
    const bytes = zipSync({
      "[Content_Types].xml": strToU8("<Types/>"),
      "_rels/.rels": strToU8("<Relationships/>"),
      "word/document.xml": strToU8("<w:document><w:body/></w:document>"),
      [part]: strToU8(huge),
    });
    const bomb = fakeDocumentFile("bomb.docx", bytes.length, bytes, []);
    assert.equal(bomb.size < 1024 * 1024, true);
    await assert.rejects(
      readAttachmentText(bomb, bomb.name, undefined),
      new RegExp(
        `DOCX XML file is too large: bomb\\.docx:${part.replace(
          /[.[\]/]/g,
          "\\$&",
        )}`,
      ),
      `a ${part} bomb reached mammoth`,
    );
  }
});

function relationships(entries: Array<[string, string]>): Uint8Array {
  return strToU8(
    `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">${entries
      .map(
        ([type, target], index) =>
          `<Relationship Id="rId${index + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/${type}" Target="${target}"/>`,
      )
      .join("")}</Relationships>`,
  );
}

// mammoth resolves the body and its styles/numbering/note parts through the
// relationships and parses whatever they point at as XML, so a target named
// "payload.bin" is inflated on the main thread even though no suffix says XML.
test("readAttachmentText refuses an oversized docx part reached through a relationship", async () => {
  const huge = strToU8("a".repeat(11 * 1024 * 1024));

  const bodyBomb = zipSync({
    "[Content_Types].xml": strToU8("<Types/>"),
    "_rels/.rels": relationships([["officeDocument", "payload.bin"]]),
    "payload.bin": huge,
  });
  const bodyFile = fakeDocumentFile("bomb.docx", bodyBomb.length, bodyBomb, []);
  assert.equal(bodyFile.size < 1024 * 1024, true);
  await assert.rejects(
    readAttachmentText(bodyFile, bodyFile.name, undefined),
    /DOCX XML file is too large: bomb\.docx:payload\.bin/,
  );

  const stylesBomb = zipSync({
    "[Content_Types].xml": strToU8("<Types/>"),
    "_rels/.rels": relationships([["officeDocument", "word/document.xml"]]),
    "word/document.xml": strToU8("<w:document><w:body/></w:document>"),
    "word/_rels/document.xml.rels": relationships([["styles", "styles.dat"]]),
    "word/styles.dat": huge,
  });
  const stylesFile = fakeDocumentFile(
    "styles.docx",
    stylesBomb.length,
    stylesBomb,
    [],
  );
  assert.equal(stylesFile.size < 1024 * 1024, true);
  await assert.rejects(
    readAttachmentText(stylesFile, stylesFile.name, undefined),
    /DOCX XML file is too large: styles\.docx:word\/styles\.dat/,
  );
});

// extractRawText never reads an image part, so a document that merely embeds a
// large picture still previews: the bound follows what mammoth parses.
test("readAttachmentText lets a docx with a large embedded image through", async () => {
  const reads: string[] = [];
  const bytes = zipSync({
    "[Content_Types].xml": strToU8("<Types/>"),
    "_rels/.rels": relationships([["officeDocument", "word/document.xml"]]),
    "word/document.xml": strToU8("<w:document><w:body/></w:document>"),
    "word/_rels/document.xml.rels": relationships([
      ["image", "media/photo.png"],
    ]),
    "word/media/photo.png": new Uint8Array(12 * 1024 * 1024),
  });
  const file = fakeDocumentFile("photo.docx", bytes.length, bytes, reads);
  const error = await readAttachmentText(file, file.name, undefined).then(
    () => null,
    (thrown: Error) => thrown,
  );
  assert.deepEqual(reads, ["photo.docx"]);
  if (error) {
    assert.doesNotMatch(error.message, /too large/);
  }
});

// mammoth hands the relationships to a real XML parser, so every attribute
// form that parser resolves has to resolve here too: a target it reaches and
// the guard does not is inflated on the main thread unbounded.
test("readAttachmentText refuses a relationship target in any XML attribute form", async () => {
  const huge = strToU8("a".repeat(11 * 1024 * 1024));
  const type =
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument";
  const forms: Array<[string, string, string]> = [
    [
      "single-quoted attributes",
      "payload.bin",
      `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id='rId1' Type='${type}' Target='payload.bin'/></Relationships>`,
    ],
    [
      "an entity-encoded target",
      "payload.bin",
      `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="${type}" Target="pay&#108;oad.bin"/></Relationships>`,
    ],
    [
      "a prefixed element name",
      "payload.bin",
      `<pkg:Relationships xmlns:pkg="http://schemas.openxmlformats.org/package/2006/relationships"><pkg:Relationship Id="rId1" Type="${type}" Target="payload.bin"/></pkg:Relationships>`,
    ],
    [
      "a target holding a decoy attribute",
      "payload.bin",
      `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id='Target="word/document.xml"' Type="${type}" Target="payload.bin"/></Relationships>`,
    ],
    [
      "a target holding a closing bracket",
      "pay>load.bin",
      `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="${type}" Target="pay>load.bin"/></Relationships>`,
    ],
  ];

  for (const [label, target, rels] of forms) {
    const bytes = zipSync({
      "[Content_Types].xml": strToU8("<Types/>"),
      "_rels/.rels": strToU8(rels),
      [target]: huge,
    });
    const bomb = fakeDocumentFile("bomb.docx", bytes.length, bytes, []);
    assert.equal(bomb.size < 1024 * 1024, true);
    await assert.rejects(
      readAttachmentText(bomb, bomb.name, undefined),
      new RegExp(
        `DOCX XML file is too large: bomb\\.docx:${target.replace(/[.>]/g, "\\$&")}$`,
      ),
      `${label} reached mammoth unbounded`,
    );
  }
});

// findPartPaths only opens the package parts and what the relationships point
// at, so an .xml part nothing references is never inflated. Custom XML data is
// a standard payload and may be large, so the suffix must not decide.
test("readAttachmentText lets a docx with a large unreferenced xml part through", async () => {
  const reads: string[] = [];
  const bytes = zipSync({
    "[Content_Types].xml": strToU8("<Types/>"),
    "_rels/.rels": relationships([["officeDocument", "word/document.xml"]]),
    "word/document.xml": strToU8("<w:document><w:body/></w:document>"),
    "word/_rels/document.xml.rels": relationships([]),
    "customXml/item1.xml": strToU8(
      `<data>${"b".repeat(11 * 1024 * 1024)}</data>`,
    ),
  });
  const file = fakeDocumentFile("custom.docx", bytes.length, bytes, reads);
  const error = await readAttachmentText(file, file.name, undefined).then(
    () => null,
    (thrown: Error) => thrown,
  );
  assert.deepEqual(reads, ["custom.docx"]);
  if (error) {
    assert.doesNotMatch(error.message, /too large/);
  }
});

// The composer empties itself before it awaits send(), so a part that only
// fails there takes the typed message with it: add() has to decide instead.
test("getDocxAttachmentError refuses an oversized part before the attachment is added", async () => {
  const bytes = zipSync({
    "[Content_Types].xml": strToU8("<Types/>"),
    "_rels/.rels": relationships([["officeDocument", "word/document.xml"]]),
    "word/document.xml": strToU8("<w:document><w:body/></w:document>"),
    "word/_rels/document.xml.rels": relationships([["styles", "styles.dat"]]),
    "word/styles.dat": strToU8("a".repeat(11 * 1024 * 1024)),
  });
  const bomb = fakeDocumentFile("styles.docx", bytes.length, bytes, []);
  assert.equal(bomb.size < 1024 * 1024, true);
  assert.equal(
    await getDocxAttachmentError(bomb),
    "DOCX XML file is too large: styles.docx:word/styles.dat",
  );

  const oversized = fakeDocumentFile(
    "huge.docx",
    60 * 1024 * 1024,
    new Uint8Array(0),
    [],
  );
  assert.equal(
    await getDocxAttachmentError(oversized),
    "DOCX file is too large: huge.docx",
  );

  const okBytes = docxBytes("<w:document><w:body/></w:document>");
  const ok = fakeDocumentFile("notes.docx", okBytes.length, okBytes, []);
  assert.equal(await getDocxAttachmentError(ok), null);
});

test("parseAttachmentText keeps an unterminated tag as plain text", () => {
  const parsed = parseAttachmentText("<attachment name=notes.txt>\nbody");
  assert.deepEqual(parsed, {
    label: null,
    text: "<attachment name=notes.txt>\nbody",
    truncated: false,
  });
});
