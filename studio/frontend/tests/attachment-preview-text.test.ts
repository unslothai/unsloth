// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  Unzip,
  UnzipInflate,
  strFromU8,
  strToU8,
  unzipSync,
  zipSync,
} from "fflate";

import { registerBundlerResolver } from "./helpers/kit.ts";
import { DOMParser } from "@xmldom/xmldom";

registerBundlerResolver();

const {
  attachmentAudioSrc,
  attachmentTextLanguage,
  countAttachmentTextLines,
  extractHtmlAttachmentText,
  extractPdfAttachmentText,
  getDocxAttachmentError,
  isAudioAttachment,
  parseAttachmentText,
  readAttachmentText,
  repackDocxAttachmentArchive,
  truncateAttachmentPreviewText,
} = await import("../src/features/chat/attachment-content.ts");
const { definePDFJSModule } = await import("unpdf");
const { readOfficeOpenXmlAttachmentContent } =
  await import("../src/features/chat/open-document.ts");
const { readRtfAttachmentContent } =
  await import("../src/features/chat/rtf.ts");
const { readIworkAttachmentContent } =
  await import("../src/features/chat/iwork.ts");

type StubNode = {
  nodeType: number;
  nodeValue?: string;
  tagName?: string;
  childNodes: StubNode[];
  parent?: StubNode;
  remove?: () => void;
};

function textNode(value: string): StubNode {
  return { nodeType: 3, nodeValue: value, childNodes: [] };
}

function element(tagName: string, ...childNodes: StubNode[]): StubNode {
  const node: StubNode = { nodeType: 1, tagName, childNodes };
  for (const child of childNodes) {
    child.parent = node;
    child.remove = () => {
      const siblings = node.childNodes;
      siblings.splice(siblings.indexOf(child), 1);
    };
  }
  return node;
}

function descendants(node: StubNode): StubNode[] {
  return node.childNodes.flatMap((child) => [child, ...descendants(child)]);
}

/** DOMParser is absent under node, so the extractor is driven over a hand-built tree. */
async function withStubDom<T>(
  build: (source: string) => StubNode,
  run: () => T | Promise<T>,
): Promise<T> {
  const original = (globalThis as { DOMParser?: unknown }).DOMParser;
  (globalThis as { DOMParser?: unknown }).DOMParser = class {
    parseFromString(source: string) {
      const body = build(source);
      return {
        body,
        querySelectorAll: (selector: string) => {
          const tags = new Set(selector.split(",").map((part) => part.trim()));
          return descendants(body).filter(
            (node) => node.tagName && tags.has(node.tagName),
          );
        },
      };
    }
  };
  try {
    return await run();
  } finally {
    (globalThis as { DOMParser?: unknown }).DOMParser = original;
  }
}

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

test("readAttachmentText previews UTF-16 registry exports as decoded text", async () => {
  const text =
    "Windows Registry Editor Version 5.00\r\n\r\n[HKEY_CURRENT_USER\\Software\\Test]";
  const utf16le = new Uint8Array(2 + text.length * 2);
  utf16le.set([0xff, 0xfe]);
  for (let index = 0; index < text.length; index += 1) {
    const codeUnit = text.charCodeAt(index);
    utf16le[2 + index * 2] = codeUnit & 0xff;
    utf16le[3 + index * 2] = codeUnit >>> 8;
  }
  const file = new File([utf16le], "export.reg");

  assert.deepEqual(await readAttachmentText(file, file.name, file.type), {
    label: null,
    text,
    truncated: false,
  });
});

test("readAttachmentText previews gettext catalogs in their declared charset", async () => {
  const before =
    'msgid ""\nmsgstr ""\n"Content-Type: text/plain; charset=ISO-8859-1\\n"\n\nmsgid "coffee"\nmsgstr "caf';
  const after = '"\n';
  const encoded = Uint8Array.from([
    ...new TextEncoder().encode(before),
    0xe9,
    ...new TextEncoder().encode(after),
  ]);
  const file = new File([encoded], "messages.po");

  assert.deepEqual(await readAttachmentText(file, file.name, file.type), {
    label: null,
    text: `${before}é${after}`,
    truncated: false,
  });
});

test("readAttachmentText reads a bounded slice of a large html file", async () => {
  const oversized = new File(
    [`<p>${"b".repeat(2_000_000)}</p>`],
    "huge.html",
    { type: "text/html" },
  );
  const { label, text, truncated } = await readAttachmentText(
    oversized,
    oversized.name,
    oversized.type,
  );

  assert.equal(label, null);
  assert.equal(truncated, true);
  assert.equal(text.length, 1_000_000);
});

test("readAttachmentText does not decode a file only the python tool reads", async () => {
  const file = new File([new Uint8Array([0x50, 0x41, 0x52, 0x31])], "t.PARQUET");
  assert.deepEqual(await readAttachmentText(file, file.name, file.type), {
    label: null,
    text: "t.PARQUET has no preview: only the python tool can read it.",
    truncated: false,
  });
});

// the adapter sends the extraction; the preview shows the markup unextracted
test("readAttachmentText previews an html file as its markup", async () => {
  const markup = "<p>Drag to rotate<br>Scroll to zoom</p>";
  const file = new File([markup], "page.html", { type: "text/html" });

  assert.deepEqual(await readAttachmentText(file, file.name, file.type), {
    label: null,
    text: markup,
    truncated: false,
  });
});

/** textContent runs a whole page onto one line, and this extraction is what the html adapter sends the model. */
test("extractHtmlAttachmentText keeps the line structure of the page", async () => {
  const extracted = await withStubDom(
    () =>
      element(
        "body",
        element("h1", textNode("Solar System Explorer")),
        element(
          "p",
          textNode("Drag  to rotate"),
          element("br"),
          textNode("Scroll to zoom"),
        ),
        element(
          "ul",
          element("li", textNode("Sun")),
          element("li", textNode("Mercury")),
        ),
        element("script", textNode("const planets = 8;")),
        element("style", textNode("body { margin: 0 }")),
      ),
    () => extractHtmlAttachmentText("<html/>"),
  );

  assert.equal(
    extracted,
    "Solar System Explorer\n\nDrag to rotate\nScroll to zoom\n\nSun\n\nMercury",
  );
});

test("isAudioAttachment matches by MIME and by extension", () => {
  assert.equal(isAudioAttachment("clip.m4a", ""), true);
  assert.equal(isAudioAttachment("clip", "audio/webm"), true);
  assert.equal(isAudioAttachment("notes.txt", "text/plain"), false);
  assert.equal(isAudioAttachment(undefined, undefined), false);
});

// CompositeAttachmentAdapter checks TextAttachmentAdapter before the
// document-specific adapters, so a browser-declared text MIME wins over a
// misleading extension in both the sent payload and its preview.
test("readAttachmentText follows text adapter precedence over document extensions", async () => {
  for (const name of ["notes.pdf", "notes.docx", "notes.html"]) {
    const file = new File([`plain text from ${name}`], name, {
      type: "text/plain",
    });
    assert.deepEqual(
      await readAttachmentText(file, file.name, file.type),
      {
        label: null,
        text: `plain text from ${name}`,
        truncated: false,
      },
    );
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

test("extractPdfAttachmentText destroys the PDF proxy after success and failure", async () => {
  const destroyed: string[] = [];
  const proxies = [
    {
      _pdfInfo: {},
      numPages: 1,
      getPage: async () => ({
        getTextContent: async () => ({
          items: [
            { str: "page one", hasEOL: true },
            { str: "page two", hasEOL: false },
          ],
        }),
      }),
      destroy: async () => {
        destroyed.push("success");
      },
    },
    {
      _pdfInfo: {},
      numPages: 1,
      getPage: async () => {
        throw new Error("page extraction failed");
      },
      destroy: async () => {
        destroyed.push("failure");
      },
    },
  ];

  await definePDFJSModule(async () => ({
    getDocument: () => ({ promise: Promise.resolve(proxies.shift()) }),
  }));
  try {
    const file = new File(["%PDF"], "small.pdf", {
      type: "application/pdf",
    });
    assert.equal(await extractPdfAttachmentText(file), "page one\npage two");
    await assert.rejects(
      extractPdfAttachmentText(file),
      /page extraction failed/,
    );
    assert.deepEqual(destroyed, ["success", "failure"]);
  } finally {
    await definePDFJSModule(() => import("unpdf/pdfjs"));
  }
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

/**
 * A relationship inside non-element markup is text to mammoth's parser, so it
 * must not select the bounded part in either direction: it cannot stand in for
 * the real target and hide it, and it cannot refuse a document mammoth reads.
 */
test("readAttachmentText ignores a relationship inside non-element markup", async () => {
  const huge = strToU8("a".repeat(11 * 1024 * 1024));
  const type =
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument";
  const wrappers: Array<[string, (tag: string) => string]> = [
    ["a comment", (tag) => `<!--${tag}-->`],
    ["a CDATA section", (tag) => `<![CDATA[${tag}]]>`],
    ["a processing instruction", (tag) => `<?guard ${tag}?>`],
  ];
  const rels = (wrap: (tag: string) => string, buried: string, live: string) =>
    strToU8(
      `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">${wrap(
        `<Relationship Id="rId0" Type="${type}" Target="${buried}"/>`,
      )}<Relationship Id="rId1" Type="${type}" Target="${live}"/></Relationships>`,
    );

  for (const [label, wrap] of wrappers) {
    const hidden = zipSync({
      "[Content_Types].xml": strToU8("<Types/>"),
      "_rels/.rels": rels(wrap, "word/document.xml", "payload.bin"),
      "word/document.xml": strToU8("<w:document><w:body/></w:document>"),
      "payload.bin": huge,
    });
    const hiddenFile = fakeDocumentFile("bomb.docx", hidden.length, hidden, []);
    await assert.rejects(
      readAttachmentText(hiddenFile, hiddenFile.name, undefined),
      /DOCX XML file is too large: bomb\.docx:payload\.bin/,
      `${label} stood in for the live relationship`,
    );

    const reads: string[] = [];
    const refused = zipSync({
      "[Content_Types].xml": strToU8("<Types/>"),
      "_rels/.rels": rels(wrap, "payload.bin", "word/document.xml"),
      "word/document.xml": strToU8("<w:document><w:body/></w:document>"),
      "word/_rels/document.xml.rels": relationships([]),
      "payload.bin": huge,
    });
    const refusedFile = fakeDocumentFile(
      "notes.docx",
      refused.length,
      refused,
      reads,
    );
    const error = await readAttachmentText(
      refusedFile,
      refusedFile.name,
      undefined,
    ).then(
      () => null,
      (thrown: Error) => thrown,
    );
    assert.deepEqual(reads, ["notes.docx"]);
    if (error) {
      assert.doesNotMatch(error.message, /too large/, label);
    }
  }
});

/** The XML declaration every real .rels file opens with is a processing instruction too, so stripping them must not cost a live relationship. */
test("readAttachmentText keeps resolving a rels file that opens with its xml declaration", async () => {
  const bytes = zipSync({
    "[Content_Types].xml": strToU8("<Types/>"),
    "_rels/.rels": strToU8(
      `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n${strFromU8(
        relationships([["officeDocument", "payload.bin"]]),
      )}`,
    ),
    "payload.bin": strToU8("a".repeat(11 * 1024 * 1024)),
  });
  const file = fakeDocumentFile("bomb.docx", bytes.length, bytes, []);
  await assert.rejects(
    readAttachmentText(file, file.name, undefined),
    /DOCX XML file is too large: bomb\.docx:payload\.bin/,
  );
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

/** Rewrites every field holding `size` down to `declared`, the way a crafted archive lies about a part. */
function understateDeclaredSizes(
  archive: Uint8Array,
  size: number,
  declared: number,
): number {
  const view = new DataView(
    archive.buffer,
    archive.byteOffset,
    archive.byteLength,
  );
  let patched = 0;
  for (let offset = 0; offset + 4 <= archive.length; offset++) {
    if (view.getUint32(offset, true) === size) {
      view.setUint32(offset, declared, true);
      patched++;
    }
  }
  return patched;
}

/**
 * Inflated sizes as jszip sees them: the whole stream, whatever the archive
 * declares. `unzipSync` cannot answer this, since it allocates each entry at
 * its declared size and stops there, which is why the declared size proves
 * nothing about what mammoth would decompress.
 */
function inflatedSizes(archive: Uint8Array): Map<string, number> {
  const sizes = new Map<string, number>();
  const unzip = new Unzip();
  unzip.register(UnzipInflate);
  unzip.onfile = (file) => {
    let size = 0;
    file.ondata = (error, chunk) => {
      if (!error) {
        size += chunk.length;
        sizes.set(file.name, size);
      }
    };
    file.start();
  };
  unzip.push(archive, true);
  return sizes;
}

/**
 * jszip takes each part's size from the central directory and inflates the part
 * in full before it can be rejected, so a lying header still expands inside
 * mammoth. fflate allocates the entry at the declared size and stops, so the
 * repack is what contains the lie.
 */
test("repackDocxAttachmentArchive bounds a part that lies about its size", () => {
  const body = strToU8("a".repeat(30 * 1024 * 1024));
  const archive = zipSync(
    {
      "[Content_Types].xml": strToU8("<Types/>"),
      "_rels/.rels": relationships([["officeDocument", "word/document.xml"]]),
      "word/document.xml": body,
    },
    { level: 9 },
  );
  assert.equal(understateDeclaredSizes(archive, body.length, 1024), 2);
  assert.equal(archive.length < 1024 * 1024, true);
  assert.equal(inflatedSizes(archive).get("word/document.xml"), body.length);

  const repacked = repackDocxAttachmentArchive("lie.docx", archive);
  assert.equal(inflatedSizes(repacked).get("word/document.xml"), 1024);
  assert.equal(unzipSync(repacked)["word/document.xml"].length, 1024);
});

test("repackDocxAttachmentArchive keeps an honest archive intact", () => {
  const files = {
    "[Content_Types].xml": strToU8("<Types/>"),
    "_rels/.rels": relationships([["officeDocument", "word/document.xml"]]),
    "word/document.xml": strToU8("<w:document><w:body/></w:document>"),
    "word/media/photo.png": new Uint8Array(4096),
  };
  const repacked = unzipSync(
    repackDocxAttachmentArchive("notes.docx", zipSync(files, { level: 9 })),
  );

  assert.deepEqual(Object.keys(repacked).sort(), Object.keys(files).sort());
  for (const [name, bytes] of Object.entries(files)) {
    assert.deepEqual(repacked[name], bytes, name);
  }
});

/** Every part can sit under the XML ceiling while the archive as a whole still unpacks to more than the webview can hold. */
test("repackDocxAttachmentArchive refuses an archive that unpacks past the ceiling", () => {
  const part = new Uint8Array(9 * 1024 * 1024);
  const files: Record<string, Uint8Array> = {
    "[Content_Types].xml": strToU8("<Types/>"),
    "_rels/.rels": relationships([["officeDocument", "word/document.xml"]]),
    "word/document.xml": strToU8("<w:document><w:body/></w:document>"),
  };
  for (let index = 0; index < 12; index++) {
    files[`word/media/photo${index}.bin`] = part;
  }

  const archive = zipSync(files, { level: 1 });
  assert.throws(
    () => repackDocxAttachmentArchive("wide.docx", archive),
    /DOCX file is too large: wide\.docx/,
  );
});

/** A preview only colours what the filename says is source; extracted document text is prose whatever the file was called. */
test("attachmentTextLanguage maps source files and leaves prose alone", () => {
  assert.equal(attachmentTextLanguage("train.py", null), "python");
  assert.equal(attachmentTextLanguage("Chart.YAML", null), "yaml");
  assert.equal(attachmentTextLanguage("page.html", null), "html");
  assert.equal(attachmentTextLanguage("notes.txt", null), null);
  assert.equal(attachmentTextLanguage("script.py", "PDF"), null);
  // the label parsed from the adapter's wrapper keeps a sent extraction unhighlighted
  assert.equal(
    attachmentTextLanguage(
      "page.html",
      parseAttachmentText("[HTML: page.html]\nDrag to rotate").label,
    ),
    null,
  );
  assert.equal(attachmentTextLanguage(undefined, null), null);
});

/**
 * Every extension and entity table here is a plain object literal, so a key
 * that names a member of Object.prototype resolves to a function rather than
 * missing. The entity case is the one that matters: a resolved target would
 * carry the source text of that function and bound a path mammoth never reads.
 */
test("prototype member names do not resolve as table entries", async () => {
  assert.equal(attachmentTextLanguage("notes.constructor", null), null);
  assert.equal(attachmentTextLanguage("notes.toString", null), null);
  assert.equal(attachmentTextLanguage("notes.py", null), "python");

  assert.equal(
    attachmentAudioSrc({ data: "AAA", format: "wav" }, "", "clip.constructor"),
    "data:audio/wav;base64,AAA",
  );

  const type =
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument";
  const bytes = zipSync({
    "[Content_Types].xml": strToU8("<Types/>"),
    "_rels/.rels": strToU8(
      `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="${type}" Target="pay&constructor;load.bin"/></Relationships>`,
    ),
    "pay&constructor;load.bin": strToU8("a".repeat(11 * 1024 * 1024)),
  });
  const bomb = fakeDocumentFile("bomb.docx", bytes.length, bytes, []);
  await assert.rejects(
    readAttachmentText(bomb, bomb.name, undefined),
    /DOCX XML file is too large: bomb\.docx:pay&constructor;load\.bin/,
  );
});

test("parseAttachmentText keeps an unterminated tag as plain text", () => {
  const parsed = parseAttachmentText("<attachment name=notes.txt>\nbody");
  assert.deepEqual(parsed, {
    label: null,
    text: "<attachment name=notes.txt>\nbody",
    truncated: false,
  });
});

test("a preview is never stricter than the adapter that took the file", async () => {
  // .html belongs to the HTML adapter, which sends a legacy page happily. The
  // preview went through the strict text decoder and threw on the same file,
  // so opening an attachment that had already been accepted failed.
  const head = new TextEncoder().encode(
    '<!doctype html><meta charset="windows-1252"><body>Caf',
  );
  const bytes = new Uint8Array([
    ...head,
    0xe9,
    ...new TextEncoder().encode("</body>"),
  ]);
  const page = new File([bytes], "page.html", { type: "text/html" });
  const preview = await readAttachmentText(page, page.name, page.type);
  assert.equal(typeof preview.text, "string");
  assert.ok(preview.text.includes("Caf"));

  // A file the text adapter does own stays strict: mojibake reaching the model
  // is worse than a message saying the encoding could not be read.
  const { UndecodableTextError } = await import(
    "../src/features/chat/text-attachment-accept.ts"
  );
  await assert.rejects(
    readAttachmentText(
      new File([bytes], "notes.srt"),
      "notes.srt",
      "text/plain",
    ),
    (error: Error) => error instanceof UndecodableTextError,
  );
});

(globalThis as { DOMParser?: unknown }).DOMParser = DOMParser;


const REL =
  "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
const MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";

function rels(entries: [string, string, string][]): string {
  return `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">${entries
    .map(
      ([id, type, target]) =>
        `<Relationship Id="${id}" Type="${REL}/${type}" Target="${target}"/>`,
    )
    .join("")}</Relationships>`;
}

function sheet(body: string): string {
  return `<worksheet xmlns="${MAIN}">${body}</worksheet>`;
}

function workbookFile(parts: Record<string, string>, name = "book.xlsx"): File {
  const zipped = zipSync(
    Object.fromEntries(
      Object.entries(parts).map(([path, xml]) => [path, strToU8(xml)]),
    ),
  );
  return new File([zipped], name);
}

const BASE_PARTS = {
  "_rels/.rels": rels([["rId1", "officeDocument", "/xl/workbook.xml"]]),
  "xl/_rels/workbook.xml.rels": rels([
    ["rIdA", "worksheet", "worksheets/sheet1.xml"],
    ["rIdB", "worksheet", "/xl/worksheets/sheet2.xml"],
    ["rIdC", "worksheet", "../xl/worksheets/sheet3.xml"],
    ["rIdS", "sharedStrings", "sharedStrings.xml"],
    ["rIdT", "styles", "styles.xml"],
  ]),
  // Listed out of archive order: the workbook, not the zip, decides sheet order.
  "xl/workbook.xml": `<workbook xmlns="${MAIN}" xmlns:r="${REL}"><sheets>
    <sheet name="Second" sheetId="2" r:id="rIdB"/>
    <sheet name="Secret" sheetId="3" state="hidden" r:id="rIdC"/>
    <sheet name="First" sheetId="1" r:id="rIdA"/>
  </sheets></workbook>`,
  "xl/sharedStrings.xml": `<sst xmlns="${MAIN}">
    <si><t>Name</t></si>
    <si><r><t>Ri</t></r><r><t>ch</t></r><rPh><t>ルビ</t></rPh></si>
  </sst>`,
  "xl/styles.xml": `<styleSheet xmlns="${MAIN}">
    <numFmts><numFmt numFmtId="164" formatCode="yyyy\\-mm\\-dd"/><numFmt numFmtId="165" formatCode="&quot;day&quot; 0"/><numFmt numFmtId="166" formatCode="yyyy-mm-dd hh:mm"/><numFmt numFmtId="167" formatCode="[h]:mm"/></numFmts>
    <cellXfs><xf numFmtId="0"/><xf numFmtId="14"/><xf numFmtId="164"/><xf numFmtId="165"/><xf numFmtId="21"/><xf numFmtId="46"/><xf numFmtId="166"/><xf numFmtId="167"/></cellXfs>
  </styleSheet>`,
  "xl/worksheets/sheet1.xml":
    sheet(`<cols><col min="-1000000000" max="0" hidden="1"/><col min="3" max="3" hidden="1"/></cols><sheetData>
    <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1"><v>9</v></c><c r="D1" t="inlineStr"><is><t>inline</t></is></c></row>
    <row r="2" hidden="1"><c r="A2"><v>hidden row</v></c></row>
    <row r="3"><c r="B3" t="b"><v>1</v></c><c r="D3" t="e"><v>#DIV/0!</v></c></row>
    <row r="4"><c r="A4"><v>0.30000000000000004</v></c><c r="B4" s="1"><v>45000</v></c><c r="D4" s="2"><v>45000.5</v></c><c r="E4" s="6"><v>45000.5</v></c></row>
    <row r="5"><c r="A5" s="3"><v>12</v></c><c r="B5" s="4"><v>1.25</v></c><c r="D5" s="5"><v>1.5</v></c><c r="E5" s="7"><v>2.25</v></c></row>
  </sheetData>`),
  "xl/worksheets/sheet2.xml": sheet(
    `<sheetData><row r="1"><c r="A1" t="str"><v>formula text</v></c></row></sheetData>`,
  ),
  "xl/worksheets/sheet3.xml": sheet(
    `<sheetData><row r="1"><c r="A1" t="str"><v>do not show</v></c></row></sheetData>`,
  ),
};

test("a workbook reads as tab-separated sheets in workbook order", async () => {
  const content = await readOfficeOpenXmlAttachmentContent(
    workbookFile(BASE_PARTS),
    "book.xlsx",
  );
  assert.equal(content.label, "XLSX");
  assert.equal(
    content.text,
    [
      "[Sheet: Second]",
      "formula text",
      "",
      "[Sheet: First]",
      "Name\tRich\tinline",
      "\tTRUE\t#DIV/0!",
      "0.3\t2023-03-15\t2023-03-15\t2023-03-15 12:00:00",
      "12\t06:00:00\t36:00:00\t54:00:00",
    ].join("\n"),
  );
});

test("a part declaring more than the XML cap is refused before it inflates", async () => {
  await assert.rejects(
    readOfficeOpenXmlAttachmentContent(
      workbookFile({
        ...BASE_PARTS,
        "xl/worksheets/sheet1.xml": sheet(" ".repeat(10 * 1024 * 1024 + 1)),
      }),
      "book.xlsx",
    ),
    /unpacks too large/,
  );
});

test("parts under the XML cap are still refused once they unpack past the total", async () => {
  const filler = sheet(" ".repeat(9 * 1024 * 1024));
  const parts: Record<string, string> = { ...BASE_PARTS };
  for (let index = 0; index < 12; index++) {
    parts[`xl/worksheets/filler${index}.xml`] = filler;
  }
  await assert.rejects(
    readOfficeOpenXmlAttachmentContent(workbookFile(parts), "book.xlsx"),
    /unpacks too large/,
  );
});

test("a stored part is charged its real size, not the size it declares", async () => {
  const name = "xl/worksheets/sheet1.xml";
  const zipped = zipSync({
    ...Object.fromEntries(
      Object.entries(BASE_PARTS).map(([path, xml]) => [path, strToU8(xml)]),
    ),
    [name]: [strToU8(sheet(" ".repeat(11 * 1024 * 1024))), { level: 0 }],
  });
  // Rewrite the central directory to declare the stored part as 1 byte.
  const view = new DataView(zipped.buffer);
  for (let offset = 0; offset < zipped.length - 46; offset++) {
    if (view.getUint32(offset, true) !== 0x02014b50) continue;
    const nameLength = view.getUint16(offset + 28, true);
    const entry = new TextDecoder().decode(
      zipped.subarray(offset + 46, offset + 46 + nameLength),
    );
    if (entry === name) view.setUint32(offset + 24, 1, true);
  }
  await assert.rejects(
    readOfficeOpenXmlAttachmentContent(
      new File([zipped], "book.xlsx"),
      "book.xlsx",
    ),
    /unpacks too large/,
  );
});

test("a row repeating one long shared string stops at the text budget", async () => {
  const cells = Array.from(
    { length: 1024 },
    () => `<c t="s"><v>0</v></c>`,
  ).join("");
  const content = await readOfficeOpenXmlAttachmentContent(
    workbookFile({
      ...BASE_PARTS,
      "xl/workbook.xml": `<workbook xmlns="${MAIN}" xmlns:r="${REL}"><sheets><sheet name="S" sheetId="1" r:id="rIdA"/></sheets></workbook>`,
      "xl/sharedStrings.xml": `<sst xmlns="${MAIN}"><si><t>${"x".repeat(1024 * 1024)}</t></si></sst>`,
      "xl/worksheets/sheet1.xml": sheet(
        `<sheetData><row>${cells}</row></sheetData>`,
      ),
    }),
    "book.xlsx",
  );
  assert.ok(content.text.length < 11 * 1024 * 1024);
  // The cells that fit are kept, not the whole row dropped.
  assert.equal(content.text.split("\t").length, 9);
  assert.match(content.text, /\[Truncated: /);
});

test("the preview reads a picked workbook as labelled text", async () => {
  const preview = await readAttachmentText(
    workbookFile(BASE_PARTS, "Book.XLSM"),
    "Book.XLSM",
    "",
  );
  assert.equal(preview.label, "XLSX");
  assert.match(preview.text, /^\[Sheet: Second\]\nformula text/);
});

const PML = "http://schemas.openxmlformats.org/presentationml/2006/main";
const DML = "http://schemas.openxmlformats.org/drawingml/2006/main";

function slide(body: string, show = ""): string {
  return `<p:sld xmlns:p="${PML}" xmlns:a="${DML}" xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"${show}><p:cSld><p:spTree>${body}</p:spTree></p:cSld></p:sld>`;
}

function shape(text: string, hidden = ""): string {
  return `<p:sp><p:nvSpPr><p:cNvPr id="2" name="s"${hidden}/></p:nvSpPr><p:txBody>${text}</p:txBody></p:sp>`;
}

const DECK_PARTS = {
  "_rels/.rels": rels([["rId1", "officeDocument", "ppt/presentation.xml"]]),
  "ppt/_rels/presentation.xml.rels": rels([
    ["rId7", "slide", "slides/slide1.xml"],
    ["rId8", "slide", "slides/slide2.xml"],
    ["rId9", "slide", "slides/slide3.xml"],
  ]),
  // The plain id comes first, as PowerPoint writes it.
  "ppt/presentation.xml": `<p:presentation xmlns:p="${PML}" xmlns:r="${REL}"><p:sldIdLst><p:sldId id="256" r:id="rId9"/><p:sldId id="257" r:id="rId8"/><p:sldId id="258" r:id="rId7"/></p:sldIdLst></p:presentation>`,
  // Archived out of deck order.
  "ppt/slides/slide1.xml": slide(
    `${shape(`<a:p><a:r><a:t>before</a:t></a:r></a:p>`)}<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="3" name="t"/></p:nvGraphicFramePr><a:graphic><a:graphicData><a:tbl><a:tr><a:tc><a:txBody><a:p><a:r><a:t>Region</a:t></a:r></a:p></a:txBody></a:tc><a:tc><a:txBody><a:p><a:r><a:t>Sales</a:t></a:r></a:p><a:p><a:r><a:t>(k)</a:t></a:r></a:p></a:txBody></a:tc></a:tr><a:tr><a:tc><a:txBody><a:p><a:r><a:t>North</a:t></a:r></a:p></a:txBody></a:tc><a:tc><a:txBody><a:p><a:r><a:t>42</a:t></a:r></a:p></a:txBody></a:tc></a:tr></a:tbl></a:graphicData></a:graphic></p:graphicFrame>${shape(`<a:p><a:r><a:t>after</a:t></a:r></a:p>`)}`,
  ),
  "ppt/slides/slide3.xml": slide(
    shape(
      `<a:p><a:r><a:t>Tit</a:t></a:r><a:r><a:t>le</a:t></a:r></a:p><a:p/><a:p><a:r><a:t>one</a:t></a:r><a:br/><a:r><a:t>two</a:t></a:r><mc:AlternateContent><mc:Choice Requires="a14"><a:r><a:t>!</a:t></a:r></mc:Choice><mc:Fallback><a:r><a:t>?</a:t></a:r></mc:Fallback></mc:AlternateContent></a:p>`,
    ) +
      shape(`<a:p><a:r><a:t>secret</a:t></a:r></a:p>`, ' hidden="1"') +
      `<mc:AlternateContent><mc:Choice Requires="a14">${shape(`<a:p><a:r><a:t>choice</a:t></a:r></a:p>`)}</mc:Choice><mc:Fallback>${shape(`<a:p><a:r><a:t>fallback</a:t></a:r></a:p>`)}</mc:Fallback></mc:AlternateContent>`,
    ' show="1"',
  ),
  "ppt/slides/slide2.xml": slide(
    shape(`<a:p><a:r><a:t>skipped slide</a:t></a:r></a:p>`),
    ' show="false"',
  ),
};

test("a presentation reads its visible slides in deck order", async () => {
  const content = await readOfficeOpenXmlAttachmentContent(
    workbookFile(DECK_PARTS, "deck.pptx"),
    "deck.pptx",
  );
  assert.equal(content.label, "PPTX");
  assert.equal(
    content.text,
    [
      "[Slide 1]",
      "Title",
      "one\ntwo!",
      "choice",
      "",
      "[Slide 3]",
      "before",
      "Region\tSales (k)",
      "North\t42",
      "after",
    ].join("\n"),
  );
});

async function readRtf(rtf: string | Uint8Array<ArrayBuffer>): Promise<string> {
  const content = await readRtfAttachmentContent(
    new File([rtf], "doc.rtf"),
    "doc.rtf",
  );
  assert.equal(content.label, "RTF");
  return content.text;
}

test("TextEdit output decodes bytes by the font charset, not the ANSI code page", async () => {
  const text = await readRtf(
    [
      "{\\rtf1\\ansi\\ansicpg936\\cocoartf2870",
      "{\\fonttbl\\f0\\fswiss\\fcharset0 Helvetica;}",
      "{\\colortbl;\\red255\\green255\\blue255;}",
      "{\\*\\expandedcolortbl;;}",
      "\\f0\\fs24 \\cf0 H\\'e9llo \\'93world\\'94\\",
      "\\",
      "Tab\there \\uc0\\u26085 \\u26412 \\",
      "}",
    ].join("\n"),
  );
  assert.equal(text, "Héllo “world”\n\nTab\there 日本");
});

test("Word-style output reads fields, tables and double-byte fonts", async () => {
  const text = await readRtf(
    new Uint8Array([
      ...new TextEncoder().encode(
        [
          "{\\rtf1\\ansi\\ansicpg1252\\deff0\\uc1",
          "{\\fonttbl{\\f0\\fnil\\fcharset134 SimSun;}{\\f1\\fcharset0 Arial;}{\\f2\\fcharset128 MS Mincho;}{\\f3\\fcharset204 Arial;}}",
          "{\\header Page header\\par}",
          "{\\info{\\title Secret title}}",
          "\\pard \\'c4\\'e3\\'ba\\'c3 \\f1 caf\\'e9\\emdash\\u8364?\\u-10179?\\u-8704?\\{x\\}\\par",
          '{\\field{\\*\\fldinst{HYPERLINK "https://x.test"}}{\\fldrslt link}}\\par',
          "{\\pict\\bin4 ",
        ].join("\n"),
      ),
      0x7d,
      0x7b,
      0x5c,
      0x7d,
      ...new TextEncoder().encode(
        [
          "}",
          "\\trowd\\cellx100\\cellx200",
          "\\pard\\intbl first\\par second\\par\\cell b\\cell\\row",
          "\\pard after\\par more \\f2\\'83e\\'83X\\'83g\\'83\\\\ \\f3\\'cf\\plain\\'c4\\'e3\\par}",
        ].join("\n"),
      ),
    ]),
  );
  assert.equal(
    text,
    "你好 café—€😀{x}\nlink\nfirst second\tb\nafter\nmore テストソ П你",
  );
});

test("an RTF reader stays bounded", async () => {
  // Nesting past the text budget is never reached, so it cannot fail the read.
  const long = await readRtf(
    `{\\rtf1 ${"x".repeat(11 * 1024 * 1024)}${"{".repeat(2000)}`,
  );
  assert.ok(long.length < 11 * 1024 * 1024);
  assert.match(long, /^x+\n\n\[Truncated: [^\n]*\]$/);
  await assert.rejects(readRtf(`{\\rtf1 ${"{".repeat(2000)}`), /nest too deeply/);
  await assert.rejects(readRtf("plain text"), /Not an RTF file/);
});

type Bytes = number[];

function varint(value: number): Bytes {
  const out: Bytes = [];
  for (; value >= 0x80; value = Math.floor(value / 0x80)) {
    out.push((value % 0x80) | 0x80);
  }
  return [...out, value];
}

const int = (field: number, value: number): Bytes => [
  ...varint(field * 8),
  ...varint(value),
];
const bytes = (field: number, value: Bytes | string): Bytes => {
  const data = typeof value === "string" ? [...strToU8(value)] : value;
  return [...varint(field * 8 + 2), ...varint(data.length), ...data];
};
const ref = (field: number, id: number): Bytes => bytes(field, int(1, id));

/** Records as iWork writes them, over two Snappy chunks, optionally ending on one- and two-byte
 *  copies that repeat the two bytes before them three more times. */
function iwa(
  objects: [number, number, Bytes][],
  overlapTail = false,
): Uint8Array {
  const data = objects.flatMap(([id, type, payload]) => {
    const info = [
      ...int(1, id),
      ...bytes(2, [...int(1, type), ...int(3, payload.length)]),
    ];
    return [...varint(info.length), ...info, ...payload];
  });
  const half = data.length >> 1;
  return new Uint8Array([
    ...chunk(data.slice(0, half), []),
    ...chunk(data.slice(half), overlapTail ? [1, 2, 2 | (1 << 2), 2, 0] : []),
  ]);
}

/** A Snappy block of `data`, whose last six bytes come from `copies` when there are any. */
function chunk(data: Bytes, copies: Bytes): Bytes {
  const literal = copies.length > 0 ? data.slice(0, -6) : data;
  const block = [...varint(data.length)];
  for (let at = 0; at < literal.length; at += 256) {
    const run = literal.slice(at, at + 256);
    block.push(60 << 2, run.length - 1, ...run);
  }
  block.push(...copies);
  const size = block.length;
  return [0, size & 0xff, (size >> 8) & 0xff, size >> 16, ...block];
}

function iworkFile(name: string, index: Uint8Array): File {
  const zipped = zipSync({
    "Index/Document.iwa": index,
    "preview.jpg": strToU8("x"),
  });
  return new File([zipped], name);
}

const readIwork = (name: string, index: Uint8Array) =>
  readIworkAttachmentContent(iworkFile(name, index), name);

test("a Pages body reads as text, with inline objects dropped", async () => {
  const file = iworkFile(
    "notes.pages",
    iwa(
      [
        [3, 2001, bytes(3, "A header, not the body")],
        [1, 10000, ref(4, 2)],
        [2, 2001, bytes(3, "Title\uFFFC\u2028Body\u0004 hahahaha")],
      ],
      true,
    ),
  );
  assert.deepEqual(await readIworkAttachmentContent(file, file.name), {
    label: "PAGES",
    text: "Title\nBody hahahaha",
  });
  const viaMime = await readAttachmentText(
    file,
    "notes",
    "application/x-iwork-pages-sffpages",
  );
  assert.equal(viaMime.label, "PAGES");
});

test("Keynote slides read in deck order, numbered past skipped slides", async () => {
  const storage = (id: number, text: string): [number, number, Bytes] => [
    id,
    2001,
    bytes(3, text),
  ];
  const content = await readIwork(
    "deck.key",
    iwa([
      [10, 2, bytes(3, [...ref(2, 20), ...ref(2, 21), ...ref(2, 23)])],
      [20, 4, [...ref(2, 30), ...ref(1, 22)]],
      // A node listed twice in the tree is still one slide.
      [22, 4, [...ref(2, 32), ...int(4, 1), ...ref(1, 21)]],
      [21, 4, ref(2, 31)],
      [23, 4, ref(2, 33)],
      // Builds come first on the wire, and set neither the slide's text nor its order.
      [
        30,
        5,
        [
          ...ref(2, 70),
          ...ref(5, 40),
          ...ref(7, 50),
          ...ref(17, 31),
          ...ref(27, 60),
        ],
      ],
      // A shape links back to a slide, which is not followed.
      [50, 2011, [...bytes(1, ref(2, 41)), ...ref(3, 31)]],
      [60, 15, ref(1, 44)],
      [70, 8, ref(1, 50)],
      [31, 5, ref(6, 42)],
      [32, 5, ref(5, 43)],
      [33, 5, ref(5, 45)],
      storage(40, "Intro"),
      storage(41, "Nested box"),
      storage(42, "End"),
      storage(43, "Skipped"),
      storage(44, "Speaker notes"),
      storage(45, "Last"),
    ]),
  );
  assert.deepEqual(content, {
    label: "KEY",
    text: "[Slide 1]\nIntro\nNested box\n\n[Slide 3]\nEnd\n\n[Slide 4]\nLast",
  });
});

function cell(type: number, flags: number, values: Bytes): Bytes {
  return [5, type, 0, 0, 0, 0, 0, 0, flags, 0, 0, 0, ...values];
}

function float64(value: number): Bytes {
  return [...new Uint8Array(new Float64Array([value]).buffer)];
}

function rowInfo(index: number, cells: (Bytes | null)[]): Bytes {
  const buffer: Bytes = [];
  const offsets: Bytes = [];
  for (const value of cells) {
    const offset = value ? buffer.length : 0xffff;
    offsets.push(offset & 0xff, offset >> 8);
    buffer.push(...(value ?? []));
  }
  return bytes(5, [
    ...int(1, index),
    ...bytes(6, buffer),
    ...bytes(7, offsets),
  ]);
}

test("Numbers sheets read their tables as rows of typed cells", async () => {
  // -12345678901234567.89: a mantissa past 2^53, exponent -2 stored with bias 0x1820.
  const mantissa = BigInt("1234567890123456789");
  const biased = 0x1820 - 2;
  const decimal = [
    ...Array.from({ length: 14 }, (_, index) =>
      Number((mantissa >> BigInt(8 * index)) & BigInt(0xff)),
    ),
    (biased & 0x7f) << 1,
    0x80 | (biased >> 7),
  ];
  // Listed out of order: the second tile first.
  const tileStorage = [
    ...bytes(1, [...int(1, 1), ...ref(2, 7)]),
    ...bytes(1, [...int(1, 0), ...ref(2, 6)]),
  ];
  const content = await readIwork(
    "budget.numbers",
    iwa([
      [1, 1, ref(1, 2)],
      [2, 2, [...bytes(1, "Budget"), ...ref(2, 3)]],
      [3, 6000, ref(2, 4)],
      [
        4,
        6001,
        [
          ...bytes(8, "Costs"),
          ...int(6, 300),
          ...int(7, 4),
          ...bytes(4, [...bytes(3, tileStorage), ...ref(4, 5)]),
        ],
      ],
      [
        5,
        6005,
        [...int(1, 1), ...bytes(3, [...int(1, 7), ...bytes(3, "Re\tnt")])],
      ],
      [
        6,
        6002,
        rowInfo(5, [
          cell(3, 0x8, [7, 0, 0, 0]),
          cell(2, 0x3, [...decimal, ...float64(0)]),
          cell(5, 0x6, [...float64(7), ...float64(86400)]),
          null,
        ]),
      ],
      [
        7,
        6002,
        rowInfo(0, [null, cell(6, 0x3, [...Array(16).fill(0), ...float64(1)])]),
      ],
    ]),
  );
  assert.deepEqual(content, {
    label: "NUMBERS",
    text: "[Sheet: Budget]\n[Table: Costs]\nRe nt\t-12345678901234567.89\t2001-01-02 00:00:00\n\tTRUE",
  });
});

test("a row repeating one long blank string spends the text budget, after the rows before it", async () => {
  const string = (id: number) => cell(3, 0x8, [id, 0, 0, 0]);
  const tiles = bytes(3, bytes(1, [...int(1, 0), ...ref(2, 6)]));
  const entry = (key: number, text: string) =>
    bytes(3, [...int(1, key), ...bytes(3, text)]);
  const content = await readIwork(
    "long.numbers",
    iwa([
      [1, 1, ref(1, 2)],
      [2, 2, [...bytes(1, "Sheet"), ...ref(2, 4)]],
      [
        4,
        6001,
        [...int(6, 3), ...int(7, 1024), ...bytes(4, [...tiles, ...ref(4, 5)])],
      ],
      [
        5,
        6005,
        [
          ...int(1, 1),
          ...entry(7, " ".repeat(2 * 1024 * 1024)),
          ...entry(8, "first"),
          ...entry(9, "after"),
        ],
      ],
      [
        6,
        6002,
        [
          ...rowInfo(2, [string(9)]),
          ...rowInfo(1, Array(1024).fill(string(7))),
          ...rowInfo(0, [string(8)]),
        ],
      ],
    ]),
  );
  assert.equal(
    content.text,
    "[Sheet: Sheet]\n[Table: ]\nfirst\n\n[Truncated: the spreadsheet has more text than one attachment carries]",
  );
});

test("an iWork reader refuses old and oversized files", async () => {
  await assert.rejects(
    readIworkAttachmentContent(
      new File([zipSync({ "index.xml.gz": strToU8("x") })], "old.pages"),
      "old.pages",
    ),
    /iWork 2013 or later/,
  );
  const huge = [...varint(200 * 1024 * 1024), 0, 0];
  await assert.rejects(
    readIwork("huge.pages", new Uint8Array([0, huge.length, 0, 0, ...huge])),
    /unpacks too large/,
  );
  // Empty records hold no bytes but still cost memory to keep.
  const records = Array.from({ length: 450_000 }, (_, index) => [
    index + 1,
    1,
    [],
  ]) as [number, number, Bytes][];
  await assert.rejects(readIwork("flood.pages", iwa(records)), /unpacks too large/);
});
