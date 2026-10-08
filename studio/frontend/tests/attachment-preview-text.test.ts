// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { DOMParser as XmlDomParser, XMLSerializer as XmlSerializer } from "@xmldom/xmldom";
import {
  Unzip,
  UnzipInflate,
  strFromU8,
  strToU8,
  unzipSync,
  zipSync,
} from "fflate";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  attachmentAudioSrc,
  attachmentTextLanguage,
  countAttachmentTextLines,
  decodeHtmlAttachmentBytes,
  extractHtmlAttachmentText,
  extractPdfAttachmentText,
  getDocxAttachmentError,
  getPdfAttachmentTextError,
  isAudioAttachment,
  isTextAttachment,
  linearizeDocxMath,
  markDocxNotes,
  parseAttachmentText,
  readAttachmentText,
  repackDocxAttachmentArchive,
  repackDocxPreviewArchive,
  truncateAttachmentPreviewText,
  writeDocxBreaksAndCheckboxes,
  writeDocxTableRows,
} = await import("../src/features/chat/attachment-content.ts");
const { definePDFJSModule } = await import("unpdf");
const { readRtfAttachmentContent } =
  await import("../src/features/chat/rtf.ts");

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

test("extractHtmlAttachmentText keeps the indentation of preformatted code", async () => {
  const extracted = await withStubDom(
    () =>
      element(
        "body",
        element("h1", textNode("Totals")),
        element(
          "pre",
          textNode("def total(items):\n    s = 0\n"),
          element("span", textNode("    for x in items:\n        s += x\n")),
          textNode("    return s\n"),
        ),
        element("p", textNode("Done  here")),
      ),
    () => extractHtmlAttachmentText("<html/>"),
  );

  assert.equal(
    extracted,
    "Totals\n\ndef total(items):\n    s = 0\n    for x in items:\n        s += x\n    return s\n\nDone here",
  );
});

test("extractHtmlAttachmentText adds no blank lines for an empty preformatted block", async () => {
  const between = await withStubDom(
    () =>
      element(
        "body",
        element("p", textNode("X")),
        element("pre", textNode("  \n  ")),
        element("p", textNode("Y")),
      ),
    () => extractHtmlAttachmentText("<html/>"),
  );
  const last = await withStubDom(
    () =>
      element(
        "body",
        element("p", textNode("X")),
        element("pre", textNode("\n")),
      ),
    () => extractHtmlAttachmentText("<html/>"),
  );

  assert.equal(between, "X\n\nY");
  assert.equal(last, "X");
});

test("extractHtmlAttachmentText drops blank lines leading a preformatted block", async () => {
  const extracted = await withStubDom(
    () =>
      element(
        "body",
        element("pre", textNode("\n  \n  first\n    second\n")),
        element("p", textNode("Z")),
      ),
    () => extractHtmlAttachmentText("<html/>"),
  );

  assert.equal(extracted, "  first\n    second\n\nZ");
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
      getFieldObjects: async () => null,
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

function singlePagePdf(
  content: string,
  resources: string,
  extra: string[],
  pageEntries = "",
  catalogEntries = "",
) {
  const objects = [
    `<< /Type /Catalog /Pages 2 0 R ${catalogEntries}>>`,
    "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources ${resources} /Contents 4 0 R ${pageEntries}>>`,
    `<< /Length ${content.length} >>\nstream\n${content}\nendstream`,
    ...extra,
  ];
  return pdfBytes(objects);
}

function pdfBytes(objects: string[]) {
  let body = "%PDF-1.4\n";
  const offsets = objects.map((object, index) => {
    const offset = body.length;
    body += `${index + 1} 0 obj\n${object}\nendobj\n`;
    return offset;
  });
  const xref = body.length;
  body += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  body += offsets
    .map((offset) => `${String(offset).padStart(10, "0")} 00000 n \n`)
    .join("");
  body += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  return Uint8Array.from(body, (char) => char.charCodeAt(0));
}

test("a scanned pdf is refused unless the python tool can open it", async () => {
  const pixels = "\x80".repeat(4);
  const scan = new File(
    [
      singlePagePdf(
        "q 200 0 0 200 0 0 cm /Im1 Do Q",
        "<< /XObject << /Im1 5 0 R >> >>",
        [
          `<< /Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceGray /BitsPerComponent 8 /Length ${pixels.length} >>\nstream\n${pixels}\nendstream`,
        ],
      ),
    ],
    "scan.pdf",
    { type: "application/pdf" },
  );
  const typed = new File(
    [
      singlePagePdf(
        "BT /F1 12 Tf 20 100 Td (quokka invoice) Tj ET",
        "<< /Font << /F1 5 0 R >> >>",
        ["<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"],
      ),
    ],
    "typed.pdf",
    { type: "application/pdf" },
  );

  const scanText = await extractPdfAttachmentText(scan);
  const typedText = await extractPdfAttachmentText(typed);
  assert.equal(scanText, "");
  assert.equal(typedText, "quokka invoice");
  assert.equal(
    (await readAttachmentText(scan, scan.name, scan.type)).text,
    "",
  );

  assert.match(
    getPdfAttachmentTextError(scan.name, scanText, false) ?? "",
    /^PDF has no readable text: scan\.pdf\./,
  );
  assert.equal(getPdfAttachmentTextError(scan.name, scanText, true), null);
  assert.equal(getPdfAttachmentTextError(typed.name, typedText, false), null);
});

test("a filled pdf form keeps the values typed into its fields", async () => {
  const font = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>";
  const field = (name: string, value: string, y: number) =>
    `<< /Type /Annot /Subtype /Widget /FT /Tx /T (${name}) /V (${value}) /Rect [60 ${y} 190 ${y + 20}] /P 3 0 R >>`;
  const checkbox = (name: string, state: string, y: number) =>
    `<< /Type /Annot /Subtype /Widget /FT /Btn /T (${name}) /V /${state} /AS /${state} /AP << /N << /Yes 10 0 R /Off 11 0 R >> >> /Rect [60 ${y} 70 ${y + 10}] /P 3 0 R >>`;
  const blank = "<< /Length 0 >>\nstream\n\nendstream";
  const filled = new File(
    [
      singlePagePdf(
        "BT /F1 12 Tf 20 100 Td (Name:) Tj ET",
        "<< /Font << /F1 5 0 R >> >>",
        [
          font,
          field("name", "Oscar Papa Quebec", 95),
          field("notes", "", 60),
          checkbox("agree", "Yes", 30),
          checkbox("newsletter", "Off", 10),
          blank,
          blank,
          `<< /Type /Annot /Subtype /Widget /FT /Tx /F 2 /T (internal) /V (hidden) /Rect [0 0 1 1] /P 3 0 R >>`,
          `<< /Type /Annot /Subtype /Widget /FT /Ch /Ff 131072 /T (status) /TU (Marital\\nstatus) /Opt [[(1) (Single)] [(2) (Married)]] /V (2) /Rect [60 150 190 170] /P 3 0 R >>`,
          `<< /Type /Annot /Subtype /Widget /FT /Tx /Ff 8192 /T (pin) /V (4321) /Rect [60 175 190 195] /P 3 0 R >>`,
        ],
        "/Annots [6 0 R 7 0 R 8 0 R 9 0 R 12 0 R 13 0 R 14 0 R] ",
        "/AcroForm << /Fields [6 0 R 7 0 R 8 0 R 9 0 R 12 0 R 13 0 R 14 0 R] >> ",
      ),
    ],
    "filled.pdf",
    { type: "application/pdf" },
  );
  const fieldsOnly = new File(
    [
      singlePagePdf(
        "",
        "<< >>",
        [field("name", "Romeo Sierra", 95)],
        "/Annots [5 0 R] ",
        "/AcroForm << /Fields [5 0 R] >> ",
      ),
    ],
    "fields-only.pdf",
    { type: "application/pdf" },
  );

  assert.equal(
    await extractPdfAttachmentText(filled),
    "Name:\nname: Oscar Papa Quebec\nagree: Yes\nMarital status: Married",
  );
  const fieldsOnlyText = await extractPdfAttachmentText(fieldsOnly);
  assert.equal(fieldsOnlyText, "name: Romeo Sierra");
  assert.equal(
    getPdfAttachmentTextError(fieldsOnly.name, fieldsOnlyText, false),
    null,
  );
});

test("a multi-page pdf form keeps each page's values with that page", async () => {
  const stream = (content: string) =>
    `<< /Length ${content.length} >>\nstream\n${content}\nendstream`;
  const page = (contents: number, annots: string) =>
    `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << /Font << /F1 10 0 R >> >> /Contents ${contents} 0 R /Annots [${annots}] >>`;
  const option = (exportName: string, tooltip: string, x: number, on: boolean) =>
    `<< /Type /Annot /Subtype /Widget /Parent 6 0 R /TU (${tooltip}) /AS /${on ? exportName : "Off"} /AP << /N << /${exportName} 12 0 R /Off 12 0 R >> >> /Rect [${x} 20 ${x + 10} 30] /P 3 0 R >>`;
  const form = new File(
    [
      pdfBytes([
        "<< /Type /Catalog /Pages 2 0 R /AcroForm << /Fields [6 0 R 9 0 R 13 0 R 14 0 R] >> >>",
        "<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
        page(5, "7 0 R 8 0 R"),
        page(11, "9 0 R 13 0 R 14 0 R"),
        stream("BT /F1 12 Tf 20 100 Td (Filing status) Tj ET"),
        "<< /FT /Btn /Ff 49152 /T (form1[0].c1_1[0]) /V /S /Kids [7 0 R 8 0 R] >>",
        option("S", "Single", 20, true),
        option("M", "Married", 60, false),
        "<< /Type /Annot /Subtype /Widget /FT /Tx /T (form1[0].f2_01[0]) /TU (Date\\nsigned) /V (2026-09-30) /Rect [60 95 190 115] /P 4 0 R >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        stream("BT /F1 12 Tf 20 100 Td (Signature) Tj ET"),
        stream(""),
        "<< /Type /Annot /Subtype /Widget /FT /Ch /Ff 2097152 /T (languages) /Opt [(English) (French) (German)] /V [(English) (German)] /Rect [60 40 190 80] /P 4 0 R >>",
        "<< /Type /Annot /Subtype /Widget /FT /Tx /T (lights) /V (Off) /Rect [60 10 190 30] /P 4 0 R >>",
      ]),
    ],
    "form.pdf",
    { type: "application/pdf" },
  );

  assert.equal(
    await extractPdfAttachmentText(form),
    "Filing status\nform1[0].c1_1[0]: S\n\nSignature\nDate signed: 2026-09-30\nlanguages: English, German\nlights: Off",
  );
});

test("a pdf with a damaged form field still reads its text", async () => {
  const damaged = new File(
    [
      singlePagePdf(
        "BT /F1 12 Tf 20 100 Td (Budget: 4200) Tj ET",
        "<< /Font << /F1 5 0 R >> >>",
        [
          "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
          "<< /Type /Annot /Subtype /Widget /FT /Tx /T null /V (x) /Rect [60 20 190 40] >>",
        ],
        "/Annots [6 0 R] ",
        "/AcroForm << /Fields [6 0 R] >> ",
      ),
    ],
    "damaged.pdf",
    { type: "application/pdf" },
  );

  assert.equal(await extractPdfAttachmentText(damaged), "Budget: 4200");
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

test("markDocxNotes numbers the references extractRawText keeps and marks the body", async () => {
  const w = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"';
  const note = (kind: string, id: number, text: string) =>
    `<w:${kind} w:id="${id}"><w:p><w:r><w:${kind}Ref/></w:r><w:r><w:t xml:space="preserve"> ${text}</w:t></w:r></w:p></w:${kind}>`;
  const notes = (kind: string, body: string) =>
    strToU8(
      `<w:${kind}s ${w}>` +
        `<w:${kind} w:type="separator" w:id="-1"><w:p><w:r><w:separator/></w:r></w:p></w:${kind}>` +
        `<w:${kind} w:type="continuationSeparator" w:id="0"><w:p><w:r><w:continuationSeparator/></w:r></w:p></w:${kind}>` +
        body +
        `</w:${kind}s>`,
    );
  const ref = (kind: string, id: number) =>
    kind === "endnote"
      ? `<w:r><w:${kind}Reference w:id="${id}">\n</w:${kind}Reference >\n</w:r>`
      : `<w:r><w:${kind}Reference w:id="${id}"/></w:r>`;
  const archive = repackDocxAttachmentArchive(
    "paper.docx",
    zipSync({
      "[Content_Types].xml": strToU8("<Types/>"),
      "_rels/.rels": relationships([["officeDocument", "word/document.xml"]]),
      "word/document.xml": strToU8(
        `<w:document ${w}><w:body><w:p><w:del w:id="9">${ref("footnote", 4)}</w:del>` +
          `<w:r><w:t>First.</w:t></w:r>${ref("footnote", 2)}` +
          `<w:r><w:t> Second.</w:t></w:r>${ref("footnote", 1)}${ref("endnote", 1)}` +
          `<w:r><w:t xml:space="preserve"> \uE0007\uE001</w:t></w:r>` +
          `<w:r><x:footnoteReference xmlns:x="http://schemas.openxmlformats.org/wordprocessingml/2006/main" x:id="3"/></w:r>` +
          `</w:p></w:body></w:document>`,
      ),
      "word/_rels/document.xml.rels": relationships([
        ["footnotes", "notes/foot.xml"],
        ["endnotes", "endnotes.xml"],
      ]),
      "word/notes/foot.xml": notes(
        "footnote",
        note("footnote", 1, "Source: LATER") +
          note("footnote", 2, "Source: EARLIER") +
          note("footnote", 3, "Source: LOCALLY DECLARED") +
          note("footnote", 4, "Source: DELETED"),
      ),
      "word/endnotes.xml": notes("endnote", note("endnote", 1, "Source: ENDNOTEBODY")),
    }),
  );
  const original = (globalThis as { DOMParser?: unknown }).DOMParser;
  (globalThis as { DOMParser?: unknown }).DOMParser = XmlDomParser;
  try {
    const marked = markDocxNotes(archive);
    const { default: mammoth } = await import("mammoth");
    const { value } = await mammoth.extractRawText({
      buffer: Buffer.from(marked.archive),
    });
    assert.equal(
      marked.label(value),
      "First.[1] Second.[2][i] \uE0007\uE001[3]\n\n" +
        "Footnotes\n[1] Source: EARLIER\n[2] Source: LATER\n[3] Source: LOCALLY DECLARED\n\n" +
        "Endnotes\n[i] Source: ENDNOTEBODY",
    );
  } finally {
    (globalThis as { DOMParser?: unknown }).DOMParser = original;
  }
});

test("linearizeDocxMath keeps equations in the body, tables and notes", async () => {
  const ns =
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" ' +
    'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"';
  const run = (text: string) => `<w:r><w:t xml:space="preserve">${text}</w:t></w:r>`;
  const m = (text: string) => `<m:r><m:t>${text}</m:t></m:r>`;
  const half = `<m:f><m:num>${m("1")}</m:num><m:den>${m("2")}</m:den></m:f>`;
  const squared = `<m:sSup><m:e>${m("v")}</m:e><m:sup>${m("2")}</m:sup></m:sSup>`;
  const mean = `<m:bar><m:barPr><m:pos m:val="top"/></m:barPr><m:e>${m("x")}</m:e></m:bar>`;
  const root = `<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>${m("n")}</m:e></m:rad>`;
  const archive = repackDocxAttachmentArchive(
    "physics.docx",
    zipSync({
      "[Content_Types].xml": strToU8("<Types/>"),
      "_rels/.rels": relationships([["officeDocument", "word/document.xml"]]),
      "word/document.xml": strToU8(
        `<w:document ${ns}><w:body>` +
          `<w:p>${run("The kinetic energy is ")}<m:oMath>${m("E=")}${half}${m("m")}${squared}</m:oMath>${run(" joules.")}` +
          `<w:r><w:footnoteReference w:id="1"/></w:r></w:p>` +
          `<w:p><m:oMathPara><m:oMath>${m("F=ma")}</m:oMath><m:oMath>${m("p=mv")}</m:oMath></m:oMathPara></w:p>` +
          `<w:tbl><w:tr><w:tc><w:p>${run("Error")}</w:p></w:tc><w:tc><w:p><m:oMath>` +
          `<m:d><m:e>${m("a+b")}</m:e></m:d><w:del w:id="2" w:author="a">${m("+c")}</w:del>${root}` +
          `</m:oMath></w:p></w:tc></w:tr></w:tbl>` +
          `<w:p><w:del w:id="3" w:author="a"><m:oMath>${m("gone")}</m:oMath></w:del></w:p>` +
          `</w:body></w:document>`,
      ),
      "word/_rels/document.xml.rels": relationships([["footnotes", "footnotes.xml"]]),
      "word/footnotes.xml": strToU8(
        `<w:footnotes ${ns}><w:footnote w:id="1"><w:p>${run("Where ")}<m:oMath>${mean}</m:oMath>${run(" is the mean.")}</w:p></w:footnote></w:footnotes>`,
      ),
    }),
  );
  const globals = globalThis as { DOMParser?: unknown; XMLSerializer?: unknown };
  const original = { DOMParser: globals.DOMParser, XMLSerializer: globals.XMLSerializer };
  globals.DOMParser = XmlDomParser;
  globals.XMLSerializer = XmlSerializer;
  try {
    const marked = markDocxNotes(linearizeDocxMath(archive));
    const { default: mammoth } = await import("mammoth");
    const { value } = await mammoth.extractRawText({ buffer: Buffer.from(marked.archive) });
    assert.equal(
      marked.label(value),
      "The kinetic energy is E=\\frac{1}{2}mv^{2} joules.[1]\n\nF=ma\np=mv\n\nError\n\n(a+b)\\sqrt{n}\n\n\n\n" +
        "Footnotes\n[1] Where \\overline{x} is the mean.",
    );
  } finally {
    Object.assign(globals, original);
  }
});

// Same cases and expected text as the backend reader's test_docx_equation_structures.
const OMML_CASES: [string, string][] = [
  [
    '<m:nary><m:naryPr><m:chr m:val="∑"/></m:naryPr><m:sub>{i=1}</m:sub><m:sup>{n}</m:sup><m:e>{i}</m:e></m:nary>',
    "∑_{i=1}^{n}i",
  ],
  ["<m:nary><m:sub>{0}</m:sub><m:sup>{1}</m:sup><m:e>{x}</m:e></m:nary>", "∫_{0}^{1}x"],
  [
    "<m:sSubSup><m:e>{x}</m:e><m:sub>{i}</m:sub><m:sup>{2}</m:sup></m:sSubSup><m:sSub><m:e>{a}</m:e><m:sub>{0}</m:sub></m:sSub>",
    "x_{i}^{2}a_{0}",
  ],
  ["<m:sPre><m:sub>{6}</m:sub><m:sup>{14}</m:sup><m:e>{C}</m:e></m:sPre>", "{}_{6}^{14}C"],
  ["<m:limUpp><m:e>{x}</m:e><m:lim>{def}</m:lim></m:limUpp>", "x^{def}"],
  ["<m:limLow><m:e>{lim}</m:e><m:lim>{n→∞}</m:lim></m:limLow>", "lim_{n→∞}"],
  [
    '<m:f><m:fPr><m:type m:val="noBar"/></m:fPr><m:num>{n}</m:num><m:den>{k}</m:den></m:f>' +
      '<m:phant><m:phantPr><m:show m:val="off"/></m:phantPr><m:e>{xyz}</m:e></m:phant>',
    "{n \\atop k}",
  ],
  ["<m:acc><m:e>{θ}</m:e></m:acc><m:rad><m:deg>{3}</m:deg><m:e>{y}</m:e></m:rad>", "θ̂\\sqrt[3]{y}"],
  [
    '<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg>{3}</m:deg><m:e>{x}</m:e></m:rad>' +
      '<m:nary><m:naryPr><m:chr m:val="∑"/><m:subHide m:val="0"/><m:supHide/></m:naryPr><m:sub>{k}</m:sub><m:sup>{n}</m:sup><m:e>{a}</m:e></m:nary>',
    "\\sqrt{x}∑_{k}a",
  ],
  ["<m:func><m:fName>{sin}</m:fName><m:e>{x}</m:e></m:func>", "sin x"],
  [
    "<m:m><m:mr><m:e>{a}</m:e><m:e>{b}</m:e></m:mr><m:mr><m:e>{c}</m:e><m:e>{d}</m:e></m:mr></m:m>",
    "a & b \\\\ c & d",
  ],
  ["<m:eqArr><m:e>{x=1}</m:e><m:e>{y=2}</m:e></m:eqArr>", "x=1\ny=2"],
  [
    '<m:d><m:e>{a}</m:e><m:e>{b}</m:e></m:d><m:d><m:dPr><m:begChr m:val="["/><m:endChr m:val=""/></m:dPr><m:e>{c}</m:e></m:d>',
    "(a|b)[c",
  ],
  [
    '<m:bar><m:e>{x}</m:e></m:bar><m:groupChr><m:groupChrPr><m:chr m:val="⏞"/><m:pos m:val="top"/></m:groupChrPr><m:e>{y}</m:e></m:groupChr>' +
      '<m:groupChr><m:groupChrPr><m:chr m:val="←"/></m:groupChrPr><m:e>{z}</m:e></m:groupChr>',
    "\\underline{x}\\overbrace{y}\\underset{←}{z}",
  ],
  [
    '{a}<w:r><w:t xml:space="preserve"> if </w:t></w:r>' +
      "<w:sdt><w:sdtPr><w:showingPlcHdr/></w:sdtPr><w:sdtContent>{prompt}</w:sdtContent></w:sdt>{b}",
    "a if b",
  ],
];

test("linearizeDocxMath writes equations as the backend does and leaves other parts untouched", () => {
  const ns =
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" ' +
    'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"';
  const archive = (body: string) =>
    zipSync({ "word/document.xml": strToU8(`<w:document ${ns}><w:body>${body}</w:body></w:document>`) });
  const globals = globalThis as { DOMParser?: unknown; XMLSerializer?: unknown };
  const original = { DOMParser: globals.DOMParser, XMLSerializer: globals.XMLSerializer };
  globals.DOMParser = XmlDomParser;
  globals.XMLSerializer = XmlSerializer;
  try {
    const plain = archive('<w:p><w:r><w:t>oMath is only a word here</w:t></w:r></w:p>');
    assert.equal(linearizeDocxMath(plain), plain);
    // How Chromium's DOMParser returns a part cut short at an XML error.
    const truncated = archive(
      '<parsererror xmlns="http://www.w3.org/1999/xhtml"/><w:p><m:oMath><m:r><m:t>x</m:t></m:r></m:oMath></w:p>',
    );
    assert.equal(linearizeDocxMath(truncated), truncated);
    const strict = zipSync({
      "word/document.xml": strToU8(
        '<w:document xmlns:w="http://purl.oclc.org/ooxml/wordprocessingml/main" xmlns:m="http://purl.oclc.org/ooxml/officeDocument/math">' +
          "<w:body><w:p><m:oMath><m:sSup><m:e><m:r><m:t>x</m:t></m:r></m:e><m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSup></m:oMath></w:p></w:body></w:document>",
      ),
    });
    assert.ok(strFromU8(unzipSync(linearizeDocxMath(strict))["word/document.xml"]).includes(">x^{2}<"));
    for (const [omml, expected] of OMML_CASES) {
      const filled = omml.replace(/\{([^{}]*)\}/g, (_, text: string) => `<m:r><m:t>${text}</m:t></m:r>`);
      const xml = strFromU8(unzipSync(linearizeDocxMath(archive(`<w:p><m:oMath>${filled}</m:oMath></w:p>`)))["word/document.xml"]);
      const doc = new XmlDomParser().parseFromString(xml, "application/xml");
      assert.equal(doc.getElementsByTagNameNS("http://schemas.openxmlformats.org/wordprocessingml/2006/main", "t")[0]?.textContent, expected);
    }
  } finally {
    Object.assign(globals, original);
  }
});

test("markDocxNotes reads Strict OOXML notes and skips unfilled content controls", () => {
  const w = 'xmlns:w="http://purl.oclc.org/ooxml/wordprocessingml/main"';
  const run = (text: string) => `<w:r><w:t xml:space="preserve">${text}</w:t></w:r>`;
  const archive = repackDocxAttachmentArchive(
    "strict.docx",
    zipSync({
      "[Content_Types].xml": strToU8("<Types/>"),
      "_rels/.rels": relationships([["officeDocument", "word/document.xml"]]),
      "word/document.xml": strToU8(`<w:document ${w}><w:body/></w:document>`),
      "word/footnotes.xml": strToU8(
        `<w:footnotes ${w}><w:footnote w:id="1"><w:p>` +
          run("Keep") +
          `<w:sdt><w:sdtPr><w:showingPlcHdr/></w:sdtPr><w:sdtContent>${run(" Click or tap here to enter text.")}</w:sdtContent></w:sdt>` +
          `<w:sdt><w:sdtPr><w:showingPlcHdr w:val="0"/></w:sdtPr><w:sdtContent>${run(" FILLED")}</w:sdtContent></w:sdt>` +
          "</w:p></w:footnote></w:footnotes>",
      ),
    }),
  );
  const original = (globalThis as { DOMParser?: unknown }).DOMParser;
  (globalThis as { DOMParser?: unknown }).DOMParser = XmlDomParser;
  try {
    assert.equal(markDocxNotes(archive).label(""), "Footnotes\n[1] Keep FILLED");
  } finally {
    (globalThis as { DOMParser?: unknown }).DOMParser = original;
  }
});

test("markDocxNotes skips move sources, deletions and text box fallbacks", () => {
  const w = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"';
  const mc = 'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"';
  const run = (text: string) => `<w:r><w:t xml:space="preserve">${text}</w:t></w:r>`;
  const box = `<w:txbxContent><w:p>${run("BOX")}</w:p></w:txbxContent>`;
  const archive = repackDocxAttachmentArchive(
    "moved.docx",
    zipSync({
      "[Content_Types].xml": strToU8("<Types/>"),
      "_rels/.rels": relationships([["officeDocument", "word/document.xml"]]),
      "word/document.xml": strToU8(`<w:document ${w}><w:body/></w:document>`),
      "word/_rels/document.xml.rels": relationships([["footnotes", "footnotes.xml"]]),
      "word/footnotes.xml": strToU8(
        `<w:footnotes ${w} ${mc}><w:footnote w:id="1"><w:p>` +
          run("Keep") +
          `<w:moveFrom w:id="7">${run(" MOVED")}</w:moveFrom>` +
          `<w:del w:id="8"><w:r><w:delText> GONE</w:delText></w:r></w:del>` +
          run(" COVID") + "<w:r><w:noBreakHyphen/></w:r>" + run("19") +
          `<w:moveTo w:id="9">${run(" MOVED")}</w:moveTo>` +
          `<w:r><mc:AlternateContent><mc:Choice Requires="wps">${box}</mc:Choice>` +
          `<mc:Fallback>${box}</mc:Fallback></mc:AlternateContent></w:r>` +
          "</w:p></w:footnote></w:footnotes>",
      ),
    }),
  );
  const original = (globalThis as { DOMParser?: unknown }).DOMParser;
  (globalThis as { DOMParser?: unknown }).DOMParser = XmlDomParser;
  try {
    assert.equal(
      markDocxNotes(archive).label(""),
      "Footnotes\n[1] Keep COVID-19 MOVED BOX",
    );
  } finally {
    (globalThis as { DOMParser?: unknown }).DOMParser = original;
  }
});

test("a Word file keeps its line breaks and which boxes are ticked", async () => {
  const ns =
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" ' +
    'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"';
  const run = (text: string) => `<w:r><w:t xml:space="preserve">${text}</w:t></w:r>`;
  const box = (checked: string, glyph: string, label: string) =>
    `<w:p><w:sdt><w:sdtPr><w14:checkbox><w14:checked w14:val="${checked}"/>` +
    '<w14:checkedState w14:val="2612" w14:font="MS Gothic"/><w14:uncheckedState w14:val="2610" w14:font="MS Gothic"/>' +
    `</w14:checkbox></w:sdtPr><w:sdtContent>${run(glyph)}</w:sdtContent></w:sdt>${run(label)}</w:p>`;
  const field = (state: string, label: string) =>
    `<w:p><w:r><w:fldChar w:fldCharType="begin"><w:ffData><w:checkBox><w:sizeAuto/>${state}</w:checkBox></w:ffData></w:fldChar></w:r>` +
    '<w:r><w:instrText xml:space="preserve"> FORMCHECKBOX </w:instrText></w:r>' +
    `<w:r><w:fldChar w:fldCharType="end"/></w:r>${run(label)}</w:p>`;
  const archive = docxBytes(
    `<w:document ${ns}><w:body>` +
      '<w:p><w:r><w:t>Jane Doe</w:t><w:br/><w:t>42 Elm Street</w:t></w:r><w:r><w:br w:type="textWrapping"/></w:r>' +
      `${run("Springfield, IL 62704")}</w:p>` +
      '<w:p><w:r><w:t>Summary</w:t><w:br w:type="page"/><w:t>Details</w:t></w:r></w:p>' +
      `<w:tbl><w:tr><w:tc><w:p>${run("Built APIs")}<w:r><w:cr/><w:t>Led team of 5</w:t></w:r></w:p></w:tc></w:tr></w:tbl>` +
      box("1", "☒", " Smoker") +
      box("0", "☐", " Diabetic") +
      field('<w:default w:val="0"/><w:checked/>', " Allergies") +
      field('<w:default w:val="0"/>', " Pregnant") +
      '<w:p><w:r><w:t xml:space="preserve">Consent </w:t><w:fldChar w:fldCharType="begin"><w:ffData><w:checkBox><w:checked/></w:checkBox></w:ffData></w:fldChar></w:r>' +
      '<w:r><w:instrText xml:space="preserve"> FORMCHECKBOX </w:instrText></w:r>' +
      `<w:r><w:fldChar w:fldCharType="end"/></w:r>${run(" given")}</w:p>` +
      "</w:body></w:document>",
  );
  const globals = globalThis as { DOMParser?: unknown; XMLSerializer?: unknown };
  const original = { DOMParser: globals.DOMParser, XMLSerializer: globals.XMLSerializer };
  globals.DOMParser = XmlDomParser;
  globals.XMLSerializer = XmlSerializer;
  try {
    const { default: mammoth } = await import("mammoth");
    const { value } = await mammoth.extractRawText({
      buffer: Buffer.from(writeDocxBreaksAndCheckboxes(archive)),
    });
    assert.equal(
      value,
      "Jane Doe\n42 Elm Street\nSpringfield, IL 62704\n\n" +
        "Summary\nDetails\n\n" +
        "Built APIs\nLed team of 5\n\n" +
        "☒ Smoker\n\n☐ Diabetic\n\n☒ Allergies\n\n☐ Pregnant\n\nConsent ☒ given\n\n",
    );
  } finally {
    Object.assign(globals, original);
  }
});

test("a Word table keeps its columns when a cell is empty", async () => {
  const ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"';
  const cell = (...paragraphs: string[]) =>
    `<w:tc>${paragraphs.map((text) => `<w:p><w:r><w:t>${text}</w:t></w:r></w:p>`).join("") || "<w:p/>"}</w:tc>`;
  const row = (...cells: string[]) => `<w:tr>${cells.join("")}</w:tr>`;
  const bytes = docxBytes(
    `<w:document ${ns}><w:body><w:p><w:r><w:t>Timetable</w:t></w:r></w:p><w:tbl>` +
      row('<w:tc><w:tcPr><w:gridSpan w:val="6"/></w:tcPr><w:p><w:r><w:t>Week 1</w:t></w:r></w:p></w:tc>') +
      row(cell("Time"), cell("Mon"), cell("Tue"), cell("Wed"), cell("Thu"), cell("Fri")) +
      row(cell("10:00"), cell("History"), cell(), cell("Maths"), cell("Art"), cell()) +
      row(
        cell("11:00"),
        '<w:tc><w:tcPr><w:gridSpan w:val="2"/></w:tcPr><w:p><w:r><w:t>Trip</w:t></w:r></w:p></w:tc>',
        cell("Music", "Room 4"),
        cell(),
        cell("PE"),
      ) +
      "</w:tbl></w:body></w:document>",
  );
  const globals = globalThis as { DOMParser?: unknown; XMLSerializer?: unknown };
  const original = { DOMParser: globals.DOMParser, XMLSerializer: globals.XMLSerializer };
  globals.DOMParser = XmlDomParser;
  globals.XMLSerializer = XmlSerializer;
  try {
    const { default: mammoth } = await import("mammoth");
    const { value } = await mammoth.extractRawText({
      buffer: Buffer.from(writeDocxTableRows(writeDocxBreaksAndCheckboxes(bytes))),
    });
    assert.equal(
      value,
      "Timetable\n\nWeek 1\n\n" +
        "Time\tMon\tTue\tWed\tThu\tFri\n\n" +
        "10:00\tHistory\t\tMaths\tArt\t\n\n" +
        "11:00\tTrip\t\tMusic Room 4\t\tPE\n\n",
    );
  } finally {
    Object.assign(globals, original);
  }
});

test("an html table keeps its columns when a cell is empty", async () => {
  const cell = (tag: string, text = "", colspan?: string) =>
    Object.assign(element(tag, ...(text ? [textNode(text)] : [])), {
      getAttribute: (name: string) => (name === "colspan" ? (colspan ?? null) : null),
    });
  const extracted = await withStubDom(
    () =>
      element(
        "body",
        element("h2", textNode("Timetable")),
        element(
          "table",
          element(
            "tbody",
            element("tr", cell("th", "Time"), cell("th", "Mon"), cell("th", "Tue"), cell("th", "Wed")),
            textNode("\n    "),
            element("tr", cell("td", "10:00"), cell("td", "History"), cell("td"), cell("td", "Maths")),
            element(
              "tr",
              cell("td", "11:00"),
              cell("td", "Trip", "2"),
              Object.assign(element("td", element("p", textNode("Art")), element("p", textNode("Room  4"))), {
                getAttribute: () => null,
              }),
            ),
          ),
        ),
        element("p", textNode("Bring a  pencil")),
      ),
    () => extractHtmlAttachmentText("<html/>"),
  );

  assert.equal(
    extracted,
    "Timetable\n\nTime\tMon\tTue\tWed\n\n10:00\tHistory\t\tMaths\n\n11:00\tTrip\t\tArt Room 4\n\nBring a pencil",
  );
});

test("a Word row keeps its columns past skipped grid cells, tabs, line breaks and nested tables", async () => {
  const ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"';
  const p = (text: string) => `<w:p><w:r><w:t>${text}</w:t></w:r></w:p>`;
  const row = (cells: string, pr = "") => `<w:tr>${pr}${cells}</w:tr>`;
  const nested = `<w:tbl>${row(`<w:tc>${p("Room")}</w:tc><w:tc>${p("4")}</w:tc>`)}</w:tbl>`;
  const bytes = docxBytes(
    `<w:document ${ns}><w:body><w:tbl>` +
      row(`<w:tc>${p("Time")}</w:tc><w:tc>${p("Mon")}</w:tc><w:tc>${p("Tue")}</w:tc>`) +
      row(`<w:tc>${p("Lab 3")}</w:tc><w:tc>${p("Lab 4")}</w:tc>`, '<w:trPr><w:gridBefore w:val="1"/></w:trPr>') +
      row(`<w:tc>${p("Only Tue")}</w:tc>`, '<w:trPr><w:gridBefore w:val="2"/></w:trPr>') +
      row(
        `<w:tc>${p("10:00")}</w:tc>` +
          `<w:tc><w:p><w:r><w:t>A</w:t><w:tab/><w:t>B</w:t></w:r></w:p></w:tc>` +
          `<w:tc>${nested}<w:p/></w:tc>`,
      ) +
      row(`<w:tc>${p("11:00")}</w:tc><w:tc><w:p><w:r><w:t>Line one</w:t><w:br/><w:t>Line two</w:t></w:r></w:p></w:tc><w:tc>${p("C")}</w:tc>`) +
      "</w:tbl></w:body></w:document>",
  );
  const globals = globalThis as { DOMParser?: unknown; XMLSerializer?: unknown };
  const original = { DOMParser: globals.DOMParser, XMLSerializer: globals.XMLSerializer };
  globals.DOMParser = XmlDomParser;
  globals.XMLSerializer = XmlSerializer;
  try {
    const { default: mammoth } = await import("mammoth");
    const { value } = await mammoth.extractRawText({
      buffer: Buffer.from(writeDocxTableRows(writeDocxBreaksAndCheckboxes(bytes))),
    });
    assert.equal(
      value,
      "Time\tMon\tTue\n\n\tLab 3\tLab 4\n\n\t\tOnly Tue\n\n10:00\tA B\tRoom 4 \n\n11:00\tLine one Line two\tC\n\n",
    );
  } finally {
    Object.assign(globals, original);
  }
});

test("an html table keeps its columns under a rowspan and leaves code in a cell alone", async () => {
  const cell = (tag: string, text = "", attributes: Record<string, string> = {}) =>
    Object.assign(element(tag, ...(text ? [textNode(text)] : [])), {
      getAttribute: (name: string) => attributes[name] ?? null,
    });
  const extracted = await withStubDom(
    () =>
      element(
        "body",
        element(
          "table",
          element(
            "tbody",
            element("tr", cell("th", "Time"), cell("th", "Mon"), cell("th", "Tue"), cell("th", "Wed")),
            element("tr", cell("td", "09:00"), cell("td", "Science", { rowspan: "2" }), cell("td", "Art"), cell("td", "PE")),
            element("tr", cell("td", "10:00"), cell("td", "Maths"), cell("td", "French")),
            element("tr", cell("td", "11:00"), cell("td", "Music", { colspan: "3" })),
            element("tr", cell("td", "ship()"), Object.assign(element("td", element("pre", textNode("def ship():\n    return 1"))), {
              getAttribute: () => null,
            })),
          ),
        ),
      ),
    () => extractHtmlAttachmentText("<html/>"),
  );

  assert.equal(
    extracted,
    "Time\tMon\tTue\tWed\n\n09:00\tScience\tArt\tPE\n\n10:00\t\tMaths\tFrench\n\n11:00\tMusic\t\t\n\nship()\n\ndef ship():\n    return 1",
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

function legacyPage(head: string, body: number[]) {
  return new Uint8Array([
    ...new TextEncoder().encode(head),
    ...body,
    ...new TextEncoder().encode("</p>"),
  ]);
}

const WINDOWS_1252_BODY = [
  0x43, 0x61, 0x66, 0xe9, 0x20, 0x80, 0x31, 0x32, 0x2c, 0x20, 0x6e, 0x61, 0xef,
  0x76, 0x65,
];
const SHIFT_JIS_BODY = [
  0x93, 0xfa, 0x96, 0x7b, 0x8c, 0xea, 0x82, 0xcc, 0x83, 0x79, 0x81, 0x5b, 0x83,
  0x57,
];

test("an html attachment is read in the encoding its page declares", async () => {
  const word = legacyPage(
    '<html><head><meta http-equiv=Content-Type content="text/html; charset=windows-1252"></head><p>',
    WINDOWS_1252_BODY,
  );
  const japanese = legacyPage('<meta charset="Shift_JIS"><p>', SHIFT_JIS_BODY);
  for (const [bytes, name, expected] of [
    [word, "report.htm", "Café €12, naïve"],
    [japanese, "page.html", "日本語のページ"],
  ] as const) {
    const file = new File([bytes], name, { type: "text/html" });
    const { text } = await readAttachmentText(file, file.name, file.type);
    assert.ok(text.includes(expected), text);
    assert.ok(!text.includes("\uFFFD"), text);
  }
});

test("decodeHtmlAttachmentBytes reads the charset the way a browser does", () => {
  const utf8 = new TextEncoder().encode(
    "<meta charset=windows-1252><p>Café</p>",
  );
  assert.equal(
    decodeHtmlAttachmentBytes(new Uint8Array([0xef, 0xbb, 0xbf, ...utf8])),
    "<meta charset=windows-1252><p>Café</p>",
  );
  assert.equal(
    decodeHtmlAttachmentBytes(new TextEncoder().encode("<p>Café</p>")),
    "<p>Café</p>",
  );
  for (const head of [
    "<!-- <meta charset=Shift_JIS> --><meta charset=windows-1252><p>",
    '<div title="<meta charset=Shift_JIS>"><meta charset=windows-1252><p>',
  ]) {
    assert.equal(
      decodeHtmlAttachmentBytes(legacyPage(head, WINDOWS_1252_BODY)),
      `${head}Café €12, naïve</p>`,
    );
  }
  assert.equal(
    decodeHtmlAttachmentBytes(
      new TextEncoder().encode('<meta charset="utf-16"><p>Café</p>'),
    ),
    '<meta charset="utf-16"><p>Café</p>',
  );
});

test("a UTF-8 html page keeps its text when its meta names a legacy charset", async () => {
  const page =
    '<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1"><p>Café €12 日本語</p>';
  const file = new File([new TextEncoder().encode(page)], "saved.html", {
    type: "text/html",
  });
  const { text } = await readAttachmentText(file, file.name, file.type);
  assert.equal(text, page);
  assert.equal(
    decodeHtmlAttachmentBytes(new TextEncoder().encode(page).subarray(0, -6), true),
    page.slice(0, -5),
  );
  const jis = legacyPage('<meta charset="iso-2022-jp"><p>', [
    0x1b, 0x24, 0x42, 0x46, 0x7c, 0x4b, 0x5c, 0x38, 0x6c, 0x1b, 0x28, 0x42,
  ]);
  assert.equal(
    decodeHtmlAttachmentBytes(jis),
    '<meta charset="iso-2022-jp"><p>日本語</p>',
  );
  const sjis = '<meta charset="Shift_JIS"><p>日本語のページです</p>';
  assert.equal(decodeHtmlAttachmentBytes(new TextEncoder().encode(sjis)), sjis);
  assert.equal(
    decodeHtmlAttachmentBytes(
      legacyPage('<meta charset="gbk"><p>', [0xd7, 0xa8, 0xd2, 0xb5]),
    ),
    '<meta charset="gbk"><p>专业</p>',
  );
});

test("a UTF-16 Markdown file previews as its text in the document viewer", async () => {
  const utf16 = new Uint8Array([0xff, 0xfe, ...Array.from("# Notes", (c) => [c.charCodeAt(0), 0]).flat()]);
  const file = new File([utf16], "notes.md", { type: "text/markdown" });
  assert.equal((await readAttachmentText(file, file.name, file.type)).text, "# Notes");
  assert.notEqual(await file.text(), "# Notes");
  const { readFile } = await import("node:fs/promises");
  const dialog = await readFile(new URL("../src/components/assistant-ui/attachment-document-dialog.tsx", import.meta.url), "utf8");
  assert.match(dialog, /blob instanceof File\s*\?\s*await readAttachmentText\(blob, source\.name, source\.contentType\)/);
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
  const long = await readRtf(
    `{\\rtf1 ${"x".repeat(11 * 1024 * 1024)}${"{".repeat(2000)}`,
  );
  assert.ok(long.length < 11 * 1024 * 1024);
  assert.match(long, /^x+\n\n\[Truncated: [^\n]*\]$/);
  await assert.rejects(readRtf(`{\\rtf1 ${"{".repeat(2000)}`), /nest too deeply/);
  await assert.rejects(readRtf("plain text"), /Not an RTF file/);
});

test("a thumbnail repack inflates only the images its kept paragraphs use", () => {
  const W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main";
  const R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
  const A = "http://schemas.openxmlformats.org/drawingml/2006/main";
  const picture = (id: string) => `<w:p><w:r><w:drawing><a:blip r:embed="${id}"/></w:drawing></w:r></w:p>`;
  const image = (id: string) =>
    `<Relationship Id="${id}" Type="${R}/image" Target="media/${id}.png"/>`;
  const bytes = zipSync({
    "[Content_Types].xml": strToU8(
      `<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="png" ContentType="image/png"/></Types>`,
    ),
    "_rels/.rels": strToU8(
      `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="d" Type="${R}/officeDocument" Target="word/document.xml"/></Relationships>`,
    ),
    "word/document.xml": strToU8(
      `<w:document xmlns:w="${W}" xmlns:r="${R}" xmlns:a="${A}"><w:body>${picture("rId1")}${picture("rId2")}</w:body></w:document>`,
    ),
    "word/_rels/document.xml.rels": strToU8(
      `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">${image("rId1")}${image("rId2")}</Relationships>`,
    ),
    "word/media/rId1.png": new Uint8Array([1, 2, 3]),
    "word/media/rId2.png": new Uint8Array([4, 5, 6]),
  });
  const names = (archive: Uint8Array) => Object.keys(unzipSync(archive)).filter((name) => name.endsWith(".png")).sort();
  assert.deepEqual(names(repackDocxPreviewArchive("a.docx", bytes, 1).archive), ["word/media/rId1.png", "word/media/rId2.png"]);
  const thumbnail = repackDocxPreviewArchive("a.docx", bytes, 1, { keptImagesOnly: true });
  assert.equal(thumbnail.truncated, true);
  assert.deepEqual(names(thumbnail.archive), ["word/media/rId1.png"]);
  const whole = repackDocxPreviewArchive("a.docx", bytes, 10, { keptImagesOnly: true });
  assert.deepEqual(names(whole.archive), ["word/media/rId1.png", "word/media/rId2.png"]);
});

test("a thumbnail repack restores only image parts the kept elements reference", () => {
  const W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main";
  const R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
  const A = "http://schemas.openxmlformats.org/drawingml/2006/main";
  const rel = (id: string, type: string, target: string) =>
    `<Relationship Id="${id}" Type="${R}/${type}" Target="${target}"/>`;
  const body =
    `<w:p><w:r><w:drawing><a:blip r:embed="rId1"/></w:drawing></w:r></w:p>` +
    `<!-- <a:blip r:embed="rId2"/> --><w:p><w:r><w:t>r:embed="rId2"</w:t></w:r></w:p>` +
    `<w:altChunk r:id="rId3"/>`;
  const bytes = zipSync({
    "[Content_Types].xml": strToU8(
      `<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="png" ContentType="image/png"/></Types>`,
    ),
    "_rels/.rels": strToU8(
      `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="d" Type="${R}/officeDocument" Target="word/document.xml"/></Relationships>`,
    ),
    "word/document.xml": strToU8(
      `<w:document xmlns:w="${W}" xmlns:r="${R}" xmlns:a="${A}"><w:body>${body}<w:p/><w:p/></w:body></w:document>`,
    ),
    "word/_rels/document.xml.rels": strToU8(
      `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">${rel("rId1", "image", "media/one.png")}${rel("rId2", "image", "media/two.png")}${rel("rId3", "aFChunk", "chunk.mht")}</Relationships>`,
    ),
    "word/media/one.png": new Uint8Array([1, 2, 3]),
    "word/media/two.png": new Uint8Array([4, 5, 6]),
    // Past the per-part ceiling, so the first pass leaves it out.
    "word/chunk.mht": new Uint8Array(11 * 1024 * 1024),
  });
  const kept = Object.keys(unzipSync(repackDocxPreviewArchive("a.docx", bytes, 3, { keptImagesOnly: true }).archive));
  assert.ok(kept.includes("word/media/one.png"));
  assert.ok(!kept.includes("word/media/two.png"));
  assert.ok(!kept.includes("word/chunk.mht"));
});

test("the text adapter claims text/plain documents but not real ones", () => {
  const cases: [string, string, boolean][] = [
    ["notes.pdf", "text/plain", true],
    ["notes.docx", "text/plain", true],
    ["a.pdf", "application/pdf", false],
    ["a.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", false],
    ["a.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", false],
    ["a.pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation", false],
    ["a.pdf", "", false],
  ];
  for (const [name, type, text] of cases) assert.equal(isTextAttachment(name, type), text, `${name} (${type || "no type"})`);
});

test("a thumbnail repack restores images in the notes its kept paragraphs refer to", () => {
  const W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main";
  const R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
  const A = "http://schemas.openxmlformats.org/drawingml/2006/main";
  const PKG = "http://schemas.openxmlformats.org/package/2006/relationships";
  const ns = `xmlns:w="${W}" xmlns:r="${R}" xmlns:a="${A}"`;
  const note = (id: string, image: string) =>
    `<w:footnote w:id="${id}"><w:p><w:r><w:drawing><a:blip r:embed="${image}"/></w:drawing></w:r></w:p></w:footnote>`;
  const ref = (id: string) => `<w:p><w:r><w:footnoteReference w:id="${id}"/></w:r></w:p>`;
  const bytes = zipSync({
    "[Content_Types].xml": strToU8(
      `<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="png" ContentType="image/png"/></Types>`,
    ),
    "_rels/.rels": strToU8(`<Relationships xmlns="${PKG}"><Relationship Id="d" Type="${R}/officeDocument" Target="word/document.xml"/></Relationships>`),
    "word/document.xml": strToU8(`<w:document ${ns}><w:body>${ref("1")}${ref("2")}</w:body></w:document>`),
    "word/_rels/document.xml.rels": strToU8(
      `<Relationships xmlns="${PKG}"><Relationship Id="f" Type="${R}/footnotes" Target="footnotes.xml"/></Relationships>`,
    ),
    "word/footnotes.xml": strToU8(`<w:footnotes ${ns}>${note("1", "rIdA")}${note("2", "rIdB")}</w:footnotes>`),
    "word/_rels/footnotes.xml.rels": strToU8(
      `<Relationships xmlns="${PKG}"><Relationship Id="rIdA" Type="${R}/image" Target="media/a.png"/><Relationship Id="rIdB" Type="${R}/image" Target="media/b.png"/></Relationships>`,
    ),
    "word/media/a.png": new Uint8Array([1]),
    "word/media/b.png": new Uint8Array([2]),
  });
  const kept = Object.keys(unzipSync(repackDocxPreviewArchive("a.docx", bytes, 1, { keptImagesOnly: true }).archive));
  assert.ok(kept.includes("word/media/a.png"));
  assert.ok(!kept.includes("word/media/b.png"));
});
