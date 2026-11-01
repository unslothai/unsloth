// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { DOMParser as XmlDomParser, XMLSerializer as XmlSerializer } from "@xmldom/xmldom";
import { strFromU8, strToU8, zipSync } from "fflate";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  extractDocxAttachmentText,
  linearizeDocxMath,
  markDocxNotes,
  repackDocxAttachmentArchive,
  writeDocxBreaksAndCheckboxes,
  writeDocxListNumbers,
  writeDocxTableRows,
} = await import("../src/features/chat/attachment-content.ts");
const { mammoth, writeDocxSymbols } = await import("../src/features/chat/docx-reader.ts");

const W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"';
const M = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"';
const W14 = 'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"';
const REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";

function rels(entries: Array<[string, string]>): Uint8Array {
  return strToU8(
    `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">${entries
      .map(([type, target], i) => `<Relationship Id="rId${i + 1}" Type="${REL}/${type}" Target="${target}"/>`)
      .join("")}</Relationships>`,
  );
}

const notes = (kind: string, ids: string[]) =>
  strToU8(
    `<w:${kind}s ${W} ${M}>${ids
      .map(
        (id) =>
          `<w:${kind} w:id="${id}"><w:p><w:r><w:t>${kind} ${id} </w:t></w:r><m:oMath><m:r><m:t>y</m:t></m:r></m:oMath>` +
          `<w:r><w:br/><w:sym w:font="Symbol" w:char="F062"/></w:r></w:p></w:${kind}>`,
      )
      .join("")}</w:${kind}s>`,
  );

// Every rewrite has something to do, and several read what an earlier one wrote.
function everyRewrite(document = "word/document.xml"): Record<string, Uint8Array> {
  return {
    "[Content_Types].xml": strToU8("<Types/>"),
    "_rels/.rels": rels([["officeDocument", document]]),
    [document]: strToU8(
      `<w:document ${W} ${M} ${W14}><w:body>` +
        `<w:p><m:oMath><m:r><w:sym w:font="Symbol" w:char="F061"/><m:t>x</m:t></m:r></m:oMath><w:r><w:t>after math</w:t></w:r></w:p>` +
        `<w:tbl><w:tr><w:tc><w:p><w:r><w:t>a</w:t><w:br/><w:t>b</w:t><w:footnoteReference w:id="1"/></w:r></w:p></w:tc>` +
        `<w:tc><w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr><w:r><w:t>item</w:t><w:tab/><w:t>tabbed</w:t></w:r></w:p></w:tc></w:tr></w:tbl>` +
        `<w:sdt><w:sdtPr><w14:checkbox><w14:checked w14:val="1"/></w14:checkbox></w:sdtPr><w:sdtContent><w:p><w:r><w:t>box</w:t></w:r></w:p></w:sdtContent></w:sdt>` +
        `<w:p><w:r><w:t>legacy</w:t><w:cr/><w:sym w:font="Wingdings" w:char="F0FC"/></w:r></w:p>` +
        `<w:p><w:r><w:t>end</w:t><w:endnoteReference w:id="2"/></w:r></w:p></w:body></w:document>`,
    ),
    [`${document.replace(/[^/]*$/, "")}_rels/${document.split("/").pop()}.rels`]: rels([
      ["footnotes", "footnotes.xml"],
      ["endnotes", "endnotes.xml"],
      ["numbering", "numbering.xml"],
    ]),
    [document.replace(/[^/]*$/, "footnotes.xml")]: notes("footnote", ["1", "3"]),
    [document.replace(/[^/]*$/, "endnotes.xml")]: notes("endnote", ["2"]),
    [document.replace(/[^/]*$/, "numbering.xml")]: strToU8(
      `<w:numbering ${W}><w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/></w:lvl></w:abstractNum>` +
        `<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num></w:numbering>`,
    ),
  };
}

/** The archive-to-archive chain extractDocxAttachmentText ran before reading one map of parts. */
async function chained(name: string, bytes: Uint8Array): Promise<string> {
  const repacked = repackDocxAttachmentArchive(name, bytes);
  const marked = markDocxNotes(linearizeDocxMath(writeDocxSymbols(repacked, 10 * 1024 * 1024)));
  const archive = writeDocxTableRows(writeDocxBreaksAndCheckboxes(writeDocxListNumbers(marked.archive)));
  const { value } = await mammoth.extractRawText({ buffer: Buffer.from(archive) } as never);
  return marked.label(value);
}

async function withDocxGlobals<T>(run: () => Promise<T>): Promise<T> {
  const globals = globalThis as { DOMParser?: unknown; XMLSerializer?: unknown };
  const original = { DOMParser: globals.DOMParser, XMLSerializer: globals.XMLSerializer };
  // The malformed fixtures would otherwise log every xmldom parse error.
  globals.DOMParser = class extends XmlDomParser {
    constructor() {
      super({ errorHandler: () => undefined });
    }
  };
  globals.XMLSerializer = XmlSerializer;
  // mammoth's node build takes a buffer, not the browser build's arrayBuffer.
  const extract = mammoth.extractRawText;
  mammoth.extractRawText = ((input: { arrayBuffer?: ArrayBuffer }) =>
    extract(input.arrayBuffer ? ({ buffer: Buffer.from(input.arrayBuffer) } as never) : (input as never))) as never;
  try {
    return await run();
  } finally {
    mammoth.extractRawText = extract;
    Object.assign(globals, original);
  }
}

async function assertSameAsChained(name: string, bytes: Uint8Array): Promise<string> {
  const file = new File([new Uint8Array(bytes)], name);
  const outcome = (promise: Promise<string>) =>
    promise.then(
      (text) => ({ text }),
      (error: Error) => ({ error: `${error.constructor.name}: ${error.message}` }),
    );
  const [direct, chain] = [await outcome(extractDocxAttachmentText(file)), await outcome(chained(name, bytes))];
  assert.deepEqual(direct, chain, name);
  return "text" in direct ? direct.text : "";
}

/** "__proto__" cannot be a key in a zipSync input, so the name is patched into the bytes. */
function withProtoPart(entries: Record<string, Uint8Array>, level: 0 | 6): Uint8Array {
  const bytes = zipSync(entries, { level });
  const from = strToU8("__PROTO__");
  for (let i = 0; i + from.length <= bytes.length; i++) {
    if (from.every((byte, j) => bytes[i + j] === byte)) bytes.set(strToU8("__proto__"), i);
  }
  return bytes;
}

/** fflate writes no zip64, so its 16-bit entry count wraps; a zip64 end record carries the full count. */
function asZip64(bytes: Uint8Array, count: number): Uint8Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const end = bytes.length - 22;
  const cdSize = view.getUint32(end + 12, true);
  const cdOffset = view.getUint32(end + 16, true);
  const record = new DataView(new ArrayBuffer(56 + 20));
  record.setUint32(0, 0x06064b50, true);
  record.setBigUint64(4, 44n, true);
  record.setUint16(12, 45, true);
  record.setUint16(14, 45, true);
  record.setBigUint64(24, BigInt(count), true);
  record.setBigUint64(32, BigInt(count), true);
  record.setBigUint64(40, BigInt(cdSize), true);
  record.setBigUint64(48, BigInt(cdOffset), true);
  record.setUint32(56, 0x07064b50, true);
  record.setBigUint64(64, BigInt(end), true);
  record.setUint32(72, 1, true);
  const out = new Uint8Array(bytes.length + 76);
  out.set(bytes.subarray(0, end));
  out.set(new Uint8Array(record.buffer), end);
  out.set(bytes.subarray(end), end + 76);
  new DataView(out.buffer).setUint16(end + 76 + 8, 0xffff, true);
  new DataView(out.buffer).setUint16(end + 76 + 10, 0xffff, true);
  return out;
}

test("extractDocxAttachmentText reads the same text as the per-rewrite archive chain", async () => {
  await withDocxGlobals(async () => {
    for (const level of [0, 6] as const) {
      const text = await assertSameAsChained("all.docx", zipSync(everyRewrite(), { level }));
      for (const piece of ["αx", "1. item", "\t", "a b[1]", "☒", "legacy\n✓", "end[i]", "Footnotes\n[1]", "Endnotes\n[i]"]) {
        assert.ok(text.includes(piece), `${piece} in ${JSON.stringify(text)}`);
      }
      await assertSameAsChained("moved.docx", zipSync(everyRewrite("other/main.xml"), { level }));
    }
  });
});

test("extractDocxAttachmentText matches the chain on names the zip round trip changes", async () => {
  await withDocxGlobals(async () => {
    const base = everyRewrite();
    const cases: Record<string, Uint8Array> = {
      "proto.docx": withProtoPart({ ...base, __PROTO__: strToU8("<x/>") }, 0),
      "proto-first.docx": withProtoPart({ __PROTO__: strToU8("<x/>"), ...base }, 6),
      "proto-main.docx": withProtoPart(
        { ...base, __PROTO__: base["word/document.xml"]!, "_rels/.rels": rels([["officeDocument", "/__PROTO__"]]) },
        0,
      ),
      "names.docx": zipSync({ ...base, "word/\ud800x.xml": strToU8(`<w:document ${W}><w:body><w:p><w:r><w:br/></w:r></w:p></w:body></w:document>`), "ünï.xml": strToU8("<a/>"), "1": strToU8("1") }),
      "locator.docx": zipSync({ ...base, ["zzPK\u0006\u0007" + "z".repeat(24)]: strToU8("q") }),
    };
    for (const count of [65534, 65535, 70000]) {
      const filler: Record<string, Uint8Array> = {};
      for (let i = Object.keys(base).length; i < count; i++) filler[`c/${i}`] = new Uint8Array(0);
      if (count <= 65535) cases[`many-${count}.docx`] = zipSync({ ...base, ...filler }, { level: 0 });
      // Past the 16-bit count, a round trip keeps only the first count % 65536 entries.
      else {
        cases[`zip64-${count}.docx`] = asZip64(zipSync({ ...base, ...filler }, { level: 0 }), count);
        cases[`zip64-late-${count}.docx`] = asZip64(zipSync({ ...filler, ...base }, { level: 0 }), count);
      }
    }
    for (const [name, bytes] of Object.entries(cases)) await assertSameAsChained(name, bytes);
  });
});

test("extractDocxAttachmentText keeps the chain's errors for unreadable archives", async () => {
  await withDocxGlobals(async () => {
    const good = zipSync(everyRewrite(), { level: 0 });
    const broken = new Uint8Array(good);
    broken.set(strToU8("not xml <<<"), strFromU8(broken, true).indexOf("<w:body>"));
    for (const [name, bytes] of Object.entries({
      "empty.docx": new Uint8Array(0),
      "text.docx": strToU8("plain text, not a zip"),
      "half.docx": good.subarray(0, good.length >> 1),
      "no-body.docx": zipSync({ "[Content_Types].xml": strToU8("<Types/>") }),
      "broken-xml.docx": zipSync({ ...everyRewrite(), "word/document.xml": strToU8(`<w:document ${W}><w:body><w:p><w:r><w:t>x</w:r></w:body>`) }),
      "stored-broken.docx": broken,
    })) {
      await assertSameAsChained(name, bytes);
    }
  });
});
