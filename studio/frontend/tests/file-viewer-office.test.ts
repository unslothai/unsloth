// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { DOMParser } from "@xmldom/xmldom";
import { strToU8, zipSync } from "fflate";

import { formatNumber, readPptx, readXlsx } from "../src/components/file-viewer/office.ts";

// DOMParser is absent under node. xmldom (mammoth's parser) stands in, given the element
// traversal the readers use and it leaves out.
{
  const probe = new DOMParser().parseFromString("<a/>", "application/xml").documentElement;
  const proto = Object.getPrototypeOf(probe) as object;
  const nextElement = (node: Node | null): Element | null => {
    while (node && node.nodeType !== 1) node = node.nextSibling;
    return node as Element | null;
  };
  Object.defineProperties(proto, {
    firstElementChild: {
      get(this: Element) {
        return nextElement(this.firstChild);
      },
    },
    nextElementSibling: {
      get(this: Element) {
        return nextElement(this.nextSibling);
      },
    },
    parentElement: {
      get(this: Element) {
        const parent = this.parentNode;
        return parent && parent.nodeType === 1 ? parent : null;
      },
    },
    children: {
      get(this: Element) {
        const out: Element[] = [];
        for (let node = this.firstElementChild; node; node = node.nextElementSibling) out.push(node);
        return out;
      },
    },
  });
  (globalThis as { DOMParser?: unknown }).DOMParser = DOMParser;
}

const MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
const REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
const PKG = "http://schemas.openxmlformats.org/package/2006/relationships";

function workbook(sheet: string, extra: Record<string, Uint8Array> = {}): Uint8Array {
  return zipSync({
    "xl/workbook.xml": strToU8(
      `<workbook xmlns="${MAIN}" xmlns:r="${REL}"><sheets><sheet name="One" sheetId="1" r:id="rId1"/></sheets></workbook>`,
    ),
    "xl/_rels/workbook.xml.rels": strToU8(
      `<Relationships xmlns="${PKG}"><Relationship Id="rId1" Type="${REL}/worksheet" Target="worksheets/sheet1.xml"/></Relationships>`,
    ),
    "xl/worksheets/sheet1.xml": strToU8(sheet),
    ...extra,
  });
}

function rowsXml(count: number, prefix = ""): string {
  const filler = "x".repeat(200);
  let rows = "";
  for (let r = 1; r <= count; r++) {
    rows += `<${prefix}row r="${r}"><${prefix}c r="A${r}"><${prefix}v>${r}</${prefix}v></${prefix}c><${prefix}c r="B${r}" t="inlineStr"><${prefix}is><${prefix}t>${filler}</${prefix}t></${prefix}is></${prefix}c></${prefix}row>`;
  }
  const ns = prefix ? `xmlns:${prefix.slice(0, -1)}="${MAIN}"` : `xmlns="${MAIN}"`;
  return `<${prefix}worksheet ${ns}><${prefix}sheetData>${rows}</${prefix}sheetData><${prefix}mergeCells count="0"/></${prefix}worksheet>`;
}

/** Sets an entry's uncompressed size in the central directory, as a zip bomb declares it. */
function declareSize(zip: Uint8Array, name: string, size: number): Uint8Array {
  const view = new DataView(zip.buffer, zip.byteOffset, zip.byteLength);
  const encoded = strToU8(name);
  for (let i = 0; i < zip.length - 46; i++) {
    if (view.getUint32(i, true) !== 0x02014b50) continue;
    const length = view.getUint16(i + 28, true);
    const entry = zip.subarray(i + 46, i + 46 + length);
    if (length === encoded.length && entry.every((byte, index) => byte === encoded[index])) {
      view.setUint32(i + 24, size, true);
      return zip;
    }
  }
  throw new Error(`no entry ${name}`);
}

test("format sections split only at semicolons outside quoted text", () => {
  assert.equal(formatNumber(12.5, '0.00 "kg; net"'), "12.50 kg; net");
  assert.equal(formatNumber(-3, '0;"minus "0'), "minus 3");
});

test("scientific formats keep their literals and negative sections", () => {
  assert.equal(formatNumber(1234, '0.00E+00 "kg"'), "1.23E+03 kg");
  assert.equal(formatNumber(-1234, "0.00E+00;(0.00E+00)"), "(1.23E+03)");
  assert.equal(formatNumber(0.00012, "0.0E+0"), "1.2E-4");
  assert.equal(formatNumber(1234, "0.0E-0"), "1.2E3");
  assert.equal(formatNumber(-1234, "0.00E+00"), "-1.23E+03");
});

test("elapsed formats show fractional seconds", () => {
  assert.equal(formatNumber(1.5 / 86400, "[h]:mm:ss.00"), "0:00:01.50");
  assert.equal(formatNumber(3725 / 86400, "[h]:mm:ss"), "1:02:05");
  assert.equal(formatNumber(90.25 / 86400, "[ss].0"), "90.3");
});

test("a workbook's unrelated parts are not inflated", () => {
  const zip = workbook(rowsXml(3), { "customXml/item1.xml": strToU8("<x/>") });
  // Declared past the unpacked ceiling: reading it would refuse the file.
  declareSize(zip, "customXml/item1.xml", 0xfffffff0);
  const [sheet] = readXlsx(zip);
  assert.equal(sheet?.rows.length, 3);
  assert.equal(sheet?.rows[2]?.[0]?.text, "3");
});

test("a large sheet is parsed only up to the rows kept, and marked truncated", () => {
  for (const prefix of ["", "x:"]) {
    const [sheet] = readXlsx(workbook(rowsXml(5200, prefix)));
    assert.equal(sheet?.rows.length, 5000, prefix);
    assert.equal(sheet?.truncated, true, prefix);
    assert.equal(sheet?.rows[4999]?.[0]?.text, "5000", prefix);
  }
  const [whole] = readXlsx(workbook(rowsXml(5000)));
  assert.equal(whole?.truncated, false);
  assert.equal(whole?.rows.length, 5000);
});

const P = "http://schemas.openxmlformats.org/presentationml/2006/main";
const A = "http://schemas.openxmlformats.org/drawingml/2006/main";
const DGM = "http://schemas.openxmlformats.org/drawingml/2006/diagram";

function deck(slide: string, extra: Record<string, Uint8Array> = {}, slideRels = ""): Uint8Array {
  return zipSync({
    "ppt/presentation.xml": strToU8(
      `<p:presentation xmlns:p="${P}" xmlns:r="${REL}"><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst><p:sldSz cx="12192000" cy="6858000"/></p:presentation>`,
    ),
    "ppt/_rels/presentation.xml.rels": strToU8(
      `<Relationships xmlns="${PKG}"><Relationship Id="rId1" Type="${REL}/slide" Target="slides/slide1.xml"/></Relationships>`,
    ),
    "ppt/slides/slide1.xml": strToU8(
      `<p:sld xmlns:p="${P}" xmlns:a="${A}" xmlns:r="${REL}" xmlns:dgm="${DGM}"><p:cSld><p:spTree>${slide}</p:spTree></p:cSld></p:sld>`,
    ),
    "ppt/slides/_rels/slide1.xml.rels": strToU8(`<Relationships xmlns="${PKG}">${slideRels}</Relationships>`),
    ...extra,
  });
}

test("a slide's native table is capped, with a row marking the cut", () => {
  const cell = "<a:tc><a:txBody><a:p><a:r><a:t>v</a:t></a:r></a:p></a:txBody></a:tc>";
  const row = `<a:tr>${cell.repeat(10)}</a:tr>`;
  const frame = `<p:graphicFrame><a:graphic><a:graphicData><a:tbl>${row.repeat(1000)}</a:tbl></a:graphicData></a:graphic></p:graphicFrame>`;
  const [box] = readPptx(deck(frame)).slides[0]!.boxes;
  assert.equal(box?.table?.length, 501);
  assert.deepEqual(box?.table?.at(-1), ["…"]);
});

test("SmartArt labels are read from the diagram's data part", () => {
  const frame = `<p:graphicFrame><a:graphic><a:graphicData><dgm:relIds r:dm="rId2" r:lo="rId3"/></a:graphicData></a:graphic></p:graphicFrame>`;
  const data = `<dgm:dataModel xmlns:dgm="${DGM}" xmlns:a="${A}"><dgm:ptLst><dgm:pt modelId="0" type="doc"><dgm:t><a:p><a:r><a:t>Doc</a:t></a:r></a:p></dgm:t></dgm:pt><dgm:pt modelId="1"><dgm:t><a:p><a:r><a:t>Plan</a:t></a:r></a:p></dgm:t></dgm:pt><dgm:pt modelId="2" type="node"><dgm:t><a:p><a:r><a:t>Ship</a:t></a:r></a:p></dgm:t></dgm:pt></dgm:ptLst></dgm:dataModel>`;
  const rels = `<Relationship Id="rId2" Type="${REL}/diagramData" Target="../diagrams/data1.xml"/>`;
  const [box] = readPptx(deck(frame, { "ppt/diagrams/data1.xml": strToU8(data) }, rels)).slides[0]!.boxes;
  assert.deepEqual(box?.paragraphs?.map((p) => p.text), ["Plan", "Ship"]);
});

test("slide pictures are kept as Blobs, and left unread for text", () => {
  const pic = `<p:pic><p:blipFill><a:blip r:embed="rId2"/></p:blipFill></p:pic>`;
  const rels = `<Relationship Id="rId2" Type="${REL}/image" Target="../media/image1.png"/>`;
  const media = { "ppt/media/image1.png": new Uint8Array([137, 80, 78, 71]), "ppt/media/clip.mp4": new Uint8Array(8) };
  const zip = deck(pic, media, rels);
  // A video is never inflated, for the viewer or the text.
  declareSize(zip, "ppt/media/clip.mp4", 0xfffffff0);
  const [box] = readPptx(zip).slides[0]!.boxes;
  assert.ok(box?.image instanceof Blob);
  assert.equal(box?.image?.type, "image/png");
  assert.equal(readPptx(zip, { images: false }).slides[0]!.boxes.length, 0);
});

test("sheet XML is read as text: entities, CDATA, rich and phonetic runs, shared formulas", () => {
  const sheet = `<x:worksheet xmlns:x="${MAIN}"><x:cols><x:col min='3' max='3' hidden='1'/></x:cols><x:sheetData>
<x:row r="1"><x:c r="A1" t="inlineStr"><x:is><x:r><x:t>a &amp; &#x42;&#67;</x:t></x:r><x:r><x:t><![CDATA[<d>]]></x:t></x:r><x:rPh><x:t>ruby</x:t></x:rPh></x:is></x:c><x:c r="B1" t="b"><x:v>1</x:v></x:c><x:c r="C1"><x:v>9</x:v></x:c></x:row>
<x:row r="2" hidden="1"><x:c r="A2"><x:v>1</x:v></x:c></x:row>
<x:row r="3"><x:c r="A3"><x:f t="shared" ref="A3:A4" si="0">B3*2+$C$1</x:f><x:v>7</x:v></x:c><x:c t="str"><x:f>"x"&amp;"y"</x:f><x:v>xy</x:v></x:c></x:row>
<x:row r="4"><x:c r="A4"><x:f t="shared" si="0"/></x:c><x:c r="B4"/></x:row>
</x:sheetData></x:worksheet>`;
  const [read] = readXlsx(workbook(sheet));
  assert.deepEqual(
    // Array.from: a hidden row is a hole, which map skips.
    Array.from(read!.rows, (row) => row?.map((cell) => cell?.text)),
    [["a & BC<d>", "TRUE"], undefined, ["7", "xy"], ["=B4*2+$C$1"]],
  );
  assert.deepEqual([...read!.hidden!.rows], [1]);
  assert.deepEqual([...read!.hidden!.columns], [2]);
});

test("a row left open ends the sheet rather than scanning on", () => {
  const [read] = readXlsx(workbook(`<worksheet xmlns="${MAIN}"><sheetData><row r="1"><c r="A1"><v>1</v></c></row><row r="2"><c r="A2"><v>2</v></c>`));
  assert.deepEqual(read?.rows.map((row) => row?.map((cell) => cell?.text)), [["1"]]);
});

test("common number formats", () => {
  const cases: [number, string, string][] = [
    [1234567.891, "#,##0.00", "1,234,567.89"],
    [12345678, "#,##0,", "12,346"],
    [-5, "#,##0.00;(#,##0.00)", "(5.00)"],
    [123, "00000", "00123"],
    [0.5, "#.##", ".5"],
    [1.5, "# ?/?", "1 1/2"],
    [0.123, "0.0%", "12.3%"],
    [150, '[>=100]"big "0;"small "0', "big 150"],
    [45000, "yyyy-mm-dd", "2023-03-15"],
    [1234.5, "[$€-407] #,##0.00", "€ 1,234.50"],
  ];
  for (const [value, format, expected] of cases) assert.equal(formatNumber(value, format), expected, format);
});
