// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { DOMParser } from "@xmldom/xmldom";
import { strToU8, zipSync } from "fflate";

import { formatNumber, readPptx, readXlsx } from "../src/components/file-viewer/office.ts";

// xmldom stands in for DOMParser, plus the element traversal it lacks.
{
  const proto = Object.getPrototypeOf(new DOMParser().parseFromString("<a/>", "application/xml").documentElement);
  const element = (node: Node | null): Element | null => {
    while (node && node.nodeType !== 1) node = node.nextSibling;
    return node as Element | null;
  };
  Object.defineProperties(proto, {
    firstElementChild: { get: function (this: Element) { return element(this.firstChild); } },
    nextElementSibling: { get: function (this: Element) { return element(this.nextSibling); } },
    parentElement: { get: function (this: Element) { return this.parentNode?.nodeType === 1 ? this.parentNode : null; } },
    children: {
      get: function (this: Element) {
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
const rels = (items: string) => strToU8(`<Relationships xmlns="${PKG}">${items}</Relationships>`);

function workbook(sheet: string, extra: Record<string, Uint8Array> = {}): Uint8Array {
  return zipSync({
    "xl/workbook.xml": strToU8(
      `<workbook xmlns="${MAIN}" xmlns:r="${REL}"><sheets><sheet name="One" sheetId="1" r:id="rId1"/></sheets></workbook>`,
    ),
    "xl/_rels/workbook.xml.rels": rels(`<Relationship Id="rId1" Type="${REL}/worksheet" Target="worksheets/sheet1.xml"/>`),
    "xl/worksheets/sheet1.xml": strToU8(sheet),
    ...extra,
  });
}

/** Sets an entry's uncompressed size in the central directory, as a zip bomb declares it. */
function declareHuge(zip: Uint8Array, name: string): Uint8Array {
  const view = new DataView(zip.buffer, zip.byteOffset, zip.byteLength);
  for (let i = 0; i < zip.length - 46; i++) {
    if (view.getUint32(i, true) !== 0x02014b50) continue;
    if (new TextDecoder().decode(zip.subarray(i + 46, i + 46 + view.getUint16(i + 28, true))) === name) {
      view.setUint32(i + 24, 0xfffffff0, true);
      return zip;
    }
  }
  throw new Error(`no entry ${name}`);
}

test("number formats", () => {
  const cases: [number, string, string][] = [
    [1234567.891, "#,##0.00", "1,234,567.89"],
    [-5, "#,##0.00;(#,##0.00)", "(5.00)"],
    [1.5, "# ?/?", "1 1/2"],
    [-1.5, "# ?/?;(# ?/?)", "(1 1/2)"],
    [45000, "yyyy-mm-dd", "2023-03-15"],
    [12.5, '0.00 "kg; net"', "12.50 kg; net"],
    [1234, '0.00E+00 "kg"', "1.23E+03 kg"],
    [-1234, "0.00E+00;(0.00E+00)", "(1.23E+03)"],
    [1.5 / 86400, "[h]:mm:ss.00", "0:00:01.50"],
    [1 / 24, '[h]:mm "hours"', "1:00 hours"],
  ];
  for (const [value, format, expected] of cases) assert.equal(formatNumber(value, format), expected, format);
});

test("xlsx: reads only the parts and rows it keeps", () => {
  let rows = "";
  for (let r = 1; r <= 5200; r++) {
    rows += `<x:row r="${r}"><x:c r="A${r}"><x:v>${r}</x:v></x:c><x:c r="B${r}" t="inlineStr"><x:is><x:t>${"x".repeat(200)}</x:t></x:is></x:c></x:row>`;
  }
  const zip = workbook(`<x:worksheet xmlns:x="${MAIN}"><x:sheetData>${rows}</x:sheetData></x:worksheet>`, {
    "customXml/item1.xml": strToU8("<x/>"),
  });
  // Past the unpacked ceiling: inflating it would refuse the file.
  const [sheet] = readXlsx(declareHuge(zip, "customXml/item1.xml"));
  assert.equal(sheet?.rows.length, 5000);
  assert.equal(sheet?.truncated, true);
});

test("xlsx: sheet XML read as text", () => {
  const [sheet] = readXlsx(
    workbook(`<worksheet xmlns="${MAIN}"><cols><col min='2' max='2' hidden='1'/></cols><sheetData>
<row r="1"><c r="A1" t="inlineStr"><is><r><t>a &amp; &#x42;</t></r><r><t><![CDATA[<c>]]></t></r><rPh><t>ruby</t></rPh></is></c><c r="B1"><v>9</v></c></row>
<row r="2"><c r="A2"><f t="shared" ref="A2:A3" si="0">B2*2+$C$1</f><v>7</v></c></row>
<row r="3"><c r="A3"><f t="shared" si="0"/></c></row>
</sheetData></worksheet>`),
  );
  assert.deepEqual(
    sheet?.rows.map((row) => row.map((cell) => cell?.text)),
    [["a & B<c>"], ["7"], ["=B3*2+$C$1"]],
  );
});

test("pptx: tables capped, SmartArt read, pictures as Blobs", () => {
  const P = "http://schemas.openxmlformats.org/presentationml/2006/main";
  const A = "http://schemas.openxmlformats.org/drawingml/2006/main";
  const DGM = "http://schemas.openxmlformats.org/drawingml/2006/diagram";
  const tc = "<a:tc><a:txBody><a:p><a:r><a:t>v</a:t></a:r></a:p></a:txBody></a:tc>";
  const text = (t: string) => `<dgm:t><a:p><a:r><a:t>${t}</a:t></a:r></a:p></dgm:t>`;
  const zip = zipSync({
    "ppt/presentation.xml": strToU8(
      `<p:presentation xmlns:p="${P}" xmlns:r="${REL}"><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst></p:presentation>`,
    ),
    "ppt/_rels/presentation.xml.rels": rels(`<Relationship Id="rId1" Type="${REL}/slide" Target="slides/slide1.xml"/>`),
    "ppt/slides/slide1.xml": strToU8(
      `<p:sld xmlns:p="${P}" xmlns:a="${A}" xmlns:r="${REL}" xmlns:dgm="${DGM}"><p:cSld><p:spTree>` +
        `<p:graphicFrame><a:graphic><a:graphicData><a:tbl>${`<a:tr>${tc.repeat(10)}</a:tr>`.repeat(1000)}</a:tbl></a:graphicData></a:graphic></p:graphicFrame>` +
        `<p:graphicFrame><a:graphic><a:graphicData><dgm:relIds r:dm="rId2"/></a:graphicData></a:graphic></p:graphicFrame>` +
        `<p:pic><p:blipFill><a:blip r:embed="rId3"/></p:blipFill></p:pic>` +
        `</p:spTree></p:cSld></p:sld>`,
    ),
    "ppt/slides/_rels/slide1.xml.rels": rels(
      `<Relationship Id="rId2" Type="${REL}/diagramData" Target="../diagrams/data1.xml"/><Relationship Id="rId3" Type="${REL}/image" Target="../media/image1.png"/>`,
    ),
    "ppt/diagrams/data1.xml": strToU8(
      `<dgm:dataModel xmlns:dgm="${DGM}" xmlns:a="${A}"><dgm:ptLst><dgm:pt type="doc">${text("Doc")}</dgm:pt><dgm:pt>${text("Plan")}</dgm:pt></dgm:ptLst></dgm:dataModel>`,
    ),
    "ppt/media/image1.png": new Uint8Array([137, 80, 78, 71]),
    "ppt/media/clip.mp4": new Uint8Array(8),
  });
  // Video is never inflated.
  const [image, table, diagram] = readPptx(declareHuge(zip, "ppt/media/clip.mp4")).slides[0]!.boxes;
  assert.equal(image?.image?.type, "image/png");
  assert.equal(table?.table?.length, 501);
  assert.deepEqual(table?.table?.at(-1), ["…"]);
  assert.deepEqual(diagram?.paragraphs?.map((p) => p.text), ["Plan"]);
});
