// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { DOMParser } from "@xmldom/xmldom";
import { strToU8, zipSync } from "fflate";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

(globalThis as { DOMParser?: unknown }).DOMParser = DOMParser;
(globalThis as { Node?: unknown }).Node ??= { ELEMENT_NODE: 1, TEXT_NODE: 3 };

const { readOpenDocumentAttachmentContent } = await import("../src/features/chat/open-document.ts");

const NS =
  'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" ' +
  'xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" ' +
  'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" ' +
  'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"';

const row = (attrs: string, ...cells: string[]) =>
  `<table:table-row ${attrs}>${cells
    .map((cell) => `<table:table-cell office:value-type="string"><text:p>${cell}</text:p></table:table-cell>`)
    .join("")}</table:table-row>`;

function odsFile(body: string, styles = ""): File {
  const bytes = zipSync({
    mimetype: strToU8("application/vnd.oasis.opendocument.spreadsheet"),
    "content.xml": strToU8(
      `<office:document-content ${NS}><office:automatic-styles>${styles}</office:automatic-styles>` +
        `<office:body><office:spreadsheet>${body}</office:spreadsheet></office:body></office:document-content>`,
    ),
  });
  return new File([bytes], "sales.ods", { type: "application/vnd.oasis.opendocument.spreadsheet" });
}

test("an attached spreadsheet keeps its filtered rows, hidden columns and hidden sheets", async () => {
  const file = odsFile(
    `<table:table table:name="Sales">` +
      `<table:table-column/><table:table-column/><table:table-column table:visibility="collapse"/>` +
      row("", "Region", "Q3", "Margin") +
      row("", "North", "100", "0.2") +
      row('table:visibility="filter"', "West", "300", "0.3") +
      row('table:visibility="collapse"', "East", "50", "0.1") +
      `<table:table-row-group table:display="false">${row("", "South", "75", "0.4")}</table:table-row-group>` +
      `</table:table>` +
      `<table:table table:name="Notes" table:style-name="ta2">${row("", "Budget cap 400")}</table:table>` +
      `<table:table table:name="Old" table:display="false">${row("", "Archived")}</table:table>`,
    `<style:style style:name="ta2" style:family="table"><style:table-properties table:display="false"/></style:style>`,
  );

  const { label, text } = await readOpenDocumentAttachmentContent(file, file.name, file.type);

  assert.equal(label, "ODS");
  assert.equal(
    text,
    [
      "[Sheet: Sales]\nRegion\tQ3\tMargin\nNorth\t100\t0.2\nWest\t300\t0.3\nEast\t50\t0.1\nSouth\t75\t0.4",
      "[Sheet: Notes]\nBudget cap 400",
      "[Sheet: Old]\nArchived",
    ].join("\n\n"),
  );
});
