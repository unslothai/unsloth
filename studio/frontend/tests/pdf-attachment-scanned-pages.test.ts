// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { extractPdfAttachmentText } = await import("../src/features/chat/attachment-content.ts");
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");
const { recordedToasts } = await import("./helpers/store-stubs/toast.ts");

function pdf(pages: { content: string; resources: string; annots?: string; rotate?: number }[], extra: string[]): Uint8Array<ArrayBuffer> {
  const first = 3 + pages.length;
  const objects = [
    "<< /Type /Catalog /Pages 2 0 R >>",
    `<< /Type /Pages /Kids [${pages.map((_, i) => `${3 + i} 0 R`).join(" ")}] /Count ${pages.length} >>`,
    ...pages.map(
      ({ resources, annots, rotate }, i) =>
        `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources ${resources} /Contents ${first + i} 0 R${annots ? ` /Annots ${annots}` : ""}${rotate ? ` /Rotate ${rotate}` : ""} >>`,
    ),
    ...pages.map(({ content }) => `<< /Length ${content.length} >>\nstream\n${content}\nendstream`),
    ...extra,
  ];
  let body = "%PDF-1.4\n";
  const offsets = objects.map((object, index) => {
    const offset = body.length;
    body += `${index + 1} 0 obj\n${object}\nendobj\n`;
    return offset;
  });
  const xref = body.length;
  body += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  body += offsets.map((offset) => `${String(offset).padStart(10, "0")} 00000 n \n`).join("");
  body += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  return Uint8Array.from(body, (char) => char.charCodeAt(0));
}

const typedPage = (words: string, font: number) => ({
  content: `BT /F1 12 Tf 20 100 Td (${words}) Tj ET`,
  resources: `<< /Font << /F1 ${font} 0 R >> >>`,
});
const scannedPage = (image: number, stamp?: { words: string; font: number; y?: number }) => ({
  content: `q 200 0 0 200 0 0 cm /Im1 Do Q${stamp ? ` BT /F1 8 Tf 10 ${stamp.y ?? 5} Td (${stamp.words}) Tj ET` : ""}`,
  resources: `<< /XObject << /Im1 ${image} 0 R >>${stamp ? ` /Font << /F1 ${stamp.font} 0 R >>` : ""} >>`,
});
const FONT = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>";
const IMAGE = `<< /Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceGray /BitsPerComponent 8 /Length 4 >>\nstream\n${"\x80".repeat(4)}\nendstream`;

const contract = () =>
  new File([pdf([typedPage("Contract terms", 7), scannedPage(8)], [FONT, IMAGE])], "contract.pdf", {
    type: "application/pdf",
  });

function serve(body: Record<string, unknown>, status = 200): string[] {
  const requests: string[] = [];
  setAuthFetchHandler((input) => {
    requests.push(input);
    return Response.json(body, { status });
  });
  return requests;
}

test.afterEach(() => {
  setAuthFetchHandler(null);
  recordedToasts.length = 0;
});

test("a scanned page of an attached pdf reaches the model instead of being dropped", async () => {
  const requests = serve({
    filename: "contract.pdf",
    text: "Contract terms\n\nANNEX B: Penalty is 50000 EUR per day.",
    truncated: false,
  });
  assert.equal(
    await extractPdfAttachmentText(contract()),
    "Contract terms\n\nANNEX B: Penalty is 50000 EUR per day.",
  );
  assert.equal(requests.length, 1);
  assert.match(requests[0]!, /\/rag\/extract-text$/);
  assert.deepEqual(recordedToasts, []);
});

test("a scanned page that cannot be read is named in a warning", async () => {
  const warning =
    "Could not read scanned page 2 of contract.pdf. Reading scanned pages needs a vision model loaded in Chat, or Tesseract OCR installed, and covers at most 50 pages.";
  serve({ filename: "contract.pdf", text: "Contract terms", truncated: false, warning });
  assert.equal(await extractPdfAttachmentText(contract()), "Contract terms");
  assert.deepEqual(recordedToasts, [{ kind: "warning", message: warning, description: undefined }]);
});

test("a typed pdf is read in the browser without a server round trip", async () => {
  const requests = serve({ text: "unused", truncated: false });
  const typed = new File(
    [pdf([typedPage("page one", 7), typedPage("page two", 7)], [FONT])],
    "typed.pdf",
    { type: "application/pdf" },
  );
  assert.equal(await extractPdfAttachmentText(typed), "page one\n\npage two");
  assert.deepEqual(requests, []);
});

test("a scanned page carrying a page number stamp is still read by the server", async () => {
  const requests = serve({ filename: "stamped.pdf", text: "Contract terms\n\nANNEX B", truncated: false });
  const stamped = new File(
    [pdf([typedPage("Contract terms", 7), scannedPage(8, { words: "Page 2", font: 7, y: 100 })], [FONT, IMAGE])],
    "stamped.pdf",
    { type: "application/pdf" },
  );
  assert.equal(await extractPdfAttachmentText(stamped), "Contract terms\n\nANNEX B");
  assert.equal(requests.length, 1);
});

for (const [edge, y] of [["footer", 5], ["header", 190]] as const) {
  test(`a scanned page with a selectable ${edge} is still read by the server`, async () => {
    const requests = serve({ filename: "invoice.pdf", text: "Invoice total: 4,250.00 EUR", truncated: false });
    const stamp = { words: "ACME Corp Confidential - Page 1 of 1", font: 5, y };
    const invoice = new File([pdf([scannedPage(6, stamp)], [FONT, IMAGE])], "invoice.pdf", {
      type: "application/pdf",
    });
    assert.equal(await extractPdfAttachmentText(invoice), "Invoice total: 4,250.00 EUR");
    assert.equal(requests.length, 1);
  });
}

test("a landscape scan's footer along the rotated page edge is still read by the server", async () => {
  const requests = serve({ filename: "landscape.pdf", text: "Invoice total", truncated: false });
  const page = {
    content: "q 200 0 0 200 0 0 cm /Im1 Do Q BT /F1 8 Tf 0 1 -1 0 195 30 Tm (ACME Corp Confidential - Page 1 of 1) Tj ET",
    resources: "<< /XObject << /Im1 6 0 R >> /Font << /F1 5 0 R >> >>",
    rotate: 90,
  };
  const file = new File([pdf([page], [FONT, IMAGE])], "landscape.pdf", { type: "application/pdf" });
  assert.equal(await extractPdfAttachmentText(file), "Invoice total");
  assert.equal(requests.length, 1);
});

test("a typed title over a background image with a footer stays in the browser", async () => {
  const requests = serve({ text: "unused", truncated: false });
  const slide = {
    content: "q 200 0 0 200 0 0 cm /Im1 Do Q BT /F1 24 Tf 20 100 Td (Agenda) Tj ET BT /F1 8 Tf 10 5 Td (Company Confidential 2026) Tj ET",
    resources: "<< /XObject << /Im1 6 0 R >> /Font << /F1 5 0 R >> >>",
  };
  const deck = new File([pdf([slide], [FONT, IMAGE])], "deck.pdf", { type: "application/pdf" });
  assert.equal(await extractPdfAttachmentText(deck), "Agenda\nCompany Confidential 2026");
  assert.deepEqual(requests, []);
});

test("a scan drawn under nested transforms is still read by the server", async () => {
  const requests = serve({ filename: "nested.pdf", text: "Invoice total", truncated: false });
  const nested = {
    content: "q 1 0 0 1 50 50 cm q 150 0 0 150 0 0 cm /Im1 Do Q Q BT /F1 8 Tf 10 5 Td (ACME Corp Confidential - Page 1 of 1) Tj ET",
    resources: "<< /XObject << /Im1 6 0 R >> /Font << /F1 5 0 R >> >>",
  };
  const file = new File([pdf([nested], [FONT, IMAGE])], "nested.pdf", { type: "application/pdf" });
  assert.equal(await extractPdfAttachmentText(file), "Invoice total");
  assert.equal(requests.length, 1);
});

test("a scan placed through a form's matrix is still read by the server", async () => {
  const requests = serve({ filename: "form.pdf", text: "Invoice total", truncated: false });
  const page = {
    content: "/Fm1 Do BT /F1 8 Tf 10 5 Td (ACME Corp Confidential - Page 1 of 1) Tj ET",
    resources: "<< /XObject << /Fm1 7 0 R >> /Font << /F1 5 0 R >> >>",
  };
  const form = "<< /Type /XObject /Subtype /Form /BBox [0 0 1 1] /Matrix [200 0 0 200 0 0] /Resources << /XObject << /Im1 6 0 R >> >> /Length 7 >>\nstream\n/Im1 Do\nendstream";
  const file = new File([pdf([page], [FONT, IMAGE, form])], "form.pdf", { type: "application/pdf" });
  assert.equal(await extractPdfAttachmentText(file), "Invoice total");
  assert.equal(requests.length, 1);
});

test("a page that is only a small image is still read by the server", async () => {
  const requests = serve({ filename: "receipt.pdf", text: "Receipt", truncated: false });
  const receipt = { content: "q 80 0 0 80 60 60 cm /Im1 Do Q", resources: "<< /XObject << /Im1 5 0 R >> >>" };
  const file = new File([pdf([receipt], [IMAGE])], "receipt.pdf", { type: "application/pdf" });
  assert.equal(await extractPdfAttachmentText(file), "Receipt");
  assert.equal(requests.length, 1);
});

// The leading q/Q scales nothing on the page; it is there so an ignored restore would enlarge the logo.
for (const [what, image] of [
  ["a logo", "q 10 0 0 10 0 0 cm Q q 20 0 0 20 5 5 cm"],
  ["an image hanging off the page", "q 400 0 0 400 190 190 cm"],
] as const) {
  test(`${what} on a typed cover does not send the deck to the server`, async () => {
    const requests = serve({ text: "unused", truncated: false });
    const cover = {
      content: `${image} /Im1 Do Q BT /F1 8 Tf 10 190 Td (Northwind Quarterly Review) Tj ET`,
      resources: "<< /XObject << /Im1 6 0 R >> /Font << /F1 5 0 R >> >>",
    };
    const deck = new File([pdf([cover], [FONT, IMAGE])], "deck.pdf", { type: "application/pdf" });
    assert.equal(await extractPdfAttachmentText(deck), "Northwind Quarterly Review");
    assert.deepEqual(requests, []);
  });
}

test("a scan stored as a stamp annotation is still read by the server", async () => {
  const requests = serve({ filename: "stamp.pdf", text: "Invoice total", truncated: false });
  const page = {
    content: "BT /F1 8 Tf 10 5 Td (ACME Corp Confidential - Page 1 of 1) Tj ET",
    resources: "<< /Font << /F1 5 0 R >> >>",
    annots: "[7 0 R]",
  };
  const stamp = "<< /Type /Annot /Subtype /Stamp /Rect [0 0 200 200] /AP << /N 8 0 R >> >>";
  const look = "<< /Type /XObject /Subtype /Form /BBox [0 0 1 1] /Resources << /XObject << /Im1 6 0 R >> >> /Length 7 >>\nstream\n/Im1 Do\nendstream";
  const file = new File([pdf([page], [FONT, IMAGE, stamp, look])], "stamp.pdf", { type: "application/pdf" });
  assert.equal(await extractPdfAttachmentText(file), "Invoice total");
  assert.equal(requests.length, 1);
});

test("an image stamp annotation on a typed cover does not send the deck to the server", async () => {
  const requests = serve({ text: "unused", truncated: false });
  const cover = {
    content: "BT /F1 8 Tf 10 190 Td (Northwind Quarterly Review) Tj ET",
    resources: "<< /Font << /F1 5 0 R >> >>",
    annots: "[7 0 R]",
  };
  const stamp = "<< /Type /Annot /Subtype /Stamp /Rect [180 180 190 190] /AP << /N 8 0 R >> >>";
  const look = "<< /Type /XObject /Subtype /Form /BBox [0 0 200 200] /Resources << /XObject << /Im1 6 0 R >> >> /Length 26 >>\nstream\n200 0 0 200 0 0 cm /Im1 Do\nendstream";
  const deck = new File([pdf([cover], [FONT, IMAGE, stamp, look])], "deck.pdf", { type: "application/pdf" });
  assert.equal(await extractPdfAttachmentText(deck), "Northwind Quarterly Review");
  assert.deepEqual(requests, []);
});

test("a blank page with no image stays in the browser", async () => {
  const requests = serve({ text: "unused", truncated: false });
  const blank = new File(
    [pdf([typedPage("page one", 7), { content: "", resources: "<< >>" }], [FONT])],
    "blank.pdf",
    { type: "application/pdf" },
  );
  assert.equal(await extractPdfAttachmentText(blank), "page one");
  assert.deepEqual(requests, []);
});

test("a pdf the server cannot read keeps the text the browser found", async () => {
  serve({ detail: "contract.pdf could not be read: no OCR" }, 422);
  assert.equal(await extractPdfAttachmentText(contract()), "Contract terms");
});
