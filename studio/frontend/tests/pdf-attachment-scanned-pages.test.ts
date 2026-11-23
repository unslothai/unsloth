// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { extractPdfAttachmentText } = await import("../src/features/chat/attachment-content.ts");
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");
const { recordedToasts } = await import("./helpers/store-stubs/toast.ts");

function pdf(pages: { content: string; resources: string }[], extra: string[]): Uint8Array<ArrayBuffer> {
  const first = 3 + pages.length;
  const objects = [
    "<< /Type /Catalog /Pages 2 0 R >>",
    `<< /Type /Pages /Kids [${pages.map((_, i) => `${3 + i} 0 R`).join(" ")}] /Count ${pages.length} >>`,
    ...pages.map(
      ({ resources }, i) =>
        `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources ${resources} /Contents ${first + i} 0 R >>`,
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
const scannedPage = (image: number, stamp?: { words: string; font: number }) => ({
  content: `q 200 0 0 200 0 0 cm /Im1 Do Q${stamp ? ` BT /F1 8 Tf 90 5 Td (${stamp.words}) Tj ET` : ""}`,
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
    [pdf([typedPage("Contract terms", 7), scannedPage(8, { words: "Page 2", font: 7 })], [FONT, IMAGE])],
    "stamped.pdf",
    { type: "application/pdf" },
  );
  assert.equal(await extractPdfAttachmentText(stamped), "Contract terms\n\nANNEX B");
  assert.equal(requests.length, 1);
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
