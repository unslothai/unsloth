// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Binary document exports for the canvas download menu. The canvas has no
// HTML renderer available outside the browser sandbox, so these embed the
// raw canvas source verbatim (monospaced, one line per row/paragraph/line) —
// the same "keep the source, change the container" approach the .md and .txt
// formats already use — instead of attempting a lossy HTML→layout conversion.

// Word treats an empty paragraph as a blank line, so keep blank source lines
// instead of collapsing them.
function splitSourceLines(source: string): string[] {
  const lines = source.split("\n");
  return lines.length > 0 ? lines : [""];
}

export async function buildArtifactDocxBlob(
  title: string,
  source: string,
): Promise<Blob> {
  const { Document, Packer, Paragraph, TextRun, HeadingLevel } = await import(
    "docx"
  );
  const doc = new Document({
    sections: [
      {
        children: [
          new Paragraph({ text: title, heading: HeadingLevel.HEADING_1 }),
          ...splitSourceLines(source).map(
            (line) =>
              new Paragraph({
                children: [new TextRun({ text: line, font: "Courier New" })],
              }),
          ),
        ],
      },
    ],
  });
  return Packer.toBlob(doc);
}

export async function buildArtifactPptxBlob(
  title: string,
  source: string,
): Promise<Blob> {
  const { default: PptxGenJS } = await import("pptxgenjs");
  const pptx = new PptxGenJS();
  const lines = splitSourceLines(source);
  // Fit the deck in slide-sized chunks so a large canvas doesn't overflow one slide.
  const LINES_PER_SLIDE = 40;
  const chunks: string[][] = [];
  for (let i = 0; i < lines.length; i += LINES_PER_SLIDE) {
    chunks.push(lines.slice(i, i + LINES_PER_SLIDE));
  }
  if (chunks.length === 0) chunks.push([""]);

  chunks.forEach((chunk, index) => {
    const slide = pptx.addSlide();
    if (index === 0) {
      slide.addText(title, {
        x: 0.4,
        y: 0.3,
        w: "90%",
        h: 0.6,
        fontSize: 24,
        bold: true,
      });
    }
    slide.addText(chunk.join("\n"), {
      x: 0.4,
      y: index === 0 ? 1.0 : 0.4,
      w: "90%",
      h: index === 0 ? 4.2 : 4.8,
      fontSize: 10,
      fontFace: "Courier New",
      valign: "top",
    });
  });

  const arrayBuffer = (await pptx.write({
    outputType: "arraybuffer",
  })) as ArrayBuffer;
  return new Blob([arrayBuffer], {
    type: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  });
}

export async function buildArtifactXlsxBlob(
  title: string,
  source: string,
): Promise<Blob> {
  const ExcelJS = (await import("exceljs")).default;
  const workbook = new ExcelJS.Workbook();
  const sheet = workbook.addWorksheet(title.slice(0, 31) || "Canvas");
  sheet.columns = [{ header: "Line", width: 8 }, { header: "Source", width: 120 }];
  splitSourceLines(source).forEach((line, index) => {
    sheet.addRow([index + 1, line]);
  });
  const arrayBuffer = await workbook.xlsx.writeBuffer();
  return new Blob([arrayBuffer], {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });
}

export async function buildArtifactPdfBlob(
  title: string,
  source: string,
): Promise<Blob> {
  const { jsPDF } = await import("jspdf");
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const marginX = 40;
  let y = 50;
  const pageHeight = doc.internal.pageSize.getHeight();
  const lineHeight = 14;

  doc.setFont("helvetica", "bold");
  doc.setFontSize(16);
  doc.text(title, marginX, y);
  y += lineHeight * 2;

  doc.setFont("courier", "normal");
  doc.setFontSize(9);
  for (const rawLine of splitSourceLines(source)) {
    // Wrap long source lines to the page width instead of clipping them.
    const wrapped = doc.splitTextToSize(rawLine || " ", 515) as string[];
    for (const line of wrapped) {
      if (y > pageHeight - 40) {
        doc.addPage();
        y = 50;
      }
      doc.text(line, marginX, y);
      y += lineHeight;
    }
  }

  return doc.output("blob");
}
