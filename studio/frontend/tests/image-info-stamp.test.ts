// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  copyPngTextChunks,
  drawInfoStamp,
  infoStampText,
  layoutInfoStamp,
  stampModelName,
  type StampContext,
} from "../src/features/images/lib/info-stamp.ts";
import { readSrc } from "./helpers/kit.ts";

function fakeContext() {
  const calls: { op: string; args: unknown[] }[] = [];
  const ctx: StampContext = {
    font: "",
    fillStyle: "",
    textBaseline: "alphabetic",
    textAlign: "start",
    fillRect: (...args) => calls.push({ op: "fillRect", args }),
    fillText: (...args) => calls.push({ op: "fillText", args }),
    // Monospace-ish: 0.6 em per character.
    measureText(text: string) {
      const px = Number.parseFloat(this.font);
      return { width: text.length * px * 0.6 };
    },
  };
  return { ctx, calls };
}

test("model name is the repo basename without the GGUF suffix", () => {
  assert.equal(
    stampModelName("unsloth/FLUX.2-klein-4B-GGUF"),
    "FLUX.2-klein-4B",
  );
  assert.equal(stampModelName("Tongyi-MAI/Z-Image-Turbo"), "Z-Image-Turbo");
  assert.equal(
    stampModelName("C:\\models\\z-image-Q4_K_M.gguf"),
    "z-image-Q4_K_M",
  );
  assert.equal(stampModelName(null), "");
});

test("stamp text is model, local date time and seed", () => {
  const created = new Date(2026, 7, 9, 13, 1, 25).getTime() / 1000;
  assert.equal(
    infoStampText({
      model: "unsloth/FLUX.2-klein-4B-GGUF",
      seed: 6630007,
      created_at: created,
    }),
    "FLUX.2-klein-4B | 2026-08-09 13:01:25 | Seed: 6630007",
  );
  assert.equal(
    infoStampText({ model: null, seed: 7, created_at: created }),
    "2026-08-09 13:01:25 | Seed: 7",
  );
});

test("a typical 1024 image gets the full line plus the brand on a bottom bar", () => {
  const { ctx, calls } = fakeContext();
  const text = "FLUX.2-klein-4B | 2026-08-09 13:01:25 | Seed: 6630007";
  const layout = drawInfoStamp(ctx, 1024, 1024, text);
  assert.equal(layout.showBrand, true);
  const bar = calls.find((c) => c.op === "fillRect");
  assert.deepEqual(bar?.args, [
    0,
    1024 - layout.barHeight,
    1024,
    layout.barHeight,
  ]);
  const texts = calls.filter((c) => c.op === "fillText").map((c) => c.args[0]);
  assert.deepEqual(texts, [text, "Unsloth"]);
});

test("a narrow image drops the brand and shrinks the line to fit", () => {
  const { ctx } = fakeContext();
  const text = "Qwen-Image-Edit-2509 | 2026-08-09 13:01:25 | Seed: 1234567890";
  const layout = layoutInfoStamp(ctx, 256, 256, text);
  assert.equal(layout.showBrand, false);
  ctx.font = `${layout.fontPx}px monospace`;
  assert.ok(ctx.measureText(text).width <= 256 - layout.padX * 2);
});

const SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

function chunk(type: string, data: number[] = []): number[] {
  const len = data.length;
  return [
    (len >>> 24) & 255,
    (len >>> 16) & 255,
    (len >>> 8) & 255,
    len & 255,
    ...[...type].map((c) => c.charCodeAt(0)),
    ...data,
    1,
    2,
    3,
    4,
  ];
}

const bytes = (s: string) => [...s].map((c) => c.charCodeAt(0));

function png(...chunks: number[][]): Uint8Array<ArrayBuffer> {
  return new Uint8Array([...SIG, ...chunks.flat()]);
}

function chunkTypes(p: Uint8Array): string[] {
  const types: string[] = [];
  let at = 8;
  while (at < p.length) {
    const len =
      (p[at] << 24) | (p[at + 1] << 16) | (p[at + 2] << 8) | p[at + 3];
    types.push(String.fromCharCode(...p.subarray(at + 4, at + 8)));
    at += 12 + len;
  }
  return types;
}

test("the stamped PNG keeps the original's recipe chunks, once", () => {
  const original = png(
    chunk("IHDR", [0, 0, 0, 1]),
    chunk("tEXt", bytes("unsloth\0{}")),
    chunk("tEXt", bytes("parameters\0a sloth")),
    chunk("IDAT", [9, 9]),
    chunk("IEND"),
  );
  const canvasPng = png(
    chunk("IHDR", [0, 0, 0, 2]),
    chunk("IDAT", [7]),
    chunk("IEND"),
  );
  const once = copyPngTextChunks(original, canvasPng);
  assert.deepEqual(chunkTypes(once), ["IHDR", "tEXt", "tEXt", "IDAT", "IEND"]);
  // The stamped pixels and header are the canvas's, not the original's.
  assert.ok(Buffer.from(once).includes(Buffer.from([0, 0, 0, 2])));
  assert.ok(
    Buffer.from(once).includes(Buffer.from(bytes("parameters\0a sloth"))),
  );
  assert.deepEqual(copyPngTextChunks(original, once), once);
});

test("an unparsable PNG on either side leaves the canvas output untouched", () => {
  const canvasPng = png(chunk("IHDR", [1]), chunk("IDAT", [7]), chunk("IEND"));
  assert.equal(
    copyPngTextChunks(new Uint8Array([1, 2, 3]), canvasPng),
    canvasPng,
  );
  const truncated = canvasPng.subarray(0, canvasPng.length - 3);
  assert.equal(copyPngTextChunks(canvasPng, truncated), truncated);
});

test("the existing download choices are unchanged and the stamp is a separate item", () => {
  const page = readSrc("features/images/images-page.tsx");
  for (const [label, format] of [
    ["PNG (original, keeps recipe)", "png"],
    ["JPEG (smaller)", "jpeg"],
    ["WebP", "webp"],
  ]) {
    const at = page.indexOf(`>\n                        ${label}\n`);
    assert.ok(at > 0, `${label} item moved`);
    const call = page.lastIndexOf("onClick", at);
    assert.ok(
      page
        .slice(call, at)
        .includes(`downloadImage(selectedSrc, selected, "${format}")`),
      `${label} no longer downloads ${format}`,
    );
  }
  assert.ok(page.includes("downloadStampedImage(selectedSrc, selected)"));
  // The one-click quick download still saves the untouched original.
  assert.match(page, /await downloadImage\(src, image, "png"\);/);
});
