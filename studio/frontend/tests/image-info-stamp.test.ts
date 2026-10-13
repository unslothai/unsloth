// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  drawInfoStamp,
  infoStampText,
  layoutInfoStamp,
  stampModelName,
  type StampContext,
} from "../src/features/images/lib/info-stamp.ts";

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
