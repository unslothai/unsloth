// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  buildExif,
  copyPngTextChunks,
  importedRecipe,
  parseA1111Parameters,
  parseComfyPrompt,
  pngTextChunks,
  readEmbeddedRecipe,
  recipeSettings,
  withJpegExif,
  withWebpExif,
} from "../src/features/images/lib/image-metadata.ts";
import { readSrc } from "./helpers/kit.ts";

// 8x8 files written by Pillow: a JFIF JPEG, a simple lossy WebP and a lossless WebP.
const JPEG_B64 =
  "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAAIAAgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwCzRRRXwR9sf//Z";
const WEBP_LOSSY_B64 =
  "UklGRjoAAABXRUJQVlA4IC4AAADQAQCdASoIAAgAAUAmJaACdLoB+AADsAD+pNf/TSPGkeNI+Yt/84ljqd3aAAAA";
const WEBP_LOSSLESS_B64 =
  "UklGRh4AAABXRUJQVlA4TBEAAAAvB8ABAAdQvCIXpf+BiOh/AAA=";

const bytes = (b64: string) => new Uint8Array(Buffer.from(b64, "base64"));

const RECIPE = JSON.stringify({
  prompt: "a sloth hiking, 日本語 ok",
  negative_prompt: "blurry",
  width: 1024,
  height: 768,
  steps: 4,
  guidance: 1,
  seed: 6630007,
  batch_seed: 6630007,
  batch_size: 1,
  model: "unsloth/FLUX.2-klein-4B-GGUF",
  workflow: "txt2img",
  loras: ["sloth-style:0.8"],
});
const PARAMETERS =
  "a sloth hiking, 日本語 ok\nNegative prompt: blurry\nSteps: 4, CFG scale: 1.0, Seed: 6630007, Size: 1024x768, Model: unsloth/FLUX.2-klein-4B-GGUF";

function chunk(type: string, data: number[] = []): number[] {
  const len = data.length;
  return [
    len >>> 24,
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
const latin1 = (s: string) => [...s].map((c) => c.charCodeAt(0));
const utf8 = (s: string) => [...new TextEncoder().encode(s)];
function png(...chunks: number[][]): Uint8Array<ArrayBuffer> {
  return new Uint8Array([
    0x89,
    0x50,
    0x4e,
    0x47,
    0x0d,
    0x0a,
    0x1a,
    0x0a,
    ...chunks.flat(),
  ]);
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

// Pillow writes tEXt for Latin-1 text and an uncompressed iTXt for anything else.
const galleryPng = () =>
  png(
    chunk("IHDR", [0, 0, 0, 1]),
    chunk("iTXt", [...latin1("unsloth\0\0\0\0\0"), ...utf8(RECIPE)]),
    chunk("iTXt", [...latin1("parameters\0\0\0\0\0"), ...utf8(PARAMETERS)]),
    chunk("IDAT", [9]),
    chunk("IEND"),
  );

test("PNG text chunks read back, tEXt and uncompressed iTXt", () => {
  const p = png(
    chunk("IHDR"),
    chunk("tEXt", latin1("parameters\0a sloth")),
    chunk("IEND"),
  );
  assert.deepEqual(pngTextChunks(p), { parameters: "a sloth" });
  assert.deepEqual(readEmbeddedRecipe(galleryPng()), {
    recipe: RECIPE,
    parameters: PARAMETERS,
  });
});

test("a re-encoded PNG keeps the original's recipe chunks, once", () => {
  const canvasPng = png(
    chunk("IHDR", [0, 0, 0, 2]),
    chunk("IDAT", [7]),
    chunk("IEND"),
  );
  const once = copyPngTextChunks(galleryPng(), canvasPng);
  assert.deepEqual(chunkTypes(once), ["IHDR", "iTXt", "iTXt", "IDAT", "IEND"]);
  assert.deepEqual(readEmbeddedRecipe(once), {
    recipe: RECIPE,
    parameters: PARAMETERS,
  });
  assert.deepEqual(copyPngTextChunks(galleryPng(), once), once);
  assert.equal(
    copyPngTextChunks(new Uint8Array([1, 2, 3]), canvasPng),
    canvasPng,
  );
});

test("JPEG export carries the recipe as EXIF after the JFIF header, idempotently", () => {
  const jpeg = bytes(JPEG_B64);
  const out = withJpegExif(jpeg, { recipe: RECIPE, parameters: PARAMETERS });
  assert.deepEqual(
    [...out.subarray(0, 4)],
    [0xff, 0xd8, 0xff, 0xe0],
    "SOI then APP0",
  );
  const app0End = 4 + ((out[4] << 8) | out[5]);
  assert.deepEqual(
    [...out.subarray(app0End, app0End + 2)],
    [0xff, 0xe1],
    "EXIF right after APP0",
  );
  assert.deepEqual(readEmbeddedRecipe(out), {
    recipe: RECIPE,
    parameters: PARAMETERS,
  });
  assert.ok(
    Buffer.from(out).subarray(-200).equals(Buffer.from(jpeg).subarray(-200)),
  );
  assert.deepEqual(
    withJpegExif(out, { recipe: RECIPE, parameters: PARAMETERS }),
    out,
  );
});

test("a recipe too big for one JPEG segment keeps the parameters text", () => {
  const huge = JSON.stringify({ prompt: "x".repeat(70_000) });
  const out = withJpegExif(bytes(JPEG_B64), {
    recipe: huge,
    parameters: PARAMETERS,
  });
  assert.deepEqual(readEmbeddedRecipe(out), { parameters: PARAMETERS });
});

for (const [name, b64] of [
  ["lossy", WEBP_LOSSY_B64],
  ["lossless", WEBP_LOSSLESS_B64],
] as const) {
  test(`WebP (${name}) export becomes VP8X with an EXIF chunk`, () => {
    const out = withWebpExif(
      bytes(b64),
      { recipe: RECIPE, parameters: PARAMETERS },
      8,
      8,
    );
    const text = Buffer.from(out).toString("latin1");
    assert.equal(text.slice(12, 16), "VP8X");
    assert.equal(out[20] & 0x08, 0x08, "EXIF flag");
    assert.equal(
      new DataView(out.buffer).getUint32(4, true),
      out.length - 8,
      "RIFF size",
    );
    assert.equal(
      (out[24] | (out[25] << 8) | (out[26] << 16)) + 1,
      8,
      "canvas width",
    );
    assert.deepEqual(readEmbeddedRecipe(out), {
      recipe: RECIPE,
      parameters: PARAMETERS,
    });
    assert.deepEqual(
      withWebpExif(out, { recipe: RECIPE, parameters: PARAMETERS }, 8, 8),
      out,
    );
  });
}

test("EXIF is big-endian with UNICODE UserComment, as piexif writes it", () => {
  const tiff = buildExif({ parameters: "hi" });
  assert.ok(tiff);
  assert.deepEqual([...tiff.subarray(0, 4)], [0x4d, 0x4d, 0, 0x2a]);
  const text = Buffer.from(tiff).toString("latin1");
  assert.ok(text.includes("UNICODE\0\0h\0i"));
  assert.equal(buildExif({}), null);
});

test("AUTOMATIC1111 parameters parse into settings", () => {
  const parsed = parseA1111Parameters(
    'masterpiece, a cat\nwearing a hat\nNegative prompt: lowres, bad hands\nSteps: 28, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 6.5, Seed: 1234567890, Size: 832x1216, Model hash: abc, Model: sdxl_base, Lora hashes: "a: 1, b: 2", Version: v1.10.1',
  );
  assert.deepEqual(parsed, {
    prompt: "masterpiece, a cat\nwearing a hat",
    negative_prompt: "lowres, bad hands",
    steps: 28,
    guidance: 6.5,
    seed: 1234567890,
    width: 832,
    height: 1216,
    model: "sdxl_base",
  });
  assert.equal(
    parseA1111Parameters("just a caption")?.prompt,
    "just a caption",
  );
});

test("Unsloth's recipe wins over the parameters text; neither gives null", () => {
  assert.equal(
    importedRecipe({ recipe: RECIPE, parameters: "other\nSteps: 1" })?.prompt,
    "a sloth hiking, 日本語 ok",
  );
  assert.equal(
    importedRecipe({ recipe: "{not json", parameters: PARAMETERS })?.steps,
    4,
  );
  assert.equal(importedRecipe({}), null);
  assert.equal(importedRecipe({ recipe: "[1,2]" }), null);
});

test("restorable settings are type-checked and fall back to the form's values", () => {
  const defaults = { width: 1024, height: 1024, steps: 8, guidance: 3.5 };
  const hostile = recipeSettings(
    {
      prompt: "p",
      width: "9999",
      steps: 2.5,
      guidance: Number.NaN,
      seed: "1",
      loras: [1, "a:1"],
      batch_size: 0,
    },
    defaults,
  );
  assert.equal(hostile.width, 1024);
  assert.equal(hostile.steps, 8);
  assert.equal(hostile.guidance, 3.5);
  assert.equal(hostile.seed, null);
  assert.deepEqual(hostile.loras, ["a:1"]);
  assert.equal(hostile.batch_size, 1);
  const full = recipeSettings(JSON.parse(RECIPE), defaults);
  assert.equal(full.seed, 6630007);
  assert.equal(full.height, 768);
  assert.deepEqual(full.loras, ["sloth-style:0.8"]);
});

test("the Images page wires downloads, the stamp setting and import", () => {
  const page = readSrc("features/images/images-page.tsx");
  for (const format of ["png", "jpeg", "webp"]) {
    assert.ok(
      page.includes(`downloadImage(selected, "${format}", selectedSrc)`),
      format,
    );
  }
  assert.match(page, /useLibrarySettingsStore\.getState\(\)\.imageInfoStamp/);
  assert.match(
    page,
    /\(image: GalleryImage\) => downloadImage\(image, "png", srcById\[image\.id\]\)/,
  );
  assert.match(
    page,
    /onImportFromImage=\{\(\) => importFileRef\.current\?\.click\(\)\}/,
  );
  assert.match(page, /onDrop=\{onImportDrop\}/);
  const store = readSrc("features/library/settings-store.ts");
  assert.match(store, /imageInfoStamp: true,/);
});

// ComfyUI's default text-to-image graph, as its SaveImage writes it into the `prompt` chunk, with
// the seed wired through a primitive node.
const COMFY_PROMPT = JSON.stringify({
  "3": {
    class_type: "KSampler",
    inputs: {
      seed: ["10", 0],
      steps: 20,
      cfg: 8,
      sampler_name: "euler",
      scheduler: "normal",
      denoise: 1,
      model: ["4", 0],
      positive: ["6", 0],
      negative: ["7", 0],
      latent_image: ["5", 0],
    },
  },
  "4": {
    class_type: "CheckpointLoaderSimple",
    inputs: { ckpt_name: "v1-5-pruned-emaonly.safetensors" },
  },
  "5": {
    class_type: "EmptyLatentImage",
    inputs: { width: 512, height: 768, batch_size: 1 },
  },
  "6": {
    class_type: "CLIPTextEncode",
    inputs: { text: "beautiful scenery nature glass bottle", clip: ["4", 1] },
  },
  "7": {
    class_type: "CLIPTextEncode",
    inputs: { text: "text, watermark", clip: ["4", 1] },
  },
  "8": {
    class_type: "VAEDecode",
    inputs: { samples: ["3", 0], vae: ["4", 2] },
  },
  "9": {
    class_type: "SaveImage",
    inputs: { filename_prefix: "ComfyUI", images: ["8", 0] },
  },
  "10": { class_type: "PrimitiveInt", inputs: { value: 156680208700286 } },
});

test("a ComfyUI PNG imports its prompt, seed, steps, CFG, size and checkpoint", () => {
  const comfyPng = png(
    chunk("IHDR"),
    chunk("tEXt", latin1(`prompt\0${COMFY_PROMPT}`)),
    chunk("tEXt", latin1('workflow\0{"nodes":[]}')),
    chunk("IEND"),
  );
  const embedded = readEmbeddedRecipe(comfyPng);
  assert.equal(embedded.comfyPrompt, COMFY_PROMPT);
  assert.deepEqual(importedRecipe(embedded), {
    prompt: "beautiful scenery nature glass bottle",
    negative_prompt: "text, watermark",
    seed: 156680208700286,
    steps: 20,
    guidance: 8,
    width: 512,
    height: 768,
    model: "v1-5-pruned-emaonly.safetensors",
  });
  assert.equal(
    parseComfyPrompt('{"1":{"class_type":"SaveImage","inputs":{}}}'),
    null,
  );
  assert.equal(parseComfyPrompt("not json"), null);
  // Unsloth's recipe still wins when a file somehow carries both.
  assert.equal(
    importedRecipe({ recipe: RECIPE, comfyPrompt: COMFY_PROMPT })?.steps,
    4,
  );
});

// A little-endian EXIF block holding one UserComment, as a non-piexif writer lays it out.
function littleEndianExifJpeg(comment: Uint8Array): Uint8Array<ArrayBuffer> {
  const tiff = new Uint8Array(8 + 18 + 18 + comment.length);
  const v = new DataView(tiff.buffer);
  tiff.set([0x49, 0x49, 0x2a, 0]);
  v.setUint32(4, 8, true);
  v.setUint16(8, 1, true);
  v.setUint16(10, 0x8769, true);
  v.setUint16(12, 4, true);
  v.setUint32(14, 1, true);
  v.setUint32(18, 26, true);
  v.setUint16(26, 1, true);
  v.setUint16(28, 0x9286, true);
  v.setUint16(30, 7, true);
  v.setUint32(32, comment.length, true);
  v.setUint32(36, 44, true);
  tiff.set(comment, 44);
  const length = 2 + 6 + tiff.length;
  return new Uint8Array([
    0xff,
    0xd8,
    0xff,
    0xe1,
    length >> 8,
    length & 0xff,
    0x45,
    0x78,
    0x69,
    0x66,
    0,
    0,
    ...tiff,
    0xff,
    0xd9,
  ]);
}

function utf16(text: string, bigEndian: boolean): Uint8Array {
  const out = new Uint8Array(8 + text.length * 2);
  out.set(latin1("UNICODE\0"));
  const v = new DataView(out.buffer);
  for (let i = 0; i < text.length; i++)
    v.setUint16(8 + i * 2, text.charCodeAt(i), !bigEndian);
  return out;
}

test("a UNICODE UserComment decodes in either byte order inside a little-endian EXIF", () => {
  for (const bigEndian of [true, false]) {
    const read = readEmbeddedRecipe(
      littleEndianExifJpeg(utf16(PARAMETERS, bigEndian)),
    );
    assert.equal(
      read.parameters,
      PARAMETERS,
      bigEndian ? "piexif-style UTF-16BE" : "byte-order UTF-16LE",
    );
  }
});

test("an imported reference count is capped at the request limit", () => {
  const defaults = { width: 1024, height: 1024, steps: 8, guidance: 3.5 };
  const capped = recipeSettings(
    { prompt: "p", workflow: "reference", reference_image_count: 1e9 },
    defaults,
  );
  assert.equal(capped.reference_image_count, 9);
  assert.equal(
    recipeSettings({ prompt: "p", reference_image_count: 3 }, defaults)
      .reference_image_count,
    3,
  );
});

test("the ComfyUI model is the loader feeding the sampler, through a LoRA loader", () => {
  const graph = JSON.parse(COMFY_PROMPT);
  // An unused checkpoint listed first, and the sampler fed by a LoRA on another UNet.
  const reordered = {
    "1": {
      class_type: "CheckpointLoaderSimple",
      inputs: { ckpt_name: "unused.safetensors" },
    },
    ...graph,
    "11": {
      class_type: "UNETLoader",
      inputs: { unet_name: "flux1-dev.safetensors" },
    },
    "12": {
      class_type: "LoraLoaderModelOnly",
      inputs: { model: ["11", 0], lora_name: "x.safetensors" },
    },
  };
  reordered["3"] = {
    ...graph["3"],
    inputs: { ...graph["3"].inputs, model: ["12", 0] },
  };
  assert.equal(
    parseComfyPrompt(JSON.stringify(reordered))?.model,
    "flux1-dev.safetensors",
  );
});

test("a file drop on the preview is always claimed before the image filter", () => {
  const page = readSrc("features/images/images-page.tsx");
  const start = page.indexOf("const onImportDrop");
  const drop = page.slice(
    start,
    page.indexOf("[importSettingsFromFile],", start),
  );
  assert.ok(
    drop.indexOf("event.preventDefault()") <
      drop.indexOf('f.type.startsWith("image/")'),
  );
  assert.match(drop, /toast\.error\(/);
  // Desktop: Tauri hands drops over as paths, so the preview is a native drop target too.
  assert.match(page, /ref=\{previewDropRef\}/);
  assert.match(
    page,
    /useNativeDropTarget\(\{ onDrop: \(paths\) => void importNativePath\(paths\[0\]\) \}\)/,
  );
});
