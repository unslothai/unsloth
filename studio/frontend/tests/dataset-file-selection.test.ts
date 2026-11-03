// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  DATASET_CLIP_EXTS,
  DATASET_IMAGE_EXTS,
  DATASET_MEDIA_EXTS,
  DATASET_TEXT_EXTS,
  chunkDatasetUpload,
  existingStemClash,
  filesFromDataTransfer,
  metadataKeyedOnSubfolders,
  oversizedChunk,
  selectDatasetFiles,
} from "../src/features/images/train/dataset-files.ts";

import { readSrcAsync, readText } from "./helpers/kit.ts";

/** Payload is stubbed because chunking reads only `size`. */
function sized(name: string, bytes: number): File {
  const file = new File([], name);
  Object.defineProperty(file, "size", { value: bytes });
  return file;
}

function picked(name: string, path?: string): File {
  const file = new File(["x"], name);
  if (path !== undefined) {
    Object.defineProperty(file, "webkitRelativePath", { value: path });
  }
  return file;
}

test("accepts exactly the extensions the backend accepts", async () => {
  const source = readText("../../backend/routes/training.py");
  const literals = (name: string) => {
    const match = new RegExp(`${name}\\s*=\\s*\\{([^}]+)\\}`).exec(source);
    assert.ok(match, `${name} not found in training.py`);
    return [...match[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]).sort();
  };

  assert.deepEqual([...DATASET_IMAGE_EXTS].sort(), literals("_DIFFUSION_DATASET_IMAGE_EXTS"));
  assert.deepEqual([...DATASET_TEXT_EXTS].sort(), literals("_DIFFUSION_DATASET_TEXT_EXTS"));

  const clipSource = readText("../../backend/core/training/diffusion_clip_formats.py");
  const clipMatch = /CLIP_EXTS\s*=\s*frozenset\(\{([^}]+)\}\)/.exec(clipSource);
  assert.ok(clipMatch, "CLIP_EXTS not found in diffusion_clip_formats.py");
  assert.deepEqual(
    [...DATASET_CLIP_EXTS].sort(),
    [...clipMatch[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]).sort(),
  );

  const overlap = DATASET_IMAGE_EXTS.filter((e) => DATASET_CLIP_EXTS.includes(e));
  assert.deepEqual(overlap, []);
  assert.equal(DATASET_MEDIA_EXTS.length, DATASET_IMAGE_EXTS.length + DATASET_CLIP_EXTS.length);
});

test("keeps images alongside the caption files paired to them", () => {
  const result = selectDatasetFiles([
    picked("cat.png"),
    picked("cat.txt"),
    picked("dog.JPEG"),
    picked("dog.caption"),
    picked("metadata.jsonl"),
  ]);

  assert.equal(result.files.length, 5);
  assert.equal(result.imageCount, 2);
  assert.equal(result.captionCount, 3);
  assert.equal(result.skipped, 0);
  assert.deepEqual(result.collisions, []);
});

test("drops files the upload endpoint would reject, and counts them", () => {
  const result = selectDatasetFiles([
    picked("cat.png"),
    picked("notes.pdf"),
    picked("README"),
  ]);

  assert.deepEqual(result.files.map((f) => f.name), ["cat.png"]);
  assert.equal(result.skipped, 2);
});

test("keeps clips alongside the caption files paired to them", () => {
  const result = selectDatasetFiles([
    picked("clip.mp4"),
    picked("clip.txt"),
    picked("second.MOV"),
    picked("metadata.jsonl"),
  ]);

  assert.equal(result.files.length, 4);
  assert.equal(result.clipCount, 2);
  assert.equal(result.imageCount, 0);
  assert.equal(result.captionCount, 2);
  assert.equal(result.skipped, 0);
  assert.deepEqual(result.collisions, []);
});

test("reports an image and a clip sharing a stem, which would share one caption sidecar", () => {
  const result = selectDatasetFiles([picked("cat.png"), picked("cat.mp4"), picked("cat.txt")]);

  assert.deepEqual(result.files.map((f) => f.name), ["cat.png", "cat.txt"]);
  assert.deepEqual(result.collisions, [{ kind: "stem", first: "cat.png", second: "cat.mp4" }]);
});

test("a clip already in the folder holds its sidecar against a new image of that stem", () => {
  const clash = existingStemClash([picked("cat.png")], ["cat.mp4"]);
  assert.deepEqual(clash, { kind: "stem", first: "cat.mp4", second: "cat.png" });
});

test("reports basenames a folder pick would flatten together", () => {
  const result = selectDatasetFiles([
    picked("cat.png", "set/train/cat.png"),
    picked("cat.png", "set/val/cat.png"),
  ]);

  assert.equal(result.files.length, 1);
  assert.deepEqual(result.collisions, [
    { kind: "name", first: "set/train/cat.png", second: "set/val/cat.png" },
  ]);
});

test("reports two images sharing a stem, which would share one caption sidecar", () => {
  const result = selectDatasetFiles([picked("cat.png"), picked("cat.jpg"), picked("cat.txt")]);

  assert.deepEqual(result.files.map((f) => f.name), ["cat.png", "cat.txt"]);
  assert.deepEqual(result.collisions, [{ kind: "stem", first: "cat.png", second: "cat.jpg" }]);
});

test("leaves case-variant names to the backend, which knows if the filesystem folds case", () => {
  const result = selectDatasetFiles([picked("Cat.png"), picked("cat.PNG")]);

  assert.equal(result.files.length, 2);
  assert.deepEqual(result.collisions, []);
});

test("folds case on the stem rule, which training.py applies whatever the filesystem does", () => {
  const result = selectDatasetFiles([picked("Cat.png"), picked("cat.jpg")]);

  assert.deepEqual(result.collisions, [{ kind: "stem", first: "Cat.png", second: "cat.jpg" }]);
});

test("clashes an extension-case pair of one stem spelling, as _shares_sidecar does", () => {
  assert.deepEqual(selectDatasetFiles([picked("cat.png"), picked("cat.PNG")]).collisions, [
    { kind: "stem", first: "cat.png", second: "cat.PNG" },
  ]);
  assert.deepEqual(selectDatasetFiles([picked("Cat.png"), picked("cat.PNG")]).collisions, []);
});

test("compares every accepted variant, since the stem exemption is not transitive", () => {
  const result = selectDatasetFiles([picked("Cat.png"), picked("cat.PNG"), picked("cat.png")]);

  assert.deepEqual(result.collisions, [
    { kind: "stem", first: "cat.PNG", second: "cat.png" },
  ]);
});

test("skips names the upload refuses outright, before any slice is committed", () => {
  const result = selectDatasetFiles([
    picked("cat.png"),
    picked(".png"),
    picked("photo..png"),
  ]);

  assert.deepEqual(result.files.map((f) => f.name), ["cat.png"]);
  assert.equal(result.skipped, 2);
});

test("keeps a dotfile named in the file dialog, where the user chose it deliberately", () => {
  const result = selectDatasetFiles([picked(".cover.png"), picked("cat.png")]);

  assert.deepEqual(result.files.map((f) => f.name), [".cover.png", "cat.png"]);
  assert.equal(result.skipped, 0);
});

test("ignores dot-directories, so re-picking a dataset folder skips its .thumbs cache", () => {
  const result = selectDatasetFiles([
    picked("cat.png", "my-photos/cat.png"),
    picked("cat.png_256.jpg", "my-photos/.thumbs/cat.png_256.jpg"),
    picked(".DS_Store", "my-photos/.DS_Store"),
  ]);

  assert.deepEqual(result.files.map((f) => f.name), ["cat.png"]);
  assert.equal(result.skipped, 0);
  assert.deepEqual(result.collisions, []);
});

test("keeps a picked folder whose own name starts with a dot", () => {
  const result = selectDatasetFiles([
    picked("cat.png", ".photos/cat.png"),
    picked("dog.png", ".photos/nested/dog.png"),
  ]);

  assert.deepEqual(result.files.map((f) => f.name), ["cat.png", "dog.png"]);
});

test("an empty or fully rejected pick yields no files", () => {
  assert.equal(selectDatasetFiles([]).files.length, 0);
  assert.equal(selectDatasetFiles([picked("notes.pdf")]).files.length, 0);
});

function fileEntry(name: string, onFile?: () => void) {
  return {
    isFile: true,
    isDirectory: false,
    name,
    file(resolve: (f: File) => void, reject: (e: Error) => void) {
      onFile?.();
      if (name === "explode.png") reject(new Error("unreadable"));
      else resolve(new File(["x"], name));
    },
  } as unknown as FileSystemEntry;
}

function dirEntry(name: string, children: FileSystemEntry[]) {
  return {
    isFile: false,
    isDirectory: true,
    name,
    createReader() {
      let cursor = 0;
      return {
        readEntries(resolve: (batch: FileSystemEntry[]) => void) {
          // Chrome never returns more than 100 entries per readEntries call.
          const batch = children.slice(cursor, cursor + 100);
          cursor += batch.length;
          resolve(batch);
        },
      };
    },
  } as unknown as FileSystemEntry;
}

function transfer(entries: FileSystemEntry[], files: File[] = []): DataTransfer {
  return {
    items: entries.map((entry) => ({ kind: "file", webkitGetAsEntry: () => entry })),
    files,
  } as unknown as DataTransfer;
}

test("reads a dropped folder past the 100-entry readEntries batch limit", async () => {
  const children = Array.from({ length: 250 }, (_, i) => fileEntry(`img_${i}.png`));
  const out = await filesFromDataTransfer(transfer([dirEntry("set", children)]));

  assert.equal(out.length, 250);
});

test("gives dropped files their folder path, so a collision names both sides", async () => {
  const out = await filesFromDataTransfer(
    transfer([
      dirEntry("set", [
        dirEntry("train", [fileEntry("cat.png")]),
        dirEntry("val", [fileEntry("cat.png")]),
      ]),
    ]),
  );

  assert.deepEqual(
    out.map((f) => (f as File & { webkitRelativePath?: string }).webkitRelativePath),
    ["set/train/cat.png", "set/val/cat.png"],
  );
  assert.deepEqual(selectDatasetFiles(out).collisions, [
    { kind: "name", first: "set/train/cat.png", second: "set/val/cat.png" },
  ]);
});

test("normalizes to the stored name, so a leading space is not a second destination", () => {
  const result = selectDatasetFiles([picked(" cat.png"), picked("cat.png")]);

  assert.equal(result.files.length, 1);
  assert.equal(result.collisions.length, 1);

  const chunks = chunkDatasetUpload([sized(" cat.png", 1), sized("cat.png", 1)], 1024 * 1024);
  assert.equal(chunks.length, 1);
});

test("rejects a drop whose items do not all resolve to entries", async () => {
  const dt = {
    items: [
      { kind: "file", webkitGetAsEntry: () => fileEntry("ok.png") },
      { kind: "file", webkitGetAsEntry: () => null },
    ],
    files: [new File(["x"], "ok.png"), new File(["x"], "ghost.png")],
  } as unknown as DataTransfer;

  await assert.rejects(filesFromDataTransfer(dt), /could not be read/);
});

test("keeps casefold-equal names in one request, which the backend can only compare there", () => {
  const files = [
    ...Array.from({ length: 499 }, (_, i) => sized(`img_${i}.png`, 1)),
    sized("Cat.png", 1),
    sized("cat.png", 1),
  ];
  const chunks = chunkDatasetUpload(files, 1024 * 1024 * 1024);

  const holding = chunks.filter((c) => c.some((f) => f.name.toLowerCase() === "cat.png"));
  assert.equal(holding.length, 1);
  assert.equal(holding[0].filter((f) => f.name.toLowerCase() === "cat.png").length, 2);
  for (const chunk of chunks) assert.ok(chunk.length <= 500 + 1);
});

test("splits on the byte cap too, not only the part count", () => {
  const mb = 1024 * 1024;
  const files = Array.from({ length: 300 }, (_, i) => sized(`img_${i}.png`, 2 * mb));
  const chunks = chunkDatasetUpload(files, 500 * mb);

  assert.ok(chunks.length > 1, "600MB under the part cap must still be split");
  for (const chunk of chunks) {
    assert.ok(chunk.reduce((n, f) => n + f.size, 0) <= 500 * mb);
  }
  assert.equal(chunks.flat().length, 300);
});

test("strips what Python strips, so a name the endpoint refuses is not read as accepted", () => {
  // trim() removes U+FEFF but Python str.strip() keeps it, so the backend sees a non-image suffix.
  const bom = selectDatasetFiles([picked("cat.png\ufeff")]);
  assert.deepEqual(bom.files, []);
  assert.equal(bom.skipped, 1);

  // str.strip() removes these but trim() does not, so the backend stores a plain cat.png.
  for (const ch of ["\u001c", "\u001d", "\u001e", "\u001f", "\u0085"]) {
    const sel = selectDatasetFiles([picked(`cat.png${ch}`)]);
    assert.equal(sel.imageCount, 1, `U+${ch.codePointAt(0)?.toString(16)} must be stripped`);
    assert.equal(sel.skipped, 0);
  }

  assert.equal(selectDatasetFiles([picked(" cat.png\t")]).imageCount, 1);
});

test("sends case-variant groups first, so their refusal lands before anything commits", () => {
  const mb = 1024 * 1024;
  const files = [
    ...Array.from({ length: 499 }, (_, i) => sized(`img_${i}.png`, mb)),
    sized("Cat.png", mb),
    sized("cat.png", mb),
  ];
  const chunks = chunkDatasetUpload(files, 500 * mb);

  assert.deepEqual(
    chunks[0].slice(0, 2).map((f) => f.name),
    ["Cat.png", "cat.png"],
    "the case-variant group must lead the first chunk",
  );
  assert.ok(chunks.length > 1, "501 files must still split on the part cap");
  assert.equal(chunks.flat().length, 501);
  assert.equal(new Set(chunks.flat()).size, 501);
});

test("names a chunk no split can fit, so the slices before it are never committed", () => {
  const mb = 1024 * 1024;
  const files = [
    ...Array.from({ length: 20 }, (_, i) => sized(`img_${i}.png`, 2 * mb)),
    sized("huge.png", 600 * mb),
  ];
  const chunks = chunkDatasetUpload(files, 500 * mb);

  assert.deepEqual(
    chunks.map((c) => c.length),
    [20, 1],
    "the oversized file must sit in a slice the endpoint would 413",
  );
  assert.equal(oversizedChunk(chunks, 500 * mb), "huge.png");
  assert.equal(oversizedChunk(chunks.slice(0, 1), 500 * mb), null);
});

test("reports a stem the dataset folder already holds, which a split top-up would 400", () => {
  const held = ["cat.png", "dog.png"];

  assert.deepEqual(existingStemClash([picked("cat.jpg", "top-up/cat.jpg")], held), {
    kind: "stem",
    first: "cat.png",
    second: "top-up/cat.jpg",
  });
  assert.equal(existingStemClash([picked("cat.png")], held), null);
  assert.equal(existingStemClash([picked("cat.PNG")], held)?.first, "cat.png");
  assert.equal(existingStemClash([picked("Cat.png")], held), null);
  assert.equal(existingStemClash([picked("cat.txt")], held), null);
  assert.equal(existingStemClash([picked("bird.png")], held), null);
});

test("flags metadata keyed on a subfolder, which flattening would silently unmatch", async () => {
  const meta = new File(
    ['{"file_name": "images/001.png", "text": "a cat"}\n{"file_name": "images/002.png"}\n'],
    "metadata.jsonl",
  );
  assert.equal(await metadataKeyedOnSubfolders([meta]), "metadata.jsonl");

  const flat = new File(['{"file_name": "001.png", "text": "a cat"}\n'], "metadata.jsonl");
  assert.equal(await metadataKeyedOnSubfolders([flat]), null);

  const win = new File(['{"file_name": "images\\\\001.png"}\n'], "metadata.jsonl");
  assert.equal(await metadataKeyedOnSubfolders([win]), "metadata.jsonl");

  const blank = new File(
    ['{"file_name": "", "image": "images/001.png"}\n'],
    "metadata.jsonl",
  );
  assert.equal(await metadataKeyedOnSubfolders([blank]), "metadata.jsonl");
});

test("refuses a partly read folder instead of uploading it as complete", async () => {
  await assert.rejects(
    filesFromDataTransfer(
      transfer([dirEntry("set", [fileEntry("ok.png"), fileEntry("explode.png")])], []),
    ),
    /unreadable/,
  );
});

test("uses the flat list when the entries API is unavailable", async () => {
  const flat = [new File(["x"], "a.png"), new File(["x"], "b.png")];
  const out = await filesFromDataTransfer({
    items: [{ kind: "file", webkitGetAsEntry: () => null }],
    files: flat,
  } as unknown as DataTransfer);

  assert.deepEqual(out.map((f) => f.name), ["a.png", "b.png"]);
});

test("the labeling grid is gated on images, not just its toggle", async () => {
  const source = await readSrcAsync("features/images/train/diffusion-train-panel.tsx");
  const guard = "{selectedDataset.image_count > 0 && (";
  const toggle = source.indexOf("<LabelingGridToggle");
  const grid = source.indexOf("<DatasetLabelingGrid");
  assert.ok(toggle > 0 && grid > 0);
  const start = source.lastIndexOf(guard, toggle);
  assert.ok(start > 0, "the toggle is not inside an image_count guard");
  let depth = 0;
  let end = -1;
  for (let i = start; i < source.length; i += 1) {
    if (source[i] === "{") depth += 1;
    else if (source[i] === "}") {
      depth -= 1;
      if (depth === 0) {
        end = i;
        break;
      }
    }
  }
  assert.ok(end > start, "unbalanced braces around the labeling grid guard");
  assert.ok(grid > start && grid < end, "the grid renders outside the image_count guard");
});
