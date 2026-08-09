// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  DATASET_IMAGE_EXTS,
  DATASET_TEXT_EXTS,
  chunkDatasetUpload,
  filesFromDataTransfer,
  metadataKeyedOnSubfolders,
  selectDatasetFiles,
} from "../src/features/images/train/dataset-files.ts";

/** a picked file of a given byte size, for the chunking tests. */
function sized(name: string, bytes: number): File {
  return new File([new Uint8Array(bytes)], name);
}

/** a picked file; `path` stands in for the webkitRelativePath a folder pick carries. */
function picked(name: string, path?: string): File {
  const file = new File(["x"], name);
  if (path !== undefined) {
    Object.defineProperty(file, "webkitRelativePath", { value: path });
  }
  return file;
}

test("accepts exactly the extensions the backend accepts", async () => {
  const source = await readFile(
    new URL("../../backend/routes/training.py", import.meta.url),
    "utf8",
  );
  const literals = (name: string) => {
    const match = new RegExp(`${name}\\s*=\\s*\\{([^}]+)\\}`).exec(source);
    assert.ok(match, `${name} not found in training.py`);
    return [...match[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]).sort();
  };

  // .caption sidecars and metadata/captions .jsonl are older caption formats the trainer
  // still reads, so a backend that widens either list must widen the picker with it.
  assert.deepEqual([...DATASET_IMAGE_EXTS].sort(), literals("_DIFFUSION_DATASET_IMAGE_EXTS"));
  assert.deepEqual([...DATASET_TEXT_EXTS].sort(), literals("_DIFFUSION_DATASET_TEXT_EXTS"));
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
    picked("clip.mp4"),
    picked("README"),
  ]);

  assert.deepEqual(result.files.map((f) => f.name), ["cat.png"]);
  assert.equal(result.skipped, 3);
});

test("reports basenames a folder pick would flatten together", () => {
  // a dataset folder is flat, so both of these would be stored as cat.png.
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
  // training.py refuses the whole batch on this; catching it here keeps the upload honest.
  const result = selectDatasetFiles([picked("cat.png"), picked("cat.jpg"), picked("cat.txt")]);

  assert.deepEqual(result.files.map((f) => f.name), ["cat.png", "cat.txt"]);
  assert.deepEqual(result.collisions, [{ kind: "stem", first: "cat.png", second: "cat.jpg" }]);
});

test("leaves case-variant names to the backend, which knows if the filesystem folds case", () => {
  // on ext4 these are two distinct files the upload accepts; refusing here would block a
  // legitimate dataset and give a reason that is false for that filesystem.
  const result = selectDatasetFiles([picked("Cat.png"), picked("cat.PNG")]);

  assert.equal(result.files.length, 2);
  assert.deepEqual(result.collisions, []);
});

test("folds case on the stem rule, which training.py applies whatever the filesystem does", () => {
  const result = selectDatasetFiles([picked("Cat.png"), picked("cat.jpg")]);

  assert.deepEqual(result.collisions, [{ kind: "stem", first: "Cat.png", second: "cat.jpg" }]);
});

test("clashes an extension-case pair of one stem spelling, as _shares_sidecar does", () => {
  // cat.png and cat.PNG share the exact stem, so both resolve to cat.txt and the backend
  // rejects them on every filesystem. Cat.png and cat.PNG differ in stem case and are exempt.
  assert.deepEqual(selectDatasetFiles([picked("cat.png"), picked("cat.PNG")]).collisions, [
    { kind: "stem", first: "cat.png", second: "cat.PNG" },
  ]);
  assert.deepEqual(selectDatasetFiles([picked("Cat.png"), picked("cat.PNG")]).collisions, []);
});

test("compares every accepted variant, since the stem exemption is not transitive", () => {
  // Cat.png exempts cat.PNG and cat.png individually, but those two share an exact stem and
  // training.py compares each new name against every earlier one.
  const result = selectDatasetFiles([picked("Cat.png"), picked("cat.PNG"), picked("cat.png")]);

  assert.deepEqual(result.collisions, [
    { kind: "stem", first: "cat.PNG", second: "cat.png" },
  ]);
});

test("skips names the upload refuses outright, before any slice is committed", () => {
  // Path(".png").suffix is empty and the endpoint rejects any name holding "..", so accepting
  // either here would commit earlier slices and then fail on a later one.
  const result = selectDatasetFiles([
    picked("cat.png"),
    picked(".png"),
    picked("photo..png"),
  ]);

  assert.deepEqual(result.files.map((f) => f.name), ["cat.png"]);
  assert.equal(result.skipped, 2);
});

test("keeps a dotfile named in the file dialog, where the user chose it deliberately", () => {
  // no webkitRelativePath means a plain multi-select, unlike a tree walk that surfaces .thumbs.
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
  // hidden entries are not "unsupported", so they are not reported as skipped.
  assert.equal(result.skipped, 0);
  assert.deepEqual(result.collisions, []);
});

test("keeps a picked folder whose own name starts with a dot", () => {
  // webkitRelativePath is rooted at the folder the user chose, so its name is not a reason to skip.
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

// -- drop handling -------------------------------------------------------------------------

/** a FileSystemFileEntry over one File. */
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

/** a FileSystemDirectoryEntry whose reader hands back `children` in 100-entry batches. */
function dirEntry(name: string, children: FileSystemEntry[]) {
  return {
    isFile: false,
    isDirectory: true,
    name,
    createReader() {
      let cursor = 0;
      return {
        readEntries(resolve: (batch: FileSystemEntry[]) => void) {
          // mirrors Chrome, which never returns more than 100 entries per call.
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

  // a single readEntries call would stop at 100 and silently drop the rest.
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
  // training.py stores Path(name).name.strip(), so " cat.png" and "cat.png" are one file.
  const result = selectDatasetFiles([picked(" cat.png"), picked("cat.png")]);

  assert.equal(result.files.length, 1);
  assert.equal(result.collisions.length, 1);

  // and the same normalisation groups them into one request rather than two repeat uploads.
  const chunks = chunkDatasetUpload([sized(" cat.png", 1), sized("cat.png", 1)], 1024 * 1024);
  assert.equal(chunks.length, 1);
});

test("rejects a drop whose items do not all resolve to entries", async () => {
  // a partly resolvable drop would otherwise upload a subset under an all-or-nothing contract.
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

  // a plain 500-file slice would put Cat.png and cat.png in separate repeat uploads, where the
  // second silently replaces the first on a case-insensitive dataset folder.
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

test("flags metadata keyed on a subfolder, which flattening would silently unmatch", async () => {
  const meta = new File(
    ['{"file_name": "images/001.png", "text": "a cat"}\n{"file_name": "images/002.png"}\n'],
    "metadata.jsonl",
  );
  assert.equal(await metadataKeyedOnSubfolders([meta]), "metadata.jsonl");

  const flat = new File(['{"file_name": "001.png", "text": "a cat"}\n'], "metadata.jsonl");
  assert.equal(await metadataKeyedOnSubfolders([flat]), null);

  // metadata written on windows keys rows with a backslash
  const win = new File(['{"file_name": "images\\\\001.png"}\n'], "metadata.jsonl");
  assert.equal(await metadataKeyedOnSubfolders([win]), "metadata.jsonl");

  // _load_metadata_captions falls back on the first TRUTHY key, so an empty file_name defers
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
