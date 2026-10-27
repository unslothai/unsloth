// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

function editLibraryImageWith({ fail = false, epochAfterRead = 0 } = {}) {
  const offered: string[] = [];
  const navigations: unknown[] = [];
  const errors: unknown[] = [];
  let epoch = 0;
  const { editLibraryImage } = loadWithStubs<{
    editLibraryImage: (navigate: (to: unknown) => Promise<void>, item: unknown) => Promise<void>;
  }>(new URL("../src/features/library/actions.ts", import.meta.url), {
    fflate: {},
    "@/config/env": {},
    "@/features/audio/audio-page-policy": {},
    "@/features/audio/workflows": {},
    "@/features/auth": { getAuthSessionEpoch: () => epoch },
    "@/features/chat": {},
    "@/features/model-picker": {},
    "@/i18n": { translate: (key: string) => key },
    "@/lib/audio-utils": {},
    "@/lib/api-base": { isTauri: false },
    "@/lib/image-fit": {
      MAX_UPLOAD_IMAGE_SIDE: 4096,
      blobToDataUrl: async () => "data:image/png;base64,AAAA",
      fitImageDataUrl: async (dataUrl: string) => `${dataUrl}#fit`,
    },
    "@/lib/native-files": {},
    "@/lib/toast": { toast: { error: (title: unknown) => errors.push(title) } },
    "@/lib/video-utils": {},
    "@/stores/image-edit-handoff": {
      useImageEditHandoffStore: { getState: () => ({ offer: (url: string) => offered.push(url) }) },
    },
    "./api": {
      LibraryFileTooLarge: class extends Error {},
      errorMessage: (error: unknown) => String(error),
      libraryItemFile: async () => {
        if (fail) throw new Error("gone");
        epoch = epochAfterRead;
        return new Blob(["x"], { type: "image/png" });
      },
    },
    "./file-kind": {},
    "./file-name": {},
    "./start-chat": {},
  });
  return editLibraryImage(async (to) => void navigations.push(to), { name: "cat.png" }).then(() => ({
    offered,
    navigations,
    errors,
  }));
}

test("Edit image hands the fitted image to the Images page and opens it", async () => {
  const { offered, navigations, errors } = await editLibraryImageWith();
  assert.deepEqual(offered, ["data:image/png;base64,AAAA#fit"]);
  assert.deepEqual(navigations, [{ to: "/images" }]);
  assert.deepEqual(errors, []);
});

test("Edit image says so when the file cannot be read, and opens nothing", async () => {
  const { offered, navigations, errors } = await editLibraryImageWith({ fail: true });
  assert.deepEqual(offered, []);
  assert.deepEqual(navigations, []);
  assert.deepEqual(errors, ["library.toast.editImageFailed"]);
});

test("Edit image hands nothing to an account that signed in during the read", async () => {
  const { offered, navigations } = await editLibraryImageWith({ epochAfterRead: 1 });
  assert.deepEqual(offered, []);
  assert.deepEqual(navigations, []);
});

test("Edit image keeps only the newest click when an earlier read finishes last", async () => {
  const offered: string[] = [];
  const reads = new Map<string, (blob: Blob) => void>();
  const { editLibraryImage } = loadWithStubs<{
    editLibraryImage: (navigate: (to: unknown) => Promise<void>, item: { name: string }) => Promise<void>;
  }>(new URL("../src/features/library/actions.ts", import.meta.url), {
    fflate: {},
    "@/config/env": {},
    "@/features/audio/audio-page-policy": {},
    "@/features/audio/workflows": {},
    "@/features/auth": { getAuthSessionEpoch: () => 0 },
    "@/features/chat": {},
    "@/features/model-picker": {},
    "@/i18n": { translate: (key: string) => key },
    "@/lib/audio-utils": {},
    "@/lib/api-base": { isTauri: false },
    "@/lib/image-fit": {
      MAX_UPLOAD_IMAGE_SIDE: 4096,
      blobToDataUrl: async (blob: Blob) => `data:${await blob.text()}`,
      fitImageDataUrl: async (dataUrl: string) => dataUrl,
    },
    "@/lib/native-files": {},
    "@/lib/toast": { toast: { error: () => {} } },
    "@/lib/video-utils": {},
    "@/stores/image-edit-handoff": {
      useImageEditHandoffStore: { getState: () => ({ offer: (url: string) => offered.push(url) }) },
    },
    "./api": {
      LibraryFileTooLarge: class extends Error {},
      errorMessage: (error: unknown) => String(error),
      libraryItemFile: (item: { name: string }) =>
        new Promise<Blob>((resolve) => reads.set(item.name, resolve)),
    },
    "./file-kind": {},
    "./file-name": {},
    "./start-chat": {},
  });
  const navigate = async () => {};
  const first = editLibraryImage(navigate, { name: "first" });
  const second = editLibraryImage(navigate, { name: "second" });
  reads.get("second")?.(new Blob(["second"]));
  await second;
  reads.get("first")?.(new Blob(["first"]));
  await first;
  assert.deepEqual(offered, ["data:second"]);
});
