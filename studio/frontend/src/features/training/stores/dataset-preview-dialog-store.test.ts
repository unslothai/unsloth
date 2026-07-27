// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { useDatasetPreviewDialogStore } from "./dataset-preview-dialog-store.ts";

const targets = [
  {
    source: "huggingface" as const,
    path: "org/alpha",
    subset: "english",
    split: "train",
  },
  {
    source: "huggingface" as const,
    path: "org/beta",
    subset: "default",
    split: "validation",
  },
  {
    source: "upload" as const,
    path: "/datasets/local.jsonl",
    subset: null,
    split: null,
  },
];

test("each dataset row opens its own complete preview target", () => {
  for (const target of targets) {
    useDatasetPreviewDialogStore.getState().openPreview(target);
    const state = useDatasetPreviewDialogStore.getState();
    assert.equal(state.open, true);
    assert.equal(state.mode, "preview");
    assert.deepEqual(state.previewTarget, target);
    state.close();
  }
});

test("closing and legacy reopening cannot retain a stale preview target", () => {
  useDatasetPreviewDialogStore.getState().openPreview(targets[1]);
  useDatasetPreviewDialogStore.getState().close();
  assert.equal(useDatasetPreviewDialogStore.getState().previewTarget, null);

  useDatasetPreviewDialogStore.getState().openPreview();
  assert.equal(useDatasetPreviewDialogStore.getState().previewTarget, null);
});

test("mapping mode remains independent from explicit preview targets", () => {
  useDatasetPreviewDialogStore.getState().openPreview(targets[0]);
  useDatasetPreviewDialogStore.getState().openMapping({} as never);
  const state = useDatasetPreviewDialogStore.getState();
  assert.equal(state.mode, "mapping");
  assert.equal(state.previewTarget, null);
});
