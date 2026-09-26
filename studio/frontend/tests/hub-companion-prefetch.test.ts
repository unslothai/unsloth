// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { buildCachedInventoryRow, buildLocalInventoryRows } =
  await import("../src/features/hub/inventory/view-models.ts");
const { buildDiscoverRows, discoveryInventorySignature } =
  await import("../src/features/hub/lib/view-models.ts");

const REPO_ID = "Qwen/Qwen-Image-2.1";
const RESULT = { id: REPO_ID, downloads: 1, likes: 1, isGguf: false };

function localRow(companionPrefetch: boolean) {
  return buildLocalInventoryRows([
    {
      id: REPO_ID,
      display_name: "Qwen-Image-2.1",
      path: "/cache/models--Qwen--Qwen-Image-2.1/snapshots/abc",
      source: "hf_cache",
      model_id: REPO_ID,
      model_format: "unknown",
      partial: true,
      companion_prefetch: companionPrefetch,
    },
  ]);
}

test("a companion-only base repo is neither on device nor partial in model Discover", () => {
  // The cached and the local listing both describe it; either one carrying the flag is enough.
  const fromCached = buildDiscoverRows(
    [RESULT],
    [
      buildCachedInventoryRow(
        {
          repo_id: REPO_ID,
          model_format: "safetensors",
          size_bytes: 1_350_989_512,
          partial: true,
          companion_prefetch: true,
        },
        "safetensors",
      ),
    ],
    [],
  );
  const fromLocal = buildDiscoverRows([RESULT], [], localRow(true));
  for (const [row] of [fromCached, fromLocal]) {
    assert.equal(row.isAvailableOnDevice, false);
    assert.equal(row.isPartialOnDevice, false);
  }
});

test("an interrupted download of the same repo still reads as partial", () => {
  const [row] = buildDiscoverRows([RESULT], [], localRow(false));
  assert.equal(row.isAvailableOnDevice, true);
  assert.equal(row.isPartialOnDevice, true);
});

test("the local listing's flag reaches the local inventory row", () => {
  assert.equal(localRow(true)[0].companionPrefetch, true);
  assert.equal(localRow(false)[0].companionPrefetch, false);
});

test("the detail pane offers a plain Download, never Run, for a companion-only repo", async () => {
  const { useSelectedModelView } =
    await import("../src/features/hub/hooks/use-selected-model-view.ts");
  const { modelDownloadState } =
    await import("../src/features/hub/catalog/model-download-state.ts");
  const { downloadActionLabel } =
    await import("../src/features/hub/catalog/use-download-card-state.ts");
  const { createElement } = await import("react");
  const { renderToStaticMarkup } = await import("react-dom/server");

  const [local] = localRow(true);
  const cached = buildCachedInventoryRow(
    {
      repo_id: REPO_ID,
      model_format: "safetensors",
      size_bytes: 1_350_989_512,
      partial: true,
      companion_prefetch: true,
    },
    "safetensors",
  );
  const [discover] = buildDiscoverRows([RESULT], [cached], [local]);
  const base = {
    selectedDiscoverRow: null,
    selectedCachedRow: null,
    selectedLocalRow: null,
    selectedHfResult: null,
    isDatasetMode: false,
  };
  const cases = {
    "local row": { ...base, selectedLocalRow: local },
    "cached row": { ...base, selectedCachedRow: cached },
    "discover row over the local row": {
      ...base,
      selectedDiscoverRow: discover,
      selectedLocalRow: local,
    },
    "discover row over the cached row": {
      ...base,
      selectedDiscoverRow: discover,
      selectedCachedRow: cached,
    },
  };
  for (const [name, input] of Object.entries(cases)) {
    let view: ReturnType<typeof useSelectedModelView> = null;
    function Harness() {
      view = useSelectedModelView(input);
      return null;
    }
    renderToStaticMarkup(createElement(Harness));
    assert.ok(view, name);
    const state = modelDownloadState(view);
    assert.equal(state.isDownloaded, false, name);
    assert.equal(state.isPartial, false, name);
    assert.equal(
      (view as { companionPrefetch?: boolean } | null)?.companionPrefetch,
      true,
      name,
    );
    assert.equal(
      downloadActionLabel(state.isPartial, state.partialResumable),
      "Download",
      name,
    );
  }
});

test("a companion fetch finishing changes the Discover memo key", () => {
  // Mid-fetch and finished, both rows are partial; only the flag moves.
  const fetching = discoveryInventorySignature([], localRow(false));
  const finished = discoveryInventorySignature([], localRow(true));
  assert.notEqual(fetching, finished);
  const cached = (companion_prefetch: boolean) =>
    buildCachedInventoryRow(
      {
        repo_id: REPO_ID,
        model_format: "safetensors",
        size_bytes: 1,
        partial: true,
        companion_prefetch,
      },
      "safetensors",
    );
  assert.notEqual(
    discoveryInventorySignature([cached(false)], []),
    discoveryInventorySignature([cached(true)], []),
  );
});

test("the download card keeps Delete for a companion-only repo", () => {
  const card = readSrc("features/hub/catalog/safetensors-download-card.tsx");
  assert.match(card, /hasCachedFiles = isDownloaded \|\| isPartial \|\| companionPrefetch/);
  assert.match(card, /canDelete =\s+hasCachedFiles &&/);
  assert.match(card, /\(isPartial \|\| companionPrefetch\) && !downloading/);
  const section = readSrc("features/hub/catalog/download-section.tsx");
  assert.match(section, /companionPrefetch=\{companionPrefetch\}/);
  const inspector = readSrc("features/hub/catalog/model-inspector.tsx");
  assert.match(inspector, /companionPrefetch=\{model\.companionPrefetch === true\}/);
});

test("a companion-only inventory row never gets the green On device dot", () => {
  const rows = readSrc("features/hub/catalog/models-catalog-rows.tsx");
  assert.match(
    rows,
    /<PartialStatusDot downloading=\{downloading\} \/>\s*\) : row\.companionPrefetch \? null : \(\s*<StatusDot tone="success" label="On device" \/>/,
  );
});
