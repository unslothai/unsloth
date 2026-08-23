// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { buildCachedInventoryRow, hasCompleteCacheCopyBeyondSelected } =
  await import("../src/features/hub/inventory/view-models.ts");
const { dedupeSameSourceHubCacheRows } = await import(
  "../src/features/hub/inventory/inventory-dedupe.ts"
);
const {
  canMigrateCachedRepoToActiveCache,
  cachedRepoVariantSources,
  cachedRepoValidationTargets,
  downloadedQuantCacheTargets,
  mergeCachedGgufVariantResults,
  toCachedRepoCopies,
} = await import("../src/features/model-picker/inventory/cache-copy-path.ts");
const {
  buildLocalPinCleanupEvidence,
  localPinInventoryNeeds,
  pinsToRemoveAfterLocalCacheDelete,
  removeQuantPinIfNoCopyRemains,
} = await import("../src/features/hub/catalog/pin-cleanup.ts");
const { remainingDownloadedGgufQuants } = await import(
  "../src/features/hub/catalog/remaining-gguf-copies.ts"
);
const {
  downloadedGgufQuantsAfterCacheDelete,
  reconcilePinsAfterCacheCopyDelete,
} = await import("../src/features/hub/catalog/pin-reconciliation.ts");
const { usePinnedModelsStore } = await import("../src/stores/pinned-models.ts");
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");

const CATALOG_ROWS = fileURLToPath(
  new URL(
    "../src/features/hub/catalog/models-catalog-rows.tsx",
    import.meta.url,
  ),
);
const GGUF_CARD = fileURLToPath(
  new URL(
    "../src/features/hub/catalog/gguf-download-card.tsx",
    import.meta.url,
  ),
);
const DOWNLOAD_SECTION = fileURLToPath(
  new URL("../src/features/hub/catalog/download-section.tsx", import.meta.url),
);
const REMAINING_GGUF_COPIES = fileURLToPath(
  new URL(
    "../src/features/hub/catalog/remaining-gguf-copies.ts",
    import.meta.url,
  ),
);
const SAFETENSORS_CARD = fileURLToPath(
  new URL(
    "../src/features/hub/catalog/safetensors-download-card.tsx",
    import.meta.url,
  ),
);
const LOCAL_ON_DEVICE_CARD = fileURLToPath(
  new URL(
    "../src/features/hub/catalog/local-on-device-card.tsx",
    import.meta.url,
  ),
);
const PIN_CLEANUP = fileURLToPath(
  new URL("../src/features/hub/catalog/pin-cleanup.ts", import.meta.url),
);
const PIN_RECONCILIATION = fileURLToPath(
  new URL("../src/features/hub/catalog/pin-reconciliation.ts", import.meta.url),
);
const PICKERS = fileURLToPath(
  new URL(
    "../src/features/model-picker/components/model-selector/pickers.tsx",
    import.meta.url,
  ),
);

test("cached view model preserves selected-copy and aggregate disk metadata", () => {
  const row = buildCachedInventoryRow(
    {
      repo_id: "Org/Model",
      load_id: "Org/Model",
      model_format: "gguf",
      size_bytes: 100,
      cache_path: "/old/models--Org--Model",
      active_cache: false,
      copy_count: 2,
      total_size_bytes: 300,
      cache_copies: [
        {
          cache_path: "/active/models--Org--Model",
          load_id: "Org/Model",
          size_bytes: 200,
          active_cache: true,
          partial: true,
          last_modified: 20,
        },
        {
          cache_path: "/old/models--Org--Model",
          load_id: "/old/models--Org--Model/snapshots/rev-old",
          size_bytes: 100,
          active_cache: false,
          partial: false,
          last_modified: 10,
        },
      ],
    },
    "gguf",
  );

  assert.equal(row.cachePath, "/old/models--Org--Model");
  assert.equal(row.bytes, 100);
  assert.equal(row.activeCache, false);
  assert.equal(row.copyCount, 2);
  assert.equal(row.totalBytes, 300);
  assert.deepEqual(
    row.cacheCopies.map((copy) => [
      copy.cachePath,
      copy.loadId,
      copy.bytes,
      copy.activeCache,
      copy.partial,
    ]),
    [
      ["/active/models--Org--Model", "Org/Model", 200, true, true],
      [
        "/old/models--Org--Model",
        "/old/models--Org--Model/snapshots/rev-old",
        100,
        false,
        false,
      ],
    ],
  );
});

test("cache-copy load identities reach picker variant selection", () => {
  const row = buildCachedInventoryRow(
    {
      repo_id: "Org/Model",
      load_id: "Org/Model",
      model_format: "gguf",
      size_bytes: 100,
      cache_path: "/active/models--Org--Model",
      active_cache: true,
      cache_copies: [
        {
          cache_path: "/active/models--Org--Model",
          load_id: "Org/Model",
          size_bytes: 100,
          active_cache: true,
          partial: false,
        },
        {
          cache_path: "/old/models--Org--Model",
          load_id: "/old/models--Org--Model/snapshots/rev-old",
          size_bytes: 200,
          active_cache: false,
          partial: false,
        },
      ],
    },
    "gguf",
  );
  const sources = cachedRepoVariantSources({
    repo_id: row.repoId,
    load_id: row.loadId,
    cache_path: row.cachePath ?? "",
    active_cache: row.activeCache,
    cache_copies: toCachedRepoCopies(row.cacheCopies),
  });
  const inactive = sources.find((source) => source.activeCache === false);
  assert.equal(
    inactive?.localPath,
    "/old/models--Org--Model/snapshots/rev-old",
  );
  assert.ok(inactive);

  const [variant] = mergeCachedGgufVariantResults([
    {
      source: inactive,
      contextLength: 8192,
      variants: [
        {
          filename: "model-Q6_K.gguf",
          quant: "Q6_K",
          size_bytes: 200,
          downloaded: true,
        },
      ],
    },
  ]);
  assert.equal(variant?.loadId, "/old/models--Org--Model/snapshots/rev-old");
  assert.equal(variant?.contextLength, 8192);
});

test("older backend rows keep truthful single-copy defaults", () => {
  const row = buildCachedInventoryRow(
    {
      repo_id: "Org/Legacy",
      size_bytes: 123,
      cache_path: "/cache/models--Org--Legacy",
    },
    "safetensors",
  );

  assert.equal(row.activeCache, null);
  assert.equal(row.copyCount, 1);
  assert.equal(row.totalBytes, 123);
  assert.deepEqual(row.cacheCopies, []);
});

test("inventory dedupe keeps live state and unions measured cache copies", () => {
  const measured = buildCachedInventoryRow(
    {
      repo_id: "Org/Model",
      model_format: "gguf",
      size_bytes: 100,
      cache_path: "/active/models--Org--Model",
      active_cache: true,
      partial: true,
      copy_count: 2,
      total_size_bytes: 300,
      cache_copies: [
        {
          cache_path: "/active/models--Org--Model",
          load_id: "Org/Model",
          size_bytes: 100,
          active_cache: true,
          partial: true,
        },
        {
          cache_path: "/older/models--Org--Model",
          load_id: "/older/models--Org--Model/snapshots/rev-older",
          size_bytes: 200,
          active_cache: false,
          partial: true,
        },
      ],
    },
    "gguf",
  );
  const live = {
    ...buildCachedInventoryRow(
      {
        repo_id: "org/model",
        model_format: "gguf",
        size_bytes: 900,
        partial: true,
        optimistic: true,
        copy_count: 0,
        total_size_bytes: 0,
        cache_copies: [
          {
            cache_path: "/older/models--Org--Model",
            size_bytes: 50,
            active_cache: false,
            partial: true,
          },
        ],
      },
      "gguf",
    ),
    liveDownload: true,
  };

  for (const cachedRows of [
    [measured, live],
    [live, measured],
  ]) {
    const [row] = dedupeSameSourceHubCacheRows({
      cachedRows,
      localRows: [],
    }).cachedRows;

    assert.equal(row.liveDownload, true);
    assert.equal(row.bytes, 900);
    assert.equal(row.cachePath, null);
    assert.equal(row.copyCount, 2);
    assert.equal(row.totalBytes, 300);
    assert.deepEqual(
      row.cacheCopies.map((copy) => [copy.cachePath, copy.loadId, copy.bytes]),
      [
        ["/active/models--Org--Model", "Org/Model", 100],
        [
          "/older/models--Org--Model",
          "/older/models--Org--Model/snapshots/rev-older",
          200,
        ],
      ],
    );
  }
});

test("whole-model pins require another complete physical copy", () => {
  assert.equal(
    hasCompleteCacheCopyBeyondSelected("/cache/selected", [
      { cachePath: "/cache/selected", partial: false },
      { cachePath: "/cache/other", partial: false },
    ]),
    true,
  );
  assert.equal(
    hasCompleteCacheCopyBeyondSelected("/cache/selected", [
      { cachePath: "/cache/selected", partial: false },
      { cachePath: "/cache/other", partial: true },
    ]),
    false,
  );
  assert.equal(
    hasCompleteCacheCopyBeyondSelected("C:\\Cache\\Model\\", [
      { cachePath: "c:/cache/model", partial: false },
    ]),
    false,
  );
  assert.equal(
    hasCompleteCacheCopyBeyondSelected("/cache/Model", [
      { cachePath: "/cache/model", partial: false },
    ]),
    true,
  );
  assert.equal(
    hasCompleteCacheCopyBeyondSelected(undefined, [
      { cachePath: "/cache/other", partial: false },
    ]),
    false,
  );
});

test("local pin cleanup separates repository and quant evidence", () => {
  assert.deepEqual(
    localPinInventoryNeeds([{ repoId: "Org/Hybrid", quant: null }]),
    { gguf: false, models: true },
  );
  assert.deepEqual(
    localPinInventoryNeeds([{ repoId: "Org/Hybrid", quant: "Q4_K_M" }]),
    { gguf: true, models: false },
  );
  const evidence = buildLocalPinCleanupEvidence(
    "Org/Hybrid",
    [
      {
        repo_id: "org/hybrid",
        cache_path: "/cache/gguf/models--Org--Hybrid",
        partial: false,
      },
    ],
    [
      {
        repo_id: "ORG/HYBRID",
        cache_path: "/cache/safetensors/models--Org--Hybrid",
        partial: false,
      },
    ],
    [],
  );

  assert.equal(evidence.plainPinMayRemain, true);
  assert.equal(evidence.ggufState, "represented");
  assert.deepEqual(
    pinsToRemoveAfterLocalCacheDelete(
      [
        { repoId: "Org/Hybrid", quant: null },
        { repoId: "Org/Hybrid", quant: "Q4_K_M" },
      ],
      evidence,
      new Set(),
    ),
    [{ repoId: "Org/Hybrid", quant: "Q4_K_M" }],
  );
  assert.deepEqual(
    pinsToRemoveAfterLocalCacheDelete(
      [
        { repoId: "Org/Hybrid", quant: null },
        { repoId: "Org/Hybrid", quant: "Q4_K_M" },
      ],
      { ...evidence, plainPinMayRemain: false },
      new Set(["q4_k_m"]),
    ),
    [{ repoId: "Org/Hybrid", quant: null }],
  );
});

test("plain repository pins remain for complete GGUF copies", () => {
  const cachedEvidence = buildLocalPinCleanupEvidence(
    "Org/GGUF",
    [
      {
        repo_id: "org/gguf",
        cache_path: "/cache/models--Org--GGUF",
        partial: false,
      },
    ],
    [],
    [],
  );
  const localEvidence = buildLocalPinCleanupEvidence(
    "Org/GGUF",
    [],
    [],
    [
      {
        source: "hf_cache",
        model_id: "ORG/GGUF",
        model_format: "gguf",
        path: "/cache/models--Org--GGUF/snapshots/revision",
        partial: false,
      },
    ],
  );

  assert.equal(cachedEvidence.plainPinMayRemain, true);
  assert.equal(localEvidence.plainPinMayRemain, true);
  assert.deepEqual(
    pinsToRemoveAfterLocalCacheDelete(
      [{ repoId: "Org/GGUF", quant: null }],
      localEvidence,
    ),
    [],
  );
});

test("post-delete reconciliation keeps plain and quant pins for a GGUF survivor", async (t) => {
  const originalPinned = usePinnedModelsStore.getState().pinned;
  t.after(() => {
    setAuthFetchHandler(null);
    usePinnedModelsStore.setState({ pinned: originalPinned });
  });
  usePinnedModelsStore.setState({
    pinned: ["Org/Hybrid", "Org/Hybrid::Q4_K_M"],
  });
  const requests: string[] = [];
  setAuthFetchHandler((input) => {
    requests.push(input);
    const body = input.startsWith("/api/hub/cached-gguf")
      ? {
          cached: [
            {
              repo_id: "org/hybrid",
              cache_path: "/cache/gguf/models--Org--Hybrid",
              cache_copies: [
                {
                  cache_path: "/cache/gguf/models--Org--Hybrid",
                  partial: false,
                },
              ],
            },
          ],
        }
      : input.startsWith("/api/hub/cached-models")
        ? { cached: [] }
        : input.startsWith("/api/hub/local")
          ? {
              models: [
                {
                  source: "hf_cache",
                  model_id: "ORG/HYBRID",
                  model_format: "gguf",
                  path: "/cache/gguf/models--Org--Hybrid/snapshots/revision",
                  partial: false,
                },
              ],
            }
          : input.startsWith("/api/hub/gguf-variants")
            ? {
                repo_id: "Org/Hybrid",
                variants: [{ quant: "Q4_K_M", downloaded: true }],
                has_vision: false,
                default_variant: null,
              }
            : null;
    if (body === null) {
      throw new Error(`Unexpected request: ${input}`);
    }
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  });

  await reconcilePinsAfterCacheCopyDelete({ repoId: "Org/Hybrid" });

  assert.deepEqual(usePinnedModelsStore.getState().pinned, [
    "Org/Hybrid",
    "Org/Hybrid::Q4_K_M",
  ]);
  assert.equal(
    requests.some((request) => request.includes("offline=true")),
    true,
  );
});

test("post-delete reconciliation cannot restore a concurrent unpin", async (t) => {
  const originalPinned = usePinnedModelsStore.getState().pinned;
  t.after(() => {
    setAuthFetchHandler(null);
    usePinnedModelsStore.setState({ pinned: originalPinned });
  });
  usePinnedModelsStore.setState({ pinned: ["Org/Concurrent"] });
  let finishLocal!: (response: Response) => void;
  setAuthFetchHandler((input) => {
    if (input.startsWith("/api/hub/cached-models")) {
      return new Response(JSON.stringify({ cached: [] }), { status: 200 });
    }
    if (input.startsWith("/api/hub/local")) {
      return new Promise((resolve) => {
        finishLocal = resolve;
      });
    }
    throw new Error(`Unexpected request: ${input}`);
  });

  const reconciliation = reconcilePinsAfterCacheCopyDelete({
    repoId: "Org/Concurrent",
  });
  usePinnedModelsStore.setState({ pinned: [] });
  finishLocal(
    new Response(JSON.stringify({ models: [] }), {
      status: 200,
      headers: { "content-type": "application/json" },
    }),
  );
  await reconciliation;

  assert.deepEqual(usePinnedModelsStore.getState().pinned, []);
});

test("local-only inventory preserves pins when cached endpoints omit a copy", () => {
  const evidence = buildLocalPinCleanupEvidence(
    "Org/Local",
    [],
    [],
    [
      {
        source: "hf_cache",
        model_id: "org/local",
        model_format: "unknown",
        path: "/cache/models--Org--Local/snapshots/revision",
        partial: false,
      },
    ],
  );

  assert.equal(evidence.plainPinMayRemain, true);
  assert.equal(evidence.ggufState, "uncertain");
  assert.deepEqual(
    pinsToRemoveAfterLocalCacheDelete(
      [{ repoId: "Org/Local", quant: "Q8_0" }],
      evidence,
    ),
    [],
  );

  const partialOnly = buildLocalPinCleanupEvidence(
    "Org/Local",
    [],
    [],
    [
      {
        source: "hf_cache",
        model_id: "org/local",
        model_format: "unknown",
        path: "/cache/models--Org--Local/snapshots/partial",
        partial: true,
      },
    ],
  );
  assert.equal(partialOnly.plainPinMayRemain, false);
  assert.equal(partialOnly.ggufState, "absent");
  assert.deepEqual(
    pinsToRemoveAfterLocalCacheDelete(
      [{ repoId: "Org/Local", quant: "Q8_0" }],
      partialOnly,
    ),
    [{ repoId: "Org/Local", quant: "Q8_0" }],
  );
});

test("quant cleanup skips unpinned probes and cannot re-add a removed pin", async (t) => {
  const originalPinned = usePinnedModelsStore.getState().pinned;
  t.after(() => usePinnedModelsStore.setState({ pinned: originalPinned }));
  let probes = 0;
  usePinnedModelsStore.setState({ pinned: [] });
  assert.equal(
    await removeQuantPinIfNoCopyRemains("Org/Model", "Q4_K_M", async () => {
      probes += 1;
      return new Set();
    }),
    false,
  );
  assert.equal(probes, 0);

  usePinnedModelsStore.setState({ pinned: ["ORG/MODEL::q4_k_m"] });
  let finishProbe!: (value: ReadonlySet<string> | null) => void;
  const cleanup = removeQuantPinIfNoCopyRemains("org/model", "Q4_K_M", () => {
    probes += 1;
    return new Promise((resolve) => {
      finishProbe = resolve;
    });
  });
  usePinnedModelsStore.setState({ pinned: [] });
  finishProbe(new Set());
  assert.equal(await cleanup, false);
  assert.deepEqual(usePinnedModelsStore.getState().pinned, []);

  usePinnedModelsStore.setState({ pinned: ["ORG/MODEL::q4_k_m"] });
  assert.equal(
    await removeQuantPinIfNoCopyRemains(
      "org/model",
      "Q4_K_M",
      async () => new Set(),
    ),
    true,
  );
  assert.deepEqual(usePinnedModelsStore.getState().pinned, []);
});

test("quant cleanup preserves pins when the deleted cache copy is unknown", async () => {
  assert.equal(
    await remainingDownloadedGgufQuants("Org/Model", undefined, []),
    null,
  );
  assert.equal(
    await remainingDownloadedGgufQuants(
      "Org/Model",
      "/cache/models--Org--Model",
      [],
    ),
    null,
  );
  assert.equal(
    await remainingDownloadedGgufQuants("Org/Model", undefined, [
      { cachePath: "" },
    ]),
    null,
  );
  assert.deepEqual(
    await remainingDownloadedGgufQuants("Org/Model", undefined, [
      { cachePath: "/cache/models--Org--Model" },
    ]),
    new Set(),
  );
  assert.deepEqual(
    await remainingDownloadedGgufQuants(
      "Org/Model",
      "/cache/models--Org--Model/snapshots/revision",
      [{ cachePath: "/cache/models--Org--Model" }],
    ),
    new Set(),
  );
});

test("browse quant cleanup resolves missing path evidence from local inventory", async (t) => {
  const originalPinned = usePinnedModelsStore.getState().pinned;
  t.after(() => {
    setAuthFetchHandler(null);
    usePinnedModelsStore.setState({ pinned: originalPinned });
  });
  const requests: string[] = [];
  setAuthFetchHandler((input) => {
    requests.push(input);
    if (input.startsWith("/api/hub/cached-gguf")) {
      return new Response(JSON.stringify({ cached: [] }), { status: 200 });
    }
    if (input.startsWith("/api/hub/local")) {
      return new Response(JSON.stringify({ models: [] }), { status: 200 });
    }
    throw new Error(`Unexpected request: ${input}`);
  });
  const loadRemaining = () =>
    downloadedGgufQuantsAfterCacheDelete({ repoId: "Org/Browse" });

  usePinnedModelsStore.setState({ pinned: [] });
  assert.equal(
    await removeQuantPinIfNoCopyRemains("Org/Browse", "Q4_K_M", loadRemaining),
    false,
  );
  assert.deepEqual(requests, []);

  usePinnedModelsStore.setState({ pinned: ["Org/Browse::Q4_K_M"] });
  assert.equal(
    await removeQuantPinIfNoCopyRemains("Org/Browse", "Q4_K_M", loadRemaining),
    true,
  );
  assert.equal(requests.length, 2);
  assert.deepEqual(usePinnedModelsStore.getState().pinned, []);
});

test("On Device management shows total disk and cleans pins by artifact", () => {
  const source = readFileSync(CATALOG_ROWS, "utf8");

  assert.match(source, /copyCount > 1[\s\S]*aggregateBytes/);
  assert.match(
    source,
    /cache locations · \{formatBytes\(aggregateBytes\)\} total/,
  );
  assert.match(source, /cacheCopies\.map/);
  assert.match(source, /copy\.cachePath/);
  assert.match(source, /hasCompleteCacheCopyBeyondSelected/);
  assert.match(source, /reconcilePinsAfterCacheCopyDelete/);
  assert.match(source, /row\.source === "hf_cache"[\s\S]*\? row\.path/);
});

test("historical cache cards download to current location instead of updating in place", () => {
  const gguf = readFileSync(GGUF_CARD, "utf8");
  const safetensors = readFileSync(SAFETENSORS_CARD, "utf8");
  const local = readFileSync(LOCAL_ON_DEVICE_CARD, "utf8");

  assert.ok(gguf.includes("getCachedModelPath(repoId, quant, cachePath)"));
  assert.match(
    gguf,
    /!downloadToCurrentCache &&[\s\S]*selected\.update_available === true/,
  );
  assert.match(gguf, /downloadToCurrentCache[\s\S]*"Download here"/);
  assert.match(
    safetensors,
    /activeCache === false && \(isDownloaded \|\| isPartial\)/,
  );
  assert.match(safetensors, /idleLabel=.*"Download here"/);
  assert.match(
    local,
    /activeCache === false && \(!isGguf \|\| selectedVariant !== null\)/,
  );
  assert.match(
    local,
    /if \(!repoId \|\| \(needsVariantSelection && !updateTargetVariant\)\) return;/,
  );
  assert.match(local, /downloadToCurrentCache[\s\S]*"Download here"/);
});

test("delete cards clean pins according to the deleted artifact", () => {
  const gguf = readFileSync(GGUF_CARD, "utf8");
  const downloadSection = readFileSync(DOWNLOAD_SECTION, "utf8");
  const remainingGgufCopies = readFileSync(REMAINING_GGUF_COPIES, "utf8");
  const safetensors = readFileSync(SAFETENSORS_CARD, "utf8");
  const local = readFileSync(LOCAL_ON_DEVICE_CARD, "utf8");
  const pinCleanup = readFileSync(PIN_CLEANUP, "utf8");
  const pinReconciliation = readFileSync(PIN_RECONCILIATION, "utf8");

  assert.match(downloadSection, /cacheCopies=\{cacheCopies\}/);
  assert.match(gguf, /remainingDownloadedGgufQuants\(/);
  assert.match(gguf, /removeQuantPinIfNoCopyRemains\(/);
  assert.match(
    remainingGgufCopies,
    /preferLocalCache: true,[\s\S]*offline: true/,
  );
  assert.match(
    remainingGgufCopies,
    /candidate === selected \|\| selected\.startsWith\(`\$\{candidate\}\/`\)/,
  );
  assert.match(gguf, /from this cache location/);
  assert.match(safetensors, /hasCompleteCacheCopyBeyondSelected/);
  assert.match(safetensors, /reconcilePinsAfterCacheCopyDelete/);
  assert.match(local, /reconcilePinsAfterCacheCopyDelete/);
  assert.match(pinReconciliation, /listCachedGguf/);
  assert.match(pinReconciliation, /listCachedModels/);
  assert.match(pinReconciliation, /listLocalModels/);
  assert.match(pinReconciliation, /inventoryNeeds\.gguf \? listCachedGguf/);
  assert.match(pinReconciliation, /inventoryNeeds\.models \? listCachedModels/);
  assert.match(pinReconciliation, /downloadedGgufQuantsInCacheCopies\(/);
  assert.match(pinReconciliation, /downloadedGgufQuantsAfterCacheDelete/);
  assert.match(pinReconciliation, /buildLocalPinCleanupEvidence\(/);
  assert.match(pinReconciliation, /removePinnedArtifactIfPresent\(/);
  assert.match(pinCleanup, /hasUnrepresentedLocalCopy/);
});

test("pinned quant targets resolve independently across physical cache copies", () => {
  const validationTargets = cachedRepoValidationTargets({
    repo_id: "Org/Model",
    load_id: "Org/Model",
    size_bytes: 100,
    cache_path: "/active/models--Org--Model",
    copy_count: 2,
    cache_copies: [
      {
        cache_path: "/older/models--Org--Model",
        load_id: "/older/models--Org--Model/snapshots/rev-old",
        size_bytes: 200,
        active_cache: false,
        partial: false,
      },
      {
        cache_path: "/active/models--Org--Model",
        load_id: "Org/Model",
        size_bytes: 100,
        active_cache: true,
        partial: false,
      },
    ],
  });
  assert.deepEqual(
    validationTargets.map((target) => [target.cachePath, target.loadId]),
    [
      ["/active/models--Org--Model", "Org/Model"],
      [
        "/older/models--Org--Model",
        "/older/models--Org--Model/snapshots/rev-old",
      ],
    ],
  );

  const targetsByQuant = downloadedQuantCacheTargets([
    { target: validationTargets[0], downloadedQuants: ["Q4_K_M"] },
    {
      target: validationTargets[1],
      downloadedQuants: ["Q6_K", "Q4_K_M"],
    },
  ]);
  assert.equal(
    targetsByQuant.get("Q4_K_M")?.cachePath,
    "/active/models--Org--Model",
  );
  assert.equal(
    targetsByQuant.get("Q6_K")?.cachePath,
    "/older/models--Org--Model",
  );
  assert.deepEqual(
    cachedRepoValidationTargets({
      repo_id: "Org/Legacy",
      size_bytes: 1,
      cache_path: "",
    }),
    [{ cachePath: undefined, copyCount: 1 }],
  );
});

test("variant management merges complete and partial quants across cache copies", () => {
  const sources = cachedRepoVariantSources({
    repo_id: "Org/Model",
    load_id: "/old/models--Org--Model/snapshots/rev-old",
    cache_path: "/old/models--Org--Model",
    active_cache: false,
    cache_copies: [
      {
        cache_path: "/active/models--Org--Model",
        load_id: "Org/Model",
        size_bytes: 80,
        active_cache: true,
        partial: true,
      },
      {
        cache_path: "/old/models--Org--Model/",
        load_id: "/old/models--Org--Model/snapshots/rev-old",
        size_bytes: 100,
        active_cache: false,
        partial: false,
      },
    ],
  });
  assert.deepEqual(sources, [
    {
      localPath: "/old/models--Org--Model/snapshots/rev-old",
      cachePath: "/old/models--Org--Model",
      loadId: "/old/models--Org--Model/snapshots/rev-old",
      activeCache: false,
    },
    {
      localPath: "/active/models--Org--Model",
      cachePath: "/active/models--Org--Model",
      loadId: "Org/Model",
      activeCache: true,
    },
  ]);

  const variants = mergeCachedGgufVariantResults([
    {
      source: sources[0],
      contextLength: 4096,
      variants: [
        {
          filename: "model-Q4_K_M.gguf",
          quant: "Q4_K_M",
          size_bytes: 100,
          downloaded: true,
        },
      ],
    },
    {
      source: sources[1],
      contextLength: 32768,
      variants: [
        {
          filename: "model-Q8_0.gguf",
          quant: "Q8_0",
          size_bytes: 200,
          downloaded: false,
          partial: true,
          partial_resumable: true,
        },
      ],
    },
  ]);
  assert.deepEqual(
    variants.map((variant) => [
      variant.quant,
      variant.downloaded,
      variant.partial,
      variant.cachePath,
      variant.loadId,
      variant.activeCache,
      variant.contextLength,
    ]),
    [
      [
        "Q4_K_M",
        true,
        undefined,
        "/old/models--Org--Model",
        "/old/models--Org--Model/snapshots/rev-old",
        false,
        4096,
      ],
      [
        "Q8_0",
        false,
        true,
        "/active/models--Org--Model",
        "Org/Model",
        true,
        32768,
      ],
    ],
  );
});

test("inactive-only quant remains migratable behind an active representative", () => {
  const cached = {
    repo_id: "Org/Model",
    load_id: "Org/Model",
    size_bytes: 100,
    cache_path: "/active/models--Org--Model",
    active_cache: true,
    cache_copies: [
      {
        cache_path: "/active/models--Org--Model",
        load_id: "Org/Model",
        size_bytes: 100,
        active_cache: true,
        partial: false,
      },
      {
        cache_path: "/older/models--Org--Model",
        load_id: "/older/models--Org--Model/snapshots/rev-older",
        size_bytes: 200,
        active_cache: false,
        partial: false,
      },
    ],
  };
  const [activeSource, olderSource] = cachedRepoVariantSources(cached);
  assert.ok(activeSource);
  assert.ok(olderSource);

  const variants = mergeCachedGgufVariantResults([
    {
      source: activeSource,
      variants: [
        {
          filename: "model-Q4_K_M.gguf",
          quant: "Q4_K_M",
          size_bytes: 100,
          downloaded: true,
        },
      ],
    },
    {
      source: olderSource,
      variants: [
        {
          filename: "model-Q8_0.gguf",
          quant: "Q8_0",
          size_bytes: 200,
          downloaded: true,
        },
      ],
    },
  ]);
  const activeQuant = variants.find((variant) => variant.quant === "Q4_K_M");
  const olderQuant = variants.find((variant) => variant.quant === "Q8_0");

  assert.equal(canMigrateCachedRepoToActiveCache(cached), true);
  assert.equal(canMigrateCachedRepoToActiveCache({}), false);
  assert.equal(activeQuant?.activeCache, true);
  assert.equal(olderQuant?.activeCache, false);
  assert.equal(
    canMigrateCachedRepoToActiveCache(cached) && !activeQuant?.activeCache,
    false,
  );
  assert.equal(
    canMigrateCachedRepoToActiveCache(cached) && !olderQuant?.activeCache,
    true,
  );
});

test("pinned and browse deletes use physical cache-copy targets", () => {
  const source = readFileSync(PICKERS, "utf8");

  assert.match(source, /cachedRepoValidationTargets\(cached\)/);
  assert.match(source, /localPath: target\.cachePath/);
  assert.match(source, /pinnedQuantValidation\.targets\.get/);
  assert.match(source, /loadId: pinnedLoadId/);
  assert.match(source, /loadId: v\.loadId \?\? loadId/);
  assert.match(
    source,
    /pinnedQuantValidation\.revision === pinnedQuantValidationRevision/,
  );
  assert.match(
    source,
    /deleteCachedModel\([\s\S]*entry\.repoId,[\s\S]*entry\.quant,[\s\S]*pinnedCachePath/,
  );
  assert.match(source, /this quantization may not be available there/);
  assert.match(
    source,
    /await onDeleteVariant\(v\.quant, v\.cachePath\);[\s\S]*removeQuantPinIfNoCopyRemains\(/,
  );
  assert.match(source, /downloadedGgufQuantsAfterCacheDelete\(/);
  assert.match(
    source,
    /pinnedCacheCopies[\s\S]*removeQuantPinIfNoCopyRemains\([\s\S]*remainingDownloadedGgufQuants\([\s\S]*pinnedCachePath,[\s\S]*pinnedCacheCopies/,
  );
  assert.match(
    source,
    /renderSoleQuantGgufRow[\s\S]*removeQuantPinIfNoCopyRemains\([\s\S]*remainingDownloadedGgufQuants\([\s\S]*c\.cache_path,[\s\S]*c\.cache_copies/,
  );
  assert.equal(
    source.match(/await removeQuantPinIfNoCopyRemains\(/g)?.length,
    3,
  );
  assert.match(
    source,
    /renderDownloadedModelRow[\s\S]*reconcilePinsAfterCacheCopyDelete\(/,
  );
  assert.match(source, /downloadToCurrentCache: canMigrateToActiveCache/);
  assert.match(
    source,
    /downloadToCurrentForVariant\s*\?\s*updateVariantBadgeLabel\s*:\s*"update available"/,
  );
  assert.match(
    source,
    /isUpdateDisabled: \(quant\) =>[\s\S]*ggufVariantMayOverlapResidentForPicker\(/,
  );
  assert.match(source, /cacheCopies=\{c\.cache_copies\}/);
  assert.match(source, /onDeleteVariant\(v\.quant, v\.cachePath\)/);
  assert.equal(
    source.match(/onDelete: async \(quant, targetCachePath\)/g)?.length,
    4,
  );
  assert.equal(
    source.match(/hfToken \|\| undefined,\s+targetCachePath,/g)?.length,
    3,
  );
  assert.match(source, /const cachedGgufByRepoId = useMemo/);
  assert.match(source, /new Map\(\s*cachedGguf\.map/);
  assert.equal(source.match(/loadId=\{cached\?\.load_id\}/g)?.length, 3);
  assert.equal(source.match(/cachePath=\{cached\?\.cache_path\}/g)?.length, 3);
  assert.equal(
    source.match(/activeCache=\{cached\?\.active_cache\}/g)?.length,
    3,
  );
  assert.equal(
    source.match(/cacheCopies=\{cached\?\.cache_copies\}/g)?.length,
    3,
  );
});
