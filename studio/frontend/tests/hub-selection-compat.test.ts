// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Compatibility and edge-case cover for the On Device selection resolver.
//
// The selection ID is the one piece of Hub state that outlives the process: it sits in the URL, so a
// bookmark, a pinned tab or a shared link written by an older Studio arrives at this code unchanged.
// The download manager's persisted jobs outlive it the same way, through localStorage. Everything
// below is an upgrade path someone can actually walk into, not a synthetic mutation:
//   - a deep link built before inventory IDs were percent-encoded,
//   - a raw `Org/Repo` ID from a response that predates `inventory_id`,
//   - a scoped download persisted before `inventoryKind` existed,
//   - a local row whose ID is a Windows or POSIX filesystem path,
//   - an ID whose percent escapes are malformed, which `decodeURIComponent` throws on.

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { dedupeSameSourceHubCacheRows } = await import(
  "../src/features/hub/inventory/inventory-dedupe.ts"
);
const {
  buildCachedInventoryRow,
  buildLocalInventoryRows,
  cachedInventoryId,
  optimisticInventoryId,
} = await import("../src/features/hub/inventory/view-models.ts");
const { resolveDownloadedSelection, resolveSelectionUrlSync } = await import(
  "../src/features/hub/lib/selection-resolution.ts"
);
const { scopedDownloadInventoryKind, downloadInventoryHintKind } = await import(
  "../src/features/hub/download-manager/download-manager-types.ts"
);

function cachedRow(
  repoId: string,
  modelFormat: "gguf" | "safetensors" | "adapter" | "checkpoint" | "unknown",
  over: Record<string, unknown> = {},
) {
  return buildCachedInventoryRow(
    {
      repo_id: repoId,
      inventory_id: cachedInventoryId(modelFormat, repoId),
      model_format: modelFormat,
      size_bytes: 100,
      ...over,
    },
    modelFormat,
  );
}

function localRow(over: Record<string, unknown>) {
  return buildLocalInventoryRows([
    {
      source: "hf_cache",
      model_format: "unknown",
      ...over,
    } as never,
  ])[0];
}

function resolve(
  selectedId: string | null,
  cachedRows: readonly unknown[],
  localRows: readonly unknown[],
) {
  return resolveDownloadedSelection({
    selectedId,
    cachedRows: cachedRows as never,
    localRows: localRows as never,
    filteredCachedRows: cachedRows as never,
    filteredLocalRows: localRows as never,
  });
}

// ---------------------------------------------------------------------------
// Deep links written by an older Studio
// ---------------------------------------------------------------------------

test("a pre-encoding deep link still selects its row", () => {
  // Studio wrote `cache:gguf:unsloth/gemma-3-270m-it` before IDs were encoded.
  const repoId = "unsloth/gemma-3-270m-it";
  const row = cachedRow(repoId, "gguf");

  assert.equal(row.id, "cache:gguf:unsloth%2Fgemma-3-270m-it");
  assert.equal(resolve(`cache:gguf:${repoId}`, [row], []).selectedId, row.id);
});

test("a raw Org/Repo deep link still selects its row", () => {
  const row = cachedRow("unsloth/gemma-3-270m-it", "safetensors");
  assert.equal(
    resolve("unsloth/gemma-3-270m-it", [row], []).selectedId,
    row.id,
  );
});

test("percent escapes resolve regardless of hex case", () => {
  const row = cachedRow("unsloth/gemma-3-270m-it", "gguf");
  // decodeURIComponent accepts either spelling; a link may carry either.
  assert.equal(
    resolve("cache:gguf:unsloth%2fgemma-3-270m-it", [row], []).selectedId,
    row.id,
  );
  assert.equal(
    resolve("cache:gguf:unsloth%2Fgemma-3-270m-it", [row], []).selectedId,
    row.id,
  );
});

test("repo IDs survive characters encodeURIComponent leaves alone", () => {
  // -_.!~*'() are not escaped, so the encoded and raw spellings coincide.
  for (const repoId of [
    "unsloth/model-v1.5",
    "unsloth/model_v2",
    "unsloth/model.gguf-test",
    "org/repo!name",
    "org/repo~name",
    "org/repo'name",
    "org/repo(1)",
  ]) {
    const row = cachedRow(repoId, "gguf");
    assert.equal(
      resolve(row.id, [row], []).selectedId,
      row.id,
      `canonical ID did not round-trip for ${repoId}`,
    );
  }
});

test("non-ASCII repo IDs round-trip through the canonical ID", () => {
  for (const repoId of [
    "组织/模型",
    "org/модель",
    "org/modèle-café",
    "org/モデル",
    "org/emoji-\u{1F600}",
  ]) {
    const row = cachedRow(repoId, "safetensors");
    assert.equal(
      resolve(row.id, [row], []).selectedId,
      row.id,
      `non-ASCII repo ID did not round-trip for ${repoId}`,
    );
  }
});

// ---------------------------------------------------------------------------
// Malformed IDs must fail safely, never throw
// ---------------------------------------------------------------------------

test("malformed selection IDs fail safely instead of throwing", () => {
  // decodeURIComponent throws URIError on every one of these. An uncaught throw
  // here surfaces as a blank Hub, because this runs inside a render.
  const malformed = [
    "cache:gguf:%",
    "cache:gguf:%2",
    "cache:gguf:%ZZ",
    "cache:gguf:%E0%A4%A",
    "cache:gguf:%C3%28",
    "cache:gguf:%ED%A0%80", // lone surrogate
    "cache:gguf:org%2F%",
  ];
  for (const id of malformed) {
    assert.doesNotThrow(
      () => resolve(id, [cachedRow("org/repo", "gguf")], []),
      `threw on ${id}`,
    );
    assert.equal(
      resolve(id, [cachedRow("org/repo", "gguf")], []).selectedId,
      null,
      `${id} should not select a row`,
    );
  }
});

test("structurally invalid selection IDs are rejected", () => {
  const row = cachedRow("org/repo", "gguf");
  for (const id of [
    "",
    ":",
    "::",
    "cache:",
    "cache::org%2Frepo",
    ":gguf:org%2Frepo",
    "cache:nosuchformat:org%2Frepo",
    "bogus:gguf:org%2Frepo",
    "cache:gguf:org%2Frepo:extra", // 4-segment, backend format_variant shape
    "ollama:gguf:llama3",
  ]) {
    assert.doesNotThrow(() => resolve(id, [row], []), `threw on ${id}`);
  }
});

// ---------------------------------------------------------------------------
// Filesystem-path row IDs: Windows, UNC and POSIX
// ---------------------------------------------------------------------------

test("a Windows path is never mistaken for a raw repo ID", () => {
  const gguf = cachedRow("org/repo", "gguf");
  for (const id of [
    "C:\\Users\\me\\models\\model.gguf",
    "D:\\models\\org\\repo",
    "\\\\server\\share\\model",
    "models\\org\\repo",
  ]) {
    assert.equal(
      resolve(id, [gguf], []).selectedId,
      null,
      `${id} was resolved onto an unrelated row`,
    );
  }
});

test("a POSIX path is never mistaken for a raw repo ID", () => {
  const gguf = cachedRow("home/me", "gguf");
  for (const id of [
    "/home/me/models/model.gguf",
    "/home/me",
    "./models/foo",
    "../models/foo",
    "/",
    "//",
  ]) {
    assert.equal(
      resolve(id, [gguf], []).selectedId,
      null,
      `${id} was resolved onto an unrelated row`,
    );
  }
});

test("a locally selected path row keeps its exact selection", () => {
  // models_dir rows are keyed by path and must still resolve by exact match.
  const row = localRow({
    id: "C:\\Users\\me\\models\\mymodel",
    load_id: "C:\\Users\\me\\models\\mymodel",
    display_name: "mymodel",
    path: "C:\\Users\\me\\models\\mymodel",
    source: "models_dir",
    model_format: "safetensors",
  });
  assert.equal(resolve(row.id, [], [row]).selectedId, row.id);
});

// ---------------------------------------------------------------------------
// One oracle for a row's format family
//
// inventory-dedupe reads a truthy partialTransport as "snapshot/model family"
// and a missing one as "gguf". The resolver must not contradict that when it
// decides whether a provisional download may adopt an unclassified row.
// ---------------------------------------------------------------------------

test("a gguf download does not adopt a snapshot partial the deduper kept apart", () => {
  const repoId = "unsloth/hybrid-repo";
  // A safetensors snapshot fragment: carries a transport, so it is model-family.
  const snapshot = localRow({
    id: repoId,
    inventory_id: `hf_cache:unknown:${encodeURIComponent(repoId)}`,
    load_id: repoId,
    display_name: "hybrid-repo",
    path: `/cache/models--unsloth--hybrid-repo`,
    model_id: repoId,
    model_format: "unknown",
    partial: true,
    partial_transport: "xet",
    partial_resumable: true,
  });

  // The deduper deliberately keeps it beside an unrelated gguf job.
  const gguf = cachedRow(repoId, "gguf", { partial: true });
  const deduped = dedupeSameSourceHubCacheRows({
    cachedRows: [gguf],
    localRows: [snapshot],
  });
  assert.equal(
    deduped.localRows.length,
    1,
    "precondition: the deduper keeps the unrelated snapshot partial",
  );

  // So the resolver must not then hand that snapshot to the cancelled gguf download.
  assert.equal(
    resolve(
      optimisticInventoryId("gguf", repoId),
      [],
      deduped.localRows,
    ).selectedId,
    null,
    "a gguf download adopted a model-family snapshot partial",
  );
});

test("a gguf download still adopts an unclassified gguf partial", () => {
  const repoId = "unsloth/gguf-repo";
  const partial = localRow({
    id: repoId,
    inventory_id: `hf_cache:unknown:${encodeURIComponent(repoId)}`,
    load_id: repoId,
    display_name: "gguf-repo",
    path: `/cache/models--unsloth--gguf-repo`,
    model_id: repoId,
    model_format: "unknown",
    partial: true,
    partial_transport: null,
  });

  assert.equal(
    resolve(optimisticInventoryId("gguf", repoId), [], [partial]).selectedId,
    partial.id,
    "cancelling a gguf download lost its own unclassified partial",
  );
});

test("a transport-less partial is still adopted, in either direction", () => {
  // Only the positive direction is provable: the backend never writes a
  // transport for a GGUF partial, but a snapshot partial with no cancel marker
  // and no manifest also reports none. So an absent transport is not evidence,
  // and both downloads must keep their own cancelled partial.
  const repoId = "unsloth/gguf-repo";
  const partial = localRow({
    id: repoId,
    inventory_id: `hf_cache:unknown:${encodeURIComponent(repoId)}`,
    load_id: repoId,
    display_name: "gguf-repo",
    path: `/cache/models--unsloth--gguf-repo`,
    model_id: repoId,
    model_format: "unknown",
    partial: true,
    partial_transport: null,
  });

  for (const format of ["gguf", "safetensors"] as const) {
    assert.equal(
      resolve(optimisticInventoryId(format, repoId), [], [partial]).selectedId,
      partial.id,
      `a ${format} download lost a transport-less partial`,
    );
  }
});

test("a known-format selection does not fall back onto a proven other family", () => {
  const repoId = "unsloth/hybrid-repo";
  const snapshot = localRow({
    id: repoId,
    inventory_id: `hf_cache:unknown:${encodeURIComponent(repoId)}`,
    load_id: repoId,
    display_name: "hybrid-repo",
    path: `/cache/models--unsloth--hybrid-repo`,
    model_id: repoId,
    model_format: "unknown",
    partial: true,
    partial_transport: "xet", // model family
  });

  assert.equal(
    resolve(cachedInventoryId("gguf", repoId), [], [snapshot]).selectedId,
    null,
    "a gguf cache selection fell back onto a model-family partial",
  );
  assert.equal(
    resolve(cachedInventoryId("safetensors", repoId), [], [snapshot])
      .selectedId,
    snapshot.id,
    "a safetensors selection lost its own model-family partial",
  );
});

// ---------------------------------------------------------------------------
// A complete row carries no transport evidence, so it must not be classified by it
// ---------------------------------------------------------------------------

test("a complete unclassified local row is suppressed by any complete cache row", () => {
  // partialTransport is null on every complete row, so reading it as "gguf family"
  // would retain the duplicate beside safetensors and drop it beside gguf. The row
  // is the same row in both cases.
  for (const format of ["gguf", "safetensors"] as const) {
    const repoId = "unsloth/complete-repo";
    const complete = cachedRow(repoId, format);
    const unknownLocal = localRow({
      id: repoId,
      inventory_id: `hf_cache:unknown:${encodeURIComponent(repoId)}`,
      load_id: repoId,
      display_name: "complete-repo",
      path: `/cache/models--unsloth--complete-repo`,
      model_id: repoId,
      model_format: "unknown",
      partial: false,
      partial_transport: null,
    });

    const deduped = dedupeSameSourceHubCacheRows({
      cachedRows: [complete],
      localRows: [unknownLocal],
    });
    assert.equal(
      deduped.localRows.length,
      0,
      `a complete unclassified row survived beside a complete ${format} row`,
    );
  }
});

// ---------------------------------------------------------------------------
// Scoped download classification, including records written before inventoryKind
// ---------------------------------------------------------------------------

test("scoped file sets classify by extension, case-insensitively", () => {
  assert.equal(scopedDownloadInventoryKind(["model.gguf"]), "gguf");
  assert.equal(scopedDownloadInventoryKind(["MODEL.GGUF"]), "gguf");
  assert.equal(scopedDownloadInventoryKind(["model.safetensors"]), "model");
  assert.equal(scopedDownloadInventoryKind(["gguf/model.safetensors"]), "model");
  assert.equal(scopedDownloadInventoryKind([]), "model");
  assert.equal(scopedDownloadInventoryKind(null), "model");
  assert.equal(scopedDownloadInventoryKind(undefined), "model");
});

test("an explicit inventory kind wins over the variant shape", () => {
  assert.equal(downloadInventoryHintKind("model", "@rag-embedding", "gguf"), "gguf");
  assert.equal(downloadInventoryHintKind("model", "Q4_K_M", "model"), "model");
  assert.equal(downloadInventoryHintKind("dataset", "@anything", "gguf"), "dataset");
});

test("an unscoped quant variant is still gguf without an explicit kind", () => {
  assert.equal(downloadInventoryHintKind("model", "Q4_K_M", undefined), "gguf");
  assert.equal(downloadInventoryHintKind("model", null, undefined), "model");
});

// ---------------------------------------------------------------------------
// URL synchronization
// ---------------------------------------------------------------------------

test("the gguf file query survives canonicalization but a stale one is dropped", () => {
  const gguf = resolveSelectionUrlSync({
    isDiscoverTab: false,
    urlModel: "cache:gguf:org/repo",
    selectionInputId: "cache:gguf:org/repo",
    resolvedSelectedId: "cache:gguf:org%2Frepo",
    resolvedModelFormat: "gguf",
  });
  assert.equal(gguf?.action, "replace");
  assert.equal(gguf?.preserveGgufFile, true);

  const safetensors = resolveSelectionUrlSync({
    isDiscoverTab: false,
    urlModel: "hf_cache:unknown:org%2Frepo",
    selectionInputId: "hf_cache:unknown:org%2Frepo",
    resolvedSelectedId: "cache:safetensors:org%2Frepo",
    resolvedModelFormat: "safetensors",
  });
  assert.equal(safetensors?.action, "replace");
  assert.equal(safetensors?.preserveGgufFile, false);
});

test("a null selection never invents a row", () => {
  assert.equal(resolve(null, [cachedRow("org/repo", "gguf")], []).selectedId, null);
  assert.equal(
    resolveSelectionUrlSync({
      isDiscoverTab: false,
      urlModel: null,
      selectionInputId: null,
      resolvedSelectedId: null,
      resolvedModelFormat: null,
    }),
    null,
  );
});

// ---------------------------------------------------------------------------
// Determinism: the resolver must be idempotent, or the URL sync effect loops
// ---------------------------------------------------------------------------

test("resolution is idempotent across every row shape combination", () => {
  const repoId = "org/repo";
  const formats = ["gguf", "safetensors", "adapter", "checkpoint", "unknown"] as const;
  const sources = ["cache", "download", "hf_cache"] as const;
  let checked = 0;

  for (const rowFormat of formats) {
    for (const partial of [false, true]) {
      for (const transport of [null, "xet"]) {
        const cached = [
          cachedRow(repoId, rowFormat, { partial, partial_transport: transport }),
        ];
        const locals = [
          localRow({
            id: repoId,
            inventory_id: `hf_cache:${rowFormat}:${encodeURIComponent(repoId)}`,
            load_id: repoId,
            display_name: "repo",
            path: "/cache/models--org--repo",
            model_id: repoId,
            model_format: rowFormat,
            partial,
            partial_transport: transport,
          }),
        ];
        for (const source of sources) {
          for (const selFormat of formats) {
            const id = `${source}:${selFormat}:${encodeURIComponent(repoId)}`;
            for (const [c, l] of [
              [cached, []],
              [[], locals],
              [cached, locals],
            ] as const) {
              const once = resolve(id, c, l).selectedId;
              const twice = resolve(once, c, l).selectedId;
              assert.equal(
                twice,
                once,
                `not idempotent for ${id}: ${once} -> ${twice}`,
              );
              checked += 1;
            }
          }
        }
      }
    }
  }
  assert.ok(checked >= 900, `expected a wide sweep, only checked ${checked}`);
});
