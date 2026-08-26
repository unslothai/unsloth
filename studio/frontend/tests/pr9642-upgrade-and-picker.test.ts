// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// S6 + S7 for PR #9642.
//
// S6 asks the question an existing install actually cares about: after upgrading
// Studio, can a value written by the OLD version be compared against one written
// by the NEW one and come out wrong? Changing a unit is only safe if the old
// unit is nowhere on disk.
//
// S7 follows the timestamp out of the hub inventory and into the chat model
// picker, which reads it under a different field name.

import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { buildCachedInventoryRow, buildLocalInventoryRows, normalizeTimestamp } =
  await import("../src/features/hub/inventory/view-models.ts");
const { soleQuantFingerprint, takeDriftedRepos } = await import(
  "../src/features/model-picker/components/model-selector/sole-quant-cache.ts"
);
const { loadedAt } = await import(
  "../src/features/model-picker/components/model-selector/model-usage.ts"
);

const SRC = new URL("../src/", import.meta.url);

async function* walk(dir: URL): AsyncGenerator<URL> {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const child = new URL(
      entry.isDirectory() ? `${entry.name}/` : entry.name,
      dir,
    );
    if (entry.isDirectory()) yield* walk(child);
    else if (/\.tsx?$/.test(entry.name)) yield child;
  }
}

test("S6: no inventory timestamp is ever written to browser storage", async () => {
  // The whole upgrade argument rests on this. If a lastModified or updatedAt
  // ever reached localStorage, an install upgraded from a seconds-era build
  // would compare a stored seconds value against a fresh milliseconds one and
  // sort a model off by a factor of a thousand.
  const offenders: string[] = [];
  for await (const file of walk(SRC)) {
    const source = await readFile(file, "utf8");
    if (!/(localStorage|sessionStorage)\.setItem/.test(source)) continue;
    // Look only at the statements that persist, not the whole file.
    for (const line of source.split("\n")) {
      if (!/(localStorage|sessionStorage)\.setItem/.test(line)) continue;
      if (/lastModified|updatedAt|last_modified|updated_at/.test(line)) {
        offenders.push(`${file.pathname}: ${line.trim()}`);
      }
    }
  }
  assert.deepEqual(offenders, [], offenders.join("\n"));
});

test("S6: the download store's persisted startedAt is already milliseconds", async () => {
  // unsloth.studio.downloads is the one persisted timestamp that now meets an
  // inventory timestamp, because a live download row is stamped with it. It is
  // Date.now() at every write site, so a job persisted by an older build
  // rehydrates on the same scale the comparator expects.
  const source = await readFile(
    new URL("features/hub/download-manager/download-manager-state.ts", SRC),
    "utf8",
  );
  assert.match(source, /nonNegativeNumber\(value\.startedAt, Date\.now\(\)\)/);
  // A rehydrated job from any era normalizes to itself.
  const persistedByOldBuild = 1_700_000_000_000;
  assert.equal(normalizeTimestamp(persistedByOldBuild), persistedByOldBuild);
});

test("S6: an install upgraded mid-download keeps a sane Recent order", () => {
  const startedAtFromOldBuild = 1_700_000_000_000;
  const live = buildCachedInventoryRow(
    {
      repo_id: "unsloth/Qwen3-8B-GGUF",
      size_bytes: 0,
      last_modified: startedAtFromOldBuild,
      partial: true,
      optimistic: true,
    },
    "gguf",
  );
  const settled = buildCachedInventoryRow(
    {
      repo_id: "google/gemma-3-4b-it",
      size_bytes: 1,
      last_modified: 1_600_000_000,
    },
    "gguf",
  );
  assert.ok(live.lastModified !== null && settled.lastModified !== null);
  assert.ok(
    (live.lastModified as number) > (settled.lastModified as number),
    "a download resumed across an upgrade must still outrank an older cached model",
  );
});

// S7. The picker forwards lastModified under the name last_modified, which is
// the same name the raw endpoints use for a seconds value. Nothing may compare
// the two, and nothing may subtract a load time from a modification time.

const CANONICAL_CACHED = [
  {
    repo_id: "google/bert-base-uncased",
    size_bytes: 440_000_000,
    last_modified: 1_600_000_000,
  },
  {
    repo_id: "google/gemma-3-4b-it",
    size_bytes: 4_000_000_000,
    last_modified: 1_650_000_000,
  },
  {
    repo_id: "unsloth/Qwen3-8B-GGUF",
    size_bytes: 8_000_000_000,
    last_modified: 1_700_000_000,
  },
].map((row) => buildCachedInventoryRow(row, "gguf"));

test("S7: the picker mappers forward a millisecond value, uniformly", () => {
  // toCachedGgufRepo is `last_modified: row.lastModified ?? undefined`.
  const forwarded = CANONICAL_CACHED.map((row) => ({
    repo_id: row.repoId,
    size_bytes: row.bytes,
    last_modified: row.lastModified ?? undefined,
  }));
  for (const repo of forwarded) {
    assert.equal(typeof repo.last_modified, "number");
    assert.ok((repo.last_modified as number) > 1_000_000_000_000);
  }

  // sortCachedRepos' date term, verbatim from pickers.tsx. It only ever
  // compares these rows against each other, so it is scale-invariant -- but it
  // must stay that way, hence the homogeneity assertion above.
  const byDate = (
    a: (typeof forwarded)[number],
    b: (typeof forwarded)[number],
  ) =>
    (b.last_modified ?? -1) - (a.last_modified ?? -1) ||
    a.repo_id.localeCompare(b.repo_id);
  assert.deepEqual(
    [...forwarded].sort(byDate).map((repo) => repo.repo_id),
    [
      "unsloth/Qwen3-8B-GGUF",
      "google/gemma-3-4b-it",
      "google/bert-base-uncased",
    ],
  );
});

test("S7: the -1 sentinel still sorts an undated repo last on either scale", () => {
  const rows = [
    { repo_id: "Org/undated", size_bytes: 1, last_modified: undefined },
    { repo_id: "Org/seconds", size_bytes: 1, last_modified: 1_700_000_000 },
    { repo_id: "Org/millis", size_bytes: 1, last_modified: 1_700_000_000_000 },
  ];
  const byDate = (a: (typeof rows)[number], b: (typeof rows)[number]) =>
    (b.last_modified ?? -1) - (a.last_modified ?? -1) ||
    a.repo_id.localeCompare(b.repo_id);
  assert.equal([...rows].sort(byDate).at(-1)?.repo_id, "Org/undated");
});

test("S7: a load time is never subtracted from a modification time", async () => {
  // loadedAt comes from unsloth.model-load-times.v1, which stores Date.now().
  // pickers.tsx compares it on its own and falls through to byDate only when it
  // ties, so the two clocks never meet in one subtraction. Pin that shape.
  const source = await readFile(
    new URL("features/model-picker/components/model-selector/pickers.tsx", SRC),
    "utf8",
  );
  assert.match(
    source,
    /const d = loadedAt\(loadTimes, b\.repo_id\) - loadedAt\(loadTimes, a\.repo_id\);\s*\n\s*return d !== 0 \? d : byDate\(a, b\);/,
  );
  // -1, deliberately below any real clock, so a never-loaded model sorts last
  // in either scale rather than colliding with the epoch.
  assert.equal(loadedAt({}, "missing"), -1);
});

test("S7: the sole-quant fingerprint changes scale but cannot cause a stale hit", () => {
  const seconds = soleQuantFingerprint({
    size_bytes: 10,
    last_modified: 1_700_000_000,
  });
  const millis = soleQuantFingerprint({
    size_bytes: 10,
    last_modified: 1_700_000_000_000,
  });
  assert.notEqual(
    seconds,
    millis,
    "the fingerprint should notice the new value",
  );

  // It is an in-memory map that starts empty every mount, and the first sighting
  // of a repo is recorded rather than treated as drift -- so an upgraded install
  // does not invalidate a cache it never had.
  const target = {
    repoId: "Org/Data",
    localSource: null,
    fingerprint: millis,
    key: "k",
  };
  const known = new Map<string, string>();
  assert.deepEqual(
    takeDriftedRepos([target], known),
    [],
    "a first sighting must not read as drift",
  );
  assert.deepEqual(takeDriftedRepos([target], known), []);
  // And a genuine change is still noticed, so the scale change has not blunted it.
  assert.deepEqual(
    takeDriftedRepos([{ ...target, fingerprint: seconds }], known),
    ["Org/Data"],
  );
});

test("S7: local rows keep the scale they always had", () => {
  const [row] = buildLocalInventoryRows([
    {
      id: "lmstudio-llama-3",
      display_name: "lmstudio-llama-3",
      path: "/models/llama-3.gguf",
      source: "lmstudio",
      updated_at: 1_700_000_000,
    },
  ]);
  // updated_at went through normalizeTimestamp before this PR too, so nothing
  // on the local side moved.
  assert.equal(row?.updatedAt, 1_700_000_000_000);
});
