// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// S1 + S2 for PR #9642. The Recent sort only works if every inventory row reaches
// the comparator in one unit, so these pin the normalizer's contract at its
// boundaries and then replay the shapes real backends actually emit on each OS.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { buildCachedInventoryRow, buildLocalInventoryRows, normalizeTimestamp } =
  await import("../src/features/hub/inventory/view-models.ts");
const { compareInventoryItemsByRecent, inventoryItemUpdatedAt } = await import(
  "../src/features/hub/catalog/inventory-sort.ts"
);

const YEAR_2000_MS = 946_684_800_000;
const YEAR_2100_MS = 4_102_444_800_000;

// S1. Every value the wire can carry, and what the normalizer must make of it.
// A backend that omits the key, a stat() that failed, and a filesystem with no
// clock all arrive here as different JSON, and all three must mean "unknown"
// rather than "1970", or they would sort above nothing and below everything.
const BOUNDARY_TABLE: ReadonlyArray<{
  readonly label: string;
  readonly input: number | null | undefined;
  readonly expected: number | null;
}> = [
  { label: "key absent from the payload", input: undefined, expected: null },
  { label: "explicit null from the hub route", input: null, expected: null },
  { label: "zero from a failed stat()", input: 0, expected: null },
  { label: "negative clock", input: -1, expected: null },
  { label: "pre-epoch", input: -1_700_000_000, expected: null },
  { label: "NaN", input: Number.NaN, expected: null },
  {
    label: "positive infinity",
    input: Number.POSITIVE_INFINITY,
    expected: null,
  },
  {
    label: "negative infinity",
    input: Number.NEGATIVE_INFINITY,
    expected: null,
  },
  { label: "one second past the epoch", input: 1, expected: 1000 },
  {
    label: "seconds, whole",
    input: 1_700_000_000,
    expected: 1_700_000_000_000,
  },
  {
    label: "seconds, fractional mtime as huggingface_hub reports it",
    input: 1_700_000_000.789,
    expected: 1_700_000_000_789,
  },
  {
    label: "last value treated as seconds",
    input: 9_999_999_999,
    expected: 9_999_999_999_000,
  },
  {
    label: "first value treated as milliseconds",
    input: 10_000_000_000,
    expected: 10_000_000_000,
  },
  {
    label: "milliseconds pass through untouched",
    input: 1_700_000_000_000,
    expected: 1_700_000_000_000,
  },
];

test("S1: normalizeTimestamp maps every wire value to a millisecond or to unknown", () => {
  for (const row of BOUNDARY_TABLE) {
    assert.equal(
      normalizeTimestamp(row.input),
      row.expected,
      `normalizeTimestamp(${String(row.input)}) for ${row.label}`,
    );
  }
});

test("S1: normalizeTimestamp is idempotent over every reachable timestamp", () => {
  // formatLocalUpdated re-normalizes whatever it is handed, and a row's
  // updatedAt has already been normalized once by the row builder, so the
  // function is applied twice on the display path. That is only safe where
  // applying it twice equals applying it once.
  //
  // The pivot is a heuristic, not a converter: it is idempotent exactly when the
  // first pass lands at or above 10_000_000_000. For a seconds input that means
  // any mtime at or after 1970-04-26. Every real filesystem timestamp qualifies
  // by a factor of ~170, so the reachable domain is safe -- but see the
  // companion test below for where the property genuinely stops holding.
  for (const row of BOUNDARY_TABLE) {
    const once = normalizeTimestamp(row.input);
    if (once !== null && once < 10_000_000_000) continue;
    assert.equal(
      normalizeTimestamp(once),
      once,
      `idempotence for ${row.label}`,
    );
  }
});

test("S1: the pivot's blind spot is confined to the first 116 days of 1970", () => {
  // Documenting the limit rather than hiding it. A raw seconds value below
  // 10_000_000 normalizes into a range the function still reads as seconds, so a
  // second pass multiplies it again. No filesystem this code can reach reports a
  // 1970 mtime for a model that exists, and an unreadable one arrives as 0 or
  // null instead, so this is unreachable rather than latent -- but it is the
  // reason normalizeTimestamp must stay a single-boundary ingestion helper.
  const lastAffectedSeconds = 9_999_999;
  assert.equal(normalizeTimestamp(lastAffectedSeconds), 9_999_999_000);
  assert.equal(normalizeTimestamp(9_999_999_000), 9_999_999_000_000);

  const firstSafeSeconds = 10_000_000;
  const onceSafe = normalizeTimestamp(firstSafeSeconds);
  assert.equal(onceSafe, 10_000_000_000);
  assert.equal(normalizeTimestamp(onceSafe), onceSafe);

  // 1970-04-26T17:46:40Z, and every real mtime is far above it.
  assert.equal(new Date(firstSafeSeconds * 1000).getUTCFullYear(), 1970);
  assert.ok(Date.now() / 1000 > firstSafeSeconds * 100);
});

test("S1: a malformed runtime value never reaches the comparator as NaN", () => {
  // The static type says number, but JSON from an older or patched backend can
  // say anything. A NaN operand would make the comparator return NaN, which is
  // where engines are free to disagree about the final order.
  const hostile: readonly unknown[] = [
    "1700000000",
    "",
    {},
    [],
    true,
    false,
    Number.NaN,
  ];
  for (const value of hostile) {
    const normalized = normalizeTimestamp(value as number);
    assert.equal(
      normalized,
      null,
      `hostile input ${JSON.stringify(value)} must normalize to null`,
    );
  }
});

// S2. One fixture family per filesystem/OS shape the backend can be sitting on.
// The PR does not branch on platform, so what varies between these is only the
// value the backend puts on the wire -- which is exactly the axis that matters.
const NOV_2023_S = 1_700_000_000;

type CachedPayload = {
  repo_id: string;
  size_bytes: number;
  last_modified?: number | null;
};

type OsFixture = {
  readonly label: string;
  readonly note: string;
  readonly cached: readonly CachedPayload[];
  readonly local: ReadonlyArray<{ name: string; updated_at?: number | null }>;
  /** Repo ids and local ids in the order Recent must produce. */
  readonly expected: readonly string[];
};

const OS_FIXTURES: readonly OsFixture[] = [
  {
    label: "linux ext4",
    note: "nanosecond mtime surfaced as a fractional float of seconds",
    cached: [
      { repo_id: "Org/older", size_bytes: 1, last_modified: NOV_2023_S },
      {
        repo_id: "Org/newer",
        size_bytes: 1,
        last_modified: NOV_2023_S + 86_400.123456,
      },
    ],
    local: [{ name: "lmstudio-mid", updated_at: NOV_2023_S + 43_200 }],
    expected: ["Org/newer", "local:lmstudio-mid", "Org/older"],
  },
  {
    label: "macos apfs",
    note: "same float seconds shape as ext4; APFS keeps nanosecond precision",
    cached: [
      {
        repo_id: "Org/newer",
        size_bytes: 1,
        last_modified: NOV_2023_S + 100.000000001,
      },
      { repo_id: "Org/older", size_bytes: 1, last_modified: NOV_2023_S },
    ],
    local: [],
    expected: ["Org/newer", "Org/older"],
  },
  {
    label: "windows ntfs",
    note: "100ns resolution, still a float of seconds through Path.stat().st_mtime",
    cached: [
      {
        repo_id: "Org/newer",
        size_bytes: 1,
        last_modified: NOV_2023_S + 0.0000001,
      },
      { repo_id: "Org/older", size_bytes: 1, last_modified: NOV_2023_S },
    ],
    local: [{ name: "lmstudio-newest", updated_at: NOV_2023_S + 10 }],
    expected: ["local:lmstudio-newest", "Org/newer", "Org/older"],
  },
  {
    label: "windows fat32 / exfat",
    note: "two-second mtime granularity, and mtime stored in local time so a DST boundary shifts it an hour",
    cached: [
      // Rounded to an even second by the filesystem, then shifted by a DST jump.
      {
        repo_id: "Org/dst-shifted",
        size_bytes: 1,
        last_modified: NOV_2023_S + 3_600,
      },
      { repo_id: "Org/even-second", size_bytes: 1, last_modified: NOV_2023_S },
    ],
    local: [],
    expected: ["Org/dst-shifted", "Org/even-second"],
  },
  {
    label: "wsl drvfs",
    note: "/mnt/c passthrough; seconds survive the translation",
    cached: [
      { repo_id: "Org/newer", size_bytes: 1, last_modified: NOV_2023_S + 5 },
      { repo_id: "Org/older", size_bytes: 1, last_modified: NOV_2023_S },
    ],
    local: [{ name: "wsl-local", updated_at: NOV_2023_S + 2 }],
    expected: ["Org/newer", "local:wsl-local", "Org/older"],
  },
  {
    label: "windows hf cache without symlinks",
    note: "no Developer Mode, so hf moves the blob into snapshots/; blob_path still resolves and mtime survives",
    cached: [
      {
        repo_id: "Org/copied-blob",
        size_bytes: 1,
        last_modified: NOV_2023_S + 7,
      },
      { repo_id: "Org/older", size_bytes: 1, last_modified: NOV_2023_S },
    ],
    local: [],
    expected: ["Org/copied-blob", "Org/older"],
  },
  {
    label: "windows broken snapshot symlink",
    note: "_blob_mtime swallows the OSError and returns 0.0, so the hub route serializes null",
    cached: [
      { repo_id: "Org/dated", size_bytes: 1, last_modified: NOV_2023_S },
      { repo_id: "Org/broken-link", size_bytes: 1, last_modified: null },
    ],
    local: [],
    expected: ["Org/dated", "Org/broken-link"],
  },
  {
    label: "network share with no usable clock",
    note: "stat() reports 0; the scanner drops the key below its > 0 gate",
    cached: [
      { repo_id: "Org/dated", size_bytes: 1, last_modified: NOV_2023_S },
      { repo_id: "Org/zero-mtime", size_bytes: 1, last_modified: 0 },
    ],
    local: [],
    expected: ["Org/dated", "Org/zero-mtime"],
  },
  {
    label: "old backend that never sends the field",
    note: "an installed Studio predating the timestamp work; every row is unknown",
    cached: [
      { repo_id: "Org/alpha", size_bytes: 1 },
      { repo_id: "Org/beta", size_bytes: 1 },
      { repo_id: "Org/gamma", size_bytes: 1 },
    ],
    local: [],
    // All unknown, so the comparator must return 0 throughout and leave the
    // caller's source order exactly as it found it.
    expected: ["Org/alpha", "Org/beta", "Org/gamma"],
  },
  {
    label: "future backend already emitting milliseconds",
    note: "forwards compatibility: the normalizer must not multiply these again",
    cached: [
      {
        repo_id: "Org/newer",
        size_bytes: 1,
        last_modified: (NOV_2023_S + 60) * 1000,
      },
      { repo_id: "Org/older", size_bytes: 1, last_modified: NOV_2023_S * 1000 },
    ],
    local: [{ name: "ms-local", updated_at: (NOV_2023_S + 30) * 1000 }],
    expected: ["Org/newer", "local:ms-local", "Org/older"],
  },
  {
    label: "mixed fleet: seconds and milliseconds for the same instant",
    note: "a rolling upgrade where one endpoint has been converted and the other has not",
    cached: [
      {
        repo_id: "Org/in-seconds",
        size_bytes: 1,
        last_modified: NOV_2023_S + 1,
      },
      {
        repo_id: "Org/in-millis",
        size_bytes: 1,
        last_modified: (NOV_2023_S + 2) * 1000,
      },
    ],
    local: [],
    expected: ["Org/in-millis", "Org/in-seconds"],
  },
];

function buildItems(fixture: OsFixture) {
  const cached = fixture.cached.map((row) => ({
    variant: "cached" as const,
    row: buildCachedInventoryRow(row, "gguf"),
  }));
  const local = fixture.local.map((entry) => {
    const [row] = buildLocalInventoryRows([
      {
        id: `local:${entry.name}`,
        display_name: entry.name,
        path: `C:\\models\\${entry.name}.gguf`,
        source: "lmstudio",
        updated_at: entry.updated_at,
      },
    ]);
    return { variant: "local" as const, row };
  });
  return [...cached, ...local];
}

for (const fixture of OS_FIXTURES) {
  test(`S2: ${fixture.label} sorts newest first (${fixture.note})`, () => {
    const items = buildItems(fixture);

    for (const item of items) {
      const value = inventoryItemUpdatedAt(item);
      if (value === null) continue;
      // A seconds value that escaped normalization would land in 1970; a
      // doubly-multiplied one would land past the year 4000. Both are caught here.
      assert.ok(
        value > YEAR_2000_MS && value < YEAR_2100_MS,
        `${fixture.label}: ${value} is not a plausible millisecond timestamp`,
      );
    }

    const sorted = [...items].sort(compareInventoryItemsByRecent);
    assert.deepEqual(
      sorted.map((item) =>
        item.variant === "cached" ? item.row.repoId : item.row.id,
      ),
      [...fixture.expected],
      `${fixture.label}: unexpected Recent order`,
    );
  });
}

test("S2: rows without a timestamp always trail rows that have one", () => {
  for (const fixture of OS_FIXTURES) {
    const sorted = [...buildItems(fixture)].sort(compareInventoryItemsByRecent);
    const stamps = sorted.map(inventoryItemUpdatedAt);
    const firstUnknown = stamps.indexOf(null);
    if (firstUnknown === -1) continue;
    assert.ok(
      stamps.slice(firstUnknown).every((value) => value === null),
      `${fixture.label}: a dated row sorted below an undated one`,
    );
  }
});

test("S2: an all-unknown inventory keeps source order, so an old backend is unaffected", () => {
  const fixture = OS_FIXTURES.find(
    (candidate) => candidate.label === "old backend that never sends the field",
  );
  assert.ok(fixture);
  const items = buildItems(fixture);
  const before = items.map((item) => item.row.id);
  const after = [...items]
    .sort(compareInventoryItemsByRecent)
    .map((i) => i.row.id);
  assert.deepEqual(after, before);
});
