// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Shared body for S3 (node) and S8 (chromium/firefox/webkit) of PR #9642.
// ES2019 makes sort stable everywhere, but only for a comparator that is pure,
// reflexive, anti-symmetric and transitive; given anything less, V8, JSC and
// SpiderMonkey may each return a different order. Proving the algebra is what
// makes Recent engine-independent. Three browsers only confirm it.

import {
  type InventoryItem,
  compareInventoryItemsByRecent,
} from "../../src/features/hub/catalog/inventory-sort.ts";
import {
  buildCachedInventoryRow,
  buildLocalInventoryRows,
} from "../../src/features/hub/inventory/view-models.ts";

const T1 = 1_700_000_000;
const T2 = 1_800_000_000;
const T3 = 1_900_000_000;

/** The distinct timestamp shapes the comparator has to reason about. */
export const ALGEBRA_VALUES: ReadonlyArray<number | undefined> = [
  undefined,
  0,
  T1,
  T2,
  T3,
  T1 * 1000,
  T3 * 1000,
];

function cachedItem(id: string, lastModified?: number): InventoryItem {
  return {
    variant: "cached",
    row: buildCachedInventoryRow(
      { repo_id: id, size_bytes: 1, last_modified: lastModified },
      "gguf",
    ),
  };
}

function localItem(id: string, updatedAt?: number): InventoryItem {
  const [row] = buildLocalInventoryRows([
    {
      id,
      display_name: id,
      path: `/models/${id}.gguf`,
      source: "lmstudio",
      updated_at: updatedAt,
    },
  ]);
  return { variant: "local", row };
}

function sign(value: number): number {
  return value < 0 ? -1 : value > 0 ? 1 : 0;
}

/** A stable sort written the obvious way, as the oracle for the engine's sort. */
function insertionSort<T>(
  input: readonly T[],
  compare: (a: T, b: T) => number,
): T[] {
  const out: T[] = [];
  for (const current of input) {
    let at = out.length;
    while (at > 0) {
      const previous = out[at - 1] as T;
      if (compare(previous, current) <= 0) break;
      at -= 1;
    }
    out.splice(at, 0, current);
  }
  return out;
}

function permutations<T>(input: readonly T[]): T[][] {
  if (input.length <= 1) return [[...input]];
  const out: T[][] = [];
  for (let i = 0; i < input.length; i += 1) {
    const head = input[i] as T;
    const rest = [...input.slice(0, i), ...input.slice(i + 1)];
    for (const tail of permutations(rest)) out.push([head, ...tail]);
  }
  return out;
}

/**
 * Runs every algebra check. Returns a list of human-readable failures so the
 * same body can be asserted by node:test or read back out of a browser page.
 */
export function runSortAlgebraChecks(): string[] {
  const failures: string[] = [];
  const fail = (message: string) => {
    if (failures.length < 40) failures.push(message);
  };

  // Both row kinds, so the cached/local asymmetry is covered in every property.
  const universe: InventoryItem[] = [];
  for (const value of ALGEBRA_VALUES) {
    universe.push(cachedItem(`Org/cached-${String(value)}`, value));
    universe.push(localItem(`local:${String(value)}`, value));
  }

  for (const a of universe) {
    const self = compareInventoryItemsByRecent(a, a);
    if (Number.isNaN(self)) fail(`reflexivity: NaN for ${a.row.id}`);
    if (self !== 0)
      fail(`reflexivity: ${a.row.id} compared to itself gave ${self}`);
  }

  for (const a of universe) {
    for (const b of universe) {
      const ab = compareInventoryItemsByRecent(a, b);
      const ba = compareInventoryItemsByRecent(b, a);
      if (Number.isNaN(ab) || Number.isNaN(ba)) {
        fail(`NaN comparing ${a.row.id} and ${b.row.id}`);
        continue;
      }
      if (sign(ab) !== -sign(ba)) {
        fail(
          `anti-symmetry: ${a.row.id} vs ${b.row.id} gave ${sign(ab)} and ${sign(ba)}`,
        );
      }
    }
  }

  for (const a of universe) {
    for (const b of universe) {
      for (const c of universe) {
        const ab = sign(compareInventoryItemsByRecent(a, b));
        const bc = sign(compareInventoryItemsByRecent(b, c));
        const ac = sign(compareInventoryItemsByRecent(a, c));
        if (ab <= 0 && bc <= 0 && ac > 0) {
          fail(
            `transitivity: ${a.row.id} <= ${b.row.id} <= ${c.row.id} but a > c`,
          );
        }
        if (ab >= 0 && bc >= 0 && ac < 0) {
          fail(
            `transitivity: ${a.row.id} >= ${b.row.id} >= ${c.row.id} but a < c`,
          );
        }
      }
    }
  }

  // Every arrangement carrying ties and unknowns must match a stable reference
  // sort, or the order on screen depends on the user's engine.
  const sample: InventoryItem[] = [
    cachedItem("Org/unknown-a"),
    localItem("local:tie-1", T2),
    cachedItem("Org/tie-2", T2),
    localItem("local:newest", T3),
    cachedItem("Org/oldest", T1),
    cachedItem("Org/unknown-b"),
  ];

  let permutationCount = 0;
  for (const arrangement of permutations(sample)) {
    permutationCount += 1;
    const engine = [...arrangement].sort(compareInventoryItemsByRecent);
    const oracle = insertionSort(arrangement, compareInventoryItemsByRecent);
    const engineIds = engine.map((item) => item.row.id).join(",");
    const oracleIds = oracle.map((item) => item.row.id).join(",");
    if (engineIds !== oracleIds) {
      fail(
        `permutation disagreed with the stable oracle: ${engineIds} vs ${oracleIds}`,
      );
    }
    if (engine.length !== arrangement.length) {
      fail(`permutation changed length: ${engineIds}`);
    }
  }
  if (permutationCount !== 720) {
    fail(`expected 720 permutations, ran ${permutationCount}`);
  }

  return failures;
}

/** A fixed ordering the browser matrix compares byte for byte across engines. */
export function canonicalRecentOrder(): string[] {
  const items: InventoryItem[] = [
    cachedItem("Org/bert-base-uncased", T1),
    cachedItem("Org/gemma-3", T1 + 100),
    localItem("local:lmstudio-newest", T3),
    localItem("local:lmstudio-mid", T2),
    cachedItem("Org/no-timestamp"),
    cachedItem("Org/already-millis", T3 * 1000 + 1),
  ];
  return [...items]
    .sort(compareInventoryItemsByRecent)
    .map((item) => item.row.id);
}
