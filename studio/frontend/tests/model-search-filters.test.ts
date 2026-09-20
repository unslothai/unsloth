// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  filterModelListing,
  hasModelSearchFilters,
  matchesRange,
  modelContextLength,
  parameterRange,
} from "../src/features/hub/lib/model-search-filters.ts";

async function collect<T>(iterator: AsyncIterable<T>): Promise<T[]> {
  const result: T[] = [];
  for await (const value of iterator) result.push(value);
  return result;
}

async function* listing(models: unknown[]) {
  yield* models;
}
const noFetch = async () => {
  throw new Error("unexpected metadata request");
};

test("parameter ranges are inclusive and exclude unknown counts only when active", () => {
  assert.equal(
    parameterRange({ minParams: 2e9, maxParams: 8e9 }),
    "min:2000000000,max:8000000000",
  );
  assert.equal(parameterRange({ maxParams: 5e8 }), "max:500000000");
  assert.equal(matchesRange(undefined), true);
  for (const count of [2e9, 4e9, 8e9])
    assert.equal(matchesRange(count, 2e9, 8e9), true);
  for (const count of [undefined, 0, NaN, Infinity, 1.9e9, 8.1e9])
    assert.equal(matchesRange(count, 2e9, 8e9), false);
  assert.equal(hasModelSearchFilters({ verifiedOnly: false }), false);
  assert.equal(hasModelSearchFilters({ minParams: 0 }), true);
});

test("context comes from text configuration rather than vision position counts", () => {
  assert.equal(modelContextLength({ max_position_embeddings: 40960 }), 40960);
  assert.equal(
    modelContextLength({
      max_position_embeddings: 256,
      text_config: { max_position_embeddings: 131072 },
    }),
    131072,
  );
  for (const field of ["n_positions", "max_seq_len", "seq_length"])
    assert.equal(modelContextLength({ [field]: 8192 }), 8192);
  assert.equal(
    modelContextLength({
      max_position_embeddings: 32768,
      rope_scaling: { factor: 4 },
    }),
    32768,
  );
  assert.equal(modelContextLength({ max_position_embeddings: -1 }), undefined);
});

test("combined filters use actual parameters, GGUF context, and verified organizations", async () => {
  const rows = [
    { name: "verified/model", safetensors: { total: 4e9 } },
    {
      name: "verified/model-GGUF",
      gguf: { total: 4e9, context_length: 32768 },
    },
    { name: "community/model", safetensors: { total: 4e9 } },
    { name: "verified/7B-misleading", safetensors: { total: 12e9 } },
    { name: "verified/unknown" },
    { name: "verified/short", safetensors: { total: 2e9 } },
  ];
  const requests: string[] = [];
  const output = await collect(
    filterModelListing(
      listing(rows),
      {
        minParams: 2e9,
        maxParams: 8e9,
        minContext: 32768,
        maxContext: 131072,
        verifiedOnly: true,
      },
      async (path) => {
        requests.push(path);
        if (path.includes("/organizations/"))
          return { isVerified: path.includes("/verified/") };
        return {
          max_position_embeddings: path.includes("/short/") ? 8192 : 131072,
        };
      },
    ),
  );
  assert.deepEqual(
    output.filter(Boolean).map((m) => (m as { name: string }).name),
    ["verified/model", "verified/model-GGUF"],
  );
  assert.equal(output.length, rows.length);
  assert.equal(
    requests.filter((p) => p.includes("/organizations/verified/")).length,
    1,
  );
  assert.equal(
    requests.some((p) => p.includes("model-GGUF/resolve")),
    false,
  );
});

test("unfiltered searches neither drop models nor fetch metadata", async () => {
  const rows = [
    { name: "user/unknown" },
    { name: "user/other", gguf: { total: 1 } },
  ];
  assert.deepEqual(
    await collect(filterModelListing(listing(rows), {}, noFetch)),
    rows,
  );
});

test("missing or inaccessible config is excluded while service errors propagate", async () => {
  const rows = [{ name: "user/gated", safetensors: { total: 2e9 } }];
  assert.deepEqual(
    await collect(
      filterModelListing(listing(rows), { minContext: 1 }, async () => null),
    ),
    [null],
  );
  await assert.rejects(
    collect(
      filterModelListing(listing(rows), { minContext: 1 }, async () => {
        throw new Error("HTTP 429");
      }),
    ),
    /HTTP 429/,
  );
});

test("a sparse result remains reachable beyond the pagination scan budget", async () => {
  const rows = Array.from({ length: 200 }, (_, i) => ({
    name: `user/${i}`,
    gguf: { total: i === 199 ? 4e9 : 70e9 },
  }));
  const iter = filterModelListing(listing(rows), { maxParams: 8e9 }, noFetch);
  for (let i = 0; i < 192; i++) assert.equal((await iter.next()).value, null);
  const rest = await collect(iter);
  assert.equal((rest.filter(Boolean)[0] as { name: string }).name, "user/199");
});

test("metadata work is bounded and cancellation closes the source iterator", async () => {
  let active = 0,
    peak = 0,
    closed = false;
  async function* source() {
    try {
      for (let i = 0; i < 20; i++) yield { name: `user/${i}` };
    } finally {
      closed = true;
    }
  }
  const iter = filterModelListing(source(), { minContext: 1 }, async () => {
    active++;
    peak = Math.max(peak, active);
    await new Promise((resolve) => setTimeout(resolve, 1));
    active--;
    return { n_positions: 4096 };
  });
  await iter.next();
  await iter.return(undefined);
  assert.equal(peak, 4);
  assert.equal(active, 0);
  assert.equal(closed, true);
});

test("context filters inspect supported language wrappers recursively", () => {
  for (const key of ["llm_config", "language_config", "thinker_config"]) {
    assert.equal(
      modelContextLength({
        [key]: { text_config: { max_position_embeddings: 32768 } },
      }),
      32768,
    );
  }
  assert.equal(
    modelContextLength({
      llm_config: { max_position_embeddings: 32768 },
      max_position_embeddings: 256,
    }),
    32768,
  );
});

test("an exact publisher match is filtered and pinned ahead of a sparse listing", async () => {
  const pinned = {
    name: "publisher/model",
    gguf: { total: 4e9, context_length: 32768 },
  };
  const rows = [
    ...Array.from({ length: 200 }, (_, i) => ({
      name: `user/${i}`,
      gguf: { total: 70e9 },
    })),
    pinned,
  ];
  const iter = filterModelListing(
    listing(rows),
    { maxParams: 8e9, minContext: 32768 },
    noFetch,
    Promise.resolve(pinned),
  );
  assert.deepEqual((await iter.next()).value, pinned);
  assert.equal((await collect(iter)).filter(Boolean).length, 0);
  const excluded = await collect(
    filterModelListing(
      listing([]),
      { maxParams: 2e9 },
      noFetch,
      Promise.resolve(pinned),
    ),
  );
  assert.deepEqual(excluded.filter(Boolean), []);
});

test("an incomplete pinned response does not hide a matching listing entry", async () => {
  const model = { name: "publisher/model-GGUF", gguf: { total: 4e9 } };
  const output = await collect(
    filterModelListing(
      listing([model]),
      { maxParams: 8e9 },
      noFetch,
      Promise.resolve({ name: model.name }),
    ),
  );
  assert.deepEqual(output.filter(Boolean), [model]);
});

test("rejected priority rows retain the pagination scan budget", async () => {
  const accepted = {
    name: "unsloth/match",
    gguf: { total: 4e9, context_length: 32768 },
  };
  const rows = await collect(
    filterModelListing(
      listing([null, null, accepted]),
      { minContext: 32768 },
      noFetch,
      Promise.resolve(accepted),
    ),
  );
  assert.deepEqual(rows, [accepted, null, null, null]);
});
