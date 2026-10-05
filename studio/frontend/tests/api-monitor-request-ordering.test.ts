// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const USE_API_MONITOR = readSrc("features/api-monitor/use-api-monitor.ts");
const LIST_GENERATION_GUARD =
  /const requestGeneration = \+\+listRequestGeneration\.current;[\s\S]{0,300}?listRequestGeneration\.current !== requestGeneration/;
const CLEAR_INVALIDATION =
  /detailRequestGeneration\.current \+= 1;[\s\S]{0,200}?retainedEntryIds\.current = new Set\(\);/;
const DETAIL_GENERATION_GUARD =
  /detailRequestGeneration\.current !== requestGeneration \|\|[\s\S]{0,100}?!retainedEntryIds\.current\.has\(id\)/;

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

test("only the newest monitor list response publishes", async () => {
  let generation = 0;
  let published = "";
  const load = async (response: Promise<string>) => {
    const requestGeneration = ++generation;
    const value = await response;
    if (generation !== requestGeneration) {
      return;
    }
    published = value;
  };
  const older = deferred<string>();
  const newer = deferred<string>();
  const first = load(older.promise);
  const second = load(newer.promise);

  newer.resolve("new");
  await second;
  older.resolve("old");
  await first;

  assert.equal(published, "new");
  assert.match(USE_API_MONITOR, LIST_GENERATION_GUARD);
});

test("clearing invalidates details that were already in flight", async () => {
  let generation = 0;
  const details: string[] = [];
  const pending = deferred<string>();
  const requestGeneration = generation;
  const receive = pending.promise.then((value) => {
    if (generation !== requestGeneration) {
      return;
    }
    details.push(value);
  });

  generation += 1;
  details.length = 0;
  pending.resolve("late secret");
  await receive;

  assert.deepEqual(details, []);
  assert.match(USE_API_MONITOR, CLEAR_INVALIDATION);
  assert.match(USE_API_MONITOR, DETAIL_GENERATION_GUARD);
});
