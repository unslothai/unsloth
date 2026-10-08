// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import {
  EMPTY_FIND_RESULT,
  type FindTarget,
  availableFindTargets,
  findTargetHolding,
  findTargetsVersion,
  onFindRequest,
  registerFindTarget,
  requestFind,
  subscribeFindTargets,
} from "../src/features/find-in-page/lib/find-targets.ts";

const target = (id: string, available: () => boolean, holds: unknown = null): FindTarget => ({
  id,
  available,
  contains: (node) => node === holds,
  search: () => undefined,
  step: () => undefined,
  result: () => EMPTY_FIND_RESULT,
});

test("the bar is offered only targets with something to search, and the one holding focus", () => {
  let open = true;
  const node = {} as Node;
  const before = findTargetsVersion();
  let heard = 0;
  const unsubscribe = subscribeFindTargets(() => {
    heard += 1;
  });
  const unregister = registerFindTarget(target("page", () => open, node));
  assert.ok(findTargetsVersion() > before);
  assert.equal(heard, 1);
  assert.deepEqual(availableFindTargets().map((candidate) => candidate.id), ["page"]);
  assert.equal(findTargetHolding(node)?.id, "page");
  assert.equal(findTargetHolding({} as Node), undefined);
  open = false;
  assert.deepEqual(availableFindTargets(), []);
  assert.equal(findTargetHolding(node), undefined);
  unregister();
  assert.equal(heard, 2);
  unsubscribe();
});

test("a replaced target's late unregister leaves its successor", () => {
  const first = registerFindTarget(target("page", () => true));
  const second = registerFindTarget(target("page", () => true));
  first();
  assert.equal(availableFindTargets().length, 1);
  second();
  assert.equal(availableFindTargets().length, 0);
});

test("requests reach the bar until it stops listening", () => {
  const asked: Array<string | null> = [];
  const stop = onFindRequest((id) => asked.push(id));
  requestFind("browser");
  requestFind(null);
  stop();
  requestFind("browser");
  assert.deepEqual(asked, ["browser", null]);
});
