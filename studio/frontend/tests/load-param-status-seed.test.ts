// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Same-checkpoint reload reconciliation for the per-model load settings group (#8039).

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import type {
  PairedLoadParamSeed,
  PairedLoadParamState,
} from "../src/features/chat/lib/resolve-paired-load-param-seed.ts";
import type {
  MlxKvBitsSeed,
  MlxKvBitsSeedState,
} from "../src/features/chat/lib/resolve-mlx-kv-bits-seed.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { resolvePairedLoadParamSeed } = await import(
  "../src/features/chat/lib/resolve-paired-load-param-seed.ts"
);
const { resolveMlxKvBitsSeed } = await import(
  "../src/features/chat/lib/resolve-mlx-kv-bits-seed.ts"
);

const DELEGATES_TO_RESOLVERS =
  /resolvePairedLoadParamSeed\(\{[\s\S]*resolveMlxKvBitsSeed\(\{/;

function paired<T>(
  control: T | null,
  loaded: T | null,
): PairedLoadParamState<T> {
  return { control, loaded };
}

function seed<T>(
  incoming: T | null | undefined,
  previous: PairedLoadParamState<T>,
  options: {
    hydratingExistingModel?: boolean;
    seedLoadParams?: boolean;
  } = {},
): PairedLoadParamSeed<T> {
  return resolvePairedLoadParamSeed({
    incoming,
    previous,
    hydratingExistingModel: options.hydratingExistingModel ?? false,
    seedLoadParams: options.seedLoadParams ?? true,
  });
}

function mlxSeed(
  incomingRequested: number | null,
  previous: MlxKvBitsSeedState,
  options: {
    hydratingExistingModel?: boolean;
    seedLoadParams?: boolean;
    isMlx?: boolean;
  } = {},
): MlxKvBitsSeed {
  return resolveMlxKvBitsSeed({
    isMlx: options.isMlx ?? true,
    mlxKvBitsDefined: true,
    incomingRequested,
    incomingReason: "ok",
    incomingTemplateReason: null,
    incomingNote: null,
    previous,
    hydratingExistingModel: options.hydratingExistingModel ?? false,
    seedLoadParams: options.seedLoadParams ?? true,
  });
}

test("a same-model reload from another client advances an undirty speculative pair", () => {
  assert.deepEqual(seed("mtp", paired("auto", "auto")), {
    control: "mtp",
    loaded: "mtp",
  });
});

test("a dirty speculative control survives while its baseline advances", () => {
  const result = seed("mtp", paired("ngram", "auto"));
  assert.deepEqual(result, { loaded: "mtp" });
});

test("a steady speculative poll touches neither field", () => {
  assert.deepEqual(seed("auto", paired("auto", "auto")), {});
  assert.deepEqual(seed("auto", paired("ngram", "auto")), {});
});

test("an omitted speculative_type leaves the pair alone on older backends", () => {
  assert.deepEqual(seed(undefined, paired("mtp", "mtp")), {});
  assert.deepEqual(seed(undefined, paired("ngram", "auto")), {});
});

test("default controls hydrate with the resident server before a loaded baseline exists", () => {
  assert.deepEqual(seed(true, paired(false, null)), {
    control: true,
    loaded: true,
  });
  assert.deepEqual(seed("mtp", paired("auto", null)), {
    control: "mtp",
    loaded: "mtp",
  });
});

test("tensor parallel and KV dtype follow the same dirty-control rule", () => {
  assert.deepEqual(seed(true, paired(false, false)), {
    control: true,
    loaded: true,
  });
  assert.deepEqual(seed("q8_0", paired("q4_0", "q4_0")), {
    control: "q8_0",
    loaded: "q8_0",
  });
  assert.deepEqual(seed(4, paired(2, 2)), { control: 4, loaded: 4 });
  assert.deepEqual(seed(4, paired(8, 2)), { loaded: 4 });
});

test("mlx kv width reconciles on a same-model reload", () => {
  assert.deepEqual(
    mlxSeed(8, {
      mlxKvBits: 4,
      loadedMlxKvBitsRequested: 4,
      mlxKvQuantReason: "ok",
      chatTemplateOverrideReason: null,
      mlxKvQuantNote: null,
    }),
    {
      mlxKvBits: 8,
      loadedMlxKvBitsRequested: 8,
      mlxKvQuantReason: "ok",
      chatTemplateOverrideReason: null,
      mlxKvQuantNote: null,
    },
  );
  const dirty = mlxSeed(8, {
    mlxKvBits: 6,
    loadedMlxKvBitsRequested: 4,
    mlxKvQuantReason: "old",
    chatTemplateOverrideReason: null,
    mlxKvQuantNote: null,
  });
  assert.equal(dirty.loadedMlxKvBitsRequested, 8);
  assert.ok(!("mlxKvBits" in dirty));
});

test("mlx verdict fields refresh when the requested width is unchanged", () => {
  const result = resolveMlxKvBitsSeed({
    isMlx: true,
    mlxKvBitsDefined: true,
    incomingRequested: 4,
    incomingReason: "new reason",
    incomingTemplateReason: "template refused",
    incomingNote: "note",
    previous: {
      mlxKvBits: 4,
      loadedMlxKvBitsRequested: 4,
      mlxKvQuantReason: "old",
      chatTemplateOverrideReason: null,
      mlxKvQuantNote: null,
    },
    hydratingExistingModel: false,
    seedLoadParams: true,
  });
  assert.deepEqual(result, {
    mlxKvQuantReason: "new reason",
    chatTemplateOverrideReason: "template refused",
    mlxKvQuantNote: "note",
  });
  assert.ok(!("mlxKvBits" in result));
  assert.ok(!("loadedMlxKvBitsRequested" in result));
});

test("the status applier delegates to the paired load-param resolvers", () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const source = readFileSync(
    path.join(
      here,
      "../src/features/chat/lib/apply-inference-status-to-store.ts",
    ),
    "utf8",
  );
  assert.match(source, DELEGATES_TO_RESOLVERS);
  assert.ok(
    !source.includes("prevState.loadedSpeculativeType === null"),
    "the inline speculative guard must not survive alongside the resolver",
  );
});
