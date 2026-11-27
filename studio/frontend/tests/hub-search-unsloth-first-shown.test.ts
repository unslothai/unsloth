// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { listModels, modelInfo } from "@huggingface/hub";
import type * as SearchModule from "../src/features/hub/hooks/use-hub-model-search.ts";
import { LruMap } from "../src/features/hub/lib/lru-map.ts";
import { mergeTaskIterators } from "../src/features/hub/lib/merge-task-iterators.ts";
import * as filters from "../src/features/hub/lib/model-search-filters.ts";
import { isGgufLike } from "../src/features/hub/lib/model-identifiers.ts";
import * as searchText from "../src/features/hub/lib/search-text.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Options = NonNullable<
  Parameters<typeof SearchModule.useHubModelSearch>[1]
>;

const UNSLOTH = [
  "unsloth/Inkling-2-Mini-GGUF",
  "unsloth/Inkling-Small-GGUF",
  "unsloth/Inkling-2-Small-GGUF",
  "unsloth/inkling-GGUF",
  "unsloth/inkling-NVFP4",
  "unsloth/Inkling-Small",
  "unsloth/Inkling-2-Mini",
  "unsloth/Inkling-2-Mini-unsloth-bnb-4bit",
];
const GENERAL = [
  "thinkingmachines/Inkling",
  ...Array.from({ length: 400 }, (_, i) => `org/Inkling-${i}`),
  "unsloth/Inkling-Small",
  "unsloth/Inkling-2-Mini",
];

function fakeHub() {
  const fetchWithTimeout = async (input: string) => {
    const url = new URL(input);
    const names = url.searchParams.get("author") === "unsloth" ? UNSLOTH : GENERAL;
    const offset = Number(url.searchParams.get("cursor") ?? 0);
    const size = Number(url.searchParams.get("limit") ?? 500);
    const rows = names.slice(offset, offset + size).map((id, i) => ({
      _id: id,
      id,
      pipeline_tag: "image-text-to-text",
      downloads: names.length - offset - i,
      lastModified: "2026-01-01T00:00:00.000Z",
    }));
    const headers = new Headers();
    if (offset + size < names.length) {
      const next = new URL(url);
      next.searchParams.set("cursor", String(offset + size));
      headers.set("Link", `<${next}>; rel="next"`);
    }
    return Response.json(rows, { headers });
  };
  return { fetchWithTimeout };
}

function search(query: string, options: Options) {
  let iterator!: AsyncGenerator<unknown>;
  let map!: (raw: unknown) => { id: string } | null;
  const mod = loadWithStubs<typeof SearchModule>(
    new URL("../src/features/hub/hooks/use-hub-model-search.ts", import.meta.url),
    {
      react: {
        useMemo: (create: () => unknown) => create(),
        useCallback: (fn: unknown) => fn,
        useState: (value: unknown) => [value, () => {}],
        useEffect: () => {},
        startTransition: (fn: () => void) => fn(),
      },
      "@/config/env": {
        usePlatformStore: (select: (s: unknown) => unknown) =>
          select({ deviceType: "linux" }),
      },
      "@/lib/model-size": { ggufRepoParams: () => undefined },
      "@/lib/hf-endpoint": {
        getHfEndpoint: () => "https://huggingface.co",
        useHfEndpoint: () => "https://huggingface.co",
      },
      "@huggingface/hub": { listModels, modelInfo },
      "../lib/hf-cache": {
        cachedModelInfo: modelInfo,
        primeCacheFromListing: () => {},
      },
      "../lib/hf-model-meta": {
        EMBEDDING_TAGS: new Set(),
        estimateSizeFromDtypes: () => undefined,
      },
      "../lib/merge-task-iterators": { mergeTaskIterators },
      "../lib/model-capabilities": { detectBaseModel: () => undefined },
      "../lib/model-identifiers": { isGgufLike },
      "../lib/fresh-ranking": { rankFresh: (models: unknown[]) => models },
      "../lib/provider-logos": { resolveOwnerProviderLogo: () => null },
      "../lib/provider-search": {
        matchProviderByQuery: () => null,
        providerSearchTerms: () => [],
        providerHubOrgs: () => [],
      },
      "../lib/network": { fetchWithTimeout: fakeHub().fetchWithTimeout },
      "../lib/unsloth-support": {
        classifyUnslothSupport: ({ modelId }: { modelId: string }) => ({
          status: modelId.endsWith("-NVFP4") ? "unsupported" : "supported",
        }),
      },
      "../lib/model-search-filters": filters,
      "../lib/lru-map": { LruMap },
      "../lib/search-text": searchText,
      "./use-hub-paginated-search": {
        MAX_RAW_ITEMS_PER_BATCH: 192,
        useHubPaginatedSearch: (
          create: (signal: AbortSignal) => AsyncGenerator<unknown>,
          mapItem: typeof map,
        ) => {
          iterator = create(new AbortController().signal);
          map = mapItem;
          return { results: [] };
        },
      },
    },
  );
  mod.useHubModelSearch(query, options);
  return { iterator, map };
}

async function shown(query: string, options: Options): Promise<string[]> {
  const { iterator, map } = search(query, options);
  const ids: string[] = [];
  for await (const raw of iterator) {
    const row = map(raw);
    if (row) ids.push(row.id);
  }
  return ids;
}

test("a GGUF-hiding search floats Unsloth uploads it can show past the GGUF and unsupported ones", async () => {
  const ids = await shown("inkling", { excludeGguf: true });
  assert.deepEqual(ids.slice(0, 3), [
    "unsloth/Inkling-Small",
    "unsloth/Inkling-2-Mini",
    "unsloth/Inkling-2-Mini-unsloth-bnb-4bit",
  ]);
  assert.equal(ids.filter((id) => id === "unsloth/Inkling-2-Mini").length, 1);
});

test("a search that keeps GGUFs still floats the org's top three as uploaded", async () => {
  const ids = await shown("inkling", {});
  assert.deepEqual(ids.slice(0, 3), UNSLOTH.slice(0, 3));
});
