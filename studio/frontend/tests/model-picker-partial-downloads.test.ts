// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Partials are listed so they can be deleted, but never loaded.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

import type {
  CachedInventoryRow,
  DiscoverRow,
  LocalInventoryRow,
  SelectedModelView,
} from "../src/features/hub/types.ts";
import { downloadActionLabel } from "../src/features/hub/catalog/use-download-card-state.ts";
import { modelDownloadState } from "../src/features/hub/catalog/model-download-state.ts";
import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { useSelectedModelView } = await import(
  "../src/features/hub/hooks/use-selected-model-view.ts"
);

function read(path: string): string {
  return readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf-8");
}

const PICKERS = read(
  "../src/features/model-picker/components/model-selector/pickers.tsx",
);
const INVENTORY = read(
  "../src/features/model-picker/inventory/use-chat-picker-inventory.ts",
);
const CHAT_ADAPTER = read("../src/features/chat/api/chat-adapter.ts");
const MODELS_TABLE = read("../src/features/hub/catalog/models-table.tsx");

test("the picker inventory lists partial snapshots instead of dropping them", () => {
  // Live downloads stay hidden: the Downloads panel owns them.
  assert.ok(!INVENTORY.includes("isCompleteCachedRow"), "old filter is gone");
  assert.match(
    INVENTORY,
    /function isListableCachedRow\(row: CachedInventoryRow\): boolean \{\n\s*return !row\.liveDownload;\n\}/,
  );
  assert.ok(
    INVENTORY.includes("isListableCachedRow(row) &&"),
    "and both cached lists use it",
  );
  assert.equal(
    INVENTORY.split("isListableCachedRow(row) &&").length - 1,
    2,
    "gguf and non-gguf alike",
  );
});

test("the flag survives the mapping, or no row downstream could tell", () => {
  assert.equal(
    INVENTORY.split("partial: row.partial,").length - 1,
    2,
    "carried onto both cached repo shapes",
  );
});

test("a partial is marked the way the Hub marks one", () => {
  const start = PICKERS.indexOf("function PartialBadge(");
  assert.ok(start > 0, "the picker has a partial mark");
  const badge = PICKERS.slice(start, PICKERS.indexOf("\n}", start));
  assert.match(badge, /size-\[calc\(5px\*var\(--ui-space-scale,1\)\)\] rounded-full bg-status-warning/);
  assert.match(badge, /aria-label="Partial download"/);
  assert.ok(
    MODELS_TABLE.includes('aria-label="Partial download"') &&
      MODELS_TABLE.includes("bg-status-warning"),
    "and the Hub still uses that dot, so the two agree",
  );
  assert.ok(
    !PICKERS.includes("&mdash;"),
    "no em dash in the tooltip, or anywhere else here",
  );
});

test("the mark promises a resume only when the transport can give one", () => {
  // A restart-only partial refetches every byte, so the picker must not say "resume".
  const start = PICKERS.indexOf("function PartialBadge(");
  const badge = PICKERS.slice(start, PICKERS.indexOf("\n}", start));
  assert.ok(badge.includes("{ resumable }: { resumable?: boolean }"));
  assert.match(
    badge,
    /resumable\n?\s*\? "Partial download\. Select to resume it/,
  );
  assert.ok(
    badge.includes('"Partial download. Select to continue it, or delete it."'),
    "and the other branch neither promises nor forbids reusing the bytes",
  );
  // False is not "restart": GGUF repo rows and old backends report false too.
  assert.ok(!badge.includes("starts over"));
  assert.ok(!badge.includes("resumable === false"));
  const chatApi = read("../src/features/chat/api/chat-api.ts");
  assert.ok(chatApi.includes("False on a GGUF repo row by design:"));
  const hub = read("../src/features/hub/catalog/use-download-card-state.ts");
  assert.ok(hub.includes('return partialResumable ? "Resume" : "Continue";'));

  const inventory = read(
    "../src/features/model-picker/inventory/use-chat-picker-inventory.ts",
  );
  assert.equal(
    inventory.split("partial_resumable: row.partialResumable,").length - 1,
    2,
    "both cached repo shapes carry it",
  );
  assert.equal(
    PICKERS.split("<PartialBadge resumable={partialResumable} />").length - 1,
    3,
    "On Device, the Hub and unaligned rows alike",
  );
  assert.equal(
    PICKERS.split("partialResumable={c.partial_resumable}").length - 1,
    2,
  );
  // /api/models/gguf-variants has no partial_resumable, so none is claimed there.
  assert.ok(!PICKERS.includes("partialResumable={variant."));
  const chatVariant = read("../src/features/chat/types/api.ts");
  const detail = chatVariant.slice(
    chatVariant.indexOf("export interface GgufVariantDetail"),
  );
  assert.ok(
    !detail.slice(0, detail.indexOf("\n}")).includes("partial_resumable"),
    "and the chat variant type does not claim the field either",
  );
  assert.equal(
    PICKERS.split("partialResumable={partialResumableSet.has(").length - 1,
    3,
    "all three Hub row renderers",
  );
});

test("complete and partial are alternatives, never both dots on one row", () => {
  assert.equal(
    PICKERS.split(
      "{partial ? <PartialBadge resumable={partialResumable} /> : null}",
    ).length - 1,
    1,
    "drawn in the unaligned branch",
  );
  assert.ok(PICKERS.includes('{alignMeta === "device" && partial ? ('));
  assert.ok(PICKERS.includes('{partial && alignMeta !== "device" ? ('));
  assert.equal(
    PICKERS.split(
      "{downloaded && !partial && !loaded ? <DownloadedBadge /> : null}",
    ).length - 1,
    2,
    "and the on-device dot yields to it in both",
  );
});

test("selecting a picker partial opens its download instead of claiming the weights", () => {
  // isDownloaded true would send a torn snapshot straight to the load path.
  assert.ok(
    PICKERS.includes("isDownloaded: !isPartial,"),
    "the pick reports what is actually on disk",
  );
});

test("Hub selections preserve download completeness and continuation state", () => {
  const capabilities = {
    canTrain: false,
    canChat: false,
    canDelete: true,
    canDownload: true,
    requiresVariant: false,
    supportsLora: false,
    supportsVision: false,
  };
  const cached: CachedInventoryRow = {
    kind: "cache",
    id: "cache:safetensors:Org%2FModel",
    loadId: "Org/Model",
    repoId: "Org/Model",
    owner: "Org",
    repo: "Model",
    isGguf: false,
    modelFormat: "safetensors",
    artifact: "unknown",
    capabilities,
    bytes: 128,
    partial: true,
    partialTransport: "http",
    partialResumable: true,
  };
  const localHfCache: LocalInventoryRow = {
    kind: "local",
    id: "hf_cache:safetensors:Org%2FModel",
    loadId: "Org/Model",
    repoId: "Org/Model",
    owner: "Org",
    title: "Model",
    source: "hf_cache",
    sourceLabel: "Hugging Face cache",
    path: "/cache/models--Org--Model",
    isGguf: false,
    modelFormat: "safetensors",
    artifact: "unknown",
    capabilities,
    updatedAt: 1,
    partial: true,
    partialTransport: "xet",
    partialResumable: false,
  };
  const discover: DiscoverRow = {
    id: "Org/Model",
    owner: "Org",
    repo: "Model",
    result: {
      id: "Org/Model",
      downloads: 0,
      likes: 0,
      isGguf: false,
    },
    isAvailableOnDevice: false,
    isPartialOnDevice: true,
    summary: "Model",
    capabilities: [],
  };
  const base = {
    selectedDiscoverRow: null,
    selectedCachedRow: null,
    selectedLocalRow: null,
    selectedHfResult: null,
    isDatasetMode: false,
  } satisfies Parameters<typeof useSelectedModelView>[0];

  const cases: Array<{
    name: string;
    input: Parameters<typeof useSelectedModelView>[0];
    kind: SelectedModelView["kind"];
    downloaded: boolean;
    partial: boolean;
    transport: string | null;
    resumable: boolean;
    action: "Download" | "Resume" | "Continue";
  }> = [
    {
      name: "direct cache row",
      input: { ...base, selectedCachedRow: cached },
      kind: "cache",
      downloaded: false,
      partial: true,
      transport: "http",
      resumable: true,
      action: "Resume",
    },
    {
      name: "discovery row backed by a cache row",
      input: { ...base, selectedDiscoverRow: discover, selectedCachedRow: cached },
      kind: "discover",
      downloaded: false,
      partial: true,
      transport: "http",
      resumable: true,
      action: "Resume",
    },
    {
      name: "discovery row backed by a local HF-cache row",
      input: {
        ...base,
        selectedDiscoverRow: discover,
        selectedLocalRow: localHfCache,
      },
      kind: "discover",
      downloaded: false,
      partial: true,
      transport: "xet",
      resumable: false,
      action: "Continue",
    },
    {
      name: "direct local HF-cache row",
      input: { ...base, selectedLocalRow: localHfCache },
      kind: "cache",
      downloaded: false,
      partial: true,
      transport: "xet",
      resumable: false,
      action: "Continue",
    },
    {
      name: "discovery partial awaiting its inventory row",
      input: { ...base, selectedDiscoverRow: discover },
      kind: "discover",
      downloaded: false,
      partial: true,
      transport: null,
      resumable: false,
      action: "Continue",
    },
    {
      name: "complete cache row",
      input: {
        ...base,
        selectedCachedRow: {
          ...cached,
          partial: false,
          partialTransport: null,
          partialResumable: false,
        },
      },
      kind: "cache",
      downloaded: true,
      partial: false,
      transport: null,
      resumable: false,
      action: "Download",
    },
  ];

  for (const entry of cases) {
    const result = { current: null as SelectedModelView | null };
    function Harness() {
      result.current = useSelectedModelView(entry.input);
      return null;
    }
    renderToStaticMarkup(createElement(Harness));
    assert.ok(result.current, entry.name);
    assert.equal(result.current.kind, entry.kind, entry.name);
    assert.equal(result.current.isDownloaded, entry.downloaded, entry.name);
    assert.equal(result.current.isPartial, entry.partial, entry.name);
    assert.equal(result.current.partialTransport, entry.transport, entry.name);
    assert.equal(result.current.partialResumable, entry.resumable, entry.name);
    const downloadState = modelDownloadState(result.current);
    assert.deepEqual(
      downloadState,
      {
        isDownloaded: entry.downloaded,
        isPartial: entry.partial,
        partialTransport: entry.transport,
        partialResumable: entry.resumable,
      },
      entry.name,
    );
    assert.equal(
      downloadActionLabel(
        downloadState.isPartial,
        downloadState.partialResumable,
      ),
      entry.action,
      entry.name,
    );
  }

  const inspector = read("../src/features/hub/catalog/model-inspector.tsx");
  assert.equal(inspector.split("{...downloadState}").length - 1, 2);
});

test("listing a partial never makes it auto-loadable", () => {
  // This guard is what makes listing partials safe.
  assert.match(
    CHAT_ADAPTER,
    /function isChattableCachedRepo\([\s\S]*?repo\.partial !== true/,
  );
  assert.ok(
    CHAT_ADAPTER.includes("row.partial !== true"),
    "and the local scan-folder rule agrees",
  );
});

test("a partial GGUF repo carries its own menu, not an empty gutter", () => {
  // A partial repo has no complete quant row, so its own row must offer delete and reveal.
  const start = PICKERS.indexOf("const renderDownloadedGgufRow");
  const row = PICKERS.slice(start, PICKERS.indexOf("\n  };", start));
  assert.ok(row.includes("const isPartialRepo = c.partial === true;"));
  assert.match(
    row,
    /\{isPartialRepo \? \(\n\s*<span className=\{ROW_ACTIONS_PINNED_CLASS\}>\n\s*<ModelRowMenu/,
    "the partial branch draws real buttons",
  );
  assert.match(
    row,
    /\) : \(\n\s*<span aria-hidden="true" className=\{cn\(ROW_ACTIONS_CLASS, "h-6"\)\}/,
    "and a complete repo still only reserves the gutter",
  );
  assert.ok(row.includes("cachePath={{ repoId: c.repo_id }}"), "reveal");
  assert.ok(row.includes('title: "Delete cached model?"'), "delete");
});

test("the partial repo delete says it removes the repo, because it does", () => {
  // No variant means a whole-repo delete, which may remove other formats too.
  const start = PICKERS.indexOf("const renderDownloadedGgufRow");
  const row = PICKERS.slice(start, PICKERS.indexOf("\n  };", start));
  assert.ok(
    row.includes("and everything downloaded under it from disk"),
    "the copy states the real scope",
  );
  assert.ok(
    !row.includes("This will remove the partly downloaded"),
    "and no longer implies only the torn bytes go",
  );
  assert.match(
    row,
    /await deleteCachedModel\(\n\s*c\.repo_id,\n\s*undefined,/,
    "still a repo-wide delete, matching the Hub row",
  );
});

test("a partial pick carries no load identity", () => {
  // The Audio handoff infers isDownloaded from loadId, so a partial must not send one.
  assert.equal(
    PICKERS.split("loadId: isPartial ? undefined : c.load_id,").length - 1,
    2,
    "both multi-quant cached-row picks drop it when the snapshot is torn",
  );
  assert.equal(
    PICKERS.split("loadId: isDownloaded ? c.load_id : undefined,").length - 1,
    1,
    "the sole-quant pick carries it only when that quant is complete",
  );
  assert.ok(
    PICKERS.includes("loadId: downloaded === true ? loadId : undefined,"),
    "the variant select still drops it the same way",
  );
  const audio = read("../src/features/audio/hooks/use-audio-handoff.ts");
  assert.match(
    audio,
    /isDownloaded: routeSearch\.loadId\n?\s*\? true/,
    "a routed loadId is still read as downloaded",
  );
  // And the route still forwards whatever the pick gives it.
  assert.ok(PICKERS.includes("audioPickSearch(id, { ...meta, task: pickedTask })"));
  assert.match(
    read("../src/features/audio/route-search.ts"),
    /loadId: pick\.loadId \?\? undefined,/,
  );
});

test("configure carries the same rule, because Run replays its metadata", () => {
  // onRun spreads this meta back into a select, so a stray loadId would reach the load path.
  assert.equal(
    PICKERS.split("onConfigure(c.repo_id, selectMeta)").length - 1,
    1,
  );
  const selector = read(
    "../src/features/model-picker/components/model-selector.tsx",
  );
  assert.match(
    selector,
    /onRun=\{\(config, isDiffusion\) =>\s*onSelect\(\s*visibleConfigTarget\.configId \?\? visibleConfigTarget\.id,\s*\{\s*\.\.\.visibleConfigTarget\.meta,/,
  );
});

test("a partial row keeps its buttons on screen instead of hiding them behind hover", () => {
  // A partial cannot load, so its menu is the only affordance and must stay visible.
  assert.ok(
    PICKERS.includes(
      'const ROW_ACTIONS_PINNED_CLASS = cn(ROW_ACTIONS_CLASS, "opacity-100");',
    ),
    "the pinned variant exists and is built from the shared one",
  );
  const gguf = PICKERS.slice(PICKERS.indexOf("const renderDownloadedGgufRow"));
  assert.ok(
    gguf
      .slice(0, gguf.indexOf("\n  };"))
      .includes("<span className={ROW_ACTIONS_PINNED_CLASS}>"),
    "GGUF partial repo row",
  );
  assert.ok(
    PICKERS.includes(
      "isPartial ? ROW_ACTIONS_PINNED_CLASS : ROW_ACTIONS_CLASS",
    ),
    "non-GGUF row pins only when the snapshot is torn",
  );
  assert.match(
    PICKERS,
    /const ROW_ACTIONS_CLASS =\n\s*"[^"]*\bopacity-0\b/,
    "the default gutter still hides",
  );
});

test("a partial never reaches the complete-download lookup", () => {
  // downloadedSet decides isDownloaded on search picks, so partials must stay out of it.
  const start = PICKERS.indexOf("const downloadedSet = useMemo(");
  assert.ok(start > 0, "downloadedSet exists");
  const set = PICKERS.slice(start, PICKERS.indexOf("const partialSet", start));
  assert.match(
    set,
    /\[\.\.\.cachedGguf, \.\.\.cachedModels\]\n\s*\.filter\(\(c\) => !c\.partial\)/,
    "both cached lists are filtered before the ids land in the set",
  );
  assert.ok(PICKERS.includes("isDownloaded: cached !== null,"));
  const lookup = PICKERS.slice(PICKERS.indexOf("const cachedIdFor = useCallback("));
  assert.match(
    lookup.slice(0, lookup.indexOf("[aliasesOf, downloadedSet]")),
    /aliasesOf\(id\)\.find\(\(alias\) => downloadedSet\.has\(alias\.toLowerCase\(\)\)\)/,
    "every alias is checked against the complete-download set",
  );
  const aliases = PICKERS.slice(PICKERS.indexOf("const aliasesOf = useCallback("));
  assert.match(
    aliases.slice(0, aliases.indexOf("[catalog]")),
    /\[id, artifact\.repoId, artifact\.upstreamRepoId\]/,
    "a row, its mirror and the vendor repo are all aliases",
  );
});

test("Hub rows can still tell a partial apart from an absent model", () => {
  assert.ok(PICKERS.includes("const partialSet = useMemo("));
  assert.equal(
    PICKERS.split("partial={isPartialRow(id)}").length - 1,
    3,
    "Recommended, its filtered twin, and the typed search list alike",
  );
  // The mark yields to a complete alias, which is what the click then loads.
  const partialRow = PICKERS.slice(PICKERS.indexOf("const isPartialRow = useCallback("));
  assert.match(
    partialRow.slice(0, partialRow.indexOf("[cachedIdFor, partialSet]")),
    /cachedIdFor\(id\) === null && partialSet\.has\(id\.toLowerCase\(\)\)/,
  );
  // The typed list renders from searchRowIds, not the curated ids.
  const search = PICKERS.slice(PICKERS.indexOf("searchRowIds.map((id) => {"));
  assert.ok(
    search
      .slice(0, search.indexOf("</ModelRow>") + 1 || 4000)
      .includes("partial={isPartialRow(id)}"),
    "the live search row marks one too",
  );
  assert.ok(
    PICKERS.includes(
      "{downloaded && !partial && !loaded ? <DownloadedBadge /> : null}",
    ),
  );
});

test("an id with a complete copy is never also marked partial", () => {
  // The cache keys by repo and format, so one id can be both complete and partial.
  assert.ok(
    PICKERS.includes(
      "partialSetFromRows([...cachedGguf, ...cachedModels], (c) => c.repo_id)",
    ),
    "the picker builds the set through the shared helper",
  );
  const dedupe = read("../src/features/hub/inventory/inventory-dedupe.ts");
  const start = dedupe.indexOf("export function partialSetFromRows");
  assert.ok(start > 0, "the helper exists");
  const body = dedupe.slice(start, dedupe.indexOf("\n}", start));
  assert.match(body, /if \(repoId && !row\.partial\) complete\.add/);
  assert.match(
    body,
    /if \(row\.partial && !complete\.has\(key\)\) partial\.add\(key\)/,
  );
  assert.ok(
    read("../src/features/hub/inventory/index.ts").includes(
      "partialSetFromRows",
    ) && read("../src/features/hub/index.ts").includes("partialSetFromRows"),
  );
  assert.ok(
    dedupe.includes('return `${normalizedRepo}\\0${modelFormat ?? "unknown"}`'),
    "repo plus format, which is what lets one id be both",
  );
});

test("a partial alone does not open the picker on the On Device tab", () => {
  // A host with only a cancelled download has nothing to load, so it must not pick that tab.
  const start = PICKERS.indexOf("export function hasDownloadedModels()");
  assert.ok(start > 0, "hasDownloadedModels exists");
  const body = PICKERS.slice(start, PICKERS.indexOf("\n}", start));
  assert.match(body, /_cachedGgufCache\.some\(\(c\) => !c\.partial\)/);
  assert.match(body, /_cachedModelsCache\.some\(\(c\) => !c\.partial\)/);
  assert.match(body, /_lmStudioCache\.length > 0/);
});

test("a torn quant inside the expander keeps its own menu", () => {
  // A partial still occupies disk, so its quant row needs reveal and delete.
  assert.match(
    PICKERS,
    /\{\(v\.downloaded \|\| v\.partial === true\) &&\n\s*\(allowPin \|\|/,
    "the menu follows disk, not completeness",
  );
  // Pinning an unloadable quant would put a dead row at the top of the list.
  assert.match(
    PICKERS,
    /pin=\{\n\s*allowPin && v\.downloaded\n\s*\? \{/,
    "pin stays for complete quants only",
  );
  assert.ok(PICKERS.includes("{v.downloaded && onConfigure && ("));
});

test("a torn quant is labelled, not left looking undownloaded", () => {
  assert.match(
    PICKERS,
    /\) : v\.partial === true \? \(\n\s*<span className="ml-1\.5 text-ui-9 font-sans font-medium text-amber-700 dark:text-amber-300">\n\s*partial\n/,
  );
});

test("the expander still lists torn quants, which is where resume lives", () => {
  // Resume is per quant, so if this filter drops partials there is no resume path.
  const vis = read(
    "../src/features/model-picker/components/model-selector/variant-visibility.ts",
  );
  assert.ok(
    vis.includes("v.downloaded === true || v.partial === true"),
    "torn quants stay listed on device",
  );
});

test("the stale reason for hiding partials is gone from the picker", () => {
  assert.ok(
    !PICKERS.includes(
      "A partially-downloaded snapshot is not on-device: listing it as loadable errors",
    ),
  );
});
