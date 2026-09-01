// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A cancelled multi-GB download used to be invisible in the picker: filtered out of the inventory
// before any list saw it, so the one screen that could delete it never showed it. The Hub always
// listed these -- marked, and opening their download rather than a load -- and the picker now
// matches. The rule that makes that safe is that a partial is listed, never loaded.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

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
  // The filter that hid them. Live downloads still go: bytes are moving and the Downloads panel
  // owns that row until they stop.
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
  // toCachedGgufRepo / toCachedModelRepo are the only things the picker sees.
  assert.equal(
    INVENTORY.split("partial: row.partial,").length - 1,
    2,
    "carried onto both cached repo shapes",
  );
});

test("a partial is marked the way the Hub marks one", () => {
  const start = PICKERS.indexOf("function PartialBadge()");
  assert.ok(start > 0, "the picker has a partial mark");
  const badge = PICKERS.slice(start, PICKERS.indexOf("\n}", start));
  assert.match(badge, /size-\[5px\] rounded-full bg-status-warning/);
  assert.match(badge, /aria-label="Partial download"/);
  assert.ok(
    !PICKERS.includes("&mdash;"),
    "no em dash in the tooltip, or anywhere else here",
  );
  assert.ok(
    MODELS_TABLE.includes('aria-label="Partial download"') &&
      MODELS_TABLE.includes("bg-status-warning"),
    "and the Hub still uses that dot, so the two agree",
  );
});

test("complete and partial are alternatives, never both dots on one row", () => {
  // The bytes are there or they are not; two dots would say both.
  assert.equal(
    PICKERS.split("{partial ? <PartialBadge /> : null}").length - 1,
    2,
    "drawn in the aligned and unaligned branches alike",
  );
  assert.equal(
    PICKERS.split(
      "{downloaded && !partial && !loaded ? <DownloadedBadge /> : null}",
    ).length - 1,
    2,
    "and the on-device dot yields to it in both",
  );
});

test("selecting a partial opens its download instead of claiming the weights", () => {
  // isDownloaded is what the load path reads. Hard-coding true on these rows sent a torn snapshot
  // straight to a load that fails on the missing shards -- the reason they were hidden at all.
  assert.ok(
    PICKERS.includes("isDownloaded: !isPartial,"),
    "the pick reports what is actually on disk",
  );
  // Hub reaches the same answer from the same field.
  assert.ok(
    read("../src/features/hub/hub-page.tsx").includes(
      "isDownloaded: !row.partial",
    ),
  );
});

test("listing a partial never makes it auto-loadable", () => {
  // The picker lists them; the background pick must still refuse them. This guard is what makes
  // widening the inventory safe, so it is not free to drift.
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
  // A complete GGUF repo keeps delete on the quant rows inside the expander, so its own row only
  // reserves the gutter. A partial repo has no complete quant to hold those actions, so that
  // reservation left the torn bytes visible and unreachable: no delete, no reveal.
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
  // Reveal and delete are the two the row owes; resume stays per-quant in the expander.
  assert.ok(row.includes("cachePath={{ repoId: c.repo_id }}"), "reveal");
  assert.ok(row.includes('title: "Delete cached model?"'), "delete");
});

test("the partial repo delete says it removes the repo, because it does", () => {
  // No variant is passed, and the backend treats an absent variant as a whole-repo delete. One
  // repo id can also hold a complete copy in another format, which that delete takes with it, so
  // wording it as "the partial download" named a smaller scope than the one that runs.
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
  // The scope claim is only true while the call stays repo-wide, so pin the call too.
  assert.match(
    row,
    /await deleteCachedModel\(\n\s*c\.repo_id,\n\s*undefined,/,
    "still a repo-wide delete, matching the Hub row",
  );
});

test("a partial pick carries no load identity", () => {
  // The Chat-to-Audio route has no isDownloaded field: audio-page.tsx infers it from the
  // forwarded loadId. A loadId names a revision already on disk, so sending one for a torn
  // snapshot told that page the weights were there and skipped the download.
  assert.equal(
    PICKERS.split("loadId: isPartial ? undefined : c.load_id,").length - 1,
    3,
    "every cached-row pick drops it when the snapshot is torn",
  );
  // The GGUF variant select set this rule first; the two must not diverge.
  assert.ok(
    PICKERS.includes("loadId: downloaded === true ? loadId : undefined,"),
    "the variant select still drops it the same way",
  );
  // What made the omission load bearing, in the page that reads it.
  const audio = read("../src/features/audio/audio-page.tsx");
  assert.match(
    audio,
    /isDownloaded: routeSearch\.loadId\n?\s*\? true/,
    "a routed loadId is still read as downloaded",
  );
  // And the route still forwards whatever the pick gives it.
  assert.ok(PICKERS.includes("loadId: meta.loadId ?? undefined,"));
});

test("configure carries the same rule, because Run replays its metadata", () => {
  // The settings page keys itself off the repo id, but onRun spreads this meta straight back
  // into a select, so a loadId left on it would reach the load path the row click already drops.
  // The sole-quant row shares one meta between its click and its gear, so it is covered once.
  assert.equal(
    PICKERS.split("onConfigure(c.repo_id, selectMeta)").length - 1,
    1,
  );
  // What makes it load bearing: Run reuses the stored meta verbatim.
  const selector = read(
    "../src/features/model-picker/components/model-selector.tsx",
  );
  assert.match(
    selector,
    /onRun=\{\(config, isDiffusion\) =>\n\s*onSelect\(visibleConfigTarget\.id, \{\n\s*\.\.\.visibleConfigTarget\.meta,/,
  );
});

test("a partial row keeps its buttons on screen instead of hiding them behind hover", () => {
  // Every other row hides the gutter until hover because the row itself is the action: click it
  // and the model loads. A partial cannot be loaded, so the menu is its ONLY affordance -- hidden,
  // the row reads as a stalled download with no controls at all.
  assert.ok(
    PICKERS.includes(
      'const ROW_ACTIONS_PINNED_CLASS = cn(ROW_ACTIONS_CLASS, "opacity-100");',
    ),
    "the pinned variant exists and is built from the shared one",
  );
  // Both partial rows use it: the GGUF repo row and the cached non-GGUF row.
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
  // The base class must keep hiding itself, or every row grows a permanent button strip.
  assert.match(
    PICKERS,
    /const ROW_ACTIONS_CLASS =\n\s*"[^"]*\bopacity-0\b/,
    "the default gutter still hides",
  );
});

test("a partial never reaches the complete-download lookup", () => {
  // downloadedSet decides isDownloaded on a search pick, and isDownloaded: true is what skips
  // download staging. Listing partials in cachedGguf / cachedModels put their ids in this set,
  // so a partial reached from Recommended or Hub search (which use handleModelClick, not the
  // On Device renderer) went straight to the loader with an incomplete snapshot.
  const start = PICKERS.indexOf("const downloadedSet = useMemo(");
  assert.ok(start > 0, "downloadedSet exists");
  const set = PICKERS.slice(start, PICKERS.indexOf("const partialSet", start));
  assert.match(
    set,
    /\[\.\.\.cachedGguf, \.\.\.cachedModels\]\n\s*\.filter\(\(c\) => !c\.partial\)/,
    "both cached lists are filtered before the ids land in the set",
  );
  // The search pick still reads that set, which is what makes the guard above load bearing.
  assert.ok(
    PICKERS.includes("isDownloaded: downloadedSet.has(id.toLowerCase()),"),
  );
});

test("Hub rows can still tell a partial apart from an absent model", () => {
  // Splitting partials out of downloadedSet would otherwise leave them looking like something
  // never fetched, so the mark comes from its own set rather than from the on-disk one.
  assert.ok(PICKERS.includes("const partialSet = useMemo("));
  assert.equal(
    PICKERS.split("partial={partialSet.has(id.toLowerCase())}").length - 1,
    3,
    "Recommended, its filtered twin, and the typed search list alike",
  );
  // The typed list is the one that was missed: it renders from searchRowIds, not from the
  // curated ids, so a partial reached by typing its name showed nothing at all.
  const search = PICKERS.slice(PICKERS.indexOf("searchRowIds.map((id) => {"));
  assert.ok(
    search
      .slice(0, search.indexOf("</ModelRow>") + 1 || 4000)
      .includes("partial={partialSet.has(id.toLowerCase())}"),
    "the live search row marks one too",
  );
  // A partial must never take the green on-disk dot; ModelRow already yields one to the other.
  assert.ok(
    PICKERS.includes(
      "{downloaded && !partial && !loaded ? <DownloadedBadge /> : null}",
    ),
  );
});

test("an id with a complete copy is never also marked partial", () => {
  // The cache keys by repo AND format, so one id survives as two rows: a complete GGUF copy and a
  // torn safetensors one. Marking on `partial` alone put that id in both sets, and since the click
  // reads downloadedSet it loaded fine while the row showed a warning dot and offered a resume.
  assert.ok(
    PICKERS.includes(
      "partialSetFromRows([...cachedGguf, ...cachedModels], (c) => c.repo_id)",
    ),
    "the picker builds the set through the shared helper",
  );
  // That helper is the whole guarantee, so its exclusion rule is asserted here too.
  const dedupe = read("../src/features/hub/inventory/inventory-dedupe.ts");
  const start = dedupe.indexOf("export function partialSetFromRows");
  assert.ok(start > 0, "the helper exists");
  const body = dedupe.slice(start, dedupe.indexOf("\n}", start));
  assert.match(body, /if \(repoId && !row\.partial\) complete\.add/);
  assert.match(
    body,
    /if \(row\.partial && !complete\.has\(key\)\) partial\.add\(key\)/,
  );
  // Both barrels have to carry it or the import above cannot resolve.
  assert.ok(
    read("../src/features/hub/inventory/index.ts").includes(
      "partialSetFromRows",
    ) && read("../src/features/hub/index.ts").includes("partialSetFromRows"),
  );
  // Two rows for one id is the case being guarded, so the key that allows it is pinned.
  assert.ok(
    dedupe.includes('return `${normalizedRepo}\\0${modelFormat ?? "unknown"}`'),
    "repo plus format, which is what lets one id be both",
  );
});

test("a partial alone does not open the picker on the On Device tab", () => {
  // hasDownloadedModels picks the first run tab. The cached lists carry partials so they can be
  // seen and removed, but a machine whose only cached row is a cancelled download has nothing to
  // load, and that tab would open on one unusable row.
  const start = PICKERS.indexOf("export function hasDownloadedModels()");
  assert.ok(start > 0, "hasDownloadedModels exists");
  const body = PICKERS.slice(start, PICKERS.indexOf("\n}", start));
  assert.match(body, /_cachedGgufCache\.some\(\(c\) => !c\.partial\)/);
  assert.match(body, /_cachedModelsCache\.some\(\(c\) => !c\.partial\)/);
  // Local sources have no partial concept, so they stay a plain emptiness check.
  assert.match(body, /_lmStudioCache\.length > 0/);
});

test("a torn quant inside the expander keeps its own menu", () => {
  // The expander lists torn quants, but the menu was gated on v.downloaded, so the one row that
  // holds bytes you might want back had no reveal and no delete. A partial still occupies disk.
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
  // Settings still need a loadable file, so the gear is untouched.
  assert.ok(PICKERS.includes("{v.downloaded && onConfigure && ("));
});

test("a torn quant is labelled, not left looking undownloaded", () => {
  // Without a label the row reads as a quant you have not fetched yet, which is the opposite of
  // what it is. Local paths already said "incomplete"; cached repos said nothing at all.
  assert.match(
    PICKERS,
    /\) : v\.partial === true \? \(\n\s*<span className="ml-1\.5 text-ui-9 font-sans font-medium text-amber-700 dark:text-amber-300">\n\s*partial\n/,
  );
});

test("the expander still lists torn quants, which is where resume lives", () => {
  // The repo row deletes the whole partial; resuming targets one quant, so it belongs to the
  // variant rows. If this filter ever drops partials there is no resume path left anywhere.
  const vis = read(
    "../src/features/model-picker/components/model-selector/variant-visibility.ts",
  );
  assert.ok(
    vis.includes("v.downloaded === true || v.partial === true"),
    "torn quants stay listed on device",
  );
});

test("the stale reason for hiding partials is gone from the picker", () => {
  // It justified a filter that no longer exists; left in place it reads as the current rule.
  assert.ok(
    !PICKERS.includes(
      "A partially-downloaded snapshot is not on-device: listing it as loadable errors",
    ),
  );
});
