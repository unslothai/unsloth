// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Models Studio only discovered -- an Ollama tag, an LM Studio folder, anything under a scan
// folder -- had no delete at all: the card offered one for hf_cache rows and nothing else, so
// years-old models found on disk could only be removed outside the app. These cover the call
// that removes them and the routing that picks it, because an Ollama row sent to the cached
// endpoint would be rejected as a malformed repo id rather than deleted.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

// The inventory API reaches authFetch through the auth barrel, which re-exports login-page.tsx.
register("./helpers/settings-api-resolver.mjs", import.meta.url);
installLocalStorageFake();

type Call = { url: string; init?: RequestInit };
const calls: Call[] = [];
let nextResponse: () => Response = () =>
  new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } });

globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
  calls.push({ url: String(input), init });
  return nextResponse();
}) as typeof fetch;

const { deleteLocalModel, fetchLocalDeleteImpact } = await import(
  "../src/features/hub/inventory/api.ts"
);

const OLLAMA_REF =
  "ollama-manifest:%2Fhome%2Fu%2F.ollama%2Fmodels%2Fmanifests%2Fregistry.ollama.ai%2Flibrary%2Fllama3%2F8b";

test("a local delete names the row by its load id, never as a repo id", async () => {
  calls.length = 0;
  await deleteLocalModel(OLLAMA_REF, "ollama");

  assert.equal(calls.length, 1);
  const [call] = calls;
  assert.match(call.url, /\/api\/hub\/delete-local$/);
  assert.equal(call.init?.method, "DELETE");
  const body = JSON.parse(String(call.init?.body));
  // The manifest reference, not the blob path: it is the only handle that names this tag
  // rather than bytes several tags may share.
  assert.equal(body.load_id, OLLAMA_REF);
  assert.equal(body.source, "ollama");
  assert.ok(
    !("repo_id" in body),
    "the cached-model endpoint would reject a manifest reference as a bad repo id",
  );
});

test("a refused delete surfaces as an error rather than a phantom success", async () => {
  calls.length = 0;
  nextResponse = () =>
    new Response(JSON.stringify({ detail: "Unload the model before deleting" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });

  await assert.rejects(
    () => deleteLocalModel("/home/u/.lmstudio/models/pub/model", "lmstudio"),
    /Unload the model before deleting/,
  );

  nextResponse = () =>
    new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } });
});

test("an unavailable preview reads as no preview, so the confirm dialog still opens", async () => {
  nextResponse = () => new Response("nope", { status: 500 });
  assert.equal(await fetchLocalDeleteImpact("/some/model", "custom"), null);

  globalThis.fetch = (async () => {
    throw new Error("offline");
  }) as typeof fetch;
  assert.equal(await fetchLocalDeleteImpact("/some/model", "custom"), null);
});

test("the preview posts, because a load id is a path-shaped value", async () => {
  calls.length = 0;
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), init });
    return new Response(
      JSON.stringify({
        load_id: OLLAMA_REF,
        source: "ollama",
        display_name: "llama3:8b",
        reclaimed_bytes: 10,
        retained_bytes: 0,
        retained_for: [],
        removed_paths: [],
        blocked_by: [],
        notes: [],
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  }) as typeof fetch;

  const impact = await fetchLocalDeleteImpact(OLLAMA_REF, "ollama");

  assert.equal(impact?.display_name, "llama3:8b");
  assert.match(calls[0].url, /\/api\/hub\/local-delete-impact$/);
  assert.equal(calls[0].init?.method, "POST");
});

// --- Routing -----------------------------------------------------------------------------

const cardSource = readFileSync(
  new URL("../src/features/hub/catalog/local-on-device-card.tsx", import.meta.url),
  "utf8",
);

test("the delete button is offered for every On Device row, not just cached ones", () => {
  // The old gate was `source === "hf_cache" && !!repoId`, which is exactly what hid the
  // button on the Ollama and scan-folder rows this feature exists for.
  assert.match(
    cardSource,
    /const canDelete =\s*\(isCachedRepo \? !!repoId : !!modelId\) && !isActive && !isLoading;/,
  );
});

test("each row reaches the endpoint that can actually delete it", () => {
  assert.match(cardSource, /if \(isCachedRepo\) \{[\s\S]*await deleteCachedModel\(/);
  assert.match(cardSource, /await deleteLocalModel\(modelId, source\);/);
  assert.match(cardSource, /const isCachedRepo = source === "hf_cache";/);
});

test("a blocked local preview disables Delete instead of buying the user a 400", () => {
  assert.match(
    cardSource,
    /blocked=\{\s*\(\(isCachedRepo \? deleteImpact : localDeleteImpact\)\?\.blocked_by\s*\.length \?\? 0\) > 0\s*\}/,
  );
});

test("the local dialog says the delete is permanent, because nothing can re-download it", () => {
  assert.match(cardSource, /Unsloth\s*\n?\s*did not download this model/);
  assert.match(cardSource, /<LocalDeleteImpactSummary impact=\{localDeleteImpact\} \/>/);
});

const rowsSource = readFileSync(
  new URL("../src/features/hub/catalog/models-catalog-rows.tsx", import.meta.url),
  "utf8",
);
const rowMenuSource = readFileSync(
  new URL(
    "../src/features/model-picker/components/model-selector/model-row-menu.tsx",
    import.meta.url,
  ),
  "utf8",
);

test("the list row menu offers a delete for discovered models, not only cached repos", () => {
  // The menu's only delete took a repo id, which a discovered model does not have -- so the
  // Ollama and scan-folder rows this feature is about had no menu delete at all.
  assert.match(
    rowsSource,
    /row\.kind === "local" && row\.source !== "hf_cache" && row\.loadId/,
  );
  assert.match(
    rowsSource,
    /const canDelete =\s*cacheDeletableRepoId !== null \|\| localDeletable !== null;/,
  );
  assert.match(
    rowsSource,
    /await deleteLocalModel\(localDeletable\.loadId, localDeletable\.source\)/,
  );
});

test("a local dataset row is not routed through the model delete", () => {
  // The local endpoint refuses a path that does not hold a model, so offering it here would
  // only turn the menu action into a 400.
  assert.match(rowsSource, /const localDeletable =\s*!isDataset &&/);
});

test("the row menu previews whichever kind of delete the row asked for", () => {
  assert.match(rowMenuSource, /localImpact\?: \{ loadId: string; source\?: string \| null \}/);
  assert.match(
    rowMenuSource,
    /useLocalDeleteImpact\(\s*deleteOpen && Boolean\(del\?\.localImpact\),/,
  );
  assert.match(
    rowMenuSource,
    /blocked=\{\s*\(\(deleteImpact \?\? localDeleteImpact\)\?\.blocked_by\.length \?\? 0\) > 0\s*\}/,
  );
});
