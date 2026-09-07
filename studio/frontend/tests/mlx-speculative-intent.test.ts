// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The requested tuple and what a runtime may overwrite. Pin and method are one setting.

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();

const {
  MLX_DRAFT_BLOCK_SIZE_RANGE,
  MLX_DRAFT_TOKENS_RANGE,
  isSelectableMlxDraftCandidate,
  mlxDraftRowCheckpoint,
  isUnavailableMlxSpeculativeMode,
  mlxDraftSelection,
  selectExternalMlxDraftCandidate,
  selectableExternalMlxDraftCandidates,
  mlxSpeculativeLoadFields,
  selectMlxSpeculativeCandidate,
  normalizeMlxDraftBlockSize,
  normalizeMlxDraftModel,
  normalizeMlxSpeculativeMode,
} = await import("../src/lib/speculative-modes.ts");
type MlxSpeculativeCandidate = import("../src/lib/speculative-modes.ts").MlxSpeculativeCandidate;

const { mlxRuntimeStateFrom, reconcileMlxSpeculativeStatus } = await import(
  "../src/features/chat/lib/mlx-runtime-state.ts"
);
const { toApiOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);
const { DEFAULT_PER_MODEL_CONFIG } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

test("an absent mode takes the caller's default, an unusable one is refused", () => {
  assert.equal(normalizeMlxSpeculativeMode(undefined, "auto"), "auto");
  assert.equal(normalizeMlxSpeculativeMode(" MTP ", "auto"), "mtp");
  assert.equal(normalizeMlxSpeculativeMode("ngram", "auto"), "off");
});

test("a pinned drafter survives only under the method that pinned it", () => {
  for (const method of ["mtp", "dflash", "eagle3"] as const) {
    assert.equal(normalizeMlxDraftModel(" org/d ", method), "org/d");
  }
  assert.equal(normalizeMlxDraftModel("org/d", "auto"), null);
  assert.equal(normalizeMlxDraftModel("org/d", "off"), null);
});

test("a draft block size is clamped to what the backend accepts, and dropped unless a method owns it", () => {
  assert.equal(normalizeMlxDraftBlockSize(1, "mtp"), 2);
  assert.equal(normalizeMlxDraftBlockSize(99, "mtp"), 16);
  assert.equal(normalizeMlxDraftBlockSize(4.4, "mtp"), 4);
  assert.equal(normalizeMlxDraftBlockSize(8, "off"), null);
  // Auto hides the depth control but any depth sent alongside it wins, overriding Auto unseen.
  assert.equal(normalizeMlxDraftBlockSize(8, "auto"), null);
  // The backend's block counts the verified token, so the control states one less at both ends.
  assert.deepEqual(
    MLX_DRAFT_TOKENS_RANGE,
    MLX_DRAFT_BLOCK_SIZE_RANGE.map((size) => size - 1),
  );
});

test("a server override travels only for a pin the server can act on", () => {
  const override = (config: Record<string, unknown>) =>
    toApiOverride({ ...DEFAULT_PER_MODEL_CONFIG, ...config });
  assert.deepEqual(
    override({
      mlxSpeculativeMode: "mtp",
      mlxDraftModel: "org/d",
      mlxDraftBlockSize: 4,
    }),
    {
      mlx_speculative_mode: "mtp",
      mlx_draft_model: "org/d",
      mlx_draft_block_size: 4,
    },
  );
  // Auto re-resolves per load and Off is the server default, so neither is sent.
  assert.deepEqual(
    override({ mlxSpeculativeMode: "auto", mlxDraftModel: "org/d" }),
    {},
  );
  assert.deepEqual(
    override({ mlxSpeculativeMode: "off", mlxDraftModel: "org/d" }),
    {},
  );
  assert.deepEqual(override({ mlxSpeculativeMode: "mtp" }), {});
});

test("a load sends the tuple only for a model MLX will serve", () => {
  const pinned = {
    mlxSpeculativeMode: "mtp",
    mlxDraftModel: "org/d",
    mlxDraftBlockSize: 6,
  };
  assert.deepEqual(mlxSpeculativeLoadFields(pinned, true), {
    mlx_speculative_mode: "mtp",
    mlx_draft_model: "org/d",
    mlx_draft_block_size: 6,
  });
  assert.deepEqual(mlxSpeculativeLoadFields(pinned, false), {
    mlx_speculative_mode: "off",
    mlx_draft_model: null,
    mlx_draft_block_size: null,
  });
  // A model with nothing remembered asks for Auto, not for Off: Off is a choice.
  assert.equal(mlxSpeculativeLoadFields({}, true).mlx_speculative_mode, "auto");
});

const MLX_STATUS = {
  is_mlx: true,
  mlx_speculative_mode: "auto",
  mlx_draft_model: "org/d",
  mlx_draft_block_size: 4,
  mlx_speculative_reason: null,
} as const;

test("a verdict is read only from an MLX response, and never over the request", () => {
  const off = mlxRuntimeStateFrom({ ...MLX_STATUS, is_mlx: false });
  assert.equal(off.loadedMlxSpeculativeMode, null);
  assert.equal(off.mlxSpeculativeReason, null);
  // The request itself is dormant off MLX, not wrong, so it is left alone.
  assert.equal("mlxSpeculativeMode" in off, false);
  const on = mlxRuntimeStateFrom(MLX_STATUS);
  // Auto is what was asked for and stays that, and pins no drafter beside it.
  assert.equal(on.loadedMlxSpeculativeMode, "auto");
  assert.equal(on.loadedMlxDraftModel, null);
});

// Auto drops both the pinned drafter and the depth, whose control it hides.
const RESIDENT = {
  mlxSpeculativeMode: "auto" as const,
  mlxDraftModel: null,
  mlxDraftBlockSize: null,
  loadedMlxSpeculativeMode: "auto" as const,
  loadedMlxDraftModel: null,
  loadedMlxDraftBlockSize: null,
  mlxSpeculativeReason: null,
};

test("a status refresh does not overwrite an edit that has not been sent yet", () => {
  // Otherwise the refresh overwrites the staged values and the reload comparison finds nothing.
  const staged = { ...RESIDENT, mlxSpeculativeMode: "dflash" as const };
  const fields = reconcileMlxSpeculativeStatus(staged, MLX_STATUS, false);
  assert.equal(fields.mlxSpeculativeMode, undefined);
  // The runtime's own half is still adopted: only the request is the user's to hold.
  assert.equal(fields.loadedMlxSpeculativeMode, "auto");
  // A model that has reported nothing holds no edit, so its first status is adopted.
  const fresh = reconcileMlxSpeculativeStatus(
    { ...RESIDENT, loadedMlxSpeculativeMode: null },
    MLX_STATUS,
    false,
  );
  assert.equal(fresh.mlxSpeculativeMode, "auto");
  // Hydrating a different model adopts that model's request instead of holding the edit.
  const hydrated = reconcileMlxSpeculativeStatus(staged, MLX_STATUS, true);
  assert.equal(hydrated.mlxSpeculativeMode, "auto");
});

function candidate(over: Record<string, unknown> = {}) {
  return {
    method: "mtp",
    repo_id: "org/d",
    label: "d",
    source: "cached",
    approximate_size_bytes: 1,
    estimated_memory_bytes: 1,
    materialization_bytes: 0,
    downloaded: true,
    compatible: true,
    runtime_supported: true,
    integration_ready: true,
    loadable: true,
    reason: null,
    recommended: false,
    ...over,
  } as MlxSpeculativeCandidate;
}

test("Auto names the drafter the backend resolved, rather than ranking the rows again", () => {
  // Auto's size and requantization rules are invisible in a row, so ranking here misnames.
  const picked = (cs: MlxSpeculativeCandidate[], resolved: string | null) =>
    selectMlxSpeculativeCandidate(cs, "auto", null, resolved)?.repo_id;
  const own = candidate({ source: "builtin", repo_id: "builtin://mtp" });
  const dflash = candidate({ method: "dflash", repo_id: "org/f" });
  const rows = [dflash, own, candidate({ repo_id: "a/also-mtp" })];

  assert.equal(picked(rows, "org/f"), "org/f");
  assert.equal(picked(rows, "builtin://mtp"), "builtin://mtp");
  // Not the row local ranking would lead with, so a reinstated one cannot pass by agreeing.
  assert.equal(picked(rows, "a/also-mtp"), "a/also-mtp");
  // Resolved to no drafter at all, which a list full of loadable rows cannot tell.
  assert.equal(picked(rows, null), undefined);
  // Named a drafter this list does not carry, so there is no row to show for it.
  assert.equal(picked(rows, "org/absent"), undefined);
  // The same repository however it is spelled.
  assert.equal(picked(rows, "  ORG/F  "), "org/f");
  assert.equal(selectMlxSpeculativeCandidate([own], "off", null, "org/f"), null);
});

test("Off selects nothing, even from a row that claims to be an Off drafter", () => {
  // The types deny this row, but the body is cast rather than validated.
  assert.equal(
    selectMlxSpeculativeCandidate([candidate({ method: "off" })], "off", null),
    null,
  );
});

test("a method one download away is offered, since the picker offers the download", () => {
  // Widened with the picker that makes it actionable.
  const undownloaded = candidate({
    loadable: false,
    downloaded: false,
    reason: "checkpoint_not_downloaded",
  });
  assert.equal(isSelectableMlxDraftCandidate(undownloaded), true);
  assert.equal(
    selectMlxSpeculativeCandidate([undownloaded], "mtp", null)?.repo_id,
    "org/d",
  );
  // A comparison the target cannot answer until it is on disk is not a refusal, so the
  // drafter stays offered: the load settles it. These are the backend's unproven reasons.
  for (const deferred of [
    "tokenizer_contract_unavailable",
    "target_config_unavailable",
    "target_weights_unmeasured",
  ]) {
    assert.equal(
      isSelectableMlxDraftCandidate(
        candidate({ loadable: false, downloaded: true, reason: deferred }),
      ),
      true,
      deferred,
    );
  }
  // What downloading cannot fix stays out, and so does the head: nothing to fetch for it.
  for (const broken of [
    { compatible: false },
    { runtime_supported: false },
    { integration_ready: false },
    { reason: "insufficient_unified_memory" },
    { source: "builtin" },
  ]) {
    assert.equal(
      isSelectableMlxDraftCandidate(candidate({ ...undownloaded, ...broken })),
      false,
      JSON.stringify(broken),
    );
  }
  // Unpinned, the method takes the first usable candidate, not the first listed.
  assert.equal(
    selectMlxSpeculativeCandidate(
      [
        candidate({ repo_id: "org/unusable", loadable: false, compatible: false }),
        candidate({ repo_id: "org/usable" }),
      ],
      "mtp",
      null,
    )?.repo_id,
    "org/usable",
  );
  // An explicit pin names one repository and takes no substitute.
  const cs = [candidate(), candidate({ repo_id: "org/other" })];
  assert.equal(selectMlxSpeculativeCandidate(cs, "mtp", "ORG/Other")?.repo_id, "org/other");
  assert.equal(selectMlxSpeculativeCandidate(cs, "mtp", "org/absent"), null);
});

test("the drafter list offers companions only, ready ones first", () => {
  const ready = candidate({ repo_id: "org/ready" });
  const download = candidate({
    repo_id: "a/needs-download",
    loadable: false,
    downloaded: false,
    reason: "checkpoint_not_downloaded",
  });
  const own = candidate({ source: "builtin", repo_id: "builtin://mtp" });
  const listed = selectableExternalMlxDraftCandidates([download, own, ready]);
  // The head is the target: nothing to choose, and what can run now leads.
  assert.deepEqual(
    listed.map((c) => c.repo_id),
    ["org/ready", "a/needs-download"],
  );
  // Not offered at all when it could not run even after downloading.
  assert.equal(
    selectableExternalMlxDraftCandidates([
      { ...download, compatible: false },
    ]).length,
    0,
  );
  // Picking the method takes the same one the list leads with. The backend lists its
  // recommendations first, so choosing the first merely selectable row would pin a download
  // over a checkpoint already on disk.
  assert.equal(
    selectMlxSpeculativeCandidate(
      [{ ...download, recommended: true }, ready],
      "mtp",
      null,
    )?.repo_id,
    "org/ready",
  );
});

test("a pin names a companion, never the target's own head", () => {
  const own = candidate({ source: "builtin", repo_id: "builtin://mtp" });
  const external = candidate({ repo_id: "org/d" });
  assert.equal(
    selectExternalMlxDraftCandidate([own, external], " ORG/D ")?.repo_id,
    "org/d",
  );
  // Naming the head selects nothing here: it is not a companion the picker can offer.
  assert.equal(selectExternalMlxDraftCandidate([own], "builtin://mtp"), null);
  assert.equal(selectExternalMlxDraftCandidate([external], null), null);
  // Choosing a checkpoint chooses the method it implements; they are one setting.
  assert.deepEqual(mlxDraftSelection(candidate({ method: "dflash" })), {
    mlxSpeculativeMode: "dflash",
    mlxDraftModel: "org/d",
  });
});

test("only a named method with no drafter is ruled out", () => {
  const cs = [candidate()];
  assert.equal(isUnavailableMlxSpeculativeMode(cs, "mtp"), false);
  assert.equal(isUnavailableMlxSpeculativeMode(cs, "dflash"), true);
  // Never Auto or Off: Off runs nothing, and Auto is the request the backend resolves itself.
  for (const listing of [cs, []]) {
    assert.equal(isUnavailableMlxSpeculativeMode(listing, "auto"), false);
    assert.equal(isUnavailableMlxSpeculativeMode(listing, "off"), false);
  }
});

test("the draft row names the pin, and offers to fetch only what would run", () => {
  const pinned = candidate({ repo_id: "org/pinned" });
  const resolved = candidate({ repo_id: "org/resolved" });
  assert.equal(mlxDraftRowCheckpoint(pinned, resolved).shown?.repo_id, "org/pinned");
  // Not the first companion listed: those span every method, so it is usually untouched here.
  assert.equal(mlxDraftRowCheckpoint(null, resolved).shown?.repo_id, "org/resolved");
  assert.equal(mlxDraftRowCheckpoint(null, null).shown, null);
  const fetchable = candidate({ repo_id: "org/resolved", downloaded: false });
  assert.equal(mlxDraftRowCheckpoint(null, fetchable).fetchable, true);
  assert.equal(mlxDraftRowCheckpoint(fetchable, fetchable).fetchable, true);
  // A pin the request would not draft with is named but not offered: fetching changes nothing.
  const foreign = candidate({ repo_id: "org/foreign", downloaded: false });
  assert.equal(mlxDraftRowCheckpoint(foreign, resolved).fetchable, false);
  assert.equal(mlxDraftRowCheckpoint(foreign, null).fetchable, false);
  assert.equal(mlxDraftRowCheckpoint(null, resolved).fetchable, false);
});


test("the control reads back the request, not the runtime's own pick", async () => {
  const { mlxRuntimeStateFrom } = await import(
    "../src/features/chat/lib/mlx-runtime-state.ts"
  );
  // The requested half of the response populates the control. Reading the effective half back
  // would rewrite an Auto request as the pick it produced, and the user could never leave Auto;
  // the response type withholds those fields from this function so it cannot.
  const state = mlxRuntimeStateFrom({
    is_mlx: true,
    mlx_speculative_mode_requested: "mtp",
    mlx_draft_model_requested: "org/asked",
    mlx_draft_block_size_requested: 2,
  });
  assert.equal(state.mlxSpeculativeMode, "mtp");
  assert.equal(state.mlxDraftModel, "org/asked");
  assert.equal(state.mlxDraftBlockSize, 2);
});
