// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { normalizePerModelConfig, pinSpeculativeMode, storedSpeculativeAuto } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { loadedConfigSignature } = await import(
  "../src/features/model-picker/model-config/config-signature.ts"
);

const { mlxDrafterChoices } = await import("../src/lib/speculative-modes.ts");

test("the drafter picker offers the mode's cached drafters and keeps a saved one visible", () => {
  const cached = [{ repo: "o/m-MTP", kind: "mtp", named: true }, { repo: "o/n-DFlash", kind: "dflash", named: false }];
  const pick = (mode: string, saved: string | null) => mlxDrafterChoices(cached, mode, saved);
  assert.deepEqual(pick("auto", null).map(([repo]: [string, string]) => repo), ["o/m-MTP", "o/n-DFlash"]);
  assert.deepEqual(pick("dflash", "o/n-DFlash"), [["o/n-DFlash", "o/n-DFlash (same architecture)"]]);
  assert.deepEqual(pick("mtp", "/local/drafter")[0], ["/local/drafter", "/local/drafter"]);
});

const specOf = (value: string) =>
  normalizePerModelConfig({ speculativeType: value }).speculativeType;

/** null means follow the global preference, so every disable alias must map to off. */
test("a stored disable alias is an override, not a fall-through to the global default", () => {
  for (const spelling of [
    "off",
    "none",
    "None",
    "NONE",
    "  none  ",
    "disable",
    "Disabled",
    "disabled",
  ]) {
    assert.equal(specOf(spelling), "off", `${spelling} must read as off`);
  }
});

test("the rest of the mapping still resolves as before", () => {
  // An unknown value is the follow-global sentinel the aliases must not become; Auto is kept.
  assert.equal(specOf("auto"), "auto");
  assert.equal(specOf("default"), "auto");
  assert.equal(specOf("bogus"), null);
  assert.equal(specOf("mtp"), "mtp");
  assert.equal(specOf("draft-mtp"), "mtp");
  assert.equal(specOf("ngram-mod"), "ngram");
  assert.equal(specOf("mtp+ngram"), "mtp+ngram");
});

test("MLX modes and an explicit Auto survive storage; the drafter only with a drafter mode", () => {
  assert.deepEqual([specOf("eagle3"), specOf("dflash+ngram")], ["eagle3", null]);
  const drafterOf = (speculativeType: string, specDraftModel: string) =>
    normalizePerModelConfig({ speculativeType, specDraftModel }).specDraftModel;
  assert.deepEqual(
    [drafterOf("auto", " o/d "), drafterOf("ngram", "o/d"), drafterOf("mtp", " ")],
    ["o/d", null, null],
  );
  // GGUF's Auto is "follow the standing preference", which storage spells null.
  assert.equal(storedSpeculativeAuto({ speculativeType: "auto" }, false).speculativeType, null);
  assert.equal(storedSpeculativeAuto({ speculativeType: "auto" }, true).speculativeType, "auto");
  // A drafter typed while the mode is unset pins the shown one, or storage would drop it.
  const pinned = pinSpeculativeMode({ speculativeType: null }, "auto", { specDraftModel: "o/d" });
  assert.equal(normalizePerModelConfig(pinned).specDraftModel, "o/d");
  assert.equal(pinSpeculativeMode({ speculativeType: "mtp" }, "auto", {}).speculativeType, undefined);
  // A new live drafter re-seeds an open editor.
  const auto = normalizePerModelConfig({ speculativeType: "auto" });
  assert.notEqual(loadedConfigSignature(auto), loadedConfigSignature({ ...auto, specDraftModel: "o/d" }));
});
