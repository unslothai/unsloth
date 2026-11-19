// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// An INT8 / FP8 image build takes LoRAs only when it loads. A selection it did not bake makes Generate reload the model
// with it (the Reapply path) instead of sending a run the backend refuses; a refusal that still arrives offers the
// same reload.

import assert from "node:assert/strict";
import test from "node:test";

import {
  LORA_RELOAD_REFUSAL,
  bakesLorasAtLoad,
  isLoraReloadRefusal,
  loraReloadReason,
} from "../src/features/images/lib/lora-reload.ts";
import { RELOAD_AND_GENERATE_LABEL, generateGate } from "../src/features/images/lib/generate-gate.ts";
import { readSrc, readText } from "./helpers/kit.ts";

const int8 = (baked: Array<{ id: string; weight: number }> = []) => ({
  loaded: true,
  family: "moss-image",
  transformer_quant: "int8",
  supports_lora: true,
  baked_lora_specs: baked,
});

test("a LoRA on an INT8 build that baked none needs a reload", () => {
  const reason = loraReloadReason(int8(), [{ id: "tarot", weight: 1 }]);
  assert.match(reason ?? "", /INT8 build takes LoRAs only when it loads/);
  assert.match(reason ?? "", /Generate reloads the model/);
  assert.match(loraReloadReason({ ...int8(), transformer_quant: "fp8" }, [{ id: "a", weight: 1 }]) ?? "", /FP8/);
});

test("the baked set, a subset of it, new weights or no LoRA run without a reload", () => {
  const baked = int8([
    { id: "tarot", weight: 1 },
    { id: "ink", weight: 0.5 },
  ]);
  assert.equal(loraReloadReason(baked, [{ id: "tarot", weight: 1 }, { id: "ink", weight: 0.5 }]), null);
  assert.equal(loraReloadReason(baked, [{ id: "tarot", weight: 0.3 }]), null);
  assert.equal(loraReloadReason(baked, []), null);
  assert.equal(loraReloadReason(baked, [{ id: "tarot", weight: 0 }, { id: "  ", weight: 1 }]), null);
  assert.notEqual(loraReloadReason(baked, [{ id: "tarot", weight: 1 }, { id: "other", weight: 1 }]), null);
  // One baked copy covers one pick.
  assert.notEqual(loraReloadReason(int8([{ id: "tarot", weight: 1 }]), [{ id: "tarot", weight: 1 }, { id: "tarot", weight: 1 }]), null);
});

test("dense and GGUF builds apply LoRAs per request", () => {
  for (const quant of [null, "", "none", "nf4"]) {
    const s = { ...int8(), transformer_quant: quant };
    assert.equal(bakesLorasAtLoad(s), false, String(quant));
    assert.equal(loraReloadReason(s, [{ id: "tarot", weight: 1 }]), null);
  }
  assert.equal(loraReloadReason({ ...int8(), supports_lora: false }, [{ id: "a", weight: 1 }]), null);
  assert.equal(loraReloadReason({ ...int8(), loaded: false }, [{ id: "a", weight: 1 }]), null);
});

test("MiniMax-H3-Image fixes the selection and weights at every precision", () => {
  const h3 = { loaded: true, family: "minimax-h3-image", transformer_quant: null, supports_lora: true, baked_lora_specs: [{ id: "a", weight: 1 }] };
  assert.equal(loraReloadReason(h3, [{ id: "a", weight: 1 }]), null);
  assert.notEqual(loraReloadReason(h3, [{ id: "a", weight: 0.5 }]), null);
  assert.notEqual(loraReloadReason(h3, []), null);
});

test("the refusal header matches the backend constant", () => {
  assert.equal(isLoraReloadRefusal(400, "lora-reload"), true);
  assert.equal(isLoraReloadRefusal(400, "memory-estimate"), false);
  assert.equal(isLoraReloadRefusal(409, "lora-reload"), false);
  const backend = readText("../../backend/core/inference/diffusion_lora.py");
  assert.match(backend, new RegExp(`LORA_RELOAD_REFUSAL = "${LORA_RELOAD_REFUSAL}"`));
});

test("Generate offers the reload rather than a refused run", () => {
  const gate = generateGate({
    busy: null,
    presetsHydrated: true,
    loaded: true,
    canRecallModel: true,
    external: false,
    switching: false,
    loraReload: loraReloadReason(int8(), [{ id: "tarot", weight: 1 }]),
    queueWaiting: true,
    canQueue: true,
  });
  assert.equal(gate.disabled, false);
  assert.equal(gate.action, "reload");
  assert.equal(gate.label, RELOAD_AND_GENERATE_LABEL);
  assert.match(gate.notice ?? "", /takes LoRAs only when it loads/);
});

test("the Images page routes Generate, Queue, Compare and refusals through the reload", () => {
  const page = readSrc("features/images/images-page.tsx");
  // Generate: reload with the selection baked, then generate.
  assert.match(page, /if \(status\?\.loaded && loraReload\) \{\s*reloadWithLoras\(true\);/);
  // The Reapply load keeps the selection even for a resident model another client loaded.
  assert.match(page, /currentLoadAdvanced\(l\.repoId, true, keepLoras\)/);
  assert.match(page, /if \(!reapplyLoad\(true\)\)/);
  // Every other caller of handleGenerate is stopped before the request.
  assert.match(page, /if \(!external && loraReload\) \{\s*toast\.error\(LORA_RELOAD_TITLE/);
  // A refusal that still arrives offers the reload.
  assert.match(page, /err instanceof LoraReloadRefusalError/);
  const api = readSrc("features/images/api.ts");
  assert.match(api, /isLoraReloadRefusal\(response\.status, response\.headers\.get\(MEMORY_REFUSAL_HEADER\)\)/);
});

test("Reload and generate survives a staged engine reload and binds to the load when it starts", () => {
  const page = readSrc("features/images/images-page.tsx");
  // Planning / companion downloads bump no load sequence, so the request waits as "next" ...
  const reload = page.slice(page.indexOf("const reloadWithLoras = useCallback("));
  assert.match(reload.slice(0, reload.indexOf("reapplyLoad(true)")), /load: "next"/);
  // ... the pending effect keeps it instead of clearing it on a sequence mismatch ...
  assert.match(page, /pending\.load !== "next" && pending\.load !== loadSeq\.current/);
  // ... handleLoad binds it to the load it starts ...
  assert.match(
    page,
    /const startLoad = \+\+loadSeq\.current;\n\s*if \(pendingRecalledGeneration\.current\?\.load === "next"\) pendingRecalledGeneration\.current\.load = startLoad;/,
  );
  // ... and a stage that never loads drops it.
  const reapply = page.slice(page.indexOf("const reapplyLoad = useCallback("));
  assert.match(reapply.slice(0, reapply.indexOf("const handleReapply")), /!started && pendingRecalledGeneration\.current\?\.load === "next"/);
});
