// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  PRECISION_REFUSAL_TITLE,
  type ResolvedControl,
  isPrecisionRefusal,
  isResolvedHonored,
  resolvedBadge,
  resolvedSeedKey,
  resolvedSelectValue,
} from "../src/lib/resolved-precision.ts";

const QUANT_OPTIONS = ["auto", "none", "int8", "fp8", "nvfp4", "mxfp8"] as const;
const toQuantOption = (v: string) =>
  QUANT_OPTIONS.find((o) => o === v || (o === "none" && v === "off")) ?? null;

test("a declined explicit precision renders a warning badge naming both sides", () => {
  // The bug: the badge only rendered for source === "auto", so an explicit FP8 the backend
  // declined showed nothing while the Precision dropdown kept advertising FP8.
  const resolved: ResolvedControl = {
    value: "off",
    requested: "fp8",
    source: "explicit",
    status: "fell_back",
    reason: "the dense bf16 transformer does not fit resident",
  };
  const badge = resolvedBadge("transformer_quant", resolved);
  assert.ok(badge);
  assert.equal(badge.label, "FP8 → OFF");
  assert.equal(badge.tone, "warn");
  assert.match(badge.tooltip, /You requested FP8/);
  assert.match(badge.tooltip, /does not fit resident/);
  assert.equal(isResolvedHonored(resolved), false);
});

test("an honored explicit request renders no badge", () => {
  const resolved: ResolvedControl = {
    value: "fp8",
    requested: "fp8",
    source: "explicit",
    status: "applied",
    reason: "engaged on the dense fast path",
  };
  assert.equal(resolvedBadge("transformer_quant", resolved), null);
  assert.equal(isResolvedHonored(resolved), true);
});

test("a backend decision still renders the neutral Auto badge", () => {
  const resolved: ResolvedControl = {
    value: "off",
    requested: null,
    source: "auto",
    status: "applied",
    reason: "not engaged (GGUF transformer loaded)",
  };
  const badge = resolvedBadge("transformer_quant", resolved);
  assert.deepEqual(badge, {
    label: "Auto: OFF",
    tone: "auto",
    tooltip: "not engaged (GGUF transformer loaded)",
  });
});

test("a control answered in another vocabulary is not reported as a fallback", () => {
  // memory_mode is REQUESTED as a mode and ENGAGED as an offload policy, so a raw string compare
  // would call every honored request a fallback. The backend's status field decides.
  const resolved: ResolvedControl = {
    value: "sequential",
    requested: "low_vram",
    source: "explicit",
    status: "applied",
    reason: "planned from measured free VRAM",
  };
  assert.equal(isResolvedHonored(resolved), true);
  assert.equal(resolvedBadge("memory_mode", resolved), null);
});

test("an older backend without requested/status keeps today's behaviour", () => {
  // No status field: an explicit control renders nothing, an auto one renders its Auto badge.
  assert.equal(
    resolvedBadge("transformer_quant", { value: "int8", source: "explicit", reason: "requested" }),
    null,
  );
  const auto = resolvedBadge("speed_mode", {
    value: "eager",
    source: "auto",
    reason: "per-kind default",
  });
  assert.equal(auto?.label, "Auto: EAGER");
  assert.equal(auto?.tone, "auto");
});

test("an older backend still flags a mismatch it can see", () => {
  // requested present but status absent: fall back to comparing, which is right for the precision
  // controls (they answer in the vocabulary they are asked in).
  const resolved: ResolvedControl = {
    value: "off",
    requested: "fp8",
    source: "explicit",
    reason: "",
  };
  assert.equal(isResolvedHonored(resolved), false);
  assert.equal(resolvedBadge("transformer_quant", resolved)?.tone, "warn");
});

test("a status this build has never heard of is not read as a decline", () => {
  // Forwards compat the OTHER way: `status` is typed wider than the backend's union precisely so a
  // newer backend can add a fourth value. Reading everything except "applied" as a failure threw
  // that away -- an honored FP8 came back as a red "FP8 → FP8" badge, and memory_mode (asked
  // "low_vram", answered "sequential") as a "LOW_VRAM → SEQUENTIAL" fallback that never happened.
  for (const status of ["partially_applied", "downgraded", "ok"]) {
    const quant: ResolvedControl = {
      value: "fp8",
      requested: "fp8",
      source: "explicit",
      status,
      reason: "engaged on the dense fast path",
    };
    assert.equal(isResolvedHonored(quant), true, status);
    assert.equal(resolvedBadge("transformer_quant", quant), null, status);
    // The select still shows the ask, not the engaged value it would have snapped to.
    assert.equal(resolvedSelectValue(quant, toQuantOption), "fp8", status);

    const memory: ResolvedControl = {
      value: "sequential",
      requested: "low_vram",
      source: "explicit",
      status,
      reason: "planned from measured free VRAM",
    };
    assert.equal(resolvedBadge("memory_mode", memory), null, status);
  }
  // The two known declines keep warning.
  for (const status of ["fell_back", "unsupported"]) {
    const resolved: ResolvedControl = {
      value: "off",
      requested: "fp8",
      source: "explicit",
      status,
      reason: "the host cannot run it",
    };
    assert.equal(isResolvedHonored(resolved), false, status);
    assert.equal(resolvedBadge("transformer_quant", resolved)?.tone, "warn", status);
  }
});

test("every off spelling counts as an honored off request", () => {
  for (const [requested, value] of [
    ["none", "off"],
    ["off", null],
    ["", "off"],
  ] as Array<[string, string | null]>) {
    assert.equal(
      isResolvedHonored({ value, requested, source: "explicit", reason: "" }),
      true,
      `${requested} -> ${value}`,
    );
  }
});

test("cpu_offload compares as a boolean and formats as On/Off", () => {
  assert.equal(
    isResolvedHonored({ value: true, requested: true, source: "explicit", reason: "" }),
    true,
  );
  const declined = resolvedBadge("cpu_offload", {
    value: false,
    requested: true,
    source: "explicit",
    status: "fell_back",
    reason: "everything fits on the GPU",
  });
  assert.equal(declined?.label, "On → Off");
});

test("the Precision select seeds from the loaded build", () => {
  // Auto stays auto (the badge names what it resolved to).
  assert.equal(
    resolvedSelectValue(
      { value: "fp8", requested: null, source: "auto", status: "applied", reason: "" },
      toQuantOption,
    ),
    "auto",
  );
  // An honored request re-selects itself.
  assert.equal(
    resolvedSelectValue(
      { value: "int8", requested: "int8", source: "explicit", status: "applied", reason: "" },
      toQuantOption,
    ),
    "int8",
  );
  // A DECLINED request snaps to what actually engaged, so the dropdown stops advertising it.
  assert.equal(
    resolvedSelectValue(
      { value: "off", requested: "fp8", source: "explicit", status: "fell_back", reason: "" },
      toQuantOption,
    ),
    "none",
  );
  // Nothing resolved: keep whatever the user has typed.
  assert.equal(resolvedSelectValue(null, toQuantOption), null);
});

test("the Attention select maps the dispatcher's own name back to its option", () => {
  const toAttentionOption = (v: string) =>
    (["auto", "native", "cudnn", "flash3", "sage"] as const).find(
      (o) => o === v || `_native_${o}` === v,
    ) ?? null;
  assert.equal(
    resolvedSelectValue(
      {
        value: "_native_cudnn",
        requested: "cudnn",
        source: "explicit",
        status: "applied",
        reason: "",
      },
      toAttentionOption,
    ),
    "cudnn",
  );
});

test("the reseed key ignores the entries the backend rewrites mid-session", () => {
  // The reseed effect used to key on JSON.stringify(resolved). The backend mutates that record at
  // GENERATION time -- speed_mode and attention_backend when the deferred compile profile engages
  // on the 3rd image, transformer_cache when the step-cache threshold flips -- so the key changed
  // with no reload behind it and the effect re-ran, overwriting a Precision the user had picked
  // but not yet loaded.
  const atLoad: Record<string, ResolvedControl> = {
    transformer_quant: { value: "off", requested: null, source: "auto", status: "applied", reason: "" },
    memory_mode: { value: "none", requested: null, source: "auto", status: "applied", reason: "" },
    attention_backend: { value: "native", requested: null, source: "auto", status: "applied", reason: "" },
    speed_mode: { value: "deferred", requested: null, source: "auto", status: "applied", reason: "" },
    transformer_cache: { value: "off", requested: null, source: "auto", status: "applied", reason: "" },
  };
  const key = resolvedSeedKey(atLoad);

  // Generation 3: the compile profile engages and the attention upgrade lands (diffusion.py).
  const afterThirdImage: Record<string, ResolvedControl> = {
    ...atLoad,
    speed_mode: { ...atLoad.speed_mode, value: "default", reason: "auto: compiled on the 3rd image" },
    attention_backend: { ...atLoad.attention_backend, value: "_native_cudnn", reason: "cuDNN upgrade" },
  };
  assert.equal(resolvedSeedKey(afterThirdImage), key, "a mid-session compile must not re-seed");
  assert.notEqual(
    JSON.stringify(afterThirdImage),
    JSON.stringify(atLoad),
    "the record really did change -- serializing it is what re-fired the effect",
  );

  // A step-cache toggle (both pages) is the same story.
  const afterCacheToggle: Record<string, ResolvedControl> = {
    ...atLoad,
    transformer_cache: { ...atLoad.transformer_cache, value: "fbcache", reason: "auto: 40 steps" },
  };
  assert.equal(resolvedSeedKey(afterCacheToggle), key, "a cache toggle must not re-seed");

  // A real reload still re-seeds: the request and the engaged value both move.
  const afterReapply: Record<string, ResolvedControl> = {
    ...atLoad,
    transformer_quant: {
      value: "off",
      requested: "fp8",
      source: "explicit",
      status: "fell_back",
      reason: "the dense bf16 transformer does not fit resident",
    },
  };
  assert.notEqual(resolvedSeedKey(afterReapply), key, "a declined Reapply must re-seed");

  // So does a load that honors a new memory mode, or a new attention request.
  assert.notEqual(
    resolvedSeedKey({
      ...atLoad,
      memory_mode: { value: "sequential", requested: "low_vram", source: "explicit", status: "applied", reason: "" },
    }),
    key,
  );
  assert.notEqual(
    resolvedSeedKey({
      ...atLoad,
      attention_backend: { value: "_native_cudnn", requested: "cudnn", source: "explicit", status: "applied", reason: "" },
    }),
    key,
  );
});

test("the reseed key tolerates an empty or absent record", () => {
  assert.equal(resolvedSeedKey(null), null);
  assert.equal(resolvedSeedKey(undefined), null);
  // An older backend sends the record without requested/status; the key is still a stable string.
  assert.equal(typeof resolvedSeedKey({}), "string");
  const older = resolvedSeedKey({
    transformer_quant: { value: "int8", source: "explicit", reason: "requested" },
  });
  assert.equal(typeof older, "string");
  assert.ok(!/undefined|NaN/.test(older ?? ""), older ?? "");
});

test("a precision refusal is recognised so it can be shown as an actionable toast", () => {
  const refusal =
    "transformer_quant='fp8' could not be used: this device cannot run a dense torchao quant " +
    "(it needs a CUDA GPU in bf16). Choose Auto to let the backend pick the fastest precision " +
    "this host can run, or Off to run the checkpoint as-is.";
  assert.equal(isPrecisionRefusal(refusal), true);
  assert.equal(isPrecisionRefusal("text_encoder_quant='int8' could not be used: nope."), true);
  assert.equal(isPrecisionRefusal("A diffusion load is already in progress."), false);
  assert.equal(PRECISION_REFUSAL_TITLE, "Requested precision is not available");
});
