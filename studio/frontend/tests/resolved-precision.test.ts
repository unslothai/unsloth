// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  DENSE_QUANT_KINDS,
  PRECISION_REFUSAL_TITLE,
  type ResolvedControl,
  isDenseQuantKind,
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
  // memory_mode is requested as a mode but engaged as an offload policy, so compare via status.
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
  // status is typed wider so newer backends can add values; only known declines are failures.
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
  assert.equal(
    resolvedSelectValue(
      { value: "fp8", requested: null, source: "auto", status: "applied", reason: "" },
      toQuantOption,
    ),
    "auto",
  );
  assert.equal(
    resolvedSelectValue(
      { value: "int8", requested: "int8", source: "explicit", status: "applied", reason: "" },
      toQuantOption,
    ),
    "int8",
  );
  assert.equal(
    resolvedSelectValue(
      { value: "off", requested: "fp8", source: "explicit", status: "fell_back", reason: "" },
      toQuantOption,
    ),
    "none",
  );
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
  // The backend mutates resolved at generation time, so keying on it re-seeded and lost edits.
  const atLoad: Record<string, ResolvedControl> = {
    transformer_quant: { value: "off", requested: null, source: "auto", status: "applied", reason: "" },
    memory_mode: { value: "none", requested: null, source: "auto", status: "applied", reason: "" },
    attention_backend: { value: "native", requested: null, source: "auto", status: "applied", reason: "" },
    family_override: { value: "auto", requested: null, source: "auto", status: "applied", reason: "" },
    speed_mode: { value: "deferred", requested: null, source: "auto", status: "applied", reason: "" },
    transformer_cache: { value: "off", requested: null, source: "auto", status: "applied", reason: "" },
  };
  const key = resolvedSeedKey(atLoad);

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

  const afterCacheToggle: Record<string, ResolvedControl> = {
    ...atLoad,
    transformer_cache: { ...atLoad.transformer_cache, value: "fbcache", reason: "auto: 40 steps" },
  };
  assert.equal(resolvedSeedKey(afterCacheToggle), key, "a cache toggle must not re-seed");

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
  assert.notEqual(
    resolvedSeedKey({
      ...atLoad,
      family_override: { value: "z-image", requested: "z-image", source: "explicit", status: "applied", reason: "" },
    }),
    key,
    "a replacement load with a different family must re-seed",
  );
});

test("the reseed key tolerates an empty or absent record", () => {
  assert.equal(resolvedSeedKey(null), null);
  assert.equal(resolvedSeedKey(undefined), null);
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

test("the dense-quant kinds are the two the backend quantises", () => {
  assert.deepEqual([...DENSE_QUANT_KINDS], ["gguf", "pipeline"]);
  assert.equal(isDenseQuantKind("gguf"), true);
  assert.equal(isDenseQuantKind("pipeline"), true);
  assert.equal(isDenseQuantKind("single_file"), false);
  assert.equal(isDenseQuantKind(" Pipeline "), true);
  assert.equal(isDenseQuantKind(null), false);
  assert.equal(isDenseQuantKind(undefined), false);
  assert.equal(isDenseQuantKind(""), false);
});

const ENCODER_OPTIONS = ["auto", "fp8", "fp8_dynamic", "int8", "nvfp4"] as const;
const toEncoderOption = (v: string) =>
  ENCODER_OPTIONS.find((o) => o === v || (o === "auto" && (v === "none" || v === "off"))) ?? null;

test("the text encoder select follows what the loaded build actually ran", () => {
  assert.equal(
    resolvedSelectValue({ value: "off", source: "auto", reason: "" }, toEncoderOption),
    "auto",
  );
  assert.equal(
    resolvedSelectValue({ value: "none", source: "auto", reason: "" }, toEncoderOption),
    "auto",
  );
  assert.equal(
    resolvedSelectValue(
      { value: "fp8_dynamic", requested: "fp8_dynamic", source: "explicit", status: "applied", reason: "" },
      toEncoderOption,
    ),
    "fp8_dynamic",
  );
  assert.equal(
    resolvedSelectValue(
      { value: "fp8", requested: "int8", source: "explicit", status: "fell_back", reason: "int8 needs resident weights" },
      toEncoderOption,
    ),
    "fp8",
  );
  assert.equal(
    resolvedSelectValue(
      { value: "off", requested: "nvfp4", source: "explicit", status: "fell_back", reason: "no Blackwell GPU" },
      toEncoderOption,
    ),
    "auto",
  );
});

test("the reseed key moves when the text encoder build changes, and only then", () => {
  const atLoad = {
    transformer_quant: { value: "fp8", requested: "fp8", source: "explicit", status: "applied", reason: "" },
    text_encoder_quant: { value: "fp8", requested: "fp8", source: "explicit", status: "applied", reason: "" },
    memory_mode: { value: "balanced", source: "auto", reason: "" },
    attention_backend: { value: "native", source: "auto", reason: "" },
  } satisfies Record<string, ResolvedControl>;
  const key = resolvedSeedKey(atLoad);

  assert.equal(
    resolvedSeedKey({
      ...atLoad,
      text_encoder_quant: { ...atLoad.text_encoder_quant, reason: "re-measured after the first image" },
    }),
    key,
    "a reason rewrite must not re-seed",
  );
  assert.notEqual(
    resolvedSeedKey({
      ...atLoad,
      text_encoder_quant: { value: "off", requested: "fp8", source: "explicit", status: "fell_back", reason: "declined" },
    }),
    key,
    "a declined encoder must re-seed",
  );
  assert.notEqual(
    resolvedSeedKey({
      ...atLoad,
      text_encoder_quant: { value: "int8", requested: "int8", source: "explicit", status: "applied", reason: "" },
    }),
    key,
  );
});

// Mirrors images-page.tsx options; keeps "Dense pinned" distinct from "Default".
const TE_OPTIONS = ["auto", "none", "fp8", "fp8_dynamic", "int8", "nvfp4"] as const;
const toTeOption = (v: string) =>
  TE_OPTIONS.find((o) => o === v || (o === "none" && v === "off")) ?? null;

test("an unset text-encoder request reseeds as Default even when a scheme engaged", () => {
  // source: "auto" keeps a family-default fp8 from being pinned into the select.
  const autoFp8: ResolvedControl = {
    value: "fp8",
    requested: null,
    source: "auto",
    status: "applied",
    reason: "selected automatically for qwen-image-2.1 (no text_encoder_quant requested)",
  };
  assert.equal(resolvedSelectValue(autoFp8, toTeOption), "auto");
});

test("a pinned dense text encoder reseeds as Dense, not Default", () => {
  const pinnedDense: ResolvedControl = {
    value: "off",
    requested: "none",
    source: "explicit",
    status: "applied",
    reason: "dense bf16 text encoder(s) loaded",
  };
  assert.equal(resolvedSelectValue(pinnedDense, toTeOption), "none");

  const declined: ResolvedControl = {
    value: "off",
    requested: "nvfp4",
    source: "explicit",
    status: "unsupported",
    reason: "nvfp4 needs Blackwell sm_100+",
  };
  assert.equal(resolvedSelectValue(declined, toTeOption), "none");
});
