// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";

import { isDenseQuantKind, isNativeEngineStatus } from "../src/lib/resolved-precision.ts";
import { readSrc } from "./helpers/kit.ts";

function precisionControlShown(status: {
  loaded?: boolean;
  model_kind?: string | null;
  dtype?: string | null;
  engine?: string | null;
} | null): boolean {
  return (
    !status?.loaded
    || (isDenseQuantKind(status.model_kind) && !isNativeEngineStatus(status))
  );
}

const DIFFUSERS_GGUF = {
  loaded: true,
  model_kind: "gguf",
  dtype: "bfloat16",
  engine: "diffusers",
};
// SdCppDiffusionBackend.status(): engine "sd_cpp" says the torchao path is unavailable.
const SD_CPP_NATIVE = {
  loaded: true,
  model_kind: "gguf",
  dtype: "gguf",
  engine: "sd_cpp",
};

test("the Precision control is offered for a diffusers GGUF and withheld from native sd.cpp", () => {
  assert.equal(precisionControlShown(null), true, "nothing loaded: the control is free");
  assert.equal(
    precisionControlShown(DIFFUSERS_GGUF),
    true,
    "a diffusers GGUF load is exactly what transformer_quant is for",
  );
  assert.equal(
    precisionControlShown({ ...DIFFUSERS_GGUF, model_kind: "pipeline" }),
    true,
    "a pipeline pick is quantised in place, so Precision applies to it",
  );
  // sd.cpp reports transformer_quant null and no `resolved` map, so an offered control snaps back.
  assert.equal(
    precisionControlShown(SD_CPP_NATIVE),
    false,
    "the native sd.cpp engine runs no torchao path, so Precision must stay withheld",
  );
});

test("the Images page gates Precision on the engine, not on model_kind alone", () => {
  // The component's own source, so the helper above cannot drift away from it.
  const source = readSrc("features/images/images-page.tsx");
  const tree = ts.createSourceFile(
    "images-page.tsx",
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  void tree;
  // Pinned on the native-engine exclusion, not the kind test beside it.
  const gate = source.includes("&& !isNativeEngineStatus(status)");
  assert.ok(
    gate,
    "the Precision gate must exclude the native engine; a bare "
      + 'status.model_kind === "gguf" offers FP8/INT8/NVFP4 to sd.cpp, which ignores them',
  );
});
