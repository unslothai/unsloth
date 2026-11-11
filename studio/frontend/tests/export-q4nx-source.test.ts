// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  localGgufPath,
  packsDirectly,
  q4nxArchFit,
  q4nxBaseModel,
  sortQ4nxVariants,
} from "../src/features/export/lib/q4nx-source.ts";

test("only Q4_0, Q4_1 and Q4_K_M pack directly, matching the backend's SOURCE_QUANTS", () => {
  for (const quant of ["Q4_0", "q4_1", "Q4_K_M"]) {
    assert.equal(packsDirectly(quant), true, quant);
  }
  for (const quant of ["Q8_0", "BF16", "Q4_K_S", "UD-Q4_K_XL", "IQ4_XS"]) {
    assert.equal(packsDirectly(quant), false, quant);
  }
});

test("the file list keeps every quant but puts the directly packed ones first", () => {
  const sorted = sortQ4nxVariants([
    { filename: "m-Q8_0.gguf", quant: "Q8_0" },
    { filename: "m-Q4_K_M.gguf", quant: "Q4_K_M" },
    { filename: "m-BF16.gguf", quant: "BF16" },
    { filename: "m-Q4_0.gguf", quant: "Q4_0" },
    { filename: "m-Q4_1.gguf", quant: "Q4_1" },
  ]);
  assert.deepEqual(
    sorted.map((v) => v.quant),
    ["Q4_0", "Q4_1", "Q4_K_M", "Q8_0", "BF16"],
  );
});

test("the original model comes from the card's base_model, else the repo without -GGUF", () => {
  assert.equal(
    q4nxBaseModel("unsloth/Qwen3-0.6B-GGUF", "Qwen/Qwen3-0.6B"),
    "Qwen/Qwen3-0.6B",
  );
  assert.equal(
    q4nxBaseModel("unsloth/Qwen3-0.6B-GGUF", null),
    "unsloth/Qwen3-0.6B",
  );
  assert.equal(
    q4nxBaseModel("unsloth/Qwen3-0.6B-gguf", "  "),
    "unsloth/Qwen3-0.6B",
  );
  assert.equal(q4nxBaseModel("", null), "");
});

test("architectures the pinned converter maps are supported, with size caveats", () => {
  assert.deepEqual(q4nxArchFit("qwen3"), { status: "supported", note: null });
  assert.deepEqual(q4nxArchFit("LLAMA"), { status: "supported", note: null });
  assert.equal(q4nxArchFit("gpt-oss").status, "supported");
  const qwen35 = q4nxArchFit("qwen35");
  assert.equal(qwen35.status, "supported");
  assert.match(qwen35.note ?? "", /0\.8B, 2B, 4B, 9B/);
});

test("a known unsupported architecture is refused, an unknown one only warned about", () => {
  const moe = q4nxArchFit("qwen3moe");
  assert.equal(moe.status, "unsupported");
  assert.match(moe.note, /qwen3moe/);
  for (const arch of [null, undefined, ""]) {
    const fit = q4nxArchFit(arch);
    assert.equal(fit.status, "unknown");
    assert.match(fit.note, /may not be supported/);
  }
});

test("a local .gguf row is the file; a folder row needs the picked variant", () => {
  assert.equal(
    localGgufPath("C:/models/m.Q4_1.gguf", null),
    "C:/models/m.Q4_1.gguf",
  );
  assert.equal(localGgufPath("/models/qwen", null), null);
  assert.equal(
    localGgufPath("/models/qwen/", { filename: "Q4_0/m-Q4_0.gguf" }),
    "/models/qwen/Q4_0/m-Q4_0.gguf",
  );
  assert.equal(
    localGgufPath("C:\\models\\qwen", { filename: "m-Q4_0.gguf" }),
    "C:\\models\\qwen\\m-Q4_0.gguf",
  );
});
