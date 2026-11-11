// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  localGgufPath,
  packsDirectly,
  q4nxArchFit,
  q4nxBaseModel,
  q4nxQuantLabel,
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

test("architectures the pinned converter accepts are supported, with size caveats", () => {
  assert.deepEqual(q4nxArchFit("qwen3"), { status: "supported", note: null });
  assert.deepEqual(q4nxArchFit("LLAMA"), { status: "supported", note: null });
  assert.equal(q4nxArchFit("gpt-oss").status, "supported");
  const qwen35 = q4nxArchFit("qwen35");
  assert.equal(qwen35.status, "supported");
  assert.match(qwen35.note ?? "", /0\.8B, 2B, 4B, 9B/);
});

test("qwen2vl is Qwen2.5-VL only; a Qwen2-VL name is refused", () => {
  // The converter would send Qwen2-VL to the dense qwen2 converter by basename.
  const vl25 = q4nxArchFit("qwen2vl", "unsloth/Qwen2.5-VL-7B-Instruct-GGUF");
  assert.equal(vl25.status, "supported");
  assert.match(vl25.note ?? "", /Only Qwen2\.5-VL/);
  assert.equal(q4nxArchFit("qwen2vl").status, "supported");
  for (const name of [
    "unsloth/Qwen2-VL-7B-Instruct-GGUF",
    "bartowski/qwen2vl-2b-GGUF",
    "C:\\models\\Qwen2_VL-2B.Q4_0.gguf",
  ]) {
    const fit = q4nxArchFit("qwen2vl", name);
    assert.equal(fit.status, "unsupported", name);
    assert.match(fit.note, /Qwen2-VL/);
  }
  // Local files have no architecture before conversion, so the name alone decides.
  for (const name of [
    "/models/Qwen2-VL-2B-Instruct-Q4_0.gguf",
    "unsloth/Qwen2-VL-7B-Instruct-GGUF /hf/models--unsloth--Qwen2-VL-7B-Instruct-GGUF/snapshots/abc/x.gguf",
  ]) {
    assert.equal(q4nxArchFit(null, name).status, "unsupported", name);
  }
  assert.equal(
    q4nxArchFit(null, "/models/Qwen2.5-VL-7B-Instruct-Q4_0.gguf").status,
    "unknown",
  );
  // Only qwen2vl GGUFs are checked by name; a dense Qwen2 repo is untouched.
  assert.equal(q4nxArchFit("qwen2", "Qwen/Qwen2-VL-ish").status, "supported");
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

test("an HF cache repo folder lists path-qualified variants that still resolve and rank", () => {
  // Backend answer for a models--org--repo row: filename and quant carry snapshots/<sha>/.
  const snap = "snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b";
  const variants = [
    { filename: `${snap}/Qwen3-0.6B-UD-Q4_K_XL.gguf`, quant: `${snap}/Qwen3-0.6B-UD-Q4_K_XL` },
    { filename: `${snap}/Qwen3-0.6B-Q4_1.gguf`, quant: `${snap}/Qwen3-0.6B-Q4_1` },
  ];
  assert.equal(packsDirectly(variants[1].quant), true);
  assert.equal(packsDirectly(variants[0].quant), false);
  assert.equal(packsDirectly("IQ4_0"), false);
  assert.equal(q4nxQuantLabel(variants[1].quant), "Qwen3-0.6B-Q4_1");
  const [first] = sortQ4nxVariants(variants);
  assert.equal(first.filename, variants[1].filename);
  assert.equal(
    localGgufPath("L:\\hub\\models--unsloth--Qwen3-0.6B-GGUF", first),
    `L:\\hub\\models--unsloth--Qwen3-0.6B-GGUF\\${snap}/Qwen3-0.6B-Q4_1.gguf`,
  );
  // A snapshot row (non-active cache) and a custom folder list plain filenames.
  assert.equal(
    localGgufPath(`/cache/models--o--r/${snap}`, { filename: "m-Q4_0.gguf" }),
    `/cache/models--o--r/${snap}/m-Q4_0.gguf`,
  );
  assert.equal(
    localGgufPath("/data/custom/qwen", { filename: "sub/m-Q4_0.gguf" }),
    "/data/custom/qwen/sub/m-Q4_0.gguf",
  );
});
