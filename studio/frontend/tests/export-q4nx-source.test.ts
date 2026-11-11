// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { modelInfo } from "@huggingface/hub";
import {
  localGgufPath,
  packsDirectly,
  Q4NX_INFO_FIELDS,
  q4nxArchFit,
  q4nxBaseModel,
  q4nxIsExpanded,
  q4nxMemoryWarning,
  q4nxNpuMemory,
  q4nxParamsFromName,
  q4nxQuantLabel,
  q4nxToggleExpanded,
  q4nxVariantLookup,
  q4nxWeights,
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
  // A generic repo id with a Qwen2-VL file: the card passes "<repo> <filename>".
  assert.equal(
    q4nxArchFit("qwen2vl", "acme/vision-GGUF Qwen2-VL-2B-Instruct-Q4_0.gguf").status,
    "unsupported",
  );
  assert.equal(
    q4nxArchFit("qwen2vl", "acme/vision-GGUF Qwen2.5-VL-3B-Instruct-Q4_0.gguf").status,
    "supported",
  );
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

test("the file list follows the active source, not a folder left picked on the other tab", () => {
  assert.deepEqual(q4nxVariantLookup("hub", "unsloth/Qwen3-0.6B-GGUF", "/models/qwen"), {
    key: "hub::unsloth/Qwen3-0.6B-GGUF",
    localDirPath: null,
  });
  assert.deepEqual(q4nxVariantLookup("hub", "", "/models/qwen"), {
    key: null,
    localDirPath: null,
  });
  assert.deepEqual(q4nxVariantLookup("local", "unsloth/Qwen3-0.6B-GGUF", "/models/qwen"), {
    key: "local::/models/qwen",
    localDirPath: "/models/qwen",
  });
  assert.deepEqual(q4nxVariantLookup("local", "x/y", null), { key: null, localDirPath: null });
});

test("a typed repo's lookup asks the Hub for its GGUF architecture", async () => {
  let url = "";
  const fakeFetch = (async (input: string | URL | Request) => {
    url = String(input);
    return new Response(
      JSON.stringify({
        _id: "x",
        id: "acme/Mixtral-GGUF",
        tags: ["base_model:mistralai/Mixtral-8x7B-v0.1"],
        gguf: { architecture: "llama", total: 1 },
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    );
  }) as typeof fetch;
  const info = (await modelInfo({
    name: "acme/Mixtral-GGUF",
    fetch: fakeFetch,
    additionalFields: Q4NX_INFO_FIELDS,
  })) as { gguf?: { architecture?: string }; tags?: string[] };
  assert.ok(new URL(url).searchParams.getAll("expand").includes("gguf"), url);
  assert.equal(info.gguf?.architecture, "llama");
  assert.ok(info.tags?.length);
});

test("without Expand quantizations the list is by model: one model's files open at a time", () => {
  let state = { expandAll: false, toggled: [] as readonly string[] };
  assert.equal(q4nxIsExpanded(state, "a/x-GGUF"), false);
  state = q4nxToggleExpanded(state, "a/x-GGUF");
  assert.equal(q4nxIsExpanded(state, "a/x-GGUF"), true);
  // Opening another model closes the first, like the chat picker.
  state = q4nxToggleExpanded(state, "b/y-GGUF");
  assert.equal(q4nxIsExpanded(state, "a/x-GGUF"), false);
  assert.equal(q4nxIsExpanded(state, "b/y-GGUF"), true);
  state = q4nxToggleExpanded(state, "b/y-GGUF");
  assert.equal(q4nxIsExpanded(state, "b/y-GGUF"), false);
});

test("Expand quantizations opens every model, and each closes on its own", () => {
  let state = { expandAll: true, toggled: [] as readonly string[] };
  assert.equal(q4nxIsExpanded(state, "a"), true);
  assert.equal(q4nxIsExpanded(state, "b"), true);
  state = q4nxToggleExpanded(state, "a");
  assert.equal(q4nxIsExpanded(state, "a"), false);
  assert.equal(q4nxIsExpanded(state, "b"), true);
  state = q4nxToggleExpanded(state, "a");
  assert.equal(q4nxIsExpanded(state, "a"), true);
});

const GIB = 1024 ** 3;

test("a file that packs directly needs its own size; any other needs a sibling's or the parameter count's", () => {
  const q4 = { filename: "m-Q4_K_M.gguf", quant: "Q4_K_M", size_bytes: 5 * GIB };
  const q8 = { filename: "m-Q8_0.gguf", quant: "Q8_0", size_bytes: 9 * GIB };
  assert.deepEqual(q4nxWeights(q4, [q8, q4]), { bytes: 5 * GIB, estimated: false });
  // Q8_0 is rounded again to 4-bit, so its own size overstates the result.
  assert.deepEqual(q4nxWeights(q8, [q8, q4]), { bytes: 5 * GIB, estimated: true });
  assert.deepEqual(q4nxWeights(q8, [q8], 10e9), { bytes: 6e9, estimated: true });
  // A repo not listed yet has only the Hub's parameter count.
  assert.deepEqual(q4nxWeights(null, [], 10e9), { bytes: 6e9, estimated: true });
  assert.equal(q4nxWeights(q8, [q8]), null);
  assert.equal(q4nxWeights(null), null);
});

test("a local model with no file that packs directly is sized from the parameter count in its name", () => {
  assert.equal(q4nxParamsFromName("Qwen3.5-9B-GGUF"), 9e9);
  assert.equal(q4nxParamsFromName("unsloth/Qwen3-0.6B-GGUF"), 0.6e9);
  assert.equal(q4nxParamsFromName("Qwen3.6-35B-A3B-GGUF"), 35e9);
  assert.equal(q4nxParamsFromName("orpheus-3b-0.1-ft-GGUF"), 3e9);
  // A quant tag or a version is not a parameter count.
  assert.equal(q4nxParamsFromName("encoder-Q2_K-GGUF"), null);
  assert.equal(q4nxParamsFromName("Z-Image-Turbo-GGUF"), null);
  assert.equal(q4nxParamsFromName("whisper-large-v3-turbo"), null);
});

test("the memory line sets the need, weights plus the 1 GB load margin, against the NPU window", () => {
  const fits = q4nxNpuMemory({ bytes: 5 * GIB, estimated: false }, 16);
  assert.equal(fits?.label, "6.0 of 16.0 GB NPU");
  assert.equal(fits?.fits, true);
  assert.equal(fits?.weightsPct, 31.25);
  assert.equal(fits?.marginPct, 6.25);
  assert.equal(q4nxMemoryWarning(fits, 16), null);

  const over = q4nxNpuMemory({ bytes: 15.5 * GIB, estimated: true }, 16);
  assert.equal(over?.label, "~16.5 of 16.0 GB NPU");
  assert.equal(over?.fits, false);
  assert.ok((over?.weightsPct ?? 0) + (over?.marginPct ?? 0) <= 100);
  assert.equal(
    q4nxMemoryWarning(over, 16),
    "Needs about 16.5 GB of NPU memory; this PC can give the NPU about 16.0 GB.",
  );
});

test("without a window from the server the line shows the need alone and never warns", () => {
  for (const windowGb of [null, undefined, 0]) {
    const memory = q4nxNpuMemory({ bytes: 5 * GIB, estimated: false }, windowGb);
    assert.equal(memory?.label, "6.0 GB NPU");
    assert.equal(memory?.fits, null);
    assert.equal(q4nxMemoryWarning(memory, windowGb), null);
  }
  assert.equal(q4nxNpuMemory(null, 16), null);
});
