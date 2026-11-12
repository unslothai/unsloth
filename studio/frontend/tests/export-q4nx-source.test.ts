// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { modelInfo } from "@huggingface/hub";
import {
  localGgufPath,
  packsDirectly,
  q4nxPacking,
  Q4NX_INFO_FIELDS,
  q4nxArchFit,
  q4nxBaseModel,
  q4nxIsExpanded,
  q4nxMemoryWarning,
  q4nxNpuMemory,
  q4nxParamsFromName,
  q4nxQuantLabel,
  q4nxQuantName,
  q4nxSummaryRows,
  q4nxToggleExpanded,
  q4nxVariantLookup,
  q4nxWeights,
  q4nxConvertibleFiles,
  q4nxIsRefusal,
  sortQ4nxVariants,
} from "../src/features/export/lib/q4nx-source.ts";

test("Q4_0 and Q4_1 pack directly everywhere; Q4_K_M only where the q4k converter runs", () => {
  for (const quant of ["Q4_0", "q4_1"]) {
    assert.equal(packsDirectly(quant), true, quant);
    assert.equal(packsDirectly(quant, "llama"), true, quant);
  }
  // q4nx_converter_pins.json sends only qwen35 and qwen35moe to the converter that unpacks Q4_K.
  assert.equal(q4nxPacking("Q4_K_M", "qwen35"), "direct");
  assert.equal(q4nxPacking("Q4_K_M", "QWEN35MOE"), "direct");
  assert.equal(q4nxPacking("Q4_K_M", "llama"), "requantized");
  assert.equal(q4nxPacking("Q4_K_M", "gemma3"), "requantized");
  // A local file's architecture is read during conversion.
  assert.equal(q4nxPacking("Qwen3.5-4B-Q4_K_M", null), "by-architecture");
  assert.equal(packsDirectly("Q4_K_M"), false);
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
  // On an architecture the q4nx converter handles, Q4_K_M ranks with the re-quantized files.
  assert.deepEqual(
    sortQ4nxVariants(
      [
        { filename: "m-Q4_K_M.gguf", quant: "Q4_K_M" },
        { filename: "m-Q8_0.gguf", quant: "Q8_0" },
        { filename: "m-Q4_1.gguf", quant: "Q4_1" },
      ],
      "llama",
    ).map((v) => v.quant),
    ["Q4_1", "Q4_K_M", "Q8_0"],
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
  assert.deepEqual(q4nxWeights(q4, [q8, q4], null, "qwen35"), { bytes: 5 * GIB, estimated: false });
  // Q8_0 is rounded again to 4-bit, so its own size overstates the result.
  assert.deepEqual(q4nxWeights(q8, [q8, q4], null, "qwen35"), { bytes: 5 * GIB, estimated: true });
  // The q4nx converter rounds Q4_K_M again, so on llama it is no exact size either.
  assert.equal(q4nxWeights(q4, [q8, q4], null, "llama"), null);
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

test("a size the converter has no model for is refused by name, before any download", () => {
  // Qwen3.8 27B reports the qwen35 architecture, which converts only at 0.8B, 2B, 4B and 9B.
  const big = q4nxArchFit("qwen35", "unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_0.gguf");
  assert.equal(big.status, "unsupported");
  assert.equal(big.status === "unsupported" && big.sized, true);
  assert.match(big.note ?? "", /27B model\. Only Qwen3\.5 0\.8B, 2B, 4B, 9B and Omni convert\./);
  // A size on the list needs no caveat.
  for (const name of ["unsloth/Qwen3.5-4B-GGUF", "unsloth/Qwen3.5-0.8B-GGUF", "acme/Qwen3.5-Omni-GGUF"]) {
    assert.deepEqual(q4nxArchFit("qwen35", name), { status: "supported", note: null }, name);
  }
  // No size in the name: the caveat stays a note.
  const unnamed = q4nxArchFit("qwen35", "acme/my-finetune-GGUF");
  assert.equal(unnamed.status, "supported");
  assert.match(unnamed.note ?? "", /0\.8B, 2B, 4B, 9B/);
  assert.deepEqual(q4nxArchFit("qwen35moe", "unsloth/Qwen3.6-35B-A3B-GGUF"), {
    status: "supported",
    note: null,
  });
  assert.equal(q4nxArchFit("qwen35moe", "acme/Qwen3.5-122B-A10B-GGUF").status, "unsupported");
  // An architecture with no size list is never refused by size.
  assert.equal(q4nxArchFit("llama", "meta/Llama-3.3-70B-GGUF").status, "supported");
  // An unsupported architecture is not a size refusal.
  const arch = q4nxArchFit("qwen3moe", "acme/x-30B-GGUF");
  assert.equal(arch.status === "unsupported" && arch.sized, undefined);
});

test("the quant is named alone, out of a variant key or a file stem", () => {
  assert.equal(q4nxQuantName("Q4_K_M"), "Q4_K_M");
  assert.equal(q4nxQuantName("Qwen3-0.6B-Q4_1"), "Q4_1");
  assert.equal(q4nxQuantName("snapshots/abc/m-UD-Q4_K_XL"), "UD-Q4_K_XL");
  assert.equal(q4nxQuantName("Qwen3.5-4B-BF16"), "BF16");
  assert.equal(q4nxQuantName("model"), "model");
});

test("the export summary describes the conversion: source, file and quant, output, size, memory fit and convertibility", () => {
  const rows = (input: Parameters<typeof q4nxSummaryRows>[0]) =>
    Object.fromEntries(q4nxSummaryRows(input).map((row) => [row.label, row.value]));
  const picked = {
    model: "unsloth/Qwen3.5-4B-GGUF",
    file: "Qwen3.5-4B-Q4_K_M.gguf",
    quant: "Q4_K_M",
    packing: q4nxPacking("Q4_K_M", "qwen35"),
    baseModel: "Qwen/Qwen3.5-4B",
    saveDirectory: "q4nx",
    size: "2.7 GB",
    memory: { fits: true, label: "3.6 of 30.8 GB NPU" },
    archFit: q4nxArchFit("qwen35", "unsloth/Qwen3.5-4B-GGUF"),
  };
  assert.deepEqual(rows(picked), {
    Model: "unsloth/Qwen3.5-4B-GGUF",
    "GGUF file": "Qwen3.5-4B-Q4_K_M.gguf",
    Quant: "Q4_K_M, packs directly",
    "Original model": "Qwen/Qwen3.5-4B",
    "Save directory": "q4nx",
    "Est. Q4NX size": "2.7 GB",
    "NPU memory": "3.6 of 30.8 GB NPU",
    Convertible: "Yes",
  });
  assert.equal(
    rows({ ...picked, archFit: q4nxArchFit("phi3", "acme/phi-GGUF") }).Convertible,
    "Only Phi-4 mini converts.",
  );

  const over = rows({
    ...picked,
    quant: "Q8_0",
    packing: q4nxPacking("Q8_0", "qwen35"),
    memory: { fits: false, label: "~16.5 of 16.0 GB NPU" },
    archFit: q4nxArchFit("qwen35", "unsloth/Qwen3.8-27B-GGUF"),
  });
  assert.equal(over.Quant, "Q8_0, re-quantized to 4-bit");
  assert.equal(
    rows({ ...picked, packing: q4nxPacking("Q4_K_M", null) }).Quant,
    "Q4_K_M, packs directly for Qwen3.5 only",
  );
  assert.equal(over["NPU memory"], "May not fit: ~16.5 of 16.0 GB NPU");
  assert.equal(over.Convertible, "No");

  assert.equal(rows({ ...picked, archFit: q4nxArchFit("llama") }).Convertible, "Yes");
  assert.equal(rows({ ...picked, archFit: q4nxArchFit(null) }).Convertible, "Checked during conversion");

  const empty = rows({
    model: null,
    file: null,
    quant: null,
    packing: null,
    baseModel: " ",
    saveDirectory: "",
    size: null,
    memory: null,
    archFit: null,
  });
  assert.deepEqual([...new Set(Object.values(empty))], ["\u2014"]);
});

test("an empty leftover quant folder and a split GGUF are not offered as a file to convert", () => {
  const files = [
    { filename: "m-Q4_0.gguf", quant: "Q4_0" },
    // One variant for the whole shard set, named by shard 1; the converter would read that shard alone.
    { filename: "BF16/m-BF16-00001-of-00002.gguf", quant: "BF16" },
    { filename: "Q8_0", quant: "Q8_0", cleanable: true },
    { filename: "m-Q4_1.gguf", quant: "Q4_1", cleanable: false },
  ];
  assert.deepEqual(
    q4nxConvertibleFiles(files).map((v) => v.quant),
    ["Q4_0", "Q4_1"],
  );
});

test("only a 4xx answer counts as a refusal; server faults and network drops stay retryable", () => {
  const failed = (status: number | null) => Object.assign(new Error("x"), { status });
  assert.equal(q4nxIsRefusal(failed(400)), true);
  assert.equal(q4nxIsRefusal(failed(422)), true);
  for (const status of [500, 501, 502, 524, null]) {
    assert.equal(q4nxIsRefusal(failed(status)), false, String(status));
  }
  assert.equal(q4nxIsRefusal(new TypeError("Failed to fetch")), false);
  assert.equal(q4nxIsRefusal(null), false);
});
