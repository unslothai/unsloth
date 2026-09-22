// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();

const { createRunConfigLink, parseRunConfigLink, MAX_RUN_CONFIG_URL_LENGTH } =
  await import("../src/features/model-picker/sharing/links.ts");
const { SHARED_CONFIG_KEYS } = await import(
  "../src/features/model-picker/sharing/fields.ts"
);
const { createRunConfigInbox, mergeSharedRunConfig } = await import(
  "../src/features/model-picker/sharing/inbox.ts"
);
const { resolveRunConfigTarget } = await import("./helpers/sharing-target.ts");
const { DEFAULT_PER_MODEL_CONFIG } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
type PerModelConfig = typeof DEFAULT_PER_MODEL_CONFIG;

const fullConfig: PerModelConfig = {
  customContextLength: 32768,
  maxSeqLength: 32768,
  kvCacheDtype: "q8_0",
  mlxKvBits: 4,
  speculativeType: "dspark",
  specDraftNMax: 8,
  specDraftCacheDtype: "q4_0",
  nParallel: 4,
  reasoningBudget: 0,
  reasoningBudgetMessage: "Think less\n回答 🦥",
  nBatch: 2048,
  nUbatch: 512,
  loadMode: "mmap+mlock",
  ctxCheckpoints: 0,
  cacheRam: -1,
  tensorParallel: true,
  disableVision: false,
  chatTemplateOverride: "",
  llamaExtraArgs: [
    "--rope-scaling",
    "yarn",
    "--yarn-orig-ctx",
    "32768",
    "--flash-attn",
    "on",
    "--no-warmup",
  ],
  gpuMemoryMode: "manual",
  gpuLayers: 0,
  nCpuMoe: 0,
  selectedGpuIds: [1, 0],
  selectedGpuIndexKind: "physical",
};

test("native schemes accept case variations without relaxing address validation", () => {
  const value = { model: "owner/model", config: { nParallel: 3 } };
  const link = createRunConfigLink(value);
  for (const scheme of ["unsloth:", "UNSLOTH:", "UnSlOtH:"]) {
    const url = link.replace("unsloth:", scheme);
    assert.deepEqual(parseRunConfigLink(url), { kind: "valid", value });
    assert.equal(parseRunConfigLink(`${url}#ignored`).kind, "invalid");
    assert.equal(
      parseRunConfigLink(
        `${url}&reasoningBudgetMessage=${"a".repeat(MAX_RUN_CONFIG_URL_LENGTH)}`,
      ).kind,
      "invalid",
    );
    assert.equal(
      parseRunConfigLink(`${scheme}//hub?model=owner/model`).kind,
      "unrelated",
    );
  }
});

test("both link formats require exactly one supported version", () => {
  for (const prefix of ["unsloth://run", "https://example.com/chat#run"]) {
    for (const query of ["", "?", "?model=owner/model", "?nParallel=3"]) {
      assert.deepEqual(parseRunConfigLink(`${prefix}${query}`), {
        kind: "invalid",
        error: "This run configuration link is missing its version.",
      });
    }
    for (const query of ["v=", "v=0", "v=2", "v=01", "v=1&v=1", "v=1&%76=1"]) {
      assert.equal(parseRunConfigLink(`${prefix}?${query}`).kind, "invalid");
    }
    assert.deepEqual(parseRunConfigLink(`${prefix}?nParallel=3&v=1`), {
      kind: "valid",
      value: { config: { nParallel: 3 } },
    });
  }
});

test("every field round-trips independently in browser and desktop links", () => {
  assert.deepEqual(
    Object.keys(fullConfig).sort(),
    [...SHARED_CONFIG_KEYS].sort(),
  );
  for (const key of SHARED_CONFIG_KEYS) {
    for (const base of [
      undefined,
      "http://localhost:8888/hub?token=private#old",
      "https://studio.example/chat",
    ]) {
      const value = { config: { [key]: fullConfig[key] } };
      const url = createRunConfigLink(value, base);
      assert.deepEqual(parseRunConfigLink(url), { kind: "valid", value }, key);
      assert.ok(!url.includes("token=private"));
    }
  }
});

test("an empty link and independently omitted model identity fields are valid", () => {
  for (const url of [
    "unsloth://run?v=1",
    "unsloth://run/?v=1",
    "http://localhost:8888/chat#run?v=1",
    "https://example.com/chat#run?v=1",
  ]) {
    assert.deepEqual(parseRunConfigLink(url), {
      kind: "valid",
      value: { config: {} },
    });
  }
  for (const identity of [
    {},
    { model: "unsloth/Model-GGUF" },
    { model: "own_er/mod_el" },
    { ggufVariant: "Q4_K_M" },
    { isGguf: false },
  ]) {
    const value = { ...identity, config: {} };
    assert.deepEqual(parseRunConfigLink(createRunConfigLink(value)), {
      kind: "valid",
      value,
    });
  }
});

test("full configs preserve argv tokens and Unicode exactly without a shell", () => {
  const value = {
    model: "unsloth/Model-GGUF",
    ggufVariant: "Q4_K_M/model-00001-of-00002.gguf",
    isGguf: true,
    config: fullConfig,
  };
  assert.deepEqual(parseRunConfigLink(createRunConfigLink(value)), {
    kind: "valid",
    value,
  });
});

test("null, false, zero and empty values survive; omissions retain the existing defaults", () => {
  const defaults = {
    ...DEFAULT_PER_MODEL_CONFIG,
    nParallel: 8,
    llamaExtraArgs: ["--metrics"],
    tensorParallel: true,
  };
  for (const patch of [
    {},
    { nParallel: 2 },
    { nParallel: null },
    { tensorParallel: false },
    { reasoningBudget: 0 },
    { llamaExtraArgs: [] },
    { llamaExtraArgs: null },
    { reasoningBudgetMessage: "" },
    { chatTemplateOverride: "" },
    { chatTemplateOverride: null },
  ]) {
    const parsed = parseRunConfigLink(createRunConfigLink({ config: patch }));
    assert.equal(parsed.kind, "valid");
    if (parsed.kind !== "valid") throw new Error("Invalid test fixture");
    assert.deepEqual(mergeSharedRunConfig(defaults, parsed.value.config), {
      ...defaults,
      ...patch,
    });
  }
  assert.deepEqual(
    mergeSharedRunConfig(defaults, { nParallel: undefined }),
    defaults,
  );
  const patch = { llamaExtraArgs: ["--metrics"] };
  const merged = mergeSharedRunConfig(defaults, patch);
  merged.llamaExtraArgs?.push("--verbose");
  assert.deepEqual(patch.llamaExtraArgs, ["--metrics"]);
  assert.deepEqual(defaults.llamaExtraArgs, ["--metrics"]);
});

test("a shared context pin replaces its legacy field while unrelated defaults remain", () => {
  const defaults = {
    ...DEFAULT_PER_MODEL_CONFIG,
    customContextLength: 4096,
    maxSeqLength: 2048,
    nParallel: 8,
  };
  for (const value of [8192, null]) {
    assert.deepEqual(mergeSharedRunConfig(defaults, { maxSeqLength: value }), {
      ...defaults,
      customContextLength: null,
      maxSeqLength: value,
    });
    assert.deepEqual(
      mergeSharedRunConfig(defaults, { customContextLength: value }),
      {
        ...defaults,
        customContextLength: value,
        maxSeqLength: null,
      },
    );
  }
  assert.deepEqual(mergeSharedRunConfig(defaults, { nParallel: 2 }), {
    ...defaults,
    nParallel: 2,
  });
  assert.deepEqual(
    mergeSharedRunConfig(defaults, { maxSeqLength: undefined }),
    defaults,
  );
  const agreed = { customContextLength: 8192, maxSeqLength: 8192 };
  assert.deepEqual(mergeSharedRunConfig(defaults, agreed), {
    ...defaults,
    ...agreed,
  });
  assert.throws(() => createRunConfigLink({ config: defaults }), /must agree/);
});

test("invalid input never produces a partial configuration", () => {
  const invalid = [
    "customContextLength=4096&maxSeqLength=8192",
    "v=2",
    "v=01",
    "nParallel=2&nParallel=3",
    "model=owner/model&model=other/model",
    "model=_owner/model",
    "model=owner_/model",
    "model=owner/_model",
    "model=owner/model_",
    "__proto__=true",
    "constructor=true",
    "hfToken=secret",
    "nParallel=2&unknown=true",
    "nParallel=0",
    "nParallel=65",
    "nParallel=2.5",
    "nParallel=%222%22",
    "nParallel=NaN",
    "tensorParallel=1",
    "tensorParallel=null",
    "disableVision=%22false%22",
    "customContextLength=127",
    "maxSeqLength=1048577",
    "reasoningBudget=-2",
    "nBatch=0",
    "nUbatch=65537",
    "ctxCheckpoints=257",
    "cacheRam=-2",
    "kvCacheDtype=%22unsupported%22",
    "speculativeType=%22unknown%22",
    "mlxKvBits=1",
    "selectedGpuIds=[1,1]",
    "selectedGpuIds=[]",
    "selectedGpuIds=[-1]",
    "selectedGpuIds=[0.5]",
    "llamaExtraArgs={}",
    "llamaExtraArgs=[3]",
    "llamaExtraArgs=[%22\\u0000%22]",
    "llamaExtraArgs=[%22\\ud800%22]",
    "llamaExtraArgs=[%22\\r%22]",
    "reasoningBudgetMessage=%FF",
    "reasoningBudgetMessage=%GG",
    "reasoningBudgetMessage=%",
    "model=/tmp/model",
    "model=C:%5Cmodels%5Cmodel",
    "model=owner/..",
    "model=owner/repo.git",
    "ggufVariant=../model.gguf",
    "ggufVariant=C:%5Cmodel.gguf",
    "ggufVariant=%00",
    "isGguf=false&ggufVariant=Q4_K_M",
  ];
  for (const query of invalid) {
    assert.equal(
      parseRunConfigLink(
        `unsloth://run?${query.startsWith("v=") ? "" : "v=1&"}${query}`,
      ).kind,
      "invalid",
      query,
    );
  }
  for (const url of [
    "unsloth://run/extra?v=1",
    "unsloth://run?v=1&model=owner/model#ignored",
    "unsloth://user@run?v=1",
    "unsloth://run:80?v=1",
  ]) {
    assert.equal(parseRunConfigLink(url).kind, "invalid", url);
  }
});

test("unrelated links are left to their existing handlers", () => {
  for (const url of [
    "invalid",
    "unsloth://open_from_hf?model=owner/model",
    "https://example.com/chat?model=owner/model",
    "https://example.com/chat#running",
    "https://example.com/a b",
    `https://example.com/chat?unrelated=${"a".repeat(MAX_RUN_CONFIG_URL_LENGTH)}`,
    "javascript:alert(1)",
  ]) {
    assert.equal(parseRunConfigLink(url).kind, "unrelated");
  }
});

test("links and field payloads have bounded sizes", () => {
  assert.equal(
    parseRunConfigLink(
      `unsloth://run?reasoningBudgetMessage=${"a".repeat(MAX_RUN_CONFIG_URL_LENGTH)}`,
    ).kind,
    "invalid",
  );
  assert.throws(() =>
    createRunConfigLink({
      config: { reasoningBudgetMessage: "🦥".repeat(2049) },
    }),
  );
  assert.throws(() =>
    createRunConfigLink({ config: { llamaExtraArgs: Array(257).fill("x") } }),
  );
  assert.throws(() =>
    createRunConfigLink({
      config: { chatTemplateOverride: "a".repeat(65537) },
    }),
  );
  assert.throws(() =>
    createRunConfigLink({ config: {} }, "file:///tmp/index.html"),
  );
  assert.throws(() =>
    createRunConfigLink({ config: {} }, "https://user:secret@example.com/"),
  );
});

test("pending imports are scoped, replaced by newer links and consumed once", () => {
  const inbox = createRunConfigInbox();
  let notifications = 0;
  const unsubscribe = inbox.subscribe(() => notifications++);
  inbox.submit({ id: "first", value: { config: { nParallel: 2 } } });
  assert.equal(inbox.take("first", "model-A"), null);
  inbox.bind("first", "model-A");
  assert.equal(inbox.take("first", "model-B"), null);
  inbox.submit({ id: "second", value: { config: { nParallel: 4 } } });
  inbox.clear("first");
  inbox.bind("first", "model-A");
  assert.equal(inbox.getSnapshot()?.id, "second");
  inbox.bind("second", "model-B");
  assert.deepEqual(inbox.take("second", "model-B"), { nParallel: 4 });
  assert.equal(inbox.take("second", "model-B"), null);
  assert.equal(inbox.getSnapshot(), null);
  assert.equal(notifications, 5);
  unsubscribe();
  inbox.submit({ id: "third", value: { config: {} } });
  assert.equal(notifications, 5);
});

test("closing the last editor cancels a delayed import without breaking StrictMode remounts", async () => {
  const inbox = createRunConfigInbox();
  const cancelled: string[] = [];
  const onCancel = (request: { id: string }) => cancelled.push(request.id);
  inbox.submit({ id: "first", value: { config: { nParallel: 2 } } });
  inbox.bind("first", "model-A");
  const release = inbox.retainEditor("model-A", onCancel);
  release();
  const releaseRemount = inbox.retainEditor("model-A", onCancel);
  await Promise.resolve();
  assert.equal(inbox.getSnapshot()?.id, "first");
  assert.deepEqual(cancelled, []);
  const releasePeer = inbox.retainEditor("model-A", onCancel);
  releaseRemount();
  await Promise.resolve();
  assert.equal(inbox.getSnapshot()?.id, "first");
  assert.deepEqual(cancelled, []);
  releasePeer();
  releasePeer();
  await Promise.resolve();
  assert.equal(inbox.getSnapshot(), null);
  assert.deepEqual(cancelled, ["first"]);
});

test("an editor cleanup cannot cancel a newer link for a different model", async () => {
  const inbox = createRunConfigInbox();
  inbox.submit({ id: "first", value: { config: {} } });
  inbox.bind("first", "model-A");
  const release = inbox.retainEditor("model-A");
  release();
  inbox.submit({ id: "second", value: { config: {} } });
  inbox.bind("second", "model-B");
  await Promise.resolve();
  assert.equal(inbox.getSnapshot()?.id, "second");
});

test("editor cleanup reports no cancellation for completed or model-only imports", async () => {
  for (const completed of [false, true]) {
    const inbox = createRunConfigInbox();
    inbox.submit({
      id: "first",
      draftKey: "model-A",
      value: { config: completed ? { nParallel: 2 } : {} },
    });
    const release = inbox.retainEditor("model-A", () =>
      assert.fail("Unexpected cancellation"),
    );
    if (completed) inbox.take("first", "model-A");
    release();
    await Promise.resolve();
    assert.equal(inbox.getSnapshot(), null);
  }
});

test("an editor cleanup cannot cancel a newer request for the same model", async () => {
  const inbox = createRunConfigInbox();
  inbox.submit({ id: "first", value: { config: {} } });
  inbox.bind("first", "model-A");
  const release = inbox.retainEditor("model-A");
  release();
  inbox.submit({ id: "second", value: { config: {} } });
  inbox.bind("second", "model-A");
  await Promise.resolve();
  assert.equal(inbox.getSnapshot()?.id, "second");
});

const selection = {
  params: { checkpoint: "owner/model" },
  activeGgufVariant: "Q4_K_M",
  loadedIsGguf: true,
  activeNativePathToken: "local-token",
  activeLoadId: "/secondary/models--owner--model/snapshots/pinned",
  models: [],
  loras: [],
};

test("omitted model identity inherits the selected model and native capability locally", () => {
  const target = resolveRunConfigTarget({ config: {} }, selection);
  assert.equal(target?.id, "owner/model");
  assert.equal(target?.meta.isGguf, true);
  assert.equal(target?.meta.ggufVariant, "Q4_K_M");
  assert.equal(target?.meta.nativePathToken, "local-token");
  assert.equal(target?.meta.loadId, selection.activeLoadId);
  assert.equal(target?.meta.isDownloaded, true);
  assert.equal(
    resolveRunConfigTarget(
      { config: {} },
      { ...selection, params: { checkpoint: "" } },
    ),
    null,
  );
});

test("an explicit native format cannot inherit a GGUF variant or native file token", () => {
  const target = resolveRunConfigTarget(
    { model: selection.params.checkpoint, isGguf: false, config: {} },
    selection,
  );
  assert.equal(target?.meta.isGguf, false);
  assert.equal(target?.meta.ggufVariant, undefined);
  assert.equal(target?.meta.nativePathToken, undefined);
  assert.equal(target?.meta.loadId, undefined);
});

test("a different model uses its inventory format without inheriting the selected model's identity", () => {
  const inventory = {
    ...selection,
    models: [{ id: "owner/other", isGguf: true, isLora: false }],
  };
  const target = resolveRunConfigTarget(
    { model: "owner/other", config: {} },
    inventory,
  );
  assert.equal(target?.meta.isGguf, true);
  assert.equal(target?.meta.ggufVariant, undefined);
  assert.equal(target?.meta.nativePathToken, undefined);
  const explicit = resolveRunConfigTarget(
    { model: "owner/other", isGguf: false, config: {} },
    inventory,
  );
  assert.equal(explicit?.meta.isGguf, false);
});

test("exported GGUF and adapter identities use existing inventory metadata", () => {
  const inventory = {
    ...selection,
    loras: [
      { id: "/models/export", exportType: "gguf" as const },
      { id: "/models/adapter", exportType: "lora" as const },
    ],
  };
  const gguf = resolveRunConfigTarget(
    { config: {} },
    {
      ...inventory,
      params: { checkpoint: "/models/export" },
      loadedIsGguf: null,
      activeGgufVariant: null,
    },
  );
  assert.equal(gguf?.meta.isGguf, true);
  assert.equal(gguf?.meta.isLora, false);
  const adapter = resolveRunConfigTarget(
    { config: {} },
    {
      ...inventory,
      params: { checkpoint: "/models/adapter" },
      loadedIsGguf: null,
      activeGgufVariant: null,
    },
  );
  assert.equal(adapter?.meta.isLora, true);
  assert.equal(adapter?.meta.isGguf, false);
});

for (const id of [
  "/models/model-Q4_K_M.gguf",
  "C:\\Models\\model-Q4_K_M.gguf",
  "model-Q4_K_M.gguf",
  "\\\\server\\models\\model.gguf",
]) {
  test(`standalone GGUF import uses the existing file identity: ${id}`, () => {
    for (const ggufVariant of [undefined, "Q4_K_M", "Q8_0"]) {
      const target = resolveRunConfigTarget(
        { ggufVariant, config: { nParallel: 3 } },
        {
          ...selection,
          params: { checkpoint: id },
          activeLoadId: id,
        },
      );
      assert.equal(target?.meta.ggufVariant, undefined);
      assert.equal(target?.meta.isGguf, true);
      assert.equal(target?.meta.isDownloaded, true);
      assert.equal(target?.meta.nativePathToken, "local-token");
      assert.equal(target?.meta.loadId, id);
    }
  });
}

for (const isGguf of [false, true]) {
  test(`settings-only links preserve the recipient's known format: GGUF=${isGguf}`, () => {
    const target = resolveRunConfigTarget(
      {
        isGguf: !isGguf,
        ggufVariant: isGguf ? undefined : "Q8_0",
        config: { nParallel: 3 },
      },
      {
        ...selection,
        loadedIsGguf: isGguf,
        activeGgufVariant: isGguf ? "Q4_K_M" : null,
      },
    );
    assert.equal(target?.meta.isGguf, isGguf);
    assert.equal(target?.meta.ggufVariant, isGguf ? "Q4_K_M" : undefined);
    assert.equal(target?.meta.isDownloaded, true);
    assert.equal(target?.meta.loadId, selection.activeLoadId);
  });
}
