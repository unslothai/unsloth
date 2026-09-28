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

const { createRunConfigLink, parseRunConfigLink } = await import(
  "./helpers/sharing-links.ts"
);
const { mergeSharedRunConfig } = await import(
  "../src/features/model-picker/sharing/fields.ts"
);
const { createRunConfigInbox } = await import(
  "../src/features/model-picker/sharing/inbox.ts"
);
const { isKnownNonGgufModel, resolveRunConfigTarget } = await import(
  "./helpers/sharing-target.ts"
);
const { DEFAULT_PER_MODEL_CONFIG } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
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

test("safetensors-only settings are neither shared nor imported", () => {
  const defaults = {
    ...DEFAULT_PER_MODEL_CONFIG,
    customContextLength: 4096,
    maxSeqLength: 2048,
    mlxKvQuant: "8" as const,
  };
  const link = createRunConfigLink({
    config: {
      customContextLength: 8192,
      maxSeqLength: 8192,
      mlxKvQuant: "4",
    },
  });
  assert.ok(!link.includes("maxSeqLength"));
  assert.ok(!link.includes("mlxKv"));
  const parsed = parseRunConfigLink(link);
  assert.ok(parsed.kind === "valid");
  assert.deepEqual(parsed.value.config, { customContextLength: 8192 });
  assert.deepEqual(
    mergeSharedRunConfig(defaults, {
      customContextLength: 8192,
      maxSeqLength: 1024,
      mlxKvQuant: "4",
    }),
    { ...defaults, customContextLength: 8192 },
  );
  for (const query of [
    "maxSeqLength=8192",
    "mlxKvQuant=4",
    "mlxKvBits=4",
    "isGguf=true",
    "isGguf=false",
  ]) {
    assert.deepEqual(parseRunConfigLink(`unsloth://run?v=1&${query}`), {
      kind: "invalid",
      error: "This run configuration link contains an unsupported setting.",
    });
  }
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

test("a linked model opens as GGUF even when the recipient lists it as safetensors", () => {
  const inventory = {
    ...selection,
    models: [{ id: "owner/other", isGguf: false, isLora: false }],
  };
  const target = resolveRunConfigTarget(
    { model: "owner/other", ggufVariant: "Q8_0", config: {} },
    inventory,
  );
  assert.equal(target?.meta.isGguf, true);
  assert.equal(target?.meta.ggufVariant, "Q8_0");
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
  assert.equal(isKnownNonGgufModel("/models/adapter", inventory), true);
  assert.equal(isKnownNonGgufModel("/models/export", inventory), false);
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

test("settings-only links keep the recipient's GGUF selection and flag a known safetensors model", () => {
  const gguf = resolveRunConfigTarget({ config: { nParallel: 3 } }, selection);
  assert.equal(gguf?.meta.isGguf, true);
  assert.equal(gguf?.meta.ggufVariant, "Q4_K_M");
  assert.equal(gguf?.meta.isDownloaded, true);
  assert.equal(gguf?.meta.loadId, selection.activeLoadId);
  assert.equal(isKnownNonGgufModel("owner/model", selection), false);
  const native = { ...selection, loadedIsGguf: false, activeGgufVariant: null };
  const target = resolveRunConfigTarget(
    { ggufVariant: "Q8_0", config: { nParallel: 3 } },
    native,
  );
  assert.equal(target?.meta.isGguf, false);
  assert.equal(target?.meta.ggufVariant, undefined);
  assert.equal(isKnownNonGgufModel("owner/model", native), true);
  assert.equal(isKnownNonGgufModel("", native), false);
  assert.equal(
    isKnownNonGgufModel("/models/model-Q4_K_M.gguf", {
      ...native,
      params: { checkpoint: "/models/model-Q4_K_M.gguf" },
    }),
    false,
  );
});

for (const model of ["owner/Model-GGUF", "owner/native"]) {
  test(`settings-only links open a recipient model of unknown format as GGUF: ${model}`, () => {
    const unknown = {
      ...selection,
      loadedIsGguf: null,
      activeGgufVariant: null,
      activeNativePathToken: null,
    };
    const value = { config: { nParallel: 3 } };
    for (const target of [
      resolveRunConfigTarget(
        value,
        { ...unknown, params: { checkpoint: "" } },
        model,
      ),
      resolveRunConfigTarget(value, {
        ...unknown,
        params: { checkpoint: model },
      }),
    ]) {
      assert.equal(target?.id, model);
      assert.equal(target?.meta.isGguf, true);
      assert.equal(target?.meta.ggufVariant, undefined);
    }
    assert.equal(isKnownNonGgufModel(model, unknown), false);
  });
}
