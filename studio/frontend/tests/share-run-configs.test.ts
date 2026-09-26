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
const { resolveRunConfigTarget } = await import("./helpers/sharing-target.ts");
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
    assert.deepEqual(
      mergeSharedRunConfig(defaults, parsed.value.config, false),
      {
        ...defaults,
        ...patch,
      },
    );
  }
  assert.deepEqual(
    mergeSharedRunConfig(defaults, { nParallel: undefined }, false),
    defaults,
  );
  const patch = { llamaExtraArgs: ["--metrics"] };
  const merged = mergeSharedRunConfig(defaults, patch, false);
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
    assert.deepEqual(
      mergeSharedRunConfig(defaults, { maxSeqLength: value }, false),
      {
        ...defaults,
        customContextLength: null,
        maxSeqLength: value,
      },
    );
    assert.deepEqual(
      mergeSharedRunConfig(defaults, { customContextLength: value }, false),
      {
        ...defaults,
        customContextLength: value,
        maxSeqLength: null,
      },
    );
  }
  assert.deepEqual(mergeSharedRunConfig(defaults, { nParallel: 2 }, false), {
    ...defaults,
    nParallel: 2,
  });
  assert.deepEqual(
    mergeSharedRunConfig(defaults, { maxSeqLength: undefined }, false),
    defaults,
  );
  const agreed = { customContextLength: 8192, maxSeqLength: 8192 };
  assert.deepEqual(mergeSharedRunConfig(defaults, agreed, false), {
    ...defaults,
    ...agreed,
  });
  assert.throws(() => createRunConfigLink({ config: defaults }), /must agree/);
});

test("GGUF imports use the requested context when both context fields are shared", () => {
  const defaults = {
    ...DEFAULT_PER_MODEL_CONFIG,
    customContextLength: 4096,
    maxSeqLength: 2048,
  };
  for (const patch of [
    { customContextLength: null, maxSeqLength: 8192 },
    { customContextLength: 8192, maxSeqLength: null },
    { customContextLength: 8192, maxSeqLength: 8192 },
  ]) {
    const parsed = parseRunConfigLink(createRunConfigLink({ config: patch }));
    assert.ok(parsed.kind === "valid");
    assert.deepEqual(
      mergeSharedRunConfig(defaults, parsed.value.config, true),
      {
        ...defaults,
        customContextLength: 8192,
        maxSeqLength: null,
      },
    );
  }
  for (const patch of [{ nParallel: 2 }, { maxSeqLength: undefined }, {}]) {
    assert.deepEqual(mergeSharedRunConfig(defaults, patch, true), {
      ...defaults,
      ...(patch.nParallel !== undefined ? { nParallel: patch.nParallel } : {}),
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

for (const [model, isGguf] of [
  ["owner/Model-GGUF", true],
  ["owner/native", false],
] as const) {
  test(`a settings-only link's format does not decide how the recipient's own model opens: GGUF=${isGguf}`, () => {
    const unknown = {
      ...selection,
      loadedIsGguf: null,
      activeGgufVariant: null,
      activeNativePathToken: null,
    };
    const value = { isGguf: !isGguf, config: { nParallel: 3 } };
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
      assert.equal(target?.meta.isGguf, isGguf);
      assert.equal(target?.meta.ggufVariant, undefined);
    }
  });
}
