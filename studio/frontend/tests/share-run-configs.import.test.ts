// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { type TestContext } from "node:test";
import type * as ImportConfig from "../src/features/model-picker/sharing/import-config.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

registerBundlerResolver();
installLocalStorageFake();
const drafts = await import(
  "../src/features/model-picker/model-config/model-config-draft.ts"
);
const { DEFAULT_PER_MODEL_CONFIG } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const fields = await import("../src/features/model-picker/sharing/fields.ts");
const inboxModule = await import(
  "../src/features/model-picker/sharing/inbox.ts"
);
type Config = typeof DEFAULT_PER_MODEL_CONFIG;
let sequence = 0;

function harness(t: TestContext, patch: Partial<Config> = { nParallel: 3 }) {
  const key = drafts.modelConfigDraftKey(`owner/model-${++sequence}`, "Q4_K_M");
  const inbox = inboxModule.createRunConfigInbox();
  const changes: Partial<Config>[] = [];
  const errors: string[] = [];
  const successes: string[] = [];
  const releaseDraft = drafts.retainModelConfigDraft(key);
  drafts.primeModelConfigDraft(
    key,
    {
      config: {
        ...DEFAULT_PER_MODEL_CONFIG,
        maxSeqLength: 4096,
        llamaExtraArgs: ["--no-warmup"],
      },
      remembered: true,
    },
    "none",
  );
  inbox.submit({ id: "import", draftKey: key, value: { config: patch } });
  const { scheduleRunConfigImport } = loadWithStubs<typeof ImportConfig>(
    new URL(
      "../src/features/model-picker/sharing/import-config.ts",
      import.meta.url,
    ),
    {
      "@/lib/toast": {
        toast: {
          error: (message: string) => errors.push(message),
          success: (message: string) => successes.push(message),
        },
      },
      "../model-config/model-config-draft": drafts,
      "./fields": fields,
      "./inbox": { ...inboxModule, runConfigInbox: inbox },
    },
  );
  const options = {
    key,
    canImport: true,
    ready: true,
    hydrated: true,
    pending: inbox.getSnapshot(),
    onImport: (value: Partial<Config>) => changes.push(value),
  };
  t.after(releaseDraft);
  return {
    inbox,
    key,
    errors,
    successes,
    changes,
    options,
    schedule: scheduleRunConfigImport,
    releaseDraft,
  };
}

test("Strict Mode cleanup leaves the request for the surviving editor and applies once", async (t) => {
  const app = harness(t);
  const firstHost = app.inbox.retainEditor(app.key);
  const cancel = app.schedule(app.options);
  assert.equal(
    drafts.readModelConfigDraft(app.key)?.config.nParallel,
    DEFAULT_PER_MODEL_CONFIG.nParallel,
  );
  cancel?.();
  firstHost();
  const secondHost = app.inbox.retainEditor(app.key);
  t.after(secondHost);
  app.schedule(app.options);
  app.schedule(app.options);
  await Promise.resolve();
  assert.equal(drafts.readModelConfigDraft(app.key)?.config.nParallel, 3);
  assert.deepEqual(app.changes, [{ nParallel: 3 }]);
  assert.equal(app.successes.length, 1);
  assert.equal(app.inbox.getSnapshot(), null);
  assert.equal(drafts.isModelConfigDraftEdited(app.key), true);
});

for (const reason of [
  "sidebar",
  "hydrating",
  "wrong draft",
  "no request",
  "no draft",
] as const) {
  test(`imports wait without consuming for ${reason}`, async (t) => {
    const app = harness(t);
    const options = { ...app.options };
    if (reason === "sidebar") options.canImport = false;
    if (reason === "hydrating") options.ready = false;
    if (reason === "wrong draft") options.key = "other";
    if (reason === "no request") options.pending = null;
    if (reason === "no draft") app.releaseDraft();
    assert.equal(app.schedule(options), undefined);
    await Promise.resolve();
    assert.equal(app.inbox.getSnapshot(), app.options.pending);
    assert.deepEqual(app.changes, []);
  });
}

for (const reason of [
  "unmount",
  "edit",
  "new link",
  "draft released",
] as const) {
  test(`a queued import cannot overwrite ${reason}`, async (t) => {
    const app = harness(t);
    const cancel = app.schedule(app.options);
    if (reason === "unmount") cancel?.();
    if (reason === "edit") {
      app.inbox.clear("import");
      drafts.patchModelConfigDraft(app.key, (config) => ({
        ...config,
        nParallel: 7,
      }));
    }
    if (reason === "new link")
      app.inbox.submit({
        id: "new",
        draftKey: app.key,
        value: { config: { nParallel: 8 } },
      });
    if (reason === "draft released") app.releaseDraft();
    await Promise.resolve();
    assert.deepEqual(app.changes, []);
    assert.deepEqual(app.successes, []);
    if (reason === "edit")
      assert.equal(drafts.readModelConfigDraft(app.key)?.config.nParallel, 7);
    if (reason === "new link") assert.equal(app.inbox.getSnapshot()?.id, "new");
    if (reason === "draft released")
      assert.equal(drafts.readModelConfigDraft(app.key), undefined);
  });
}

test("failed hydration preserves the import without changing settings or remembered state", async (t) => {
  const app = harness(t);
  const before = drafts.readModelConfigDraft(app.key);
  app.schedule({ ...app.options, hydrated: false });
  await Promise.resolve();
  assert.equal(drafts.readModelConfigDraft(app.key), before);
  assert.equal(drafts.isModelConfigDraftEdited(app.key), false);
  assert.equal(app.inbox.getSnapshot(), app.options.pending);
  assert.equal(app.errors.length, 1);
  assert.deepEqual(app.changes, []);
  app.schedule(app.options);
  await Promise.resolve();
  assert.equal(app.inbox.getSnapshot(), null);
  assert.deepEqual(app.changes, [{ nParallel: 3 }]);
  assert.equal(app.successes.length, 1);
  assert.equal(drafts.readModelConfigDraft(app.key)?.remember, true);
});

test("model-only links do not mark settings edited or report a settings import", async (t) => {
  const app = harness(t, {});
  app.schedule(app.options);
  await Promise.resolve();
  assert.equal(app.inbox.getSnapshot(), null);
  assert.equal(drafts.isModelConfigDraftEdited(app.key), false);
  assert.deepEqual(app.successes, []);
});

for (const reason of ["edit", "closed editor", "new link"] as const) {
  test(`late hydration cannot revive an import cancelled by ${reason}`, async (t) => {
    const app = harness(t);
    const release = app.inbox.retainEditor(app.key);
    t.after(release);
    app.schedule({ ...app.options, hydrated: false });
    await Promise.resolve();
    if (reason === "closed editor") {
      release();
    } else if (reason === "new link") {
      app.inbox.submit({
        id: "new",
        draftKey: app.key,
        value: { config: { nParallel: 8 } },
      });
    } else {
      app.inbox.clear("import");
      drafts.patchModelConfigDraft(app.key, (config) => ({
        ...config,
        nParallel: 7,
      }));
    }
    await Promise.resolve();
    app.schedule(app.options);
    await Promise.resolve();
    assert.deepEqual(app.changes, []);
    assert.deepEqual(app.successes, []);
    assert.equal(app.errors.length, 1);
    if (reason === "edit")
      assert.equal(drafts.readModelConfigDraft(app.key)?.config.nParallel, 7);
    if (reason === "new link") assert.equal(app.inbox.getSnapshot()?.id, "new");
  });
}

test("review lists only changed fields including cleared context aliases and exact prompt/argv values", async (t) => {
  const patch = {
    customContextLength: 8192,
    reasoningBudgetMessage: "Read <this>\nand then continue",
    llamaExtraArgs: ["--rope-scaling", "yarn"],
    tensorParallel: DEFAULT_PER_MODEL_CONFIG.tensorParallel,
  };
  const app = harness(t, patch);
  drafts.setExtraArgsEditForDraft(app.key, {
    text: "unfinished '",
    source: "--no-warmup",
  });
  app.schedule(app.options);
  await Promise.resolve();
  assert.deepEqual(app.changes, [
    {
      customContextLength: 8192,
      maxSeqLength: null,
      reasoningBudgetMessage: patch.reasoningBudgetMessage,
      llamaExtraArgs: patch.llamaExtraArgs,
    },
  ]);
  assert.equal(drafts.readExtraArgsEditForDraft(app.key), undefined);
  assert.equal(drafts.readModelConfigDraft(app.key)?.remember, true);
});

test("omitted extra arguments preserve an existing raw edit", async (t) => {
  const app = harness(t);
  const edit = { text: "unfinished '", source: "--no-warmup" };
  drafts.setExtraArgsEditForDraft(app.key, edit);
  app.schedule(app.options);
  await Promise.resolve();
  assert.deepEqual(drafts.readExtraArgsEditForDraft(app.key), edit);
  assert.deepEqual(
    drafts.readModelConfigDraft(app.key)?.config.llamaExtraArgs,
    ["--no-warmup"],
  );
});
