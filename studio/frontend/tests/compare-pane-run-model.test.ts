// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  isLoadingPick,
  loadComparePaneModel,
  prepareComparePaneRun,
  registerComparePaneRunPreparer,
} from "../src/features/chat/compare-run-ownership.ts";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { checkpointMatchesPick } = await import(
  "../src/features/chat/lib/resident-model-match.ts"
);

const adapterSource = readSrc("features/chat/api/chat-adapter.ts");
const chatPageSource = readSrc("features/chat/chat-page.tsx");
const generalCompareSource = chatPageSource.slice(
  chatPageSource.indexOf("const GeneralCompareContent"),
  chatPageSource.indexOf("function formatProjectChatDate"),
);

const idle = async () => {};
const live = () => new AbortController().signal;

test("a compare pane run is prepared for its own pane and pair", async () => {
  const calls: string[] = [];
  const unregister = registerComparePaneRunPreparer("pair-a", async (pane) => {
    calls.push(pane);
  });
  await prepareComparePaneRun("pair-a", "model1", live(), idle);
  await prepareComparePaneRun("pair-a", "model2", live(), idle);
  await prepareComparePaneRun("pair-b", "model1", live(), idle);
  await prepareComparePaneRun("pair-a", "base", live(), idle);
  await prepareComparePaneRun(undefined, "model1", live(), idle);
  assert.deepEqual(calls, ["model1", "model2"]);
  unregister();
  await prepareComparePaneRun("pair-a", "model1", live(), idle);
  assert.deepEqual(calls, ["model1", "model2"]);
});

test("a pane checks its model only after a load in flight lands", async () => {
  const events: string[] = [];
  const unregister = registerComparePaneRunPreparer("pair-e", async (pane) => {
    events.push(`prepare ${pane}`);
  });
  const wait = async () => {
    await new Promise((resolve) => setTimeout(resolve, 5));
    events.push("load landed");
  };
  await prepareComparePaneRun("pair-e", "model2", live(), wait);
  await prepareComparePaneRun("pair-e", "base", live(), wait);
  await prepareComparePaneRun("pair-x", "model1", live(), wait);
  assert.deepEqual(events, ["load landed", "prepare model2"]);
  unregister();
});

test("a pane uses the preparer registered while it waited", async () => {
  const calls: string[] = [];
  const stale = registerComparePaneRunPreparer("pair-f", async () => {
    calls.push("stale");
  });
  let fresh = () => {};
  await prepareComparePaneRun("pair-f", "model1", live(), async () => {
    stale();
    fresh = registerComparePaneRunPreparer("pair-f", async () => {
      calls.push("fresh");
    });
  });
  assert.deepEqual(calls, ["fresh"]);
  fresh();
});

test("the preparer gets the run's abort signal", async () => {
  let seen: AbortSignal | undefined;
  const unregister = registerComparePaneRunPreparer("pair-g", async (_, signal) => {
    seen = signal;
  });
  const signal = live();
  await prepareComparePaneRun("pair-g", "model1", signal, idle);
  assert.equal(seen, signal);
  unregister();
});

test("a failed pane load stops the pane run", async () => {
  const unregister = registerComparePaneRunPreparer("pair-c", async () => {
    throw new Error("not loaded");
  });
  await assert.rejects(
    prepareComparePaneRun("pair-c", "model1", live(), idle),
    /not loaded/,
  );
  unregister();
});

test("a stale unregister keeps the newer preparer", async () => {
  const calls: string[] = [];
  const first = registerComparePaneRunPreparer("pair-d", async () => {
    calls.push("first");
  });
  const second = registerComparePaneRunPreparer("pair-d", async () => {
    calls.push("second");
  });
  first();
  await prepareComparePaneRun("pair-d", "model1", live(), idle);
  assert.deepEqual(calls, ["second"]);
  second();
});

function paneModel(active: boolean) {
  const state = { active, loading: false, loads: 0, cancels: 0 };
  let finishLoad = (_landed: boolean) => {};
  const model = {
    name: "gemma-3-270m-it",
    isActive: () => state.active,
    isLoading: () => state.loading,
    load: () => {
      state.loads += 1;
      state.loading = true;
      return new Promise<void>((resolve) => {
        finishLoad = (landed) => {
          state.loading = false;
          state.active = landed;
          resolve();
        };
      });
    },
    cancelLoad: () => {
      state.cancels += 1;
      finishLoad(false);
    },
  };
  return { state, model, finishLoad: (landed: boolean) => finishLoad(landed) };
}

test("a pane whose model is active runs without loading", async () => {
  const pane = paneModel(true);
  await loadComparePaneModel(pane.model, live());
  assert.equal(pane.state.loads, 0);
});

test("a pane loads its own model before it runs", async () => {
  const pane = paneModel(false);
  const run = loadComparePaneModel(pane.model, live());
  assert.equal(pane.state.loads, 1);
  pane.finishLoad(true);
  await run;
});

test("a pane whose load did not land is not answered", async () => {
  const pane = paneModel(false);
  const run = loadComparePaneModel(pane.model, live());
  pane.finishLoad(false);
  await assert.rejects(run, /gemma-3-270m-it is not loaded/);
});

test("stopping a pane cancels the load it started", async () => {
  const pane = paneModel(false);
  const controller = new AbortController();
  const run = loadComparePaneModel(pane.model, controller.signal);
  controller.abort();
  await assert.rejects(run, { name: "AbortError" });
  assert.equal(pane.state.cancels, 1);
  assert.equal(pane.state.active, false);
});

test("stopping a pane during load preflight cancels the load once it starts", async () => {
  const pane = paneModel(false);
  const load = pane.model.load;
  pane.model.load = () => {
    const started = load();
    pane.state.loading = false;
    setTimeout(() => {
      pane.state.loading = true;
    }, 150);
    return started;
  };
  const controller = new AbortController();
  const run = loadComparePaneModel(pane.model, controller.signal);
  controller.abort();
  await assert.rejects(run, { name: "AbortError" });
  assert.equal(pane.state.cancels, 1);
  assert.equal(pane.state.active, false);
});

test("a stopped pane leaves a newer pane's load alone", async () => {
  const stopped = paneModel(false);
  const newer = paneModel(false);
  const controller = new AbortController();
  const stoppedRun = loadComparePaneModel(stopped.model, controller.signal);
  stopped.state.loading = false;
  controller.abort();
  const newerRun = loadComparePaneModel(newer.model, live());
  // Same model and quant in both panes: the stopped pane's watch would see the newer load as its own.
  stopped.state.loading = true;
  await new Promise((resolve) => setTimeout(resolve, 250));
  assert.equal(stopped.state.cancels, 0);
  stopped.finishLoad(false);
  newer.finishLoad(true);
  await assert.rejects(stoppedRun, { name: "AbortError" });
  await newerRun;
});

test("a pane's loading pick is its own model and quant", () => {
  const sel = { id: "unsloth/Qwen3-0.6B-GGUF", ggufVariant: "Q4_K_M" };
  assert.ok(isLoadingPick({ id: sel.id, ggufVariant: "Q4_K_M" }, sel));
  assert.ok(!isLoadingPick({ id: sel.id, ggufVariant: "Q8_0" }, sel));
  assert.ok(!isLoadingPick({ id: "unsloth/other-GGUF", ggufVariant: "Q4_K_M" }, sel));
  assert.ok(!isLoadingPick(null, sel));
  assert.ok(isLoadingPick({ id: "org/m", ggufVariant: null }, { id: "org/m" }));
});

test("stopping a pane leaves a load it did not start alone", async () => {
  const pane = paneModel(false);
  const controller = new AbortController();
  const run = loadComparePaneModel(pane.model, controller.signal);
  pane.state.loading = false;
  controller.abort();
  pane.finishLoad(true);
  await assert.rejects(run, { name: "AbortError" });
  assert.equal(pane.state.cancels, 0);
});

test("the chat adapter prepares a compare pane before it reads the model", () => {
  const prepare = adapterSource.indexOf("await prepareComparePaneRun(");
  const firstRead = adapterSource.indexOf(
    "let runtime = useChatRuntimeStore.getState();",
  );
  assert.ok(prepare > 0, "the run must prepare its compare pane");
  assert.ok(prepare < firstRead, "before the checkpoint is read");
  assert.match(
    adapterSource.slice(prepare, firstRead),
    /abortSignal,\s*\(\) => waitForModelReady\(abortSignal\),/,
  );
});

test("Refresh or Edit in a compare pane loads that pane's own model", () => {
  assert.match(
    generalCompareSource,
    /registerComparePaneRunPreparer\(pairId, async \(pane, signal\) => \{\s*const sel = pane === "model1" \? model1 : model2;/,
  );
  assert.match(generalCompareSource, /loadPaneModel\(\{\s*id: sel\.id,/);
  assert.match(generalCompareSource, /checkpointMatchesPick\(\s*store\(\)\.params\.checkpoint,/);
  assert.match(
    generalCompareSource,
    /isLoading: \(\) => isLoadingPick\(store\(\)\.loadingModelPick, sel\),/,
  );

  assert.match(
    generalCompareSource,
    /\.\.\.\(sel\.ggufVariant \|\| sel\.id\.toLowerCase\(\)\.endsWith\("\.gguf"\)\s*\? \{ isGguf: true \}/,
    "a standalone GGUF pane reloads with GGUF load fields, as shared Send does",
  );
  assert.match(
    generalCompareSource,
    /throwOnError: true,\s*previousConfig: currentRuntimePerModelConfig\(\{\s*includeMaxSeqLength: true,/,
    "a failed pane load restores the resident model's settings, as a regular pick does",
  );
  assert.match(chatPageSource, /loadPaneModel=\{selectModel\}/);
  assert.match(chatPageSource, /cancelPaneModelLoad=\{cancelLoading\}/);
});

test("a standalone GGUF pane matches the quant the backend derives", () => {
  const file = "/models/Qwen3-0.6B-Q4_K_M.gguf";
  assert.ok(checkpointMatchesPick(file, "Q4_K_M", { id: file }));
  assert.ok(
    checkpointMatchesPick("unsloth/Qwen3-0.6B-GGUF", "Q4_K_M", {
      id: "unsloth/Qwen3-0.6B-GGUF",
      ggufVariant: "Q4_K_M",
    }),
  );
  assert.ok(
    !checkpointMatchesPick("unsloth/Qwen3-0.6B-GGUF", "Q8_0", {
      id: "unsloth/Qwen3-0.6B-GGUF",
      ggufVariant: "Q4_K_M",
    }),
  );
  assert.ok(
    !checkpointMatchesPick("/models/other.gguf", "Q4_K_M", { id: file }),
  );
});
