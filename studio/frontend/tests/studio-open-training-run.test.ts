// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as LibraryOriginModule from "../src/features/library/origin.ts";
import type * as NavigationModule from "../src/features/studio/use-studio-navigation.ts";
import {
  isTrainingRunActive,
  shouldShowTrainingView,
  useTrainingRuntimeStore,
} from "../src/features/training/stores/training-runtime-store.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Slot = { value?: unknown; deps?: unknown[]; cleanup?: unknown };

let slots: Slot[] = [];
let cursor = 0;
let pendingEffects: Array<() => void> = [];
let dirty = false;

const react = {
  useState: (init: unknown) => {
    const slot = (slots[cursor++] ??= {
      value: typeof init === "function" ? init() : init,
    });
    return [
      slot.value,
      (next: unknown) => {
        slot.value = next;
        dirty = true;
      },
    ];
  },
  useRef: (init: unknown) =>
    (slots[cursor++] ??= { value: { current: init } }).value,
  useCallback: (fn: unknown) => fn,
  useEffect: (effect: () => unknown, deps: unknown[]) => {
    const slot = (slots[cursor++] ??= {});
    if (slot.deps && deps.every((dep, i) => Object.is(dep, slot.deps?.[i])))
      return;
    slot.deps = deps;
    pendingEffects.push(() => {
      if (typeof slot.cleanup === "function") slot.cleanup();
      slot.cleanup = effect();
    });
  },
};

const trainingStub = {
  isTrainingRunActive,
  shouldShowTrainingView,
  useTrainingRuntimeStore: Object.assign(
    (selector: (state: unknown) => unknown) =>
      selector(useTrainingRuntimeStore.getState()),
    useTrainingRuntimeStore,
  ),
};

const { useStudioNavigation: renderStudioNavigation } = loadWithStubs<
  typeof NavigationModule
>(new URL("../src/features/studio/use-studio-navigation.ts", import.meta.url), {
  react,
  "zustand/react/shallow": { useShallow: (selector: unknown) => selector },
  "@/features/training": trainingStub,
});

const { useLibraryOrigin: libraryOrigin } = loadWithStubs<
  typeof LibraryOriginModule
>(new URL("../src/features/library/origin.ts", import.meta.url), {
  "@/features/audio/workflows": { isAudioWorkflowId: () => false },
  "@/features/training": trainingStub,
  "@tanstack/react-router": { useNavigate: () => () => Promise.resolve() },
});

function mountStudio() {
  const mine: Slot[] = [];
  slots = mine;
  const stopWatching = useTrainingRuntimeStore.subscribe(() => {
    dirty = true;
  });
  let view!: ReturnType<typeof renderStudioNavigation>;
  const render = () => {
    do {
      slots = mine;
      dirty = false;
      cursor = 0;
      pendingEffects = [];
      view = renderStudioNavigation();
      for (const run of pendingEffects) run();
    } while (dirty);
    return view;
  };
  render();
  return {
    render,
    get view() {
      return view;
    },
    unmount() {
      stopWatching();
      for (const slot of mine) {
        if (typeof slot?.cleanup === "function") slot.cleanup();
      }
    },
  };
}

function status(jobId: string, running: boolean) {
  useTrainingRuntimeStore.getState().applyStatus({
    job_id: jobId,
    phase: running ? "training" : "completed",
    is_training_running: running,
    eval_enabled: false,
    message: "",
    error: null,
  });
  assert.equal(
    isTrainingRunActive(useTrainingRuntimeStore.getState()),
    running,
  );
}

/** Status the store kept from an earlier visit to Studio. */
function lastSeenJob(jobId: string | null, running = false) {
  useTrainingRuntimeStore.getState().resetRuntime();
  if (jobId) status(jobId, running);
}

const openFromRecents = (runId: string) =>
  useTrainingRuntimeStore.getState().setSelectedHistoryRunId(runId);

const openFromLibrary = (runId: string) =>
  libraryOrigin()({ id: `model:${runId}`, runId } as never)?.open();

const visible = (studio: ReturnType<typeof mountStudio>) => {
  const view = studio.render();
  return [view.activeTab, view.selectedHistoryRunId];
};

for (const [entry, open] of [
  ["Recents", openFromRecents],
  ["Library", openFromLibrary],
] as const) {
  test(`opening the running job from ${entry} on another page mounts on the live run`, () => {
    lastSeenJob("job-live", true);
    open("job-live");
    const studio = mountStudio();
    assert.deepEqual(visible(studio), ["current-run", "job-live"]);
    status("job-live", true);
    assert.deepEqual(visible(studio), ["current-run", "job-live"]);
    studio.unmount();
  });
}

test("Recents, Library and the History grid open the running job on the live run while the Train page is open", () => {
  lastSeenJob("job-live", true);
  const studio = mountStudio();
  assert.equal(studio.render().activeTab, "current-run");

  for (const open of [
    openFromRecents,
    openFromLibrary,
    (runId: string) => studio.view.handleHistoryRunSelected(runId),
  ]) {
    studio.view.handleTabChange("history");
    assert.deepEqual(visible(studio), ["history", null]);

    open("job-old");
    assert.deepEqual(visible(studio), ["history", "job-old"]);

    open("job-live");
    assert.deepEqual(visible(studio), ["current-run", "job-live"]);
    assert.equal(useTrainingRuntimeStore.getState().currentRunViewActive, true);
  }
  studio.unmount();
});

test("a run opened before any status arrives moves to the live run once status says it is training", () => {
  lastSeenJob(null);
  openFromRecents("job-live");
  const studio = mountStudio();
  assert.deepEqual(visible(studio), ["history", "job-live"]);
  status("job-live", true);
  assert.deepEqual(visible(studio), ["current-run", "job-live"]);
  studio.unmount();
});

test("finished runs open as past runs", () => {
  lastSeenJob("job-live", true);
  openFromRecents("job-old");
  const studio = mountStudio();
  assert.deepEqual(visible(studio), ["history", "job-old"]);
  status("job-live", true);
  assert.deepEqual(visible(studio), ["history", "job-old"]);
  studio.unmount();

  lastSeenJob("job-live", false);
  openFromRecents("job-live");
  const finished = mountStudio();
  assert.deepEqual(visible(finished), ["history", "job-live"]);
  finished.unmount();
});

test("the live run turns into its own past run when training ends", () => {
  lastSeenJob("job-live", true);
  openFromRecents("job-live");
  const studio = mountStudio();
  assert.deepEqual(visible(studio), ["current-run", "job-live"]);
  status("job-live", false);
  assert.deepEqual(visible(studio), ["history", "job-live"]);
  studio.unmount();
});

test("a run replaced by another job shows its own history, never the other job's live view, whatever order status arrives in", () => {
  const responses: Array<[string, boolean]> = [
    ["job-old", true],
    ["job-live", true],
    ["job-old", false],
  ];
  const orders = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
  ];
  for (const order of orders) {
    lastSeenJob("job-old", true);
    const studio = mountStudio();
    openFromRecents("job-old");
    assert.deepEqual(visible(studio), ["current-run", "job-old"]);
    for (const index of order) {
      status(...responses[index]);
      const [tab, selected] = visible(studio);
      assert.equal(selected, "job-old", `order ${order}`);
      if (tab === "current-run") {
        assert.equal(useTrainingRuntimeStore.getState().jobId, "job-old");
      }
    }
    status("job-live", true);
    assert.deepEqual(visible(studio), ["history", "job-old"]);
    studio.unmount();
  }
});

test("Back and the History tab open the History grid and keep it through the next poll", () => {
  lastSeenJob("job-live", true);
  const studio = mountStudio();

  openFromRecents("job-old");
  assert.deepEqual(visible(studio), ["history", "job-old"]);
  studio.view.clearHistorySelection();
  assert.deepEqual(visible(studio), ["history", null]);
  status("job-live", true);
  assert.deepEqual(visible(studio), ["history", null]);

  openFromRecents("job-live");
  assert.deepEqual(visible(studio), ["current-run", "job-live"]);
  studio.view.handleTabChange("history");
  assert.deepEqual(visible(studio), ["history", null]);
  status("job-live", true);
  assert.deepEqual(visible(studio), ["history", null]);

  openFromRecents("job-live");
  status("job-live", false);
  assert.deepEqual(visible(studio), ["history", "job-live"]);
  studio.view.clearHistorySelection();
  assert.deepEqual(visible(studio), ["history", null]);
  studio.unmount();
});

for (const [label, next] of [
  ["nothing", null],
  ["another job", "job-next"],
] as const) {
  test(`deleting a run opened live, after it finished, returns to the History grid while ${label} is training`, () => {
    lastSeenJob("job-live", true);
    const studio = mountStudio();
    openFromRecents("job-live");
    assert.deepEqual(visible(studio), ["current-run", "job-live"]);
    status("job-live", false);
    if (next) status(next, true);
    assert.deepEqual(visible(studio), ["history", "job-live"]);
    // The sidebar clears the selection after a delete.
    useTrainingRuntimeStore.getState().setSelectedHistoryRunId(null);
    assert.deepEqual(visible(studio), ["history", null]);
    studio.unmount();
  });
}

test("the Train page fading out after a route change does not clear the run opened from Recents", () => {
  lastSeenJob("job-live", true);
  openFromRecents("job-live");
  const fading = mountStudio();
  const studio = mountStudio();
  fading.unmount();
  assert.deepEqual(visible(studio), ["current-run", "job-live"]);
  studio.unmount();
  assert.equal(useTrainingRuntimeStore.getState().selectedHistoryRunId, null);
});
