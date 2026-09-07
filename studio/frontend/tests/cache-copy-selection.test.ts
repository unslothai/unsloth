// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { modelIdsMatch } from "../src/features/hub/lib/model-identity.ts";
import {
  modelIdsMatchForPicker,
  soleQuantRowState,
} from "../src/features/model-picker/components/model-selector/row-identity.ts";

test("Hub carries the chat loader's physical target into resident and loading state", () => {
  const source = readFileSync(
    new URL("../src/features/hub/hub-page.tsx", import.meta.url),
    "utf8",
  );
  const start = source.indexOf("const activeCheckpoint =");
  assert.ok(start >= 0);
  const resident = new Function(
    "checkpoint",
    "residentCheckpoint",
    "activeLoadId",
    "isExternalModelId",
    `${source.slice(start, source.indexOf(";", start) + 1)} return activeCheckpoint;`,
  );
  const repo = "Org/Model";
  const path = "/old/hub/models--Org--Model/snapshots/abc";
  assert.equal(
    resident(repo, repo, path, () => false),
    path,
  );
  assert.equal(
    resident(repo, repo, null, () => false),
    repo,
  );
  assert.equal(
    resident(repo, null, path, () => false),
    null,
  );
  const loadingStart = source.indexOf("const isLoadingThisModel = useMemo(");
  const loadingEnd = source.indexOf("\n\n", loadingStart);
  assert.ok(loadingStart >= 0 && loadingEnd > loadingStart);
  const loading = new Function(
    "useMemo",
    "loadingModel",
    "selectedModel",
    "modelIdsMatch",
    `${source.slice(loadingStart, loadingEnd)} return isLoadingThisModel;`,
  );
  for (const runId of [repo, path]) {
    assert.equal(
      loading(
        (fn: () => boolean) => fn(),
        { id: repo, loadId: path },
        { resource: { runId } },
        modelIdsMatch,
      ),
      runId === path,
    );
  }
});

test("the runtime loader and in-flight guard distinguish cache targets", () => {
  const source = readFileSync(
    new URL(
      "../src/features/chat/hooks/use-chat-model-runtime.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const start = source.indexOf("if (!forceReload &&");
  assert.ok(start >= 0);
  const condition = source.slice(start + 4, source.indexOf(") {", start));
  const skip = new Function(
    "forceReload",
    "modelId",
    "params",
    "ggufVariant",
    "currentVariant",
    "selection",
    "loadPath",
    "useChatRuntimeStore",
    "modelIdsMatchForPicker",
    `return ${condition};`,
  );
  const inFlightStart = source.indexOf("const loadingSamePick =");
  const declaration = source.slice(
    inFlightStart,
    source.indexOf(";", inFlightStart) + 1,
  );
  const loadingSame = new Function(
    "inFlightLoad",
    "modelId",
    "ggufVariant",
    "nativePathToken",
    "loadPath",
    "modelIdsMatchForPicker",
    `${declaration} return loadingSamePick;`,
  );
  for (const loadPath of ["Org/Model", "/old/snapshot"]) {
    const same = loadPath === "Org/Model";
    assert.equal(
      skip(
        false,
        "Org/Model",
        { checkpoint: "Org/Model" },
        "Q8_0",
        "Q8_0",
        { loadId: loadPath },
        loadPath,
        { getState: () => ({ activeLoadId: null }) },
        modelIdsMatchForPicker,
      ),
      same,
    );
    assert.equal(
      loadingSame(
        {
          id: "Org/Model",
          loadId: "Org/Model",
          ggufVariant: "Q8_0",
          nativePathToken: null,
        },
        "Org/Model",
        "Q8_0",
        null,
        loadPath,
        modelIdsMatchForPicker,
      ),
      same,
    );
  }
});

test("finishing a load cannot clear another copy's in-flight ownership", () => {
  const source = readFileSync(
    new URL(
      "../src/features/chat/stores/chat-runtime-store.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const start = source.indexOf("clearLoadingModelPick: (expected) =>");
  assert.ok(start >= 0);
  const end = source.indexOf("  setModelRequiresTrustRemoteCode:", start);
  const expression = source
    .slice(start + "clearLoadingModelPick: ".length, end)
    .trim()
    .replace(/,$/, "");
  const current = {
    id: "Org/Model",
    loadId: "/old/snapshot",
    ggufVariant: "Q8_0",
    nativePathToken: null,
  };
  let state: { loadingModelPick: typeof current | null } = {
    loadingModelPick: current,
  };
  const clear = new Function("set", `return ${expression};`)(
    (update: (value: typeof state) => typeof state) => {
      state = update(state);
    },
  );
  clear({ ...current, loadId: "Org/Model" });
  assert.equal(state.loadingModelPick, current);
  clear(current);
  assert.equal(state.loadingModelPick, null);
});

test("chat's actual selection guard switches between copies of the same quant", () => {
  const source = readFileSync(
    new URL("../src/features/chat/chat-page.tsx", import.meta.url),
    "utf8",
  );
  const handler = source.indexOf("const handleCheckpointChange =");
  const start = source.indexOf("const isSameLoadedModel =", handler);
  assert.ok(start > handler);
  const declaration = source.slice(start, source.indexOf(";", start) + 1);
  const guard = new Function(
    "value",
    "currentCheckpoint",
    "currentVariant",
    "meta",
    "store",
    "modelIdsMatchForPicker",
    `${declaration} return isSameLoadedModel;`,
  );
  for (const [current, target, same] of [
    [null, "/old/snapshot", false],
    ["/old/snapshot", "Org/Model", false],
    ["/old/snapshot", "/old/snapshot", true],
    ["C:\\Cache\\Snapshot", "c:/cache/snapshot", true],
    ["/old/snapshot", undefined, true],
  ] as const) {
    assert.equal(
      guard(
        "Org/Model",
        "Org/Model",
        "Q8_0",
        { ggufVariant: "Q8_0", loadId: target },
        { activeLoadId: current },
        modelIdsMatchForPicker,
      ),
      same,
    );
  }
});

test("only the loaded copy of a duplicated quant is selected and marked loaded", () => {
  for (const activeLoadId of [null, "/old/snapshot"]) {
    const states = ["Org/Model", "/old/snapshot"].map((loadId) =>
      soleQuantRowState({
        pickerValue: "Org/Model",
        repoId: "Org/Model",
        quant: "Q8_0",
        loadedModelId: "Org/Model",
        activeGgufVariant: "Q8_0",
        loadId,
        activeLoadId,
      }),
    );
    assert.deepEqual(
      states,
      activeLoadId
        ? [
            { selected: false, loaded: false },
            { selected: true, loaded: true },
          ]
        : [
            { selected: true, loaded: true },
            { selected: false, loaded: false },
          ],
    );
  }
});
