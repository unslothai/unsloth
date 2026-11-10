// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Fine-tune's base model and the Hugging Face Dataset node use the Train page's own pickers.

import assert from "node:assert/strict";
import test from "node:test";

import { byTestId, flushMicrotasks } from "./helpers/flows.ts";
import { fakeReact } from "./helpers/fake-react.ts";
import { elementsOf, loadWithStubs, type StubElement, stubEveryImport, stubJsxRuntime } from "./helpers/module-stubs.ts";
import { hfDatasetChoice, pickedSplit, shownSplit } from "../src/features/flows/editor/dataset-choice.ts";

test("the node's params map onto the picker's values, with an unset split running as train", () => {
  assert.deepEqual(hfDatasetChoice({}), { repo: null, subset: null, split: "train" });
  assert.deepEqual(
    hfDatasetChoice({ repo: " openai/gsm8k ", subset: "main", split: "test" }),
    { repo: "openai/gsm8k", subset: "main", split: "test" },
  );
  assert.deepEqual(hfDatasetChoice({ repo: "", subset: "", split: "" }), { repo: null, subset: null, split: null });
});

test("a saved split expression survives the dropdown, which lists only split names", () => {
  assert.equal(shownSplit("train[:10%]"), "train");
  assert.equal(shownSplit("train+test"), "train");
  assert.equal(shownSplit(null), null);
  // Auto-select of the split it shows keeps the slice; picking another split replaces it.
  assert.equal(pickedSplit("train[:10%]", "train"), "train[:10%]");
  assert.equal(pickedSplit("train[:10%]", "test"), "test");
  assert.equal(pickedSplit("train", null), "");
  // `v` on the way to `validation` is no split name, so auto-select would snap it back to train.
  assert.notEqual(shownSplit("v"), pickedSplit("v", "train"));
});

// ---- The picker components, rendered with deterministic hooks (no DOM). ----

let active = fakeReact();
const react = new Proxy({}, {
  get: (_t, key) => (key === "memo" ? (c: unknown) => c : key === "useId" ? () => "id" : active.react[key as keyof typeof active.react]),
});
const src = (rel: string) => new URL(`../src/features/flows/${rel}`, import.meta.url);
const named = (els: StubElement[], name: string) =>
  els.find((e) => (e.type as { name?: string }).name === name) as StubElement;
const call = (el: StubElement, prop: string, ...args: unknown[]) =>
  (el.props[prop] as (...a: unknown[]) => unknown)(...args);

test("Fine-tune's base model sends the backend a plain repo string plus the picked On device copy", () => {
  // Backend contract: the run reads `base_model` and `base_model_cache: { known_cached, local_path }`.
  const { FlowTrainModelPicker } = loadWithStubs<{ FlowTrainModelPicker: (p: unknown) => StubElement }>(
    src("editor/model-picker-field.tsx"),
    stubEveryImport({ react, "react/jsx-runtime": stubJsxRuntime() }),
  );
  const { ParamField } = loadWithStubs<{ ParamField: (p: unknown) => StubElement }>(
    src("editor/param-fields.tsx"),
    stubEveryImport({ react, "react/jsx-runtime": stubJsxRuntime(), "./model-picker-field": { FlowTrainModelPicker } }),
  );
  const patches: unknown[] = [];
  const plain: unknown[] = [];
  const field = (onPatch?: (p: unknown) => void) =>
    ParamField({ spec: { id: "base_model", kind: "text" }, value: "unsloth/a", onChange: (v: unknown) => plain.push(v), onPatch });
  const picker = field((p) => patches.push(p));
  assert.equal(picker.type, FlowTrainModelPicker);
  assert.equal(picker.props.value, "unsloth/a");
  const selector = FlowTrainModelPicker(picker.props);
  call(selector, "onValueChange", "unsloth/b", { knownCached: true, localPath: "/models/b" });
  call(selector, "onValueChange", "unsloth/c", null);
  assert.deepEqual(patches, [
    { base_model: "unsloth/b", base_model_cache: { known_cached: true, local_path: "/models/b" } },
    { base_model: "unsloth/c", base_model_cache: null },
  ]);
  // Without onPatch (e.g. a node with no inspector patching) the plain repo string still lands.
  call(FlowTrainModelPicker(field().props), "onValueChange", "unsloth/d", null);
  assert.deepEqual(plain, ["unsloth/d"]);
});

for (const [source, fetches] of [["/api/flows", 2], ["/api/models", 1]] as const)
  test(`a picker over ${source} fetches its options ${fetches === 2 ? "fresh on every open" : "once and reuses them"}`, async () => {
    const fetched: string[] = [];
    const { ParamField } = loadWithStubs<{ ParamField: (p: unknown) => StubElement }>(
      src("editor/param-fields.tsx"),
      stubEveryImport({
        react,
        "react/jsx-runtime": stubJsxRuntime(),
        "@/features/auth": {
          getAuthSessionEpoch: () => 1,
          authFetch: async (url: string) => {
            fetched.push(url);
            return { ok: true, json: async () => ["a", "b"] };
          },
        },
        "./remote-options": await import("../src/features/flows/editor/remote-options.ts"),
      }),
    );
    const spec = { id: "flow", kind: "text", options_from: source };
    for (let open = 0; open < 2; open++) {
      const hooks = fakeReact();
      active = hooks;
      const field = ParamField({ spec, value: "", onChange: () => undefined });
      const render = () => hooks.render(field.type as (p: unknown) => unknown, field.props);
      render();
      await flushMicrotasks();
      const input = named(elementsOf(render()), "Input");
      assert.equal(input.props.placeholder, "2 choices");
      hooks.unmount();
    }
    assert.equal(fetched.length, fetches);
  });

function datasetFields(params: Record<string, unknown>) {
  const patches: Record<string, unknown>[] = [];
  const { FlowHfDatasetFields } = loadWithStubs<{ FlowHfDatasetFields: (p: unknown) => unknown }>(
    src("editor/dataset-picker-field.tsx"),
    stubEveryImport({
      react,
      "react/jsx-runtime": stubJsxRuntime(),
      "./dataset-choice": { hfDatasetChoice, pickedSplit, shownSplit },
      "@/features/hub": { hfApiToken: () => null, useHfTokenStore: () => null, useOnlineStatus: () => true },
    }),
  );
  const hooks = fakeReact();
  const props = { params, onChange: (p: Record<string, unknown>) => patches.push(p) };
  active = hooks;
  const els = elementsOf(hooks.render(FlowHfDatasetFields, props));
  const splitInput = byTestId(els, "flows-hf-split-expression") as StubElement;
  return { patches, els, splitInput, dropdowns: named(els, "HfDatasetSubsetSplitSelectors") };
}

test("the Hugging Face Dataset node: a new repo clears the subset, and the eval split is hidden", () => {
  const saved = { repo: "openai/gsm8k", subset: "main", split: "train[:10%]" };
  const { patches, els, dropdowns } = datasetFields(saved);
  call(named(els, "DatasetSelector"), "onValueChange", "org/other");
  assert.deepEqual(patches, [{ repo: "org/other", subset: "" }]);
  assert.equal(dropdowns.props.hideEvalSplit, true);
  assert.equal(dropdowns.props.datasetName, "openai/gsm8k");
  assert.equal(dropdowns.props.datasetSplit, "train", "the dropdown shows the split name of the saved expression");
  call(dropdowns, "setDatasetSubset", null);
  assert.deepEqual(patches.at(-1), { subset: "" });
});

test("a saved split expression survives the dropdown's auto-select, and the full expression stays editable", () => {
  const { patches, dropdowns, splitInput } = datasetFields({ repo: "openai/gsm8k", split: "train[:10%]" });
  call(dropdowns, "setDatasetSplit", "train");
  call(dropdowns, "setDatasetSplit", "test");
  assert.deepEqual(patches, [{ split: "train[:10%]" }, { split: "test" }]);
  assert.equal(splitInput.props.value, "train[:10%]");
  call(splitInput, "onChange", { target: { value: "train+test" } });
  assert.deepEqual(patches.at(-1), { split: "train+test" });
});

test("typing a split expression is not replaced by the dropdown's auto-select mid-word", () => {
  const { patches, dropdowns, splitInput } = datasetFields({ repo: "openai/gsm8k", split: "v" });
  call(splitInput, "onFocus");
  call(dropdowns, "setDatasetSplit", "train");
  assert.deepEqual(patches, [], "no snap back to train while typing");
  call(splitInput, "onBlur");
  call(dropdowns, "setDatasetSplit", "train");
  assert.deepEqual(patches, [{ split: "train" }], "after leaving the field the dropdown may pick again");
});
