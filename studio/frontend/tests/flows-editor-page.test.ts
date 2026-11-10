// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The editor page around the canvas: leaving, read-only flows, the toolbar and shortcuts.

import assert from "node:assert/strict";
import test, { mock } from "node:test";

import { installLocalStorageFake, registerBundlerResolver } from "./helpers/kit.ts";
import {
  elementsOf,
  loadWithStubs,
  stubEveryImport,
  stubJsxRuntime,
  type StubElement,
} from "./helpers/module-stubs.ts";

registerBundlerResolver();

const flows = (rel: string) => new URL(`../src/features/flows/${rel}`, import.meta.url);

test("leaving the editor waits for a successful save, or an explicit Leave anyway", async () => {
  const warnings: { title: string; options: Record<string, () => void> }[] = [];
  let save: () => Promise<boolean | "stuck"> = async () => true;
  const page = loadWithStubs<{ stayUnlessSaved: () => Promise<boolean> }>(
    flows("editor/flows-editor-page.tsx"),
    stubEveryImport({
      "./use-autosave": { flushSave: () => save() },
      sonner: {
        toast: {
          warning: (title: string, options: { action: { onClick: () => void }; onDismiss: () => void }) =>
            warnings.push({ title, options: { leave: options.action.onClick, dismiss: options.onDismiss } }),
        },
      },
    }),
  );
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    assert.equal(await page.stayUnlessSaved(), false, "a saved flow leaves at once");
    assert.equal(warnings.length, 0);

    save = async () => false;
    const failed = page.stayUnlessSaved();
    await Promise.resolve();
    await Promise.resolve();
    assert.equal(warnings.at(-1)?.title, "This flow isn't saved");
    warnings.at(-1)?.options.leave();
    assert.equal(await failed, false, "Leave anyway leaves");

    const dismissed = page.stayUnlessSaved();
    await Promise.resolve();
    await Promise.resolve();
    warnings.at(-1)?.options.dismiss();
    assert.equal(await dismissed, true, "dismissing the warning stays");

    save = () => new Promise(() => undefined);
    const stuck = page.stayUnlessSaved();
    mock.timers.tick(10_000);
    await Promise.resolve();
    await Promise.resolve();
    assert.equal(warnings.at(-1)?.title, "Still saving this flow");
    warnings.at(-1)?.options.dismiss();
    assert.equal(await stuck, true);
  } finally {
    mock.timers.reset();
  }
});

test("a read-only flow's toolbar cannot edit the document", () => {
  const ui = { saveState: "saved", snapToGrid: false, wireStyle: "bus", tidyDirection: "LR" };
  const docState = { meta: { name: "F", kind: "pipeline" }, past: [1], future: [1] };
  const bar = loadWithStubs<{
    Toolbar: (p: { readOnly?: boolean }) => unknown;
  }>(
    flows("editor/toolbar.tsx"),
    stubEveryImport({
      react: { memo: <T,>(f: T) => f },
      "react/jsx-runtime": stubJsxRuntime(),
      "../stores/doc-store": { useDocStore: (pick: (s: unknown) => unknown) => pick(docState) },
      "../stores/ui-store": { useUiStore: (pick: (s: unknown) => unknown) => pick(ui) },
      "./placement": { usePlacement: (pick: (s: unknown) => unknown) => pick({ armed: null }) },
      "@xyflow/react": { useReactFlow: () => ({}) },
    }),
  );
  const labels = ["Undo", "Redo", "Add group", "Add note", "Add reroute"];
  const editing = (tree: unknown) => {
    const all = elementsOf(tree);
    const buttons = all.filter((e) => typeof e.props.label === "string" && labels.some((l) => (e.props.label as string).startsWith(l)));
    const tidy = all.find((e) => typeof e.type === "function" && e.type.name === "TidyButton") as StubElement;
    const tidyHalves = elementsOf((tidy.type as (p: unknown) => unknown)(tidy.props)).filter((e) => "disabled" in e.props);
    const name = all.find((e) => e.type === "input") as StubElement;
    return { buttons, tidyHalves, name };
  };
  const locked = editing(bar.Toolbar({ readOnly: true }));
  assert.equal(locked.buttons.length, labels.length);
  for (const b of locked.buttons) assert.equal(b.props.disabled, true, String(b.props.label));
  assert.equal(locked.tidyHalves.length, 2, "both halves of the Tidy button");
  for (const half of locked.tidyHalves) assert.equal(half.props.disabled, true);
  assert.equal(locked.name.props.readOnly, true);

  const open = editing(bar.Toolbar({ readOnly: false }));
  for (const b of open.buttons) assert.ok(!b.props.disabled, String(b.props.label));
  for (const half of open.tidyHalves) assert.equal(half.props.disabled, false);
  assert.equal(open.name.props.readOnly, false);
});

test("a block from a newer revision opens the flow read-only", async () => {
  const { needsNewerStudio, useCatalogStore } = await import(
    "../src/features/flows/stores/catalog-store.ts"
  );
  useCatalogStore.setState({
    specs: { "text.template": { type: "text.template", v: 1 } as never },
  });
  assert.equal(needsNewerStudio({ type: "text.template", v: 1 }), false);
  assert.equal(needsNewerStudio({ type: "text.template" }), false);
  assert.equal(needsNewerStudio({ type: "text.template", v: 2 }), true);
  assert.equal(needsNewerStudio({ type: "made.up" }), true);
});

test("Escape closes the inspector as well as clearing the selection, or just Quick Add when open", async () => {
  const { fireWindowEvent } = installLocalStorageFake();
  const uiStore = await import("../src/features/flows/stores/ui-store.ts");
  const keys = loadWithStubs<{ useEditorShortcuts: (active: boolean) => void }>(
    new URL("../src/features/flows/editor/use-editor-shortcuts.ts", import.meta.url),
    stubEveryImport({
      react: { useEffect: (fn: () => void) => fn() },
      "@xyflow/react": { useReactFlow: () => ({}) },
      "../stores/ui-store": uiStore,
      "../stores/doc-store": await import("../src/features/flows/stores/doc-store.ts"),
      "./history-keys": await import("../src/features/flows/editor/history-keys.ts"),
    }),
  );
  keys.useEditorShortcuts(true);
  const ui = () => uiStore.useUiStore.getState();
  const escape = () => fireWindowEvent("keydown", { key: "Escape", target: null, preventDefault() {} });

  ui().setQuickAdd({ x: 0, y: 0 } as never);
  ui().setSelection(["a"], []);
  ui().openInspector("a");
  escape();
  assert.equal(ui().quickAdd, null);
  assert.deepEqual(ui().selectedNodes, ["a"], "the first Escape only closes Quick Add");
  escape();
  assert.deepEqual(ui().selectedNodes, []);
  assert.equal(ui().inspectorNodeId, null);
});

test("every fit stops at a zoom where card text still shows", async () => {
  const { FIT_MIN_ZOOM } = await import("../src/features/flows/editor/canvas-rules.ts");
  assert.ok(FIT_MIN_ZOOM > 0.45);
});
