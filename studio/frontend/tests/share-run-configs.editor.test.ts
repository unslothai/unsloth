// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";
import type { SharedRunConfigControls as Controls } from "../src/features/model-picker/sharing/config-controls.tsx";
import type { SharedRunConfigReview as Review } from "../src/features/model-picker/sharing/config-review.tsx";
import * as events from "../src/features/model-picker/sharing/editor-events.ts";
import type { SharedRunConfigLinkEditor as LinkEditor } from "../src/features/model-picker/sharing/link-editor.tsx";
import type { SharedRunConfigLinkHandler as LinkHandler } from "../src/features/model-picker/sharing/link-handler.tsx";
import type { ShareRunConfigDialog as ShareDialog } from "../src/features/model-picker/sharing/share-dialog.tsx";
import {
  installLocalStorageFake,
  readText,
  registerBundlerResolver,
} from "./helpers/kit.ts";
import {
  type StubElement,
  loadWithStubs,
  stubJsxRuntime,
} from "./helpers/module-stubs.ts";

registerBundlerResolver();
installLocalStorageFake();
const fields = await import("../src/features/model-picker/sharing/fields.ts");
const sharedArgs = await import(
  "../src/features/model-picker/sharing/extra-args.ts"
);
const links = await import("./helpers/sharing-links.ts");
const { DEFAULT_PER_MODEL_CONFIG } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { modelConfigDraftKey } = await import(
  "../src/features/model-picker/model-config/model-config-draft.ts"
);
const { createRunConfigInbox, mergeSharedRunConfig } = await import(
  "../src/features/model-picker/sharing/inbox.ts"
);
const targetModule = await import("./helpers/sharing-target.ts");
const { reconcileGpuSelection } = await import("../src/hooks/gpu-selection.ts");

test("unresolved shared GGUFs disable Load and skip model metadata requests", () => {
  const source = ts.createSourceFile(
    "model-config-page.tsx",
    readText("../src/features/model-picker/components/model-config-page.tsx"),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  const declarations = new Map<string, string>();
  let disabled: string | undefined;
  const visit = (node: ts.Node) => {
    if (ts.isVariableDeclaration(node)) {
      declarations.set(node.name.getText(source), node.getText(source));
    }
    if (ts.isJsxOpeningElement(node)) {
      const props = node.attributes.properties.filter(ts.isJsxAttribute);
      if (
        props.some(
          (prop) =>
            prop.name.getText(source) === "onClick" &&
            prop.initializer?.getText(source) === "{handleRun}",
        )
      ) {
        const value = props.find(
          (prop) => prop.name.getText(source) === "disabled",
        )?.initializer;
        assert.ok(value && ts.isJsxExpression(value) && value.expression);
        disabled = value.expression.getText(source);
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(source);
  assert.ok(disabled);
  const body = ["sharedVariantUnresolved", "contextFetchKey", "handleRun"]
    .map((name) => {
      const declaration = declarations.get(name);
      assert.ok(declaration);
      return `const ${declaration};`;
    })
    .join("\n");
  const evaluate = new Function(
    "target",
    "isRunConfigVariantUnresolved",
    ts.transpile(
      `${body}\nreturn { contextFetchKey, disabled: ${disabled}, run: handleRun };`,
    ),
  );
  for (const ggufVariant of [undefined, "model-Q4_K_M.gguf"]) {
    const state = evaluate(
      { isGguf: true, ggufVariant, meta: { source: "hub" } },
      targetModule.isRunConfigVariantUnresolved,
    );
    assert.equal(state.contextFetchKey, null);
    assert.equal(state.disabled, true);
    assert.equal(state.run(), undefined);
  }
});

function elements(node: unknown): StubElement[] {
  if (Array.isArray(node)) return node.flatMap(elements);
  if (!node || typeof node !== "object" || !("props" in node)) return [];
  const element = node as StubElement;
  return [element, ...elements(element.props.children)];
}

function text(node: unknown): string {
  if (Array.isArray(node)) return node.map(text).join("");
  if (node && typeof node === "object" && "props" in node)
    return text((node as StubElement).props.children);
  return typeof node === "string" || typeof node === "number"
    ? String(node)
    : "";
}

test("sharing empty arguments and templates preserves recipient overrides unless explicitly selected", () => {
  const checkbox = Symbol("checkbox");
  const textarea = Symbol("textarea");
  let states: unknown[] = [];
  let cursor = 0;
  const { ShareRunConfigDialog } = loadWithStubs<{
    ShareRunConfigDialog: typeof ShareDialog;
  }>(
    new URL(
      "../src/features/model-picker/sharing/share-dialog.tsx",
      import.meta.url,
    ),
    {
      "react/jsx-runtime": stubJsxRuntime(),
      react: {
        useId: () => "share",
        useMemo: (create: () => unknown) => create(),
        useState: <T>(initial: T | (() => T)) => {
          const index = cursor++;
          if (index >= states.length) {
            states.push(
              typeof initial === "function" ? (initial as () => T)() : initial,
            );
          }
          return [
            states[index],
            (update: T | ((current: T) => T)) => {
              states[index] =
                typeof update === "function"
                  ? (update as (current: T) => T)(states[index] as T)
                  : update;
            },
          ];
        },
      },
      "@/components/ui/button": { Button: "button" },
      "@/components/ui/checkbox": { Checkbox: checkbox },
      "@/components/ui/dialog": {
        Dialog: "dialog",
        DialogContent: "content",
        DialogDescription: "description",
        DialogHeader: "header",
        DialogTitle: "title",
      },
      "@/components/ui/select": {
        Select: "select",
        SelectContent: "options",
        SelectItem: "option",
        SelectTrigger: "trigger",
        SelectValue: "value",
      },
      "@/components/ui/textarea": { Textarea: textarea },
      "@/lib/api-base": { isTauri: true },
      "@/lib/copy-to-clipboard": {},
      "@/lib/toast": {},
      "../model-config/per-model-config": { DEFAULT_PER_MODEL_CONFIG },
      "./extra-args": sharedArgs,
      "./fields": fields,
      "./links": links,
    },
  );
  const recipient = {
    ...DEFAULT_PER_MODEL_CONFIG,
    llamaExtraArgs: ["--threads", "8"],
    chatTemplateOverride: "{{ messages }}",
  };
  for (const llamaExtraArgs of [undefined, null, [], ["--threads", "4"]]) {
    states = [];
    const config = {
      ...DEFAULT_PER_MODEL_CONFIG,
      llamaExtraArgs,
      chatTemplateOverride: "",
    };
    const render = () => {
      cursor = 0;
      return elements(
        ShareRunConfigDialog({
          target: {
            id: "owner/Model-GGUF",
            displayName: "Model",
            ggufVariant: "Q4_K_M",
            isGguf: true,
            apiLoadable: true,
            meta: { source: "hub", isLora: false },
          },
          config,
          onClose: () => undefined,
        }),
      );
    };
    const importedConfig = (tree: StubElement[]) => {
      const link = tree.find((element) => element.type === textarea)?.props
        .value;
      assert.equal(typeof link, "string");
      const parsed = links.parseRunConfigLink(link as string);
      assert.equal(parsed.kind, "valid");
      assert.ok(parsed.kind === "valid");
      return mergeSharedRunConfig(recipient, parsed.value.config, true);
    };
    const tree = render();
    const templateChoice = tree.find(
      (element) =>
        element.type === checkbox &&
        element.props.id === "share-chatTemplateOverride",
    );
    assert.ok(templateChoice);
    assert.equal(templateChoice.props.checked, false);
    assert.equal(
      text(
        tree.find(
          (element) => element.props.id === "share-chatTemplateOverride-detail",
        ),
      ),
      "Default",
    );
    assert.equal(
      importedConfig(tree).chatTemplateOverride,
      recipient.chatTemplateOverride,
    );
    (templateChoice.props.onCheckedChange as (checked: boolean) => void)(true);
    assert.equal(importedConfig(render()).chatTemplateOverride, "");
    const choice = tree.find(
      (element) =>
        element.type === checkbox &&
        element.props.id === "share-llamaExtraArgs",
    );
    const nonempty = (llamaExtraArgs?.length ?? 0) > 0;
    assert.deepEqual(
      importedConfig(tree).llamaExtraArgs,
      nonempty ? llamaExtraArgs : recipient.llamaExtraArgs,
    );
    if (llamaExtraArgs === undefined) {
      assert.equal(choice, undefined);
      continue;
    }
    assert.ok(choice);
    assert.equal(choice.props.checked, nonempty);
    if (!nonempty) {
      (choice.props.onCheckedChange as (checked: boolean) => void)(true);
      const selected = render();
      assert.equal(
        selected.find((element) => element.props.id === "share-llamaExtraArgs")
          ?.props.checked,
        true,
      );
      assert.ok(text(selected).includes("No extra arguments"));
      assert.deepEqual(importedConfig(selected).llamaExtraArgs, llamaExtraArgs);
    }
  }
});

test("Share opens and closes its dialog; dismissing a pending import gives feedback", async () => {
  const inbox = createRunConfigInbox();
  const target = {
    id: "owner/Model-GGUF",
    displayName: "Model",
    ggufVariant: "Q4_K_M",
    isGguf: true,
    apiLoadable: true,
    meta: { source: "hub" as const, isLora: false },
  };
  const key = modelConfigDraftKey(target.id, target.ggufVariant);
  inbox.submit({
    id: "pending",
    draftKey: key,
    value: { config: { nParallel: 3 } },
  });
  let sharing = false;
  let release: (() => void) | undefined;
  const notices: { message: string; id: string; description: string }[] = [];
  const dialog = Symbol("share dialog");
  const button = Symbol("button");
  const { SharedRunConfigControls } = loadWithStubs<{
    SharedRunConfigControls: typeof Controls;
  }>(
    new URL(
      "../src/features/model-picker/sharing/config-controls.tsx",
      import.meta.url,
    ),
    {
      "react/jsx-runtime": stubJsxRuntime(),
      react: {
        lazy: () => dialog,
        Suspense: Symbol("suspense"),
        useState: () => [
          sharing,
          (value: boolean) => {
            sharing = value;
          },
        ],
        useSyncExternalStore: (_subscribe: unknown, get: () => unknown) =>
          get(),
        useEffect: () => undefined,
        useLayoutEffect: (effect: () => (() => void) | undefined) => {
          release ??= effect();
        },
      },
      "@/components/ui/button": { Button: button },
      "@/lib/toast": {
        toast: {
          info: (
            message: string,
            options: { id: string; description: string },
          ) => notices.push({ message, ...options }),
        },
      },
      "../model-config/model-config-draft": {
        modelConfigDraftKey,
      },
      "./inbox": { runConfigInbox: inbox },
      "./import-config": { scheduleRunConfigImport: () => undefined },
    },
  );
  const props = {
    className: "h-9 rounded-full",
    target,
    config: DEFAULT_PER_MODEL_CONFIG,
    ready: true,
    hydrated: true,
    canImport: true,
    disabled: false,
    onImport: () => undefined,
  };
  const render = () => elements(SharedRunConfigControls(props));
  const initial = render();
  assert.equal(
    initial.some((element) => element.type === dialog),
    false,
  );
  const share = initial.find((element) => element.type === button);
  assert.ok(share);
  (share.props.onClick as () => void)();
  const opened = render();
  const shownDialog = opened.find((element) => element.type === dialog);
  assert.ok(shownDialog);
  (shownDialog.props.onClose as () => void)();
  assert.equal(
    render().some((element) => element.type === dialog),
    false,
  );
  assert.ok(release);
  assert.equal(notices.length, 0);
  release();
  await Promise.resolve();
  assert.equal(inbox.getSnapshot(), null);
  assert.equal(notices.length, 1);
  assert.equal(notices[0].id, "pending");
  assert.match(
    notices[0].description,
    /editor closed before the settings were imported/,
  );
});

test("edit cancellation includes contained controls and excludes portaled dialog controls", (t) => {
  class NodeFake extends EventTarget {
    children = new Set<NodeFake>();
    contains(node: NodeFake) {
      return node === this || this.children.has(node);
    }
  }
  const original = Object.getOwnPropertyDescriptor(globalThis, "Node");
  Object.defineProperty(globalThis, "Node", {
    value: NodeFake,
    configurable: true,
  });
  t.after(() => {
    if (original) Object.defineProperty(globalThis, "Node", original);
    else Reflect.deleteProperty(globalThis, "Node");
  });
  const editor = new NodeFake();
  const numericInput = new NodeFake();
  const portaledTextarea = new NodeFake();
  editor.children.add(numericInput);
  const currentTarget = editor as unknown as Node;
  assert.equal(
    events.isRunConfigEditorChange({ currentTarget, target: numericInput }),
    true,
  );
  assert.equal(
    events.isRunConfigEditorChange({ currentTarget, target: portaledTextarea }),
    false,
  );
  assert.equal(
    events.isRunConfigEditorChange({
      currentTarget,
      target: new EventTarget(),
    }),
    false,
  );
});

test("review renders field labels and full prompt/argument values as text without HTML injection", () => {
  const { SharedRunConfigReview } = loadWithStubs<{
    SharedRunConfigReview: typeof Review;
  }>(
    new URL(
      "../src/features/model-picker/sharing/config-review.tsx",
      import.meta.url,
    ),
    {
      "react/jsx-runtime": stubJsxRuntime(),
      "./fields": fields,
    },
  );
  assert.equal(
    SharedRunConfigReview({
      config: null,
      draftConfig: DEFAULT_PER_MODEL_CONFIG,
      currentConfig: DEFAULT_PER_MODEL_CONFIG,
    }),
    null,
  );
  const prompt = "<img src=x onerror=alert(1)>\nContinue after thinking";
  const args = ["--rope-scaling", "yarn"];
  const imported = {
    reasoningBudgetMessage: prompt,
    llamaExtraArgs: args,
    maxSeqLength: null,
  };
  const tree = SharedRunConfigReview({
    config: imported,
    draftConfig: { ...DEFAULT_PER_MODEL_CONFIG, ...imported },
    currentConfig: { ...DEFAULT_PER_MODEL_CONFIG, ...imported },
  });
  assert.ok(text(tree).includes("Settings changed by link (3)"));
  assert.ok(text(tree).includes("Reasoning budget message"));
  assert.ok(text(tree).includes(prompt));
  assert.ok(text(tree).includes("--rope-scaling yarn"));
  assert.ok(text(tree).includes("Default"));
  assert.ok(
    elements(tree).every(
      (element) => !("dangerouslySetInnerHTML" in element.props),
    ),
  );
  assert.ok(
    text(
      SharedRunConfigReview({
        config: {},
        draftConfig: DEFAULT_PER_MODEL_CONFIG,
        currentConfig: DEFAULT_PER_MODEL_CONFIG,
      }),
    ).includes("already match"),
  );
  for (const changed of [
    { reasoningBudgetMessage: "Edited through the other editor" },
    { llamaExtraArgs: ["--threads", "4"] },
    DEFAULT_PER_MODEL_CONFIG,
  ]) {
    assert.equal(
      SharedRunConfigReview({
        config: imported,
        draftConfig: { ...DEFAULT_PER_MODEL_CONFIG, ...imported, ...changed },
        currentConfig: { ...DEFAULT_PER_MODEL_CONFIG, ...imported, ...changed },
      }),
      null,
    );
  }
  assert.notEqual(
    SharedRunConfigReview({
      config: imported,
      draftConfig: { ...DEFAULT_PER_MODEL_CONFIG, ...imported },
      currentConfig: {
        ...DEFAULT_PER_MODEL_CONFIG,
        ...imported,
        llamaExtraArgs: [...args],
      },
    }),
    null,
  );
  for (const llamaExtraArgs of [null, []]) {
    const config = { llamaExtraArgs };
    const draftConfig = { ...DEFAULT_PER_MODEL_CONFIG, ...config };
    const tree = SharedRunConfigReview({
      config,
      draftConfig,
      currentConfig: draftConfig,
    });
    assert.ok(text(tree).includes("No extra arguments"));
  }
  const adjusted = SharedRunConfigReview({
    config: { llamaExtraArgs: args },
    draftConfig: { ...DEFAULT_PER_MODEL_CONFIG, llamaExtraArgs: args },
    currentConfig: { ...DEFAULT_PER_MODEL_CONFIG, llamaExtraArgs: [] },
  });
  assert.ok(text(adjusted).includes("No extra arguments"));
  assert.ok(text(adjusted).includes("Requested: --rope-scaling yarn"));
});

test("GPU reconciliation keeps imported settings visible and explains removed or filtered GPU choices", () => {
  const { SharedRunConfigReview } = loadWithStubs<{
    SharedRunConfigReview: typeof Review;
  }>(
    new URL(
      "../src/features/model-picker/sharing/config-review.tsx",
      import.meta.url,
    ),
    {
      "react/jsx-runtime": stubJsxRuntime(),
      "./fields": fields,
    },
  );
  const imported = {
    selectedGpuIds: [0, 1],
    selectedGpuIndexKind: "physical" as const,
    nParallel: 3,
  };
  const draftConfig = { ...DEFAULT_PER_MODEL_CONFIG, ...imported };
  for (const [indexKind, deviceIds] of [
    [null, []],
    ["vulkan", [0, 1]],
    ["physical", [0, 2]],
  ] as const) {
    const reconciled = reconcileGpuSelection(
      imported.selectedGpuIds,
      imported.selectedGpuIndexKind,
      indexKind,
      [...deviceIds],
    );
    const currentConfig = {
      ...draftConfig,
      selectedGpuIds: reconciled.ids ?? undefined,
      selectedGpuIndexKind:
        reconciled.ids === null ? undefined : reconciled.indexKind,
    };
    const tree = SharedRunConfigReview({
      config: imported,
      draftConfig,
      currentConfig,
    });
    assert.ok(text(tree).includes("Settings changed by link (3)"));
    const values = elements(tree)
      .filter((element) => element.type === "dd")
      .map(text);
    assert.ok(values.includes("3"));
    assert.ok(
      values.some((value) =>
        value.startsWith(reconciled.ids ? "[0]" : "Default"),
      ),
    );
    assert.ok(text(tree).includes("Requested: [0,1]"));
    assert.ok(text(tree).includes("unsupported values will not be used"));
  }
});

test("settings-only chooser keeps the import while accepting recipient-local models", () => {
  const inbox = createRunConfigInbox();
  const dialog = Symbol("dialog");
  const input = Symbol("input");
  const button = Symbol("button");
  let modelInput = "";
  const runtime = { params: { checkpoint: "" }, settingsHydrated: true };
  const { SharedRunConfigLinkEditor } = loadWithStubs<{
    SharedRunConfigLinkEditor: typeof LinkEditor;
  }>(
    new URL(
      "../src/features/model-picker/sharing/link-editor.tsx",
      import.meta.url,
    ),
    {
      "react/jsx-runtime": stubJsxRuntime(),
      react: {
        useState: () => [
          modelInput,
          (value: string) => {
            modelInput = value;
          },
        ],
        useRef: () => ({ current: null }),
        useEffect: () => undefined,
      },
      "@/components/ui/button": { Button: button },
      "@/components/ui/input": { Input: input },
      "@/components/ui/dialog": {
        Dialog: dialog,
        DialogContent: "content",
        DialogDescription: "description",
        DialogHeader: "header",
        DialogTitle: "title",
      },
      "@/features/auth": {
        hasAuthToken: () => true,
        mustChangePassword: () => false,
      },
      "@/features/chat": {
        isExternalModelId: () => false,
        useChatRuntimeStore: (select: (state: typeof runtime) => unknown) =>
          select(runtime),
      },
      "@/features/hub": {
        useHfTokenStore: () => "",
        useInventoryVersion: () => 0,
      },
      "@tanstack/react-router": {
        useNavigate: () => undefined,
        useRouterState: () => ({ pathname: "/chat" }),
      },
      "./inbox": { runConfigInbox: inbox },
      "./link-lifecycle": {},
      "./target": targetModule,
    },
  );
  const value = { config: { nParallel: 3 } };
  const render = () => {
    const pending = inbox.getSnapshot();
    assert.ok(pending);
    return elements(SharedRunConfigLinkEditor({ pending, chatSearch: null }));
  };
  for (const model of [
    "owner/model",
    "/home/models/my model.gguf",
    "C:\\Models\\my model.gguf",
    "\\\\server\\models\\model.gguf",
    "/mnt/c/Models/model.gguf",
    "./models/native",
    "~/models/native",
    "ollama-manifest:registry.ollama.ai/library/llama3/latest",
  ]) {
    inbox.submit({ id: "local-choice", value });
    modelInput = "";
    let tree = render();
    assert.equal(
      tree.find((element) => element.type === dialog)?.props.open,
      true,
    );
    assert.equal(
      tree.find((element) => element.type === button)?.props.disabled,
      true,
    );
    const modelField = tree.find((element) => element.type === input);
    assert.ok(modelField);
    (modelField.props.onChange as (event: unknown) => void)({
      target: { value: model },
    });
    assert.equal(inbox.getSnapshot()?.value, value);
    tree = render();
    assert.equal(
      tree.find((element) => element.type === button)?.props.disabled,
      false,
      model,
    );
    const form = tree.find((element) => element.type === "form");
    assert.ok(form);
    (form.props.onSubmit as (event: unknown) => void)({
      preventDefault: () => undefined,
    });
    assert.equal(inbox.getSnapshot()?.selectedModel, model);
    assert.equal(inbox.getSnapshot()?.value, value);
    assert.equal(
      render().find((element) => element.type === dialog)?.props.open,
      false,
    );
  }
  for (const model of [
    "",
    "https://example.com/model.gguf",
    "external::provider::model",
    "/model\u0000.gguf",
    "/model\ud800.gguf",
  ]) {
    assert.equal(targetModule.isRunConfigModelInput(model), false, model);
  }
});

test("startup intake waits for mount effects and survives strict effect replay", async () => {
  const browser = installLocalStorageFake();
  const inbox = createRunConfigInbox();
  const effects: (() => undefined | (() => void))[] = [];
  let received = 0;
  const editor = Symbol("lazy link editor");
  let authChanged: (() => void) | undefined;
  let disposed = false;
  let revisions = 0;
  Object.assign(window.location, {
    href: "http://localhost/chat",
    origin: "http://localhost",
  });
  const { SharedRunConfigLinkHandler } = loadWithStubs<{
    SharedRunConfigLinkHandler: typeof LinkHandler;
  }>(
    new URL(
      "../src/features/model-picker/sharing/link-handler.tsx",
      import.meta.url,
    ),
    {
      "react/jsx-runtime": stubJsxRuntime(),
      react: {
        lazy: () => editor,
        Suspense: Symbol("suspense"),
        useState: () => [
          0,
          () => {
            revisions += 1;
          },
        ],
        useSyncExternalStore: (_subscribe: unknown, get: () => unknown) =>
          get(),
        useEffect: (effect: () => undefined | (() => void)) =>
          effects.push(effect),
      },
      "./inbox": { runConfigInbox: inbox },
      "./receive-link": {
        receiveStartupRunConfigUrl: () => {
          received += 1;
        },
        subscribeRunConfigSession: (onChange: () => void) => {
          authChanged = onChange;
          return () => {
            disposed = true;
          };
        },
      },
    },
  );
  const chatSearch = { new: "retained-draft" };
  const render = () => SharedRunConfigLinkHandler({ chatSearch });
  assert.equal(render(), null);
  const cancelled = effects[0]();
  assert.equal(received, 0);
  cancelled?.();
  await Promise.resolve();
  assert.equal(received, 0);
  disposed = false;
  const cleanup = effects[0]();
  assert.equal(received, 0);
  await Promise.resolve();
  assert.equal(received, 1);
  for (const event of ["hashchange", "popstate"]) {
    window.location.href =
      "http://localhost/chat#run?v=1&model=owner/model&nParallel=3";
    browser.fireWindowEvent(event, {});
    assert.equal(inbox.getSnapshot(), null);
    assert.equal(received, 1);
  }
  inbox.submit({ id: "link", value: { config: { nParallel: 3 } } });
  const shown = elements(render()).find((element) => element.type === editor);
  assert.equal(shown?.props.pending, inbox.getSnapshot());
  assert.equal(shown?.props.chatSearch, chatSearch);
  authChanged?.();
  assert.equal(revisions, 1);
  inbox.clear("link");
  assert.equal(render(), null);
  cleanup?.();
  assert.equal(disposed, true);
});
