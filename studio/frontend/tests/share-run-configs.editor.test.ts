// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type { ModelPickTarget } from "../src/features/model-picker/components/model-selector/types.ts";
import type { PerModelConfig } from "../src/features/model-picker/model-config/per-model-config.ts";
import type { SharedRunConfigControls as Controls } from "../src/features/model-picker/sharing/config-controls.tsx";
import type { SharedRunConfigReview as Review } from "../src/features/model-picker/sharing/config-ui.tsx";
import type { SharedRunConfigActions as Actions } from "../src/features/model-picker/sharing/config-ui.tsx";
import * as events from "../src/features/model-picker/sharing/editor-events.ts";
import type { SharedRunConfigLinkEditor as LinkEditor } from "../src/features/model-picker/sharing/link-editor.tsx";
import type { SharedRunConfigLinkHandler as LinkHandler } from "../src/features/model-picker/sharing/link-handler.tsx";
import type { ShareRunConfigDialog as ShareDialog } from "../src/features/model-picker/sharing/share-dialog.tsx";
import {
  installLocalStorageFake,
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
const { mergeSharedRunConfig } = fields;
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
const { createRunConfigInbox } = await import(
  "../src/features/model-picker/sharing/inbox.ts"
);
const targetModule = await import("./helpers/sharing-target.ts");
const { reconcileGpuSelection } = await import("../src/hooks/gpu-selection.ts");

const { SharedRunConfigReview } = loadWithStubs<{
  SharedRunConfigReview: typeof Review;
}>(
  new URL(
    "../src/features/model-picker/sharing/config-ui.tsx",
    import.meta.url,
  ),
  {
    "react/jsx-runtime": stubJsxRuntime(),
    react: {},
    "@/components/ui/button": {},
    "../model-config/model-config-draft": {},
    "./inbox": {},
    "./import-config": {},
    "./share-dialog": {},
    "./fields": fields,
  },
);

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

function shareDialogHarness(
  config: PerModelConfig,
  desktop = true,
  target: ModelPickTarget = {
    id: "owner/Model-GGUF",
    displayName: "Model",
    ggufVariant: "Q4_K_M",
    isGguf: true,
    apiLoadable: true,
    meta: { source: "hub", isLora: false },
  },
) {
  const checkbox = Symbol("checkbox");
  const textarea = Symbol("textarea");
  const states: unknown[] = [];
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
      "@/lib/api-base": { isTauri: desktop },
      "@/lib/copy-to-clipboard": {},
      "@/lib/toast": {},
      "../model-config/per-model-config": { DEFAULT_PER_MODEL_CONFIG },
      "./extra-args": sharedArgs,
      "./fields": fields,
      "./links": links,
    },
  );
  return () => {
    cursor = 0;
    const tree = elements(
      ShareRunConfigDialog({
        target,
        config,
        onClose: () => undefined,
      }),
    );
    const link = tree.find((element) => element.type === textarea)?.props.value;
    assert.equal(typeof link, "string");
    const parsed = links.parseRunConfigLink(link as string);
    assert.ok(parsed.kind === "valid");
    return {
      tree,
      link: link as string,
      value: parsed.value,
      config: parsed.value.config,
      choice: (key: string) =>
        tree.find(
          (element) =>
            element.type === checkbox && element.props.id === `share-${key}`,
        ),
    };
  };
}

test("local sharing preserves the recipient's quant unless the sender explicitly includes it", () => {
  const selection = {
    params: { checkpoint: "unsloth/Qwen3-8B-GGUF" },
    activeGgufVariant: "Q4_K_M",
    loadedIsGguf: true,
    activeNativePathToken: null,
    activeLoadId: null,
    models: [],
    loras: [],
  };
  for (const id of [
    "/models/Qwen3-8B",
    "/Users/test/Models/Qwen3-8B",
    "C:\\Models\\Qwen3-8B",
    "/mnt/c/Models/Qwen3-8B",
  ]) {
    const render = shareDialogHarness(DEFAULT_PER_MODEL_CONFIG, true, {
      id,
      displayName: "Qwen3-8B",
      ggufVariant: "Q8_0",
      isGguf: true,
      apiLoadable: true,
      meta: { source: "local", isLora: false },
    });
    const initial = render();
    const variant = initial.choice("variant");
    assert.ok(variant);
    assert.equal(variant.props.checked, false);
    assert.equal(initial.value.model, undefined);
    assert.equal(initial.value.ggufVariant, undefined);
    assert.equal(initial.value.isGguf, undefined);
    const unchanged = targetModule.resolveRunConfigTarget(
      initial.value,
      selection,
    );
    assert.equal(unchanged?.meta.ggufVariant, "Q4_K_M");
    assert.equal(unchanged?.meta.isDownloaded, true);
    (variant.props.onCheckedChange as (checked: boolean) => void)(true);
    const explicit = render();
    assert.equal(explicit.value.ggufVariant, "Q8_0");
    const changed = targetModule.resolveRunConfigTarget(
      explicit.value,
      selection,
    );
    assert.equal(changed?.meta.ggufVariant, "Q8_0");
    assert.notEqual(changed?.meta.isDownloaded, true);
  }
  const shareable = shareDialogHarness(DEFAULT_PER_MODEL_CONFIG)();
  assert.equal(shareable.choice("variant")?.props.checked, true);
  assert.equal(shareable.value.ggufVariant, "Q4_K_M");
});

for (const [address, destination] of [
  ["http://localhost:8888", "browser"],
  ["http://127.0.0.1:8888", "browser"],
  ["http://127.10.20.30:8888", "browser"],
  ["http://[::1]:8888", "browser"],
  ["http://192.168.1.20:8888", "desktop"],
  ["http://10.0.0.2:8888", "desktop"],
  ["http://[fd00::1]:8888", "desktop"],
  ["https://studio.example.com", "desktop"],
  ["https://studio.trycloudflare.com", "desktop"],
  ["https://studio.ngrok.app", "desktop"],
  ["https://localhost.example.com", "desktop"],
]) {
  test(`share dialog defaults safely at ${address}`, (t) => {
    const previous = window.location;
    Object.assign(window, { location: new URL(`${address}/chat?private=1`) });
    t.after(() => Object.assign(window, { location: previous }));
    const render = shareDialogHarness(DEFAULT_PER_MODEL_CONFIG, false);
    const initial = render();
    const select = initial.tree.find((element) => element.type === "select");
    assert.ok(select);
    assert.equal(select.props.value, destination);
    assert.equal(
      new URL(initial.link).protocol,
      destination === "browser" ? new URL(address).protocol : "unsloth:",
    );
    (select.props.onValueChange as (value: string) => void)("browser");
    const explicit = new URL(render().link);
    assert.equal(explicit.origin, new URL(address).origin);
    assert.equal(explicit.search, "?run=1");
  });
}

test("extra arguments are always selectable and empty arguments are shared only when selected", () => {
  const recipient = {
    ...DEFAULT_PER_MODEL_CONFIG,
    llamaExtraArgs: ["--threads", "8"],
    chatTemplateOverride: "{{ messages }}",
    reasoningBudgetMessage: "Recipient message",
  };
  for (const llamaExtraArgs of [undefined, null, [], ["--threads", "4"]]) {
    const render = shareDialogHarness({
      ...DEFAULT_PER_MODEL_CONFIG,
      llamaExtraArgs,
      chatTemplateOverride: "",
      reasoningBudgetMessage: "Sender instruction",
    });
    const { tree, config, choice } = render();
    const imported = mergeSharedRunConfig(recipient, config, true);
    const messageChoice = choice("reasoningBudgetMessage");
    assert.ok(messageChoice);
    assert.equal(messageChoice.props.disabled, true);
    assert.equal(messageChoice.props.checked, false);
    assert.equal(
      imported.reasoningBudgetMessage,
      recipient.reasoningBudgetMessage,
    );
    assert.match(text(tree), /Custom reasoning messages cannot be shared/);
    assert.equal(choice("chatTemplateOverride"), undefined);
    assert.equal(imported.chatTemplateOverride, recipient.chatTemplateOverride);
    const argsChoice = choice("llamaExtraArgs");
    const nonempty = (llamaExtraArgs?.length ?? 0) > 0;
    assert.deepEqual(
      imported.llamaExtraArgs,
      nonempty ? llamaExtraArgs : recipient.llamaExtraArgs,
    );
    assert.ok(argsChoice);
    assert.equal(argsChoice.props.disabled, false);
    assert.equal(argsChoice.props.checked, nonempty);
    if (!nonempty) {
      (argsChoice.props.onCheckedChange as (checked: boolean) => void)(true);
      const selected = render();
      assert.equal(selected.choice("llamaExtraArgs")?.props.checked, true);
      assert.match(text(selected.tree), /No extra arguments/);
      assert.deepEqual(selected.config.llamaExtraArgs, llamaExtraArgs ?? null);
    }
  }
});

test("chat templates are absent from sharing options and generated links", () => {
  for (const chatTemplateOverride of [null, "", "{{ messages }}"]) {
    const { choice, config, link } = shareDialogHarness({
      ...DEFAULT_PER_MODEL_CONFIG,
      chatTemplateOverride,
    })();
    assert.equal(choice("chatTemplateOverride"), undefined);
    assert.equal(Object.hasOwn(config, "chatTemplateOverride"), false);
    assert.equal(link.includes("chatTemplateOverride"), false);
  }
});

test("automatic GPU settings preserve recipient overrides until explicitly selected", () => {
  const automatic = {
    gpuMemoryMode: "auto" as const,
    gpuLayers: -1,
    nCpuMoe: 0,
    selectedGpuIds: null,
    selectedGpuIndexKind: null,
  };
  const manual = {
    gpuMemoryMode: "manual" as const,
    gpuLayers: 20,
    nCpuMoe: 4,
    selectedGpuIds: [1, 0],
    selectedGpuIndexKind: "physical" as const,
  };
  const recipient = { ...DEFAULT_PER_MODEL_CONFIG, ...manual };
  const render = shareDialogHarness({
    ...DEFAULT_PER_MODEL_CONFIG,
    ...automatic,
  });
  const initial = render();
  assert.deepEqual(initial.config, {});
  assert.deepEqual(
    mergeSharedRunConfig(recipient, initial.config, true),
    recipient,
  );
  for (const key of Object.keys(automatic)) {
    const choice = initial.choice(key);
    assert.ok(choice, key);
    assert.equal(choice.props.checked, false, key);
    assert.equal(choice.props.disabled, false, key);
    (choice.props.onCheckedChange as (checked: boolean) => void)(true);
  }
  assert.deepEqual(render().config, automatic);
  assert.deepEqual(mergeSharedRunConfig(recipient, render().config, true), {
    ...recipient,
    ...automatic,
  });
  const selected = shareDialogHarness({
    ...DEFAULT_PER_MODEL_CONFIG,
    ...manual,
  })();
  assert.deepEqual(selected.config, manual);
  for (const key of Object.keys(manual)) {
    assert.equal(selected.choice(key)?.props.checked, true, key);
  }
  for (const gpu of [{}, { gpuMemoryMode: "auto" as const }]) {
    assert.deepEqual(
      shareDialogHarness({ ...DEFAULT_PER_MODEL_CONFIG, ...gpu })().config,
      {},
    );
  }
});

test("Share opens and closes its dialog", () => {
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
  const dialog = Symbol("share dialog");
  const button = Symbol("button");
  const { SharedRunConfigActions } = loadWithStubs<{
    SharedRunConfigActions: typeof Actions;
  }>(
    new URL(
      "../src/features/model-picker/sharing/config-ui.tsx",
      import.meta.url,
    ),
    {
      "react/jsx-runtime": stubJsxRuntime(),
      react: {
        useState: () => [
          sharing,
          (value: boolean) => {
            sharing = value;
          },
        ],
        useSyncExternalStore: (_subscribe: unknown, get: () => unknown) =>
          get(),
        useEffect: () => undefined,
      },
      "@/components/ui/button": { Button: button },
      "../model-config/model-config-draft": {
        modelConfigDraftKey,
      },
      "./inbox": { runConfigInbox: inbox },
      "./import-config": { scheduleRunConfigImport: () => undefined },
      "./fields": fields,
      "./share-dialog": { ShareRunConfigDialog: dialog },
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
  const render = () => elements(SharedRunConfigActions(props));
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
  assert.equal(inbox.getSnapshot()?.id, "pending");
});

test("closing an editor before its sharing UI loads cancels the import, while effect replay retains it", async () => {
  const inbox = createRunConfigInbox();
  const target = {
    id: "owner/model",
    displayName: "Model",
    isGguf: false,
    apiLoadable: true,
    meta: { source: "hub" as const, isLora: false },
  };
  const key = modelConfigDraftKey(target.id, undefined);
  inbox.submit({
    id: "pending",
    draftKey: key,
    value: { config: { nParallel: 3 } },
  });
  const effects: (() => (() => void) | undefined)[] = [];
  const notices: { id: string; description: string }[] = [];
  const actions = Symbol("lazy sharing UI");
  const { SharedRunConfigControls } = loadWithStubs<{
    SharedRunConfigControls: typeof Controls;
  }>(
    new URL(
      "../src/features/model-picker/sharing/config-controls.tsx",
      import.meta.url,
    ),
    {
      "react/jsx-runtime": stubJsxRuntime(),
      "@/components/lazy-import-boundary": {
        LazyImportBoundary: "boundary",
        LazyImportFailure: "failure",
      },
      "@/components/ui/button": { Button: Symbol("button") },
      react: {
        lazy: () => actions,
        Suspense: Symbol("suspense"),
        useLayoutEffect: (effect: () => (() => void) | undefined) => {
          effects.push(effect);
        },
      },
      "@/lib/toast": {
        toast: {
          info: (_message: string, options: (typeof notices)[number]) =>
            notices.push(options),
        },
      },
      "../model-config/model-config-draft": {
        modelConfigDraftKey,
        isExtraArgsHydratedForDraft: () => false,
      },
      "./variant": { isRunConfigVariantUnresolved: () => false },
      "./inbox": { runConfigInbox: inbox },
    },
  );
  const props = {
    className: "h-9 rounded-full",
    target,
    config: DEFAULT_PER_MODEL_CONFIG,
    ready: true,
    isDiffusion: false,
    canImport: true,
    disabled: false,
    onImport: () => undefined,
  };
  SharedRunConfigControls({ ...props, canImport: false });
  assert.equal(effects[0](), undefined);
  const tree = elements(SharedRunConfigControls(props));
  assert.equal(tree[0].type, "boundary");
  const fallback = tree[0].props.fallback as StubElement;
  assert.equal(fallback.props.disabled, true);
  assert.equal(fallback.props.className, props.className);
  assert.equal(
    tree.find((element) => element.type === actions)?.props.target,
    target,
  );
  const release = effects[1]();
  assert.ok(release);
  release();
  SharedRunConfigControls(props);
  const releaseRemounted = effects[2]();
  await Promise.resolve();
  assert.equal(inbox.getSnapshot()?.id, "pending");
  assert.ok(releaseRemounted);
  assert.equal(notices.length, 0);
  releaseRemounted();
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

test("review renders field labels and argument values as text", () => {
  assert.equal(
    SharedRunConfigReview({
      config: null,
      draftConfig: DEFAULT_PER_MODEL_CONFIG,
      currentConfig: DEFAULT_PER_MODEL_CONFIG,
    }),
    null,
  );
  const args = ["--rope-scaling", "yarn"];
  const imported = {
    nParallel: 3,
    llamaExtraArgs: args,
    maxSeqLength: null,
  };
  const tree = SharedRunConfigReview({
    config: imported,
    draftConfig: { ...DEFAULT_PER_MODEL_CONFIG, ...imported },
    currentConfig: { ...DEFAULT_PER_MODEL_CONFIG, ...imported },
  });
  assert.ok(text(tree).includes("Settings changed by link (3)"));
  assert.ok(text(tree).includes("Parallel slots"));
  assert.ok(text(tree).includes("--rope-scaling yarn"));
  assert.ok(text(tree).includes("Default"));
  assert.match(text(tree), /checked saves these settings for future loads/);
  assert.match(
    text(tree),
    /unchecked deletes any saved settings for this model/,
  );
  assert.match(text(tree), /without loading to keep your saved settings/);
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
    { nParallel: 7 },
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

test("review identifies a linked repository and explains uncached downloads, including model-only links", () => {
  for (const config of [{}, { nParallel: 3 }]) {
    const draftConfig = { ...DEFAULT_PER_MODEL_CONFIG, ...config };
    const props = { config, draftConfig, currentConfig: draftConfig };
    const tree = SharedRunConfigReview({ ...props, model: "owner/model" });
    assert.match(
      text(tree),
      /This link selected owner\/model from Hugging Face/,
    );
    assert.match(
      text(tree),
      /Loading downloads any model files that are not already cached/,
    );
    assert.doesNotMatch(
      text(SharedRunConfigReview(props)),
      /This link selected/,
    );
  }
});

test("review identifies an explicit quant selection and possible download without a linked model or changed settings", () => {
  for (const model of [undefined, "owner/Model-GGUF"]) {
    for (const config of [{}, { nParallel: 3 }]) {
      const draftConfig = { ...DEFAULT_PER_MODEL_CONFIG, ...config };
      const tree = SharedRunConfigReview({
        config,
        model,
        ggufVariant: "Q8_0",
        draftConfig,
        currentConfig: draftConfig,
      });
      assert.match(text(tree), /This link selected GGUF variant Q8_0/);
      assert.match(
        text(tree),
        /Loading downloads any model files that are not already cached/,
      );
      assert.doesNotMatch(text(tree), /Link settings already match/);
      if (!model) assert.doesNotMatch(text(tree), /from Hugging Face/);
      if (!model && Object.keys(config).length === 0) {
        assert.match(text(tree), /GGUF variant selected by link/);
      }
    }
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
    "models/owner/checkpoint-500",
    "models\\checkpoint-500",
    "models/my native model",
    "checkpoint-500",
    ".\\models\\checkpoint-500",
    "../models/checkpoint-500",
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
      "@/components/lazy-import-boundary": {
        LazyImportBoundary: "boundary",
        LazyImportFailure: "failure",
      },
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
  const boundary = render() as StubElement;
  assert.equal(boundary.type, "boundary");
  const failure = boundary.props.fallback as StubElement;
  assert.equal(failure.type, "failure");
  (failure.props.onDismiss as () => void)();
  assert.equal(render(), null);
  cleanup?.();
  assert.equal(disposed, true);
});
