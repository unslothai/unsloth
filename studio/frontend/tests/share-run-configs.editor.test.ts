// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type { ModelPickTarget } from "../src/features/model-picker/components/model-selector/types.ts";
import type { PerModelConfig } from "../src/features/model-picker/model-config/per-model-config.ts";
import type * as Controls from "../src/features/model-picker/sharing/config-controls.tsx";
import type * as ConfigUi from "../src/features/model-picker/sharing/config-ui.tsx";
import type * as LinkEditor from "../src/features/model-picker/sharing/link-editor.tsx";
import type * as LinkHandler from "../src/features/model-picker/sharing/link-handler.tsx";
import type * as ShareDialog from "../src/features/model-picker/sharing/share-dialog.tsx";
import {
  installLocalStorageFake,
  readSrc,
  registerBundlerResolver,
} from "./helpers/kit.ts";
import {
  type StubElement,
  loadWithStubs,
  stubJsxRuntime,
} from "./helpers/module-stubs.ts";

registerBundlerResolver();
installLocalStorageFake();
const [fields, sharedArgs, links, perModel, drafts, inboxes, targets] =
  await Promise.all([
    import("../src/features/model-picker/sharing/fields.ts"),
    import("../src/features/model-picker/sharing/extra-args.ts"),
    import("./helpers/sharing-links.ts"),
    import("../src/features/model-picker/model-config/per-model-config.ts"),
    import("../src/features/model-picker/model-config/model-config-draft.ts"),
    import("../src/features/model-picker/sharing/inbox.ts"),
    import("./helpers/sharing-target.ts"),
  ]);
const { mergeSharedRunConfig } = fields;
const { createRunConfigLink, parseRunConfigLink } = links;
const { DEFAULT_PER_MODEL_CONFIG: D } = perModel;
const { modelConfigDraftKey } = drafts;
const { createRunConfigInbox } = inboxes;
const { isKnownNonGgufModel, resolveRunConfigTarget } = targets;

type Inbox = ReturnType<typeof createRunConfigInbox>;
type Selection = Parameters<typeof resolveRunConfigTarget>[1];
const hubTarget: ModelPickTarget = {
  id: "owner/Model-GGUF",
  displayName: "Model",
  ggufVariant: "Q4_K_M",
  isGguf: true,
  apiLoadable: true,
  meta: { source: "hub", isLora: false },
};

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

const find = (tree: StubElement[], type: unknown) =>
  tree.find((element) => element.type === type);
const summaryText = (node: unknown) => text(find(elements(node), "summary"));
const call = (element: StubElement | undefined, name: string, arg?: unknown) =>
  (element?.props[name] as (value?: unknown) => unknown)(arg);
const toggle = (element: StubElement | undefined, checked: boolean) =>
  call(element, "onCheckedChange", checked);
const tags = new Proxy({}, { get: (_target, name) => name });

// State by call order, effects recorded (never run); unlisted imports render as their names.
function load<T>(file: string, stubs: Record<string, unknown> = {}) {
  const states: unknown[] = [];
  const effects: (() => (() => void) | undefined)[] = [];
  const effect = (run: (typeof effects)[number]) => effects.push(run);
  let cursor = 0;
  const known: Record<string, unknown> = {
    react: {
      lazy: () => "lazy",
      Suspense: "Suspense",
      useId: () => "share",
      useMemo: (create: () => unknown) => create(),
      useRef: () => ({ current: null }),
      useSyncExternalStore: (_: unknown, get: () => unknown) => get(),
      useEffect: effect,
      useLayoutEffect: effect,
      useState: (initial: unknown) => {
        const index = cursor++;
        if (index >= states.length) {
          states.push(typeof initial === "function" ? initial() : initial);
        }
        const set = (next: unknown) => {
          states[index] =
            typeof next === "function" ? next(states[index]) : next;
        };
        return [states[index], set];
      },
    },
    "react/jsx-runtime": stubJsxRuntime(),
    "../model-config/model-config-draft": {
      modelConfigDraftKey,
      isExtraArgsHydratedForDraft: () => false,
    },
    "../model-config/per-model-config": perModel,
    "./extra-args": sharedArgs,
    "./fields": fields,
    "./links": links,
    ...stubs,
  };
  const module = loadWithStubs<T>(
    new URL(`../src/features/model-picker/sharing/${file}`, import.meta.url),
    new Proxy(known, {
      has: () => true,
      get: (target, name: string) => (name in target ? target[name] : tags),
    }),
  );
  const render = <R>(draw: () => R) => {
    cursor = 0;
    return draw();
  };
  return { module, states, effects, render };
}

const loadConfigUi = (inbox = createRunConfigInbox()) =>
  load<typeof ConfigUi>("config-ui.tsx", {
    "./inbox": { runConfigInbox: inbox },
    "./import-config": { scheduleRunConfigImport: () => undefined },
  });
const { SharedRunConfigReview } = loadConfigUi().module;
const review = (
  config: Partial<PerModelConfig> | null,
  props: Partial<Parameters<typeof SharedRunConfigReview>[0]> = {},
  selection: { model?: string; ggufVariant?: string } = {},
) => {
  const draftConfig = { ...D, ...config };
  return SharedRunConfigReview({
    imported: config && { changes: config, ...selection },
    draftConfig,
    currentConfig: draftConfig,
    ...props,
  });
};

function shareDialog(config: PerModelConfig, target = hubTarget, tauri = true) {
  const { module, render } = load<typeof ShareDialog>("share-dialog.tsx", {
    "@/lib/api-base": { isTauri: tauri },
  });
  return () => {
    const tree = elements(
      render(() =>
        module.ShareRunConfigDialog({ target, config, onClose: () => {} }),
      ),
    );
    const link = find(tree, "Textarea")?.props.value as string;
    const parsed = parseRunConfigLink(link);
    assert.ok(parsed.kind === "valid");
    const choice = (key: string) =>
      tree.find((element) => element.props.id === `share-${key}`);
    const { value } = parsed;
    return { tree, link, value, config: value.config, choice };
  };
}

test("links keep null, false, zero and empty values and drop native-only settings", () => {
  const defaults = { ...D, nParallel: 8, llamaExtraArgs: ["--metrics"] };
  const empty = [{}, { nParallel: null }, { tensorParallel: false }];
  const zero = [{ reasoningBudget: 0 }, { reasoningBudgetMessage: "" }];
  for (const patch of [...empty, ...zero, { llamaExtraArgs: [] }]) {
    const parsed = parseRunConfigLink(createRunConfigLink({ config: patch }));
    assert.ok(parsed.kind === "valid");
    const merged = mergeSharedRunConfig(defaults, parsed.value.config);
    assert.deepEqual(merged, { ...defaults, ...patch });
  }
  const native = { maxSeqLength: 8192, mlxKvQuant: "4" as const };
  const link = createRunConfigLink({ config: { nParallel: 2, ...native } });
  assert.ok(!/maxSeqLength|mlxKv/.test(link));
  const omitted = { nParallel: undefined, ...native };
  assert.deepEqual(mergeSharedRunConfig(defaults, omitted), defaults);
  const patch = { llamaExtraArgs: ["--metrics"] };
  mergeSharedRunConfig(defaults, patch).llamaExtraArgs?.push("--verbose");
  const kept = [patch, defaults].map((config) => config.llamaExtraArgs);
  assert.deepEqual(kept, [["--metrics"], ["--metrics"]]);
  for (const query of ["maxSeqLength=8192", "mlxKvQuant=4", "isGguf=true"]) {
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
  inbox.bind("second", "model-B");
  assert.deepEqual(inbox.take("second", "model-B"), { nParallel: 4 });
  assert.equal(inbox.take("second", "model-B"), null);
  assert.equal(inbox.getSnapshot(), null);
  assert.equal(notifications, 5);
  unsubscribe();
  inbox.submit({ id: "third", value: { config: {} } });
  assert.equal(notifications, 5);
});

test("editor cleanup cancels only the last editor's own unfinished import, surviving remounts", async () => {
  const inbox = createRunConfigInbox();
  const cancelled: string[] = [];
  const onCancel = (request: { id: string }) => cancelled.push(request.id);
  inbox.submit({ id: "first", value: { config: { nParallel: 2 } } });
  inbox.bind("first", "model-A");
  inbox.retainEditor("model-A", onCancel)();
  const releaseRemount = inbox.retainEditor("model-A", onCancel);
  await Promise.resolve();
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
  for (const nextKey of ["model-A", "model-B"]) {
    inbox.submit({ id: "old", value: { config: {} } });
    inbox.bind("old", "model-A");
    inbox.retainEditor("model-A")();
    inbox.submit({ id: "new", value: { config: {} } });
    inbox.bind("new", nextKey);
    await Promise.resolve();
    assert.equal(inbox.getSnapshot()?.id, "new", nextKey);
  }
  for (const config of [{}, { nParallel: 2 }]) {
    inbox.submit({ id: "done", draftKey: "model-A", value: { config } });
    const release = inbox.retainEditor("model-A", () => assert.fail());
    if ("nParallel" in config) inbox.take("done", "model-A");
    release();
    await Promise.resolve();
    assert.equal(inbox.getSnapshot(), null);
  }
});

const selection: Selection = {
  params: { checkpoint: "owner/model" },
  activeGgufVariant: "Q4",
  loadedIsGguf: true,
  activeNativePathToken: "token",
  activeLoadId: "/cache/owner/model",
  models: [],
  loras: [],
};
const unknown = { loadedIsGguf: null, activeGgufVariant: null };
const at = (checkpoint: string, extra: Partial<Selection> = {}) => ({
  ...unknown,
  activeNativePathToken: null,
  params: { checkpoint },
  ...extra,
});
const other = (isGguf: boolean) => ({
  models: [{ id: "owner/other", isGguf, isLora: false }],
});
const lora = (id: string, exportType: "gguf" | "lora") =>
  at(id, { loras: [{ id, exportType }] });
const file = "C:\\m.gguf";
const local = { isGguf: true, nativePathToken: "token", isDownloaded: true };
const fresh = { nativePathToken: undefined, loadId: undefined };
const none = { ggufVariant: undefined };
const u = undefined;
const c = { config: {} };
const q8 = { ggufVariant: "Q8", config: {} };

test("run config targets resolve model identity, format and local capability", () => {
  // [link value, recipient selection, chosen model, expected id and meta]
  const rows = [
    [c, {}, u, { id: "owner/model", ggufVariant: "Q4", ...local }],
    [c, {}, u, { loadId: "/cache/owner/model" }],
    [c, at(""), u, null],
    [{ ...q8, model: "owner/other" }, other(false), u, { ...fresh }],
    [{ ...q8, model: "owner/other" }, other(false), u, { isGguf: true }],
    [{ ...q8, model: "owner/other" }, other(false), u, { ggufVariant: "Q8" }],
    [{ ...c, model: "owner/other" }, other(true), u, { ...none, ...fresh }],
    [c, lora("/e", "gguf"), u, { isGguf: true, isLora: false }],
    [c, lora("/a", "lora"), u, { isGguf: false, isLora: true }],
    [q8, { params: { checkpoint: file } }, u, { ...none, ...local }],
    [
      q8,
      { activeLoadId: file, params: { checkpoint: file } },
      u,
      { loadId: file },
    ],
    [q8, { loadedIsGguf: false }, u, { isGguf: false, ...none }],
    [c, at(""), "a/n", { id: "a/n", isGguf: true, ...none }],
    [c, at("a/n"), u, { id: "a/n", isGguf: true, ...none }],
  ] as const;
  for (const [value, overrides, chosen, expected] of rows) {
    const state = { ...selection, ...overrides };
    const target = resolveRunConfigTarget(value, state, chosen);
    if (!expected) {
      assert.equal(target, null);
      continue;
    }
    const actual: Record<string, unknown> = { id: target?.id, ...target?.meta };
    for (const [key, field] of Object.entries(expected)) {
      assert.deepEqual(actual[key], field, key);
    }
  }
  const loras = [
    { id: "/e", exportType: "gguf" as const },
    { id: "/a", exportType: "lora" as const },
  ];
  const native = { ...selection, loadedIsGguf: false, loras };
  for (const [id, state, expected] of [
    ["owner/model", selection, false],
    ["owner/model", native, true],
    ["", native, false],
    [file, { ...native, params: { checkpoint: file } }, false],
    ["/a", native, true],
    ["/e", native, false],
    ["a/n", { ...selection, ...unknown }, false],
  ] as const) {
    assert.equal(isKnownNonGgufModel(id, state), expected, id);
  }
});

test("local sharing keeps the recipient's quant unless the sender includes it", () => {
  const recipient = { ...selection, activeNativePathToken: null };
  const opened = (value: Parameters<typeof resolveRunConfigTarget>[0]) =>
    resolveRunConfigTarget(value, recipient)?.meta;
  const meta = { source: "local" as const, isLora: false };
  const render = shareDialog(D, { ...hubTarget, id: "/m/Qwen", meta });
  const initial = render();
  assert.equal(initial.choice("variant")?.props.checked, false);
  assert.deepEqual(initial.value, { config: {} });
  assert.equal(opened(initial.value)?.ggufVariant, "Q4");
  assert.equal(opened(initial.value)?.isDownloaded, true);
  toggle(initial.choice("variant"), true);
  assert.equal(render().value.ggufVariant, "Q4_K_M");
  assert.equal(opened(render().value)?.ggufVariant, "Q4_K_M");
  assert.notEqual(opened(render().value)?.isDownloaded, true);
  const shareable = shareDialog(D)();
  assert.equal(shareable.choice("variant")?.props.checked, true);
  assert.equal(shareable.value.ggufVariant, "Q4_K_M");
});

test("share dialog defaults to Web only on loopback addresses", () => {
  const previous = window.location;
  try {
    for (const address of ["http://localhost:8888", "http://192.168.1.20:8888"]) {
      Object.assign(window, { location: new URL(`${address}/chat`) });
      assert.equal(new URL(shareDialog(D)().link).protocol, "unsloth:", address);
    }
    for (const [address, destination] of [
      ["http://localhost:8888", "browser"],
      ["http://127.10.20.30:8888", "browser"],
      ["http://[::1]:8888", "browser"],
      ["http://192.168.1.20:8888", "desktop"],
      ["https://localhost.example.com", "desktop"],
    ]) {
      Object.assign(window, { location: new URL(`${address}/chat?p=1`) });
      const render = shareDialog(D, hubTarget, false);
      const select = find(render().tree, "Select");
      assert.equal(select?.props.value, destination, address);
      const { protocol } = new URL(render().link);
      const http = new URL(address).protocol;
      assert.equal(protocol, destination === "browser" ? http : "unsloth:");
      call(select, "onValueChange", "browser");
      const explicit = new URL(render().link);
      assert.equal(explicit.origin, new URL(address).origin);
      assert.equal(explicit.search, "?run=1");
      const note = text(render().tree);
      assert.equal(
        note.includes("Anyone with it and your password can sign in"),
        destination === "desktop",
        address,
      );
    }
  } finally {
    Object.assign(window, { location: previous });
  }
});

test("extra arguments are always selectable; custom text, templates and native settings are not", () => {
  const recipient = {
    ...D,
    llamaExtraArgs: ["--threads", "8"],
    chatTemplateOverride: "{{ messages }}",
    reasoningBudgetMessage: "Recipient message",
  };
  for (const llamaExtraArgs of [undefined, null, [], ["--threads", "4"]]) {
    const render = shareDialog({
      ...D,
      llamaExtraArgs,
      chatTemplateOverride: "{{ sender }}",
      maxSeqLength: 4096,
      mlxKvQuant: "8",
      reasoningBudgetMessage: "Sender instruction",
    });
    const { tree, config, choice, link } = render();
    const message = choice("reasoningBudgetMessage")?.props;
    assert.deepEqual([message?.disabled, message?.checked], [true, false]);
    assert.match(text(tree), /Custom reasoning messages cannot be shared/);
    const hidden = ["chatTemplateOverride", "maxSeqLength", "mlxKvQuant"];
    assert.ok(!hidden.some(choice));
    assert.ok(!/chatTemplateOverride|maxSeqLength|mlxKv|isGguf/.test(link));
    assert.deepEqual(mergeSharedRunConfig(recipient, config), {
      ...recipient,
      ...(llamaExtraArgs?.length && { llamaExtraArgs }),
    });
    const args = choice("llamaExtraArgs");
    assert.equal(args?.props.disabled, false);
    assert.equal(args?.props.checked, Boolean(llamaExtraArgs?.length));
    if (!llamaExtraArgs?.length) {
      toggle(args, true);
      const selected = render();
      assert.equal(selected.choice("llamaExtraArgs")?.props.checked, true);
      assert.match(text(selected.tree), /No extra arguments/);
      assert.deepEqual(selected.config.llamaExtraArgs, llamaExtraArgs ?? null);
    }
  }
});

test("automatic GPU settings preserve recipient overrides until explicitly selected", () => {
  // [field, automatic value, manual value]
  const gpu = [
    ["gpuMemoryMode", "auto", "manual"],
    ["gpuLayers", -1, 20],
    ["nCpuMoe", 0, 4],
    ["selectedGpuIds", null, [1, 0]],
    ["selectedGpuIndexKind", null, "physical"],
  ] as const;
  const pick = (index: 1 | 2): Partial<PerModelConfig> =>
    Object.fromEntries(gpu.map((row) => [row[0], row[index]]));
  const [automatic, manual] = [pick(1), pick(2)];
  const render = shareDialog({ ...D, ...automatic });
  assert.deepEqual(render().config, {});
  for (const key of Object.keys(automatic)) {
    const choice = render().choice(key);
    assert.equal(choice?.props.checked || choice?.props.disabled, false);
    toggle(choice, true);
  }
  assert.deepEqual(render().config, automatic);
  assert.deepEqual(mergeSharedRunConfig({ ...D, ...manual }, render().config), {
    ...D,
    ...automatic,
    tensorSplit: null,
  });
  assert.deepEqual(shareDialog({ ...D, ...manual })().config, manual);
  assert.deepEqual(shareDialog(D)().config, {});
});

function pendingImport() {
  const inbox = createRunConfigInbox();
  const draftKey = modelConfigDraftKey(hubTarget.id, hubTarget.ggufVariant);
  inbox.submit({
    id: "pending",
    draftKey,
    value: { config: { nParallel: 3 } },
  });
  return inbox;
}
const controlProps = {
  className: "h-9 rounded-full",
  target: hubTarget,
  config: D,
  ready: true,
  canImport: true,
  disabled: false,
  onImport: () => undefined,
};

test("Share opens and closes its dialog without touching the pending import", () => {
  const inbox = pendingImport();
  const { module, render } = loadConfigUi(inbox);
  const draftKey = modelConfigDraftKey(hubTarget.id, hubTarget.ggufVariant);
  const props = { ...controlProps, draftKey, hydrated: true };
  const tree = () =>
    elements(render(() => module.SharedRunConfigActions(props)));
  assert.equal(find(tree(), "ShareRunConfigDialog"), undefined);
  call(find(tree(), "Button"), "onClick");
  call(find(tree(), "ShareRunConfigDialog"), "onClose");
  assert.equal(find(tree(), "ShareRunConfigDialog"), undefined);
  assert.equal(inbox.getSnapshot()?.id, "pending");
});

test("closing an editor before its sharing UI loads cancels the import, while effect replay retains it", async () => {
  const inbox = pendingImport();
  const notices: { id: string; description: string }[] = [];
  const { module, effects } = load<typeof Controls>("config-controls.tsx", {
    "@/lib/toast": {
      toast: { info: (_: string, notice: never) => notices.push(notice) },
    },
    "./inbox": { runConfigInbox: inbox },
    "./target": { isRunConfigVariantUnresolved: () => false },
  });
  const props = { ...controlProps, isDiffusion: false };
  module.SharedRunConfigControls({ ...props, canImport: false });
  assert.equal(effects[0](), undefined);
  const tree = elements(module.SharedRunConfigControls(props));
  assert.equal(tree[0].type, "LazyImportBoundary");
  const fallback = (tree[0].props.fallback as StubElement).props;
  assert.deepEqual(
    [fallback.disabled, fallback.className],
    [true, props.className],
  );
  assert.equal(find(tree, "lazy")?.props.target, hubTarget);
  assert.equal(find(tree, "lazy")?.props.hydrated, false);
  effects[1]()?.();
  module.SharedRunConfigControls(props);
  const releaseRemounted = effects[2]();
  await Promise.resolve();
  assert.equal(inbox.getSnapshot()?.id, "pending");
  assert.equal(notices.length, 0);
  releaseRemounted?.();
  await Promise.resolve();
  assert.equal(inbox.getSnapshot(), null);
  assert.equal(notices[0].id, "pending");
  assert.match(notices[0].description, /editor closed before the settings/);
});

test("edit cancellation includes contained controls and excludes portaled dialog controls", (t) => {
  class NodeFake extends EventTarget {
    child?: NodeFake;
    contains = (node: NodeFake) => node === this || node === this.child;
  }
  // Bare node has no DOM Node global; this file runs in its own process.
  Object.assign(globalThis, { Node: NodeFake });
  t.after(() => Reflect.deleteProperty(globalThis, "Node"));
  const editor = new NodeFake();
  editor.child = new NodeFake();
  const currentTarget = editor as unknown as Node;
  const events = load<typeof Controls>("config-controls.tsx").module;
  for (const [target, expected] of [
    [editor.child, true],
    [new NodeFake(), false],
    [new EventTarget(), false],
  ] as const) {
    const changed = events.isRunConfigEditorChange({ currentTarget, target });
    assert.equal(changed, expected);
  }
});

test("review renders field labels, values and device adjustments as text", () => {
  assert.equal(review(null), null);
  const args = ["--rope-scaling", "yarn"];
  const imported = {
    nParallel: 3,
    llamaExtraArgs: args,
    customContextLength: null,
    kvCacheDtype: "q8_0",
    selectedGpuIds: [0, 1],
  };
  const tree = review(imported);
  assert.equal(summaryText(tree), "Settings changed by link5");
  const values = elements(tree)
    .filter((e) => e.type === "dd")
    .map(text);
  assert.deepEqual(values, ["Default", "q8_0", "3", args.join(" "), "[0,1]"]);
  assert.ok(text(tree).includes("Parallel slots"));
  assert.ok(
    elements(tree).every((e) => !("dangerouslySetInnerHTML" in e.props)),
  );
  assert.ok(text(review({})).includes("already match"));
  for (const changed of [
    { nParallel: 7 },
    { llamaExtraArgs: ["-t", "4"] },
    D,
  ]) {
    const edited = { ...D, ...imported, ...changed };
    const props = { draftConfig: edited, currentConfig: edited };
    assert.equal(review(imported, props), null);
  }
  const copied = { ...D, ...imported, llamaExtraArgs: [...args] };
  assert.notEqual(review(imported, { currentConfig: copied }), null);
  assert.match(text(review({ llamaExtraArgs: null })), /No extra arguments/);
  const cleared = { currentConfig: { ...D, llamaExtraArgs: [] } };
  const adjusted = text(review({ llamaExtraArgs: args }, cleared));
  assert.match(adjusted, /No extra arguments/);
  assert.match(adjusted, /Requested --rope-scaling yarn, adjusted for this/);
});

test("review states what loading does to this model's saved settings", () => {
  const notes = [
    "Loading replaces your saved settings for this model. Close without loading to keep them.",
    "Loading saves these settings for this model.",
    "Loading clears your saved settings for this model. Close without loading to keep them.",
    "To keep these settings for next time, tick “Remember for this model”.",
  ];
  notes.forEach((note, index) => {
    const remember = index < 2;
    const hasSavedSettings = index % 2 === 0;
    const shown = text(
      review({ nParallel: 3 }, { remember, hasSavedSettings }),
    );
    assert.deepEqual(
      notes.filter((entry) => shown.includes(entry)),
      [note],
    );
  });
});

test("review identifies a linked repository or quant and explains possible downloads", () => {
  const hub = "If the model isn’t on this device, loading downloads it from";
  const quant = "If this variant isn’t on this device, loading downloads it.";
  for (const config of [{}, { nParallel: 3 }]) {
    const plain = text(review(config));
    assert.doesNotMatch(plain, /owner\/model|Hugging Face|downloads/);
    for (const [model, ggufVariant, summary, note] of [
      ["owner/model", u, "owner/model", hub],
      ["owner/Model-GGUF", "Q8_0", "owner/Model-GGUF · Q8_0", hub],
      [u, "Q8_0", "Q8_0", quant],
    ] as const) {
      const tree = review(config, {}, { model, ggufVariant });
      assert.ok(summaryText(tree).endsWith(summary));
      assert.ok(text(tree).includes(note));
      assert.doesNotMatch(text(tree), /Link settings already match/);
      if (!model) assert.doesNotMatch(text(tree), /from Hugging Face/);
      if (!model && !("nParallel" in config)) {
        assert.match(text(tree), /GGUF variant selected by link/);
      }
    }
  }
});

const yes = () => true;
const no = () => false;
const useRouterState = () => ({ pathname: "/chat" });

function linkEditor(runtime: object, inbox: Inbox) {
  const useChatRuntimeStore = (select: (state: object) => unknown) =>
    select(runtime);
  const { module, render, states } = load<typeof LinkEditor>(
    "link-editor.tsx",
    {
      "@/features/auth": { hasAuthToken: yes, mustChangePassword: no },
      "@/features/chat": { isExternalModelId: no, useChatRuntimeStore },
      "@/features/hub": { useHfTokenStore: no, useInventoryVersion: no },
      "@tanstack/react-router": { useNavigate: no, useRouterState },
      "./inbox": { runConfigInbox: inbox },
      "./target": targets,
    },
  );
  const draw = () => {
    const pending = inbox.getSnapshot();
    assert.ok(pending);
    const props = { pending, chatSearch: null };
    return elements(render(() => module.SharedRunConfigLinkEditor(props)));
  };
  return { draw, states };
}

test("settings-only chooser accepts recipient-local models and skips known GGUF selections", () => {
  const inbox = createRunConfigInbox();
  const empty = { params: { checkpoint: "" }, settingsHydrated: true };
  const { draw, states } = linkEditor(empty, inbox);
  const value = { config: { nParallel: 3 } };
  for (const model of [
    "owner/model",
    "C:\\Models\\my model.gguf",
    "./models/native",
    "ollama-manifest:registry.ollama.ai/library/llama3/latest",
  ]) {
    inbox.submit({ id: "local-choice", value });
    states.length = 0;
    assert.equal(find(draw(), "Dialog")?.props.open, true);
    assert.match(text(find(draw(), "Dialog")), /Choose a GGUF model/);
    assert.equal(find(draw(), "Button")?.props.disabled, true);
    call(find(draw(), "Input"), "onChange", { target: { value: model } });
    assert.equal(find(draw(), "Button")?.props.disabled, false, model);
    call(find(draw(), "form"), "onSubmit", { preventDefault: () => {} });
    assert.equal(inbox.getSnapshot()?.selectedModel, model);
    assert.equal(inbox.getSnapshot()?.value, value);
    assert.equal(find(draw(), "Dialog")?.props.open, false);
  }
  const bad = ["", "https://e.com/m.gguf", "external::provider::model"];
  for (const model of [...bad, "/m\u0000.gguf", "/m\ud800.gguf"]) {
    assert.equal(targets.isRunConfigModelInput(model), false, model);
  }
  const runtime = {
    ...at("a/n"),
    models: [],
    loras: [],
    settingsHydrated: true,
  };
  for (const [loadedIsGguf, model, open] of [
    [false, u, true],
    [true, u, false],
    [null, u, false],
    [false, "owner/Model-GGUF", false],
  ] as const) {
    inbox.submit({ id: "known", value: { model, config: { nParallel: 3 } } });
    const shown = linkEditor({ ...runtime, loadedIsGguf }, inbox).draw();
    assert.equal(find(shown, "Dialog")?.props.open, open, String(loadedIsGguf));
  }
});

test("startup intake waits for mount effects and survives strict effect replay", async () => {
  const inbox = createRunConfigInbox();
  let received = 0;
  let authChanged = () => {};
  let disposed = false;
  const handler = load<typeof LinkHandler>("link-handler.tsx", {
    "./inbox": { runConfigInbox: inbox },
    "./receive-link": {
      receiveStartupRunConfigUrl: () => received++,
      subscribeRunConfigSession: (onChange: () => void) => {
        authChanged = onChange;
        return () => {
          disposed = true;
        };
      },
    },
  });
  const { module, effects, states, render } = handler;
  const chatSearch = { new: "retained-draft" };
  const draw = () =>
    render(() => module.SharedRunConfigLinkHandler({ chatSearch }));
  assert.equal(draw(), null);
  effects[0]()?.();
  await Promise.resolve();
  assert.equal(received, 0);
  disposed = false;
  const cleanup = effects[0]();
  assert.equal(received, 0);
  await Promise.resolve();
  assert.equal(received, 1);
  inbox.submit({ id: "link", value: { config: { nParallel: 3 } } });
  const shown = find(elements(draw()), "lazy");
  assert.equal(shown?.props.pending, inbox.getSnapshot());
  assert.equal(shown?.props.chatSearch, chatSearch);
  authChanged();
  assert.equal(states[0], 1);
  const failure = (draw() as StubElement).props.fallback as StubElement;
  assert.equal(failure.type, "LazyImportFailure");
  call(failure, "onDismiss");
  assert.equal(draw(), null);
  cleanup?.();
  assert.equal(disposed, true);
});

test("Load and Save refuse while a shared GGUF variant is unresolved", () => {
  const page = readSrc(
    "features/model-picker/components/model-config-page.tsx",
  ).replace(/\s+/g, " ");
  for (const handler of ["handleRun", "handleSave"]) {
    const body = page.slice(page.indexOf(`const ${handler} = () => {`));
    assert.match(
      body.slice(0, 80),
      /^const \w+ = \(\) => \{ if \(sharedVariantUnresolved\) \{ return; \}/,
    );
    const button = page.slice(0, page.indexOf(`onClick={${handler}}`));
    assert.match(
      button.slice(button.lastIndexOf("disabled={")),
      /^disabled=\{ sharedVariantUnresolved \|\|/,
    );
  }
});
