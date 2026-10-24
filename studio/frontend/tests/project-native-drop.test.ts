// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "@typescript/typescript6";
import { create } from "zustand";

import * as dropPaths from "../src/features/native-intents/drop-paths.ts";
import type { NativeIntent } from "../src/features/native-intents/types.ts";
import type { useNativeModelDrop } from "../src/features/native-intents/use-native-drop.ts";
import {
  loadWithStubs,
  stubJsxRuntime,
  type StubElement,
} from "./helpers/module-stubs.ts";

type DropOptions = Parameters<typeof useNativeModelDrop>[0];
type View = {
  mode: "project" | "single" | "compare";
  projectId?: string;
  threadId?: string;
  newThreadNonce?: string;
  pairId?: string;
};
const source = ts.createSourceFile(
  "chat-page.tsx",
  readFileSync(
    new URL("../src/features/chat/chat-page.tsx", import.meta.url),
    "utf8",
  ),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);

function findNode(predicate: (node: ts.Node) => boolean): ts.Node {
  let result: ts.Node | undefined;
  function visit(node: ts.Node) {
    if (!result && predicate(node)) result = node;
    if (!result) ts.forEachChild(node, visit);
  }
  visit(source);
  assert.ok(result, "expected ChatPage's native-drop wiring to exist");
  return result;
}

function declaration(name: string): string {
  return `const ${findNode((node) => ts.isVariableDeclaration(node) && node.name.getText(source) === name).getText(source)};`;
}

// Execute the shipped callbacks/options and JSX, rather than repeating their
// predicates in the test. ChatPage's unrelated services need no DOM or mocks.
function evaluate<T>(body: string, bindings: Record<string, unknown>): T {
  const { outputText } = ts.transpileModule(body, {
    fileName: "chat-drop-fragment.tsx",
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.CommonJS,
      jsx: ts.JsxEmit.ReactJSX,
    },
  });
  return new Function(
    "require",
    "exports",
    ...Object.keys(bindings),
    outputText,
  )(() => stubJsxRuntime(), {}, ...Object.values(bindings)) as T;
}

function pageOptions(view: View, nonce: string, store: unknown): DropOptions {
  const keyStart = findNode(
    (node) =>
      ts.isVariableDeclaration(node) &&
      node.name.getText(source) === "artifactViewKey",
  ).parent.parent.getStart(source);
  const keyEnd = findNode(
    (node) =>
      ts.isVariableDeclaration(node) &&
      node.name.getText(source) === "attachmentScope",
  ).parent.parent.end;
  const names = [
    "handleNativeImageDrop",
    "handleNativeDocumentDrop",
    "handleNativeAudioDrop",
    "handleNativeVideoDrop",
    "nativeModelDropState",
  ];
  return evaluate(
    `${source.text.slice(keyStart, keyEnd)}\n${names.map(declaration).join("\n")} return nativeModelDropState;`,
    {
      view,
      search: {},
      projectNewThreadNonce: nonce,
      active: true,
      nativePathLeasesSupported: true,
      hasActiveModel: false,
      loadingModel: null,
      modelLoading: false,
      useCallback: (callback: unknown) => callback,
      useNativeIntentStore: store,
      useNativeModelDrop: (options: DropOptions) => options,
      handleNativeModelDropAutoLoad: () => {
        throw new Error("unexpected model load");
      },
    },
  );
}

function consumerContexts(view: View, targetKey: string, dragging = true) {
  const runtime = findNode(
    (node) =>
      ts.isJsxElement(node) &&
      node.openingElement.tagName.getText(source) === "ChatRuntimeProvider" &&
      node.openingElement.attributes.properties.some(
        (prop) =>
          ts.isJsxAttribute(prop) &&
          prop.name.getText(source) === "modelType" &&
          prop.initializer?.getText(source) === '"base"',
      ),
  );
  const rendered = evaluate<StubElement>(
    `return (${runtime.getText(source)});`,
    {
      ChatRuntimeProvider: "runtime",
      baseView: view,
      projectNewThreadNonce: "draft",
      baseBackgrounded: false,
      markProjectRuntimeReady: () => {},
      handleCompactionCountChange: () => {},
      NativeAttachmentTargetContext: { Provider: "target" },
      NativeAttachmentOwnerScopeContext: { Provider: "owner" },
      baseAttachmentOwnerScope: view.mode === "project" ? targetKey : null,
      NativeAttachmentDragContext: { Provider: "drag" },
      baseAttachmentTargetKey: targetKey,
      nativeModelDropState: { status: dragging ? "attach" : "idle" },
      ProjectLanding: "project",
      SingleContent: "single",
      TemporaryChatSaveBridge: "temporary-save",
      currentProject: { name: "Project" },
      currentProjectItems: [],
      rotateProjectNewThreadNonce: () => {},
      currentProjectItemsLoaded: true,
      projectsLoading: false,
      projectRuntimeReady: true,
    },
  );
  const found: Array<{
    component: unknown;
    target: unknown;
    dragging: unknown;
    ownerScope: unknown;
  }> = [];
  function walk(value: unknown, target?: unknown, drag?: unknown, ownerScope?: unknown) {
    if (Array.isArray(value)) {
      for (const child of value) walk(child, target, drag, ownerScope);
      return;
    }
    if (!value || typeof value !== "object" || !("props" in value)) return;
    const node = value as StubElement;
    if (node.type === "target") target = node.props.value;
    if (node.type === "drag") drag = node.props.value;
    if (node.type === "owner") ownerScope = node.props.value;
    if (node.type === "project" || node.type === "single")
      found.push({ component: node.type, target, dragging: drag, ownerScope });
    walk(node.props.children, target, drag, ownerScope);
  }
  walk(rendered);
  return found;
}

function intent(path: string): NativeIntent {
  return {
    id: path,
    kind: "attachment",
    sourceKind: "drop",
    displayLabel: path,
    path: {
      token: path,
      kind: "attachment",
      displayLabel: path,
      allowedOperations: ["attach"],
      expiresAtMs: Date.now() + 60_000,
    },
  };
}

async function harness(
  view: View,
  nonce = "draft-one",
  register = async (path: string) => intent(path),
) {
  const { useNativeIntentStore: realStore } = loadWithStubs<
    typeof import("../src/features/native-intents/store.ts")
  >(
    new URL("../src/features/native-intents/store.ts", import.meta.url),
    { zustand: { create } },
    { relativePassthrough: true },
  );
  const store = Object.assign(
    (select: (state: ReturnType<typeof realStore.getState>) => unknown) =>
      select(realStore.getState()),
    { getState: realStore.getState },
  );
  let options = pageOptions(view, nonce, store);
  const optionsRef = { current: options };
  let overlay: ReturnType<typeof useNativeModelDrop> = { status: "idle" };
  let mounted = false;
  let dispose: (() => void) | undefined;
  const errors: string[] = [];
  let deliver!: (event: {
    payload: {
      type: string;
      paths: string[];
      position: { x: number; y: number };
    };
  }) => Promise<void>;
  let installed!: () => void;
  const ready = new Promise<void>((resolve) => {
    installed = resolve;
  });
  const hook = loadWithStubs<{ useNativeModelDrop: typeof useNativeModelDrop }>(
    new URL(
      "../src/features/native-intents/use-native-drop.ts",
      import.meta.url,
    ),
    {
      "@/lib/api-base": { isTauri: true },
      "@/lib/toast": {
        toast: { error: (message: string) => errors.push(message) },
      },
      react: {
        useRef: () => optionsRef,
        useState: () => [
          overlay,
          (
            next: typeof overlay | ((prev: typeof overlay) => typeof overlay),
          ) => {
            overlay = typeof next === "function" ? next(overlay) : next;
          },
        ],
        useEffect: (effect: () => (() => void) | undefined) => {
          if (!mounted) {
            mounted = true;
            dispose = effect();
          }
        },
      },
      "./api": {
        registerNativeAttachmentPath: register,
        registerNativeModelPath: () => {
          throw new Error("unexpected model registration");
        },
      },
      "./drop-paths": dropPaths,
      "./native-drop-targets": { nativeDropTargetAt: () => null },
      "./store": { useNativeIntentStore: store },
      "@tauri-apps/api/webview": {
        getCurrentWebview: () => ({
          onDragDropEvent: (callback: typeof deliver) => {
            deliver = callback;
            installed();
            return Promise.resolve(() => {});
          },
        }),
      },
    },
  );
  hook.useNativeModelDrop(options);
  await ready;
  return {
    store: realStore,
    errors,
    get options() {
      return options;
    },
    get overlay() {
      return overlay;
    },
    dispatch: (type: string, paths: string[]) =>
      deliver({ payload: { type, paths, position: { x: 0, y: 0 } } }),
    navigate(nextView: View, nextNonce = nonce) {
      options = pageOptions(nextView, nextNonce, store);
      hook.useNativeModelDrop(options);
    },
    dispose: () => dispose?.(),
  };
}

for (const [label, view] of [
  ["project new chat", { mode: "project", projectId: "project-one" }],
  [
    "standalone new chat",
    { mode: "single", newThreadNonce: "standalone-draft" },
  ],
  ["existing single chat", { mode: "single", threadId: "existing-thread" }],
] as const) {
  test(`${label} accepts files and provides the queue target and drag state to its composer`, async () => {
    const h = await harness(view);
    try {
      const key = h.options.attachmentTargetKey!;
      assert.deepEqual(consumerContexts(view, key), [
        {
          component: view.mode,
          target: key,
          dragging: true,
          ownerScope: view.mode === "project" ? key : null,
        },
      ]);
      const paths = [
        "/tmp/notes.pdf",
        "/tmp/photo.png",
        "/tmp/voice.wav",
        "/tmp/movie.mp4",
      ];
      await h.dispatch("enter", paths);
      assert.deepEqual(h.overlay, { status: "attach" });
      await h.dispatch("drop", paths);
      assert.deepEqual(h.errors, []);
      const state = h.store.getState();
      assert.deepEqual(
        state.takeOpenDocumentAttachments(key).map((item) => item.id),
        paths.slice(0, 2),
      );
      assert.deepEqual(
        state.takeAudioAttachments(key).map((item) => item.id),
        [paths[2]],
      );
      assert.deepEqual(
        state.takeVideoAttachments(key).map((item) => item.id),
        [paths[3]],
      );
      assert.equal(
        state.registeringImageDrops +
          state.registeringAudioDrops +
          state.registeringVideoDrops,
        0,
      );
      await h.dispatch("drop", ["/tmp/another-photo.png"]);
      assert.deepEqual(
        h.store
          .getState()
          .takeImageAttachments(key)
          .map((item) => item.id),
        ["/tmp/another-photo.png"],
      );
    } finally {
      h.dispose();
    }
  });
}

test("Compare rejects attachments and models before registering a path", async () => {
  let registrations = 0;
  const h = await harness(
    { mode: "compare", pairId: "comparison" },
    "draft",
    async (path) => {
      registrations += 1;
      return intent(path);
    },
  );
  try {
    await h.dispatch("enter", ["/tmp/notes.pdf"]);
    assert.equal(h.overlay.status, "invalid");
    await h.dispatch("drop", ["/tmp/notes.pdf"]);
    await h.dispatch("drop", ["/tmp/model.gguf"]);
    assert.equal(registrations, 0);
    assert.deepEqual(h.errors, [
      h.options.dropsUnsupportedReason,
      h.options.dropsUnsupportedReason,
    ]);
  } finally {
    h.dispose();
  }
});

for (const [label, nextView, nextNonce] of [
  [
    "another project",
    { mode: "project", projectId: "project-two" },
    "draft-one",
  ],
  [
    "another draft in the same project",
    { mode: "project", projectId: "project-one" },
    "draft-two",
  ],
] as const) {
  test(`an in-flight drop cannot attach to ${label}`, async () => {
    let release!: () => void;
    const registering = new Promise<void>((resolve) => {
      release = resolve;
    });
    const h = await harness(
      { mode: "project", projectId: "project-one" },
      "draft-one",
      async (path) => {
        await registering;
        return intent(path);
      },
    );
    try {
      const originalKey = h.options.attachmentTargetKey!;
      const pending = h.dispatch("drop", ["/tmp/notes.pdf"]);
      assert.equal(h.store.getState().registeringImageDrops, 1);
      h.navigate(nextView, nextNonce);
      const nextKey = h.options.attachmentTargetKey!;
      assert.notEqual(originalKey, nextKey);
      release();
      await pending;
      assert.deepEqual(
        h.store.getState().takeOpenDocumentAttachments(nextKey),
        [],
      );
      assert.deepEqual(
        h.store
          .getState()
          .takeOpenDocumentAttachments(originalKey)
          .map((item) => item.id),
        ["/tmp/notes.pdf"],
      );
      assert.deepEqual(h.errors, []);
    } finally {
      release();
      h.dispose();
    }
  });
}


// Re-evaluate the shipped render fragment with one persistent ref, as ChatPage
// does while the same mounted runtime changes routes.
function ownerScopeRenders() {
  const start = findNode(
    (node) =>
      ts.isVariableDeclaration(node) &&
      node.name.getText(source) === "nativeAttachmentScopesByThread",
  ).parent.parent.getStart(source);
  const end = findNode(
    (node) =>
      ts.isVariableDeclaration(node) &&
      node.name.getText(source) === "baseAttachmentOwnerScope",
  ).parent.parent.end;
  const remembered = { current: new Map<string, string>() };
  return {
    remembered: remembered.current,
    render({
      baseView,
      baseAttachmentTargetKey,
      projectChatShownId = null,
      nativeAttachmentTargetKey = baseAttachmentTargetKey,
    }: {
      baseView: View | null;
      baseAttachmentTargetKey: string;
      projectChatShownId?: string | null;
      nativeAttachmentTargetKey?: string;
    }): string | null {
      return evaluate<string | null>(
        `${source.text.slice(start, end)} return baseAttachmentOwnerScope;`,
        {
          useRef: () => remembered,
          baseView,
          baseAttachmentTargetKey,
          projectChatShownId,
          nativeAttachmentTargetKey,
        },
      );
    },
  };
}

test("project owners remember saved threads without binding an unsaved draft", () => {
  const h = ownerScopeRenders();
  const first = {
    baseView: { mode: "project", projectId: "first-project" } as View,
    baseAttachmentTargetKey: "project:first-project:first-draft",
  };
  assert.equal(h.render(first), first.baseAttachmentTargetKey);
  assert.equal(h.remembered.size, 0);

  assert.equal(
    h.render({ ...first, projectChatShownId: "first-thread" }),
    first.baseAttachmentTargetKey,
  );
  assert.equal(h.remembered.get("first-thread"), first.baseAttachmentTargetKey);
  assert.equal(
    h.render({
      baseView: { mode: "single", threadId: "first-thread" },
      baseAttachmentTargetKey: "single:first-thread",
    }),
    first.baseAttachmentTargetKey,
  );

  const second = {
    baseView: { mode: "project", projectId: "second-project" } as View,
    baseAttachmentTargetKey: "project:second-project:second-draft",
    projectChatShownId: "second-thread",
  };
  assert.equal(h.render(second), second.baseAttachmentTargetKey);
  assert.equal(h.remembered.get("second-thread"), second.baseAttachmentTargetKey);
  assert.equal(
    h.render({
      baseView: { mode: "single", threadId: "first-thread" },
      baseAttachmentTargetKey: "single:first-thread",
    }),
    first.baseAttachmentTargetKey,
  );
  assert.equal(h.remembered.size, 2);
});

test("Compare keeps the background project's owner while unrelated singles remain unscoped", () => {
  const h = ownerScopeRenders();
  const baseView: View = { mode: "project", projectId: "project" };
  const baseAttachmentTargetKey = "project:project:original-draft";
  h.render({ baseView, baseAttachmentTargetKey, projectChatShownId: "saved" });
  assert.equal(
    h.render({
      baseView,
      baseAttachmentTargetKey,
      nativeAttachmentTargetKey: "compare:comparison",
    }),
    baseAttachmentTargetKey,
  );
  assert.equal(h.remembered.get("saved"), baseAttachmentTargetKey);
  assert.equal(
    h.render({
      baseView: { mode: "single", threadId: "saved" },
      baseAttachmentTargetKey: "single:saved",
      nativeAttachmentTargetKey: "compare:comparison",
    }),
    baseAttachmentTargetKey,
  );
  for (const unrelated of [
    null,
    { mode: "single", threadId: "unrelated" },
    { mode: "single", newThreadNonce: "new-standalone" },
  ] as Array<View | null>) {
    assert.equal(
      h.render({
        baseView: unrelated,
        baseAttachmentTargetKey: "single:unrelated",
      }),
      null,
    );
  }
});
