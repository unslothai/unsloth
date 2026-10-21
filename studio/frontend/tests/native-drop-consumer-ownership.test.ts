// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "@typescript/typescript6";
import { create } from "zustand";

import type { NativeIntent } from "../src/features/native-intents/types.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const source = ts.createSourceFile(
  "thread.tsx",
  readFileSync(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
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
  assert.ok(result, "expected the native attachment consumer source");
  return result;
}

function evaluate<T>(body: string, bindings: Record<string, unknown>): T {
  const { outputText } = ts.transpileModule(body, {
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.CommonJS,
    },
  });
  return new Function(...Object.keys(bindings), outputText)(
    ...Object.values(bindings),
  ) as T;
}

type Kind = "Image" | "OpenDocument" | "Audio" | "Video";
type Destination = {
  key: string;
  draft: string;
  project: string | null;
  runtimeId: string;
};
const original: Destination = {
  key: "project:project-a:draft-one",
  draft: "draft-one",
  project: "project-a",
  runtimeId: "__LOCALID_shared",
};

// The real owner assignment is evaluated separately from the effects because
// React completes render (including this assignment) before running effects.
function ownerFor(destination: Destination): string {
  const declaration = findNode(
    (node) =>
      ts.isVariableDeclaration(node) &&
      node.name.getText(source) === "nativeAttachmentOwner",
  );
  const assignment = findNode(
    (node) =>
      ts.isBinaryExpression(node) &&
      node.left.getText(source) === "nativeAttachmentOwnerRef.current",
  );
  assert.ok(ts.isBinaryExpression(assignment));
  return evaluate(
    `const ${declaration.getText(source)}; return (${assignment.right.getText(source)});`,
    {
      threadListItemId: destination.runtimeId,
      nativeAttachmentOwnerScope: destination.project
        ? `project:${destination.project}:${destination.draft}`
        : null,
    },
  );
}

function drainEffect(kind: Kind): string {
  const node = findNode(
    (node) =>
      ts.isCallExpression(node) &&
      node.expression.getText(source) === "useEffect" &&
      node.arguments[0]?.getText(source).includes(`.take${kind}Attachments(`),
  );
  assert.ok(ts.isCallExpression(node));
  return `return { run: ${node.arguments[0]!.getText(source)}, dependencies: ${node.arguments[1]!.getText(source)} };`;
}

const effects = Object.fromEntries(
  (["Image", "OpenDocument", "Audio", "Video"] as const).map((kind) => [
    kind,
    drainEffect(kind),
  ]),
) as Record<Kind, string>;

function attachment(kind: Kind): NativeIntent {
  const name = {
    Image: "photo.png",
    OpenDocument: "notes.txt",
    Audio: "clip.wav",
    Video: "movie.mp4",
  }[kind];
  return {
    id: name,
    kind: "attachment",
    sourceKind: "drop",
    displayLabel: name,
    path: {
      token: name,
      kind: "attachment",
      displayLabel: name,
      allowedOperations: ["attach"],
      expiresAtMs: Date.now() + 60_000,
    },
  };
}

async function settle() {
  // Native reads and normalization both cross promise boundaries. A macrotask
  // lets all continuations and synchronous store subscribers settle together.
  await new Promise<void>((resolve) => setImmediate(resolve));
}

function harness(
  kind: Kind,
  stage: "native read" | "adapter add" = "native read",
  initial = original,
) {
  const { useNativeIntentStore: store } = loadWithStubs<
    typeof import("../src/features/native-intents/store.ts")
  >(
    new URL("../src/features/native-intents/store.ts", import.meta.url),
    { zustand: { create } },
    { relativePassthrough: true },
  );
  let destination = initial;
  let keyRef = { current: destination.key };
  let identityRef = { current: ownerFor(destination) };
  let composerRef = { current: destination.runtimeId };
  let cleanup: (() => void) | undefined;
  let previousDependencies: unknown[] | undefined;
  let firstRead = true;
  let firstAdd = true;
  let addAttempts = 0;
  let blocked!: () => void;
  const suspended = new Promise<void>((resolve) => {
    blocked = resolve;
  });
  let release!: () => void;
  const readGate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const attached: Array<{ key: string; name: string }> = [];
  const errors: string[] = [];
  const cancelled: string[] = [];
  const cancelledAdds = new WeakSet<File>();
  const materializing = new Map<string, boolean>();
  const aui = {
    composer: () => ({
      addAttachment: async (file: File) => {
        cancelledAdds.delete(file);
        addAttempts += 1;
        assert.ok(
          addAttempts <= 3,
          "a cancelled add must not loop through the same drain",
        );
        const owner = ownerFor(destination);
        if (firstAdd && stage === "adapter add") {
          firstAdd = false;
          blocked();
          await readGate;
          // The guarded adapter settles without mutating a different owner.
          // Its own generation checks have separate integration coverage.
          if (ownerFor(destination) !== owner) {
            cancelledAdds.add(file);
            return;
          }
        }
        attached.push({ key: destination.key, name: file.name });
      },
    }),
  };
  const intent = attachment(kind);

  function mount(next: Destination, remount = false) {
    destination = next;
    if (remount) {
      cleanup?.();
      cleanup = undefined;
      previousDependencies = undefined;
      keyRef = { current: next.key };
      identityRef = { current: ownerFor(next) };
      composerRef = { current: next.runtimeId };
    } else {
      keyRef.current = next.key;
      identityRef.current = ownerFor(next);
      composerRef.current = next.runtimeId;
    }
    const bindings = {
      nativeAttachmentTargetKey: next.key,
      nativeAttachmentTargetKeyRef: keyRef,
      composerIdentityRef: composerRef,
      nativeAttachmentOwnerRef: identityRef,
      nativeAttachmentOwner: identityRef.current,
      wasAttachmentAddCancelled: (file: File) => cancelledAdds.has(file),
      useNativeIntentStore: store,
      aui,
      cancelQueuedSendRef: { current: () => cancelled.push(destination.key) },
      toast: { error: (message: string) => errors.push(message) },
      nativeAttachmentIntentToFile: async (item: NativeIntent) => {
        if (firstRead) {
          firstRead = false;
          if (stage === "native read") {
            blocked();
            await readGate;
          }
        }
        return new File(["fixture"], item.displayLabel);
      },
      normalizeChatImage: async (file: File) => file,
      setMaterializingDroppedImages: (value: boolean) =>
        materializing.set(next.key, value),
      setMaterializingDroppedOpenDocuments: (value: boolean) =>
        materializing.set(next.key, value),
      setMaterializingDroppedAudio: (value: boolean) =>
        materializing.set(next.key, value),
      setMaterializingDroppedVideo: (value: boolean) =>
        materializing.set(next.key, value),
    };
    const effect = evaluate<{
      run: () => (() => void) | undefined;
      dependencies: unknown[];
    }>(effects[kind], bindings);
    if (
      previousDependencies?.every((value, index) =>
        Object.is(value, effect.dependencies[index]),
      )
    )
      return;
    cleanup?.();
    previousDependencies = effect.dependencies;
    cleanup = effect.run();
  }

  store.getState()[`add${kind}Attachments`](initial.key, [intent]);
  mount(initial);
  assert.equal(
    firstRead,
    false,
    "the originating composer must have started reading the file",
  );
  return {
    store,
    intent,
    attached,
    errors,
    cancelled,
    materializing,
    suspended,
    get addAttempts() {
      return addAttempts;
    },
    mount,
    async finishRead() {
      release();
      await settle();
    },
    cleanup() {
      release();
      cleanup?.();
    },
    pending(key: string) {
      return store.getState()[`take${kind}Attachments`](key);
    },
  };
}

for (const kind of ["Image", "OpenDocument", "Audio", "Video"] as const) {
  for (const stage of ["native read", "adapter add"] as const) {
    for (const [label, next, remount] of [
      [
        "another project",
        {
          ...original,
          key: "project:project-b:draft-one",
          project: "project-b",
        },
        true,
      ],
      [
        "a new draft in the same project",
        { ...original, key: "project:project-a:draft-two", draft: "draft-two" },
        false,
      ],
    ] as const) {
      test(`${kind}: a delayed ${stage} stays with its draft when the runtime ID is reused for ${label}`, async () => {
        const h = harness(kind, stage);
        try {
          await h.suspended;
          h.mount(next, remount);
          await h.finishRead();
          assert.deepEqual(
            h.attached,
            [],
            "the newly opened composer must not receive the previous draft's file",
          );
          assert.deepEqual(h.pending(next.key), []);
          assert.deepEqual(
            h.pending(original.key),
            [h.intent],
            "the originating draft keeps the file available",
          );
          assert.deepEqual(h.errors, []);
          assert.deepEqual(h.cancelled, []);
        } finally {
          h.cleanup();
        }
      });
    }

    for (const remount of [false, true]) {
      test(`${kind}: first persistence ${remount ? "with a remount" : "under the mounted composer"} preserves a delayed ${stage}`, async () => {
        const h = harness(kind, stage);
        const persisted = { ...original, key: "single:persisted-thread" };
        try {
          await h.suspended;
          h.mount(persisted, remount);
          await h.finishRead();
          assert.deepEqual(h.attached, [
            { key: persisted.key, name: h.intent.displayLabel },
          ]);
          assert.deepEqual(h.pending(original.key), []);
          assert.deepEqual(h.pending(persisted.key), []);
          assert.equal(h.materializing.get(persisted.key), false);
          assert.deepEqual(h.errors, []);
        } finally {
          h.cleanup();
        }
      });
    }
  }
}
