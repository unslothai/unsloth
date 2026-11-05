// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createLibraryComposerAttachment } from "../src/features/library/composer-attachment.ts";
import { loadWithStubs, stubJsxRuntime, type StubElement } from "./helpers/module-stubs.ts";

type Slot = {
  value?: unknown;
  deps?: unknown[];
  cleanup?: (() => void) | void;
};

const sameDeps = (a?: unknown[], b?: unknown[]) =>
  Boolean(a && b && a.length === b.length && a.every((value, i) => Object.is(value, b[i])));
const turn = () => new Promise<void>((resolve) => setImmediate(resolve));
const file = () => new File(["saved report"], "report.txt", { type: "text/plain" });

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

function pickerHarness({
  read = async () => file(),
  add = async () => {},
}: {
  read?: () => Promise<File>;
  add?: (file: File) => Promise<void>;
} = {}) {
  type Instance = { slots: Slot[]; target: string | null; closes: number; onClose: () => void };
  let current: Instance;
  let cursor = 0;
  let effects: (() => void)[] = [];
  const slot = () => current.slots[cursor++] ??= {};
  const memo = (make: () => unknown, deps?: unknown[]) => {
    const entry = slot();
    if (!("value" in entry) || !sameDeps(entry.deps, deps)) {
      entry.value = make();
      entry.deps = deps;
    }
    return entry.value;
  };
  const effect = (run: () => (() => void) | void, deps?: unknown[]) => {
    const entry = slot();
    if (!sameDeps(entry.deps, deps)) {
      entry.deps = deps;
      effects.push(() => { entry.cleanup?.(); entry.cleanup = run(); });
    }
  };
  const react = {
    useContext: () => current.target,
    useState: (initial: unknown) => {
      const entry = slot();
      if (!("value" in entry)) entry.value = typeof initial === "function" ? initial() : initial;
      return [entry.value, (value: unknown) => {
        entry.value = typeof value === "function" ? value(entry.value) : value;
      }];
    },
    useRef: (initial: unknown) => memo(() => ({ current: initial }), []),
    useMemo: memo,
    useCallback: (callback: unknown, deps: unknown[]) => memo(() => callback, deps),
    useEffect: effect,
    useLayoutEffect: effect,
  };
  const aui = { composer: () => ({ getState: () => ({ isEditing: true }), addAttachment: add }) };
  const { LibraryAttachmentPicker } = loadWithStubs<{
    LibraryAttachmentPicker: (props: { onClose: () => void }) => StubElement;
  }>(new URL("../src/components/assistant-ui/library-attachment-picker.tsx", import.meta.url), {
    react,
    "react/jsx-runtime": stubJsxRuntime(),
    "@assistant-ui/react": { useAui: () => aui },
    "@/components/ui/command": { CommandItem: "CommandItem" },
    "@/components/ui/dialog": {},
    "@/components/ui/spinner": {},
    "@/features/auth": { getAuthSessionEpoch: () => 1 },
    "@/features/library": {
      createLibraryComposerAttachment,
      getLibrary: async () => ({ items: [{ id: "upload:report", name: "report.txt", model: null,
        sizeBytes: 12, contentType: "text/plain", source: "uploaded", updatedAt: 1 }] }),
      libraryItemFile: read,
      fileKind: () => "document",
      LibraryFileTooLarge: class extends Error {},
    },
    "@/lib/audio-utils": {},
    "@/lib/video-utils": {},
    "@/i18n": { useT: () => (key: string) => key },
  });
  return {
    mount(): Instance {
      const instance: Instance = { slots: [], target: "single:existing", closes: 0,
        onClose: () => { instance.closes += 1; } };
      return instance;
    },
    render(instance: Instance) {
      current = instance;
      cursor = 0;
      effects = [];
      const tree = LibraryAttachmentPicker({ onClose: instance.onClose });
      for (const run of effects) run();
      return tree;
    },
    unmount(instance: Instance) {
      for (const entry of instance.slots) entry.cleanup?.();
    },
  };
}

function selectedRow(tree: StubElement): StubElement {
  const visit = (value: unknown): StubElement | undefined => {
    if (Array.isArray(value)) return value.map(visit).find(Boolean);
    if (value === null || typeof value !== "object") return;
    const node = value as StubElement;
    return node.type === "CommandItem" ? node : visit(node.props?.children);
  };
  const row = visit(tree);
  assert.ok(row, "the loaded Library file must be selectable");
  return row;
}

test("leaving Chat while the picker reads a file cancels attachment and closes the old picker", async () => {
  const read = deferred<File>();
  const added: File[] = [];
  const host = pickerHarness({ read: () => read.promise, add: async (value) => { added.push(value); } });
  const picker = host.mount();
  host.render(picker);
  await turn();
  (selectedRow(host.render(picker)).props.onSelect as () => void)();
  picker.target = null;
  host.render(picker);
  read.resolve(file());
  await turn();
  assert.deepEqual(added, []);
  assert.equal(picker.closes, 1);
  host.unmount(picker);
});

test("an old adapter finishing after the picker reopens does not close the replacement", async () => {
  const added = deferred<void>();
  let starts = 0;
  let open = true;
  const host = pickerHarness({ add: async () => { starts += 1; await added.promise; } });
  const old = host.mount();
  old.onClose = () => { old.closes += 1; open = false; };
  host.render(old);
  await turn();
  (selectedRow(host.render(old)).props.onSelect as () => void)();
  await turn();
  assert.equal(starts, 1, "the actual attachment adapter must have started");
  host.unmount(old);
  const replacement = host.mount();
  host.render(replacement);
  await turn();
  selectedRow(host.render(replacement));
  added.resolve();
  await turn();
  assert.equal(old.closes, 0);
  assert.equal(replacement.closes, 0);
  assert.equal(open, true);
  host.unmount(replacement);
});
