// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs, stubJsxRuntime } from "./helpers/module-stubs.ts";

type Reply = { columns: string[]; preview_samples: Record<string, string>[] };

function harness() {
  const pending = new Map<string, (reply: Reply) => void>();
  const shared = { preview: null as unknown, dataset: "" };
  let effects: (() => void | (() => void))[] = [];
  const react = {
    useCallback: (fn: unknown) => fn,
    useEffect: (fn: () => void | (() => void)) => {
      effects.push(fn);
    },
    useRef: (value: unknown) => ({ current: value }),
    useState: (value: unknown) => [value, () => {}],
  };
  const passthrough = new Proxy({}, { get: () => "x" });
  const { RlDatasetRoles } = loadWithStubs<{
    RlDatasetRoles: (props: { objective: string }) => unknown;
  }>(new URL("../src/features/studio/sections/rl/rl-dataset-roles.tsx", import.meta.url), {
    react,
    "react/jsx-runtime": stubJsxRuntime(),
    "zustand/react/shallow": { useShallow: (fn: unknown) => fn },
    "@/components/new-badge": passthrough,
    "@/components/ui/button": passthrough,
    "@/components/ui/select": passthrough,
    "@/features/auth": { getAuthSessionEpoch: () => 0 },
    "@/features/hub": { getHfToken: () => "" },
    "@/features/training": {
      checkDatasetFormat: ({ datasetName }: { datasetName: string }) =>
        new Promise<Reply>((resolve) => pending.set(datasetName, resolve)),
      previewCell: () => "",
      useRlWorkspaceStore: (select: (s: unknown) => unknown) =>
        select({ previewRow: shared.preview, setPreviewRow: (row: unknown) => (shared.preview = row) }),
      useTrainingConfigStore: Object.assign(
        (select: (s: unknown) => unknown) =>
          select({
            datasetSource: "huggingface",
            dataset: shared.dataset,
            rlRoleMapping: {},
            setRlRoleMapping: () => {},
          }),
        { setState: () => {} },
      ),
      useTrainingRuntimeStore: (select: (s: unknown) => unknown) => select({ isStarting: false }),
      RL_IGNORE: "__ignore",
      RL_REQUIRED_ROLES: { grpo: [] },
      RL_ROLES: { grpo: [] },
      missingRlRoles: () => [],
      syncRlMapping: (_o: unknown, _c: unknown, mapping: unknown) => mapping,
    },
    "@/i18n": { useT: () => (key: string) => key },
    "@/lib/utils": { cn: (...parts: unknown[]) => parts.join(" ") },
    "@hugeicons/core-free-icons": passthrough,
    "@hugeicons/react": passthrough,
  });
  const mount = (dataset: string) => {
    shared.dataset = dataset;
    effects = [];
    RlDatasetRoles({ objective: "grpo" });
    const cleanups = effects.map((fn) => fn()).filter((c) => typeof c === "function");
    return () => cleanups.forEach((c) => (c as () => void)());
  };
  const reply = (dataset: string) =>
    pending.get(dataset)!({ columns: ["q"], preview_samples: [{ q: dataset }] });
  return { mount, reply, pending, shared };
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

test("an unmounted role panel's late reply does not replace the new dataset's preview row", async () => {
  const h = harness();
  const unmountA = h.mount("A");
  await settle();
  assert.ok(h.pending.has("A"));
  unmountA(); // GRPO -> SFT while A is still being inspected
  h.mount("B"); // dataset B picked, back on GRPO
  await settle();
  h.reply("B");
  await settle();
  h.reply("A");
  await settle();
  assert.deepEqual(h.shared.preview, { q: "B" });
});

test("a panel unmounted before its deferred load starts never inspects the dataset", async () => {
  const h = harness();
  h.mount("A")();
  await settle();
  assert.equal(h.pending.has("A"), false);
});

test("switching or clearing the dataset drops the previous dataset's preview row at once", async () => {
  const h = harness();
  h.mount("A");
  await settle();
  h.reply("A");
  await settle();
  assert.deepEqual(h.shared.preview, { q: "A" });
  h.mount("B");
  await settle();
  assert.equal(h.shared.preview, null); // B still loading: no reference from A
  h.reply("B");
  await settle();
  assert.deepEqual(h.shared.preview, { q: "B" });
  h.mount("");
  await settle();
  assert.equal(h.shared.preview, null);
});
