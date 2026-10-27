// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A load that unloads another model says so once: chat in its own toast, Images and Video on the
// "Model loaded" toast. Other runtimes are resynced, never the loading one (a self-eject cancels).

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import type * as ActionModule from "../src/features/settings/lib/model-memory-action.ts";
import { useSettingsDialogStore } from "../src/features/settings/stores/settings-dialog-store.ts";
import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import { calls } from "./helpers/toast-stub.mjs";

register("./helpers/toast-resolver.mjs", import.meta.url);
// model-lifecycle-events dispatches on `window`; stand one up before importing it.
const win = new EventTarget();
(globalThis as { window?: unknown }).window = win;

const events = await import("../src/lib/model-lifecycle-events.ts");
const lib = await import("../src/lib/evicted-toast.ts");
type Kind = "chat" | "audio" | "image" | "video";
const slots = (kind: Kind, ...models: string[]) =>
  models.map((model) => ({ kind, model }));

test("evicted_slots wins, bare names read as chat, anything else is nothing", () => {
  const bad = [{ kind: "x", model: "y" }, { kind: "video", model: "" }, null];
  const mixed = {
    evicted_slots: [...slots("image", "F"), ...bad],
    evicted: ["Q"],
  };
  const junk = [null, "x", 4, {}, { evicted: "Q" }, { evicted_slots: {} }];
  const cases: [unknown, unknown][] = [
    [mixed, slots("image", "F")],
    [{ evicted: ["Q", 3, ""] }, slots("chat", "Q")],
    ...junk.map((answer): [unknown, unknown] => [answer, []]),
  ];
  for (const [answer, want] of cases) {
    assert.deepEqual(
      lib.evictedSlotsFrom(answer),
      want,
      JSON.stringify(answer),
    );
  }
});

test("the notice names what went and where to load it again", () => {
  const cases: [ReturnType<typeof slots>, string][] = [
    [slots("chat", "A"), "Select it in chat to load it back."],
    [slots("chat", "A", "B"), "Select them in chat to load them back."],
    [slots("image", "A"), "Load it again on the Images page."],
    [slots("image", "A", "B"), "Load them again on the Images page."],
    [slots("video", "A"), "Load it again on the Video page."],
    [slots("video", "A", "B"), "Load them again on the Video page."],
    [slots("audio", "A"), "Load it again on its audio page."],
    [slots("audio", "A", "B"), "Load them again on their audio pages."],
    [
      [...slots("chat", "A"), ...slots("audio", "B")],
      "Load them again where you use them.",
    ],
  ];
  for (const [evicted, description] of cases) {
    const title = `Unloaded ${evicted.map((s) => s.model).join(", ")} to make room`;
    assert.deepEqual(lib.evictedNotice(evicted), { title, description });
  }
  assert.equal(lib.evictedNotice([]), null);
  // Negative control: main's POST answer has no eviction fields, so "Model loaded" stays bare.
  assert.deepEqual(
    lib.evictedToastOptions(lib.evictedSlotsFrom({ loaded: true })),
    {},
  );
  assert.deepEqual(lib.evictedToastOptions(slots("chat", "Q")), {
    description: "Unloaded Q to make room. Select it in chat to load it back.",
  });
});

test("chat raises exactly one info toast, and none when nothing was unloaded", () => {
  calls.length = 0;
  lib.toastEvicted([]);
  lib.toastEvicted([...slots("image", "F"), ...slots("video", "L")]);
  const description = "Load them again where you use them.";
  const title = "Unloaded F, L to make room";
  assert.deepEqual(calls, [{ kind: "info", title, options: { description } }]);
});

test("the other runtimes are resynced, the loading one never is", () => {
  const seen: string[] = [];
  type Detail = CustomEvent<{ runtime: string }>;
  const record = (tag: string) => (e: Event) =>
    seen.push(`${tag}:${(e as Detail).detail.runtime}`);
  win.addEventListener(events.MODEL_EJECTED_EVENT, record("ejected"));
  win.addEventListener(events.MODEL_LIFECYCLE_EVENT, record("lifecycle"));
  const all = [
    slots("image", "A"),
    slots("video", "B"),
    slots("chat", "C"),
  ].flat();
  const cases: [ReturnType<typeof slots>, Kind, string[]][] = [
    [all, "image", ["ejected:video", "lifecycle:chat"]],
    [all, "video", ["ejected:image", "lifecycle:chat"]],
    [slots("image", "A"), "image", []],
    [slots("audio", "W"), "chat", ["lifecycle:chat"]],
  ];
  for (const [evicted, self, want] of cases) {
    seen.length = 0;
    lib.resyncEvicted(evicted, self);
    assert.deepEqual(seen, want, self);
  }
});

const media = (kind: Kind) => [
  'toast.success("Model loaded", evictedToastOptions(evicted));',
  "pendingEvicted.current = evictedSlotsFrom(answer);",
  `resyncEvicted(pendingEvicted.current, "${kind}");`,
];
const SECOND_TOAST = [
  'toast.success("Model loaded");',
  "toastEvicted(",
  "toast.info(`Unloaded",
];
const PINS: [string, string[], string[]][] = [
  ["features/images/images-page.tsx", media("image"), SECOND_TOAST],
  ["features/video/video-page.tsx", media("video"), SECOND_TOAST],
  [
    "features/chat/hooks/use-chat-model-runtime.ts",
    ["toastEvicted(evictedSlots);", 'resyncEvicted(evictedSlots, "chat");'],
    ["Select it again to load it back."],
  ],
];

for (const [path, required, forbidden] of PINS) {
  test(`${path} raises one eviction notice and never ejects itself`, () => {
    const source = readSrc(path);
    for (const pin of required)
      assert.ok(source.includes(pin), `missing: ${pin}`);
    for (const pin of forbidden)
      assert.ok(!source.includes(pin), `a second toast: ${pin}`);
  });
}

function actionFor(owner: boolean) {
  useSettingsDialogStore.setState({ open: false, scrollTarget: null });
  const url = new URL(
    "../src/features/settings/lib/model-memory-action.ts",
    import.meta.url,
  );
  return loadWithStubs<typeof ActionModule>(url, {
    "../stores/settings-dialog-store": { useSettingsDialogStore },
    "@/features/auth/account-session": { isAccountOwner: () => owner },
  }).modelMemorySettingsAction();
}

test("Open settings is the owner's only, and lands on Model memory", () => {
  assert.equal(actionFor(false), undefined);
  assert.equal(useSettingsDialogStore.getState().open, false);
  const action = actionFor(true);
  assert.equal(action?.label, "Open settings");
  action?.onClick();
  const { open, activeTab, scrollTarget } = useSettingsDialogStore.getState();
  assert.deepEqual(
    [open, activeTab, scrollTarget],
    [true, "resources", "resources-model-memory"],
  );
});
