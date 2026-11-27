// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type * as SectionModule from "../src/features/settings/components/model-auto-switch-section.tsx";
import { type StubElement, loadWithStubs, stubJsxRuntime } from "./helpers/module-stubs.ts";

const UNSAVED = "settings.general.modelAutoSwitch.idleUnsaved";
const IDLE_ERROR = "settings.general.modelAutoSwitch.idleError";

Object.assign(globalThis, { window: new EventTarget() });

const Input = () => null;
let slots: { value: unknown }[] = [];
let cursor = 0;
let mounting = false;
let puts: Record<string, unknown>[] = [];
let stored: Record<string, unknown> = {};
let hold: Promise<void> | null = null;

const react = {
  useState<T>(initial: T) {
    const slot = (slots[cursor++] ??= { value: initial });
    return [slot.value, (next: T) => { slot.value = next; }];
  },
  useEffect(effect: () => void) {
    if (mounting) effect();
  },
};

const section = loadWithStubs<typeof SectionModule>(
  new URL("../src/features/settings/components/model-auto-switch-section.tsx", import.meta.url),
  {
    react,
    "react/jsx-runtime": stubJsxRuntime(),
    "@/components/ui/button": { Button: () => null },
    "@/components/ui/input": { Input },
    "@/components/ui/switch": { Switch: () => null },
    "@/i18n": { useT: () => (key: string) => key },
    "../api/openai-auto-switch": {
      loadOpenAIAutoSwitchSettings: async () => ({ ...stored }),
      updateOpenAIAutoSwitchSettings: async (update: Record<string, unknown>) => {
        puts.push(update);
        await hold;
        stored = { ...stored, ...update };
        return { ...stored };
      },
    },
    "./settings-row": { SettingsRow: () => null },
    "./settings-section": { SettingsSection: () => null },
  },
);

const flush = () => new Promise((resolve) => setImmediate(resolve));

function walk(node: unknown, found: StubElement[] = []): StubElement[] {
  if (Array.isArray(node)) {
    for (const child of node) walk(child, found);
  } else if (node && typeof node === "object" && "props" in node) {
    const element = node as StubElement;
    found.push(element);
    walk(element.props.children, found);
    walk(element.props.below, found);
  }
  return found;
}

function mount(component: (props: Record<string, never>) => unknown) {
  slots = [];
  const render = () => {
    cursor = 0;
    return walk(component({}));
  };
  mounting = true;
  render();
  mounting = false;
  return {
    input() {
      const element = render().find((node) => node.type === Input);
      assert.ok(element);
      return element.props as {
        value: string;
        onChange: (event: { target: { value: string } }) => void;
        onKeyDown?: (event: { key: string; nativeEvent: { isComposing: boolean } }) => void;
      };
    },
    shows: (text: string) => render().some((node) => node.props.children === text),
  };
}

function type(view: ReturnType<typeof mount>, value: string) {
  view.input().onChange({ target: { value } });
}

function press(view: ReturnType<typeof mount>, key = "Enter", isComposing = false) {
  view.input().onKeyDown?.({ key, nativeEvent: { isComposing } });
}

async function chat(seconds: number) {
  puts = [];
  hold = null;
  stored = { enabled: true, autoUnloadIdleSeconds: seconds, mediaAutoUnloadIdleSeconds: 0 };
  const view = mount(section.ModelAutoSwitchSection);
  await flush();
  return view;
}

test("Enter saves the chat idle field", async () => {
  const view = await chat(60);
  type(view, "0");
  press(view);
  await flush();
  assert.deepEqual(puts, [{ enabled: true, autoUnloadIdleSeconds: 0 }]);
  assert.equal(view.input().value, "0");
  assert.equal(view.shows(UNSAVED), false);
});

test("Enter on an invalid value shows the error and writes nothing", async () => {
  const view = await chat(60);
  type(view, "30");
  press(view);
  await flush();
  assert.deepEqual(puts, []);
  assert.equal(view.shows(IDLE_ERROR), true);
});

test("Enter that confirms an IME composition saves nothing", async () => {
  const view = await chat(60);
  type(view, "0");
  press(view, "Enter", true);
  await flush();
  assert.deepEqual(puts, []);
});

test("other keys save nothing", async () => {
  const view = await chat(60);
  type(view, "0");
  press(view, "Tab");
  await flush();
  assert.deepEqual(puts, []);
});

test("the row says when the typed value is not the saved one", async () => {
  const view = await chat(60);
  assert.equal(view.shows(UNSAVED), false);
  type(view, "0");
  assert.equal(view.shows(UNSAVED), true);
  type(view, "060");
  assert.equal(view.shows(UNSAVED), false);
});

test("the unsaved note stays away while the save is running", async () => {
  const view = await chat(60);
  let release = () => {};
  hold = new Promise((resolve) => { release = resolve; });
  type(view, "0");
  press(view);
  assert.equal(view.shows(UNSAVED), false);
  release();
  await flush();
  assert.equal(view.shows(UNSAVED), false);
});

test("Enter while a save is running writes nothing more", async () => {
  const view = await chat(60);
  let release = () => {};
  hold = new Promise((resolve) => { release = resolve; });
  type(view, "0");
  press(view);
  press(view);
  press(view);
  release();
  await flush();
  assert.equal(puts.length, 1);
});

test("Enter in a field that is off writes nothing", async () => {
  await chat(60);
  stored.enabled = false;
  const off = mount(section.ModelAutoSwitchSection);
  await flush();
  type(off, "0");
  press(off);
  await flush();
  assert.deepEqual(puts, []);
});

test("Enter saves the media idle field alone", async () => {
  puts = [];
  hold = null;
  stored = { enabled: true, autoUnloadIdleSeconds: 60, mediaAutoUnloadIdleSeconds: 120 };
  const Row = section.MediaIdleUnloadRow;
  const view = mount((props) => {
    const element = section.MediaIdleUnloadSetting(props) as unknown as StubElement;
    assert.equal(element.type, Row);
    return Row(element.props as Parameters<typeof Row>[0]);
  });
  await flush();
  assert.equal(view.input().value, "120");
  type(view, "0");
  press(view);
  await flush();
  assert.deepEqual(puts, [{ mediaAutoUnloadIdleSeconds: 0 }]);
});
