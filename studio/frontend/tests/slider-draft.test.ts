// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Draft = {
  draft: number | null;
  sliderProps: {
    value: number[];
    onPointerDown: (event: { pointerId: number }) => void;
    onValueChange: (value: number[]) => void;
    onValueCommit: (value: number[]) => void;
    onLostPointerCapture: (event: { pointerId: number }) => void;
  };
};

const documentEvents = new EventTarget();
Object.defineProperty(globalThis, "document", {
  value: documentEvents,
  configurable: true,
  writable: true,
});
const teardown: (() => void)[] = [];
test.afterEach(() => {
  for (const cleanup of teardown.splice(0)) cleanup();
});

const FINGER = { pointerId: 1 };

function mountDraft(value: number, onDraft?: (value: number | null) => void) {
  const slots: unknown[] = [];
  const synced: unknown[] = [];
  const cleanups: (() => void)[] = [];
  let cursor = 0;
  let flushing = false;
  let pendingOwned: number | null = null;
  const react = {
    useState: (initial: unknown) => {
      const at = cursor++;
      if (!(at in slots)) slots[at] = initial;
      return [
        slots[at],
        (next: unknown) => {
          slots[at] = next;
          if (flushing) synced.push(next);
        },
      ];
    },
    useRef: (initial: unknown) => {
      const at = cursor++;
      if (!(at in slots)) slots[at] = { current: initial };
      return slots[at];
    },
    useEffectEvent: (handler: (event?: Event) => void) => {
      const at = cursor++;
      if (!(at in slots)) slots[at] = { handler, call: (event?: Event) => (slots[at] as { handler: (event?: Event) => void }).handler(event) };
      const event = slots[at] as { handler: (event?: Event) => void; call: (event?: Event) => void };
      event.handler = handler;
      return event.call;
    },
    useLayoutEffect: (effect: () => (() => void) | void, deps: unknown[]) => {
      const at = cursor++;
      const previous = slots[at] as unknown[] | undefined;
      if (
        previous &&
        deps.every((value, index) => Object.is(value, previous[index]))
      ) return;
      slots[at] = deps;
      const cleanup = effect();
      if (cleanup) cleanups.push(cleanup);
    },
  };
  const { useSliderDraft } = loadWithStubs<{
    useSliderDraft: (
      value: number,
      onCommit: (value: number) => void,
      onDraft?: (value: number | null) => void,
    ) => Draft;
  }>(new URL("../src/hooks/use-slider-draft.ts", import.meta.url), {
    react,
    "react-dom": {
      flushSync: (update: () => void) => {
        flushing = true;
        update();
        if (pendingOwned !== null) {
          owned = pendingOwned;
          pendingOwned = null;
        }
        SliderRow();
        flushing = false;
      },
    },
  });
  const commits: number[] = [];
  let owned = value;
  const SliderRow = () => {
    cursor = 0;
    return useSliderDraft(owned, (next) => commits.push(next), onDraft);
  };
  const unmount = () => {
    for (const cleanup of cleanups.splice(0)) cleanup();
  };
  const setOwned = (next: number) => {
    owned = next;
  };
  teardown.push(unmount);
  const queueOwned = (next: number) => {
    pendingOwned = next;
  };
  return { render: SliderRow, commits, synced, unmount, setOwned, queueOwned };
}

test("a drag moves the thumb locally and hands its owner one settled value", () => {
  const { render, commits } = mountDraft(0.5);
  render().sliderProps.onPointerDown(FINGER);
  render().sliderProps.onValueChange([0.6]);
  render().sliderProps.onValueChange([0.7]);
  const dragging = render();
  assert.equal(dragging.draft, 0.7);
  assert.deepEqual(dragging.sliderProps.value, [0.7]);
  assert.deepEqual(commits, [], "the owner re-renders once per drag, not once per pointer move");

  dragging.sliderProps.onValueCommit([0.7]);
  dragging.sliderProps.onLostPointerCapture(FINGER);
  assert.deepEqual(commits, [0.7], "Radix's own commit at release is not a second one");
  assert.equal(render().draft, null);
});

test("a release right behind the last move commits that move", () => {
  const { render, commits } = mountDraft(0.5);
  render().sliderProps.onPointerDown(FINGER);
  const row = render();
  row.sliderProps.onValueChange([0.6]);
  row.sliderProps.onValueChange([0.9]);
  row.sliderProps.onValueCommit([0.6]);
  row.sliderProps.onLostPointerCapture(FINGER);
  assert.deepEqual(commits, [0.9], "Radix commits the value its last render saw, a move behind the pointer");
});

test("each move is drawn in the frame it arrives in", () => {
  const { render, synced } = mountDraft(0.5);
  render().sliderProps.onPointerDown(FINGER);
  render().sliderProps.onValueChange([0.6]);
  render().sliderProps.onValueChange([0.9]);
  assert.deepEqual(synced, [0.6, 0.9], "a plain state update waits for a later task and trails the pointer by a frame");
});

test("a drag that ends where it started commits nothing and leaves no draft", () => {
  const { render, commits } = mountDraft(0.5);
  render().sliderProps.onPointerDown(FINGER);
  render().sliderProps.onValueChange([0.6]);
  render().sliderProps.onValueChange([0.5]);
  render().sliderProps.onLostPointerCapture(FINGER);
  assert.equal(render().draft, null);
  assert.deepEqual(commits, []);
});

test("a drag cut short keeps the value it reached, as a native range input does", () => {
  const { render, commits } = mountDraft(0.5);
  render().sliderProps.onPointerDown(FINGER);
  render().sliderProps.onValueChange([0.8]);
  render().sliderProps.onLostPointerCapture(FINGER);
  assert.deepEqual(commits, [0.8], "a pointercancel or a stolen capture threw the drag away");
  assert.equal(render().draft, null);
});

test("a drag whose row goes away keeps the value it reached, as main's live writes did", () => {
  const previews: (number | null)[] = [];
  const cut = mountDraft(0.5, (next) => previews.push(next));
  cut.render().sliderProps.onPointerDown(FINGER);
  cut.render().sliderProps.onValueChange([0.8]);
  cut.unmount();
  assert.deepEqual(cut.commits, [0.8], "Escape closed the model picker mid-drag and dropped the dragged value");
  assert.deepEqual(previews, [0.8, null], "an owner that follows the drag hears that it ended");

  const ended = mountDraft(0.5);
  ended.render().sliderProps.onPointerDown(FINGER);
  ended.render().sliderProps.onValueChange([0.8]);
  ended.render().sliderProps.onLostPointerCapture(FINGER);
  ended.unmount();
  assert.deepEqual(ended.commits, [0.8], "a drag that already ended is not committed twice");
});

test("a change from outside after the last move wins at release, as it did with main's live writes", () => {
  const held = mountDraft(0.7);
  held.render().sliderProps.onPointerDown(FINGER);
  held.render().sliderProps.onValueChange([0.9]);
  held.setOwned(0.6);
  held.render().sliderProps.onLostPointerCapture(FINGER);
  assert.deepEqual(held.commits, [], "a model load that landed while the thumb was held still was overwritten by the dragged value");
  assert.equal(held.render().draft, null);

  const moved = mountDraft(0.7);
  moved.render().sliderProps.onPointerDown(FINGER);
  moved.render().sliderProps.onValueChange([0.9]);
  moved.setOwned(0.6);
  moved.render().sliderProps.onValueChange([0.8]);
  moved.render().sliderProps.onLostPointerCapture(FINGER);
  assert.deepEqual(moved.commits, [0.8], "a move after the change is the newer intent");
});

test("a second finger on the slider keeps the drag going after the first one lifts", () => {
  const { render, commits } = mountDraft(0.5);
  render().sliderProps.onPointerDown(FINGER);
  render().sliderProps.onPointerDown({ pointerId: 2 });
  render().sliderProps.onValueChange([0.6]);
  render().sliderProps.onValueCommit([0.6]);
  render().sliderProps.onLostPointerCapture(FINGER);
  render().sliderProps.onValueChange([0.8]);
  assert.equal(render().draft, 0.8, "the second finger's moves were dropped once the first one lifted");
  assert.deepEqual(commits, []);
  render().sliderProps.onLostPointerCapture({ pointerId: 2 });
  assert.deepEqual(commits, [0.8]);
  assert.equal(render().draft, null);
});

test("an owner can follow the drag and hears when it ends", () => {
  const previews: (number | null)[] = [];
  const { render } = mountDraft(10, (next) => previews.push(next));
  render().sliderProps.onPointerDown(FINGER);
  render().sliderProps.onValueChange([12]);
  render().sliderProps.onValueChange([14]);
  render().sliderProps.onLostPointerCapture(FINGER);
  assert.deepEqual(previews, [12, 14, null]);
});

test("keyboard steps go straight to the owner", () => {
  const { render, commits } = mountDraft(4);
  render().sliderProps.onValueChange([5]);
  render().sliderProps.onValueCommit([5]);
  assert.equal(render().draft, null);
  assert.deepEqual(commits, [5]);
});

test("the Run settings and model picker sliders drag on a draft", () => {
  for (const [file, rows] of [
    ["features/chat/chat-settings-sheet.tsx", 1],
    ["features/model-picker/components/model-config-page.tsx", 3],
  ] as const) {
    const source = readSrc(file);
    assert.equal(source.match(/= useSliderDraft\(/g)?.length, rows, file);
    assert.doesNotMatch(source, /onValueChange=\{\(\[\w+\]\) => (?:onChange|setContextSliderValue)\(/, file);
  }
});

test("what an owner shows beside its slider follows the drag", () => {
  const picker = readSrc("features/model-picker/components/model-config-page.tsx");
  assert.match(picker, /onDraft=\{setDraftPercent\}/);
  assert.match(picker, /\{shownPercent !== defaultPercent && \(/, "the VRAM advice contradicted the dragged percent");
  assert.match(picker, /displayValue=\{isMlx && windowUnknown && !dragging \? "—" : undefined\}/, "an unknown MLX window showed a dash for the whole drag");
  assert.match(picker, /\{renderWarning\(shown\)\}/);
  const music = readSrc("features/audio/components/music-edit-inputs.tsx");
  assert.match(music, /onDraft=\{setDraggedExtendS\}/);
  assert.match(music, /tailS=\{action === "extend" \? \(draggedExtendS \?\? draft\.extendS\) : 0\}/, "the extend tail stopped growing while Add seconds was dragged");
  assert.match(
    readSrc("features/chat/chat-settings-sheet.tsx"),
    /const sliderProps = commitWhileDragging\s*\? \{ value: \[value\], onValueChange: \(\[v\]: number\[\]\) => commit\(v\) \}\s*: draftProps;/,
  );
  assert.match(
    readSrc("features/audio/pages/music-page.tsx"),
    /label="Variations"[\s\S]*?onChange=\{onChange\}\s*commitWhileDragging/,
    "the reload notice under Variations named the count from before the drag",
  );
  const edit = readSrc("features/audio/pages/edit-page.tsx");
  assert.match(edit, /onChange=\{\(speed\) => setDelivery\(\{ speed \}\)\}\s*commitWhileDragging/, "the Generate blocker and the edit-pass text lagged a Speed drag");
  assert.match(edit, /onChange=\{\(pitchSteps\) => setDelivery\(\{ pitchSteps \}\)\}\s*commitWhileDragging/);
});

test("a sampling change re-renders the Run settings panel, not the chat page", () => {
  const page = readSrc("features/chat/chat-page.tsx");
  assert.match(page, /const inferenceCheckpoint = useChatRuntimeStore\(\s*\(state\) => state\.params\.checkpoint,\s*\);/);
  for (const file of ["features/chat/chat-page.tsx", "features/chat/hooks/use-chat-model-runtime.ts"]) {
    assert.doesNotMatch(readSrc(file), /useChatRuntimeStore\(\(\w+\) => \w+\.params\)/, `${file} re-rendered the whole chat page for every sampling change`);
  }
  assert.match(readSrc("features/chat/chat-settings-sheet.tsx"), /const params = useChatRuntimeStore\(\(s\) => s\.params\);/);
});

test("a Run settings commit lands on the params the store holds when it is made", () => {
  assert.match(
    readSrc("features/chat/chat-settings-sheet.tsx"),
    /function set<K extends keyof InferenceParams>\(key: K\) \{\s*return \(v: InferenceParams\[K\]\) => \{\s*const nextParams = \{\s*\.\.\.useChatRuntimeStore\.getState\(\)\.params,/,
    "a slider that unmounts in the commit that switched the model commits from its last render, and that render's params would switch the model back",
  );
});

test("a track click after a number input blur commits against the flushed owner", () => {
  const held = mountDraft(0.5);
  const stale = held.render().sliderProps;
  stale.onPointerDown(FINGER);
  held.queueOwned(0.9);
  stale.onValueChange([0.3]);
  stale.onLostPointerCapture(FINGER);
  assert.deepEqual(held.commits, [0.3]);
});

test("a removed capture target releases its pointer through the document", () => {
  const held = mountDraft(0.5);
  held.render().sliderProps.onPointerDown(FINGER);
  held.render().sliderProps.onValueChange([0.7]);
  documentEvents.dispatchEvent(Object.assign(new Event("lostpointercapture"), FINGER));
  assert.deepEqual(held.commits, [0.7]);
  held.setOwned(0.7);
  held.render().sliderProps.onValueCommit([0.8]);
  assert.deepEqual(held.commits, [0.7, 0.8]);
  held.unmount();
  documentEvents.dispatchEvent(Object.assign(new Event("lostpointercapture"), FINGER));
  assert.deepEqual(held.commits, [0.7, 0.8]);
});

test("a draft callback that unmounts the row still commits the latest move", () => {
  let remove = () => {};
  const held = mountDraft(0.5, (next) => {
    if (next !== null) remove();
  });
  remove = held.unmount;
  held.render().sliderProps.onPointerDown(FINGER);
  held.render().sliderProps.onValueChange([0.8]);
  assert.deepEqual(held.commits, [0.8]);
  assert.equal(held.render().draft, null);
});
