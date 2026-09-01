// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

class FakeElement {
  parentElement: FakeElement | null = null;
}

let hit: FakeElement | null = null;

Object.assign(globalThis, {
  HTMLElement: FakeElement,
  // devicePixelRatio 2: a physical drop position is twice the CSS position the
  // DOM is hit-tested in, which is the conversion this module owns.
  window: { devicePixelRatio: 2, location: { protocol: "http:" } },
  document: {
    elementFromPoint: () => hit,
    // setAppliedInterfaceZoom writes the mac chrome vars here on its way through.
    documentElement: { style: { setProperty: () => undefined } },
  },
});

const { nativeDropTargetAt, registerNativeDropTarget } = await import(
  "../src/features/native-intents/native-drop-targets.ts"
);
const { nativeDropPointToCss } = await import(
  "../src/features/native-intents/native-drop-position.ts"
);
const { setAppliedInterfaceZoom } = await import(
  "../src/features/settings/lib/interface-scale-runtime.ts"
);

const asElement = (value: FakeElement) => value as unknown as HTMLElement;

test("no registered target means the window handler keeps the drop", () => {
  hit = new FakeElement();
  assert.equal(nativeDropTargetAt({ x: 10, y: 10 }), null);
});

test("a drop over a registered element resolves to it", () => {
  const zone = new FakeElement();
  hit = zone;
  const unregister = registerNativeDropTarget(asElement(zone), {
    onDrop: () => undefined,
  });
  assert.equal(nativeDropTargetAt({ x: 10, y: 10 }), asElement(zone));
  unregister();
  assert.equal(nativeDropTargetAt({ x: 10, y: 10 }), null);
});

test("a drop on a child resolves to its registered ancestor", () => {
  const zone = new FakeElement();
  const child = new FakeElement();
  child.parentElement = zone;
  hit = child;
  const unregister = registerNativeDropTarget(asElement(zone), {
    onDrop: () => undefined,
  });
  assert.equal(nativeDropTargetAt({ x: 10, y: 10 }), asElement(zone));
  unregister();
});

// The bug: nested zones both matched by bounds, so the outer one could win and
// swallow a drop meant for the dialog sitting inside it.
test("the innermost registered ancestor wins", () => {
  const outer = new FakeElement();
  const inner = new FakeElement();
  const child = new FakeElement();
  inner.parentElement = outer;
  child.parentElement = inner;
  hit = child;
  const stopOuter = registerNativeDropTarget(asElement(outer), {
    onDrop: () => undefined,
  });
  const stopInner = registerNativeDropTarget(asElement(inner), {
    onDrop: () => undefined,
  });
  assert.equal(nativeDropTargetAt({ x: 10, y: 10 }), asElement(inner));
  stopInner();
  assert.equal(nativeDropTargetAt({ x: 10, y: 10 }), asElement(outer));
  stopOuter();
});

// wry reports CSS pixels on macOS (NSView points) and GTK (widget coords), and
// device pixels only on WebView2. Scaling everything by devicePixelRatio put
// every hit test at half the real position on a Retina Mac, so nothing matched.
function pointSeenFor(userAgent: string): { x: number; y: number } {
  const zone = new FakeElement();
  const seen: Array<{ x: number; y: number }> = [];
  hit = zone;
  Object.defineProperty(globalThis, "navigator", {
    value: { userAgent },
    configurable: true,
  });
  Object.assign(globalThis.document, {
    elementFromPoint: (x: number, y: number) => {
      seen.push({ x, y });
      return zone;
    },
  });
  const unregister = registerNativeDropTarget(asElement(zone), {
    onDrop: () => undefined,
  });
  nativeDropTargetAt({ x: 120, y: 80 });
  unregister();
  return seen[0];
}

test("a macOS drop position is hit-tested as-is", () => {
  assert.deepEqual(pointSeenFor("Mozilla/5.0 (Macintosh; Intel Mac OS X)"), {
    x: 120,
    y: 80,
  });
});

test("a Linux drop position is hit-tested as-is", () => {
  assert.deepEqual(pointSeenFor("Mozilla/5.0 (X11; Linux x86_64)"), {
    x: 120,
    y: 80,
  });
});

for (const userAgent of [
  "Mozilla/5.0 (Macintosh; Intel Mac OS X)",
  "Mozilla/5.0 (X11; Linux x86_64)",
]) {
  test(`${userAgent} drop positions follow webview zoom`, () => {
    Object.defineProperty(globalThis, "navigator", {
      value: { userAgent },
      configurable: true,
    });
    assert.deepEqual(nativeDropPointToCss({ x: 120, y: 80 }, 2, 0.5), {
      x: 240,
      y: 160,
    });
  });
}

// Every other case passes the zoom explicitly, which leaves the default argument
// unexercised. That default is the seam between the scale store and the drop path, and
// wiring it to the wrong getter would keep every one of those cases green.
test("the default zoom comes from the applied interface scale", () => {
  Object.defineProperty(globalThis, "navigator", {
    value: { userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X)" },
    configurable: true,
  });
  assert.deepEqual(nativeDropPointToCss({ x: 120, y: 80 }, 2), {
    x: 120,
    y: 80,
  });
  setAppliedInterfaceZoom(0.5);
  try {
    assert.deepEqual(nativeDropPointToCss({ x: 120, y: 80 }, 2), {
      x: 240,
      y: 160,
    });
  } finally {
    setAppliedInterfaceZoom(1);
  }
});

test("a Windows drop position is divided by the scale factor", () => {
  assert.deepEqual(pointSeenFor("Mozilla/5.0 (Windows NT 10.0; Win64)"), {
    x: 60,
    y: 40,
  });
});

// Webview zoom moves devicePixelRatio either side of the monitor scale, and
// elementFromPoint wants CSS pixels. Zoomed in first, then out.
for (const ratio of [3, 1.5]) {
  test(`a Windows drop position follows a devicePixelRatio of ${ratio}`, () => {
    globalThis.window.devicePixelRatio = ratio;
    try {
      assert.deepEqual(pointSeenFor("Mozilla/5.0 (Windows NT 10.0; Win64)"), {
        x: 120 / ratio,
        y: 80 / ratio,
      });
    } finally {
      globalThis.window.devicePixelRatio = 2;
    }
  });
}

// The chat-wide handler has to ask before acting, or a drop aimed at a dialog's
// own zone lands as a chat attachment behind it.
test("the chat drop handler defers to a registered target", async () => {
  const source = await readFile(
    new URL(
      "../src/features/native-intents/use-native-drop.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    source,
    /if \(nativeDropTargetAt\(event\.payload\.position\)\) \{\s*publish\(\{ status: "idle" \}\);\s*return;/,
  );
});

test("the shared image picker owns native drops and ignores stale reads", async () => {
  const source = await readFile(
    new URL("../src/components/image-dropzone.tsx", import.meta.url),
    "utf8",
  );
  assert.match(source, /const nativeDropRef = useNativeDropTarget\(\{/);
  assert.match(source, /ref=\{nativeDropRef\}/);
  assert.match(source, /registerNativeAttachmentPath\(path\)/);
  assert.match(source, /readNativeAttachmentFile\(intent\.path\.token\)/);
  // A read outliving the picker would land on whoever holds `onChange` now, and
  // the native policy takes fewer formats than the picker's own image/*.
  assert.match(source, /if \(!mounted\.current \|\| claimed !== selection\.current\) return;/);
  assert.match(source, /NATIVE_IMAGE_EXTS\.includes\(/);
  // Index-keyed reference slots keep the picker mounted when one is removed.
  assert.match(source, /if \(seen\.current === value\) return;\s*seen\.current = value;\s*selection\.current \+= 1;/);
});

// The picker rejects a format the native side would refuse anyway, so the two
// lists have to stay in step or a droppable image starts being turned away.
test("the picker's droppable formats match the native path policy", async () => {
  const picker = await readFile(
    new URL("../src/components/image-dropzone.tsx", import.meta.url),
    "utf8",
  );
  const rust = await readFile(
    new URL("../../src-tauri/src/native_path_policy.rs", import.meta.url),
    "utf8",
  );
  const listed = (source: string, pattern: RegExp) =>
    [...(source.match(pattern)?.[1].matchAll(/"([a-z0-9]+)"/g) ?? [])]
      .map((match) => match[1])
      .sort();

  assert.deepEqual(
    listed(picker, /NATIVE_IMAGE_EXTS\s*=\s*\[([^\]]+)\]/),
    listed(rust, /IMAGE_ATTACHMENT_EXTS:\s*&\[&str\]\s*=\s*&\[([^\]]+)\]/),
  );
});

// Tauri repeats "over" for every cursor move, and useNativeModelDrop sits in
// ChatPage, so an unconditional setState there rerenders the page per event.
test("the chat drop overlay only publishes a changed state", async () => {
  const source = await readFile(
    new URL(
      "../src/features/native-intents/use-native-drop.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    source,
    /setDropState\(\(prev\) => \(sameDropState\(prev, next\) \? prev : next\)\)/,
  );
  assert.doesNotMatch(source, /payload\.type !== "drop"\) \{\s*setDropState\(/);
});
