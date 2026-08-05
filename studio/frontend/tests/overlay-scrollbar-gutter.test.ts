// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  OVERLAY_SCROLLBAR_GUTTER_VAR,
  applyOverlayScrollbarGutter,
  measureOverlayScrollbarGutter,
  watchOverlayScrollbarGutter,
} from "../src/lib/overlay-scrollbar.ts";

const PROBE_WIDTH = 60;

type Node = {
  style: { cssText: string };
  scrollTop: number;
  children: Node[];
  appendChild: (child: Node) => Node;
  offsetWidth: number;
  clientWidth: number;
  getBoundingClientRect: () => { top: number; right: number; height: number };
};

/** Whether the probe overrides a non-interactive body. */
function optsIntoHitTesting(node: Node): boolean {
  return /(^|;)\s*pointer-events\s*:\s*auto\s*(;|$)/.test(node.style.cssText);
}

/** Simulates independent scrollbar hit-test and layout widths. */
function fakeDocument({
  railPx,
  layoutPx,
  contentReachable = true,
  bodyPointerEventsNone = false,
}: {
  railPx: number;
  layoutPx: number;
  contentReachable?: boolean;
  bodyPointerEventsNone?: boolean;
}) {
  const vars = new Map<string, string>();
  const bodyChildren: Node[] = [];
  const documentElement = {
    style: {
      setProperty: (name: string, value: string) => vars.set(name, value),
      removeProperty: (name: string) => vars.delete(name),
    },
  };

  function createElement(): Node {
    const node: Node = {
      style: { cssText: "" },
      scrollTop: 0,
      children: [],
      appendChild: (child) => {
        node.children.push(child);
        return child;
      },
      offsetWidth: PROBE_WIDTH,
      clientWidth: PROBE_WIDTH - layoutPx,
      getBoundingClientRect: () => ({
        top: 0,
        right: PROBE_WIDTH,
        height: PROBE_WIDTH,
      }),
    };
    return node;
  }

  const doc = {
    createElement,
    documentElement,
    body: {
      appendChild: (child: Node) => {
        bodyChildren.push(child);
        return child;
      },
      removeChild: (child: Node) => {
        bodyChildren.splice(bodyChildren.indexOf(child), 1);
        return child;
      },
    },
    elementFromPoint: (x: number) => {
      const probe = bodyChildren[0];
      if (!probe) {
        return null;
      }
      // Radix disables pointer events on the body while a modal is open.
      if (bodyPointerEventsNone && !optsIntoHitTesting(probe)) {
        return documentElement;
      }
      // The rail occupies the trailing columns of the padding box.
      if (x >= PROBE_WIDTH - railPx) {
        return probe;
      }
      return contentReachable ? probe.children[0] : probe;
    },
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
  };

  return { doc: doc as unknown as Document, vars, bodyChildren };
}

test("an overlay scrollbar's hit strip is measured, not assumed", () => {
  // WebKitGTK 4.1 measured a 21px hit-test strip.
  const { doc, bodyChildren } = fakeDocument({ railPx: 21, layoutPx: 0 });

  assert.equal(measureOverlayScrollbarGutter(doc), 21);
  // The probe must always be removed.
  assert.deepEqual(bodyChildren, []);
});

test("a scrollbar that takes layout width reserves nothing", () => {
  // Chromium and WebView2 already displace the content.
  const { doc, vars } = fakeDocument({ railPx: 0, layoutPx: 10 });

  assert.equal(measureOverlayScrollbarGutter(doc), 0);
  applyOverlayScrollbarGutter(doc);
  assert.equal(vars.has(OVERLAY_SCROLLBAR_GUTTER_VAR), false);
});

test("an unreadable sweep leaves the layout alone rather than guessing", () => {
  const { doc, vars, bodyChildren } = fakeDocument({
    railPx: 0,
    layoutPx: 0,
    contentReachable: false,
  });

  assert.equal(applyOverlayScrollbarGutter(doc), 0);
  assert.equal(vars.has(OVERLAY_SCROLLBAR_GUTTER_VAR), false);
  assert.deepEqual(bodyChildren, []);
});

test("an open modal's pointer-events:none does not erase the gutter", () => {
  // Re-measurement can run while a modal disables body pointer events.
  const { doc, vars } = fakeDocument({
    railPx: 21,
    layoutPx: 0,
    bodyPointerEventsNone: true,
  });

  assert.equal(measureOverlayScrollbarGutter(doc), 21);
  assert.equal(applyOverlayScrollbarGutter(doc), 21);
  assert.equal(vars.get(OVERLAY_SCROLLBAR_GUTTER_VAR), "21px");
});

test("the measured width is published in px for the CSS utility", () => {
  const { doc, vars } = fakeDocument({ railPx: 21, layoutPx: 0 });

  assert.equal(applyOverlayScrollbarGutter(doc), 21);
  assert.equal(vars.get(OVERLAY_SCROLLBAR_GUTTER_VAR), "21px");
});

test("regaining focus re-measures, so a changed scrollbar setting is picked up", () => {
  // Changing the macOS scrollbar setting is followed by an app refocus.
  let railPx = 0;
  const { doc, vars } = fakeDocument({ railPx: 0, layoutPx: 0 });
  const live = doc as unknown as {
    elementFromPoint: (x: number) => unknown;
    body: { appendChild: (c: unknown) => unknown };
  };
  const bodyProbes: { children: unknown[] }[] = [];
  const appendChild = live.body.appendChild;
  live.body.appendChild = (child: unknown) => {
    bodyProbes.push(child as { children: unknown[] });
    return appendChild(child);
  };
  live.elementFromPoint = (x: number) => {
    const probe = bodyProbes[bodyProbes.length - 1];
    if (x >= PROBE_WIDTH - railPx) {
      return probe;
    }
    return probe.children[0];
  };

  const handlers = new Map<string, () => void>();
  const win = {
    document: doc,
    addEventListener: (type: string, fn: () => void) => handlers.set(type, fn),
    removeEventListener: (type: string) => handlers.delete(type),
  } as unknown as Window;

  const stop = watchOverlayScrollbarGutter(win);
  assert.equal(vars.has(OVERLAY_SCROLLBAR_GUTTER_VAR), false);

  railPx = 15;
  handlers.get("focus")?.();
  assert.equal(vars.get(OVERLAY_SCROLLBAR_GUTTER_VAR), "15px");

  // Turning overlay scrollbars off must remove the gutter again.
  railPx = 0;
  handlers.get("focus")?.();
  assert.equal(vars.has(OVERLAY_SCROLLBAR_GUTTER_VAR), false);

  stop();
  assert.equal(handlers.size, 0);
});

test("right-edge action lists reserve the gutter they publish", async () => {
  const css = await readFile(
    new URL("../src/index.css", import.meta.url),
    "utf8",
  );
  // The utility must read the variable written by the probe.
  assert.match(
    css,
    new RegExp(
      `\\.overlay-scrollbar-gutter\\s*\\{[^}]*padding-right:\\s*var\\(${OVERLAY_SCROLLBAR_GUTTER_VAR},\\s*0px\\)`,
    ),
  );

  const pickers = await readFile(
    new URL(
      "../src/features/model-picker/components/model-selector/pickers.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // Every model row must sit inside the gutter wrapper.
  assert.match(
    pickers,
    /"model-list-scroll[^"]*overflow-y-auto[^"]*"[\s\S]{0,800}"overlay-scrollbar-gutter",/,
  );

  const apiKeysTab = await readFile(
    new URL(
      "../src/features/settings/tabs/api-keys-tab.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // Preserve classic padding and move every API-key row into the gutter.
  assert.match(
    apiKeysTab,
    /"hover-scrollbar[^"]*overflow-y-auto[^"]*\bpr-1\b[^"]*"[\s\S]{0,200}<div className="overlay-scrollbar-gutter">[\s\S]{0,300}<ApiKeyRow/,
  );

  const projectSourceDropzone = await readFile(
    new URL(
      "../src/features/rag/components/project-source-dropzone.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // Keep staged-source remove actions inside the gutter.
  assert.match(
    projectSourceDropzone,
    /<ul className="[^"]*overlay-scrollbar-gutter[^"]*max-h-52[^"]*overflow-y-auto[^"]*">[\s\S]{0,1000}aria-label={`Remove \${entry\.file\.name}`}/,
  );
});
