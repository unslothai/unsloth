// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// radix reports a toast click as outside the dialog, so a modal dialog closed under its action.

import assert from "node:assert/strict";
import test from "node:test";
import {
  type StubElement,
  loadWithStubs,
  stubJsxRuntime,
} from "./helpers/module-stubs.ts";

type OutsideEvent = {
  target: { closest: (selector: string) => unknown };
  defaultPrevented: boolean;
  preventDefault: () => void;
};

// depth-first search for the first element whose stub type matches
function findByType(node: unknown, type: string): StubElement | null {
  if (Array.isArray(node)) {
    for (const child of node) {
      const hit = findByType(child, type);
      if (hit) return hit;
    }
    return null;
  }
  if (!node || typeof node !== "object") return null;
  const element = node as StubElement;
  if (element.type === type) return element;
  return findByType((element.props ?? {}).children, type);
}

function outsideEvent(insideToaster: boolean): OutsideEvent {
  return {
    target: {
      closest: (selector) =>
        insideToaster && selector === "[data-sonner-toaster]" ? {} : null,
    },
    defaultPrevented: false,
    preventDefault() {
      this.defaultPrevented = true;
    },
  };
}

test("an interaction on a toast does not dismiss the dialog under it", () => {
  const { DialogContent } = loadWithStubs<{
    DialogContent: (props: {
      onInteractOutside: (event: OutsideEvent) => void;
    }) => StubElement;
  }>(new URL("../src/components/ui/dialog.tsx", import.meta.url), {
    "react/jsx-runtime": stubJsxRuntime(),
    react: {
      createContext: () => ({ Provider: "Provider" }),
      useContext: () => null,
    },
    "radix-ui": {
      Dialog: new Proxy({}, { get: (_, part) => `Dialog.${String(part)}` }),
    },
    "@/components/app-readiness": { AppPortalGate: "AppPortalGate" },
    "@/components/ui/button": { Button: "Button" },
    "@/lib/utils": { cn: (...classes: unknown[]) => classes.join(" ") },
    "@hugeicons/core-free-icons": { Cancel01Icon: {} },
    "@hugeicons/react": { HugeiconsIcon: "HugeiconsIcon" },
  });

  const seen: OutsideEvent[] = [];
  const content = findByType(
    DialogContent({ onInteractOutside: (event) => seen.push(event) }),
    "Dialog.Content",
  );
  assert.ok(content);
  const onInteractOutside = content.props.onInteractOutside as (
    event: OutsideEvent,
  ) => void;

  const onToast = outsideEvent(true);
  const elsewhere = outsideEvent(false);
  onInteractOutside(onToast);
  onInteractOutside(elsewhere);

  assert.equal(onToast.defaultPrevented, true, "a toast click closed the dialog");
  assert.equal(elsewhere.defaultPrevented, false, "a real outside click must still dismiss");
  assert.deepEqual(seen, [onToast, elsewhere], "the caller's handler was dropped");
});

test("an interaction on a toast does not dismiss the sheet under it", () => {
  const { SheetContent } = loadWithStubs<{
    SheetContent: (props: {
      onInteractOutside: (event: OutsideEvent) => void;
      showCloseButton: boolean;
    }) => StubElement;
  }>(new URL("../src/components/ui/sheet.tsx", import.meta.url), {
    "react/jsx-runtime": stubJsxRuntime(),
    react: { useState: () => [null, () => {}] },
    "radix-ui": {
      Dialog: new Proxy({}, { get: (_, part) => `Dialog.${String(part)}` }),
    },
    "@/components/app-readiness": { AppPortalGate: "AppPortalGate" },
    "@/components/ui/button": { Button: "Button" },
    "@/components/ui/dialog": {
      DialogPortalContainerContext: { Provider: "Provider" },
    },
    "@/lib/utils": { cn: (...classes: unknown[]) => classes.join(" ") },
    "@hugeicons/core-free-icons": { Cancel01Icon: {} },
    "@hugeicons/react": { HugeiconsIcon: "HugeiconsIcon" },
  });

  const seen: OutsideEvent[] = [];
  const content = findByType(
    SheetContent({
      onInteractOutside: (event) => seen.push(event),
      showCloseButton: false,
    }),
    "Dialog.Content",
  );
  assert.ok(content);
  const onInteractOutside = content.props.onInteractOutside as (
    event: OutsideEvent,
  ) => void;

  const onToast = outsideEvent(true);
  const elsewhere = outsideEvent(false);
  onInteractOutside(onToast);
  onInteractOutside(elsewhere);

  assert.equal(onToast.defaultPrevented, true, "a toast click closed the sheet");
  assert.equal(elsewhere.defaultPrevented, false, "a real outside click must still dismiss");
  assert.deepEqual(seen, [onToast, elsewhere], "the caller's handler was dropped");
});
