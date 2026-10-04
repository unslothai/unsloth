// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Radix closes an `AlertDialogPrimitive.Action` dialog on click unless the handler
 * defaults the event, and SYNCHRONOUSLY, so the close races the in-flight
 * `/api/shutdown`: the error path then re-enables a dialog already unmounted.
 * The harness models that contract rather than grepping for `preventDefault`, so
 * it still fails if the call survives and the behaviour breaks another way.
 */

import assert from "node:assert/strict";
import test from "node:test";
import {
  type StubElement,
  loadWithStubs,
  stubJsxRuntime,
} from "./helpers/module-stubs.ts";

type Outcome = { ok: boolean } | { throws: true };

/** Depth-first search for the first element whose stub type matches. */
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

/** `useState` is cell-backed across renders, so the error path's reset is observable. */
async function clickStopServer(outcome: Outcome) {
  const toasts: string[] = [];
  const fetched: string[] = [];
  const cells: unknown[] = [];
  let cursor = 0;
  let afterShutdownCalls = 0;
  let open = true;

  const body = { innerHTML: "" };
  const priorDocument = Object.getOwnPropertyDescriptor(globalThis, "document");
  Object.defineProperty(globalThis, "document", {
    value: { body },
    configurable: true,
    writable: true,
  });

  try {
    const { ShutdownDialog } = loadWithStubs<{
      ShutdownDialog: (props: {
        open: boolean;
        onOpenChange: (next: boolean) => void;
        onAfterShutdown?: () => void;
      }) => StubElement;
    }>(new URL("../src/components/shutdown-dialog.tsx", import.meta.url), {
      "react/jsx-runtime": stubJsxRuntime(),
      react: {
        useState: (initial: unknown) => {
          const slot = cursor++;
          if (cells.length <= slot) cells[slot] = initial;
          return [cells[slot], (next: unknown) => (cells[slot] = next)];
        },
      },
      "@/features/auth": {
        authFetch: async (path: string) => {
          fetched.push(path);
          if ("throws" in outcome) throw new TypeError("Failed to fetch");
          return { ok: outcome.ok };
        },
      },
      "@/shared/toast": {
        toastError: (message: string) => toasts.push(message),
      },
      "@/components/ui/alert-dialog": {
        AlertDialog: "AlertDialog",
        AlertDialogAction: "AlertDialogAction",
        AlertDialogCancel: "AlertDialogCancel",
        AlertDialogContent: "AlertDialogContent",
        AlertDialogDescription: "AlertDialogDescription",
        AlertDialogFooter: "AlertDialogFooter",
        AlertDialogHeader: "AlertDialogHeader",
        AlertDialogTitle: "AlertDialogTitle",
      },
    });

    const render = () => {
      cursor = 0;
      return ShutdownDialog({
        open,
        onOpenChange: (next: boolean) => (open = next),
        onAfterShutdown: () => afterShutdownCalls++,
      });
    };

    const action = findByType(render(), "AlertDialogAction");
    assert.ok(action, "no AlertDialogAction in shutdown-dialog.tsx");

    let defaultPrevented = false;
    const event = {
      preventDefault: () => (defaultPrevented = true),
      get defaultPrevented() {
        return defaultPrevented;
      },
    };

    const onClick = action.props.onClick as (event: unknown) => void;
    assert.equal(typeof onClick, "function", "Stop server has no onClick");
    onClick(event);

    // Radix's close, applied before the request settles. Main fails here.
    if (!defaultPrevented) open = false;

    await new Promise((resolve) => setImmediate(resolve));

    return {
      open,
      toasts,
      fetched,
      afterShutdownCalls,
      bodyReplaced: body.innerHTML !== "",
      stopping: cells[0] as boolean,
      rerendered: findByType(render(), "AlertDialogAction"),
    };
  } finally {
    if (priorDocument)
      Object.defineProperty(globalThis, "document", priorDocument);
    else delete (globalThis as { document?: unknown }).document;
  }
}

test("a rejected shutdown leaves the dialog open and retryable", async () => {
  const result = await clickStopServer({ ok: false });

  assert.deepEqual(result.fetched, ["/api/shutdown"]);
  assert.deepEqual(result.toasts, ["Failed to shut down server"]);
  assert.equal(result.open, true, "dialog closed under the error toast");
  assert.equal(result.stopping, false, "Stop server left disabled");
  assert.equal(
    result.rerendered?.props.disabled,
    false,
    "Stop server not clickable again",
  );
  assert.equal(result.bodyReplaced, false);
  assert.equal(result.afterShutdownCalls, 0);
});

test("an unreachable server leaves the dialog open and retryable", async () => {
  const result = await clickStopServer({ throws: true });

  assert.deepEqual(result.toasts, ["Could not reach server"]);
  assert.equal(result.open, true, "dialog closed under the error toast");
  assert.equal(result.stopping, false, "Stop server left disabled");
  assert.equal(result.bodyReplaced, false);
  assert.equal(result.afterShutdownCalls, 0);
});

test("an accepted shutdown still tears the page down", async () => {
  const result = await clickStopServer({ ok: true });

  assert.deepEqual(result.toasts, []);
  assert.equal(result.afterShutdownCalls, 1, "beforeunload never released");
  assert.equal(result.bodyReplaced, true, "stopped page never rendered");
  assert.equal(result.stopping, true, "Stop server re-enabled on the way out");
});
