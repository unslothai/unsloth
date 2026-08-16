// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

/**
 * The React half of the code-block layout hold. See ./code-block-layout.ts for what it decides
 * and why; this file only wires that decision to the thread root's DOM attribute.
 *
 * Separate file so the decision can be unit-tested without pulling assistant-ui and React into
 * the test runner, which strips types rather than bundling.
 */

import {
  CODE_BLOCK_LAYOUT_ATTRIBUTE,
  type CodeBlockLayoutController,
  createCodeBlockLayoutController,
} from "@/components/assistant-ui/code-block-layout";
import { useAuiState } from "@assistant-ui/react";
import { type RefObject, useEffect, useRef } from "react";

/**
 * Writes `data-code-block-layout` onto the thread root.
 *
 * Renders nothing. It is mounted inside ThreadPrimitive.Root and handed a ref to it, so the only
 * component that re-renders when the thread starts or stops running is this one. Putting the
 * attribute on the root as a rendered prop instead would subscribe the component that owns
 * ThreadPrimitive.Root to the run state, and that component is memoised precisely so a render of
 * it does not reconcile the whole message list.
 */
export function CodeBlockLayoutSignal({
  rootRef,
  settleMs,
}: {
  rootRef: RefObject<HTMLElement | null>;
  settleMs?: number;
}): null {
  const isRunning = useAuiState(({ thread }) => thread.isRunning);
  const controllerRef = useRef<CodeBlockLayoutController | null>(null);

  useEffect(() => {
    // Captured once. React clears refs before running an unmounting element's cleanup, so
    // reading rootRef.current in the cleanup below would find null and leave the attribute
    // saying "settled" on a root that no longer has anything driving it.
    const root = rootRef.current;
    const controller = createCodeBlockLayoutController({
      settleMs,
      onChange: (layout) => {
        root?.setAttribute(CODE_BLOCK_LAYOUT_ATTRIBUTE, layout);
      },
    });
    controllerRef.current = controller;
    // Written before anything is armed, so the DOM never says something the controller does not.
    // Its absence already means held, so this is about legibility rather than correctness.
    root?.setAttribute(CODE_BLOCK_LAYOUT_ATTRIBUTE, controller.layout());
    return () => {
      controller.dispose();
      controllerRef.current = null;
      root?.setAttribute(CODE_BLOCK_LAYOUT_ATTRIBUTE, "building");
    };
  }, [rootRef, settleMs]);

  useEffect(() => {
    controllerRef.current?.setRunning(isRunning);
  }, [isRunning]);

  return null;
}
