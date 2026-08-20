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
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useAuiState } from "@assistant-ui/react";
import { type RefObject, useEffect, useLayoutEffect, useRef } from "react";

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

  // A reply leaving (or entering) the edit textarea is a remount that no run state reports.
  // thread.tsx renders an editing assistant message as a bare <textarea> and any other one as
  // its rendered parts, so ending an edit mounts a brand new element for every code block in
  // that reply while the thread is quiet and has therefore already been released. Measured in
  // Chromium 151 on a 2,169px block: re-created in the released state it lays out at
  // streamdown's 200px `contain-intrinsic-size` fallback for exactly one frame, moving
  // everything below it by 1,969px and back; re-created under the hold it is 2,169px from the
  // first frame and nothing moves.
  //
  // useLayoutEffect, not useEffect: the attribute has to be back on the root within the commit
  // that created those elements. A passive effect runs after the browser has painted that
  // commit, and that paint IS the flicker frame.
  const editingMessageId = useChatRuntimeStore((s) => s.editingMessageId);
  const seenEditingRef = useRef(false);
  useLayoutEffect(() => {
    // The first run is the mount, which is already held, and where the controller-owning
    // effect below has not run yet anyway.
    if (!seenEditingRef.current) {
      seenEditingRef.current = true;
      return;
    }
    controllerRef.current?.remeasure();
  }, [editingMessageId]);

  return null;
}
