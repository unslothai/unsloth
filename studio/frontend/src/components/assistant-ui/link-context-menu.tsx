// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { ContextMenu, ContextMenuTrigger } from "@/components/ui/context-menu";
import { authFetch } from "@/features/auth";
import { Slot } from "radix-ui";
import {
  type ComponentProps,
  type MouseEvent as ReactMouseEvent,
  type ReactElement,
  type ReactNode,
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { sandboxFilePath } from "./sandbox-files";

const WebLinkMenuContent = lazy(() =>
  import("./link-menu-content").then((module) => ({ default: module.WebLinkMenuContent })),
);
const FileMenuContent = lazy(() =>
  import("./link-menu-content").then((module) => ({ default: module.FileMenuContent })),
);

// Passed through so a menu can sit inside another asChild trigger (a dialog's).
type TriggerProps = Omit<ComponentProps<typeof ContextMenuTrigger>, "asChild" | "children">;
/** Mounted on first right-click (then replayed), so streamed links don't re-render a Radix menu per token. */
function LazyContextMenu({
  triggerProps,
  content,
  children,
}: { triggerProps: TriggerProps; content: ReactNode; children: ReactElement }) {
  const [live, setLive] = useState(false);
  const triggerRef = useRef<HTMLElement | null>(null);
  // An outer asChild trigger passes its own ref; keep both.
  const { ref: outerRef, ...restTriggerProps } = triggerProps;
  const setTrigger = useCallback(
    (node: HTMLSpanElement | null) => {
      triggerRef.current = node;
      if (typeof outerRef === "function") outerRef(node);
      else if (outerRef) outerRef.current = node;
    },
    [outerRef],
  );
  const replayRef = useRef<{ x: number; y: number } | null>(null);
  useEffect(() => {
    const point = replayRef.current;
    if (!live || !point) return;
    replayRef.current = null;
    const frame = requestAnimationFrame(() =>
      triggerRef.current?.dispatchEvent(
        new MouseEvent("contextmenu", { bubbles: true, cancelable: true, button: 2, clientX: point.x, clientY: point.y }),
      ),
    );
    return () => cancelAnimationFrame(frame);
  }, [live]);
  if (live) {
    return (
      <ContextMenu>
        <ContextMenuTrigger asChild={true} className="select-text" {...restTriggerProps} ref={setTrigger}>
          {children}
        </ContextMenuTrigger>
        {content}
      </ContextMenu>
    );
  }
  return (
    <Slot.Root
      className="select-text"
      {...triggerProps}
      onContextMenu={(event: ReactMouseEvent<HTMLElement>) => {
        triggerProps.onContextMenu?.(event as ReactMouseEvent<HTMLSpanElement>);
        if (event.defaultPrevented) return;
        event.preventDefault();
        replayRef.current = { x: event.clientX, y: event.clientY };
        setLive(true);
      }}
    >
      {children}
    </Slot.Root>
  );
}

export function WebLinkContextMenu({
  href,
  children,
  ...triggerProps
}: { href: string; children: ReactElement } & TriggerProps) {
  if (!/^https?:\/\//i.test(href)) return children;
  return (
    <LazyContextMenu triggerProps={triggerProps} content={
        <Suspense fallback={null}>
          <WebLinkMenuContent href={href} />
        </Suspense>
      }>
      {children}
    </LazyContextMenu>
  );
}

export type ContextFile = {
  name: string;
  contentType?: string;
  load: () => Promise<Blob>;
  open?: () => void;
  sandbox?: { sessionId: string; file: string };
};

export function loadSandboxFile(sessionId: string, file: string): Promise<Blob> {
  return authFetch(sandboxFilePath(sessionId, file)).then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.blob();
  });
}

export function FileContextMenu({
  file,
  children,
  ...triggerProps
}: { file: ContextFile; children: ReactElement } & TriggerProps) {
  return (
    <LazyContextMenu triggerProps={triggerProps} content={
        <Suspense fallback={null}>
          <FileMenuContent file={file} />
        </Suspense>
      }>
      {children}
    </LazyContextMenu>
  );
}
