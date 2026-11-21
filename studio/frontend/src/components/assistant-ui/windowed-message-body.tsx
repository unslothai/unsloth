// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAuiState } from "@assistant-ui/react";
import { type ReactNode, useCallback, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { isWebKitGTK } from "@/components/ui/theme-view-transition";
import { observeMessageWindow } from "./message-window-controller";
import { useAdjustForContentInsertedAbove } from "./use-intent-aware-autoscroll";

const ENABLED = typeof navigator !== "undefined"
  && isWebKitGTK(navigator.platform, navigator.userAgent);

export function WindowedMessageBody({ children, enabled = true }: {
  children: ReactNode;
  enabled?: boolean;
}) {
  return ENABLED ? <MessageBody enabled={enabled}>{children}</MessageBody> : children;
}

function MessageBody({ children, enabled }: { children: ReactNode; enabled: boolean }) {
  const messageId = useAuiState(({ message }) => message.id);
  const longThread = useAuiState(({ thread }) => thread.messages.length >= 100);
  const holder = useRef<HTMLDivElement>(null);
  // Restoration corrections go through the autoscroll owner, which keeps its intent in sync.
  const adjustAbove = useAdjustForContentInsertedAbove();
  const adjustRef = useRef(adjustAbove);
  adjustRef.current = adjustAbove;
  const [adjust] = useState(() => (deltaPx: number) => {
    adjustRef.current(deltaPx);
  });
  const [content] = useState(() => {
    const element = document.createElement("div");
    element.style.display = "contents";
    return element;
  });
  const attach = useCallback((node: HTMLDivElement | null) => {
    holder.current = node;
    if (!node) return;
    // Attach before child layout effects and the progressive-mount anchor correction.
    const root = node.closest<HTMLElement>("[data-role][data-message-id]");
    root?.setAttribute("data-message-windowed", "");
    node.append(content);
    return () => {
      content.remove();
      root?.removeAttribute("data-message-windowed");
      holder.current = null;
    };
  }, [content]);
  useLayoutEffect(() => {
    const shell = holder.current;
    const root = shell?.closest<HTMLElement>("[data-role][data-message-id]");
    if (!root || !shell) return;
    const stop = enabled && longThread ? observeMessageWindow(root, shell, content, adjust) : undefined;
    return stop;
  }, [adjust, content, enabled, longThread, messageId]);
  return (
    <>
      <div ref={attach} style={{ display: "contents" }} />
      {createPortal(children, content)}
    </>
  );
}
