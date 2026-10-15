// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Button } from "@/components/ui/button";
import {
  resolveToolConfirmation,
  ToolApprovalGoneError,
} from "@/features/chat/api/chat-api";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import {
  COMPOSER_INPUT_SELECTOR,
  isSurfaceBackgrounded,
  useShortcut,
} from "@/features/settings";
import type {
  ToolCallMessagePartComponent,
  ToolCallMessagePartStatus,
} from "@assistant-ui/react";
import { useCallback, useEffect, useState } from "react";
import { useChatActive, useChatNavigationStore } from "@/features/chat";

/** Shown only when the adapter registered a backend-gated pending call for this card. */
export function ToolConfirmationControls({
  toolCallId,
  toolName,
  result,
  status,
}: {
  toolCallId?: string;
  toolName: string;
  result: unknown;
  status?: ToolCallMessagePartStatus;
}) {
  const confirmation = useChatRuntimeStore((s) =>
    toolCallId &&
    Object.prototype.hasOwnProperty.call(s.toolConfirmations, toolCallId)
      ? s.toolConfirmations[toolCallId]
      : undefined,
  );
  const allowToolAlways = useChatRuntimeStore((s) => s.allowToolAlways);
  const clearToolConfirmation = useChatRuntimeStore(
    (s) => s.clearToolConfirmation,
  );
  const autoAllowKey = confirmation?.autoAllowKey ?? "";
  // Sharing the user's image is asked every time: no Always allow, no keyboard chord.
  const disclosure = confirmation?.imageDisclosure;
  const autoAllowed = useChatRuntimeStore(
    (s) =>
      !disclosure &&
      (s.alwaysAllowToolsBySession.get(autoAllowKey)?.has(toolName) ?? false),
  );

  const [decided, setDecided] = useState(false);
  const [pending, setPending] = useState<"allow" | "deny" | null>(null);
  // "retry" can succeed on another press; "gone" means the backend no longer holds the approval.
  const [failure, setFailure] = useState<"retry" | "gone" | null>(null);
  const failed = failure !== null;

  const awaiting =
    confirmation !== undefined &&
    result === undefined &&
    status?.type === "running";
  const showControls = awaiting && !decided;

  const resolve = useCallback(
    async (decision: "allow" | "deny", alsoAlways = false) => {
      if (!toolCallId || !confirmation) return;
      setPending(decision);
      setFailure(null);
      try {
        const ok = await resolveToolConfirmation(
          confirmation.sessionId,
          confirmation.approvalId,
          decision,
        );
        if (ok) {
          // Record the session grant only after the backend accepted, or a failed press auto-approves.
          if (alsoAlways && autoAllowKey) allowToolAlways(autoAllowKey, toolName);
          // Hide only once the backend confirms, or the generation stays blocked with no retry.
          setDecided(true);
          clearToolConfirmation(toolCallId);
        } else {
          setFailure("retry");
        }
      } catch (err) {
        setFailure(err instanceof ToolApprovalGoneError ? "gone" : "retry");
      } finally {
        setPending(null);
      }
    },
    [
      toolCallId,
      confirmation,
      clearToolConfirmation,
      allowToolAlways,
      autoAllowKey,
      toolName,
    ],
  );

  useEffect(() => {
    if (showControls && autoAllowed && pending === null && !failed) {
      void resolve("allow");
    }
  }, [showControls, autoAllowed, pending, failed, resolve]);

  // Chords only while this card asks and the chat is visible; with a second request parked, both
  // cards fall back to buttons.
  const chatActive = useChatActive();
  const soleRequest = useChatRuntimeStore(
    (s) => Object.keys(s.toolConfirmations).length === 1,
  );
  // A sidebar selection also handles Escape without consuming it; defer so dismissing it cannot deny.
  const selectionActive = useChatNavigationStore((s) => s.selectionActive);
  const keyboardReady =
    !disclosure &&
    chatActive &&
    soleRequest &&
    !selectionActive &&
    showControls &&
    pending === null &&
    !(autoAllowed && !failed);
  // The Chat route stays mounted under a dialog; never answer a tool call the user cannot see.
  const chatCovered = () => isSurfaceBackgrounded(COMPOSER_INPUT_SELECTOR);
  useShortcut(
    "approveToolRequest",
    () => {
      if (chatCovered()) return;
      void resolve("allow");
    },
    {
      enabled: keyboardReady,
      // Enter belongs to the composer while it has focus.
      skipInTextFields: true,
    },
  );
  useShortcut(
    "declineToolRequest",
    () => {
      if (chatCovered()) return;
      void resolve("deny");
    },
    {
      enabled: keyboardReady,
      skipInTextFields: true,
      // Escape is allowed from the composer (it types nothing there); Enter is not, since it sends.
      textFieldException: COMPOSER_INPUT_SELECTOR,
    },
  );

  if (!showControls) return null;
  // Auto-approved tools resolve silently unless the post fails.
  if (autoAllowed && !failed) return null;

  return (
    <div className="flex flex-wrap items-center gap-2 pt-1">
      {disclosure ? (
        <p className="w-full text-xs text-muted-foreground">
          Send your attached image ({Math.ceil(disclosure.size_bytes / 1024)}{" "}
          KB) to {disclosure.server} ({disclosure.tool}) at{" "}
          {disclosure.destination}? The server may keep it.
        </p>
      ) : null}
      <Button
        size="xs"
        disabled={pending !== null || failure === "gone"}
        onClick={() => void resolve("allow")}
      >
        {disclosure ? "Share image once" : "Allow"}
      </Button>
      {disclosure ? null : (
        <Button
          size="xs"
          variant="outline"
          disabled={pending !== null || failure === "gone"}
          onClick={() => void resolve("allow", true)}
        >
          Always allow
        </Button>
      )}
      <Button
        size="xs"
        variant="destructive"
        disabled={pending !== null || failure === "gone"}
        onClick={() => void resolve("deny")}
      >
        Deny
      </Button>
      {failure !== null ? (
        <span className="text-xs text-destructive">
          {failure === "gone"
            ? "This request is no longer waiting for an answer."
            : "Could not send your decision. Try again."}
        </span>
      ) : null}
    </div>
  );
}

export function withToolConfirmation(
  Component: ToolCallMessagePartComponent,
): ToolCallMessagePartComponent {
  const WithToolConfirmation: ToolCallMessagePartComponent = (props) => (
    <>
      <Component {...props} />
      <ToolConfirmationControls
        toolCallId={props.toolCallId}
        toolName={props.toolName}
        result={props.result}
        status={props.status}
      />
    </>
  );
  return WithToolConfirmation;
}
