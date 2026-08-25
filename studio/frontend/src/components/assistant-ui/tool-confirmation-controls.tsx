// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Button } from "@/components/ui/button";
import { resolveToolConfirmation } from "@/features/chat/api/chat-api";
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

/**
 * Allow / Always allow / Deny controls for a tool call paused awaiting the
 * user's confirmation. Rendered alongside every tool card (built-in and
 * MCP) so the gate works for all tools, not just the ones using the
 * fallback renderer.
 *
 * A card is "awaiting" only when the adapter registered a backend-gated
 * pending call for it (see `toolConfirmations` in the runtime store), so
 * non-gated cards -- toggle off, or external-provider tools that already
 * ran -- never show controls.
 */
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
  const autoAllowed = useChatRuntimeStore(
    (s) =>
      s.alwaysAllowToolsBySession.get(autoAllowKey)?.has(toolName) ?? false,
  );

  const [decided, setDecided] = useState(false);
  const [pending, setPending] = useState<"allow" | "deny" | null>(null);
  const [failed, setFailed] = useState(false);

  // Still awaiting our decision: a gated pending entry exists, the tool has
  // not produced a result, and the card is in its running state.
  const awaiting =
    confirmation !== undefined &&
    result === undefined &&
    status?.type === "running";
  const showControls = awaiting && !decided;

  const resolve = useCallback(
    async (decision: "allow" | "deny") => {
      if (!toolCallId || !confirmation) return;
      setPending(decision);
      setFailed(false);
      try {
        const ok = await resolveToolConfirmation(
          confirmation.sessionId,
          confirmation.approvalId,
          decision,
        );
        if (ok) {
          // Only hide the controls once the backend confirms it matched the
          // pending call -- otherwise the generation would stay blocked with
          // no way to retry.
          setDecided(true);
          clearToolConfirmation(toolCallId);
        } else {
          setFailed(true);
        }
      } catch {
        setFailed(true);
      } finally {
        setPending(null);
      }
    },
    [toolCallId, confirmation, clearToolConfirmation],
  );

  // Tools the user marked "Always allow" (this session) approve themselves.
  useEffect(() => {
    if (showControls && autoAllowed && pending === null && !failed) {
      void resolve("allow");
    }
  }, [showControls, autoAllowed, pending, failed, resolve]);

  // ⏎ / Esc, only while this card is asking: an auto-approved one answers
  // itself. Off route the chat pane is hidden rather than unmounted, so
  // without the active check a bare key would decide a request nobody can
  // see. With a second request parked, in the other Compare pane or further
  // up the thread, the chord has no way to say which one it means, so both
  // cards fall back to their buttons.
  const chatActive = useChatActive();
  const soleRequest = useChatRuntimeStore(
    (s) => Object.keys(s.toolConfirmations).length === 1,
  );
  // A sidebar selection answers Escape already, and that listener does not
  // consume the key, so both would run off one press and deny a call the user
  // was only dismissing a selection with. Escape there costs nothing to undo
  // and this does not, so this is the one that waits. Enter goes with it: the
  // buttons are still there, and a card that takes half its keys is worse to
  // explain than one that takes none.
  const selectionActive = useChatNavigationStore((s) => s.selectionActive);
  const keyboardReady =
    chatActive &&
    soleRequest &&
    !selectionActive &&
    showControls &&
    pending === null &&
    !(autoAllowed && !failed);
  // The Chat route stays mounted under a dialog, so `keyboardReady` still says
  // yes while the request is hidden behind one. Answering a tool call the user
  // cannot see must not be reachable by accident, so both chords ask here.
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
      // A request usually arrives with the composer still focused from the
      // prompt that caused it, and Escape types nothing there, so the gate
      // would hold the decline back at the one moment it is most wanted. Enter
      // above gets no such pass: that key sends. Every other field keeps its
      // Escape, the queued-prompt editor and the settings search included.
      textFieldException: COMPOSER_INPUT_SELECTOR,
    },
  );

  if (!showControls) return null;
  // Auto-approved tools resolve silently unless the post fails.
  if (autoAllowed && !failed) return null;

  return (
    <div className="flex flex-wrap items-center gap-2 pt-1">
      <Button
        size="xs"
        disabled={pending !== null}
        onClick={() => void resolve("allow")}
      >
        Allow
      </Button>
      <Button
        size="xs"
        variant="outline"
        disabled={pending !== null}
        onClick={() => {
          if (autoAllowKey) allowToolAlways(autoAllowKey, toolName);
          void resolve("allow");
        }}
      >
        Always allow
      </Button>
      <Button
        size="xs"
        variant="destructive"
        disabled={pending !== null}
        onClick={() => void resolve("deny")}
      >
        Deny
      </Button>
      {failed ? (
        <span className="text-xs text-destructive">
          Could not send your decision. Try again.
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
