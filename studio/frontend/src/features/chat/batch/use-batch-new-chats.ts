// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { useAui } from "@assistant-ui/react";
import type { useNavigate } from "@tanstack/react-router";
import { type RefObject, useEffect } from "react";
import { AUTH_SESSION_CLEARED_EVENT } from "@/features/auth";
import { openFilePicker } from "@/lib/open-file-picker";
import { toast } from "@/lib/toast";
import { wasAttachmentAddCancelled } from "../draft-attachment-adapter";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import {
  type BatchFinish,
  type ThreadListLike,
  batchTargetKey,
  batchThreadReader,
  claimBatchStep,
  resetBatchNewChats,
  startBatchNewChats,
  stopBatchNewChats,
  useBatchNewChatsStore,
} from "./batch-new-chats";

type Aui = ReturnType<typeof useAui>;
type Navigate = ReturnType<typeof useNavigate>;

const BATCH_TOAST_ID = "batch-new-chats";

// crypto.randomUUID is undefined in non-secure contexts (HTTP over a LAN IP).
function newChatNonce(): string {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function chatCount(count: number): string {
  return count === 1 ? "1 chat" : `${count} chats`;
}

const notify = {
  progress: (index: number, total: number, fileName: string) => {
    toast.loading(`Sending to new chats: ${index + 1} of ${total}`, {
      id: BATCH_TOAST_ID,
      description: fileName,
      duration: Number.POSITIVE_INFINITY,
      action: {
        label: "Stop",
        onClick: (event) => {
          // Keep the toast: the stop result replaces it instead of closing with the click.
          event.preventDefault();
          stopBatchNewChats();
        },
      },
    });
  },
  finish: (result: BatchFinish) => {
    const base = { id: BATCH_TOAST_ID, duration: 8000, action: undefined, description: undefined };
    if (result.kind === "done") {
      toast.success(`Sent the prompt to ${chatCount(result.total)}`, base);
    } else if (result.kind === "stopped") {
      toast.info(`Stopped after ${result.sent} of ${result.total}`, base);
    } else {
      toast.error(`Stopped at ${result.index + 1} of ${result.total}: ${result.fileName}`, {
        ...base,
        description: result.reason,
      });
    }
  },
};

if (typeof window !== "undefined") {
  // The next account must not inherit another account's files.
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () => {
    resetBatchNewChats();
    toast.dismiss(BATCH_TOAST_ID);
  });
}

export function pickFilesForNewChats(aui: Aui, navigate: Navigate, accept: string): void {
  const composer = aui.composer().getState();
  const prompt = composer.text.trim();
  if (!prompt) {
    toast.error("Type the prompt first", {
      description: "Each file you pick is sent with this message in its own new chat.",
    });
    return;
  }
  if (composer.attachments.length > 0) {
    toast.error("Remove the attached files first", {
      description: "Pick the files to send in the next step instead.",
    });
    return;
  }
  openFilePicker(accept, (files) => {
    const started = startBatchNewChats(prompt, files, {
      newNonce: newChatNonce,
      openNewChat: (nonce) => {
        const runtime = useChatRuntimeStore.getState();
        runtime.setActiveThreadId(null);
        runtime.setActiveProjectId(null);
        runtime.setIncognito(false);
        void navigate({ to: "/chat", search: { new: nonce } });
      },
      notify,
    });
    if (started) {
      aui.composer().setText("");
    } else {
      toast.error("A batch is already running", {
        description: "Stop it or wait for it to finish first.",
      });
    }
  });
}

// A new chat's composer can remount while a step is being sent; the newest one for a key sends.
const composerForms = new Map<string, RefObject<HTMLFormElement | null>>();

// Hidden tabs pause animation frames; the timer keeps an unattended batch moving.
function nextFrame(): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, 100);
    requestAnimationFrame(() => {
      clearTimeout(timer);
      resolve();
    });
  });
}

export function useBatchNewChatsComposer(
  aui: Aui,
  targetKey: string | null,
  formRef: RefObject<HTMLFormElement | null>,
): void {
  useEffect(() => {
    if (!targetKey) return;
    composerForms.set(targetKey, formRef);
    let disposed = false;
    let readyTicks = 0;
    // Polls only while a step waits on this chat, so idle composers keep no timer.
    let poll: ReturnType<typeof setInterval> | null = null;
    const stopPoll = () => {
      if (poll !== null) clearInterval(poll);
      poll = null;
    };
    // The route switches before the runtime does: claim only once the empty new chat is on screen.
    const onFreshChat = () => {
      const item = aui.threadListItem().getState();
      return (
        !item.remoteId &&
        aui.thread().getState().messages.length === 0 &&
        !aui.thread().getState().isRunning
      );
    };
    const tryClaim = () => {
      if (disposed) return;
      const run = useBatchNewChatsStore.getState().run;
      if (!run || run.phase !== "opening" || batchTargetKey(run.nonce) !== targetKey) {
        readyTicks = 0;
        stopPoll();
        return;
      }
      poll ??= setInterval(tryClaim, 150);
      readyTicks = onFreshChat() ? readyTicks + 1 : 0;
      if (readyTicks < 2) return;
      const claim = claimBatchStep(targetKey);
      if (!claim) return;
      stopPoll();
      const draftId = aui.threadListItem().getState().id;
      void (async () => {
        try {
          await aui.composer().addAttachment(claim.file);
          if (wasAttachmentAddCancelled(claim.file)) {
            throw new Error("The chat changed before the file was attached.");
          }
          aui.composer().setText(claim.prompt);
          // Let the composer render the file and text, so its send checks see them.
          await nextFrame();
          await nextFrame();
          // A New chat opened meanwhile can reuse this blank draft and clear it.
          const draft = aui.composer().getState();
          if (
            aui.threadListItem().getState().id !== draftId ||
            draft.text.trim() !== claim.prompt ||
            draft.attachments.length === 0
          ) {
            throw new Error("The chat changed before the message was sent.");
          }
          const form = formRef.current ?? composerForms.get(targetKey)?.current;
          if (!form) {
            throw new Error("The chat changed before the message was sent.");
          }
          const runtime = aui.threads().__internal_getAssistantRuntime?.();
          if (!runtime) throw new Error("Chat isn't ready yet.");
          // Through the form, so this send takes every check a typed send does.
          form.requestSubmit();
          const read = batchThreadReader(
            runtime.threads as unknown as ThreadListLike,
            draftId,
          );
          claim.sent({
            read,
            unsent: () => {
              // A draft left unsent counts as unsent, so the send timeout still ends the batch.
              if (aui.threadListItem().getState().id !== draftId) return true;
              return aui.composer().getState().text.trim() === claim.prompt;
            },
          });
        } catch (error) {
          claim.failed(error instanceof Error ? error.message : String(error));
        }
      })();
    };
    tryClaim();
    const unsubscribe = useBatchNewChatsStore.subscribe(tryClaim);
    return () => {
      disposed = true;
      if (composerForms.get(targetKey) === formRef) composerForms.delete(targetKey);
      stopPoll();
      unsubscribe();
    };
  }, [aui, targetKey, formRef]);
}
