// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Kept apart from actions.ts so pages outside the Library can start a chat without loading it.
import type { useNavigate } from "@tanstack/react-router";
import { clearNewChatDraft, useChatRuntimeStore } from "@/features/chat";
import { toast } from "@/lib/toast";
import {
  type LibraryChatHandoff,
  useLibraryChatHandoffStore,
} from "./chat-handoff-store";

type Navigate = ReturnType<typeof useNavigate>;

// crypto.randomUUID only exists in secure contexts, and Studio is also served over plain http to
// the LAN.
function createNonce(): string {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export function resetToNewChat(): void {
  clearNewChatDraft();
  const runtime = useChatRuntimeStore.getState();
  runtime.setActiveThreadId(null);
  runtime.setActiveProjectId(null);
  runtime.setIncognito(false);
}

/** Open a fresh chat with these files attached in the composer. */
export function startLibraryChat(
  navigate: Navigate,
  handoff: LibraryChatHandoff,
): void {
  const nonce = createNonce();
  resetToNewChat();
  useLibraryChatHandoffStore.getState().offer(`single:${nonce}`, handoff);
  void navigate({ to: "/chat", search: { new: nonce } });
}

/** Open a fresh chat with a generated image or clip attached, named after its prompt. */
export async function chatAboutMedia(
  navigate: Navigate,
  src: string,
  prompt: string,
  extension: string,
): Promise<void> {
  try {
    const blob = await (await fetch(src)).blob();
    const base = prompt.replace(/[\\/:*?"<>|\s]+/g, " ").trim().slice(0, 60) || "Untitled";
    startLibraryChat(navigate, { files: [new File([blob], `${base}.${extension}`, { type: blob.type })] });
  } catch (error) {
    toast.error("Could not open the file", {
      description: error instanceof Error ? error.message : String(error),
    });
  }
}
