// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth/api";

export type StoredAttachmentFile = { id: string; sandboxPath: string };

type Attachment = {
  name: string;
  content?: readonly unknown[];
  storedFile?: StoredAttachmentFile;
};

/** Keeps the original bytes for the python tool. Null when that fails, leaving the inline text. */
export async function uploadAttachmentFile(
  file: File,
): Promise<StoredAttachmentFile | null> {
  const form = new FormData();
  form.append("file", file);
  try {
    const response = await authFetch("/api/chat/attachment-files", {
      method: "POST",
      body: form,
    });
    const { id, sandboxPath } = (await response.json()) as Record<
      string,
      unknown
    >;
    return typeof id === "string" && typeof sandboxPath === "string"
      ? { id, sandboxPath }
      : null;
  } catch {
    return null;
  }
}

function isStored(
  attachment: unknown,
): attachment is Attachment & { storedFile: StoredAttachmentFile } {
  const stored = (attachment as Attachment | null)?.storedFile;
  return (
    typeof stored?.id === "string" && typeof stored.sandboxPath === "string"
  );
}

// What ChatCompletionRequest.sandbox_attachments takes. Past it the request is refused outright,
// so a long thread carries its most recent files rather than failing every turn.
const MAX_SANDBOX_ATTACHMENTS = 64;

function carriedSandboxPaths(
  messages: readonly { attachments?: readonly unknown[] }[],
): Set<string> {
  const order = new Set<string>();
  for (const message of messages) {
    for (const attachment of message.attachments ?? []) {
      if (!isStored(attachment)) continue;
      const { sandboxPath } = attachment.storedFile;
      // Deleted first, so a file attached again late in the thread counts as recent, not as old.
      order.delete(sandboxPath);
      order.add(sandboxPath);
    }
  }
  return new Set([...order].slice(-MAX_SANDBOX_ATTACHMENTS));
}

/** Names every stored attachment's sandbox copy, and asks the backend to put the copies there. */
export function withSandboxAttachmentPaths<
  M extends { attachments?: readonly unknown[] },
>(messages: readonly M[]) {
  const sandboxAttachments: Array<{ id: string; name: string }> = [];
  const carried = carriedSandboxPaths(messages);
  const listed = new Set<string>();
  const annotated = messages.map((message) => {
    if (!message.attachments?.some(isStored)) return message;
    const attachments = message.attachments.map((attachment) => {
      if (!isStored(attachment)) return attachment;
      const { id, sandboxPath } = attachment.storedFile;
      // Dropped from the request, so it is not told to open a file that was never copied.
      if (!carried.has(sandboxPath)) return attachment;
      if (!listed.has(sandboxPath)) {
        listed.add(sandboxPath);
        // The path's own last segment, so the backend derives this exact path again.
        sandboxAttachments.push({ id, name: sandboxPath.split("/").pop()! });
      }
      const note = `[${attachment.name} is saved at ${sandboxPath} in the python tool's working directory]`;
      return {
        ...attachment,
        content: [{ type: "text", text: note }, ...(attachment.content ?? [])],
      };
    });
    return { ...message, attachments } as M;
  });
  return { messages: annotated, sandboxAttachments };
}
