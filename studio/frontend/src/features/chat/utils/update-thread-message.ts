import { MessageRepository } from "@assistant-ui/core/internal";
import type { ExportedMessageRepository, ThreadMessage } from "@assistant-ui/react";
import { saveChatMessage } from "../api/chat-api";

type ThreadImportExport = {
  export: () => ExportedMessageRepository;
  import: (data: ExportedMessageRepository) => void;
};

function parseTaggedTextToContent(text: string): ThreadMessage["content"] {
  const parts: any[] = [];
  const tagRegex = /(<\/?(THINK|TOOL)>)/g;
  let lastIndex = 0;
  let match;
  let currentType: "text" | "reasoning" | "tool" = "text";

  while ((match = tagRegex.exec(text)) !== null) {
    const fullTag = match[0];
    const tagName = match[2];
    const index = match.index;

    if (index > lastIndex) {
      const content = text.substring(lastIndex, index);
      const trimmedContent = content.trim();
      if (trimmedContent) {
        parts.push({ type: currentType, text: trimmedContent });
      }
    }

    if (fullTag.startsWith("</")) {
      currentType = "text";
    } else {
      currentType = tagName === "THINK" ? "reasoning" : "tool";
    }
    lastIndex = index + fullTag.length;
  }

  if (lastIndex < text.length) {
    const remainingText = text.substring(lastIndex).trim();
    if (remainingText) {
      parts.push({ type: currentType, text: remainingText });
    }
  }

  return parts.length > 0 ? parts : text;
}

export async function updateThreadMessage(args: {
  thread: ThreadImportExport;
  messageId: string;
  remoteId: string | undefined;
  newText: string;
}) {
  const { thread, messageId, remoteId, newText } = args;
  const updatedContent = parseTaggedTextToContent(newText);

  const exported = thread.export();
  const repo = new MessageRepository();
  repo.import(exported);
  const message = repo.getMessage(messageId);
  if (message) message.content = updatedContent;
  thread.import(repo.export());

  if (remoteId) {
    try {
      await saveChatMessage({
        id: messageId,
        threadId: remoteId,
        role: "assistant",
        content: updatedContent,
        createdAt: Date.now(),
      });
    } catch (e) {
      console.error("Backend sync failed:", e);
    }
  }

  return updatedContent;
}