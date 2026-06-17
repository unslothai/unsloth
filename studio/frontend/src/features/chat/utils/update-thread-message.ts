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
      if (content.trim()) parts.push({ type: currentType, text: content });
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
    if (remainingText) parts.push({ type: currentType, text: remainingText });
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

  const currentExport = thread.export();

  const targetMessageEntry = currentExport.messages.find(m => m.message.id === messageId);
  
  if (!targetMessageEntry) {
    throw new Error("MESSAGE_NOT_FOUND");
  }

  const originalParentId = targetMessageEntry.parentId; 
  const originalCreatedAt = targetMessageEntry.message.createdAt;

  const updatedMessages = currentExport.messages.map((m) => {
    if (m.message.id === messageId) {
      return {
        ...m,
        message: {
          ...m.message,
          content: updatedContent,
        },
      };
    }
    return m;
  });

  const nextExport = {
    ...currentExport,
    messages: updatedMessages,
  };

  thread.import(nextExport);

  if (remoteId) {
    try {
      await saveChatMessage({
        id: messageId,
        threadId: remoteId,
        parentId: originalParentId,
        role: "assistant",
        content: updatedContent,
        createdAt: originalCreatedAt ? Number(originalCreatedAt) : Date.now(),
      });
    } catch (e) {
      console.error("Backend sync failed:", e);
      throw e;
    }
  }

  return updatedContent;
}