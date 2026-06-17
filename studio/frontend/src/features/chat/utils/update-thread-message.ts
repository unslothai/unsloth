import type { ExportedMessageRepository, ThreadMessage } from "@assistant-ui/react";
import { saveChatMessage } from "../api/chat-api";

type ThreadImportExport = {
  export: () => ExportedMessageRepository;
  import: (data: ExportedMessageRepository) => void;
};

type ContentPart = { type: "text" | "reasoning" | "tool"; text: string };

/**
 * Converts structured message content (with reasoning/tools) into a tagged string
 * suitable for editing in a plain-text textarea.
 */
export function extractTaggedText(content: any): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return "";

  const open = "\u003C"; // <
  const close = "\u003E"; // >

  return content
    .map((part: any) => {
      const text = part.text || part.content || (typeof part === 'string' ? part : "");
      if (!text) return "";
      
      switch (part.type) {
        case 'reasoning': 
          return `${open}THINK${close}\n${text}\n${open}/THINK${close}`;
        case 'tool_call':
        case 'tool': 
          return `${open}TOOL${close}\n${text}\n${open}/TOOL${close}`;
        default: 
          return text;
      }
    })
    .filter(Boolean)
    .join('\n\n');
}

function parseTaggedTextToContent(text: string): ThreadMessage["content"] {
  const parts: ContentPart[] = [];
  const tagRegex = /(<\/?(THINK|TOOL)>)/g;
  let lastIndex = 0;
  let match;
  let currentType: ContentPart["type"] = "text";

  while ((match = tagRegex.exec(text)) !== null) {
    const fullTag = match[0];
    const tagName = match[2];
    const index = match.index;

    if (index > lastIndex) {
      const content = text.substring(lastIndex, index);
      if (content.trim()) parts.push({ type: currentType, text: content });
    }

    currentType = fullTag.startsWith("</") ? "text" : (tagName === "THINK" ? "reasoning" : "tool");
    lastIndex = index + fullTag.length;
  }

  if (lastIndex < text.length) {
    const remainingText = text.substring(lastIndex).trim();
    if (remainingText) parts.push({ type: currentType, text: remainingText });
  }

  return parts.length > 0 ? (parts as any) : text; 
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
    throw new Error(`Message with ID ${messageId} not found in thread.`);
  }

  const { parentId: originalParentId } = targetMessageEntry; 
  const { createdAt: originalCreatedAt } = targetMessageEntry.message;

  const updatedMessages = currentExport.messages.map((m) => 
    m.message.id === messageId 
      ? { ...m, message: { ...m.message, content: updatedContent } } 
      : m
  );

  thread.import({ ...currentExport, messages: updatedMessages });

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
      console.error("Backend sync failed for message update:", e);
      throw e;
    }
  }

  return updatedContent;
}