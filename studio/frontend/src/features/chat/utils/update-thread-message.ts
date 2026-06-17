import type { ExportedMessageRepository, ThreadMessage } from "@assistant-ui/react";
import { saveChatMessage } from "../api/chat-api";

type ThreadImportExport = {
  export: () => ExportedMessageRepository;
  import: (data: ExportedMessageRepository) => void;
};

type ContentPart = { type: "text" | "reasoning" | "tool"; text: string };

/**
 * Extracts only the editable text and reasoning from a message, 
 * ignoring structured parts like tool calls that cannot be edited as plain text.
 */
export function extractTaggedText(content: any): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return "";

  const open = "\u003C"; // <
  const close = "\u003E"; // >

  return content
    .map((part: any) => {
      if (typeof part === 'string') return part;
      if (!part) return "";
      
      // Only extract text from 'text' or 'reasoning' parts.
      // Tool calls/responses are ignored here so they aren't accidentally 
      // deleted or corrupted by the user in the textarea.
      const text = part.text || part.content || "";
      if (!text) return "";
      
      switch (part.type) {
        case 'reasoning': 
          return `${open}THINK${close}\n${text}\n${open}/THINK${close}`;
        case 'text':
        default:
          return text;
      }
    })
    .filter(Boolean)
    .join('\n\n');
}

function parseTaggedTextToContent(text: string): ContentPart[] {
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

  return parts;
}

export async function updateThreadMessage(args: {
  thread: ThreadImportExport;
  messageId: string;
  remoteId: string | undefined;
  newText: string;
}) {
  const { thread, messageId, remoteId, newText } = args;
  const parsedEditableContent = parseTaggedTextToContent(newText);
  const currentExport = thread.export();

  const targetMessageEntry = currentExport.messages.find(m => m.message.id === messageId);
  if (!targetMessageEntry) {
    throw new Error(`Message with ID ${messageId} not found in thread.`);
  }

  const { parentId: originalParentId } = targetMessageEntry; 
  const { createdAt: originalCreatedAt } = targetMessageEntry.message;

  // MERGE STRATEGY:
  // We want to preserve original tool_calls/responses but update the text/reasoning.
  const updatedMessages = currentExport.messages.map((m) => {
    if (m.message.id !== messageId) return m;

    const originalContent = m.message.content;
    let finalContent: any[] = [];

    if (Array.isArray(originalContent)) {
      // 1. Find the index of the very first editable part (text or reasoning)
      const firstEditableIndex = originalContent.findIndex((part: any) => 
        part.type === 'text' || part.type === 'reasoning'
      );

      if (firstEditableIndex === -1) {
        // If there was no text at all, just put the new text at the start 
        // and keep the tools.
        const nonEditableParts = originalContent.filter((part: any) => 
          part.type !== 'text' && part.type !== 'reasoning'
        );
        finalContent = [...parsedEditableContent, ...nonEditableParts];
      } else {
        // 2. PRESERVE ORDER: 
        // - Keep everything that came BEFORE the first piece of text (e.g., early tool calls).
        // - Inject the entire new edited block at that first slot.
        // - Keep everything that came AFTER, but filter out the old editable text.
        const before = originalContent.slice(0, firstEditableIndex);
        const after = originalContent.slice(firstEditableIndex + 1).filter((part: any) => 
          part.type !== 'text' && part.type !== 'reasoning'
        );
        
        finalContent = [...before, ...parsedEditableContent, ...after];
      }
    } else {
      // If the original was just a string, we just use the parsed array.
      finalContent = parsedEditableContent;
    }

    return {
      ...m,
      message: {
        ...m.message,
        content: finalContent,
      },
    };
  }) as typeof currentExport.messages;

  // Snapshot for rollback
  const originalExport = currentExport;
  thread.import({ ...currentExport, messages: updatedMessages });

  if (remoteId && !remoteId.startsWith("__LOCALID_")) {
    try {
      await saveChatMessage({
        id: messageId,
        threadId: remoteId,
        parentId: originalParentId,
        role: "assistant",
        content: (updatedMessages.find(m => m.message.id === messageId)?.message.content) || [],
        createdAt: originalCreatedAt ? Number(originalCreatedAt) : Date.now(),
      });
    } catch (e) {
      // ROLLBACK: If backend sync fails, revert the UI to the original state
      thread.import(originalExport);
      console.error("Backend sync failed for message update. Rolling back UI.", e);
      throw e;
    }
  }

  return (updatedMessages.find(m => m.message.id === messageId)?.message.content) || [];
}