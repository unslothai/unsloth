import type { ExportedMessageRepository, ThreadMessage } from "@assistant-ui/react";
import { saveChatMessage } from "../api/chat-api";

type ThreadImportExport = {
  export: () => ExportedMessageRepository;
  import: (data: ExportedMessageRepository) => void;
};

type ContentPart = { type: "text" | "reasoning" | "tool"; text: string; slot?: number };

export function isEditablePart(part: any): boolean {
  return part?.type === 'text' || part?.type === 'reasoning';
}

function toolLabel(part: any): string {
  const name = typeof part?.toolName === 'string' && part.toolName ? part.toolName : part?.type;
  // The marker is one line delimited by angle brackets, so a label carrying
  // either would not survive the round trip.
  const label = typeof name === 'string' ? name.replace(/[<>\n]+/g, ' ').trim() : "";
  return label || "tool";
}

/**
 * Extracts the editable text and reasoning from a message, with a numbered placeholder
 * marker recording where each non-editable part sat among the prose.
 */
export function extractTaggedText(content: any): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return "";

  const open = "\u003C"; // <
  const close = "\u003E"; // >
  let slot = 0;

  return content
    .map((part: any) => {
      if (typeof part === 'string') return part;
      if (!part) return "";

      if (!isEditablePart(part)) {
        slot += 1;
        return `${open}TOOL ${slot}: ${toolLabel(part)}${close}`;
      }

      const text = part.text || part.content || "";
      if (!text) return "";

      // Trim the text first so we don't accumulate newlines
      // around the tags on every save.
      if (part.type === 'reasoning') {
        return `${open}THINK${close}\n${text.trim()}\n${open}/THINK${close}`;
      }
      return text;
    })
    .filter(Boolean)
    .join('\n\n');
}

function parseTaggedTextToContent(text: string): ContentPart[] {
  const parts: ContentPart[] = [];
  // A tool marker is one whole token carrying its slot number. Requiring the number
  // keeps a reply that merely writes <TOOL>name</TOOL> in its prose, and a marker the
  // user half-deleted, from being read as a marker.
  const tagRegex = /<\/?THINK>|<TOOL (\d+): ([^<>\n]*)>/g;
  let lastIndex = 0;
  let match;
  let currentType: ContentPart["type"] = "text";

  while ((match = tagRegex.exec(text)) !== null) {
    const fullTag = match[0];
    const index = match.index;

    if (index > lastIndex) {
      // Trim the extracted content to remove any leading/trailing
      // newlines created by the tag wrapping process.
      const content = text.substring(lastIndex, index).trim();
      if (content) parts.push({ type: currentType, text: content });
    }
    lastIndex = index + fullTag.length;

    if (match[1] !== undefined) {
      parts.push({ type: "tool", text: fullTag, slot: Number(match[1]) });
      continue;
    }
    currentType = fullTag.startsWith("</") ? "text" : "reasoning";
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
  isIncognito: boolean; // <--- ADD THIS
}) {
  const { thread, messageId, remoteId, newText, isIncognito } = args;
  const parsedEditableContent = parseTaggedTextToContent(newText);
  const currentExport = thread.export();

  const targetMessageEntry = currentExport.messages.find(m => m.message.id === messageId);
  if (!targetMessageEntry) {
    throw new Error(`Message with ID ${messageId} not found in thread.`);
  }

  const { parentId: originalParentId } = targetMessageEntry;
  const { createdAt: originalCreatedAt } = targetMessageEntry.message;

  const updatedMessages = currentExport.messages.map((m) => {
    if (m.message.id !== messageId) return m;

    const originalContent = m.message.content;
    const finalContent: any[] = [];

    // Text the editor produced, appended to the run before it rather than opening a
    // second text part, so a save never multiplies the parts of a reply.
    const pushText = (text: string) => {
      const last = finalContent[finalContent.length - 1];
      if (last && last.type === 'text') {
        last.text = `${last.text}\n\n${text}`;
        return;
      }
      finalContent.push({ type: 'text', text });
    };

    if (Array.isArray(originalContent)) {
      const nonEditableParts = originalContent.filter(
        (part: any) => !isEditablePart(part)
      );
      const restored = new Set<number>();

      for (const part of parsedEditableContent) {
        if (part.type !== 'tool') {
          if (part.type === 'text') pushText(part.text);
          else finalContent.push(part);
          continue;
        }
        const slot = (part.slot ?? 0) - 1;
        if (nonEditableParts[slot] && !restored.has(slot)) {
          restored.add(slot);
          finalContent.push(nonEditableParts[slot]);
        } else {
          // No part of this reply answers to that marker, so it is prose: keep it.
          pushText(part.text);
        }
      }

      // A card whose marker the user deleted still belongs to the reply.
      nonEditableParts.forEach((part, i) => {
        if (!restored.has(i)) finalContent.push(part);
      });
    } else {
      for (const part of parsedEditableContent) {
        if (part.type === 'text' || part.type === 'tool') pushText(part.text);
        else finalContent.push(part);
      }
    }

    return {
      ...m,
      message: {
        ...m.message,
        content: finalContent,
      },
    };
  }) as typeof currentExport.messages;

  const originalExport = currentExport;
  thread.import({ ...currentExport, messages: updatedMessages });

  // If it's NOT incognito, we attempt to save to the DB regardless of the ID.
  if (remoteId && !isIncognito) {
    try {
      await saveChatMessage(
        {
          id: messageId,
          threadId: remoteId,
          parentId: originalParentId,
          role: "assistant",
          content: (updatedMessages.find(m => m.message.id === messageId)?.message.content) || [],
          createdAt: originalCreatedAt ? Number(originalCreatedAt) : Date.now(),
        },
        { allowGenerationEdit: true },
      );
    } catch (e) {
      thread.import(originalExport);
      console.error("Backend sync failed for message update. Rolling back UI.", e);
      throw e;
    }
  }

  return (updatedMessages.find(m => m.message.id === messageId)?.message.content) || [];
}
