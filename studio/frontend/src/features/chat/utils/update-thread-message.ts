import type { ExportedMessageRepository, ThreadMessage } from "@assistant-ui/react";
import { saveChatMessage } from "../api/chat-api";
import type { MessageRecord } from "../types";
import { exportedItemToRecord } from "./delete-thread-message";
import { RESEARCH_METADATA_KEYS } from "./research-message-sync";

// Mirrors studio_db._SERVER_MANAGED_LINK_KEYS. The backend detaches a finished run only when
// the edit drops its ownership claim; display fields such as timing and incomplete stay.
const SERVER_OWNED_METADATA_KEYS: readonly string[] = [
  ...RESEARCH_METADATA_KEYS,
  "generationRunId",
  "generationSeq",
  "generationStatus",
  "generationSettled",
];

function withoutServerOwnership(record: MessageRecord): MessageRecord {
  const metadata = record.metadata as Record<string, unknown> | undefined;
  if (!metadata) return record;
  const kept = Object.fromEntries(
    Object.entries(metadata).filter(
      ([key]) => !SERVER_OWNED_METADATA_KEYS.includes(key),
    ),
  );
  const { metadata: _owned, ...rest } = record;
  return Object.keys(kept).length > 0 ? { ...rest, metadata: kept } : rest;
}

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
          // Trim the text first so we don't accumulate newlines
          // around the tags on every save.
          return `${open}THINK${close}\n${text.trim()}\n${open}/THINK${close}`;
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
      // Trim the extracted content to remove any leading/trailing
      // newlines created by the tag wrapping process.
      const content = text.substring(lastIndex, index).trim();
      if (content) parts.push({ type: currentType, text: content });
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

  const updatedMessages = currentExport.messages.map((m) => {
    if (m.message.id !== messageId) return m;

    const originalContent = m.message.content;
    let finalContent: any[] = [];

    if (Array.isArray(originalContent)) {
      const firstEditableIndex = originalContent.findIndex((part: any) =>
        part.type === 'text' || part.type === 'reasoning'
      );

      if (firstEditableIndex === -1) {
        const nonEditableParts = originalContent.filter((part: any) =>
          part.type !== 'text' && part.type !== 'reasoning'
        );
        finalContent = [...parsedEditableContent, ...nonEditableParts];
      } else {
        const before = originalContent.slice(0, firstEditableIndex);
        const after = originalContent.slice(firstEditableIndex + 1).filter((part: any) =>
          part.type !== 'text' && part.type !== 'reasoning'
        );
        finalContent = [...before, ...parsedEditableContent, ...after];
      }
    } else {
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

  const originalExport = currentExport;
  thread.import({ ...currentExport, messages: updatedMessages });

  const editedMessage = updatedMessages.find(m => m.message.id === messageId)?.message;

  // If it's NOT incognito, we attempt to save to the DB regardless of the ID.
  if (remoteId && !isIncognito && editedMessage) {
    try {
      await saveChatMessage(
        withoutServerOwnership(
          exportedItemToRecord(remoteId, originalParentId, editedMessage),
        ),
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
