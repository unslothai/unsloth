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

type ContentPart = { type: "text" | "reasoning" | "tool"; text: string; slot?: number };

// A raw string is prose that extractTaggedText emits as-is, without a marker. It has to
// count as editable here too, or the restoration list gains a slot the editor never
// numbered and every marker after it restores the wrong part.
function isEditablePart(part: any): boolean {
  return typeof part === 'string' || part?.type === 'text' || part?.type === 'reasoning';
}

function toolLabel(part: any): string {
  const name = typeof part?.toolName === 'string' && part.toolName ? part.toolName : part?.type;
  // The marker is one line delimited by angle brackets, so a label carrying
  // either would not survive the round trip.
  const label = typeof name === 'string' ? name.replace(/[<>\n]+/g, ' ').trim() : "";
  return label || "tool";
}

// Prose can itself spell a marker -- a reply explaining this very syntax. Escaping it
// on the way into the editor, and undoing that on the way out, keeps the parser from
// reading the reply's own words as the card's placeholder. The backslash count carries
// through, so text that already contains an escaped marker round-trips too.
function escapeMarkers(text: string): string {
  return text.replace(/<(\\*)TOOL (\d+: )/g, "<\\$1TOOL $2");
}

function unescapeMarkers(text: string): string {
  return text.replace(/<\\(\\*)TOOL (\d+: )/g, "<$1TOOL $2");
}

/**
 * Extracts the editable text and reasoning from a message, with a numbered placeholder
 * marker recording where each non-editable part sat among the prose.
 */
export function extractTaggedText(content: any): string {
  if (typeof content === 'string') return escapeMarkers(content);
  if (!Array.isArray(content)) return "";

  const open = "\u003C"; // <
  const close = "\u003E"; // >
  let slot = 0;

  return content
    .map((part: any) => {
      if (typeof part === 'string') return escapeMarkers(part);
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
        return `${open}THINK${close}\n${escapeMarkers(text.trim())}\n${open}/THINK${close}`;
      }
      return escapeMarkers(text);
    })
    .filter(Boolean)
    .join('\n\n');
}

// extractTaggedText joins parts with a blank line and puts a newline inside the THINK
// tags. Only that separator may be removed: trimming instead would eat a reply's own
// leading whitespace, and four spaces are an indented code block, not padding.
function stripSeparators(text: string, afterTag: boolean, beforeTag: boolean): string {
  let out = text;
  if (afterTag) out = out.replace(/^\n\n?/, "");
  if (beforeTag) out = out.replace(/\n\n?$/, "");
  return out;
}

function parseTaggedTextToContent(text: string): ContentPart[] {
  const parts: ContentPart[] = [];
  // A tool marker is one whole token carrying its slot number. Requiring the number
  // keeps a reply that merely writes <TOOL>name</TOOL> in its prose, and a marker the
  // user half-deleted, from being read as a marker.
  const tagRegex = /<\/?THINK>|<TOOL (\d+): ([^<>\n]*)>/g;
  let lastIndex = 0;
  let match;
  let sawTag = false;
  let currentType: ContentPart["type"] = "text";

  while ((match = tagRegex.exec(text)) !== null) {
    const fullTag = match[0];
    const index = match.index;

    if (index > lastIndex) {
      const content = stripSeparators(
        text.substring(lastIndex, index), sawTag, true,
      );
      if (content) parts.push({ type: currentType, text: unescapeMarkers(content) });
    }
    sawTag = true;
    lastIndex = index + fullTag.length;

    if (match[1] !== undefined) {
      parts.push({ type: "tool", text: fullTag, slot: Number(match[1]) });
      continue;
    }
    currentType = fullTag.startsWith("</") ? "text" : "reasoning";
  }

  if (lastIndex < text.length) {
    const remainingText = stripSeparators(text.substring(lastIndex), sawTag, false);
    if (remainingText) parts.push({ type: currentType, text: unescapeMarkers(remainingText) });
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
