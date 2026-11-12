import type { ExportedMessageRepository } from "@assistant-ui/react";
import { saveChatMessage } from "../api/chat-api";
import type { MessageRecord } from "../types";
import { exportedItemToRecord } from "./delete-thread-message";
import { RESEARCH_METADATA_KEYS } from "./research-message-sync";
import { withMessageRevision } from "./saved-context-usage";

// Mirrors studio_db._SERVER_MANAGED_LINK_KEYS.
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

function withoutGeminiContinuationReplay<T>(metadata: T): T {
  if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) {
    return metadata;
  }
  const record = metadata as Record<string, unknown>;
  const custom = record.custom;
  if (!custom || typeof custom !== "object" || Array.isArray(custom)) {
    return metadata;
  }
  const customRecord = custom as Record<string, unknown>;
  if (!("geminiContinuationReplay" in customRecord)) {
    return metadata;
  }
  const { geminiContinuationReplay: _stale, ...keptCustom } = customRecord;
  return { ...record, custom: keptCustom } as T;
}

type ThreadImportExport = {
  export: () => ExportedMessageRepository;
  import: (data: ExportedMessageRepository) => void;
};

type ContentPart = { type: "text" | "reasoning" | "tool"; text: string; slot?: number };

// Raw strings are prose without a marker; they must count as editable or slots misalign.
// A typed part whose text is not a string cannot be edited: it joins the placeholders.
function isEditablePart(part: any): boolean {
  if (typeof part === 'string') return true;
  if (part?.type !== 'text' && part?.type !== 'reasoning') return false;
  const raw = part.text || part.content;
  return raw == null || typeof raw === 'string';
}

function toolLabel(part: any): string {
  const name = typeof part?.toolName === 'string' && part.toolName ? part.toolName : part?.type;
  const label = typeof name === 'string' ? name.replace(/[<>\r\n]+/g, ' ').trim() : "";
  return label || "tool";
}

// Escape marker-like prose so the parser does not read it as a placeholder.
function escapeMarkers(text: string): string {
  return text
    .replace(/<(\/?)(\\*)THINK>/g, "<$1\\$2THINK>")
    .replace(/<(\\*)TOOL (\d+: )/g, "<\\$1TOOL $2");
}

function unescapeMarkers(text: string): string {
  return text
    .replace(/<(\/?)\\(\\*)THINK>/g, "<$1$2THINK>")
    .replace(/<\\(\\*)TOOL (\d+: )/g, "<$1TOOL $2");
}

/**
 * Extracts the editable text and reasoning from a message, with a numbered placeholder
 * marker recording where each non-editable part sat among the prose.
 */
export function extractTaggedText(content: any): string {
  if (typeof content === 'string') return escapeMarkers(content);
  if (!Array.isArray(content)) return "";

  const open = "\u003C";
  const close = "\u003E";
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

      // Trim first so newlines do not accumulate around the tags on every save.
      if (part.type === 'reasoning') {
        return `${open}THINK${close}\n${escapeMarkers(text.trim())}\n${open}/THINK${close}`;
      }
      return escapeMarkers(text);
    })
    .filter(Boolean)
    .join('\n\n');
}

// Remove only the separator; trimming would eat meaningful leading whitespace.
function stripSeparators(text: string, afterTag: boolean, beforeTag: boolean): string {
  let out = text;
  if (afterTag) out = out.replace(/^\n\n?/, "");
  if (beforeTag) out = out.replace(/\n\n?$/, "");
  return out;
}

function parseTaggedTextToContent(text: string): ContentPart[] {
  const parts: ContentPart[] = [];
  // Requiring the slot number keeps prose and half-deleted markers from matching.
  const tagRegex = /<\/?THINK>|<TOOL (\d+): ([^<>\r\n]*)>/g;
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
  isIncognito: boolean;
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

    // Append to the previous text part so a save never multiplies parts.
    // Only parts this save created merge: a restored card keeps its own payload.
    const ownTextParts = new Set<object>();
    const pushText = (text: string) => {
      const last = finalContent[finalContent.length - 1];
      if (last && ownTextParts.has(last)) {
        last.text = `${last.text}\n\n${text}`;
        return;
      }
      const part = { type: 'text', text };
      ownTextParts.add(part);
      finalContent.push(part);
    };

    if (Array.isArray(originalContent)) {
      const nonEditableParts = originalContent.filter(
        (part: any) => part && !isEditablePart(part)
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

    const previousContent = m.message.content;
    const previousTools = Array.isArray(previousContent)
      ? previousContent.filter((part) => part && !isEditablePart(part))
      : [];
    const previousText = extractTaggedText(previousContent);
    const unchanged = extractTaggedText(finalContent) === previousText
      && JSON.stringify(finalContent.filter((part) => part && !isEditablePart(part))) === JSON.stringify(previousTools);
    // An untouched save writes back the stored parts so invisible fields survive. Not `unchanged`:
    // tagged text trims reasoning, so that would drop a whitespace-only reasoning edit.
    const untouched = newText === previousText && Array.isArray(previousContent);

    return {
      ...m,
      message: {
        ...m.message,
        content: untouched ? previousContent : finalContent,
        metadata:
          unchanged
            ? withoutGeminiContinuationReplay(m.message.metadata)
            : withMessageRevision(withoutGeminiContinuationReplay(m.message.metadata)),
      },
    };
  }) as typeof currentExport.messages;

  const originalExport = currentExport;
  thread.import({ ...currentExport, messages: updatedMessages });

  const editedMessage = updatedMessages.find(m => m.message.id === messageId)?.message;

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
