// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  MessageRecord,
  ModelType,
  ParsedConversation,
  ProjectRecord,
  ThreadRecord,
} from "../types";
import { compareStoredMessages } from "./message-order";

type Dict = Record<string, unknown>;

const MODEL_TYPES = new Set<string>(["base", "lora", "model1", "model2"]);

// Mirrors studio_db._SERVER_MANAGED_LINK_KEYS: a restored copy belongs to no server run.
const SERVER_MANAGED_LINK_KEYS = new Set<string>([
  "researchRunId",
  "researchRun",
  "researchStatus",
  "researchPlanRevision",
  "serverManaged",
  "generationRunId",
  "generationSeq",
  "generationStatus",
  "generationSettled",
]);

// A backup is a file that arrived from somewhere, so its settings snapshot is
// untrusted input, not a saved preference. These are the only keys a restored chat
// carries. Everything else in ThreadScopedSettings either turns a capability on
// (toolsEnabled, codeToolsEnabled, mcpEnabledForChat, webFetchToolsEnabled,
// deepResearchEnabled, artifactsEnabled, the rag* group), silences the approval
// prompt (permissionMode "off" sets confirm_tool_calls false and never pauses) or
// injects text the sender chose (systemPrompt, systemVariables). Restoring those
// together lets a sent backup arm a chat that runs tools unattended under the
// importer's account on their first message.
// An ALLOWLIST, so a setting added later is dropped until someone decides it is
// safe to restore, rather than shipping restorable by default.
const RESTORABLE_SETTING_KEYS = new Set<string>([
  "temperature",
  "topP",
  "topK",
  "minP",
  "minPMode",
  "repetitionPenalty",
  "presencePenalty",
  "seed",
  "reasoningEnabled",
  "reasoningEffort",
]);

function isDict(value: unknown): value is Dict {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// The export emits content_json exactly as it was stored, so the importer has to
// accept every shape storage accepts, not just the one it writes today. A string
// becomes a text part rather than nothing: the backup is the user's last copy, and
// dropping the text of a message is not recoverable. Same normalisation
// oaiContentToParts already does for the OpenAI path in chat-import.ts.
function messageContent(raw: unknown): MessageRecord["content"] {
  if (Array.isArray(raw)) {
    return raw.map((part) =>
      isDict(part) ? withoutServerLinks(part) : part,
    ) as MessageRecord["content"];
  }
  if (typeof raw === "string" && raw.trim()) {
    return [{ type: "text", text: raw }] as MessageRecord["content"];
  }
  return [] as MessageRecord["content"];
}

// The third carrier of the same vector as systemPrompt and a project's instructions:
// toOpenAIMessages passes a stored role "system" straight into the next request, so a backup
// could hand whoever wrote it the system role on the importer's account. The text is the
// user's and a backup is their last copy, so it comes back as an ordinary turn, visible in the
// transcript, rather than being dropped or replayed with authority.
function restorableRole(role: string): MessageRecord["role"] {
  return (role === "system" ? "user" : role) as MessageRecord["role"];
}

function restorableSettings(settings: Dict): Dict {
  return Object.fromEntries(
    Object.entries(settings).filter(([key]) => RESTORABLE_SETTING_KEYS.has(key)),
  );
}

function str(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function withoutServerLinks(value: Dict): Dict {
  return Object.fromEntries(
    Object.entries(value).filter(([key]) => !SERVER_MANAGED_LINK_KEYS.has(key)),
  );
}

function detachMetadata(metadata: Dict): Dict {
  const detached = withoutServerLinks(metadata);
  if (isDict(detached.custom)) detached.custom = withoutServerLinks(detached.custom);
  return detached;
}

export function isStudioChatBackup(value: unknown): value is Dict {
  return (
    isDict(value) &&
    typeof value.version === "number" &&
    Array.isArray(value.threads) &&
    Array.isArray(value.messages)
  );
}

export function studioBackupProjects(backup: Dict): ProjectRecord[] {
  const projects: ProjectRecord[] = [];
  if (!Array.isArray(backup.projects)) return projects;
  for (const project of backup.projects) {
    if (!isDict(project)) continue;
    const id = str(project.id);
    const name = str(project.name);
    if (!id || !name) continue;
    const createdAt = num(project.createdAt) ?? Date.now();
    projects.push({
      id,
      name,
      // Dropped for the same reason as systemPrompt above: chat-adapter.ts wraps a
      // project's instructions in <project_instructions> and unshifts them as a
      // system message on the next send, so restoring them lets whoever wrote the
      // backup put text in the system prompt of the importer's chats. The name and
      // the grouping are what a restore is for; the instructions are not.
      instructions: "",
      archived: project.archived === true,
      createdAt,
      updatedAt: num(project.updatedAt) ?? createdAt,
    });
  }
  return projects;
}

export function studioBackupToConversations(
  backup: Dict,
  fallbackTitle: string,
  knownProjectIds: ReadonlySet<string> = new Set(),
): ParsedConversation[] {
  const threads = (backup.threads as unknown[]).filter(
    (thread): thread is Dict => isDict(thread) && str(thread.id) !== null,
  );
  const threadIds = new Map<string, string>();
  const pairIds = new Map<string, string>();
  for (const thread of threads) {
    threadIds.set(thread.id as string, crypto.randomUUID());
    const pairId = str(thread.pairId);
    if (pairId && !pairIds.has(pairId)) pairIds.set(pairId, crypto.randomUUID());
  }

  const messageIds = new Map<string, string>();
  const messagesByThread = new Map<string, Dict[]>();
  for (const message of backup.messages as unknown[]) {
    if (!isDict(message)) continue;
    const id = str(message.id);
    const threadId = str(message.threadId);
    if (!id || !threadId || !threadIds.has(threadId) || messageIds.has(id)) continue;
    if (!str(message.role)) continue;
    messageIds.set(id, crypto.randomUUID());
    const bucket = messagesByThread.get(threadId);
    if (bucket) bucket.push(message);
    else messagesByThread.set(threadId, [message]);
  }

  const conversations: ParsedConversation[] = [];
  threads.forEach((thread, index) => {
    // The export dumps every thread row, and a chat whose only turn was deleted has none left.
    const source = messagesByThread.get(thread.id as string) ?? [];
    const threadId = threadIds.get(thread.id as string) as string;
    const ordered = source
      .map((message) => ({
        raw: message,
        id: message.id as string,
        role: message.role as string,
        createdAt: num(message.createdAt) ?? 0,
      }))
      .sort(compareStoredMessages);

    let previousTs = Number.NEGATIVE_INFINITY;
    const messages = ordered.map(({ raw, id, createdAt }): MessageRecord => {
      const ts = Math.max(previousTs + 1, createdAt);
      previousTs = ts;
      const record: MessageRecord = {
        id: messageIds.get(id) as string,
        threadId,
        role: restorableRole(raw.role as string),
        content: messageContent(raw.content),
        createdAt: ts,
      };
      if (typeof raw.parentId === "string") {
        record.parentId = messageIds.get(raw.parentId) ?? null;
      } else if (raw.parentId === null) {
        record.parentId = null;
      }
      if (Array.isArray(raw.attachments)) {
        record.attachments = raw.attachments as MessageRecord["attachments"];
      }
      if (isDict(raw.metadata)) record.metadata = detachMetadata(raw.metadata);
      return record;
    });

    const forkedFromThreadId = threadIds.get(str(thread.forkedFromThreadId) ?? "");
    const forkedFromMessageId = messageIds.get(
      str(thread.forkedFromMessageId) ?? "",
    );
    // Points into this thread's own messages, so it remaps like any other id. Dropping it
    // would restore the fork without its "Continued from chat" divider.
    const forkBoundaryMessageId = messageIds.get(
      str(thread.forkBoundaryMessageId) ?? "",
    );
    const projectId = str(thread.projectId);
    const pairId = str(thread.pairId);
    const modelId = str(thread.modelId);
    const createdAt = num(thread.createdAt) ?? messages[0]?.createdAt ?? Date.now();
    conversations.push({
      title: str(thread.title) ?? `${fallbackTitle} ${index + 1}`,
      threadId,
      messages,
      archived: thread.archived === true,
      createdAt,
      thread: {
        modelType: MODEL_TYPES.has(thread.modelType as string)
          ? (thread.modelType as ModelType)
          : "base",
        ...(modelId ? { modelId } : {}),
        ...(str(thread.modelGgufVariant)
          ? { modelGgufVariant: thread.modelGgufVariant as string }
          : {}),
        ...(pairId ? { pairId: pairIds.get(pairId) } : {}),
        projectId: projectId && knownProjectIds.has(projectId) ? projectId : null,
        createdAt,
        updatedAt: num(thread.updatedAt) ?? createdAt,
        ...(forkedFromThreadId
          ? {
              forkedFromThreadId,
              ...(forkedFromMessageId ? { forkedFromMessageId } : {}),
            }
          : {}),
        ...(forkBoundaryMessageId ? { forkBoundaryMessageId } : {}),
        ...(isDict(thread.settings) &&
        Object.keys(restorableSettings(thread.settings)).length > 0
          ? {
              settings: restorableSettings(
                thread.settings,
              ) as ThreadRecord["settings"],
            }
          : {}),
      },
    });
  });
  return conversations;
}
