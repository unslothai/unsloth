import {
  createPlatformChat,
  createPlatformSession,
  deleteAllPlatformChats,
  deletePlatformChat,
  deletePlatformSessionMessage,
  deletePlatformSessions,
  getPlatformChat,
  getPlatformSession,
  isPlatformApiError,
  listAllPlatformChats,
  listAllPlatformSessions,
  updatePlatformChat,
  updatePlatformSession,
  type PlatformChatDto,
  type PlatformSessionDto,
  type PlatformSessionMessageDto,
} from "@/integrations/platform-backend";
import type { MessageRecord, ProjectRecord, ThreadRecord } from "../types";
import {
  clearPlatformChatOverlay,
  deletePlatformProjectOverlay,
  deletePlatformThreadOverlays,
  getPlatformProjectOverlay,
  getPlatformThreadOverlay,
  setPlatformProjectOverlay,
  setPlatformThreadOverlay,
} from "./platform-chat-overlay";

export const GENERAL_CHAT_NAME = "General";
export const PLATFORM_CHAT_FANOUT_CONCURRENCY = 4;

export interface PlatformChatFanoutMetrics {
  chatCount: number;
  sessionRequests: number;
  peakConcurrency: number;
  durationMs: number;
}

let latestFanoutMetrics: PlatformChatFanoutMetrics = {
  chatCount: 0,
  sessionRequests: 0,
  peakConcurrency: 0,
  durationMs: 0,
};
const sessionChatCache = new Map<string, string>();
const generalChatIds = new Set<string>();

export function getPlatformChatFanoutMetrics(): PlatformChatFanoutMetrics {
  return { ...latestFanoutMetrics };
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function timestamp(value: unknown, date: unknown, fallback = Date.now()): number {
  const numeric = typeof value === "number" ? value : Number(value);
  if (Number.isFinite(numeric) && numeric > 0) return numeric;
  if (typeof date === "string") {
    const parsed = Date.parse(date);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

function promptConfig(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? { ...(value as Record<string, unknown>) }
    : {};
}

export function mapPlatformChatToProject(dto: PlatformChatDto): ProjectRecord {
  const id = stringValue(dto.id);
  const name = stringValue(dto.name);
  if (!id || !name) {
    throw new TypeError("Rag Platform Chat yanıtında id veya ad eksik.");
  }
  if (name === GENERAL_CHAT_NAME) generalChatIds.add(id);
  const createdAt = timestamp(dto.create_time, dto.create_date);
  const overlay = getPlatformProjectOverlay(id);
  return {
    id,
    name,
    instructions: stringValue(promptConfig(dto.prompt_config).system),
    archived: overlay.archived ?? false,
    rootPath: overlay.rootPath ?? null,
    sandboxPath: overlay.sandboxPath ?? null,
    datasetIds: stringArray(dto.dataset_ids),
    platformLlmId: stringValue(dto.llm_id) || null,
    createdAt,
    updatedAt: timestamp(dto.update_time, dto.update_date, createdAt),
  };
}

export function mapPlatformSessionToThread(
  dto: PlatformSessionDto,
  fallbackChatId?: string,
): ThreadRecord {
  const id = stringValue(dto.id);
  const chatId = stringValue(dto.chat_id) || fallbackChatId || "";
  if (!id || !chatId) {
    throw new TypeError("Rag Platform Session yanıtında id veya chat_id eksik.");
  }
  sessionChatCache.set(id, chatId);
  const overlay = getPlatformThreadOverlay(id);
  const createdAt = timestamp(dto.create_time, dto.create_date);
  return {
    id,
    title: stringValue(dto.name) || "New Chat",
    modelType: overlay.modelType ?? "base",
    modelId: overlay.modelId,
    pairId: overlay.pairId,
    projectId: generalChatIds.has(chatId) ? null : chatId,
    archived: overlay.archived ?? false,
    createdAt,
    updatedAt: timestamp(dto.update_time, dto.update_date, createdAt),
    openaiCodeExecContainerId: overlay.openaiCodeExecContainerId,
    anthropicCodeExecContainerId: overlay.anthropicCodeExecContainerId,
    forkedFromThreadId: overlay.forkedFromThreadId,
    forkedFromMessageId: overlay.forkedFromMessageId,
  };
}

function normalizedContent(value: unknown): MessageRecord["content"] {
  if (Array.isArray(value)) return value as MessageRecord["content"];
  if (typeof value === "string") return [{ type: "text", text: value }];
  if (value == null) return [];
  return [{ type: "text", text: JSON.stringify(value) }];
}

export function mapPlatformSessionMessages(
  session: PlatformSessionDto,
): MessageRecord[] {
  const threadId = stringValue(session.id);
  if (!threadId) return [];
  const rawMessages = Array.isArray(session.messages)
    ? (session.messages as PlatformSessionMessageDto[])
    : [];
  const references = Array.isArray(session.reference) ? session.reference : [];
  const baseTime = timestamp(session.create_time, session.create_date);
  let parentId: string | null = null;
  let assistantIndex = 0;

  return rawMessages.flatMap((raw, index) => {
    const role = stringValue(raw.role);
    if (role !== "system" && role !== "user" && role !== "assistant") return [];
    const platformMessageId = stringValue(raw.id) || null;
    const id = platformMessageId
      ? `${platformMessageId}:${role}:${index}`
      : `${threadId}:${role}:${index}`;
    const reference = role === "assistant" ? references[assistantIndex++] : undefined;
    const message: MessageRecord = {
      id,
      threadId,
      parentId,
      role,
      content: normalizedContent(raw.content),
      createdAt: timestamp(
        raw.create_time ?? raw.created_at,
        undefined,
        baseTime + index,
      ),
      metadata: {
        platformMessageId,
        ...(reference === undefined ? {} : { platformReference: reference }),
      },
    };
    parentId = id;
    return [message];
  });
}

async function mapBounded<T, R>(
  values: T[],
  limit: number,
  task: (value: T) => Promise<R>,
): Promise<{ values: R[]; peak: number }> {
  const output = new Array<R>(values.length);
  let cursor = 0;
  let active = 0;
  let peak = 0;
  async function worker() {
    for (;;) {
      const index = cursor++;
      if (index >= values.length) return;
      active += 1;
      peak = Math.max(peak, active);
      try {
        output[index] = await task(values[index]);
      } finally {
        active -= 1;
      }
    }
  }
  await Promise.all(
    Array.from({ length: Math.min(limit, values.length) }, () => worker()),
  );
  return { values: output, peak };
}

export async function listPlatformProjectsForChat(args: {
  includeArchived?: boolean;
} = {}): Promise<ProjectRecord[]> {
  return (await listAllPlatformChats())
    .filter((chat) => stringValue(chat.name) !== GENERAL_CHAT_NAME)
    .map(mapPlatformChatToProject)
    .filter((project) => args.includeArchived !== false || !project.archived);
}

export async function getPlatformProjectForChat(
  projectId: string,
): Promise<ProjectRecord | null> {
  try {
    return mapPlatformChatToProject(await getPlatformChat(projectId));
  } catch (error) {
    if (isPlatformApiError(error) && error.httpStatus === 404) return null;
    throw error;
  }
}

export async function createPlatformProjectForChat(input: {
  name: string;
  datasetIds?: string[];
  instructions?: string;
}): Promise<ProjectRecord> {
  const name = input.name.trim();
  if (!name) throw new Error("Project name is required.");
  if (name.localeCompare(GENERAL_CHAT_NAME, undefined, { sensitivity: "accent" }) === 0) {
    throw new Error(`“${GENERAL_CHAT_NAME}” is reserved for chats outside a project.`);
  }
  return mapPlatformChatToProject(
    await createPlatformChat({
      name,
      dataset_ids: [...new Set(input.datasetIds ?? [])],
      ...(input.instructions?.trim()
        ? { prompt_config: { system: input.instructions.trim() } }
        : {}),
    }),
  );
}

export async function updatePlatformProjectForChat(
  projectId: string,
  patch: Partial<ProjectRecord>,
): Promise<ProjectRecord> {
  if (
    patch.name !== undefined &&
    patch.name.trim().localeCompare(GENERAL_CHAT_NAME, undefined, {
      sensitivity: "accent",
    }) === 0
  ) {
    throw new Error(
      `“${GENERAL_CHAT_NAME}” is reserved for chats outside a project.`,
    );
  }
  if (
    patch.archived !== undefined ||
    patch.rootPath !== undefined ||
    patch.sandboxPath !== undefined
  ) {
    setPlatformProjectOverlay(projectId, {
      ...(patch.archived !== undefined ? { archived: patch.archived } : {}),
      ...(patch.rootPath !== undefined ? { rootPath: patch.rootPath } : {}),
      ...(patch.sandboxPath !== undefined
        ? { sandboxPath: patch.sandboxPath }
        : {}),
    });
  }
  const hasServerPatch =
    patch.name !== undefined ||
    patch.datasetIds !== undefined ||
    patch.instructions !== undefined ||
    patch.platformLlmId !== undefined;
  if (!hasServerPatch) {
    const current = await getPlatformProjectForChat(projectId);
    if (!current) throw new Error(`Project ${projectId} was not found.`);
    return current;
  }
  const raw = await getPlatformChat(projectId);
  const currentPrompt = promptConfig(raw.prompt_config);
  const updated = await updatePlatformChat(projectId, {
    ...(patch.name !== undefined ? { name: patch.name.trim() } : {}),
    ...(patch.datasetIds !== undefined
      ? { dataset_ids: [...new Set(patch.datasetIds)] }
      : {}),
    ...(patch.platformLlmId !== undefined
      ? { llm_id: patch.platformLlmId ?? "" }
      : {}),
    ...(patch.instructions !== undefined
      ? {
          prompt_config: {
            ...currentPrompt,
            system: patch.instructions.trim(),
          },
        }
      : {}),
  });
  return mapPlatformChatToProject(updated);
}

export async function deletePlatformProjectForChat(
  projectId: string,
): Promise<void> {
  await deletePlatformChat(projectId);
  deletePlatformProjectOverlay(projectId);
}

export async function ensureGeneralPlatformChat(): Promise<ProjectRecord> {
  const findGeneral = (chats: PlatformChatDto[]) =>
    chats.find((chat) => stringValue(chat.name) === GENERAL_CHAT_NAME);
  const existing = findGeneral(await listAllPlatformChats());
  if (existing) return mapPlatformChatToProject(existing);
  try {
    return mapPlatformChatToProject(
      await createPlatformChat({ name: GENERAL_CHAT_NAME, dataset_ids: [] }),
    );
  } catch (error) {
    // A second tab may have won the unique-name race. Re-read before failing.
    const raced = findGeneral(await listAllPlatformChats());
    if (raced) return mapPlatformChatToProject(raced);
    throw error;
  }
}

async function resolveChatIdForSession(sessionId: string): Promise<string | null> {
  const cached = sessionChatCache.get(sessionId);
  if (cached) return cached;
  const chats = await listAllPlatformChats();
  for (const chat of chats) mapPlatformChatToProject(chat);
  const { values } = await mapBounded(
    chats,
    PLATFORM_CHAT_FANOUT_CONCURRENCY,
    async (chat) => {
      const chatId = stringValue(chat.id);
      if (!chatId) return null;
      const sessions = await listAllPlatformSessions(chatId);
      for (const session of sessions) {
        const id = stringValue(session.id);
        if (id) sessionChatCache.set(id, chatId);
      }
      return sessions.some((session) => stringValue(session.id) === sessionId)
        ? chatId
        : null;
    },
  );
  return values.find((value): value is string => Boolean(value)) ?? null;
}

export async function listPlatformThreadsForChat(args: {
  projectId?: string | null;
  includeArchived?: boolean;
  pairId?: string;
  modelType?: ThreadRecord["modelType"];
} = {}): Promise<ThreadRecord[]> {
  const started = performance.now();
  const chats =
    args.projectId === null
      ? [await ensureGeneralPlatformChat()]
      : args.projectId
        ? [(await getPlatformProjectForChat(args.projectId))].filter(
            (chat): chat is ProjectRecord => Boolean(chat),
          )
        : (await listAllPlatformChats()).map(mapPlatformChatToProject);
  const { values, peak } = await mapBounded(
    chats,
    PLATFORM_CHAT_FANOUT_CONCURRENCY,
    async (chat) =>
      (await listAllPlatformSessions(chat.id)).map((session) =>
        mapPlatformSessionToThread(session, chat.id),
      ),
  );
  latestFanoutMetrics = {
    chatCount: chats.length,
    sessionRequests: chats.length,
    peakConcurrency: peak,
    durationMs: Math.max(0, performance.now() - started),
  };
  return values
    .flat()
    .filter((thread) => args.includeArchived !== false || !thread.archived)
    .filter((thread) => !args.pairId || thread.pairId === args.pairId)
    .filter((thread) => !args.modelType || thread.modelType === args.modelType)
    .sort(
      (a, b) =>
        (b.updatedAt ?? b.createdAt) - (a.updatedAt ?? a.createdAt),
    );
}

export async function getPlatformThreadForChat(
  sessionId: string,
): Promise<ThreadRecord | null> {
  const chatId = await resolveChatIdForSession(sessionId);
  if (!chatId) return null;
  try {
    return mapPlatformSessionToThread(
      await getPlatformSession(chatId, sessionId),
      chatId,
    );
  } catch (error) {
    if (isPlatformApiError(error) && error.httpStatus === 404) return null;
    throw error;
  }
}

export async function createPlatformThreadForChat(
  thread: ThreadRecord,
): Promise<ThreadRecord> {
  const chat = thread.projectId
    ? await getPlatformProjectForChat(thread.projectId)
    : await ensureGeneralPlatformChat();
  if (!chat) throw new Error(`Project ${thread.projectId} was not found.`);
  const session = await createPlatformSession(chat.id, {
    name: thread.title.trim() || "New Chat",
  });
  const mapped = mapPlatformSessionToThread(session, chat.id);
  setPlatformThreadOverlay(mapped.id, {
    modelType: thread.modelType,
    modelId: thread.modelId,
    pairId: thread.pairId,
    archived: thread.archived,
    openaiCodeExecContainerId: thread.openaiCodeExecContainerId,
    anthropicCodeExecContainerId: thread.anthropicCodeExecContainerId,
    forkedFromThreadId: thread.forkedFromThreadId,
    forkedFromMessageId: thread.forkedFromMessageId,
  });
  return { ...mapped, ...getPlatformThreadOverlay(mapped.id) };
}

export async function updatePlatformThreadForChat(
  sessionId: string,
  patch: Partial<ThreadRecord>,
): Promise<ThreadRecord | null> {
  const unsupportedPatch = {
    ...(patch.archived !== undefined ? { archived: patch.archived } : {}),
    ...(patch.modelType !== undefined ? { modelType: patch.modelType } : {}),
    ...(patch.modelId !== undefined ? { modelId: patch.modelId } : {}),
    ...(patch.pairId !== undefined ? { pairId: patch.pairId } : {}),
    ...(patch.openaiCodeExecContainerId !== undefined
      ? { openaiCodeExecContainerId: patch.openaiCodeExecContainerId }
      : {}),
    ...(patch.anthropicCodeExecContainerId !== undefined
      ? { anthropicCodeExecContainerId: patch.anthropicCodeExecContainerId }
      : {}),
    ...(patch.forkedFromThreadId !== undefined
      ? { forkedFromThreadId: patch.forkedFromThreadId }
      : {}),
    ...(patch.forkedFromMessageId !== undefined
      ? { forkedFromMessageId: patch.forkedFromMessageId }
      : {}),
  };
  if (Object.keys(unsupportedPatch).length > 0) {
    setPlatformThreadOverlay(sessionId, unsupportedPatch);
  }
  const chatId = await resolveChatIdForSession(sessionId);
  if (!chatId) return null;
  if (patch.projectId !== undefined && patch.projectId !== chatId) {
    throw new Error(
      "Rag Platform does not support moving an existing chat between projects.",
    );
  }
  const session =
    patch.title !== undefined
      ? await updatePlatformSession(chatId, sessionId, {
          name: patch.title.trim(),
        })
      : await getPlatformSession(chatId, sessionId);
  return mapPlatformSessionToThread(session, chatId);
}

export async function deletePlatformThreadsForChat(
  sessionIds: string[],
): Promise<void> {
  const groups = new Map<string, string[]>();
  for (const sessionId of sessionIds) {
    const chatId = await resolveChatIdForSession(sessionId);
    if (!chatId) continue;
    groups.set(chatId, [...(groups.get(chatId) ?? []), sessionId]);
  }
  await Promise.all(
    [...groups].map(([chatId, ids]) => deletePlatformSessions(chatId, ids)),
  );
  deletePlatformThreadOverlays(sessionIds);
  for (const sessionId of sessionIds) sessionChatCache.delete(sessionId);
}

export async function listPlatformMessagesForChat(
  sessionId: string,
): Promise<MessageRecord[]> {
  const chatId = await resolveChatIdForSession(sessionId);
  if (!chatId) return [];
  return mapPlatformSessionMessages(
    await getPlatformSession(chatId, sessionId),
  );
}

export async function getPlatformMessageForChat(
  sessionId: string,
  messageId: string,
): Promise<MessageRecord | null> {
  return (
    (await listPlatformMessagesForChat(sessionId)).find(
      (message) => message.id === messageId,
    ) ?? null
  );
}

export async function deletePlatformMessageTurnForChat(
  sessionId: string,
  messageId: string,
): Promise<MessageRecord[]> {
  const chatId = await resolveChatIdForSession(sessionId);
  if (!chatId) return [];
  const messages = await listPlatformMessagesForChat(sessionId);
  const selected = messages.find((message) => message.id === messageId);
  const platformMessageId = selected?.metadata?.platformMessageId;
  if (typeof platformMessageId !== "string" || !platformMessageId) {
    throw new Error("This introductory message cannot be deleted.");
  }
  return mapPlatformSessionMessages(
    await deletePlatformSessionMessage(chatId, sessionId, platformMessageId),
  );
}

export async function clearPlatformChatsForChat(): Promise<void> {
  await deleteAllPlatformChats();
  clearPlatformChatOverlay();
  sessionChatCache.clear();
}

export async function buildPlatformChatExportForChat() {
  const projects = await listPlatformProjectsForChat({ includeArchived: true });
  const threads = await listPlatformThreadsForChat({ includeArchived: true });
  const { values } = await mapBounded(
    threads,
    PLATFORM_CHAT_FANOUT_CONCURRENCY,
    (thread) => listPlatformMessagesForChat(thread.id),
  );
  return {
    exportedAt: new Date().toISOString(),
    version: 1 as const,
    threadCount: threads.length,
    projects,
    threads,
    messages: values.flat(),
  };
}
