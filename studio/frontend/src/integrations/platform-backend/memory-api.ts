import { platformRequest } from "./client";
import { asRecord } from "./model-types";
import type {
  CreatePlatformMemoryInput,
  PlatformMemory,
  PlatformMemoryListResult,
  PlatformMemoryMessage,
  PlatformMemoryMessageListResult,
  PlatformMemoryType,
  UpdatePlatformMemoryInput,
} from "./memory-types";
import { mapPlatformMemory, mapPlatformMemoryMessage } from "./memory-types";

const segment = (value: string) => encodeURIComponent(value.trim());
const messageSegment = (memoryId: string, messageId: string) =>
  `${segment(memoryId)}:${segment(messageId)}`;
const mappedList = <T>(value: unknown, map: (item: unknown) => T | null) =>
  (Array.isArray(value) ? value : [])
    .map(map)
    .filter((item): item is T => item !== null);
const count = (value: unknown, fallback: number) => {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : fallback;
};

export async function listPlatformMemories(
  query: {
    page: number;
    pageSize: number;
    keywords?: string;
    memoryType?: PlatformMemoryType;
    storageType?: string;
  },
  signal?: AbortSignal,
): Promise<PlatformMemoryListResult> {
  const data = asRecord(
    await platformRequest("/memories", {
      query: {
        page: query.page,
        page_size: query.pageSize,
        keywords: query.keywords?.trim() || undefined,
        memory_type: query.memoryType,
        storage_type: query.storageType,
      },
      signal,
    }),
  );
  const items = mappedList(data.memory_list, mapPlatformMemory);
  return { items, total: count(data.total_count, items.length) };
}

export async function createPlatformMemory(
  input: CreatePlatformMemoryInput,
  signal?: AbortSignal,
): Promise<PlatformMemory> {
  const raw = await platformRequest("/memories", {
    method: "POST",
    json: {
      name: input.name.trim(),
      memory_type: input.memoryTypes,
      embd_id: input.embeddingModelId,
      llm_id: input.llmId,
    },
    signal,
  });
  const mapped = mapPlatformMemory(raw);
  if (!mapped) throw new TypeError("Rag Platform hafıza yanıtı geçersiz.");
  return mapped;
}

export async function getPlatformMemoryConfig(
  memoryId: string,
  signal?: AbortSignal,
): Promise<PlatformMemory> {
  const raw = await platformRequest(
    `/memories/${segment(memoryId)}/config`,
    { signal },
  );
  const mapped = mapPlatformMemory(raw);
  if (!mapped) throw new TypeError("Rag Platform hafıza ayarı geçersiz.");
  return mapped;
}

export function updatePlatformMemory(
  memoryId: string,
  input: UpdatePlatformMemoryInput,
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest(`/memories/${segment(memoryId)}`, {
    method: "PUT",
    json: {
      ...(input.name !== undefined ? { name: input.name.trim() } : {}),
      ...(input.permissions ? { permissions: input.permissions } : {}),
      ...(input.llmId ? { llm_id: input.llmId } : {}),
      ...(input.embeddingModelId ? { embd_id: input.embeddingModelId } : {}),
      ...(input.memoryTypes ? { memory_type: input.memoryTypes } : {}),
      ...(input.memorySize !== undefined ? { memory_size: input.memorySize } : {}),
      ...(input.forgettingPolicy ? { forgetting_policy: input.forgettingPolicy } : {}),
      ...(input.temperature !== undefined ? { temperature: input.temperature } : {}),
      ...(input.description !== undefined ? { description: input.description } : {}),
      ...(input.systemPrompt !== undefined ? { system_prompt: input.systemPrompt } : {}),
      ...(input.userPrompt !== undefined ? { user_prompt: input.userPrompt } : {}),
    },
    signal,
  });
}

export function deletePlatformMemory(memoryId: string, signal?: AbortSignal) {
  return platformRequest<void>(`/memories/${segment(memoryId)}`, {
    method: "DELETE",
    signal,
  });
}

export async function listPlatformMemoryMessages(
  memoryId: string,
  query: { page: number; pageSize: number; keywords?: string; agentIds?: string[] },
  signal?: AbortSignal,
): Promise<PlatformMemoryMessageListResult> {
  const data = asRecord(
    await platformRequest(`/memories/${segment(memoryId)}`, {
      query: {
        page: query.page,
        page_size: query.pageSize,
        keywords: query.keywords?.trim() || undefined,
        agent_id: query.agentIds,
      },
      signal,
    }),
  );
  const messages = asRecord(data.messages);
  const items = mappedList(messages.message_list, mapPlatformMemoryMessage);
  return {
    items,
    total: count(messages.total_count, items.length),
    storageType: typeof data.storage_type === "string" ? data.storage_type : "table",
  };
}

export async function listRecentPlatformMemoryMessages(
  memoryIds: string[],
  query: { limit?: number; agentId?: string; sessionId?: string } = {},
  signal?: AbortSignal,
): Promise<PlatformMemoryMessage[]> {
  const data = await platformRequest("/messages", {
    query: {
      memory_id: memoryIds,
      limit: query.limit ?? 10,
      agent_id: query.agentId?.trim() || undefined,
      session_id: query.sessionId?.trim() || undefined,
    },
    signal,
  });
  return mappedList(data, mapPlatformMemoryMessage);
}

export async function searchPlatformMemoryMessages(
  memoryIds: string[],
  query: string,
  options: { topN?: number; similarityThreshold?: number; keywordWeight?: number; agentId?: string; sessionId?: string } = {},
  signal?: AbortSignal,
): Promise<PlatformMemoryMessage[]> {
  const data = await platformRequest("/messages/search", {
    query: {
      memory_id: memoryIds,
      query: query.trim(),
      top_n: options.topN ?? 10,
      similarity_threshold: options.similarityThreshold ?? 0.2,
      keywords_similarity_weight: options.keywordWeight ?? 0.7,
      agent_id: options.agentId?.trim() || undefined,
      session_id: options.sessionId?.trim() || undefined,
    },
    signal,
  });
  return mappedList(data, mapPlatformMemoryMessage);
}

export function addPlatformMemoryMessage(
  input: { memoryIds: string[]; agentId: string; sessionId: string; userInput: string; agentResponse: string },
  signal?: AbortSignal,
) {
  return platformRequest<void>("/messages", {
    method: "POST",
    json: {
      memory_id: input.memoryIds,
      agent_id: input.agentId.trim(),
      session_id: input.sessionId.trim(),
      user_input: input.userInput,
      agent_response: input.agentResponse,
    },
    signal,
  });
}

export function updatePlatformMemoryMessageStatus(
  memoryId: string,
  messageId: string,
  status: boolean,
  signal?: AbortSignal,
) {
  return platformRequest<void>(`/messages/${messageSegment(memoryId, messageId)}`, {
    method: "PUT",
    json: { status },
    signal,
  });
}

export function forgetPlatformMemoryMessage(
  memoryId: string,
  messageId: string,
  signal?: AbortSignal,
) {
  return platformRequest<void>(`/messages/${messageSegment(memoryId, messageId)}`, {
    method: "DELETE",
    signal,
  });
}

export async function getPlatformMemoryMessageContent(
  memoryId: string,
  messageId: string,
  signal?: AbortSignal,
): Promise<PlatformMemoryMessage> {
  const raw = await platformRequest(
    `/messages/${messageSegment(memoryId, messageId)}/content`,
    { signal },
  );
  const mapped = mapPlatformMemoryMessage(raw);
  if (!mapped) throw new TypeError("Rag Platform mesaj yanıtı geçersiz.");
  return mapped;
}
