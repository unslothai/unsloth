import { platformRequest } from "./client";
import type {
  PlatformChatCreateRequest,
  PlatformChatDto,
  PlatformChatPage,
  PlatformChatUpdateRequest,
  PlatformSessionDeleteResult,
  PlatformSessionDto,
  PlatformSessionListQuery,
} from "./chat-types";

function nonNegativeInteger(value: unknown, fallback: number): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : fallback;
}

export async function listPlatformChats(
  query: {
    page?: number;
    pageSize?: number;
    keywords?: string;
    orderby?: "create_time" | "update_time" | "name";
    desc?: boolean;
  } = {},
  signal?: AbortSignal,
): Promise<PlatformChatPage> {
  const data = await platformRequest<{
    chats?: unknown;
    total?: unknown;
  }>("/chats", {
    query: {
      page: query.page ?? 1,
      page_size: query.pageSize ?? 100,
      keywords: query.keywords?.trim() || undefined,
      orderby: query.orderby ?? "update_time",
      desc: query.desc ?? true,
    },
    signal,
  });
  const chats = Array.isArray(data?.chats)
    ? (data.chats as PlatformChatDto[])
    : [];
  return { chats, total: nonNegativeInteger(data?.total, chats.length) };
}

export async function listAllPlatformChats(
  signal?: AbortSignal,
): Promise<PlatformChatDto[]> {
  const pageSize = 100;
  const first = await listPlatformChats({ page: 1, pageSize }, signal);
  const chats = [...first.chats];
  for (let page = 2; chats.length < first.total; page += 1) {
    const next = await listPlatformChats({ page, pageSize }, signal);
    chats.push(...next.chats);
    if (next.chats.length === 0) break;
  }
  return chats;
}

export function getPlatformChat(
  chatId: string,
  signal?: AbortSignal,
): Promise<PlatformChatDto> {
  return platformRequest(`/chats/${encodeURIComponent(chatId)}`, { signal });
}

export function createPlatformChat(
  payload: PlatformChatCreateRequest,
  signal?: AbortSignal,
): Promise<PlatformChatDto> {
  return platformRequest("/chats", {
    method: "POST",
    json: payload,
    signal,
  });
}

export function updatePlatformChat(
  chatId: string,
  payload: PlatformChatUpdateRequest,
  signal?: AbortSignal,
): Promise<PlatformChatDto> {
  return platformRequest(`/chats/${encodeURIComponent(chatId)}`, {
    method: "PATCH",
    json: payload,
    signal,
  });
}

/** Full-replacement compatibility contract. Product UI uses PATCH. */
export function replacePlatformChat(
  chatId: string,
  payload: PlatformChatCreateRequest,
  signal?: AbortSignal,
): Promise<PlatformChatDto> {
  return platformRequest(`/chats/${encodeURIComponent(chatId)}`, {
    method: "PUT",
    json: payload,
    signal,
  });
}

export function deletePlatformChat(
  chatId: string,
  signal?: AbortSignal,
): Promise<void> {
  return platformRequest(`/chats/${encodeURIComponent(chatId)}`, {
    method: "DELETE",
    signal,
  });
}

export function deleteAllPlatformChats(signal?: AbortSignal): Promise<void> {
  return platformRequest("/chats", {
    method: "DELETE",
    json: { delete_all: true },
    signal,
  });
}

function sessionPath(chatId: string, sessionId?: string): string {
  const collection = `/chats/${encodeURIComponent(chatId)}/sessions`;
  return sessionId
    ? `${collection}/${encodeURIComponent(sessionId)}`
    : collection;
}

export function listPlatformSessions(
  chatId: string,
  query: PlatformSessionListQuery = {},
  signal?: AbortSignal,
): Promise<PlatformSessionDto[]> {
  return platformRequest(sessionPath(chatId), {
    query: {
      page: query.page ?? 1,
      page_size: query.pageSize ?? 100,
      id: query.id,
      name: query.name?.trim() || undefined,
      orderby: query.orderby ?? "update_time",
      desc: query.desc ?? true,
    },
    signal,
  });
}

export async function listAllPlatformSessions(
  chatId: string,
  signal?: AbortSignal,
): Promise<PlatformSessionDto[]> {
  const pageSize = 100;
  const sessions: PlatformSessionDto[] = [];
  const seenIds = new Set<string>();
  for (let page = 1; ; page += 1) {
    const next = await listPlatformSessions(
      chatId,
      { page, pageSize, orderby: "update_time", desc: true },
      signal,
    );
    let newIds = 0;
    for (const session of Array.isArray(next) ? next : []) {
      const id = typeof session.id === "string" ? session.id : "";
      if (!id || !seenIds.has(id)) {
        sessions.push(session);
        if (id) seenIds.add(id);
        newIds += 1;
      }
    }
    if (!Array.isArray(next) || next.length < pageSize) break;
    if (newIds === 0) break;
  }
  return sessions;
}

export function createPlatformSession(
  chatId: string,
  payload: { name?: string } = {},
  signal?: AbortSignal,
): Promise<PlatformSessionDto> {
  return platformRequest(sessionPath(chatId), {
    method: "POST",
    json: payload,
    signal,
  });
}

export function getPlatformSession(
  chatId: string,
  sessionId: string,
  signal?: AbortSignal,
): Promise<PlatformSessionDto> {
  return platformRequest(sessionPath(chatId, sessionId), { signal });
}

export function updatePlatformSession(
  chatId: string,
  sessionId: string,
  payload: { name: string },
  signal?: AbortSignal,
): Promise<PlatformSessionDto> {
  return platformRequest(sessionPath(chatId, sessionId), {
    method: "PATCH",
    json: payload,
    signal,
  });
}

/** Deprecated upstream alias retained only for contract/auth verification. */
export function updatePlatformSessionCompatibility(
  chatId: string,
  sessionId: string,
  payload: { name: string },
  signal?: AbortSignal,
): Promise<PlatformSessionDto> {
  return platformRequest(sessionPath(chatId, sessionId), {
    method: "PUT",
    json: payload,
    signal,
  });
}

/** Deprecated upstream alias; Phase 8 owns the canonical recommendation UI. */
export function getPlatformRelatedQuestionsCompatibility(
  question: string,
  signal?: AbortSignal,
): Promise<string[]> {
  return platformRequest("/sessions/related_questions", {
    method: "POST",
    json: { question },
    signal,
  });
}

export async function deletePlatformSessions(
  chatId: string,
  sessionIds: string[],
  signal?: AbortSignal,
): Promise<PlatformSessionDeleteResult> {
  const data = await platformRequest<Record<string, unknown>>(
    sessionPath(chatId),
    { method: "DELETE", json: { ids: sessionIds }, signal },
  );
  const deleted = data?.deleted_ids ?? data?.deletedIds;
  const missing = data?.not_found_ids ?? data?.notFoundIds;
  return {
    deletedIds: Array.isArray(deleted)
      ? deleted.filter((id): id is string => typeof id === "string")
      : [...sessionIds],
    notFoundIds: Array.isArray(missing)
      ? missing.filter((id): id is string => typeof id === "string")
      : [],
  };
}

export function deletePlatformSessionMessage(
  chatId: string,
  sessionId: string,
  messageId: string,
  signal?: AbortSignal,
): Promise<PlatformSessionDto> {
  return platformRequest(
    `${sessionPath(chatId, sessionId)}/messages/${encodeURIComponent(messageId)}`,
    { method: "DELETE", signal },
  );
}
