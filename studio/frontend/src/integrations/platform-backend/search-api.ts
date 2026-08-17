import { mapPlatformChatReference } from "./chat-completion-api";
import { platformOpenResponse, platformRequest } from "./client";
import { PlatformApiError } from "./errors";
import { asRecord, stringValue } from "./model-types";
import { parsePlatformSseStream } from "./sse";
import type {
  PlatformSearchApp,
  PlatformSearchConfig,
  PlatformSearchListResult,
  PlatformSearchStreamEvent,
} from "./search-types";
import {
  mapPlatformSearchApp,
  serializePlatformSearchConfig,
} from "./search-types";

const segment = (value: string) => encodeURIComponent(value.trim());
const list = (value: unknown) =>
  (Array.isArray(value) ? value : [])
    .map(mapPlatformSearchApp)
    .filter((item): item is PlatformSearchApp => item !== null);
const total = (value: unknown, fallback: number) => {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : fallback;
};

export async function listPlatformSearches(
  query: {
    page: number;
    pageSize: number;
    keywords?: string;
    ownerIds?: string[];
  },
  signal?: AbortSignal,
): Promise<PlatformSearchListResult> {
  const data = asRecord(
    await platformRequest("/searches", {
      query: {
        page: query.page,
        page_size: query.pageSize,
        keywords: query.keywords?.trim() || undefined,
        owner_ids: query.ownerIds,
        orderby: "update_time",
        desc: true,
      },
      signal,
    }),
  );
  const items = list(data.search_apps);
  return { items, total: total(data.total, items.length) };
}

export async function createPlatformSearch(
  input: { name: string; description?: string },
  signal?: AbortSignal,
): Promise<string> {
  const data = asRecord(
    await platformRequest("/searches", {
      method: "POST",
      json: {
        name: input.name.trim(),
        description: input.description?.trim() ?? "",
      },
      signal,
    }),
  );
  const id = stringValue(data.search_id).trim();
  if (!id) throw new TypeError("Rag Platform arama oluşturma yanıtı geçersiz.");
  return id;
}

export async function getPlatformSearch(
  searchId: string,
  signal?: AbortSignal,
): Promise<PlatformSearchApp> {
  const raw = await platformRequest(`/searches/${segment(searchId)}`, {
    signal,
  });
  const mapped = mapPlatformSearchApp(raw);
  if (!mapped) throw new TypeError("Rag Platform arama yanıtı geçersiz.");
  return mapped;
}

export async function updatePlatformSearch(
  searchId: string,
  input: { name: string; description: string; config: PlatformSearchConfig },
  signal?: AbortSignal,
): Promise<PlatformSearchApp> {
  const raw = await platformRequest(`/searches/${segment(searchId)}`, {
    method: "PUT",
    json: {
      name: input.name.trim(),
      description: input.description,
      search_config: serializePlatformSearchConfig(input.config),
    },
    signal,
  });
  const mapped = mapPlatformSearchApp(raw);
  if (!mapped)
    throw new TypeError("Rag Platform arama güncelleme yanıtı geçersiz.");
  return mapped;
}

export function deletePlatformSearch(searchId: string, signal?: AbortSignal) {
  return platformRequest<void>(`/searches/${segment(searchId)}`, {
    method: "DELETE",
    signal,
  });
}

async function* streamCompletionPath(
  path: string,
  question: string,
  datasetIds: string[],
  signal?: AbortSignal,
): AsyncGenerator<PlatformSearchStreamEvent> {
  const handle = await platformOpenResponse(path, {
    method: "POST",
    json: {
      question: question.trim(),
      ...(datasetIds.length ? { kb_ids: datasetIds } : {}),
    },
    signal,
    timeoutMs: 180_000,
  });
  try {
    const { response } = handle;
    if (!response.body) {
      throw new PlatformApiError("Rag Platform stream gövdesi döndürmedi.", {
        httpStatus: response.status,
        code: "MISSING_STREAM_BODY",
        endpoint: path,
      });
    }
    let previous = "";
    let answerEnded = false;
    for await (const event of parsePlatformSseStream(response.body, signal)) {
      if (event.terminal) {
        yield { type: "done" };
        return;
      }
      let envelope: Record<string, unknown>;
      try {
        envelope = asRecord(JSON.parse(event.data));
      } catch {
        throw new PlatformApiError("Rag Platform SSE yanıtı geçersiz.", {
          httpStatus: response.status,
          code: "INVALID_SSE_FRAME",
          endpoint: path,
        });
      }
      const code = Number(envelope.code ?? 0);
      const data = asRecord(envelope.data);
      if (code !== 0) {
        throw new PlatformApiError(
          stringValue(envelope.message) ||
            "Rag Platform aramayı tamamlayamadı.",
          { httpStatus: response.status, code, endpoint: path },
        );
      }
      const rawAnswer = answerEnded ? "" : stringValue(data.answer);
      const doneMarker = rawAnswer.indexOf("[DONE]");
      const incoming =
        doneMarker === -1 ? rawAnswer : rawAnswer.slice(0, doneMarker);
      if (doneMarker !== -1) answerEnded = true;
      const delta = incoming.startsWith(previous)
        ? incoming.slice(previous.length)
        : incoming;
      if (incoming) previous = incoming;
      if (delta) yield { type: "answer", answer: delta };
      const reference = mapPlatformChatReference(data.reference);
      if (reference.chunks.length || reference.documentAggregations.length) {
        yield { type: "reference", reference };
      }
      if (data.final === true) yield { type: "done" };
    }
  } finally {
    handle.close();
  }
}

export function streamPlatformSearchCompletion(
  searchId: string,
  question: string,
  datasetIds: string[] = [],
  signal?: AbortSignal,
) {
  return streamCompletionPath(
    `/searches/${segment(searchId)}/completions`,
    question,
    datasetIds,
    signal,
  );
}

/** Contract adapter for the active singular Go alias. */
export function streamPlatformSearchCompletionAlias(
  searchId: string,
  question: string,
  datasetIds: string[] = [],
  signal?: AbortSignal,
) {
  return streamCompletionPath(
    `/searches/${segment(searchId)}/completion`,
    question,
    datasetIds,
    signal,
  );
}
