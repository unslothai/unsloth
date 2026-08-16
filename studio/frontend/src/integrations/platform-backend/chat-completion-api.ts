import { platformOpenResponse, platformRequest } from "./client";
import { PlatformApiError } from "./errors";
import { parsePlatformSseStream } from "./sse";
import type {
  PlatformChatCitation,
  PlatformChatCompletionRequest,
  PlatformChatReference,
  PlatformChatStreamEvent,
  PlatformChatUsage,
  PlatformFeedbackRequest,
  PlatformMindMapNode,
} from "./chat-completion-types";
import type { PlatformSessionDto } from "./chat-types";

const COMPLETION_TIMEOUT_MS = 180_000;
export const PLATFORM_CHAT_AUDIO_MAX_BYTES = 25 * 1024 * 1024;
export const PLATFORM_CHAT_AUDIO_MAX_DURATION_MS = 120_000;
const ALLOWED_AUDIO_TYPES = new Set([
  "audio/wav",
  "audio/x-wav",
  "audio/mpeg",
  "audio/mp4",
  "audio/aac",
  "audio/flac",
  "audio/ogg",
  "audio/webm",
  "audio/opus",
  "audio/x-ms-wma",
]);

type Envelope = { code?: unknown; message?: unknown; data?: unknown };
type FrameData = Record<string, unknown>;

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function numberValue(value: unknown): number | null {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function objectValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function pageFromPositions(value: unknown): number | null {
  if (!Array.isArray(value) || value.length === 0) return null;
  const first = Array.isArray(value[0]) ? value[0]?.[0] : value[0];
  const page = numberValue(first);
  return page !== null && page >= 0 ? Math.floor(page) : null;
}

export function platformChatCitations(
  reference: PlatformChatReference | null,
) {
  return (reference?.chunks ?? []).map((citation) => ({
    id: citation.id,
    filename: citation.filename,
    page: citation.page,
    score: citation.score,
    text: citation.text,
    documentId: citation.documentId,
    chunkId: citation.chunkId,
    datasetId: citation.datasetId,
    source: "platform" as const,
  }));
}

export function mapPlatformChatReference(
  value: unknown,
): PlatformChatReference {
  const reference = objectValue(value) ?? {};
  const rawChunks = Array.isArray(reference.chunks) ? reference.chunks : [];
  const rawAggregations = Array.isArray(reference.doc_aggs)
    ? reference.doc_aggs
    : [];
  const chunks = rawChunks.flatMap((raw, index): PlatformChatCitation[] => {
    const chunk = objectValue(raw);
    if (!chunk) return [];
    const chunkId = stringValue(chunk.id ?? chunk.chunk_id) || null;
    const documentId = stringValue(chunk.document_id ?? chunk.doc_id) || null;
    const filename =
      stringValue(
        chunk.document_name ??
          chunk.doc_name ??
          chunk.docnm_kwd ??
          chunk.filename,
      ) ||
      `Kaynak ${index + 1}`;
    const score = numberValue(
      chunk.similarity ?? chunk.score ?? chunk.vector_similarity,
    );
    return [
      {
        id: chunkId ?? `${documentId ?? "source"}-${index}`,
        chunkId,
        documentId,
        datasetId: stringValue(chunk.dataset_id ?? chunk.kb_id) || null,
        filename,
        text: stringValue(chunk.content ?? chunk.content_with_weight),
        page: pageFromPositions(chunk.positions ?? chunk.position_int),
        score,
      },
    ];
  });
  const documentAggregations = rawAggregations.flatMap((raw) => {
    const aggregation = objectValue(raw);
    if (!aggregation) return [];
    return [
      {
        documentId:
          stringValue(aggregation.doc_id ?? aggregation.document_id) || null,
        filename:
          stringValue(
            aggregation.doc_name ??
              aggregation.document_name ??
              aggregation.filename,
          ) || "Belge",
        count: numberValue(aggregation.count ?? aggregation.chunk_count),
      },
    ];
  });
  return { chunks, documentAggregations };
}

function mapUsage(frame: FrameData): PlatformChatUsage | null {
  const raw = objectValue(frame.usage) ?? frame;
  const promptTokens = numberValue(raw.prompt_tokens);
  const completionTokens = numberValue(raw.completion_tokens);
  const totalTokens = numberValue(raw.total_tokens);
  if (
    promptTokens === null &&
    completionTokens === null &&
    totalTokens === null
  ) {
    return null;
  }
  return {
    ...(promptTokens === null ? {} : { promptTokens }),
    ...(completionTokens === null ? {} : { completionTokens }),
    ...(totalTokens === null ? {} : { totalTokens }),
  };
}

function envelopeCode(value: unknown): number | string {
  return typeof value === "number" || typeof value === "string" ? value : 0;
}

function isSuccessCode(value: unknown): boolean {
  return value === undefined || value === null || value === 0 || value === "0";
}

function cumulativeDelta(previous: string, incoming: string): string {
  if (!incoming) return "";
  if (incoming === previous) return "";
  if (incoming.startsWith(previous)) return incoming.slice(previous.length);
  if (previous.startsWith(incoming)) return "";
  return incoming;
}

/** Normalize the native Rag Platform completion protocol. */
export async function* streamPlatformChatCompletion(
  request: PlatformChatCompletionRequest,
  signal?: AbortSignal,
): AsyncGenerator<PlatformChatStreamEvent> {
  const handle = await platformOpenResponse("/chat/completions", {
    method: "POST",
    json: {
      chat_id: request.chatId,
      session_id: request.sessionId,
      question: request.question,
      stream: true,
      legacy: request.legacy ?? false,
      ...(request.thinking && request.thinking !== "default"
        ? { thinking: request.thinking }
        : {}),
    },
    signal,
    timeoutMs: COMPLETION_TIMEOUT_MS,
  });

  const { response } = handle;
  let text = "";
  let reasoning = "";
  let reference: PlatformChatReference | null = null;
  let usage: PlatformChatUsage | null = null;
  let messageId: string | null = null;
  let chatId: string | null = request.chatId;
  let sessionId: string | null = request.sessionId;
  let terminal = false;
  let reasoningActive = false;
  let previousWireAnswer = "";
  let previousFrame = "";

  try {
    const contentType =
      response.headers.get("content-type")?.toLowerCase() ?? "";
    if (!response.body) {
      throw new PlatformApiError("Rag Platform stream gövdesi döndürmedi.", {
        httpStatus: response.status,
        code: "MISSING_STREAM_BODY",
        endpoint: "/chat/completions",
      });
    }
    if (!contentType.includes("text/event-stream")) {
      const body = (await response.json().catch(() => null)) as Envelope | null;
      if (!body || !isSuccessCode(body.code)) {
        throw new PlatformApiError(
          stringValue(body?.message) ||
            "Rag Platform completion isteğini reddetti.",
          {
            httpStatus: response.status,
            code: envelopeCode(body?.code ?? "INVALID_STREAM_RESPONSE"),
            endpoint: "/chat/completions",
          },
        );
      }
      throw new PlatformApiError(
        "Rag Platform SSE yerine geçersiz yanıt döndürdü.",
        {
          httpStatus: response.status,
          code: "INVALID_STREAM_RESPONSE",
          endpoint: "/chat/completions",
        },
      );
    }

    for await (const event of parsePlatformSseStream(response.body, signal)) {
      if (signal?.aborted) throw signal.reason;
      if (!event.terminal && event.data === previousFrame) continue;
      previousFrame = event.data;
      let envelope: Envelope;
      try {
        envelope = JSON.parse(event.data) as Envelope;
      } catch (cause) {
        throw new PlatformApiError(
          "Rag Platform geçersiz SSE verisi döndürdü.",
          {
            httpStatus: response.status,
            code: "INVALID_STREAM_FRAME",
            endpoint: "/chat/completions",
            cause,
          },
        );
      }
      if (!isSuccessCode(envelope.code)) {
        const message =
          stringValue(envelope.message) || "Rag Platform stream hatası.";
        yield { type: "error", code: envelopeCode(envelope.code), message };
        throw new PlatformApiError(message, {
          httpStatus: response.status,
          code: envelopeCode(envelope.code),
          endpoint: "/chat/completions",
        });
      }
      if (envelope.data === true || event.terminal) {
        terminal = true;
        if (reasoningActive) {
          reasoningActive = false;
          yield { type: "reasoning-end", text: reasoning };
        }
        yield {
          type: "final",
          terminal: true,
          messageId,
          chatId,
          sessionId,
          text,
          reasoning,
          reference,
          usage,
        };
        break;
      }
      const frame = objectValue(envelope.data);
      if (!frame) continue;
      messageId = stringValue(frame.id) || messageId;
      chatId = stringValue(frame.chat_id) || chatId;
      sessionId = stringValue(frame.session_id) || sessionId;

      if (frame.start_to_think === true && !reasoningActive) {
        reasoningActive = true;
        yield { type: "reasoning-start" };
      }

      const incomingAnswer = stringValue(frame.answer);
      if (incomingAnswer) {
        const isCumulative = request.legacy === true || frame.final === true;
        const delta = isCumulative
          ? cumulativeDelta(text, incomingAnswer)
          : incomingAnswer;
        previousWireAnswer = incomingAnswer;
        if (reasoningActive) {
          reasoning += delta;
          if (delta) yield { type: "reasoning-delta", delta, text: reasoning };
        } else {
          text += delta;
          if (delta) yield { type: "text-delta", delta, text };
        }
      } else if (request.legacy && previousWireAnswer) {
        previousWireAnswer = "";
      }

      if (frame.end_to_think === true && reasoningActive) {
        reasoningActive = false;
        yield { type: "reasoning-end", text: reasoning };
      }

      if (frame.reference !== undefined) {
        reference = mapPlatformChatReference(frame.reference);
        yield { type: "reference-update", reference };
      }
      const nextUsage = mapUsage(frame);
      if (nextUsage) {
        usage = nextUsage;
        yield { type: "usage", usage };
      }
      if (frame.final === true) {
        yield {
          type: "final",
          terminal: false,
          messageId,
          chatId,
          sessionId,
          text,
          reasoning,
          reference,
          usage,
        };
      }
    }
    if (!terminal) {
      throw new PlatformApiError(
        "Rag Platform stream bağlantısı tamamlanmadan kesildi.",
        {
          httpStatus: response.status,
          code: "STREAM_INTERRUPTED",
          endpoint: "/chat/completions",
        },
      );
    }
  } finally {
    handle.close();
  }
}

export function updatePlatformMessageFeedback(
  chatId: string,
  sessionId: string,
  messageId: string,
  payload: PlatformFeedbackRequest,
  signal?: AbortSignal,
): Promise<PlatformSessionDto> {
  return platformRequest(
    `/chats/${encodeURIComponent(chatId)}/sessions/${encodeURIComponent(sessionId)}/messages/${encodeURIComponent(messageId)}/feedback`,
    { method: "PUT", json: payload, signal },
  );
}

function normalizeMindMap(
  value: unknown,
  path = "root",
): PlatformMindMapNode | null {
  const node = objectValue(value);
  if (!node) return null;
  const label =
    stringValue(
      node.label ?? node.name ?? node.title ?? node.topic ?? node.id,
    ) || (path === "root" ? "Mindmap" : path);
  const id = stringValue(node.id) || `${path}-${label}`;
  const rawChildren = Array.isArray(node.children)
    ? node.children
    : Array.isArray(node.nodes)
      ? node.nodes
      : [];
  const children = rawChildren.flatMap((child, index) => {
    const normalized = normalizeMindMap(child, `${path}-${index}`);
    return normalized ? [normalized] : [];
  });
  if (
    path === "root" &&
    children.length === 0 &&
    !stringValue(node.label ?? node.name ?? node.title ?? node.topic ?? node.id)
  ) {
    return null;
  }
  return {
    id,
    label,
    children,
  };
}

export async function generatePlatformMindMap(
  question: string,
  datasetIds: string[],
  signal?: AbortSignal,
): Promise<PlatformMindMapNode | null> {
  const data = await platformRequest<unknown>("/chat/mindmap", {
    method: "POST",
    json: { question, kb_ids: datasetIds },
    signal,
    timeoutMs: COMPLETION_TIMEOUT_MS,
  });
  return normalizeMindMap(data);
}

export function getPlatformRecommendations(
  question: string,
  signal?: AbortSignal,
): Promise<string[]> {
  return platformRequest("/chat/recommendation", {
    method: "POST",
    json: { question },
    signal,
    timeoutMs: COMPLETION_TIMEOUT_MS,
  });
}

export function synthesizePlatformChatSpeech(
  text: string,
  signal?: AbortSignal,
): Promise<Blob> {
  return platformRequest("/chat/audio/speech", {
    method: "POST",
    json: { text },
    signal,
    timeoutMs: COMPLETION_TIMEOUT_MS,
    responseType: "blob",
  });
}

function audioExtension(type: string): string {
  if (type.includes("webm")) return ".webm";
  if (type.includes("ogg") || type.includes("opus")) return ".ogg";
  if (type.includes("mp4") || type.includes("aac")) return ".m4a";
  if (type.includes("mpeg")) return ".mp3";
  if (type.includes("flac")) return ".flac";
  return ".wav";
}

export async function transcribePlatformChatAudio(
  blob: Blob,
  signal?: AbortSignal,
): Promise<string> {
  const mime = (blob.type || "audio/wav").split(";", 1)[0]?.toLowerCase() ?? "";
  if (!ALLOWED_AUDIO_TYPES.has(mime)) {
    throw new TypeError("Rag Platform bu ses biçimini desteklemiyor.");
  }
  if (blob.size <= 0 || blob.size > PLATFORM_CHAT_AUDIO_MAX_BYTES) {
    throw new TypeError("Ses kaydı boş veya 25 MB sınırını aşıyor.");
  }
  const form = new FormData();
  form.append("file", blob, `recording${audioExtension(mime)}`);
  form.append("stream", "false");
  const data = await platformRequest<{ text?: unknown }>(
    "/chat/audio/transcription",
    {
      method: "POST",
      body: form,
      signal,
      timeoutMs: COMPLETION_TIMEOUT_MS,
    },
  );
  return stringValue(data?.text).trim();
}
