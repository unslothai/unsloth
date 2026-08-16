import { platformOpenResponse } from "./client";
import { PlatformApiError } from "./errors";
import { parsePlatformSseStream } from "./sse";
import type {
  PlatformAgentCompletionRequest,
  PlatformAgentRunRequest,
  PlatformAgentStreamEvent,
} from "./agent-types";

const STREAM_TIMEOUT_MS = 180_000;

function text(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function object(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

async function* openAgentStream(
  endpoint: string,
  json: unknown,
  signal?: AbortSignal,
): AsyncGenerator<PlatformAgentStreamEvent> {
  const handle = await platformOpenResponse(endpoint, {
    method: "POST",
    json,
    signal,
    timeoutMs: STREAM_TIMEOUT_MS,
  });
  let terminal = false;
  let messageId: string | null = null;
  let sessionId: string | null = null;
  try {
    const contentType =
      handle.response.headers.get("content-type")?.toLowerCase() ?? "";
    if (!handle.response.body || !contentType.includes("text/event-stream")) {
      const payload = await handle.response.json().catch(() => null);
      const body = object(payload);
      throw new PlatformApiError(
        text(body?.message) || "Rag Platform agent stream yanıtı geçersiz.",
        {
          httpStatus: handle.response.status,
          code:
            (body?.code as number | string | undefined) ??
            "INVALID_AGENT_STREAM",
          endpoint,
        },
      );
    }

    for await (const frame of parsePlatformSseStream(
      handle.response.body,
      signal,
    )) {
      if (frame.terminal) {
        terminal = true;
        yield { type: "done", messageId, sessionId };
        break;
      }
      let parsed: unknown;
      try {
        parsed = JSON.parse(frame.data) as unknown;
      } catch (cause) {
        throw new PlatformApiError(
          "Rag Platform geçersiz agent SSE verisi döndürdü.",
          {
            httpStatus: handle.response.status,
            code: "INVALID_AGENT_STREAM_FRAME",
            endpoint,
            cause,
          },
        );
      }
      const envelope = object(parsed);
      if (
        envelope &&
        "code" in envelope &&
        envelope.code !== 0 &&
        envelope.code !== "0"
      ) {
        const message =
          text(envelope.message) || "Rag Platform agent stream hatası.";
        yield {
          type: "error",
          message,
          code: envelope.code as number | string,
        };
        throw new PlatformApiError(message, {
          httpStatus: handle.response.status,
          code: envelope.code as number | string,
          endpoint,
        });
      }
      const payload = object(envelope?.data) ?? envelope ?? {};
      messageId = text(payload.message_id ?? payload.id) || messageId;
      sessionId = text(payload.session_id) || sessionId;
      const eventName = text(payload.event) || frame.event || "message";
      if (eventName === "done" || payload.done === true) {
        terminal = true;
        yield { type: "done", messageId, sessionId };
        break;
      }
      if (eventName === "error") {
        const message =
          text(payload.message ?? payload.error) ||
          "Rag Platform agent çalıştırması hata verdi.";
        yield { type: "error", message };
        throw new PlatformApiError(message, {
          httpStatus: handle.response.status,
          code: "AGENT_STREAM_ERROR",
          endpoint,
        });
      }
      yield {
        type: "event",
        event: eventName,
        data: payload.data ?? payload,
        messageId,
        sessionId,
      };
    }
    if (!terminal) {
      throw new PlatformApiError(
        "Rag Platform agent stream'i tamamlanmadan kapandı.",
        {
          httpStatus: handle.response.status,
          code: "INCOMPLETE_AGENT_STREAM",
          endpoint,
        },
      );
    }
  } finally {
    handle.close();
  }
}

export function streamAgentCompletion(
  request: PlatformAgentCompletionRequest,
  signal?: AbortSignal,
) {
  return openAgentStream(
    "/agents/chat/completions",
    {
      agent_id: request.agentId,
      query: request.query,
      stream: true,
      ...(request.sessionId ? { session_id: request.sessionId } : {}),
      ...(request.inputs ? { inputs: request.inputs } : {}),
      ...(request.files ? { files: request.files } : {}),
      ...(request.returnTrace ? { return_trace: true } : {}),
    },
    signal,
  );
}

export function streamAgentRun(
  request: PlatformAgentRunRequest,
  signal?: AbortSignal,
) {
  const query = new URLSearchParams();
  if (request.sessionId) query.set("session_id", request.sessionId);
  if (request.version) query.set("version", request.version);
  const suffix = query.size ? `?${query.toString()}` : "";
  return openAgentStream(
    `/agents/${encodeURIComponent(request.agentId)}/run${suffix}`,
    { user_input: request.userInput },
    signal,
  );
}
