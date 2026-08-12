import { PlatformApiError } from "./errors";
import type { PlatformCode, PlatformEnvelope } from "./types";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export function isPlatformEnvelope<TData = unknown>(
  value: unknown,
): value is PlatformEnvelope<TData> {
  return isRecord(value) && "code" in value && "data" in value;
}

function envelopeMessage(value: PlatformEnvelope<unknown>): string {
  return typeof value.message === "string" && value.message.trim()
    ? value.message
    : "Rag Platform isteği tamamlanamadı.";
}

export function unwrapPlatformEnvelope<TData>(
  value: unknown,
  context: {
    endpoint: string;
    httpStatus: number;
    requestId?: string;
  },
): TData {
  if (!isPlatformEnvelope<TData>(value)) {
    throw new PlatformApiError("Rag Platform geçersiz bir yanıt döndürdü.", {
      httpStatus: context.httpStatus,
      code: "INVALID_RESPONSE",
      endpoint: context.endpoint,
      requestId: context.requestId,
    });
  }

  if (value.code !== 0) {
    throw new PlatformApiError(envelopeMessage(value), {
      httpStatus: context.httpStatus,
      code: value.code as PlatformCode,
      endpoint: context.endpoint,
      requestId: context.requestId,
    });
  }

  return value.data;
}
