import {
  getPlatformBackendConfig,
  resolvePlatformUrl,
  type PlatformBackendConfig,
  type PlatformPathMode,
} from "./config";
import { unwrapPlatformEnvelope } from "./envelope";
import { PlatformApiError, isPlatformApiError } from "./errors";
import type {
  PlatformEnvelope,
  PlatformQuery,
  PlatformResponseType,
} from "./types";

export interface PlatformRequestOptions {
  method?: string;
  token?: string | null;
  query?: PlatformQuery;
  json?: unknown;
  body?: BodyInit | null;
  headers?: HeadersInit;
  signal?: AbortSignal;
  timeoutMs?: number;
  responseType?: PlatformResponseType;
  pathMode?: PlatformPathMode;
  getRetries?: number;
  credentials?: RequestCredentials;
  config?: PlatformBackendConfig;
}

const RETRYABLE_HTTP_STATUSES = new Set([502, 503, 504]);
const MAX_GET_RETRIES = 2;

function appendQuery(url: string, query: PlatformQuery | undefined): string {
  if (!query) return url;
  const [base, fragment] = url.split("#", 2);
  const separator = base.includes("?") ? "&" : "?";
  const params = new URLSearchParams();

  for (const [key, rawValue] of Object.entries(query)) {
    if (rawValue === null || rawValue === undefined) continue;
    const values = Array.isArray(rawValue) ? rawValue : [rawValue];
    for (const value of values) params.append(key, String(value));
  }

  const serialized = params.toString();
  if (!serialized) return url;
  return `${base}${separator}${serialized}${fragment ? `#${fragment}` : ""}`;
}

function requestIdFrom(response: Response): string | undefined {
  return (
    response.headers.get("x-request-id") ??
    response.headers.get("x-correlation-id") ??
    response.headers.get("request-id") ??
    undefined
  );
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return undefined;
  }
}

function backendErrorMessage(body: unknown, status: number): string {
  if (
    typeof body === "object" &&
    body !== null &&
    "message" in body &&
    typeof body.message === "string" &&
    body.message.trim()
  ) {
    return body.message;
  }
  return status >= 500
    ? "Rag Platform şu anda isteği tamamlayamıyor."
    : "Rag Platform isteği reddetti.";
}

function errorCode(body: unknown, status: number): number | string {
  if (
    typeof body === "object" &&
    body !== null &&
    "code" in body &&
    (typeof body.code === "number" || typeof body.code === "string")
  ) {
    return body.code;
  }
  return `HTTP_${status}`;
}

function createAbortContext(
  signal: AbortSignal | undefined,
  timeoutMs: number,
) {
  const controller = new AbortController();
  let timedOut = false;
  const onAbort = () => controller.abort(signal?.reason);

  if (signal?.aborted) controller.abort(signal.reason);
  else signal?.addEventListener("abort", onAbort, { once: true });

  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort(new DOMException("Request timed out", "TimeoutError"));
  }, timeoutMs);

  return {
    signal: controller.signal,
    didTimeOut: () => timedOut,
    cleanup: () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    },
  };
}

async function waitForRetry(
  attempt: number,
  signal?: AbortSignal,
): Promise<void> {
  const delayMs = 80 * 2 ** attempt + Math.floor(Math.random() * 40);
  await new Promise<void>((resolve, reject) => {
    const finish = () => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    };
    const timer = setTimeout(finish, delayMs);
    const onAbort = () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
      reject(signal?.reason ?? new DOMException("Aborted", "AbortError"));
    };
    if (signal?.aborted) onAbort();
    else signal?.addEventListener("abort", onAbort, { once: true });
  });
}

async function executeRequest<TData>(
  endpoint: string,
  url: string,
  options: PlatformRequestOptions,
  method: string,
  headers: Headers,
  body: BodyInit | null | undefined,
  responseType: PlatformResponseType,
  timeoutMs: number,
): Promise<TData> {
  const abortContext = createAbortContext(options.signal, timeoutMs);

  try {
    const response = await fetch(url, {
      method,
      headers,
      body,
      signal: abortContext.signal,
      credentials: options.credentials ?? "same-origin",
    });
    const requestId = requestIdFrom(response);

    if (response.status === 204) return undefined as TData;

    if (responseType === "blob" && response.ok) {
      return (await response.blob()) as TData;
    }

    const text = await response.text();
    const parsed = text ? parseJson(text) : undefined;

    if (!response.ok) {
      throw new PlatformApiError(backendErrorMessage(parsed, response.status), {
        httpStatus: response.status,
        code: errorCode(parsed, response.status),
        endpoint,
        requestId,
      });
    }

    if (!text) return undefined as TData;
    if (responseType === "text") return text as TData;
    if (responseType === "json") {
      if (parsed === undefined) {
        throw new PlatformApiError("Rag Platform geçersiz JSON döndürdü.", {
          httpStatus: response.status,
          code: "INVALID_RESPONSE",
          endpoint,
          requestId,
        });
      }
      return parsed as TData;
    }

    return unwrapPlatformEnvelope<TData>(parsed, {
      endpoint,
      httpStatus: response.status,
      requestId,
    });
  } catch (error) {
    if (isPlatformApiError(error)) throw error;
    if (abortContext.didTimeOut()) {
      throw new PlatformApiError("Rag Platform isteği zaman aşımına uğradı.", {
        httpStatus: null,
        code: "CLIENT_TIMEOUT",
        endpoint,
        cause: error,
      });
    }
    if (options.signal?.aborted || abortContext.signal.aborted) {
      throw new PlatformApiError("Rag Platform isteği iptal edildi.", {
        httpStatus: null,
        code: "CLIENT_ABORTED",
        endpoint,
        cause: error,
      });
    }
    throw new PlatformApiError("Rag Platform bağlantısı kurulamadı.", {
      httpStatus: null,
      code: "NETWORK_ERROR",
      endpoint,
      cause: error,
    });
  } finally {
    abortContext.cleanup();
  }
}

export async function platformRequest<TData>(
  endpoint: string,
  options: PlatformRequestOptions = {},
): Promise<TData> {
  if (options.json !== undefined && options.body !== undefined) {
    throw new TypeError(
      "Rag Platform isteğinde json ve body birlikte kullanılamaz.",
    );
  }

  const method = (options.method ?? "GET").toUpperCase();
  const config = options.config ?? getPlatformBackendConfig();
  const url = appendQuery(
    resolvePlatformUrl(endpoint, options.pathMode ?? "api", config),
    options.query,
  );
  const headers = new Headers(options.headers);
  let body = options.body;

  if (options.token) headers.set("Authorization", `Bearer ${options.token}`);
  if (options.json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(options.json);
  }

  const responseType = options.responseType ?? "envelope";
  const timeoutMs = Math.max(1, options.timeoutMs ?? config.requestTimeoutMs);
  const retries =
    method === "GET"
      ? Math.min(MAX_GET_RETRIES, Math.max(0, options.getRetries ?? 1))
      : 0;

  for (let attempt = 0; ; attempt += 1) {
    try {
      return await executeRequest<TData>(
        endpoint,
        url,
        options,
        method,
        headers,
        body,
        responseType,
        timeoutMs,
      );
    } catch (error) {
      const canRetry =
        attempt < retries &&
        isPlatformApiError(error) &&
        (error.code === "NETWORK_ERROR" ||
          (error.httpStatus !== null &&
            RETRYABLE_HTTP_STATUSES.has(error.httpStatus)));
      if (!canRetry) throw error;
      try {
        await waitForRetry(attempt, options.signal);
      } catch (retryError) {
        throw new PlatformApiError("Rag Platform isteği iptal edildi.", {
          httpStatus: null,
          code: "CLIENT_ABORTED",
          endpoint,
          cause: retryError,
        });
      }
    }
  }
}

export type { PlatformEnvelope };
