import { platformRequest } from "./client";
import { PlatformApiError, isPlatformApiError } from "./errors";
import { getPlatformDataset, updatePlatformDataset } from "./dataset-api";
import {
  mapConnectorOAuthStart,
  mapPlatformConnector,
  mapPlatformConnectorLog,
  type CreatePlatformConnectorInput,
  type PlatformConnector,
  type PlatformConnectorLogsPage,
  type PlatformConnectorOAuthSource,
  type PlatformConnectorOAuthStart,
  type PlatformGoogleConnectorSource,
  type UpdatePlatformConnectorInput,
} from "./connector-types";

const OAUTH_PENDING_CODE = 106;
const OAUTH_POLL_INTERVAL_MAX_MS = 5_000;

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function array(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function numberValue(value: unknown): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

export async function listPlatformConnectors(
  signal?: AbortSignal,
): Promise<PlatformConnector[]> {
  const data = await platformRequest<unknown>("/connectors", { signal });
  const values = Array.isArray(data) ? data : array(record(data).connectors);
  return values.map(mapPlatformConnector).filter((item) => item.id);
}

export async function createPlatformConnector(
  input: CreatePlatformConnectorInput,
  signal?: AbortSignal,
): Promise<PlatformConnector> {
  const data = await platformRequest<unknown>("/connectors", {
    method: "POST",
    json: {
      name: input.name,
      source: input.source,
      config: input.config,
      ...(input.refreshFrequency === undefined
        ? {}
        : { refresh_freq: input.refreshFrequency }),
      ...(input.pruneFrequency === undefined
        ? {}
        : { prune_freq: input.pruneFrequency }),
      ...(input.timeoutSeconds === undefined
        ? {}
        : { timeout_secs: input.timeoutSeconds }),
    },
    signal,
  });
  return mapPlatformConnector(data);
}

export async function getPlatformConnector(
  connectorId: string,
  signal?: AbortSignal,
): Promise<PlatformConnector> {
  const data = await platformRequest<unknown>(
    `/connectors/${encodeURIComponent(connectorId)}`,
    { signal },
  );
  return mapPlatformConnector(data);
}

export async function updatePlatformConnector(
  connectorId: string,
  input: UpdatePlatformConnectorInput,
  signal?: AbortSignal,
): Promise<PlatformConnector> {
  const data = await platformRequest<unknown>(
    `/connectors/${encodeURIComponent(connectorId)}`,
    {
      method: "PATCH",
      json: {
        ...(input.config === undefined ? {} : { config: input.config }),
        ...(input.refreshFrequency === undefined
          ? {}
          : { refresh_freq: input.refreshFrequency }),
        ...(input.pruneFrequency === undefined
          ? {}
          : { prune_freq: input.pruneFrequency }),
        ...(input.timeoutSeconds === undefined
          ? {}
          : { timeout_secs: input.timeoutSeconds }),
        ...(input.reschedule === undefined
          ? {}
          : { reschedule: input.reschedule }),
        ...(input.status === undefined ? {} : { status: input.status }),
      },
      signal,
    },
  );
  return mapPlatformConnector(data);
}

export async function deletePlatformConnector(
  connectorId: string,
  signal?: AbortSignal,
): Promise<boolean> {
  return platformRequest<boolean>(
    `/connectors/${encodeURIComponent(connectorId)}`,
    { method: "DELETE", signal },
  );
}

export async function testPlatformConnector(
  connectorId: string,
  signal?: AbortSignal,
): Promise<boolean> {
  return platformRequest<boolean>(
    `/connectors/${encodeURIComponent(connectorId)}/test`,
    { method: "POST", signal, timeoutMs: 45_000 },
  );
}

export async function rebuildPlatformConnector(
  connectorId: string,
  datasetId: string,
  signal?: AbortSignal,
): Promise<boolean> {
  return platformRequest<boolean>(
    `/connectors/${encodeURIComponent(connectorId)}/rebuild`,
    { method: "POST", json: { kb_id: datasetId }, signal, timeoutMs: 45_000 },
  );
}

/**
 * A rebuild task is visible to the sync worker only after the connector is
 * linked through the dataset's `connectors` contract. Preserve every existing
 * link, create the selected one when absent, then rebuild it.
 */
export async function linkAndRebuildPlatformConnector(
  connectorId: string,
  datasetId: string,
  signal?: AbortSignal,
): Promise<boolean> {
  const dataset = await getPlatformDataset(datasetId, signal);
  const links = array(dataset.connectors)
    .map(record)
    .map((item) => ({
      id: typeof item.id === "string" ? item.id : "",
      auto_parse:
        typeof item.auto_parse === "string" ? item.auto_parse : "1",
    }))
    .filter((item) => item.id);

  if (!links.some((item) => item.id === connectorId)) {
    await updatePlatformDataset(
      datasetId,
      { connectors: [...links, { id: connectorId, auto_parse: "1" }] },
      signal,
    );
  }

  return rebuildPlatformConnector(connectorId, datasetId, signal);
}

export async function listPlatformConnectorLogs(
  connectorId: string,
  options: { page?: number; pageSize?: number } = {},
  signal?: AbortSignal,
): Promise<PlatformConnectorLogsPage> {
  const data = record(
    await platformRequest<unknown>(
      `/connectors/${encodeURIComponent(connectorId)}/logs`,
      {
        query: {
          page: options.page ?? 1,
          page_size: options.pageSize ?? 20,
        },
        signal,
      },
    ),
  );
  return {
    total: numberValue(data.total),
    logs: array(data.logs).map(mapPlatformConnectorLog),
  };
}

export async function startGoogleConnectorOAuth(
  source: PlatformGoogleConnectorSource,
  credentials: Record<string, unknown>,
  redirectUri: string,
  signal?: AbortSignal,
): Promise<PlatformConnectorOAuthStart> {
  const data = await platformRequest<unknown>("/connectors/google/oauth/web/start", {
    method: "POST",
    query: { type: source },
    json: { credentials, redirect_uri: redirectUri },
    signal,
    timeoutMs: 30_000,
  });
  return mapConnectorOAuthStart(data);
}

export async function startBoxConnectorOAuth(
  clientId: string,
  clientSecret: string,
  redirectUri: string,
  signal?: AbortSignal,
): Promise<PlatformConnectorOAuthStart> {
  const data = await platformRequest<unknown>("/connectors/box/oauth/web/start", {
    method: "POST",
    json: {
      client_id: clientId,
      client_secret: clientSecret,
      redirect_uri: redirectUri,
    },
    signal,
    timeoutMs: 30_000,
  });
  return mapConnectorOAuthStart(data);
}

function callbackEndpoint(source: PlatformConnectorOAuthSource): string {
  return source === "box"
    ? "/connectors/box/oauth/web/callback"
    : `/connectors/${source}/oauth/web/callback`;
}

export async function completeConnectorOAuthCallback(
  source: PlatformConnectorOAuthSource,
  query: {
    state: string;
    code?: string;
    error?: string;
    errorDescription?: string;
  },
  signal?: AbortSignal,
): Promise<{ success: boolean }> {
  const html = await platformRequest<string>(callbackEndpoint(source), {
    pathMode: "root",
    token: null,
    redirectOnUnauthorized: false,
    responseType: "text",
    getRetries: 0,
    query: {
      state: query.state,
      code: query.code,
      error: query.error,
      error_description: query.errorDescription,
    },
    signal,
    timeoutMs: 30_000,
  });
  return {
    success:
      !query.error &&
      !/(["']status["']\s*:\s*["']error["'])/i.test(html) &&
      !/authorization failed/i.test(html),
  };
}

async function pollConnectorOAuthResultOnce(
  source: PlatformConnectorOAuthSource,
  flowId: string,
  signal?: AbortSignal,
): Promise<unknown> {
  const endpoint =
    source === "box"
      ? "/connectors/box/oauth/web/result"
      : "/connectors/google/oauth/web/result";
  return platformRequest<unknown>(endpoint, {
    method: "POST",
    query: source === "box" ? undefined : { type: source },
    json: { flow_id: flowId },
    signal,
  });
}

function wait(delayMs: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const finish = () => signal?.removeEventListener("abort", abort);
    const timer = window.setTimeout(() => {
      finish();
      resolve();
    }, delayMs);
    const abort = () => {
      window.clearTimeout(timer);
      finish();
      reject(signal?.reason ?? new DOMException("Aborted", "AbortError"));
    };
    if (signal?.aborted) abort();
    else signal?.addEventListener("abort", abort, { once: true });
  });
}

async function waitUntilVisible(signal?: AbortSignal): Promise<void> {
  if (typeof document === "undefined" || document.visibilityState !== "hidden")
    return;
  await new Promise<void>((resolve, reject) => {
    const finish = () => {
      document.removeEventListener("visibilitychange", onVisibility);
      signal?.removeEventListener("abort", onAbort);
    };
    const onVisibility = () => {
      if (document.visibilityState === "hidden") return;
      finish();
      resolve();
    };
    const onAbort = () => {
      finish();
      reject(signal?.reason ?? new DOMException("Aborted", "AbortError"));
    };
    document.addEventListener("visibilitychange", onVisibility);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

export async function waitForConnectorOAuthResult(
  source: PlatformConnectorOAuthSource,
  flowId: string,
  options: { signal?: AbortSignal; timeoutMs?: number } = {},
): Promise<unknown> {
  const startedAt = Date.now();
  const timeoutMs = Math.max(1_000, options.timeoutMs ?? 120_000);
  let attempt = 0;
  while (Date.now() - startedAt < timeoutMs) {
    await waitUntilVisible(options.signal);
    try {
      const data = record(
        await pollConnectorOAuthResultOnce(source, flowId, options.signal),
      );
      return data.credentials;
    } catch (error) {
      if (
        !isPlatformApiError(error) ||
        Number(error.code) !== OAUTH_PENDING_CODE
      ) {
        throw error;
      }
    }
    await wait(
      Math.min(OAUTH_POLL_INTERVAL_MAX_MS, 600 * 2 ** attempt),
      options.signal,
    );
    attempt += 1;
  }
  throw new PlatformApiError("OAuth sonucu beklenirken zaman aşımı oluştu.", {
    httpStatus: null,
    code: "OAUTH_RESULT_TIMEOUT",
    endpoint: "/connectors/*/oauth/web/result",
  });
}
