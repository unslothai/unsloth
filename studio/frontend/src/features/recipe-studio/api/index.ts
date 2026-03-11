// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { authFetch } from "@/features/auth";

const DEFAULT_BASE = "/api/data-recipe";

export const DATA_DESIGNER_API_BASE =
  import.meta.env.VITE_DATA_DESIGNER_API ?? DEFAULT_BASE;

export type JobCreateResponse = {
  // biome-ignore lint/style/useNamingConvention: api schema
  job_id: string;
};

export type JobStatusResponse = {
  // biome-ignore lint/style/useNamingConvention: api schema
  job_id: string;
  status: string;
  stage?: string | null;
  // biome-ignore lint/style/useNamingConvention: api schema
  current_column?: string | null;
  // biome-ignore lint/style/useNamingConvention: api schema
  completed_columns?: string[] | null;
  batch?: {
    idx?: number | null;
    total?: number | null;
  };
  progress?: {
    done?: number | null;
    total?: number | null;
    percent?: number | null;
    // biome-ignore lint/style/useNamingConvention: api schema
    eta_sec?: number | null;
    rate?: number | null;
    ok?: number | null;
    failed?: number | null;
  };
  // biome-ignore lint/style/useNamingConvention: api schema
  column_progress?: {
    done?: number | null;
    total?: number | null;
    percent?: number | null;
    // biome-ignore lint/style/useNamingConvention: api schema
    eta_sec?: number | null;
    rate?: number | null;
    ok?: number | null;
    failed?: number | null;
  };
  // biome-ignore lint/style/useNamingConvention: api schema
  model_usage?: Record<string, unknown>;
  rows?: number | null;
  cols?: number | null;
  error?: string | null;
  // biome-ignore lint/style/useNamingConvention: api schema
  has_analysis?: boolean;
  // biome-ignore lint/style/useNamingConvention: api schema
  dataset_rows?: number | null;
  // biome-ignore lint/style/useNamingConvention: api schema
  artifact_path?: string | null;
  // biome-ignore lint/style/useNamingConvention: api schema
  started_at?: number | null;
  // biome-ignore lint/style/useNamingConvention: api schema
  finished_at?: number | null;
};

export type JobDatasetResponse = {
  dataset?: unknown[];
  total?: number;
  limit?: number;
  offset?: number;
};

export type JobEvent = {
  event: string;
  id: number | null;
  payload: Record<string, unknown>;
};

export type SeedInspectRequest = {
  // biome-ignore lint/style/useNamingConvention: api schema
  dataset_name: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  hf_token?: string;
  subset?: string;
  split?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  preview_size?: number;
};

export type SeedInspectUploadRequest = {
  filename: string;
  // base64 payload without data URL prefix
  content_base64: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  preview_size?: number;
  // biome-ignore lint/style/useNamingConvention: api schema
  seed_source_type?: "local" | "unstructured";
  // biome-ignore lint/style/useNamingConvention: api schema
  unstructured_chunk_size?: number;
  // biome-ignore lint/style/useNamingConvention: api schema
  unstructured_chunk_overlap?: number;
};

export type SeedInspectResponse = {
  // biome-ignore lint/style/useNamingConvention: api schema
  dataset_name: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  resolved_path: string;
  columns: string[];
  // biome-ignore lint/style/useNamingConvention: api schema
  preview_rows: Record<string, unknown>[];
  split?: string | null;
  subset?: string | null;
};

export type ValidateError = {
  message: string;
  path?: string | null;
  code?: string | null;
};

export type ValidateResponse = {
  valid: boolean;
  errors: ValidateError[];
  // biome-ignore lint/style/useNamingConvention: api schema
  raw_detail?: string | null;
};

export type McpToolsListRequest = {
  // biome-ignore lint/style/useNamingConvention: api schema
  mcp_providers: Record<string, unknown>[];
  // biome-ignore lint/style/useNamingConvention: api schema
  timeout_sec?: number;
};

export type McpToolsProviderResult = {
  name: string;
  tools: string[];
  error?: string | null;
};

export type McpToolsListResponse = {
  providers: McpToolsProviderResult[];
  // biome-ignore lint/style/useNamingConvention: api schema
  duplicate_tools: Record<string, string[]>;
};

async function parseErrorResponse(response: Response): Promise<string> {
  const text = (await response.text()).trim();
  if (!text) {
    return "Request failed.";
  }
  try {
    const parsed = JSON.parse(text) as {
      detail?: string;
      message?: string;
      // biome-ignore lint/style/useNamingConvention: api schema
      raw_detail?: string;
    };
    return (
      parsed.detail ??
      parsed.message ??
      parsed.raw_detail ??
      text
    );
  } catch {
    return text;
  }
}

async function postJson<T>(path: string, payload: unknown): Promise<T> {
  const response = await authFetch(`${DATA_DESIGNER_API_BASE}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(await parseErrorResponse(response));
  }

  return response.json();
}

async function getJson<T>(path: string): Promise<T> {
  const response = await authFetch(`${DATA_DESIGNER_API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(await parseErrorResponse(response));
  }
  return response.json();
}

function parseJobEvent(rawEvent: string): JobEvent | null {
  const lines = rawEvent.split(/\r?\n/);
  let eventName = "message";
  let id: number | null = null;
  const dataLines: string[] = [];

  for (const line of lines) {
    if (!line) {
      continue;
    }
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim() || "message";
      continue;
    }
    if (line.startsWith("id:")) {
      const value = Number(line.slice(3).trim());
      id = Number.isFinite(value) ? value : null;
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (dataLines.length === 0) {
    return null;
  }
  let payload: Record<string, unknown>;
  try {
    payload = JSON.parse(dataLines.join("\n")) as Record<string, unknown>;
  } catch {
    return null;
  }
  return {
    event: eventName,
    id,
    payload,
  };
}

export async function validateRecipe(
  payload: unknown,
): Promise<ValidateResponse> {
  return postJson<ValidateResponse>("/validate", payload);
}

export async function createRecipeJob(payload: unknown): Promise<JobCreateResponse> {
  return postJson<JobCreateResponse>("/jobs", payload);
}

export async function getRecipeJobStatus(jobId: string): Promise<JobStatusResponse> {
  return getJson<JobStatusResponse>(`/jobs/${jobId}/status`);
}

export async function getRecipeJobAnalysis(
  jobId: string,
): Promise<Record<string, unknown>> {
  return getJson<Record<string, unknown>>(`/jobs/${jobId}/analysis`);
}

export async function getRecipeJobDataset(
  jobId: string,
  options?: {
    limit?: number;
    offset?: number;
  },
): Promise<JobDatasetResponse> {
  const limit = options?.limit ?? 20;
  const offset = options?.offset ?? 0;
  return getJson<JobDatasetResponse>(
    `/jobs/${jobId}/dataset?limit=${limit}&offset=${offset}`,
  );
}

export async function cancelRecipeJob(jobId: string): Promise<JobStatusResponse> {
  return postJson<JobStatusResponse>(`/jobs/${jobId}/cancel`, {});
}

export async function inspectSeedDataset(
  payload: SeedInspectRequest,
): Promise<SeedInspectResponse> {
  return postJson<SeedInspectResponse>("/seed/inspect", payload);
}

export async function inspectSeedUpload(
  payload: SeedInspectUploadRequest,
): Promise<SeedInspectResponse> {
  return postJson<SeedInspectResponse>("/seed/inspect-upload", payload);
}

export async function listMcpTools(
  payload: McpToolsListRequest,
): Promise<McpToolsListResponse> {
  return postJson<McpToolsListResponse>("/mcp/tools", payload);
}

export async function streamRecipeJobEvents(options: {
  jobId: string;
  signal: AbortSignal;
  lastEventId?: number | null;
  onOpen?: () => void;
  onEvent: (event: JobEvent) => void;
}): Promise<void> {
  const headers = new Headers();
  let query = "";
  if (typeof options.lastEventId === "number") {
    headers.set("Last-Event-ID", String(options.lastEventId));
    query = `?after=${options.lastEventId}`;
  }

  const response = await authFetch(
    `${DATA_DESIGNER_API_BASE}/jobs/${options.jobId}/events${query}`,
    {
      method: "GET",
      headers,
      signal: options.signal,
    },
  );
  if (!response.ok) {
    throw new Error(await parseErrorResponse(response));
  }
  if (!response.body) {
    throw new Error("Job stream unavailable.");
  }

  options.onOpen?.();

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    let separatorIndex = buffer.search(/\r?\n\r?\n/);
    while (separatorIndex >= 0) {
      const rawEvent = buffer.slice(0, separatorIndex);
      const separatorLength = buffer[separatorIndex] === "\r" ? 4 : 2;
      buffer = buffer.slice(separatorIndex + separatorLength);

      if (rawEvent.startsWith("retry:")) {
        separatorIndex = buffer.search(/\r?\n\r?\n/);
        continue;
      }

      const parsed = parseJobEvent(rawEvent);
      if (parsed) {
        options.onEvent(parsed);
      }
      separatorIndex = buffer.search(/\r?\n\r?\n/);
    }
  }
}

// NOTE: preview endpoints removed from harness.
