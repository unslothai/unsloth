const DEFAULT_BASE = "/api/data-recipe";

export const DATA_DESIGNER_API_BASE =
  import.meta.env.VITE_DATA_DESIGNER_API ?? DEFAULT_BASE;

export type PreviewResponse = {
  dataset?: unknown[];
  // biome-ignore lint/style/useNamingConvention: api schema
  processor_artifacts?: Record<string, unknown>;
  analysis?: Record<string, unknown>;
};

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
};

export type JobEvent = {
  event: string;
  id: number | null;
  payload: Record<string, unknown>;
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
  const response = await fetch(`${DATA_DESIGNER_API_BASE}${path}`, {
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
  const response = await fetch(`${DATA_DESIGNER_API_BASE}${path}`);
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
  const payload = JSON.parse(dataLines.join("\n")) as Record<string, unknown>;
  return {
    event: eventName,
    id,
    payload,
  };
}

export async function previewRecipe(payload: unknown): Promise<PreviewResponse> {
  return postJson<PreviewResponse>("/preview", payload);
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
  limit = 20,
): Promise<JobDatasetResponse> {
  return getJson<JobDatasetResponse>(`/jobs/${jobId}/dataset?limit=${limit}`);
}

export async function cancelRecipeJob(jobId: string): Promise<JobStatusResponse> {
  return postJson<JobStatusResponse>(`/jobs/${jobId}/cancel`, {});
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

  const response = await fetch(
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

// NOTE: tools + seed inspect/preview endpoints removed from harness.
