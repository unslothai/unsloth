const DEFAULT_BASE = "";

export const DATA_DESIGNER_API_BASE =
  import.meta.env.VITE_DATA_DESIGNER_API ?? DEFAULT_BASE;

export type PreviewResponse = {
  dataset?: unknown[];
  // biome-ignore lint/style/useNamingConvention: api schema
  processor_artifacts?: Record<string, unknown>;
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

export type ToolsResponse = {
  // biome-ignore lint/style/useNamingConvention: api schema
  tools_by_provider: Record<string, string[]>;
  tools: string[];
};

export type SeedInspectResponse = {
  // biome-ignore lint/style/useNamingConvention: api schema
  repo_id: string;
  splits: string[];
  // biome-ignore lint/style/useNamingConvention: api schema
  globs_by_split: Record<string, string>;
  columns: string[];
};

export type SeedPreviewResponse = {
  rows: Record<string, unknown>[];
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

export async function previewRecipe(payload: unknown): Promise<PreviewResponse> {
  return postJson<PreviewResponse>("/preview", payload);
}

export async function validateRecipe(
  payload: unknown,
): Promise<ValidateResponse> {
  return postJson<ValidateResponse>("/validate", payload);
}

export async function listRecipeTools(payload: unknown): Promise<ToolsResponse> {
  return postJson<ToolsResponse>("/tools", payload);
}

export async function inspectSeedDataset(payload: unknown): Promise<SeedInspectResponse> {
  return postJson<SeedInspectResponse>("/seed/inspect", payload);
}

export async function previewSeedDataset(payload: unknown): Promise<SeedPreviewResponse> {
  return postJson<SeedPreviewResponse>("/seed/preview", payload);
}
