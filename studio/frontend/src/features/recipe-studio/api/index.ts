const DEFAULT_BASE = "/api/data-recipe";

export const DATA_DESIGNER_API_BASE =
  import.meta.env.VITE_DATA_DESIGNER_API ?? DEFAULT_BASE;

export type PreviewResponse = {
  dataset?: unknown[];
  // biome-ignore lint/style/useNamingConvention: api schema
  processor_artifacts?: Record<string, unknown>;
  analysis?: Record<string, unknown>;
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

export async function previewRecipe(payload: unknown): Promise<PreviewResponse> {
  return postJson<PreviewResponse>("/preview", payload);
}

export async function validateRecipe(
  payload: unknown,
): Promise<ValidateResponse> {
  return postJson<ValidateResponse>("/validate", payload);
}

// NOTE: tools + seed inspect/preview endpoints removed from harness.
