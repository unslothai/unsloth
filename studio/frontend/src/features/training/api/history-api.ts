


import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type {
  TrainingRunDeleteResponse,
  TrainingRunDetailResponse,
  TrainingRunListResponse,
  TrainingRunSummary,
} from "../types/history";

const readError = (r: Response): Promise<string> => readFastApiError(r);

export class HistoryRequestError extends Error {
  readonly status: number | null;
  readonly errorCode: string | null;

  constructor(
    message: string,
    status: number | null,
    errorCode: string | null = null,
  ) {
    super(message);
    this.name = "HistoryRequestError";
    this.status = status;
    this.errorCode = errorCode;
  }
}

async function readHistoryRequestError(
  response: Response,
): Promise<HistoryRequestError> {
  const fallbackResponse = response.clone();
  try {
    const payload = (await response.json()) as { detail?: unknown };
    const detail = payload.detail;
    if (detail && typeof detail === "object" && !Array.isArray(detail)) {
      const structured = detail as { code?: unknown; message?: unknown };
      if (typeof structured.message === "string" && structured.message) {
        return new HistoryRequestError(
          structured.message,
          response.status,
          typeof structured.code === "string" ? structured.code : null,
        );
      }
    }
  } catch {
    return new HistoryRequestError(
      await readError(fallbackResponse),
      response.status,
    );
  }
  return new HistoryRequestError(
    await readError(fallbackResponse),
    response.status,
  );
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw await readHistoryRequestError(response);
  }
  return (await response.json()) as T;
}

export async function listTrainingRuns(
  limit = 50,
  offset = 0,
  signal?: AbortSignal,
): Promise<TrainingRunListResponse> {
  const response = await authFetch(
    `/api/train/runs?limit=${limit}&offset=${offset}`,
    { signal },
  );
  return parseJson<TrainingRunListResponse>(response);
}

export async function getTrainingRun(
  runId: string,
  signal?: AbortSignal,
): Promise<TrainingRunDetailResponse> {
  const response = await authFetch(
    `/api/train/runs/${encodeURIComponent(runId)}`,
    { signal },
  );
  return parseJson<TrainingRunDetailResponse>(response);
}

export async function deleteTrainingRun(
  runId: string,
  options?: { deleteArtifacts?: boolean; signal?: AbortSignal },
): Promise<TrainingRunDeleteResponse> {
  const query = options?.deleteArtifacts ? "?delete_artifacts=true" : "";
  const response = await authFetch(
    `/api/train/runs/${encodeURIComponent(runId)}${query}`,
    { method: "DELETE", signal: options?.signal },
  );
  return parseJson<TrainingRunDeleteResponse>(response);
}

export async function renameTrainingRun(
  runId: string,
  displayName: string | null,
  signal?: AbortSignal,
): Promise<TrainingRunSummary> {
  const response = await authFetch(
    `/api/train/runs/${encodeURIComponent(runId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_name: displayName }),
      signal,
    },
  );
  return parseJson<TrainingRunSummary>(response);
}
