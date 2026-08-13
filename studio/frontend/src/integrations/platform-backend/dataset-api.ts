import { platformRequest } from "./client";
import { unwrapPlatformEnvelope } from "./envelope";
import type {
  PlatformDatasetDto,
  PlatformDatasetListQuery,
  PlatformDatasetListResult,
  PlatformDatasetUpdateRequest,
  PlatformDatasetWriteRequest,
} from "./dataset-types";
import type { PlatformEnvelope } from "./types";

interface DatasetListEnvelope extends PlatformEnvelope<PlatformDatasetDto[]> {
  total_datasets?: unknown;
}

function positiveInteger(value: unknown, fallback: number): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : fallback;
}

export async function listPlatformDatasets(
  query: PlatformDatasetListQuery,
  signal?: AbortSignal,
): Promise<PlatformDatasetListResult> {
  const endpoint = "/datasets";
  const envelope = await platformRequest<DatasetListEnvelope>(endpoint, {
    query: {
      page: query.page,
      page_size: query.pageSize,
      name: query.name?.trim() || undefined,
      orderby: query.orderby ?? "update_time",
      desc: query.desc ?? true,
    },
    responseType: "json",
    signal,
  });
  const items = unwrapPlatformEnvelope<PlatformDatasetDto[]>(envelope, {
    endpoint,
    httpStatus: 200,
  });
  const safeItems = Array.isArray(items) ? items : [];
  return {
    items: safeItems,
    total: positiveInteger(envelope.total_datasets, safeItems.length),
  };
}

export function getPlatformDataset(
  datasetId: string,
  signal?: AbortSignal,
): Promise<PlatformDatasetDto> {
  return platformRequest(`/datasets/${encodeURIComponent(datasetId)}`, {
    signal,
  });
}

export function createPlatformDataset(
  payload: PlatformDatasetWriteRequest,
  signal?: AbortSignal,
): Promise<PlatformDatasetDto> {
  return platformRequest("/datasets", {
    method: "POST",
    json: payload,
    signal,
  });
}

export function updatePlatformDataset(
  datasetId: string,
  payload: PlatformDatasetUpdateRequest,
  signal?: AbortSignal,
): Promise<PlatformDatasetDto> {
  return platformRequest(`/datasets/${encodeURIComponent(datasetId)}`, {
    method: "PUT",
    json: payload,
    signal,
  });
}

export function deletePlatformDatasets(
  datasetIds: string[],
  signal?: AbortSignal,
): Promise<void> {
  return platformRequest("/datasets", {
    method: "DELETE",
    json: { ids: datasetIds },
    signal,
  });
}
