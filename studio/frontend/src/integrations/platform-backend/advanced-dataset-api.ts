import { platformRequest } from "./client";
import type {
  PlatformArtifactGraph,
  PlatformArtifactList,
  PlatformArtifactPage,
  PlatformArtifactProbe,
  PlatformDatasetSkillPage,
  PlatformDatasetSearchResponse,
  PlatformDocumentMetadataBatchRequest,
  PlatformDocumentMetadataBatchResult,
  PlatformDocumentStatusBatchResult,
  PlatformEmbeddingCheckResponse,
  PlatformEmbeddingRunResult,
  PlatformGraphData,
  PlatformIndexStartResult,
  PlatformIndexTask,
  PlatformIndexType,
  PlatformIngestionLog,
  PlatformIngestionLogList,
  PlatformIngestionSummary,
  PlatformMetadataConfig,
  PlatformMetadataSummary,
  PlatformSkillInfo,
  PlatformSkillSearchConfig,
  PlatformSkillSearchConfigRequest,
  PlatformSkillSearchResponse,
  PlatformSkillSpace,
  PlatformSkillSpaceDeleteResult,
  PlatformSkillSpaceUpdateRequest,
  PlatformSkillSpaceWriteRequest,
  PlatformSkillTreeNode,
  PlatformTagCount,
} from "./advanced-dataset-types";

const encode = encodeURIComponent;
const encodePath = (value: string) =>
  value.split("/").map(encodeURIComponent).join("/");

export function getDatasetMetadataConfig(
  datasetId: string,
  signal?: AbortSignal,
): Promise<PlatformMetadataConfig> {
  return platformRequest(`/datasets/${encode(datasetId)}/metadata/config`, {
    signal,
  });
}

export function updateDatasetMetadataConfig(
  datasetId: string,
  payload: PlatformMetadataConfig,
  signal?: AbortSignal,
): Promise<PlatformMetadataConfig> {
  return platformRequest(`/datasets/${encode(datasetId)}/metadata/config`, {
    method: "PUT",
    json: payload,
    signal,
  });
}

export function getFlattenedDatasetMetadata(
  datasetIds: string[],
  signal?: AbortSignal,
): Promise<Record<string, unknown>> {
  return platformRequest("/datasets/metadata/flattened", {
    query: { dataset_ids: datasetIds.join(",") },
    signal,
  });
}

export function getDatasetMetadataSummary(
  datasetId: string,
  documentIds: string[] = [],
  signal?: AbortSignal,
): Promise<PlatformMetadataSummary> {
  return platformRequest(`/datasets/${encode(datasetId)}/metadata/summary`, {
    query: { doc_ids: documentIds.length ? documentIds.join(",") : undefined },
    signal,
  });
}

/** API-only compatibility alias. Product UI uses getDatasetMetadataSummary. */
export function getLegacyDocumentMetadataSummaryCompatibility(
  datasetId: string,
  documentIds: string[] = [],
  signal?: AbortSignal,
): Promise<PlatformMetadataSummary> {
  return platformRequest("/document/metadata/summary", {
    method: "POST",
    json: { kb_id: datasetId, doc_ids: documentIds },
    signal,
  });
}

/** API-only compatibility alias. Product UI uses dataset-scoped mutations. */
export function setLegacyDocumentMetadataCompatibility(
  documentId: string,
  metadata: Record<string, string | number | Array<string | number>>,
  signal?: AbortSignal,
): Promise<boolean> {
  return platformRequest("/document/set_meta", {
    method: "POST",
    json: { doc_id: documentId, meta: JSON.stringify(metadata) },
    signal,
  });
}

export function updateDocumentMetadataConfig(
  datasetId: string,
  documentId: string,
  payload: Record<string, unknown>,
  signal?: AbortSignal,
): Promise<Record<string, unknown>> {
  return platformRequest(
    `/datasets/${encode(datasetId)}/documents/${encode(documentId)}/metadata/config`,
    { method: "PUT", json: payload, signal },
  );
}

export function batchUpdateDatasetMetadata(
  datasetId: string,
  payload: PlatformDocumentMetadataBatchRequest,
  signal?: AbortSignal,
): Promise<PlatformDocumentMetadataBatchResult> {
  return platformRequest(`/datasets/${encode(datasetId)}/metadata/update`, {
    method: "POST",
    json: payload,
    signal,
  });
}

export function patchDatasetDocumentMetadata(
  datasetId: string,
  payload: PlatformDocumentMetadataBatchRequest,
  signal?: AbortSignal,
): Promise<PlatformDocumentMetadataBatchResult> {
  return platformRequest(`/datasets/${encode(datasetId)}/documents/metadatas`, {
    method: "PATCH",
    json: payload,
    signal,
  });
}

export function batchUpdateDatasetDocumentStatus(
  datasetId: string,
  documentIds: string[],
  status: 0 | 1,
  signal?: AbortSignal,
): Promise<PlatformDocumentStatusBatchResult> {
  return platformRequest(
    `/datasets/${encode(datasetId)}/documents/batch-update-status`,
    { method: "POST", json: { doc_ids: documentIds, status }, signal },
  );
}

export function listDatasetTags(
  datasetId: string,
  signal?: AbortSignal,
): Promise<PlatformTagCount[]> {
  return platformRequest(`/datasets/${encode(datasetId)}/tags`, { signal });
}

export function aggregateDatasetTags(
  datasetIds: string[],
  signal?: AbortSignal,
): Promise<PlatformTagCount[]> {
  return platformRequest("/datasets/tags/aggregation", {
    query: { dataset_ids: datasetIds.join(",") },
    signal,
  });
}

export function renameDatasetTag(
  datasetId: string,
  fromTag: string,
  toTag: string,
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest(`/datasets/${encode(datasetId)}/tags`, {
    method: "PUT",
    json: { from_tag: fromTag, to_tag: toTag },
    signal,
  });
}

export function removeDatasetTags(
  datasetId: string,
  tags: string[],
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest(`/datasets/${encode(datasetId)}/tags`, {
    method: "DELETE",
    json: { tags },
    signal,
  });
}

export function getDatasetGraph(
  datasetId: string,
  signal?: AbortSignal,
): Promise<PlatformGraphData> {
  return platformRequest(`/datasets/${encode(datasetId)}/graph`, { signal });
}

export function searchDatasets(
  datasetIds: string[],
  question: string,
  signal?: AbortSignal,
): Promise<PlatformDatasetSearchResponse> {
  return platformRequest("/datasets/search", {
    method: "POST",
    json: {
      dataset_ids: datasetIds,
      question,
      page: 1,
      size: 30,
      use_kg: false,
      top_k: 1024,
      force_refresh: false,
    },
    signal,
  });
}

export function hasDatasetArtifacts(
  datasetId: string,
  signal?: AbortSignal,
): Promise<PlatformArtifactProbe> {
  return platformRequest(`/datasets/${encode(datasetId)}/any_artifact`, {
    signal,
  });
}

export function listDatasetArtifacts(
  datasetId: string,
  query: { page: number; pageSize: number; pageType?: string },
  signal?: AbortSignal,
): Promise<PlatformArtifactList> {
  return platformRequest(`/datasets/${encode(datasetId)}/artifacts`, {
    query: {
      page: query.page,
      page_size: query.pageSize,
      page_type: query.pageType,
    },
    signal,
  });
}

export function getDatasetArtifactGraph(
  datasetId: string,
  node?: string,
  signal?: AbortSignal,
): Promise<PlatformArtifactGraph> {
  return platformRequest(`/datasets/${encode(datasetId)}/artifacts/graph`, {
    query: { node },
    signal,
  });
}

export function clearDatasetArtifacts(
  datasetId: string,
  signal?: AbortSignal,
): Promise<Record<string, unknown>> {
  return platformRequest(`/datasets/${encode(datasetId)}/artifacts`, {
    method: "DELETE",
    signal,
  });
}

export function getDatasetArtifactPage(
  datasetId: string,
  pageType: string,
  slug: string,
  signal?: AbortSignal,
): Promise<PlatformArtifactPage | null> {
  return platformRequest(
    `/datasets/${encode(datasetId)}/artifacts/${encode(pageType)}/${encodePath(slug)}`,
    { signal },
  );
}

export function updateDatasetArtifactPage(
  datasetId: string,
  pageType: string,
  slug: string,
  payload: { content_md: string; title?: string; comments?: string },
  signal?: AbortSignal,
): Promise<PlatformArtifactPage> {
  return platformRequest(
    `/datasets/${encode(datasetId)}/artifacts/${encode(pageType)}/${encodePath(slug)}`,
    { method: "PUT", json: payload, signal },
  );
}

export function startDatasetIndex(
  datasetId: string,
  type: PlatformIndexType,
  signal?: AbortSignal,
): Promise<PlatformIndexStartResult> {
  return platformRequest(`/datasets/${encode(datasetId)}/index`, {
    method: "POST",
    query: { type },
    signal,
  });
}

export function getDatasetIndexStatus(
  datasetId: string,
  type: PlatformIndexType,
  signal?: AbortSignal,
): Promise<PlatformIndexTask | null> {
  return platformRequest(`/datasets/${encode(datasetId)}/index`, {
    query: { type },
    signal,
  });
}

export function deleteDatasetIndex(
  datasetId: string,
  type: PlatformIndexType,
  wipe: boolean,
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest(`/datasets/${encode(datasetId)}/${type}`, {
    method: "DELETE",
    query: { wipe },
    signal,
  });
}

export function deleteDatasetIndexByQuery(
  datasetId: string,
  type: PlatformIndexType,
  wipe: boolean,
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest(`/datasets/${encode(datasetId)}/index`, {
    method: "DELETE",
    query: { type, wipe },
    signal,
  });
}

export function runDatasetEmbedding(
  datasetId: string,
  signal?: AbortSignal,
): Promise<PlatformEmbeddingRunResult> {
  return platformRequest(`/datasets/${encode(datasetId)}/embedding`, {
    method: "POST",
    signal,
  });
}

export function checkDatasetEmbedding(
  datasetId: string,
  embeddingId: string,
  checkNum?: number,
  signal?: AbortSignal,
): Promise<PlatformEmbeddingCheckResponse> {
  return platformRequest(`/datasets/${encode(datasetId)}/embedding/check`, {
    method: "POST",
    json: { embd_id: embeddingId, check_num: checkNum },
    signal,
  });
}

export function getDatasetIngestionSummary(
  datasetId: string,
  signal?: AbortSignal,
): Promise<PlatformIngestionSummary> {
  return platformRequest(`/datasets/${encode(datasetId)}/ingestions/summary`, {
    signal,
  });
}

export function listDatasetIngestionLogs(
  datasetId: string,
  query: {
    page: number;
    pageSize: number;
    orderby?: string;
    desc?: boolean;
    operationStatus?: string[];
    createDateFrom?: string;
    createDateTo?: string;
    logType?: "dataset" | "file";
    keywords?: string;
  },
  signal?: AbortSignal,
): Promise<PlatformIngestionLogList> {
  return platformRequest(`/datasets/${encode(datasetId)}/ingestions`, {
    query: {
      page: query.page,
      page_size: query.pageSize,
      orderby: query.orderby ?? "create_time",
      desc: query.desc ?? true,
      operation_status: query.operationStatus,
      create_date_from: query.createDateFrom,
      create_date_to: query.createDateTo,
      log_type: query.logType ?? "dataset",
      keywords: query.keywords,
    },
    signal,
  });
}

export function getDatasetIngestionLog(
  datasetId: string,
  logId: string,
  signal?: AbortSignal,
): Promise<PlatformIngestionLog> {
  return platformRequest(
    `/datasets/${encode(datasetId)}/ingestions/${encode(logId)}`,
    { signal },
  );
}

export function hasDatasetSkills(
  datasetId: string,
  signal?: AbortSignal,
): Promise<PlatformArtifactProbe> {
  return platformRequest(`/datasets/${encode(datasetId)}/any_skill`, {
    signal,
  });
}

export function getDatasetSkillTree(
  datasetId: string,
  signal?: AbortSignal,
): Promise<PlatformSkillTreeNode | null> {
  return platformRequest(`/datasets/${encode(datasetId)}/skills`, { signal });
}

export function getDatasetSkillPage(
  datasetId: string,
  skillKeyword: string,
  signal?: AbortSignal,
): Promise<PlatformDatasetSkillPage | null> {
  return platformRequest(
    `/datasets/${encode(datasetId)}/skills/${encodePath(skillKeyword)}`,
    { signal },
  );
}

export function getGlobalSkillSearchConfig(
  spaceId: string,
  embeddingId = "",
  signal?: AbortSignal,
): Promise<PlatformSkillSearchConfig> {
  return platformRequest("/skills/config", {
    query: { space_id: spaceId, embd_id: embeddingId || undefined },
    signal,
  });
}

export function updateGlobalSkillSearchConfig(
  payload: PlatformSkillSearchConfigRequest,
  signal?: AbortSignal,
): Promise<PlatformSkillSearchConfig> {
  return platformRequest("/skills/config", {
    method: "POST",
    json: payload,
    signal,
  });
}

export function searchGlobalSkills(
  payload: {
    space_id: string;
    query: string;
    page: number;
    page_size: number;
    sort_by?: "name" | "update_time" | "create_time" | "relevance";
    sort_order?: "asc" | "desc";
  },
  signal?: AbortSignal,
): Promise<PlatformSkillSearchResponse> {
  return platformRequest("/skills/search", {
    method: "POST",
    json: payload,
    signal,
  });
}

export function indexGlobalSkills(
  spaceId: string,
  skills: PlatformSkillInfo[],
  embeddingId?: string,
  signal?: AbortSignal,
): Promise<{ indexed_count: number }> {
  return platformRequest("/skills/index", {
    method: "POST",
    json: { skills, space_id: spaceId, embd_id: embeddingId },
    signal,
  });
}

export function reindexGlobalSkills(
  spaceId: string,
  embeddingId?: string,
  signal?: AbortSignal,
): Promise<Record<string, unknown>> {
  return platformRequest("/skills/reindex", {
    method: "POST",
    json: { space_id: spaceId, embd_id: embeddingId },
    signal,
  });
}

export function deleteGlobalSkillIndex(
  spaceId: string,
  skillId: string,
  signal?: AbortSignal,
): Promise<boolean> {
  return platformRequest("/skills/index", {
    method: "DELETE",
    query: { space_id: spaceId, skill_id: skillId },
    signal,
  });
}

export function listGlobalSkillSpaces(signal?: AbortSignal): Promise<{
  spaces: PlatformSkillSpace[];
  total: number;
}> {
  return platformRequest("/skills/spaces", { signal });
}

export function createGlobalSkillSpace(
  payload: PlatformSkillSpaceWriteRequest,
  signal?: AbortSignal,
): Promise<PlatformSkillSpace> {
  return platformRequest("/skills/spaces", {
    method: "POST",
    json: payload,
    signal,
  });
}

export function getGlobalSkillSpace(
  spaceId: string,
  signal?: AbortSignal,
): Promise<PlatformSkillSpace> {
  return platformRequest(`/skills/spaces/${encode(spaceId)}`, { signal });
}

export function updateGlobalSkillSpace(
  spaceId: string,
  payload: PlatformSkillSpaceUpdateRequest,
  signal?: AbortSignal,
): Promise<PlatformSkillSpace> {
  return platformRequest(`/skills/spaces/${encode(spaceId)}`, {
    method: "PUT",
    json: payload,
    signal,
  });
}

export function deleteGlobalSkillSpace(
  spaceId: string,
  signal?: AbortSignal,
): Promise<PlatformSkillSpaceDeleteResult> {
  return platformRequest(`/skills/spaces/${encode(spaceId)}`, {
    method: "DELETE",
    signal,
  });
}

export function getGlobalSkillSpaceByFolder(
  folderId: string,
  signal?: AbortSignal,
): Promise<PlatformSkillSpace> {
  return platformRequest("/skills/space/by-folder", {
    query: { folder_id: folderId },
    signal,
  });
}
