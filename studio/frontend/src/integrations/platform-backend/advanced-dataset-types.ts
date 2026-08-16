export type PlatformIndexType = "graph" | "raptor" | "mindmap";

export interface PlatformMetadataField {
  key: string;
  type: string;
  description: string | null;
  enum: string[];
}

export interface PlatformMetadataConfig {
  metadata: PlatformMetadataField[];
  built_in_metadata: PlatformMetadataField[];
}

export interface PlatformMetadataSummary {
  summary: Record<string, unknown>;
}

export interface PlatformDocumentMetadataSelector {
  document_ids: string[];
  metadata_condition: Record<string, unknown>;
}

export interface PlatformDocumentMetadataUpdate {
  key: string;
  value: unknown;
  match?: unknown;
  valueType?: string;
}

export interface PlatformDocumentMetadataDelete {
  key: string;
  value?: unknown;
}

export interface PlatformDocumentMetadataBatchRequest {
  selector: PlatformDocumentMetadataSelector;
  updates: PlatformDocumentMetadataUpdate[];
  deletes: PlatformDocumentMetadataDelete[];
}

export interface PlatformDocumentMetadataBatchResult {
  updated: number;
  matched_docs: number;
}

export interface PlatformDocumentStatusBatchResult {
  [key: string]: unknown;
}

export interface PlatformTagCount {
  key: string;
  count: number;
}

export interface PlatformGraphData {
  graph?: unknown;
  mind_map?: unknown;
  [key: string]: unknown;
}

export interface PlatformDatasetSearchResponse {
  chunks: Array<Record<string, unknown>>;
  doc_aggs: Array<Record<string, unknown>>;
  labels?: Record<string, number>;
  total: number;
}

export interface PlatformArtifactProbe {
  has: boolean;
}

export interface PlatformArtifactListItem {
  slug: string;
  title: string;
  page_type: string;
}

export interface PlatformArtifactList {
  total: number;
  items: PlatformArtifactListItem[];
}

export interface PlatformArtifactPage {
  slug?: string;
  title?: string;
  page_type?: string;
  content_md?: string;
  content_md_rendered?: string;
  [key: string]: unknown;
}

export interface PlatformArtifactGraph {
  entities: Array<Record<string, unknown>>;
  relations: Array<Record<string, unknown>>;
}

export interface PlatformIndexStartResult {
  task_id: string;
}

export interface PlatformIndexTask {
  id: string;
  doc_id: string;
  task_type: string;
  progress: number;
  progress_msg?: string;
  begin_at?: string;
  process_duration?: number;
  [key: string]: unknown;
}

export interface PlatformEmbeddingRunResult {
  scheduled_count: number;
}

export interface PlatformEmbeddingCheckResult {
  chunk_id: string;
  doc_id?: string;
  doc_name?: string;
  vector_field?: string;
  vector_dim?: number;
  cos_sim?: number;
  reason?: string;
}

export interface PlatformEmbeddingCheckResponse {
  summary: {
    kb_id: string;
    model: string;
    sampled: number;
    valid: number;
    avg_cos_sim: number;
    min_cos_sim: number;
    max_cos_sim: number;
    match_mode: string;
  };
  results: PlatformEmbeddingCheckResult[];
}

export interface PlatformIngestionSummary {
  doc_num: number;
  chunk_num: number;
  token_num: number;
  status: unknown;
}

export interface PlatformIngestionLog {
  id: string;
  task_type?: string;
  operation_status?: string;
  progress?: number;
  progress_msg?: string;
  create_date?: string;
  update_date?: string;
  document_id?: string;
  parser_id?: string;
  dsl?: unknown;
  [key: string]: unknown;
}

export interface PlatformIngestionLogList {
  total: number;
  logs: PlatformIngestionLog[];
}

export interface PlatformSkillTreeNode {
  skill_kwd?: string;
  name?: string;
  title?: string;
  description?: string;
  children?: PlatformSkillTreeNode[];
  [key: string]: unknown;
}

export interface PlatformDatasetSkillPage {
  skill_kwd?: string;
  name?: string;
  title?: string;
  description?: string;
  content_md?: string;
  [key: string]: unknown;
}

export interface PlatformSkillFieldWeight {
  enabled: boolean;
  weight: number;
}

export interface PlatformSkillFieldConfig {
  name: PlatformSkillFieldWeight;
  tags: PlatformSkillFieldWeight;
  description: PlatformSkillFieldWeight;
  content: PlatformSkillFieldWeight;
}

export interface PlatformSkillSearchConfig {
  id?: string;
  space_id: string;
  embd_id: string;
  vector_similarity_weight: number;
  similarity_threshold: number;
  field_config: PlatformSkillFieldConfig;
  rerank_id?: string;
  top_k: number;
  index_version?: string;
  status?: string;
  [key: string]: unknown;
}

export interface PlatformSkillSearchConfigRequest {
  space_id: string;
  embd_id: string;
  vector_similarity_weight: number;
  similarity_threshold: number;
  field_config: PlatformSkillFieldConfig;
  rerank_id: string;
  top_k: number;
}

export interface PlatformSkillSearchResult {
  skill_id: string;
  folder_id: string;
  name: string;
  description: string;
  tags: string[];
  score: number;
  bm25_score?: number;
  vector_score?: number;
  index_version?: string;
  create_time?: number;
  version?: string;
}

export interface PlatformSkillSearchResponse {
  skills: PlatformSkillSearchResult[];
  total: number;
  query: string;
  search_type: "keyword" | "vector" | "hybrid" | string;
}

export interface PlatformSkillInfo {
  id: string;
  folder_id: string;
  name: string;
  description: string;
  tags: string[];
  content: string;
  version: string;
}

export interface PlatformSkillSpace {
  id: string;
  tenant_id?: string;
  name: string;
  folder_id: string;
  description?: string;
  embd_id?: string;
  rerank_id?: string;
  top_k?: number;
  status?: "active" | "deleting" | "deleted" | string;
  create_time?: number | string;
  update_time?: number | string;
  [key: string]: unknown;
}

export interface PlatformSkillSpaceWriteRequest {
  name: string;
  description: string;
  embd_id: string;
  rerank_id: string;
}

export interface PlatformSkillSpaceUpdateRequest
  extends PlatformSkillSpaceWriteRequest {
  top_k: number;
}

export interface PlatformSkillSpaceDeleteResult {
  deleting: boolean;
  space_id: string;
}
