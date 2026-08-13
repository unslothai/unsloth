export type PlatformDatasetPermission = "me" | "team";

export type PlatformDatasetChunkMethod =
  | "naive"
  | "book"
  | "email"
  | "laws"
  | "manual"
  | "one"
  | "paper"
  | "picture"
  | "presentation"
  | "qa"
  | "table"
  | "tag"
  | "resume";

export interface PlatformDatasetDto {
  id?: unknown;
  name?: unknown;
  description?: unknown;
  document_count?: unknown;
  embedding_model?: unknown;
  permission?: unknown;
  chunk_method?: unknown;
  parser_id?: unknown;
  parser_config?: unknown;
  pipeline_id?: unknown;
  create_date?: unknown;
  create_time?: unknown;
  update_date?: unknown;
  update_time?: unknown;
}

export interface PlatformDatasetListQuery {
  desc?: boolean;
  name?: string;
  orderby?: "create_time" | "update_time";
  page: number;
  pageSize: number;
}

export interface PlatformDatasetListResult {
  items: PlatformDatasetDto[];
  total: number;
}

export interface PlatformDatasetWriteRequest {
  name: string;
  description?: string;
  embedding_model: string;
  permission: PlatformDatasetPermission;
  chunk_method?: PlatformDatasetChunkMethod;
  parser_config?: Record<string, unknown>;
  pipeline_id?: string;
  parse_type?: 2;
}

export interface PlatformDatasetUpdateRequest {
  name?: string;
  description?: string;
  embedding_model?: string;
  permission?: PlatformDatasetPermission;
  parser_id?: PlatformDatasetChunkMethod;
  parser_config?: Record<string, unknown>;
  pipeline_id?: string;
  parse_type?: 1 | 2;
}
