export interface PlatformPromptConfigDto {
  system?: unknown;
  prologue?: unknown;
  [key: string]: unknown;
}

export interface PlatformChatDto {
  id?: unknown;
  name?: unknown;
  description?: unknown;
  llm_id?: unknown;
  dataset_ids?: unknown;
  prompt_config?: unknown;
  status?: unknown;
  create_date?: unknown;
  create_time?: unknown;
  update_date?: unknown;
  update_time?: unknown;
  [key: string]: unknown;
}

export interface PlatformChatCreateRequest {
  name: string;
  dataset_ids?: string[];
  llm_id?: string;
  description?: string;
  prompt_config?: Record<string, unknown>;
}

export interface PlatformChatUpdateRequest {
  name?: string;
  dataset_ids?: string[];
  llm_id?: string;
  description?: string;
  prompt_config?: Record<string, unknown>;
}

export interface PlatformSessionMessageDto {
  id?: unknown;
  role?: unknown;
  content?: unknown;
  create_time?: unknown;
  created_at?: unknown;
  [key: string]: unknown;
}

export interface PlatformSessionDto {
  id?: unknown;
  chat_id?: unknown;
  name?: unknown;
  messages?: unknown;
  reference?: unknown;
  user_id?: unknown;
  create_date?: unknown;
  create_time?: unknown;
  update_date?: unknown;
  update_time?: unknown;
  [key: string]: unknown;
}

export interface PlatformChatPage {
  chats: PlatformChatDto[];
  total: number;
}

export interface PlatformSessionListQuery {
  id?: string;
  name?: string;
  page?: number;
  pageSize?: number;
  orderby?: "create_time" | "update_time";
  desc?: boolean;
}

export interface PlatformSessionDeleteResult {
  deletedIds: string[];
  notFoundIds: string[];
}
