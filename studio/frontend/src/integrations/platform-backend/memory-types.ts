import { asRecord, stringArray, stringValue } from "./model-types";

export const PLATFORM_MEMORY_TYPES = [
  "raw",
  "semantic",
  "episodic",
  "procedural",
] as const;
export type PlatformMemoryType = (typeof PLATFORM_MEMORY_TYPES)[number];
export type PlatformMemoryPermission = "me" | "team";
export type PlatformForgettingPolicy = "FIFO";

export interface PlatformMemory {
  id: string;
  name: string;
  ownerName: string;
  tenantId: string;
  memoryTypes: PlatformMemoryType[];
  storageType: string;
  embeddingModelId: string;
  llmId: string;
  permissions: PlatformMemoryPermission;
  description: string;
  memorySize: number;
  forgettingPolicy: PlatformForgettingPolicy;
  temperature: number;
  systemPrompt: string;
  userPrompt: string;
  createTime: number | null;
  updateTime: number | null;
}

export interface PlatformMemoryMessage {
  messageId: string;
  memoryId: string;
  userId: string;
  agentId: string;
  sessionId: string;
  status: boolean;
  content: string;
  extract: unknown[];
  validAt: string;
  invalidAt: string;
  forgetAt: string;
  agentName: string;
}

export interface PlatformMemoryListResult {
  items: PlatformMemory[];
  total: number;
}

export interface PlatformMemoryMessageListResult {
  items: PlatformMemoryMessage[];
  total: number;
  storageType: string;
}

export interface CreatePlatformMemoryInput {
  name: string;
  memoryTypes: PlatformMemoryType[];
  embeddingModelId: string;
  llmId: string;
}

export interface UpdatePlatformMemoryInput {
  name?: string;
  permissions?: PlatformMemoryPermission;
  llmId?: string;
  embeddingModelId?: string;
  memoryTypes?: PlatformMemoryType[];
  memorySize?: number;
  forgettingPolicy?: PlatformForgettingPolicy;
  temperature?: number;
  description?: string;
  systemPrompt?: string;
  userPrompt?: string;
}

const MEMORY_TYPE_BITS: Array<[number, PlatformMemoryType]> = [
  [1, "raw"],
  [2, "semantic"],
  [4, "episodic"],
  [8, "procedural"],
];

function numberValue(value: unknown, fallback = 0): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function memoryTypes(value: unknown): PlatformMemoryType[] {
  if (typeof value === "number" || /^\d+$/.test(stringValue(value))) {
    const mask = numberValue(value);
    return MEMORY_TYPE_BITS.flatMap(([bit, type]) =>
      (mask & bit) !== 0 ? [type] : [],
    );
  }
  return stringArray(value).filter((type): type is PlatformMemoryType =>
    PLATFORM_MEMORY_TYPES.includes(type as PlatformMemoryType),
  );
}

export function mapPlatformMemory(value: unknown): PlatformMemory | null {
  const dto = asRecord(value);
  const id = stringValue(dto.id).trim();
  const name = stringValue(dto.name).trim();
  if (!id || !name) return null;
  return {
    id,
    name,
    ownerName: stringValue(dto.owner_name).trim(),
    tenantId: stringValue(dto.tenant_id).trim(),
    memoryTypes: memoryTypes(dto.memory_type),
    storageType: stringValue(dto.storage_type).trim() || "table",
    embeddingModelId: stringValue(dto.embd_id).trim(),
    llmId: stringValue(dto.llm_id).trim(),
    permissions: dto.permissions === "team" ? "team" : "me",
    description: stringValue(dto.description),
    memorySize: numberValue(dto.memory_size, 5 * 1024 * 1024),
    forgettingPolicy: "FIFO",
    temperature: numberValue(dto.temperature, 0.5),
    systemPrompt: stringValue(dto.system_prompt),
    userPrompt: stringValue(dto.user_prompt),
    createTime: dto.create_time == null ? null : numberValue(dto.create_time),
    updateTime: dto.update_time == null ? null : numberValue(dto.update_time),
  };
}

export function mapPlatformMemoryMessage(
  value: unknown,
): PlatformMemoryMessage | null {
  const dto = asRecord(value);
  const rawMessageId = dto.message_id ?? dto.id;
  const messageId =
    typeof rawMessageId === "string" || typeof rawMessageId === "number"
      ? String(rawMessageId).trim()
      : "";
  if (!messageId) return null;
  return {
    messageId,
    memoryId: stringValue(dto.memory_id).trim(),
    userId: stringValue(dto.user_id).trim(),
    agentId: stringValue(dto.agent_id).trim(),
    sessionId: stringValue(dto.session_id).trim(),
    status: dto.status === true || dto.status === 1 || dto.status === "1",
    content: stringValue(dto.content),
    extract: Array.isArray(dto.extract) ? dto.extract : [],
    validAt: stringValue(dto.valid_at),
    invalidAt: stringValue(dto.invalid_at),
    forgetAt: stringValue(dto.forget_at),
    agentName: stringValue(dto.agent_name),
  };
}
