export interface PlatformUserDto {
  avatar?: unknown;
  color_schema?: unknown;
  create_date?: unknown;
  create_time?: unknown;
  email?: unknown;
  id?: unknown;
  is_active?: unknown;
  is_superuser?: unknown;
  language?: unknown;
  login_channel?: unknown;
  nickname?: unknown;
  timezone?: unknown;
  update_date?: unknown;
  update_time?: unknown;
}

export interface PlatformUser {
  active: boolean;
  avatar: string | null;
  colorScheme: string;
  createdAt: number | null;
  email: string;
  id: string;
  language: string;
  loginChannel: string;
  nickname: string;
  superuser: boolean;
  timezone: string;
  updatedAt: number | null;
}

export interface PlatformTenantModelsDto {
  asr_id?: unknown;
  embd_id?: unknown;
  img2txt_id?: unknown;
  llm_id?: unknown;
  name?: unknown;
  ocr_id?: unknown;
  parser_ids?: unknown;
  rerank_id?: unknown;
  role?: unknown;
  tenant_id?: unknown;
  tts_id?: unknown;
}

export interface PlatformTenantModels {
  asrModelId: string;
  chatModelId: string;
  embeddingModelId: string;
  imageToTextModelId: string;
  name: string;
  ocrModelId: string;
  parserIds: string;
  rerankModelId: string;
  role: string;
  tenantId: string;
  textToSpeechModelId: string;
}

export interface PlatformLoginChannelDto {
  channel?: unknown;
  display_name?: unknown;
  icon?: unknown;
}

export interface PlatformLoginChannel {
  channel: string;
  displayName: string;
  icon: string;
}

export interface PlatformSystemAuthConfigDto {
  disablePasswordLogin?: unknown;
  registerEnabled?: unknown;
}

export interface PlatformAuthCapabilities {
  loginChannels: PlatformLoginChannel[];
  passwordLoginEnabled: boolean;
  registrationEnabled: boolean;
}

export interface PlatformAuthResult {
  token: string;
  user: PlatformUser;
}

export interface PlatformProfileUpdate {
  avatar?: string | null;
  colorScheme?: string;
  language?: string;
  nickname?: string;
  timezone?: string;
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function timestampValue(primary: unknown, fallback: unknown): number | null {
  const numeric =
    typeof primary === "number"
      ? primary
      : typeof primary === "string" && primary.trim()
        ? Number(primary)
        : Number.NaN;
  if (Number.isFinite(numeric) && numeric > 0) return numeric;

  if (typeof fallback !== "string" || !fallback.trim()) return null;
  const parsed = Date.parse(fallback);
  return Number.isFinite(parsed) ? parsed : null;
}

export function mapPlatformUser(dto: PlatformUserDto): PlatformUser {
  return {
    id: stringValue(dto.id),
    email: stringValue(dto.email),
    nickname: stringValue(dto.nickname),
    avatar: typeof dto.avatar === "string" && dto.avatar ? dto.avatar : null,
    colorScheme: stringValue(dto.color_schema),
    createdAt: timestampValue(dto.create_time, dto.create_date),
    language: stringValue(dto.language),
    timezone: stringValue(dto.timezone),
    updatedAt: timestampValue(dto.update_time, dto.update_date),
    loginChannel: stringValue(dto.login_channel),
    active: dto.is_active === "1" || dto.is_active === true,
    superuser: dto.is_superuser === true,
  };
}

export function mapPlatformTenantModels(
  dto: PlatformTenantModelsDto,
): PlatformTenantModels {
  return {
    tenantId: stringValue(dto.tenant_id),
    name: stringValue(dto.name),
    role: stringValue(dto.role),
    chatModelId: stringValue(dto.llm_id),
    embeddingModelId: stringValue(dto.embd_id),
    rerankModelId: stringValue(dto.rerank_id),
    asrModelId: stringValue(dto.asr_id),
    textToSpeechModelId: stringValue(dto.tts_id),
    imageToTextModelId: stringValue(dto.img2txt_id),
    ocrModelId: stringValue(dto.ocr_id),
    parserIds: stringValue(dto.parser_ids),
  };
}

export function mapPlatformLoginChannel(
  dto: PlatformLoginChannelDto,
): PlatformLoginChannel | null {
  const channel = stringValue(dto.channel).trim();
  if (!channel) return null;
  return {
    channel,
    displayName: stringValue(dto.display_name).trim() || channel,
    icon: stringValue(dto.icon).trim() || "sso",
  };
}
