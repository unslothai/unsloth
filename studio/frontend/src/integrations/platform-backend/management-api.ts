import { encryptPlatformPassword } from "./auth-crypto";
import { platformRequest } from "./client";
import { PlatformApiError } from "./errors";
import type {
  ManagementJson,
  ManagementOperation,
  ManagementSnapshot,
} from "./management-types";
import { redactManagementData } from "./management-types";

type JsonObject = Record<string, ManagementJson>;

export interface AdminSession {
  email: string;
  token: string;
}

export interface AdminDashboardOptions {
  signal?: AbortSignal;
  token: string;
}

function adminRequest<T>(
  endpoint: string,
  token: string,
  options: {
    body?: unknown;
    method?: string;
    query?: Record<string, string | number | boolean | null | undefined>;
    signal?: AbortSignal;
  } = {},
): Promise<T> {
  return platformRequest<T>(endpoint, {
    method: options.method,
    token,
    authorizationScheme: "raw",
    json: options.body,
    query: options.query,
    signal: options.signal,
    redirectOnUnauthorized: false,
    getRetries: 0,
  });
}

function normalizedAuthorization(value: string | null): string {
  const token = value?.trim().replace(/^Bearer\s+/i, "") ?? "";
  if (!token) {
    throw new PlatformApiError("Yönetici oturum anahtarı alınamadı.", {
      code: "MISSING_ADMIN_AUTHORIZATION",
      endpoint: "/admin/login",
      httpStatus: 200,
    });
  }
  return token;
}

export async function loginPlatformAdmin(
  email: string,
  password: string,
  options: { publicKeyPem?: string; signal?: AbortSignal } = {},
): Promise<AdminSession> {
  let authorization: string | null = null;
  await platformRequest<unknown>("/admin/login", {
    method: "POST",
    token: null,
    json: {
      email: email.trim(),
      password: encryptPlatformPassword(password, options.publicKeyPem),
    },
    signal: options.signal,
    redirectOnUnauthorized: false,
    onResponse: (response) => {
      authorization = response.headers.get("authorization");
    },
  });
  return { email: email.trim(), token: normalizedAuthorization(authorization) };
}

export async function logoutPlatformAdmin(token: string, signal?: AbortSignal) {
  try {
    await adminRequest("/admin/logout", token, { method: "POST", signal });
  } finally {
    // The caller owns the in-memory token and always clears it in finally.
  }
}

export function checkPlatformAdmin(token: string, signal?: AbortSignal) {
  return adminRequest<unknown>("/admin/auth", token, { signal });
}

const ADMIN_READS: Array<{
  endpoint: string;
  key: string;
  label: string;
  query?: Record<string, string | number>;
}> = [
  { endpoint: "/admin/version", key: "version", label: "Sürüm" },
  { endpoint: "/admin/reports", key: "reports", label: "Raporlar" },
  { endpoint: "/admin/users", key: "users", label: "Kullanıcılar", query: { page: 1, page_size: 50 } },
  { endpoint: "/admin/services", key: "services", label: "Servisler" },
  { endpoint: "/admin/variables", key: "variables", label: "Değişkenler" },
  { endpoint: "/admin/configs", key: "configs", label: "Yapılandırmalar" },
  { endpoint: "/admin/config/log", key: "log", label: "Log seviyesi" },
  { endpoint: "/admin/environments", key: "environments", label: "Ortamlar" },
  { endpoint: "/admin/queue", key: "queue", label: "Kuyruk" },
  { endpoint: "/admin/queue/messages", key: "queue-messages", label: "Kuyruk mesajları" },
  { endpoint: "/admin/ingestors", key: "ingestors", label: "Ingestor'lar" },
  { endpoint: "/admin/ingestion/tasks", key: "ingestion-tasks", label: "Ingestion görevleri" },
  { endpoint: "/admin/ingestion/tasks/summary", key: "ingestion-summary", label: "Ingestion özeti" },
  { endpoint: "/admin/sandbox/providers", key: "sandbox-providers", label: "Sandbox sağlayıcıları" },
  { endpoint: "/admin/sandbox/config", key: "sandbox-config", label: "Sandbox yapılandırması" },
  { endpoint: "/admin/all-models", key: "all-models", label: "Tüm modeller" },
  { endpoint: "/admin/roles", key: "roles", label: "Roller" },
  { endpoint: "/admin/roles/resource", key: "resources", label: "İzin kaynakları" },
  { endpoint: "/admin/data/summary", key: "data-summary", label: "Veri özeti" },
  { endpoint: "/admin/data/storage", key: "data-storage", label: "Depolama sağlığı" },
  { endpoint: "/admin/data/index", key: "data-index", label: "Arama motoru sağlığı" },
  { endpoint: "/admin/data/orphan", key: "data-orphan", label: "Yetim veri" },
];

export async function getPlatformAdminDashboard({ token, signal }: AdminDashboardOptions) {
  await checkPlatformAdmin(token, signal);
  return Promise.all(
    ADMIN_READS.map(async (read): Promise<ManagementSnapshot> => {
      const data = await adminRequest<unknown>(read.endpoint, token, {
        query: read.query,
        signal,
      });
      return { key: read.key, label: read.label, data: redactManagementData(data) };
    }),
  );
}

function interpolateEndpoint(
  template: string,
  parameters: Record<string, string>,
): string {
  return template.replace(/:([A-Za-z0-9_]+)|<([^>]+)>|\*([A-Za-z0-9_]+)/g, (_, a, b, c) => {
    const name = a || b || c;
    const value = parameters[name]?.trim();
    if (!value) throw new TypeError(`${name} yolu için değer zorunludur.`);
    const encodedValue = name === "username" && template.startsWith("/admin/users/")
      ? encodeAdminUsername(value)
      : value;
    return encodeURIComponent(encodedValue);
  });
}

function encodeAdminUsername(username: string): string {
  const value = username.trim();
  if (!value) throw new TypeError("Kullanıcı e-postası zorunludur.");
  const bytes = new TextEncoder().encode(value);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return globalThis.btoa(binary);
}

export async function executeManagementOperation(
  operation: ManagementOperation,
  input: {
    adminToken?: string;
    auditReason?: string;
    body?: JsonObject;
    pathParameters?: Record<string, string>;
    query?: Record<string, string>;
    signal?: AbortSignal;
  },
): Promise<ManagementJson> {
  if (operation.requiresAuditReason && !input.auditReason?.trim()) {
    throw new TypeError("Bu işlem için denetim gerekçesi zorunludur.");
  }
  const endpoint = interpolateEndpoint(operation.endpoint, input.pathParameters ?? {});
  const body = input.body
    ? {
        ...input.body,
        ...(operation.requiresAuditReason ? { audit_reason: input.auditReason?.trim() } : {}),
      }
    : operation.requiresAuditReason
      ? { audit_reason: input.auditReason?.trim() }
      : undefined;
  const options = {
    method: operation.method,
    json: body,
    query: input.query,
    signal: input.signal,
    getRetries: 0,
    redirectOnUnauthorized: !operation.needsAdminToken,
  } as const;
  const result = operation.needsAdminToken
    ? await adminRequest<unknown>(endpoint, input.adminToken ?? "", {
        method: operation.method,
        body,
        query: input.query,
        signal: input.signal,
      })
    : await platformRequest<unknown>(endpoint, options);
  return redactManagementData(result);
}

export function listPlatformTenants(signal?: AbortSignal) {
  return platformRequest<unknown[]>("/tenants", { signal, getRetries: 0 });
}

export function getPlatformTenant(tenantId: string, signal?: AbortSignal) {
  return platformRequest<unknown>(`/tenants/${encodeURIComponent(tenantId)}`, {
    signal,
    getRetries: 0,
  });
}

export function updatePlatformTenant(tenantId: string, name: string, signal?: AbortSignal) {
  return platformRequest<unknown>(`/tenants/${encodeURIComponent(tenantId)}`, {
    method: "PUT",
    json: { name: name.trim() },
    signal,
  });
}

export function listPlatformTenantMembers(tenantId: string, signal?: AbortSignal) {
  return platformRequest<unknown[]>(`/tenants/${encodeURIComponent(tenantId)}/users`, {
    signal,
    getRetries: 0,
  });
}

export function invitePlatformTenantMember(tenantId: string, email: string, signal?: AbortSignal) {
  return platformRequest<unknown>(`/tenants/${encodeURIComponent(tenantId)}/users`, {
    method: "POST",
    json: { email: email.trim() },
    signal,
  });
}

export function removePlatformTenantMember(tenantId: string, userId: string, signal?: AbortSignal) {
  return platformRequest<unknown>(`/tenants/${encodeURIComponent(tenantId)}/users`, {
    method: "DELETE",
    json: { user_id: userId },
    signal,
  });
}

export function updatePlatformTenantMemberRole(
  tenantId: string,
  userId: string,
  role: "admin" | "normal",
  signal?: AbortSignal,
) {
  return platformRequest<unknown>(
    `/tenants/${encodeURIComponent(tenantId)}/users/${encodeURIComponent(userId)}/role`,
    { method: "PUT", json: { role }, signal },
  );
}

export interface PublicTokenPair {
  beta: string;
  tenantId: string;
  token: string;
}

function publicTokenPair(value: unknown): PublicTokenPair {
  const record = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  if (
    typeof record.token !== "string" ||
    !record.token ||
    typeof record.beta !== "string" ||
    !record.beta ||
    typeof record.tenant_id !== "string" ||
    !record.tenant_id
  ) {
    throw new PlatformApiError("Public/embed token yanıtı geçersiz.", {
      code: "INVALID_PUBLIC_TOKEN_RESPONSE",
      endpoint: "/admin/users/:username/tokens",
      httpStatus: 200,
    });
  }
  return { beta: record.beta, tenantId: record.tenant_id, token: record.token };
}

export function listPlatformUserPublicTokens(
  adminToken: string,
  username: string,
  signal?: AbortSignal,
) {
  return adminRequest<unknown[]>(`/admin/users/${encodeURIComponent(encodeAdminUsername(username))}/tokens`, adminToken, {
    signal,
  }).then((value) => redactManagementData(value));
}

export async function createPlatformUserPublicToken(
  adminToken: string,
  username: string,
  signal?: AbortSignal,
): Promise<PublicTokenPair> {
  const value = await adminRequest<unknown>(
    `/admin/users/${encodeURIComponent(encodeAdminUsername(username))}/tokens`,
    adminToken,
    { method: "POST", signal },
  );
  return publicTokenPair(value);
}

export function revokePlatformUserPublicToken(
  adminToken: string,
  username: string,
  token: string,
  signal?: AbortSignal,
) {
  return adminRequest<unknown>(
    `/admin/users/${encodeURIComponent(encodeAdminUsername(username))}/tokens/${encodeURIComponent(token)}`,
    adminToken,
    { method: "DELETE", signal },
  );
}

export async function rotatePlatformUserPublicToken(
  adminToken: string,
  username: string,
  currentToken: string,
  signal?: AbortSignal,
): Promise<PublicTokenPair> {
  const replacement = await createPlatformUserPublicToken(adminToken, username, signal);
  try {
    await revokePlatformUserPublicToken(adminToken, username, currentToken, signal);
    return replacement;
  } catch (cause) {
    // Restore the pre-rotation state when old-token revoke fails. This avoids
    // silently leaving two live public credentials after a partial mutation.
    await revokePlatformUserPublicToken(adminToken, username, replacement.token, signal).catch(
      () => undefined,
    );
    throw cause;
  }
}

export function acceptPlatformTenantInvite(tenantId: string, signal?: AbortSignal) {
  return platformRequest<unknown>(`/tenants/${encodeURIComponent(tenantId)}`, {
    method: "PATCH",
    signal,
  });
}

export function listPlatformChatChannels(signal?: AbortSignal) {
  return platformRequest<unknown[]>("/chat-channels", { signal, getRetries: 0 });
}

export function createPlatformChatChannel(
  input: { chatId?: string; channel: string; config: JsonObject; name: string },
  signal?: AbortSignal,
) {
  return platformRequest<unknown>("/chat-channels", {
    method: "POST",
    json: {
      name: input.name.trim(),
      channel: input.channel.trim(),
      config: input.config,
      ...(input.chatId?.trim() ? { chat_id: input.chatId.trim() } : {}),
    },
    signal,
  });
}

export function listPlatformCompilationTemplateGroups(signal?: AbortSignal) {
  return platformRequest<unknown[]>("/compilation_template_groups", { signal, getRetries: 0 });
}

export function listPlatformCompilationBuiltins(signal?: AbortSignal) {
  return platformRequest<unknown[]>("/compilation_templates/builtins", { signal, getRetries: 0 });
}

export function listPlatformCompilationWikiPresets(signal?: AbortSignal) {
  return platformRequest<unknown[]>("/compilation_templates/wiki_presets", { signal, getRetries: 0 });
}

export function getPlatformDifyHealth(signal?: AbortSignal) {
  return platformRequest<unknown>("/dify/retrieval/health", {
    signal,
    token: null,
    getRetries: 0,
    redirectOnUnauthorized: false,
  });
}

export interface AimlapiAuthorizationStart {
  expiresIn: number;
  interval: number;
  requestId: string;
  verificationUri: string;
}

export interface AimlapiAuthorizationPoll {
  apiKey?: string;
  status: string;
}

export async function startPlatformAimlapiAuthorization(
  signal?: AbortSignal,
): Promise<AimlapiAuthorizationStart> {
  const value = await platformRequest<unknown>("/llm/aimlapi/authorize/start", {
    method: "POST",
    signal,
  });
  const record = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  const verificationUri = typeof record.verification_uri === "string" ? record.verification_uri : "";
  let parsed: URL;
  try {
    parsed = new URL(verificationUri);
  } catch {
    throw new PlatformApiError("AIMLAPI doğrulama URL'si geçersiz.", {
      code: "INVALID_AIMLAPI_AUTHORIZATION_RESPONSE",
      endpoint: "/llm/aimlapi/authorize/start",
      httpStatus: 200,
    });
  }
  if (
    typeof record.request_id !== "string" ||
    !record.request_id ||
    parsed.protocol !== "https:"
  ) {
    throw new PlatformApiError("AIMLAPI authorization yanıtı geçersiz.", {
      code: "INVALID_AIMLAPI_AUTHORIZATION_RESPONSE",
      endpoint: "/llm/aimlapi/authorize/start",
      httpStatus: 200,
    });
  }
  return {
    requestId: record.request_id,
    verificationUri: parsed.toString(),
    interval: typeof record.interval === "number" ? record.interval : 5,
    expiresIn: typeof record.expires_in === "number" ? record.expires_in : 900,
  };
}

export async function pollPlatformAimlapiAuthorization(
  requestId: string,
  signal?: AbortSignal,
): Promise<AimlapiAuthorizationPoll> {
  const value = await platformRequest<unknown>("/llm/aimlapi/authorize/poll", {
    method: "POST",
    json: { request_id: requestId },
    signal,
  });
  const record = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  if (typeof record.status !== "string" || !record.status) {
    throw new PlatformApiError("AIMLAPI poll yanıtı geçersiz.", {
      code: "INVALID_AIMLAPI_POLL_RESPONSE",
      endpoint: "/llm/aimlapi/authorize/poll",
      httpStatus: 200,
    });
  }
  return {
    status: record.status,
    ...(typeof record.api_key === "string" && record.api_key ? { apiKey: record.api_key } : {}),
  };
}

const PRIMARY_PHASE14_OPERATIONS: readonly ManagementOperation[] = [
  { id: "admin-user-create", area: "admin", label: "Kullanıcı oluştur", description: "Kullanıcı adı, parola ve rol ile yönetici kullanıcısı oluşturur.", method: "POST", endpoint: "/admin/users", danger: false, needsAdminToken: true, bodyTemplate: { username: "", password: "", role: "user" } },
  { id: "admin-user-password", area: "admin", label: "Kullanıcı parolasını değiştir", description: "Seçili kullanıcının parolasını sıfırlar.", method: "PUT", endpoint: "/admin/users/:username/password", pathParameters: ["username"], danger: true, needsAdminToken: true, requiresAuditReason: true, bodyTemplate: { new_password: "" } },
  { id: "admin-user-activate", area: "admin", label: "Kullanıcı durumunu değiştir", description: "Kullanıcıyı etkinleştirir veya devre dışı bırakır.", method: "PUT", endpoint: "/admin/users/:username/activate", pathParameters: ["username"], danger: true, needsAdminToken: true, requiresAuditReason: true, bodyTemplate: { activate_status: "on" } },
  { id: "admin-user-grant", area: "admin", label: "Yönetici yetkisi ver", description: "Kullanıcıya superuser yetkisi verir.", method: "PUT", endpoint: "/admin/users/:username/admin", pathParameters: ["username"], danger: true, needsAdminToken: true, requiresAuditReason: true },
  { id: "admin-user-revoke", area: "admin", label: "Yönetici yetkisini kaldır", description: "Kullanıcıdan superuser yetkisini kaldırır.", method: "DELETE", endpoint: "/admin/users/:username/admin", pathParameters: ["username"], danger: true, needsAdminToken: true, requiresAuditReason: true },
  { id: "admin-user-delete", area: "admin", label: "Kullanıcıyı sil", description: "Kullanıcı ve ilişkili yönetim kaydını siler.", method: "DELETE", endpoint: "/admin/users/:username", pathParameters: ["username"], danger: true, needsAdminToken: true, requiresAuditReason: true },
  { id: "admin-service-start", area: "admin", label: "Servisi başlat", description: "Servis operasyonunu start olarak çalıştırır.", method: "POST", endpoint: "/admin/services/:service_id", pathParameters: ["service_id"], danger: true, needsAdminToken: true, requiresAuditReason: true },
  { id: "admin-service-stop", area: "admin", label: "Servisi durdur", description: "Servis operasyonunu shutdown olarak çalıştırır.", method: "DELETE", endpoint: "/admin/services/:service_id", pathParameters: ["service_id"], danger: true, needsAdminToken: true, requiresAuditReason: true },
  { id: "admin-service-restart", area: "admin", label: "Servisi yeniden başlat", description: "Servis operasyonunu restart olarak çalıştırır.", method: "PUT", endpoint: "/admin/services/:service_id", pathParameters: ["service_id"], danger: true, needsAdminToken: true, requiresAuditReason: true },
  { id: "admin-variable-set", area: "admin", label: "Değişken ayarla", description: "Yönetim değişkenini günceller.", method: "PUT", endpoint: "/admin/variables", danger: true, needsAdminToken: true, requiresAuditReason: true, bodyTemplate: { var_name: "", var_value: "" } },
  { id: "admin-log-level", area: "admin", label: "Log seviyesini ayarla", description: "Çalışma zamanı log seviyesini günceller.", method: "PUT", endpoint: "/admin/config/log", danger: true, needsAdminToken: true, requiresAuditReason: true, bodyTemplate: { log_level: "INFO" } },
  { id: "admin-queue-publish", area: "admin", label: "Kuyruğa test görevi yayınla", description: "Ingestion test görevi yayınlar.", method: "POST", endpoint: "/admin/queue/messages", danger: true, needsAdminToken: true, requiresAuditReason: true, bodyTemplate: { message: "" } },
  { id: "admin-queue-pull", area: "admin", label: "Kuyruktan mesaj çek", description: "Belirli sayıda mesajı ACK/NACK politikasıyla çeker.", method: "PUT", endpoint: "/admin/queue/messages", danger: true, needsAdminToken: true, requiresAuditReason: true, bodyTemplate: { message_count: 1, ack_policy: "NACK" } },
  { id: "admin-ingestion-stop", area: "admin", label: "Ingestion görevini durdur", description: "Görev kimlikleri için stop ister.", method: "PUT", endpoint: "/admin/ingestion/tasks", danger: true, needsAdminToken: true, requiresAuditReason: true, bodyTemplate: { tasks: [] } },
  { id: "admin-ingestion-remove", area: "admin", label: "Ingestion görevini kaldır", description: "Görev kayıtlarını kaldırır.", method: "DELETE", endpoint: "/admin/ingestion/tasks", danger: true, needsAdminToken: true, requiresAuditReason: true, bodyTemplate: { tasks: [] } },
  { id: "admin-ingestor-shutdown", area: "admin", label: "Ingestor'ı güvenli kapat", description: "Taze heartbeat ile doğrulanan hedefe NATS üzerinden gerçek graceful-shutdown komutu gönderir.", method: "DELETE", endpoint: "/admin/ingestors", danger: true, needsAdminToken: true, requiresAuditReason: true, bodyTemplate: { ingestor_name: "" } },
  { id: "admin-sandbox-test", area: "admin", label: "Sandbox bağlantısını test et", description: "Provider yapılandırmasını kalıcılaştırmadan test eder.", method: "POST", endpoint: "/admin/sandbox/test", danger: false, needsAdminToken: true, bodyTemplate: { provider_type: "", config: {} } },
  { id: "admin-sandbox-save", area: "admin", label: "Sandbox yapılandırmasını kaydet", description: "Sandbox provider yapılandırmasını etkinleştirir.", method: "POST", endpoint: "/admin/sandbox/config", danger: true, needsAdminToken: true, requiresAuditReason: true, bodyTemplate: { provider_type: "", config: {} } },
  { id: "admin-orphan-purge", area: "admin", label: "Yetim veriyi temizle", description: "Sahipsiz backend verisini geri alınamaz biçimde temizler.", method: "DELETE", endpoint: "/admin/data/orphan", danger: true, needsAdminToken: true, requiresAuditReason: true },
  { id: "tenant-accept", area: "tenant", label: "Davet kabul et", description: "Bekleyen tenant davetini kabul eder.", method: "PATCH", endpoint: "/tenants/:tenant_id", pathParameters: ["tenant_id"], danger: false },
  { id: "tenant-detail", area: "tenant", label: "Tenant detayını getir", description: "Aktif üyelik kapsamındaki tenant detayını getirir.", method: "GET", endpoint: "/tenants/:tenant_id", pathParameters: ["tenant_id"], danger: false },
  { id: "tenant-update", area: "tenant", label: "Tenant adını güncelle", description: "Yalnızca tenant sahibi görünen adı günceller.", method: "PUT", endpoint: "/tenants/:tenant_id", pathParameters: ["tenant_id"], danger: true, requiresAuditReason: true, bodyTemplate: { name: "" } },
  { id: "tenant-invite", area: "tenant", label: "Takım üyesi davet et", description: "E-posta adresiyle üyelik daveti gönderir.", method: "POST", endpoint: "/tenants/:tenant_id/users", pathParameters: ["tenant_id"], danger: false, bodyTemplate: { email: "" } },
  { id: "tenant-remove", area: "tenant", label: "Takım üyesini kaldır", description: "Tenant üyesini kaldırır; son yönetici koruması backend tarafından doğrulanır.", method: "DELETE", endpoint: "/tenants/:tenant_id/users", pathParameters: ["tenant_id"], danger: true, requiresAuditReason: true, bodyTemplate: { user_id: "" } },
  { id: "tenant-role", area: "tenant", label: "Tenant üye rolünü güncelle", description: "Kabul edilmiş üyeyi normal veya admin rolüne geçirir; owner değiştirilemez.", method: "PUT", endpoint: "/tenants/:tenant_id/users/:user_id/role", pathParameters: ["tenant_id", "user_id"], danger: true, requiresAuditReason: true, bodyTemplate: { role: "normal" } },
  { id: "channel-create", area: "channels", label: "Kanal oluştur", description: "Chat kanalını yapılandırır ve yayınlama runtime'ına bağlar.", method: "POST", endpoint: "/chat-channels", danger: false, bodyTemplate: { name: "", channel: "", config: {}, chat_id: "" } },
  { id: "channel-detail", area: "channels", label: "Kanal detayını getir", description: "Seçili kanalın güvenli yapılandırma görünümünü getirir.", method: "GET", endpoint: "/chat-channels/:channel_id", pathParameters: ["channel_id"], danger: false },
  { id: "channel-update", area: "channels", label: "Kanalı güncelle", description: "Kanal yapılandırmasını günceller.", method: "PATCH", endpoint: "/chat-channels/:channel_id", pathParameters: ["channel_id"], danger: false, bodyTemplate: { name: "", config: {} } },
  { id: "channel-runtime", area: "channels", label: "Kanal runtime durumunu getir", description: "Canlı kanal durumunu gösterir.", method: "GET", endpoint: "/chat-channels/:channel_id/runtime", pathParameters: ["channel_id"], danger: false },
  { id: "channel-delete", area: "channels", label: "Kanalı sil", description: "Kanalı ve runtime bağlantısını kaldırır.", method: "DELETE", endpoint: "/chat-channels/:channel_id", pathParameters: ["channel_id"], danger: true, requiresAuditReason: true },
  { id: "template-create", area: "templates", label: "Template grubu oluştur", description: "Compilation template grubunu oluşturur.", method: "POST", endpoint: "/compilation_template_groups", danger: false, bodyTemplate: { name: "", description: "", templates: [] } },
  { id: "template-detail", area: "templates", label: "Template grubu detayını getir", description: "Seçili compilation template grubunu getirir.", method: "GET", endpoint: "/compilation_template_groups/:group_id", pathParameters: ["group_id"], danger: false },
  { id: "template-update", area: "templates", label: "Template grubunu güncelle", description: "Compilation template grubunu günceller.", method: "PUT", endpoint: "/compilation_template_groups/:group_id", pathParameters: ["group_id"], danger: false, bodyTemplate: { name: "", description: "", templates: [] } },
  { id: "template-delete", area: "templates", label: "Template grubunu sil", description: "Compilation template grubunu siler.", method: "DELETE", endpoint: "/compilation_template_groups/:group_id", pathParameters: ["group_id"], danger: true, requiresAuditReason: true },
  { id: "chatbot-info", area: "bots", label: "Chatbot bilgisini getir", description: "Tenant'a ait chatbot paylaşım bilgisini doğrular.", method: "GET", endpoint: "/chatbots/:dialog_id/info", pathParameters: ["dialog_id"], danger: false },
  { id: "chatbot-run", area: "bots", label: "Chatbot yanıtını test et", description: "Non-stream chatbot completion çalıştırır.", method: "POST", endpoint: "/chatbots/:dialog_id/completions", pathParameters: ["dialog_id"], danger: false, bodyTemplate: { question: "", stream: false } },
  { id: "agentbot-inputs", area: "bots", label: "Agentbot girdilerini getir", description: "Agent başlangıç formu girdilerini listeler.", method: "GET", endpoint: "/agentbots/:agent_id/inputs", pathParameters: ["agent_id"], danger: false },
  { id: "agentbot-log", area: "bots", label: "Agentbot logunu getir", description: "Belirli agent çalıştırmasının tanılama kaydını getirir.", method: "GET", endpoint: "/agentbots/:agent_id/logs/:log_id", pathParameters: ["agent_id", "log_id"], danger: false },
  { id: "agentbot-run", area: "bots", label: "Agentbot yanıtını test et", description: "Non-stream agentbot completion çalıştırır.", method: "POST", endpoint: "/agentbots/:agent_id/completions", pathParameters: ["agent_id"], danger: false, bodyTemplate: { question: "", stream: false } },
  { id: "searchbot-detail", area: "bots", label: "Searchbot detayını getir", description: "Search app paylaşım detayını getirir.", method: "GET", endpoint: "/searchbots/detail", queryParameters: ["search_id"], danger: false },
  { id: "searchbot-ask", area: "bots", label: "Searchbot'a sor", description: "Search app üzerinden soru çalıştırır.", method: "POST", endpoint: "/searchbots/ask", danger: false, bodyTemplate: { search_id: "", question: "" } },
  { id: "searchbot-retrieval", area: "bots", label: "Searchbot retrieval test", description: "Veri kümesi üzerinde retrieval testi çalıştırır.", method: "POST", endpoint: "/searchbots/retrieval_test", danger: false, bodyTemplate: { question: "", kb_id: [], page: 1, size: 10 } },
  { id: "searchbot-related", area: "bots", label: "İlgili soruları üret", description: "Kullanıcı onayına sunulacak ilgili soru adaylarını üretir.", method: "POST", endpoint: "/searchbots/related_questions", danger: false, bodyTemplate: { search_id: "", question: "" } },
  { id: "searchbot-mindmap", area: "bots", label: "Searchbot mindmap üret", description: "Search app için mindmap verisi üretir.", method: "POST", endpoint: "/searchbots/mindmap", danger: false, bodyTemplate: { search_id: "", question: "", kb_ids: [] } },
  { id: "dify-health", area: "compatibility", label: "Dify retrieval health", description: "Dış retrieval protokolünün sağlık sözleşmesini doğrular.", method: "GET", endpoint: "/dify/retrieval/health", danger: false },
];

// These routes are intentionally explicit rather than an arbitrary URL console:
// every entry comes from the Phase 14 route inventory. The body editor remains a
// JSON object because several admin DTOs are provider/edition dependent; the
// adapter never guesses fields or persists credentials. Non-GET operations are
// always confirmation + audit-reason gated by the UI.
const ADDITIONAL_PHASE14_CONTRACTS = `
GET /admin/all-models/:model_name
GET /admin/fingerprint
GET /admin/license
POST /admin/license
POST /admin/license/config
GET /admin/log_levels
PUT /admin/log_levels
DELETE /admin/providers
GET /admin/providers
POST /admin/providers
GET /admin/providers/:provider_name
POST /admin/providers/:provider_name/connection
DELETE /admin/providers/:provider_name/instances
GET /admin/providers/:provider_name/instances
POST /admin/providers/:provider_name/instances
GET /admin/providers/:provider_name/instances/:instance_name
PUT /admin/providers/:provider_name/instances/:instance_name
GET /admin/providers/:provider_name/instances/:instance_name/balance
GET /admin/providers/:provider_name/instances/:instance_name/connection
DELETE /admin/providers/:provider_name/instances/:instance_name/models
GET /admin/providers/:provider_name/instances/:instance_name/models
POST /admin/providers/:provider_name/instances/:instance_name/models
PATCH /admin/providers/:provider_name/instances/:instance_name/models/*model_name
GET /admin/providers/:provider_name/models
GET /admin/providers/:provider_name/models/:model_name
POST /admin/reports
POST /admin/roles
DELETE /admin/roles/:role_name
GET /admin/roles/:role_name
PUT /admin/roles/:role_name
DELETE /admin/roles/:role_name/default-models
GET /admin/roles/:role_name/default-models
PATCH /admin/roles/:role_name/default-models
DELETE /admin/roles/:role_name/permission
GET /admin/roles/:role_name/permission
POST /admin/roles/:role_name/permission
GET /admin/sandbox/providers/:provider_id/schema
GET /admin/service_types/:service_type
GET /admin/services/:service_id
GET /admin/system/fingerprint
GET /admin/system/license
POST /admin/system/license
PUT /admin/system/license/config
GET /admin/users/:username
GET /admin/users/:username/activity
GET /admin/users/:username/agents
GET /admin/users/:username/chats
DELETE /admin/users/:username/data
GET /admin/users/:username/dataset
GET /admin/users/:username/datasets
GET /admin/users/:username/default-models
GET /admin/users/:username/files
GET /admin/users/:username/index
GET /admin/users/:username/keys
POST /admin/users/:username/keys
DELETE /admin/users/:username/keys/:key
GET /admin/users/:username/models
GET /admin/users/:username/permission
GET /admin/users/:username/providers
GET /admin/users/:username/providers/:provider_name/instances
GET /admin/users/:username/providers/:provider_name/instances/:instance_name/models
GET /admin/users/:username/quota
PUT /admin/users/:username/role
GET /admin/users/:username/searches
GET /admin/users/:username/storage
GET /admin/users/:username/summary
GET /admin/users/:username/tokens
POST /admin/users/:username/tokens
DELETE /admin/users/:username/tokens/:token
GET /admin/users/activity
DELETE /admin/users/data
GET /admin/users/documents
GET /admin/users/index
GET /admin/users/plan/summary
GET /admin/users/quota
GET /admin/users/quota/summary
GET /admin/users/reports
GET /admin/users/storage
GET /admin/users/summary
GET /admin/variables/:var_name
GET /all-models
GET /all-models/:model_name
GET /system/config/log
PUT /system/config/log
GET /system/environments
GET /system/oceanbase/status
GET /system/variables
PUT /system/variables
GET /system/variables/:var_name
GET /tenant/list
`.trim();

const additionalPhase14Operations: ManagementOperation[] = ADDITIONAL_PHASE14_CONTRACTS
  .split("\n")
  .map((contract) => {
    const [method, endpoint] = contract.trim().split(/\s+/, 2) as [ManagementOperation["method"], string];
    const pathParameters = [...endpoint.matchAll(/(?:[:*])([A-Za-z0-9_]+)/g)].map((match) => match[1]);
    const danger = method !== "GET";
    const area: ManagementOperation["area"] = endpoint.startsWith("/tenant/") ? "tenant" : "admin";
    return {
      id: `contract-${method.toLowerCase()}-${endpoint.replace(/[^A-Za-z0-9]+/g, "-")}`,
      area,
      label: `${method} ${endpoint}`,
      description: "Kaynak koddan doğrulanmış Faz 14 sözleşmesini typed yönetim adapter'ı üzerinden çalıştırır.",
      method,
      endpoint,
      ...(pathParameters.length > 0 ? { pathParameters } : {}),
      danger,
      needsAdminToken: endpoint.startsWith("/admin/"),
      requiresAuditReason: danger,
      ...(danger ? { bodyTemplate: {} } : {}),
    };
  });

export const PHASE14_OPERATIONS: readonly ManagementOperation[] = [
  ...PRIMARY_PHASE14_OPERATIONS,
  ...additionalPhase14Operations,
];
