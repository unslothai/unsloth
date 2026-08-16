type UnknownRecord = Record<string, unknown>;

export interface PlatformServiceStatus {
  id: string;
  label: string;
  status: "healthy" | "degraded" | "unknown";
  type: string | null;
  latencyMs: number | null;
}

export interface PlatformOperationsStatus {
  overall: "healthy" | "degraded";
  services: PlatformServiceStatus[];
  taskExecutorCount: number;
}

export interface PlatformStatsPoint {
  at: string;
  value: number;
}

export interface PlatformUsageStats {
  pageViews: PlatformStatsPoint[];
  uniqueVisitors: PlatformStatsPoint[];
  speed: PlatformStatsPoint[];
  tokensThousands: PlatformStatsPoint[];
  rounds: PlatformStatsPoint[];
  thumbsUp: PlatformStatsPoint[];
}

export interface PlatformApiToken {
  id: string;
  label: string;
  maskedToken: string;
  createdAt: string | null;
  /** Ephemeral, memory-only value required by the backend DELETE path. */
  revokeKey: string;
}

export interface PlatformCreatedApiToken {
  token: string;
  compatibilityToken: string | null;
}

export interface PlatformLangfuseConfig {
  configured: boolean;
  host: string;
  maskedPublicKey: string;
  projectId: string | null;
  projectName: string | null;
}

export interface PlatformLangfuseInput {
  host: string;
  publicKey: string;
  secretKey: string;
}

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function numberValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function timestampValue(value: unknown): string | null {
  const timestamp = numberValue(value);
  if (timestamp === null) return null;
  const milliseconds =
    timestamp < 1_000_000_000_000 ? timestamp * 1_000 : timestamp;
  const date = new Date(milliseconds);
  return Number.isNaN(date.getTime()) ? null : date.toISOString();
}

export function maskCredential(value: string): string {
  const trimmed = value.trim();
  if (trimmed.length <= 8) return "••••••••";
  return `${trimmed.slice(0, 4)}••••${trimmed.slice(-4)}`;
}

export function mapPlatformOperationsStatus(
  value: unknown,
): PlatformOperationsStatus {
  const source = isRecord(value) ? value : {};
  const services = Object.entries(source).flatMap(([id, raw]) => {
    if (id === "task_executor_heartbeats") return [];
    if (!isRecord(raw)) return [];
    const rawStatus = stringValue(raw.status)?.toLowerCase() ?? "unknown";
    const status: PlatformServiceStatus["status"] =
      rawStatus === "green" || rawStatus === "ok" || rawStatus === "healthy"
        ? "healthy"
        : rawStatus === "red" || rawStatus === "error" || rawStatus === "down"
          ? "degraded"
          : "unknown";
    return [
      {
        id,
        label: id.replaceAll("_", " "),
        status,
        type:
          stringValue(raw.type) ??
          stringValue(raw.storage) ??
          stringValue(raw.database),
        latencyMs: numberValue(raw.elapsed),
      },
    ];
  });
  const heartbeats = isRecord(source.task_executor_heartbeats)
    ? source.task_executor_heartbeats
    : {};
  return {
    overall: services.some((service) => service.status !== "healthy")
      ? "degraded"
      : "healthy",
    services,
    taskExecutorCount: Object.keys(heartbeats).length,
  };
}

function mapStatsSeries(value: unknown): PlatformStatsPoint[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((entry) => {
    if (!Array.isArray(entry) || entry.length < 2) return [];
    const at = stringValue(entry[0]);
    const point = numberValue(entry[1]);
    return at && point !== null ? [{ at, value: point }] : [];
  });
}

export function mapPlatformUsageStats(value: unknown): PlatformUsageStats {
  const source = isRecord(value) ? value : {};
  return {
    pageViews: mapStatsSeries(source.pv),
    uniqueVisitors: mapStatsSeries(source.uv),
    speed: mapStatsSeries(source.speed),
    tokensThousands: mapStatsSeries(source.tokens),
    rounds: mapStatsSeries(source.round),
    thumbsUp: mapStatsSeries(source.thumb_up),
  };
}

export function mapPlatformApiTokens(value: unknown): PlatformApiToken[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((entry, index) => {
    if (!isRecord(entry)) return [];
    const token = stringValue(entry.token);
    if (!token) return [];
    const createdAt =
      stringValue(entry.create_date) ?? timestampValue(entry.create_time);
    return [
      {
        id: `${index}-${maskCredential(token)}`,
        label: stringValue(entry.source) ?? `API token ${index + 1}`,
        maskedToken: maskCredential(token),
        createdAt,
        revokeKey: token,
      },
    ];
  });
}

export function mapCreatedPlatformApiToken(
  value: unknown,
): PlatformCreatedApiToken {
  if (!isRecord(value)) throw new TypeError("API token yanıtı geçersiz.");
  const token = stringValue(value.token);
  if (!token) throw new TypeError("API token yanıtında token bulunamadı.");
  return {
    token,
    compatibilityToken: stringValue(value.beta),
  };
}

export function mapPlatformLangfuseConfig(
  value: unknown,
): PlatformLangfuseConfig | null {
  if (!isRecord(value)) return null;
  const host = stringValue(value.host);
  const publicKey = stringValue(value.public_key);
  if (!host || !publicKey) return null;
  return {
    configured: true,
    host,
    maskedPublicKey: maskCredential(publicKey),
    projectId: stringValue(value.project_id),
    projectName: stringValue(value.project_name),
  };
}

export function normalizePlatformLangfuseInput(
  input: PlatformLangfuseInput,
): PlatformLangfuseInput {
  const host = input.host.trim().replace(/\/+$/, "");
  const publicKey = input.publicKey.trim();
  const secretKey = input.secretKey.trim();
  if (!host || !publicKey || !secretKey) {
    throw new TypeError("Langfuse alanlarının tümü zorunludur.");
  }
  const parsed = new URL(host);
  if (parsed.protocol !== "https:" && parsed.protocol !== "http:") {
    throw new TypeError("Langfuse adresi HTTP veya HTTPS olmalıdır.");
  }
  return { host, publicKey, secretKey };
}
