const SECRET_KEY =
  /(secret|token|password|credential|authorization|api[_-]?key|private[_-]?key)/i;

export const PLATFORM_CONNECTOR_SOURCES = [
  "rss",
  "s3",
  "notion",
  "rest_api",
  "r2",
  "google_cloud_storage",
  "oci_storage",
  "slack",
  "confluence",
  "jira",
  "google_drive",
  "gmail",
  "discord",
  "webdav",
  "moodle",
  "s3_compatible",
  "dropbox",
  "box",
  "airtable",
  "asana",
  "github",
  "gitlab",
  "imap",
  "bitbucket",
  "zendesk",
  "seafile",
  "mysql",
  "postgresql",
  "bigquery",
  "dingtalk_ai_table",
  "onedrive",
  "outlook",
  "salesforce",
  "azure_blob",
] as const;

export type PlatformConnectorSource =
  (typeof PLATFORM_CONNECTOR_SOURCES)[number];
export type PlatformGoogleConnectorSource = "google-drive" | "gmail";
export type PlatformConnectorOAuthSource =
  | PlatformGoogleConnectorSource
  | "box";

export interface PlatformConnector {
  id: string;
  name: string;
  source: string;
  status: string;
  inputType: string;
  refreshFrequency: number;
  pruneFrequency: number;
  timeoutSeconds: number;
  indexingStartedAt: string | null;
  configSummary: Record<string, unknown>;
}

export interface PlatformConnectorLog {
  id: string;
  connectorId: string;
  taskType: string;
  datasetId: string;
  datasetName: string;
  status: string;
  newDocuments: number;
  totalDocuments: number;
  removedDocuments: number;
  errorCount: number;
  errorMessage: string;
  startedAt: string | null;
  updatedAt: string | null;
}

export interface PlatformConnectorLogsPage {
  total: number;
  logs: PlatformConnectorLog[];
}

export interface PlatformConnectorOAuthStart {
  flowId: string;
  authorizationUrl: string;
  expiresIn: number;
}

export interface CreatePlatformConnectorInput {
  name: string;
  source: PlatformConnectorSource;
  config: Record<string, unknown>;
  refreshFrequency?: number;
  pruneFrequency?: number;
  timeoutSeconds?: number;
}

export interface UpdatePlatformConnectorInput {
  config?: Record<string, unknown>;
  refreshFrequency?: number;
  pruneFrequency?: number;
  timeoutSeconds?: number;
  reschedule?: boolean;
  status?: "CANCEL" | "SCHEDULE";
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function text(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function numberValue(value: unknown): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function nullableText(value: unknown): string | null {
  const parsed = text(value).trim();
  return parsed || null;
}

export function redactConnectorSecrets(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(redactConnectorSecrets);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>).map(([key, item]) => [
      key,
      SECRET_KEY.test(key) ? "[redacted]" : redactConnectorSecrets(item),
    ]),
  );
}

export function mapPlatformConnector(value: unknown): PlatformConnector {
  const dto = record(value);
  return {
    id: text(dto.id),
    name: text(dto.name),
    source: text(dto.source),
    status: text(dto.status),
    inputType: text(dto.input_type),
    refreshFrequency: numberValue(dto.refresh_freq),
    pruneFrequency: numberValue(dto.prune_freq),
    timeoutSeconds: numberValue(dto.timeout_secs),
    indexingStartedAt: nullableText(dto.indexing_start),
    configSummary: record(redactConnectorSecrets(record(dto.config))),
  };
}

export function mapPlatformConnectorLog(value: unknown): PlatformConnectorLog {
  const dto = record(redactConnectorSecrets(value));
  return {
    id: text(dto.id),
    connectorId: text(dto.connector_id),
    taskType: text(dto.task_type),
    datasetId: text(dto.kb_id),
    datasetName: text(dto.kb_name),
    status: text(dto.status),
    newDocuments: numberValue(dto.new_docs_indexed),
    totalDocuments: numberValue(dto.total_docs_indexed),
    removedDocuments: numberValue(dto.docs_removed_from_index),
    errorCount: numberValue(dto.error_count),
    errorMessage: text(dto.error_msg),
    startedAt: nullableText(dto.time_started),
    updatedAt: nullableText(dto.update_date),
  };
}

export function mapConnectorOAuthStart(
  value: unknown,
): PlatformConnectorOAuthStart {
  const dto = record(value);
  return {
    flowId: text(dto.flow_id),
    authorizationUrl: text(dto.authorization_url),
    expiresIn: numberValue(dto.expires_in),
  };
}

