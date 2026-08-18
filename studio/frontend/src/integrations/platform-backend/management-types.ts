export type ManagementScalar = string | number | boolean | null;
export type ManagementJson =
  | ManagementScalar
  | ManagementJson[]
  | { [key: string]: ManagementJson };

export interface ManagementRecord {
  id: string;
  label: string;
  values: Record<string, ManagementJson>;
}

export interface ManagementSnapshot {
  key: string;
  label: string;
  data: ManagementJson;
}

export type ManagementArea =
  | "admin"
  | "tenant"
  | "bots"
  | "channels"
  | "templates"
  | "compatibility";

export interface ManagementOperation {
  area: ManagementArea;
  bodyTemplate?: Record<string, ManagementJson>;
  danger: boolean;
  description: string;
  endpoint: string;
  id: string;
  label: string;
  method: "DELETE" | "GET" | "PATCH" | "POST" | "PUT";
  needsAdminToken?: boolean;
  pathParameters?: string[];
  queryParameters?: string[];
  requiresAuditReason?: boolean;
}

const SECRET_KEY =
  /(authorization|credential|password|secret|token|api[_-]?key|private[_-]?key|access[_-]?key)/i;

function scalar(value: unknown): ManagementJson {
  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }
  return String(value);
}

export function redactManagementData(
  value: unknown,
  key = "",
  seen = new WeakSet<object>(),
): ManagementJson {
  if (SECRET_KEY.test(key) && value !== null && value !== "") return "••••••••";
  if (Array.isArray(value)) {
    if (seen.has(value)) return "[circular]";
    seen.add(value);
    return value.map((entry) => redactManagementData(entry, key, seen));
  }
  if (value && typeof value === "object") {
    if (seen.has(value)) return "[circular]";
    seen.add(value);
    return Object.fromEntries(
      Object.entries(value).map(([childKey, childValue]) => [
        childKey,
        redactManagementData(childValue, childKey, seen),
      ]),
    );
  }
  return scalar(value);
}

function recordId(value: Record<string, unknown>, index: number): string {
  for (const key of ["id", "user_id", "tenant_id", "name", "email", "username"]) {
    const candidate = value[key];
    if (typeof candidate === "string" && candidate.trim()) return candidate;
  }
  return String(index + 1);
}

export function toManagementRecords(value: unknown): ManagementRecord[] {
  const source = Array.isArray(value)
    ? value
    : value && typeof value === "object"
      ? [value]
      : [];
  return source.map((entry, index) => {
    const raw: Record<string, unknown> =
      entry && typeof entry === "object" && !Array.isArray(entry)
        ? (entry as Record<string, unknown>)
        : { value: entry };
    const id = recordId(raw, index);
    const labelCandidate =
      raw.name ?? raw.nickname ?? raw.email ?? raw.username ?? raw.type ?? id;
    return {
      id,
      label: typeof labelCandidate === "string" ? labelCandidate : id,
      values: redactManagementData(raw) as Record<string, ManagementJson>,
    };
  });
}

export function parseManagementJson(value: string): Record<string, ManagementJson> {
  const parsed: unknown = value.trim() ? JSON.parse(value) : {};
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new TypeError("İstek gövdesi bir JSON nesnesi olmalıdır.");
  }
  return parsed as Record<string, ManagementJson>;
}
