import type { PlatformConnectorOAuthSource } from "./connector-types";

const STORAGE_KEY = "rag-platform.connector-oauth.pending";
export const CONNECTOR_OAUTH_MESSAGE = "rag-platform-connector-oauth";

export interface PendingConnectorOAuth {
  source: PlatformConnectorOAuthSource;
  flowId: string;
  returnTo: string;
  startedAt: number;
}

export interface ConnectorOAuthMessage {
  type: typeof CONNECTOR_OAUTH_MESSAGE;
  source: PlatformConnectorOAuthSource;
  flowId: string;
  status: "success" | "error";
}

export function savePendingConnectorOAuth(value: PendingConnectorOAuth): void {
  sessionStorage.setItem(STORAGE_KEY, JSON.stringify(value));
}

export function readPendingConnectorOAuth(): PendingConnectorOAuth | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(STORAGE_KEY) ?? "null") as
      | Partial<PendingConnectorOAuth>
      | null;
    if (
      !value ||
      !["google-drive", "gmail", "box"].includes(value.source ?? "") ||
      typeof value.flowId !== "string" ||
      typeof value.returnTo !== "string" ||
      typeof value.startedAt !== "number"
    )
      return null;
    return value as PendingConnectorOAuth;
  } catch {
    return null;
  }
}

export function clearPendingConnectorOAuth(flowId?: string): void {
  const current = readPendingConnectorOAuth();
  if (!flowId || current?.flowId === flowId) sessionStorage.removeItem(STORAGE_KEY);
}

export function connectorOAuthRedirectUri(
  source: PlatformConnectorOAuthSource,
): string {
  return `${window.location.origin}/connector-oauth/${source}/callback`;
}

export function openConnectorOAuthWindow(
  source: PlatformConnectorOAuthSource,
  flowId: string,
  authorizationUrl: string,
): Window | null {
  const popup = window.open(
    authorizationUrl,
    `rag-platform-oauth:${source}:${flowId}`,
    "popup,width=620,height=760,noopener=false",
  );
  return popup;
}

export function parseConnectorOAuthWindowName(
  name: string,
): Pick<PendingConnectorOAuth, "source" | "flowId"> | null {
  const match = /^rag-platform-oauth:(google-drive|gmail|box):(.+)$/.exec(name);
  return match
    ? { source: match[1] as PlatformConnectorOAuthSource, flowId: match[2] }
    : null;
}

export function matchesConnectorOAuthCorrelation(
  source: PlatformConnectorOAuthSource,
  state: string,
  windowName: string,
  pending: PendingConnectorOAuth | null = readPendingConnectorOAuth(),
): boolean {
  const expected = parseConnectorOAuthWindowName(windowName) ?? pending;
  return Boolean(
    state && expected && expected.source === source && expected.flowId === state,
  );
}
