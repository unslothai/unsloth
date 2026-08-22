export type McpOAuthFormPayload = {
  readonly useOauth: boolean;
  readonly oauthClientId?: string | null;
  readonly oauthClientSecret?: string;
};

export function buildMcpOAuthFormPayload(
  useOauth: boolean,
  clientId: string,
  clientSecret: string,
): McpOAuthFormPayload {
  if (!useOauth) {
    return { useOauth: false };
  }
  return {
    useOauth: true,
    oauthClientId: clientId.trim() || null,
    ...(clientSecret ? { oauthClientSecret: clientSecret } : {}),
  };
}
