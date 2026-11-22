// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const SCHEMELESS_HOST = /^(\[[0-9a-f:.]+\]|[a-z0-9.-]+)(:\d+)?([/?#].*)?$/i;
// Pasted full endpoints: the client appends these itself, so keeping one requests /chat/completions/models.
const ENDPOINT_SUFFIX =
  /\/(chat\/completions|completions|responses|messages|models)\/*$/i;
const LOCAL_HOST_NAME =
  /^(\[.*\]|\d{1,3}(\.\d{1,3}){3}|[^.]+|.+\.(localhost|local|lan|internal|home\.arpa))$/i;

// The API key rides on every request, so a bare public domain (api.openai.com/v1) must not default to cleartext.
function defaultScheme(host: string, port: string | undefined): string {
  if (port && port !== ":443") return "http";
  return LOCAL_HOST_NAME.test(host) ? "http" : "https";
}

export function normalizeProviderBaseUrl(
  input: string,
  options: { appendOpenAiVersionPath?: boolean } = {},
): string | null {
  let trimmed = input.trim();
  if (!trimmed) return null;
  const schemeless = trimmed.includes("://")
    ? null
    : SCHEMELESS_HOST.exec(trimmed);
  if (schemeless) {
    trimmed = `${defaultScheme(schemeless[1], schemeless[2])}://${trimmed}`;
  }
  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    throw new Error("Base URL must be a valid URL.");
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("Base URL must use http or https.");
  }
  parsed.pathname = parsed.pathname.replace(ENDPOINT_SUFFIX, "");
  if (options.appendOpenAiVersionPath) {
    const pathname = parsed.pathname.replace(/\/+$/, "");
    if (!pathname) {
      parsed.pathname = "/v1";
    }
  }
  return parsed.toString().replace(/\/+$/, "");
}
