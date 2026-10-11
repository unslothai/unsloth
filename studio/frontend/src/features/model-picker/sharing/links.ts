// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isValidRepoId as isShareableModelId } from "@/features/deep-links";
import type { PerModelConfig } from "../model-config/per-model-config";
import {
  SHARED_CONFIG_FIELDS,
  SHARED_CONFIG_KEYS,
  type SharedConfigKey,
  isSharedConfigKey,
} from "./fields";
import {
  MAX_RUN_CONFIG_URL_LENGTH,
  isRunConfigLink,
  nativeRunAddress,
} from "./inbox";

export const DESKTOP_RUN_CONFIG_URL_WARNING_LENGTH = 2_083;
export type SharedRunConfig = {
  model?: string;
  ggufVariant?: string;
  config: Partial<PerModelConfig>;
};
export type RunConfigLinkResult =
  | { kind: "unrelated" }
  | { kind: "invalid"; error: string }
  | { kind: "valid"; value: SharedRunConfig };

const variantSegment = /^[A-Za-z0-9_][A-Za-z0-9._ -]*$/;
const windowsDevice = /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9]) *(?:\.|$)/i;
const malformedEscape = /%(?![0-9a-f]{2})/i;
export { isShareableModelId };

function validVariant(value: string): boolean {
  return (
    value.length <= 512 &&
    value
      .split("/")
      .every(
        (part) =>
          part.length <= 255 &&
          variantSegment.test(part) &&
          !part.endsWith(".") &&
          !part.endsWith(" ") &&
          !windowsDevice.test(part),
      )
  );
}

function fieldError(key: SharedConfigKey): Error {
  const field = SHARED_CONFIG_FIELDS[key];
  return new Error(field.error ?? `The setting “${field.label}” is invalid.`);
}

function linkQuery(raw: string, url: URL, native: boolean): string {
  if (invalidUrlCharacters(raw)) {
    throw new Error("This link contains invalid URL characters.");
  }
  if (
    url.username ||
    url.password ||
    (native && (url.port || url.hash || !nativeRunAddress.test(raw)))
  ) {
    throw new Error("This run configuration link has an invalid address.");
  }
  const query = native ? url.search.slice(1) : url.hash.slice(5);
  if (malformedEscape.test(query)) {
    throw new Error("This run configuration link has invalid URL encoding.");
  }
  try {
    decodeURIComponent(query);
  } catch {
    throw new Error("This run configuration link has invalid text encoding.");
  }
  return query;
}

function readParameter(
  result: SharedRunConfig,
  key: string,
  value: string,
): void {
  if (key === "v") {
    if (value !== "1") {
      throw new Error(
        "This run configuration link uses an unsupported version.",
      );
    }
  } else if (key === "model") {
    if (!isShareableModelId(value)) {
      throw new Error("Use a Hugging Face model ID such as owner/model.");
    }
    result.model = value;
  } else if (key === "ggufVariant") {
    if (!validVariant(value)) {
      throw new Error("The GGUF variant is invalid.");
    }
    result.ggufVariant = value;
  } else if (isSharedConfigKey(key)) {
    const { valid } = SHARED_CONFIG_FIELDS[key];
    let decoded: unknown = value;
    try {
      decoded = valid(value) ? value : JSON.parse(value);
    } catch {
      throw fieldError(key);
    }
    if (!valid(decoded)) {
      throw fieldError(key);
    }
    Object.assign(result.config, { [key]: decoded });
  } else {
    throw new Error(
      "This run configuration link contains an unsupported setting.",
    );
  }
}

function parseParameters(query: string): SharedRunConfig {
  const seen = new Set<string>();
  const result: SharedRunConfig = { config: {} };
  for (const [key, value] of new URLSearchParams(query)) {
    if (seen.has(key)) {
      throw new Error(
        "This run configuration link contains a repeated setting.",
      );
    }
    seen.add(key);
    readParameter(result, key, value);
  }
  if (!seen.has("v")) {
    throw new Error("This run configuration link is missing its version.");
  }
  return result;
}

function invalidUrlCharacters(raw: string): boolean {
  for (const character of raw) {
    const code = character.codePointAt(0) ?? 0;
    if (
      code <= 32 ||
      (code >= 0x7f && code <= 0x9f) ||
      (code >= 0xd800 && code <= 0xdfff)
    ) {
      return true;
    }
  }
  return false;
}

export function parseRunConfigLink(raw: string): RunConfigLinkResult {
  if (!isRunConfigLink(raw)) {
    return { kind: "unrelated" };
  }
  if (raw.length > MAX_RUN_CONFIG_URL_LENGTH) {
    return {
      kind: "invalid",
      error: "This run configuration link is too long.",
    };
  }
  const url = new URL(raw);
  try {
    const query = linkQuery(raw, url, url.protocol === "unsloth:");
    return { kind: "valid", value: parseParameters(query) };
  } catch (error) {
    return {
      kind: "invalid",
      error:
        error instanceof Error
          ? error.message
          : "Invalid run configuration link.",
    };
  }
}

function encodeParameters(value: SharedRunConfig): URLSearchParams {
  const params = new URLSearchParams({ v: "1" });
  if (value.model !== undefined) {
    params.set("model", value.model);
  }
  if (value.ggufVariant !== undefined) {
    params.set("ggufVariant", value.ggufVariant);
  }
  for (const key of SHARED_CONFIG_KEYS) {
    const field = value.config[key];
    if (field !== undefined) {
      if (!SHARED_CONFIG_FIELDS[key].valid(field)) {
        throw fieldError(key);
      }
      params.set(
        key,
        typeof field === "string" ? field : JSON.stringify(field),
      );
    }
  }
  return params;
}

function browserLink(params: URLSearchParams, address: string): string {
  const url = new URL(address);
  if (
    (url.protocol !== "http:" && url.protocol !== "https:") ||
    url.username ||
    url.password
  ) {
    throw new Error("Use an HTTP or HTTPS Unsloth Web address.");
  }
  url.pathname = "/chat";
  // Force document navigation from /chat: link intake ignores same-document hash changes.
  url.search = "?run=1";
  url.hash = `run?${params}`;
  return url.href;
}

export function createRunConfigLink(
  value: SharedRunConfig,
  browserUrl?: string,
): string {
  const params = encodeParameters(value);
  const link =
    browserUrl === undefined
      ? `unsloth://run?${params}`
      : browserLink(params, browserUrl);
  const parsed = parseRunConfigLink(link);
  if (parsed.kind === "invalid") {
    throw new Error(parsed.error);
  }
  return link;
}
