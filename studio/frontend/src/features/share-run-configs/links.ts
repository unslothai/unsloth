// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PerModelConfig } from "../model-picker/model-config/per-model-config";
import {
  SHARED_CONFIG_FIELDS,
  SHARED_CONFIG_KEYS,
  type SharedConfigKey,
  isSharedConfigKey,
} from "./fields";

export const MAX_RUN_CONFIG_URL_LENGTH = 16_384;
export type SharedRunConfig = {
  model?: string;
  ggufVariant?: string;
  isGguf?: boolean;
  config: Partial<PerModelConfig>;
};
export type RunConfigLinkResult =
  | { kind: "unrelated" }
  | { kind: "invalid"; error: string }
  | { kind: "valid"; value: SharedRunConfig };

const repoSegment = /^[A-Za-z0-9_](?:[A-Za-z0-9._-]*[A-Za-z0-9_])?$/;
const variantSegment = /^[A-Za-z0-9_][A-Za-z0-9._ -]*$/;
const windowsDevice = /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9]) *(?:\.|$)/i;
const malformedEscape = /%(?![0-9a-f]{2})/i;
const nativeAddress = /^unsloth:\/\/run\/?(?:\?|$)/;
const webAddress = /^https?:\/\//i;

function oversizedRunLink(raw: string): RunConfigLinkResult {
  const fragment = raw.indexOf("#");
  const web =
    webAddress.test(raw) &&
    fragment >= 0 &&
    (raw.slice(fragment, fragment + 5) === "#run?" ||
      raw.slice(fragment) === "#run");
  return nativeAddress.test(raw) || web
    ? { kind: "invalid", error: "This run configuration link is too long." }
    : { kind: "unrelated" };
}

export function isShareableModelId(model: string): boolean {
  const segments = model.split("/");
  return (
    segments.length === 2 &&
    !model.endsWith(".git") &&
    segments.every(
      (segment) =>
        segment.length <= 96 &&
        segment === segment.trim() &&
        repoSegment.test(segment) &&
        !segment.includes("..") &&
        !segment.includes("--"),
    )
  );
}

function validVariant(value: string): boolean {
  return (
    value.length > 0 &&
    value.length <= 512 &&
    value
      .split("/")
      .every(
        (part) =>
          part.length <= 255 &&
          part === part.trim() &&
          variantSegment.test(part) &&
          !part.endsWith(".") &&
          !part.endsWith(" ") &&
          !windowsDevice.test(part),
      )
  );
}

function decodeField(key: string, value: string): unknown {
  if (
    isSharedConfigKey(key) &&
    !SHARED_CONFIG_FIELDS[key].text &&
    SHARED_CONFIG_FIELDS[key].valid(value)
  ) {
    return value;
  }
  if (
    isSharedConfigKey(key) &&
    SHARED_CONFIG_FIELDS[key].text &&
    value !== "null"
  ) {
    return value.startsWith('"') ? JSON.parse(value) : value;
  }
  return JSON.parse(value);
}

function linkQuery(raw: string, url: URL, native: boolean): string {
  if (invalidUrlCharacters(raw)) {
    throw new Error("This link contains invalid URL characters.");
  }
  if (
    url.username ||
    url.password ||
    (native && (url.port || url.hash || !nativeAddress.test(raw)))
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
  switch (key) {
    case "v": {
      if (value !== "1") {
        throw new Error(
          "This run configuration link uses an unsupported version.",
        );
      }
      return;
    }
    case "model": {
      if (!isShareableModelId(value)) {
        throw new Error("Use a Hugging Face model ID such as owner/model.");
      }
      result.model = value;
      return;
    }
    case "ggufVariant": {
      if (!validVariant(value)) {
        throw new Error("The GGUF variant is invalid.");
      }
      result.ggufVariant = value;
      return;
    }
    case "isGguf": {
      if (value !== "true" && value !== "false") {
        throw new Error("The model format must be true or false.");
      }
      result.isGguf = value === "true";
      return;
    }
    default:
      readConfigField(result.config, key, value);
  }
}

function readConfigField(
  config: Partial<PerModelConfig>,
  key: string,
  value: string,
): void {
  if (!isSharedConfigKey(key)) {
    throw new Error(
      "This run configuration link contains an unsupported setting.",
    );
  }
  let decoded: unknown;
  try {
    decoded = decodeField(key, value);
  } catch {
    throw new Error(`The setting “${key}” is invalid.`);
  }
  validateConfigField(key, decoded);
  Object.assign(config, { [key]: decoded });
}

function validateConfigField(key: SharedConfigKey, value: unknown): void {
  if (!SHARED_CONFIG_FIELDS[key].valid(value)) {
    throw new Error(
      SHARED_CONFIG_FIELDS[key].error ??
        `The setting “${SHARED_CONFIG_FIELDS[key].label}” is invalid.`,
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
  if (result.isGguf === false && result.ggufVariant !== undefined) {
    throw new Error("A GGUF variant requires a GGUF model.");
  }
  const { customContextLength, maxSeqLength } = result.config;
  if (
    customContextLength != null &&
    maxSeqLength != null &&
    customContextLength !== maxSeqLength
  ) {
    throw new Error(
      "Context length and max sequence length must agree. Share only one, or use the same value for both.",
    );
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
  if (raw.length > MAX_RUN_CONFIG_URL_LENGTH) {
    return oversizedRunLink(raw);
  }
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    return { kind: "unrelated" };
  }
  const native = url.protocol === "unsloth:" && url.hostname === "run";
  const web =
    (url.protocol === "http:" || url.protocol === "https:") &&
    (url.hash === "#run" || url.hash.startsWith("#run?"));
  if (!(native || web)) {
    return { kind: "unrelated" };
  }
  try {
    return {
      kind: "valid",
      value: parseParameters(linkQuery(raw, url, native)),
    };
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
  if (value.isGguf !== undefined) {
    params.set("isGguf", String(value.isGguf));
  }
  for (const key of SHARED_CONFIG_KEYS) {
    const field = value.config[key];
    if (field !== undefined) {
      validateConfigField(key, field);
      params.set(
        key,
        typeof field === "string" && !SHARED_CONFIG_FIELDS[key].text
          ? field
          : JSON.stringify(field),
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
    throw new Error("Use an HTTP or HTTPS Studio address.");
  }
  url.pathname = "/chat";
  url.search = "";
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
  if (parsed.kind !== "valid") {
    throw new Error(
      parsed.kind === "invalid"
        ? parsed.error
        : "Invalid run configuration link.",
    );
  }
  return link;
}
