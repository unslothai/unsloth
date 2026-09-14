// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Composer, RAG and model-load toggles the chat runtime store mirrors to /api/chat/settings so
// they follow the installation rather than one browser's localStorage. Every value has to
// survive a round trip through a browser's storage, so each is re-validated here before it
// reaches the backend, which rejects the whole patch on a single out-of-contract field.

import type { PersistedChatSettings } from "../api/chat-settings-api";

const MIRRORED_BOOLEAN_KEYS = [
  "reasoningEnabled",
  "toolsEnabled",
  "codeToolsEnabled",
  "imageToolsEnabled",
  "webFetchToolsEnabled",
  "deepResearchEnabled",
  "artifactsEnabled",
  "showCanvasMenuItem",
  "searchImages",
  "mcpEnabledForChat",
  "confirmToolCalls",
  "ragOcrScanned",
  "ragCaptionFigures",
  "expandQuantizations",
  "showAllQuantizations",
  "fitOnDeviceOnly",
] as const satisfies readonly (keyof PersistedChatSettings)[];

const MIRRORED_ENUM_VALUES = {
  // "full" (Full access) is session-only and never persisted.
  permissionMode: ["ask", "auto", "off"],
  ragMode: ["hybrid", "lexical", "dense"],
  ragAutoInject: ["auto", "on", "off"],
  speculativeType: ["auto", "ngram", "off"],
  gpuMemoryMode: ["auto", "manual"],
} as const satisfies Partial<
  Record<keyof PersistedChatSettings, readonly string[]>
>;

// One year, the ceiling ChatSettingsPayload and the run route both enforce. A larger value would
// be dropped from the patch and then 400 the run, so it is bounded where it is set.
export const MAX_RESEARCH_MODEL_TIMEOUT_SECONDS = 365 * 24 * 3600;
// 0 is the unlimited sentinel, not a very short budget, so it sits below the run route's finite
// floor. A stored 1..9 would 400 every run.
export const MIN_FINITE_RESEARCH_MODEL_TIMEOUT_SECONDS = 10;

// Bounds match the ge/le the backend payload enforces on the same fields.
const MIRRORED_NUMBER_BOUNDS = {
  ragTopK: { min: 1, max: 50, integer: true },
  ragAutoInjectMinScore: { min: 0, max: 1, integer: false },
  researchModelTimeoutSeconds: {
    min: 0,
    minPositive: MIN_FINITE_RESEARCH_MODEL_TIMEOUT_SECONDS,
    max: MAX_RESEARCH_MODEL_TIMEOUT_SECONDS,
    integer: true,
  },
} as const satisfies Partial<
  Record<
    keyof PersistedChatSettings,
    { min: number; minPositive?: number; max: number; integer: boolean }
  >
>;

export const MIRRORED_SETTING_KEYS = [
  ...MIRRORED_BOOLEAN_KEYS,
  ...(Object.keys(
    MIRRORED_ENUM_VALUES,
  ) as (keyof typeof MIRRORED_ENUM_VALUES)[]),
  ...(Object.keys(
    MIRRORED_NUMBER_BOUNDS,
  ) as (keyof typeof MIRRORED_NUMBER_BOUNDS)[]),
  "researchWebsitePolicy",
  "ragSource",
] as const satisfies readonly (keyof PersistedChatSettings)[];

const MAX_RESEARCH_POLICY_DOMAINS = 1000;
// 253 is the maximum length of a DNS name.
const MAX_DOMAIN_LENGTH = 253;
const MAX_RAG_KB_ID_LENGTH = 256;

export function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === "object" && !Array.isArray(value);
}

function sanitizeDomainList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter(
      (entry): entry is string =>
        typeof entry === "string" && entry.length <= MAX_DOMAIN_LENGTH,
    )
    .slice(0, MAX_RESEARCH_POLICY_DOMAINS);
}

function sanitizeResearchWebsitePolicy(
  value: unknown,
): PersistedChatSettings["researchWebsitePolicy"] | undefined {
  if (!isRecord(value)) return undefined;
  return {
    allowedDomains: sanitizeDomainList(value.allowedDomains),
    blockedDomains: sanitizeDomainList(value.blockedDomains),
  };
}

export function sanitizeRagSource(
  value: unknown,
): PersistedChatSettings["ragSource"] | undefined {
  if (!isRecord(value)) return undefined;
  if (value.type === "thread") return { type: "thread" };
  if (
    value.type === "kb" &&
    typeof value.kbId === "string" &&
    value.kbId.length > 0 &&
    value.kbId.length <= MAX_RAG_KB_ID_LENGTH
  ) {
    return { type: "kb", kbId: value.kbId };
  }
  return undefined;
}

export function sanitizeBoundedNumber(
  value: unknown,
  {
    min,
    minPositive,
    max,
    integer,
  }: { min: number; minPositive?: number; max: number; integer: boolean },
): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) return undefined;
  if (integer && !Number.isInteger(value)) return undefined;
  // A sentinel below the floor stays legal; anything between the two is not.
  if (minPositive !== undefined && value > min && value < minPositive) {
    return undefined;
  }
  return value >= min && value <= max ? value : undefined;
}

/** Copy the mirrored settings `value` holds in-contract onto `settings`. */
export function assignSanitizedMirroredSettings(
  value: Record<string, unknown>,
  settings: PersistedChatSettings,
): void {
  const target = settings as Record<string, unknown>;
  for (const key of MIRRORED_BOOLEAN_KEYS) {
    if (typeof value[key] === "boolean") target[key] = value[key];
  }
  for (const [key, allowed] of Object.entries(MIRRORED_ENUM_VALUES)) {
    const candidate = value[key];
    if (
      typeof candidate === "string" &&
      (allowed as readonly string[]).includes(candidate)
    ) {
      target[key] = candidate;
    }
  }
  for (const [key, bounds] of Object.entries(MIRRORED_NUMBER_BOUNDS)) {
    const sanitized = sanitizeBoundedNumber(value[key], bounds);
    if (sanitized !== undefined) target[key] = sanitized;
  }
  const researchWebsitePolicy = sanitizeResearchWebsitePolicy(
    value.researchWebsitePolicy,
  );
  if (researchWebsitePolicy) {
    settings.researchWebsitePolicy = researchWebsitePolicy;
  }
  const ragSource = sanitizeRagSource(value.ragSource);
  if (ragSource) settings.ragSource = ragSource;
}

/** Map a stored RAG auto-inject value onto the three-way control. Storage predating that control
 *  holds "true"/"false", which the backend rejects, so the migration has to run before a
 *  backfill sends the stored value. */
export function normalizeStoredRagAutoInject(
  raw: string,
): "auto" | "on" | "off" {
  if (raw === "auto" || raw === "on" || raw === "off") return raw;
  return raw === "false" ? "off" : "auto";
}

/** Whether a resident model's own baseline currently owns a mirrored setting. speculativeType and
 *  gpuMemoryMode are each written with a loaded* shadow that hydration cannot set, so moving the
 *  editable half alone while a shadow holds the other splits the pair. With no shadow the stored
 *  preference is what the next load reads, and hydration has to apply it or the load sends a
 *  default and persists that default back over the server's value. */
export function loadShadowOwnsMirroredSetting(
  key: string,
  shadows: {
    loadedSpeculativeType: string | null;
    loadedGpuMemoryMode: "auto" | "manual" | null;
  },
): boolean {
  if (key === "speculativeType") return shadows.loadedSpeculativeType !== null;
  if (key === "gpuMemoryMode") return shadows.loadedGpuMemoryMode !== null;
  return false;
}

// storage predating the level control holds only the confirm toggle: on -> ask, off -> off.
export function normalizeStoredPermissionMode(
  rawMode: string | null,
  legacyConfirm: boolean | null,
): "ask" | "auto" | "off" {
  if (rawMode === "ask" || rawMode === "auto" || rawMode === "off") {
    return rawMode;
  }
  if (legacyConfirm === null) return "auto";
  return legacyConfirm ? "ask" : "off";
}

/** Whether `settings` carries none of the mirrored values. */
export function hasNoMirroredSettings(
  settings: PersistedChatSettings,
): boolean {
  return MIRRORED_SETTING_KEYS.every((key) => settings[key] === undefined);
}
