// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The chat settings that describe one conversation rather than the installation: the composer
// pills, the permission level and the retrieval controls. Editing one with a chat open writes
// this snapshot onto the thread, and reopening that thread applies it back.

import type {
  PermissionMode,
  RagAutoInject,
  RagMode,
  RagSource,
  ReasoningEffort,
} from "../stores/chat-runtime-store";
import {
  isRecord,
  sanitizeBoundedNumber,
  sanitizeRagSource,
} from "./mirrored-chat-settings.ts";

/** "full" (Full access) is session-only, so a thread never carries it. */
export type ThreadPermissionMode = Exclude<PermissionMode, "full">;

export interface ThreadScopedSettings {
  reasoningEnabled?: boolean;
  reasoningEffort?: ReasoningEffort;
  toolsEnabled?: boolean;
  codeToolsEnabled?: boolean;
  imageToolsEnabled?: boolean;
  webFetchToolsEnabled?: boolean;
  deepResearchEnabled?: boolean;
  artifactsEnabled?: boolean;
  mcpEnabledForChat?: boolean;
  permissionMode?: ThreadPermissionMode;
  ragEnabled?: boolean;
  ragSource?: RagSource;
  ragMode?: RagMode;
  ragTopK?: number;
  ragAutoInject?: RagAutoInject;
  ragAutoInjectMinScore?: number;
}

const THREAD_SCOPED_BOOLEAN_KEYS = [
  "reasoningEnabled",
  "toolsEnabled",
  "codeToolsEnabled",
  "imageToolsEnabled",
  "webFetchToolsEnabled",
  "deepResearchEnabled",
  "artifactsEnabled",
  "mcpEnabledForChat",
  "ragEnabled",
] as const satisfies readonly (keyof ThreadScopedSettings)[];

const THREAD_SCOPED_ENUM_VALUES = {
  reasoningEffort: ["none", "minimal", "low", "medium", "high", "max", "xhigh"],
  permissionMode: ["ask", "auto", "off"],
  ragMode: ["hybrid", "lexical", "dense"],
  ragAutoInject: ["auto", "on", "off"],
} as const satisfies Partial<
  Record<keyof ThreadScopedSettings, readonly string[]>
>;

// bounds match the ge/le PATCH /api/chat/threads/{id} enforces on the same fields.
const THREAD_SCOPED_NUMBER_BOUNDS = {
  ragTopK: { min: 1, max: 50, integer: true },
  ragAutoInjectMinScore: { min: 0, max: 1, integer: false },
} as const satisfies Partial<
  Record<
    keyof ThreadScopedSettings,
    { min: number; max: number; integer: boolean }
  >
>;

export const THREAD_SCOPED_SETTING_KEYS = [
  ...THREAD_SCOPED_BOOLEAN_KEYS,
  ...(Object.keys(
    THREAD_SCOPED_ENUM_VALUES,
  ) as (keyof typeof THREAD_SCOPED_ENUM_VALUES)[]),
  ...(Object.keys(
    THREAD_SCOPED_NUMBER_BOUNDS,
  ) as (keyof typeof THREAD_SCOPED_NUMBER_BOUNDS)[]),
  "ragSource",
] as const satisfies readonly (keyof ThreadScopedSettings)[];

export type ThreadScopedSettingKey =
  (typeof THREAD_SCOPED_SETTING_KEYS)[number];

const THREAD_SCOPED_SETTING_KEY_SET: ReadonlySet<string> = new Set(
  THREAD_SCOPED_SETTING_KEYS,
);

export function isThreadScopedSettingKey(
  key: string,
): key is ThreadScopedSettingKey {
  return THREAD_SCOPED_SETTING_KEY_SET.has(key);
}

// derived on apply, so unstored, but loadPermissionMode falls back to the confirm toggle: writing
// it globally would turn one chat's permission level into every other browser's default.
const THREAD_DERIVED_SETTING_KEYS: ReadonlySet<string> = new Set([
  "confirmToolCalls",
]);

/** whether an edit to `key` belongs to the open chat rather than the installation. */
export function isThreadOwnedSettingKey(key: string): boolean {
  return (
    THREAD_SCOPED_SETTING_KEY_SET.has(key) ||
    THREAD_DERIVED_SETTING_KEYS.has(key)
  );
}

// the patch model is extra="forbid", so one out-of-contract field would 400 the whole write.
export function sanitizeThreadScopedSettings(
  value: unknown,
): ThreadScopedSettings {
  const settings: ThreadScopedSettings = {};
  if (!isRecord(value)) return settings;
  const target = settings as Record<string, unknown>;
  for (const key of THREAD_SCOPED_BOOLEAN_KEYS) {
    if (typeof value[key] === "boolean") target[key] = value[key];
  }
  for (const [key, allowed] of Object.entries(THREAD_SCOPED_ENUM_VALUES)) {
    const candidate = value[key];
    if (
      typeof candidate === "string" &&
      (allowed as readonly string[]).includes(candidate)
    ) {
      target[key] = candidate;
    }
  }
  for (const [key, bounds] of Object.entries(THREAD_SCOPED_NUMBER_BOUNDS)) {
    const sanitized = sanitizeBoundedNumber(value[key], bounds);
    if (sanitized !== undefined) target[key] = sanitized;
  }
  const ragSource = sanitizeRagSource(value.ragSource);
  if (ragSource) settings.ragSource = ragSource;
  return settings;
}

/** whether `settings` carries any thread-scoped value at all. */
export function hasThreadScopedSettings(
  settings: ThreadScopedSettings | null | undefined,
): boolean {
  if (!settings) return false;
  return THREAD_SCOPED_SETTING_KEYS.some((key) => settings[key] !== undefined);
}
