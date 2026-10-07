// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const GUARD_FIELDS = ["expectedTitle", "expectedOpeningMessageId"] as const;

/** The desktop frontend may run against an older backend that drops these fields. */
export function schemaDeclaresRepairGuards(document: unknown): boolean {
  if (typeof document !== "object" || document === null) return false;
  const components = (document as { components?: unknown }).components;
  if (typeof components !== "object" || components === null) return false;
  const schemas = (components as { schemas?: unknown }).schemas;
  if (typeof schemas !== "object" || schemas === null) return false;
  const patch = (schemas as Record<string, unknown>).ChatThreadPatch;
  if (typeof patch !== "object" || patch === null) return false;
  const properties = (patch as { properties?: unknown }).properties;
  if (typeof properties !== "object" || properties === null) return false;
  const declared = properties as Record<string, unknown>;
  return GUARD_FIELDS.every((field) => field in declared);
}

export interface GuardProbe {
  supported: boolean;
  /** Only a parsed schema settles it; a transient 401 or 503 must not be cached. */
  settled: boolean;
}

export function readGuardProbe(ok: boolean, document: unknown): GuardProbe {
  if (!ok) return { supported: false, settled: false };
  return { supported: schemaDeclaresRepairGuards(document), settled: true };
}
