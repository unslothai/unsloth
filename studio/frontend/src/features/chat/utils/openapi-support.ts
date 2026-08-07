// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Whether the served schema declares the conditional title patch.
 *
 *  The desktop app ships its own frontend against a separately installed
 *  backend, and an older one drops `expectedTitle` and applies the write
 *  unguarded. Anything unrecognised reads as unsupported. */
export function schemaDeclaresExpectedTitle(document: unknown): boolean {
  if (typeof document !== "object" || document === null) return false;
  const components = (document as { components?: unknown }).components;
  if (typeof components !== "object" || components === null) return false;
  const schemas = (components as { schemas?: unknown }).schemas;
  if (typeof schemas !== "object" || schemas === null) return false;
  const patch = (schemas as Record<string, unknown>).ChatThreadPatch;
  if (typeof patch !== "object" || patch === null) return false;
  const properties = (patch as { properties?: unknown }).properties;
  if (typeof properties !== "object" || properties === null) return false;
  return "expectedTitle" in (properties as Record<string, unknown>);
}

export interface GuardProbe {
  supported: boolean;
  /** Only a schema that arrived and parsed settles the question. Anything else
   *  is a moment in time: a 401 while the token warms up, a 503 during startup.
   *  Remembering those would park the migration for the whole session. */
  settled: boolean;
}

export function readGuardProbe(ok: boolean, document: unknown): GuardProbe {
  if (!ok) return { supported: false, settled: false };
  return { supported: schemaDeclaresExpectedTitle(document), settled: true };
}
