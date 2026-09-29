// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The monitor's Unload button releases whatever the backend has resident. /unload matches
// on the internal id, which the monitor rows do not carry, so the id has to be read from
// the status first. Its own plain module because the page is .tsx (router, motion, icons)
// and cannot be imported by the node --test suite.

/** The model the backend has resident right now, as the Unload button needs it. */
export type ResidentModel = {
  /** The id POST /api/inference/unload matches on. */
  checkpoint: string;
  /** Every spelling of that same model (load path, advertised repo id), for the store clear. */
  aliases: string[];
};

export type UnloadResidentDeps = {
  /** One GET /api/inference/status, resolved to the resident model or null. */
  readResident: () => Promise<ResidentModel | null>;
  /** One POST /api/inference/unload naming that id. */
  unload: (checkpoint: string) => Promise<void>;
};

export type UnloadResidentResult = {
  /** Aliases of every model this run really unloaded, in the order they were unloaded. */
  unloadedAliases: string[];
  /** A model still resident when the run ended, or null once nothing is loaded. */
  stillResident: string | null;
};

// One retry. An API auto-switch can replace the model between the status read and the
// unload reaching the backend's lifecycle gate, and /unload naming a model a load already
// replaced is a documented 200 no-op, so a single pass reports success with the new model
// still holding the VRAM. A model that arrives after the retry is a fresh load rather than
// this click's target, so the run names it instead of chasing it.
export const UNLOAD_RESIDENT_PASSES = 2;

export async function unloadResident(
  deps: UnloadResidentDeps,
  passes: number = UNLOAD_RESIDENT_PASSES,
): Promise<UnloadResidentResult> {
  const unloadedAliases: string[] = [];
  let resident = await deps.readResident();
  for (let pass = 0; pass < passes && resident !== null; pass += 1) {
    await deps.unload(resident.checkpoint);
    unloadedAliases.push(...resident.aliases);
    // Re-read rather than assume: only the backend knows whether that id matched.
    resident = await deps.readResident();
  }
  return { unloadedAliases, stillResident: resident?.checkpoint ?? null };
}
