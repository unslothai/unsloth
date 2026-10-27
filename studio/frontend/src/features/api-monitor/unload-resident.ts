// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// /unload matches the internal id, so read it from status first. Plain module for node --test.

export type ResidentModel = {
  checkpoint: string;
  aliases: string[];
};

export type UnloadResidentDeps = {
  readResident: () => Promise<ResidentModel | null>;
  unload: (checkpoint: string) => Promise<void>;
};

export type UnloadResidentResult = {
  unloadedAliases: string[];
  stillResident: string | null;
};

// One retry: an auto-switch can replace the model between read and unload, and /unload on a
// replaced model is a 200 no-op.
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
