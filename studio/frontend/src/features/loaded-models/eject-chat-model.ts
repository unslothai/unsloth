// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The eject sequence for one chat row. Its own plain module, taking its I/O as
// deps, because loaded-models-api.ts reaches the chat store and cannot be
// imported by the node --test suite. Mirrors api-monitor/unload-resident.ts.

// Extension-qualified so node --test resolves it, as merge-task-iterators.ts does.
import { unloadResident } from "../api-monitor/unload-resident.ts";

/** What /api/inference/status says is resident, already resolved. */
export type ResidentChatModel = {
  /** The id /unload matches on. */
  checkpoint: string;
  /** Every spelling of it: status reports the load path, the store the repo id. */
  aliases: string[];
};

export type EjectChatModelDeps = {
  readResident: () => Promise<ResidentChatModel | null>;
  unload: (modelPath: string) => Promise<void>;
  /** modelIdsMatch, injected so this module stays free of path aliases. */
  matches: (left: string, right: string) => boolean;
};

export type EjectChatModelResult = {
  /** Every spelling this run released, for clearing the picker selection. */
  unloadedAliases: string[];
  /** The target if it survived the run, else null. */
  stillResident: string | null;
};

/**
 * Release the model `target` names, and only that one.
 *
 * Read, unload, re-read, as the API monitor's Unload does, since an API
 * auto-switch can replace the model mid-eject and /unload naming a replaced
 * model is a documented 200 no-op. Unlike that single button, this is a per-row
 * control, so the read is scoped: a switch that landed before the click leaves
 * another model active, and unloading that frees something nobody asked to.
 */
export async function ejectChatModel(
  target: string,
  deps: EjectChatModelDeps,
): Promise<EjectChatModelResult> {
  const { unloadedAliases, stillResident } = await unloadResident({
    readResident: async () => {
      const resident = await deps.readResident();
      if (!resident) return null;
      const namesTarget = resident.aliases.some((alias) =>
        deps.matches(target, alias),
      );
      return namesTarget ? resident : null;
    },
    unload: deps.unload,
  });
  if (unloadedAliases.length > 0) {
    return { unloadedAliases, stillResident };
  }
  // Never active during this eject: a cached row, or a switch beat the click.
  // Name it directly, since the Transformers backend can still hold it.
  await deps.unload(target);
  return { unloadedAliases: [target], stillResident: null };
}
