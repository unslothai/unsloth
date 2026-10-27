// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Separate module with injected deps: loaded-models-api.ts reaches the chat store and
// cannot be imported by node --test.

// Extension-qualified so node --test resolves it.
import { unloadResident } from "../api-monitor/unload-resident.ts";

export type ResidentChatModel = {
  checkpoint: string;
  /** Status reports the load path, the store the repo id. */
  aliases: string[];
};

export type EjectChatModelDeps = {
  readResident: () => Promise<ResidentChatModel | null>;
  unload: (modelPath: string) => Promise<void>;
  matches: (left: string, right: string) => boolean;
  cachedRow?: boolean;
  readCached?: () => Promise<string[]>;
};

export type EjectChatModelResult = {
  unloadedAliases: string[];
  stillResident: string | null;
  replacedBy: string | null;
};

/**
 * Read, unload, re-read: /unload naming a replaced model is a 200 no-op, and the read is
 * scoped so a model switched in before the click is never unloaded.
 */
export async function ejectChatModel(
  target: string,
  deps: EjectChatModelDeps,
): Promise<EjectChatModelResult> {
  const seen: { resident: ResidentChatModel | null } = { resident: null };
  const { unloadedAliases, stillResident } = await unloadResident({
    readResident: async () => {
      const resident = await deps.readResident();
      seen.resident = resident;
      if (!resident) return null;
      const namesTarget = resident.aliases.some((alias) =>
        deps.matches(target, alias),
      );
      return namesTarget ? resident : null;
    },
    unload: deps.unload,
  });
  if (unloadedAliases.length > 0) {
    return { unloadedAliases, stillResident, replacedBy: null };
  }
  // Transformers keeps a replaced model in backend.models (reported as `loaded`), so check
  // what the runtime really holds before calling the row somebody else's.
  const holdsTarget = (names: string[] | undefined) =>
    names?.some((name) => deps.matches(target, name)) ?? false;
  // Re-read even for a cached row: another tab may have evicted it since the card polled, and
  // unloading a name nobody holds would still stop the active model's queues. A failed read
  // keeps the row's own word, as before.
  const cachedNow = await deps.readCached?.().catch(() => undefined);
  if (cachedNow === undefined ? deps.cachedRow : holdsTarget(cachedNow)) {
    // A cached row is never the active model, so the scoped read can never
    // match it. Name it directly: the Transformers backend can still hold it.
    await deps.unload(target);
    // /unload answers 200 for a name no longer held, so confirm with a re-read.
    const survived = holdsTarget(await deps.readCached?.());
    return {
      unloadedAliases: survived ? [] : [target],
      stillResident: survived ? target : null,
      replacedBy: null,
    };
  }
  // Nothing unloaded: /unload naming a replaced model is a 200 no-op, so do not fire it.
  return {
    unloadedAliases: [],
    stillResident: null,
    replacedBy: seen.resident?.checkpoint ?? null,
  };
}
