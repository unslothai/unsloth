// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A bounded cache of Shiki tokenisations, keyed on the WHOLE source.
//
// It exists because `@streamdown/code` 1.1.1 keeps its own tokenisation Map forever: no size cap,
// no clear, no unmount hook, and a key built from the source's length plus its first and last 100
// characters. A STREAMED fence reaches the highlighter once per refresh window with a longer
// prefix each time, so one reply mints one full dual-theme tokenisation per window and every one
// of them survives for the life of the tab. Measured on a 32 KB python fence streamed over about
// five seconds: +10.8 MB of retained V8 heap per fence, still there after the reply is unmounted
// and the heap is force-collected, against +0.64 MB for the same fence delivered in one update.
//
// Two properties matter, and both are load-bearing:
//
//   PREFIX EVICTION  every earlier frame of a growing fence is a prefix of the current one, so
//                    storing frame N drops frames 1..N-1 of the same fence immediately. A
//                    streaming reply occupies ONE entry while it streams, not one per window.
//                    A size cap alone would not do this: the prefixes would sit in the cache
//                    until later replies pushed them out one at a time. It is ONE-DIRECTIONAL on
//                    purpose; see the comment at the eviction itself.
//   A CHARACTER CAP  tokens cost roughly 17x their source in retained heap, so a budget in
//                    characters of source is a budget in megabytes. A count cap alone would let
//                    64 large fences pin far more than 64 small ones.

export type TokenCacheEntry<T> = {
  /** Group the entry belongs to. Only entries in the same group evict each other. */
  group: string;
  /** The source that produced `result`. Prefix eviction compares against this. */
  code: string;
  result: T;
};

export type TokenCache<T> = {
  get: (group: string, code: string) => T | null;
  set: (group: string, code: string, result: T) => void;
  stats: () => { entries: number; chars: number };
};

export type TokenCacheOptions = {
  /** Characters of source held across all entries. */
  maxChars: number;
  /** Hard entry count, so a thread of tiny fences cannot grow the Map without bound. */
  maxEntries: number;
};

// A group must not be confusable with a source that happens to contain the separator, so the key
// joins on a character no markdown fence can hold.
const SEPARATOR = "\u0000";

export function createTokenCache<T>(options: TokenCacheOptions): TokenCache<T> {
  // Insertion-ordered, and every read re-inserts, so the first key is the least recently used.
  const entries = new Map<string, TokenCacheEntry<T>>();
  let chars = 0;

  const drop = (key: string, entry: TokenCacheEntry<T>) => {
    entries.delete(key);
    chars -= entry.code.length;
  };

  const trim = () => {
    while (
      entries.size > options.maxEntries ||
      (chars > options.maxChars && entries.size > 1)
    ) {
      const oldestKey = entries.keys().next().value;
      if (oldestKey === undefined) return;
      const oldest = entries.get(oldestKey);
      if (!oldest) return;
      drop(oldestKey, oldest);
    }
  };

  return {
    get: (group, code) => {
      const key = `${group}${SEPARATOR}${code}`;
      const entry = entries.get(key);
      if (!entry) return null;
      entries.delete(key);
      entries.set(key, entry);
      return entry.result;
    },
    set: (group, code, result) => {
      const key = `${group}${SEPARATOR}${code}`;
      const existing = entries.get(key);
      if (existing) drop(key, existing);
      // Deleting the current entry while iterating a Map is defined behaviour and does not skip
      // the rest, so this walk is safe.
      for (const [otherKey, other] of entries) {
        if (other.group !== group) continue;
        // ONE DIRECTION ONLY: drop what the new code EXTENDS, never what extends it.
        //
        // Evicting both ways live-locks the app. Two fences can be on screen at once where one is
        // a prefix of the other, which is what a page showing the same snippet at two lengths
        // does. Storing the long one would evict the short one, the short one's next render would
        // miss and evict the long one, and so on: a miss returns null, which schedules an
        // asynchronous tokenisation, whose callback re-renders, which misses again. Measured on
        // the pair below before this was one-directional: two misses per round, forever, with the
        // cache stuck at one entry. It hung a CI job for thirty minutes.
        //
        // The direction kept here is the one the memory win needs. Every earlier frame of a
        // GROWING fence is a prefix of the current one, so this still collapses a streamed reply
        // onto one entry. The reverse case, a source that SHRINKS, is left to the size cap.
        if (code.startsWith(other.code)) {
          drop(otherKey, other);
        }
      }
      entries.set(key, { group, code, result });
      chars += code.length;
      trim();
    },
    stats: () => ({ entries: entries.size, chars }),
  };
}
