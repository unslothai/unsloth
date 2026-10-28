// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";

export type SpookyWelcome = {
  key: TranslationKey;
  name?: string;
  sloth: string;
};

type Line = [TranslationKey, TranslationKey | null, string];

// [key, key with {name} or null, sloth picture]
const LINES: Line[] = [
  [
    "halloween.greeting.boo",
    "halloween.greeting.booName",
    "sloth shock large.png",
  ],
  ["halloween.greeting.trickOrTreat", null, "large sloth cheeky.png"],
  ["halloween.greeting.brewing", null, "large sloth fire.png"],
  ["halloween.greeting.ghosts", null, "sloth shy large.png"],
];

const HALLOWEEN: Line = [
  "halloween.greeting.happyHalloween",
  "halloween.greeting.happyHalloweenName",
  "large sloth yay.png",
];

/** October 30 and 31 (local time) always say Happy Halloween; other days pick a line. */
export function buildSpookyWelcome(
  name: string,
  now: Date = new Date(),
): SpookyWelcome {
  const [key, namedKey, sloth] =
    now.getMonth() === 9 && now.getDate() >= 30
      ? HALLOWEEN
      : LINES[Math.floor(Math.random() * LINES.length)];
  return name && namedKey ? { key: namedKey, name, sloth } : { key, sloth };
}
