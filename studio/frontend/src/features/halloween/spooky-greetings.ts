// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";

export type SpookyWelcome = {
  key: TranslationKey;
  name?: string;
  sloth: string;
};

// [key, key with {name} or null, sloth picture]
const LINES: [TranslationKey, TranslationKey | null, string][] = [
  [
    "halloween.greeting.boo",
    "halloween.greeting.booName",
    "sloth shock large.png",
  ],
  ["halloween.greeting.trickOrTreat", null, "large sloth cheeky.png"],
  ["halloween.greeting.brewing", null, "large sloth fire.png"],
  ["halloween.greeting.pumpkins", null, "large sloth yay.png"],
  ["halloween.greeting.ghosts", null, "sloth shy large.png"],
  ["halloween.greeting.manor", null, "sloth sir large.png"],
  ["halloween.greeting.onlyTreats", null, "large sloth heart.png"],
  ["halloween.greeting.moon", null, "large sloth glasses.png"],
  ["halloween.greeting.skeletons", null, "Large sloth Question mark.png"],
  [
    "halloween.greeting.haunting",
    "halloween.greeting.hauntingName",
    "large sloth wave.png",
  ],
  ["halloween.greeting.candyBag", null, "large sloth drink.png"],
  ["halloween.greeting.broomstick", null, "large sloth thumbs.png"],
];

export function buildSpookyWelcome(name: string): SpookyWelcome {
  const [key, namedKey, sloth] =
    LINES[Math.floor(Math.random() * LINES.length)];
  return name && namedKey ? { key: namedKey, name, sloth } : { key, sloth };
}
