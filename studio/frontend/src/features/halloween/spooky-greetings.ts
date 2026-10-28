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
  ["halloween.greeting.pumpkins", null, "large sloth yay.png"],
  [
    "halloween.greeting.brewing",
    "halloween.greeting.brewingName",
    "large sloth fire.png",
  ],
  ["halloween.greeting.trickOrTreat", null, "large sloth cheeky.png"],
  [
    "halloween.greeting.manor",
    "halloween.greeting.manorName",
    "sloth sir large.png",
  ],
];

const HALLOWEEN: Line = [
  "halloween.greeting.happyHalloween",
  "halloween.greeting.happyHalloweenName",
  "large sloth wave.png",
];

/** On October 30 and 31 (local time) Happy Halloween joins the lines a new chat picks from. */
export function buildSpookyWelcome(
  name: string,
  now: Date = new Date(),
): SpookyWelcome {
  const pool =
    now.getMonth() === 9 && now.getDate() >= 30 ? [...LINES, HALLOWEEN] : LINES;
  const [key, namedKey, sloth] = pool[Math.floor(Math.random() * pool.length)];
  return name && namedKey ? { key: namedKey, name, sloth } : { key, sloth };
}
