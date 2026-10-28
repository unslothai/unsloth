// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";

export type SpookyWelcome = {
  key: TranslationKey;
  name?: string;
  sloth: string;
  enlarge?: boolean;
};

const TRICK_OR_TREAT = "sloth halloween trick or treat.png";
const WITCH = "sloth halloween witch.png";
const DEAD = "sloth halloween dead.png";
const VAMPIRE = "sloth halloween vampire.png";
const CHEEKY = "large sloth cheeky.png";
const SHOCK = "sloth shock large.png";
// Wide drawings fill less of the square canvas, so they render slightly larger.
const ENLARGED = new Set([TRICK_OR_TREAT, WITCH, VAMPIRE]);

// [key, key with {name} or null, sloth pictures to pick from]
type Line = [TranslationKey, TranslationKey | null, string[]];

const LINES: Line[] = [
  ["halloween.greeting.pumpkins", null, [TRICK_OR_TREAT]],
  ["halloween.greeting.brewing", null, [WITCH]],
  [
    "halloween.greeting.trickOrTreat",
    null,
    [TRICK_OR_TREAT, CHEEKY, DEAD, SHOCK],
  ],
  ["halloween.greeting.manor", "halloween.greeting.manorName", [VAMPIRE]],
  ["halloween.greeting.spirits", null, [SHOCK]],
  [
    "halloween.greeting.mysteries",
    null,
    [TRICK_OR_TREAT, WITCH, DEAD, VAMPIRE, SHOCK, CHEEKY],
  ],
  ["halloween.greeting.conjure", null, [WITCH]],
];

const HALLOWEEN: Line = [
  "halloween.greeting.happyHalloween",
  "halloween.greeting.happyHalloweenName",
  [TRICK_OR_TREAT, WITCH, DEAD, VAMPIRE, SHOCK],
];

const pick = <T>(items: T[]): T =>
  items[Math.floor(Math.random() * items.length)];

/** On October 30 and 31 (local time) Happy Halloween joins the lines a new chat picks from. */
export function buildSpookyWelcome(
  name: string,
  now: Date = new Date(),
): SpookyWelcome {
  const pool =
    now.getMonth() === 9 && now.getDate() >= 30 ? [...LINES, HALLOWEEN] : LINES;
  const [key, namedKey, sloths] = pick(pool);
  const sloth = pick(sloths);
  const enlarge = ENLARGED.has(sloth);
  return name && namedKey
    ? { key: namedKey, name, sloth, enlarge }
    : { key, sloth, enlarge };
}
