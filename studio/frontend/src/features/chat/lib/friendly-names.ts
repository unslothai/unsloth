// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Friendly default names for auto-created OpenAI shell containers, used by the
 * chat-adapter's lazy-create path (Code pill on, no thread container, non-default
 * TTL). Goal: a memorable label like "otter" instead of "chat-abc12345"; users
 * can still rename via the Unsloth alias map.
 *
 * The list is curated to be unambiguous, non-offensive nouns from natural
 * categories (animals, plants, geography, materials, weather), avoid
 * technical/political/brand words, and stay small (~1.5 KB).
 *
 * Collisions are tolerated: the real unique key is the ``cntr_*`` id, not the
 * name. A short random hex suffix keeps same-name picks visually distinct.
 */

const WORDS = [
  // animals
  "otter",
  "falcon",
  "heron",
  "lynx",
  "marten",
  "stoat",
  "raven",
  "magpie",
  "salmon",
  "trout",
  "perch",
  "tortoise",
  "gecko",
  "iguana",
  "axolotl",
  "narwhal",
  "manatee",
  "dolphin",
  "porpoise",
  "octopus",
  "cuttlefish",
  "nautilus",
  "starfish",
  "urchin",
  "anemone",
  "coral",
  "puffin",
  "kestrel",
  "osprey",
  "buzzard",
  "kingfisher",
  "robin",
  "wren",
  "finch",
  "sparrow",
  "thrush",
  "siskin",
  "warbler",
  "tanager",
  "oriole",
  "hare",
  "badger",
  "weasel",
  "ferret",
  "polecat",
  "civet",
  "tapir",
  "okapi",
  "ibex",
  "chamois",
  // plants & trees
  "alder",
  "aspen",
  "birch",
  "cedar",
  "cypress",
  "elder",
  "elm",
  "fir",
  "ginkgo",
  "hawthorn",
  "hazel",
  "hemlock",
  "holly",
  "juniper",
  "larch",
  "linden",
  "maple",
  "oak",
  "olive",
  "pine",
  "rowan",
  "spruce",
  "sycamore",
  "willow",
  "yew",
  "thistle",
  "fern",
  "moss",
  "ivy",
  "clover",
  "heather",
  "lavender",
  "rosemary",
  "sage",
  "thyme",
  "myrtle",
  "laurel",
  "magnolia",
  // geography / landscape
  "harbor",
  "atoll",
  "lagoon",
  "estuary",
  "fjord",
  "delta",
  "isthmus",
  "mesa",
  "plateau",
  "valley",
  "ridge",
  "summit",
  "glade",
  "meadow",
  "moor",
  "heath",
  "tundra",
  "savanna",
  "prairie",
  "steppe",
  "bayou",
  "marsh",
  "fen",
  "grotto",
  "cavern",
  "canyon",
  "ravine",
  "gorge",
  "knoll",
  "dell",
  "vale",
  "coast",
  // materials / minerals / colors
  "amber",
  "agate",
  "onyx",
  "opal",
  "jade",
  "quartz",
  "obsidian",
  "basalt",
  "granite",
  "marble",
  "slate",
  "flint",
  "lapis",
  "topaz",
  "garnet",
  "pearl",
  "coral",
  "ivory",
  "ebony",
  "copper",
  "cobalt",
  "indigo",
  "saffron",
  "vermilion",
  "ochre",
  "umber",
  "sienna",
  "russet",
  // weather / sky / time
  "aurora",
  "comet",
  "ember",
  "frost",
  "gale",
  "harvest",
  "monsoon",
  "nebula",
  "solstice",
  "twilight",
  "zephyr",
  "drizzle",
  "tempest",
  "halcyon",
  "equinox",
  "rainbow",
  "horizon",
  "meridian",
  "zenith",
  "comet",
  // misc tactile / cozy nouns
  "lantern",
  "kettle",
  "compass",
  "anchor",
  "beacon",
  "harbor",
  "voyage",
  "trellis",
  "cottage",
  "thicket",
  "orchard",
  "bramble",
  "haystack",
  "snowfall",
  "campfire",
];

/** RFC 4122-ish 4-character lowercase hex suffix using crypto.randomUUID. */
function randomHexSuffix(): string {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID().replace(/-/g, "").slice(0, 4);
  }
  // Older-browser fallback. Math.random is fine: display suffix, not a token.
  return Math.floor(Math.random() * 0xffff)
    .toString(16)
    .padStart(4, "0");
}

/**
 * Returns an English-word name with a short random hex suffix (e.g.
 * "kestrel-3f9c"), so repeated words stay distinguishable in the picker.
 */
export function pickFriendlyContainerName(): string {
  const word = WORDS[Math.floor(Math.random() * WORDS.length)] ?? "container";
  return `${word}-${randomHexSuffix()}`;
}
