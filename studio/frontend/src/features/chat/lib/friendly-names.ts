// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Friendly default names for auto-created OpenAI shell containers.
 * Used by the chat-adapter when the lazy-create path fires (Code pill
 * on, no thread container yet, user has set a non-default TTL). The
 * goal is a human-memorable label like "otter" or "harbor" instead of
 * "chat-abc12345" — the user can still rename via the Studio-side
 * alias map.
 *
 * The list is curated to:
 *   - Be unambiguous, non-offensive nouns from natural categories
 *     (animals, plants, geography, materials, weather).
 *   - Avoid technical / political / brand words that might read as
 *     odd in a chat UI.
 *   - Stay reasonably small so the bundle cost is negligible (~200
 *     entries × ~7 bytes ≈ 1.5 KB).
 *
 * Collisions are tolerated — the container's real unique key is its
 * ``cntr_*`` id, not its name. A short random hex suffix is appended
 * to make accidental same-name collisions visually distinct in the
 * picker list.
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
  // Older browser fallback. Math.random is fine here — this is a
  // display suffix, not a security token.
  return Math.floor(Math.random() * 0xffff)
    .toString(16)
    .padStart(4, "0");
}

/**
 * Returns a single English-word name with a short random hex suffix.
 *
 * Example output: "kestrel-3f9c", "harbor-a012".
 *
 * The suffix keeps containers visually distinguishable in the picker
 * when the same word recurs across creations.
 */
export function pickFriendlyContainerName(): string {
  const word = WORDS[Math.floor(Math.random() * WORDS.length)] ?? "container";
  return `${word}-${randomHexSuffix()}`;
}
