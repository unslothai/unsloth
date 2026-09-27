// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Color themes (the `palette` preference). Tokens live in index.css under
// html[data-palette]. Keep ids in sync with public/theme-boot.js and the
// backend `palette` Literal in routes/settings.py.

export const UNSLOTH_THEME_IDS = ["standard", "classic", "minimal"] as const;

export const FLAVOR_THEME_IDS = [
  "matcha",
  "blueberry",
  "tangerine",
  "plum",
  "neon-cyberpunk",
  "cherry",
  "mint",
  "honey",
  "taro",
  "espresso",
  "butterfly-pea",
  "macaron",
  "wasabi",
  "earl-grey",
  "peach",
  "licorice",
  "yuzu",
  "dragon-fruit",
  "oat-milk",
  "cinnamon",
  "cotton-candy",
] as const;

export const COLOR_THEME_IDS = [
  ...UNSLOTH_THEME_IDS,
  ...FLAVOR_THEME_IDS,
] as const;

export type ColorThemeId = (typeof COLOR_THEME_IDS)[number];

export type ThemeModeColors = {
  accent: string;
  background: string;
  foreground: string;
  /** Flavor themes only: card and menu color. */
  surface?: string;
};

type ColorThemeMeta = {
  /** Flavor themes only; Unsloth themes use i18n keys. */
  name?: string;
  light: ThemeModeColors;
  dark: ThemeModeColors;
};

export const COLOR_THEMES: Record<ColorThemeId, ColorThemeMeta> = {
  standard: {
    light: { accent: "#17b88b", background: "#fefefd", foreground: "#262626" },
    dark: { accent: "#17b88b", background: "#181818", foreground: "#dfdfdf" },
  },
  classic: {
    light: { accent: "#339cff", background: "#ffffff", foreground: "#1a1c1f" },
    dark: { accent: "#4dabff", background: "#181818", foreground: "#dfdfdf" },
  },
  minimal: {
    light: { accent: "#171717", background: "#ffffff", foreground: "#171717" },
    dark: { accent: "#ededed", background: "#181818", foreground: "#dfdfdf" },
  },
  matcha: {
    name: "Matcha",
    light: {
      accent: "#437a22",
      background: "#f6f8f1",
      foreground: "#1d2618",
      surface: "#eef2e6",
    },
    dark: {
      accent: "#8cc265",
      background: "#151a15",
      foreground: "#e1ead9",
      surface: "#1b221b",
    },
  },
  blueberry: {
    name: "Blueberry",
    light: {
      accent: "#4a5bdc",
      background: "#f6f7fc",
      foreground: "#1c2140",
      surface: "#eef0f9",
    },
    dark: {
      accent: "#7c8cff",
      background: "#141726",
      foreground: "#dde2f5",
      surface: "#1b1f33",
    },
  },
  tangerine: {
    name: "Tangerine Dreams",
    light: {
      accent: "#e36a00",
      background: "#fdf8f3",
      foreground: "#2f1f10",
      surface: "#f9efe5",
    },
    dark: {
      accent: "#ff8c2b",
      background: "#1b1510",
      foreground: "#f3e5d6",
      surface: "#241c15",
    },
  },
  plum: {
    name: "Plum Praline",
    light: {
      accent: "#8b3fd1",
      background: "#fbf7fd",
      foreground: "#2a1836",
      surface: "#f4ecf8",
    },
    dark: {
      accent: "#c77dff",
      background: "#1a1320",
      foreground: "#eadff2",
      surface: "#22192a",
    },
  },
  "neon-cyberpunk": {
    name: "Neon Cyberpunk",
    light: {
      accent: "#0a0a0f",
      background: "#f6f6f1",
      foreground: "#0a0a0f",
      surface: "#ecece5",
    },
    dark: {
      accent: "#fcee0a",
      background: "#070b14",
      foreground: "#eafcff",
      surface: "#0d1424",
    },
  },
  cherry: {
    name: "Cherry Cola",
    light: {
      accent: "#c8102e",
      background: "#fdf6f6",
      foreground: "#300f12",
      surface: "#f9ebeb",
    },
    dark: {
      accent: "#ff4d5e",
      background: "#1b1112",
      foreground: "#f4dfe0",
      surface: "#241718",
    },
  },
  mint: {
    name: "Mint",
    light: {
      accent: "#0f9e80",
      background: "#f3f9f7",
      foreground: "#12302a",
      surface: "#e9f4f1",
    },
    dark: {
      accent: "#4fd1b3",
      background: "#111b1a",
      foreground: "#dcefeb",
      surface: "#172322",
    },
  },
  honey: {
    name: "Honey",
    light: {
      accent: "#c98a00",
      background: "#fdf9ef",
      foreground: "#2a2310",
      surface: "#f8f1de",
    },
    dark: {
      accent: "#f2b631",
      background: "#1a1710",
      foreground: "#f0e6cf",
      surface: "#231f15",
    },
  },
  taro: {
    name: "Ube Chiffon",
    light: {
      accent: "#7b5fc4",
      background: "#f9f7fc",
      foreground: "#241d33",
      surface: "#f1edf8",
    },
    dark: {
      accent: "#b69cf0",
      background: "#17141f",
      foreground: "#e6e0f2",
      surface: "#1e1a28",
    },
  },
  espresso: {
    name: "Espresso Amore",
    light: {
      accent: "#8a5a33",
      background: "#faf6f2",
      foreground: "#2b201a",
      surface: "#f3ece5",
    },
    dark: {
      accent: "#d69a5e",
      background: "#1c1613",
      foreground: "#eadfd6",
      surface: "#251d19",
    },
  },
  "butterfly-pea": {
    name: "Butterfly Pea",
    light: {
      accent: "#0077c8",
      background: "#f4f9fd",
      foreground: "#0f2233",
      surface: "#e9f2fa",
    },
    dark: {
      accent: "#38b6ff",
      background: "#0f1520",
      foreground: "#dcebf8",
      surface: "#141c2a",
    },
  },
  macaron: {
    name: "Macaron",
    light: {
      accent: "#d61f55",
      background: "#fdf6f8",
      foreground: "#33121d",
      surface: "#f9ecf0",
    },
    dark: {
      accent: "#ff4f7b",
      background: "#1d1216",
      foreground: "#f3e0e6",
      surface: "#26181d",
    },
  },
  wasabi: {
    name: "Wasabi",
    light: {
      accent: "#4f7d00",
      background: "#f8faf2",
      foreground: "#1a2410",
      surface: "#eef3e3",
    },
    dark: {
      accent: "#b4e33d",
      background: "#121710",
      foreground: "#e3eed8",
      surface: "#181f15",
    },
  },
  "earl-grey": {
    name: "Earl Grey",
    light: {
      accent: "#4f6b8a",
      background: "#f6f7f9",
      foreground: "#1c2129",
      surface: "#eceff3",
    },
    dark: {
      accent: "#9db4cf",
      background: "#15171b",
      foreground: "#e2e6ec",
      surface: "#1c1f24",
    },
  },
  peach: {
    name: "Peach",
    light: {
      accent: "#e0703d",
      background: "#fdf7f3",
      foreground: "#33211a",
      surface: "#f9eee7",
    },
    dark: {
      accent: "#ffa07a",
      background: "#1c1614",
      foreground: "#f2e4dc",
      surface: "#251d1a",
    },
  },
  licorice: {
    name: "Licorice",
    light: {
      accent: "#d12d34",
      background: "#fafafa",
      foreground: "#111114",
      surface: "#f2f2f3",
    },
    dark: {
      accent: "#e5484d",
      background: "#0e0e10",
      foreground: "#ececf1",
      surface: "#16161a",
    },
  },
  yuzu: {
    name: "Yuzu Citrus",
    light: {
      accent: "#d4c400",
      background: "#fcfcf2",
      foreground: "#22260f",
      surface: "#f5f6e4",
    },
    dark: {
      accent: "#e8dd3a",
      background: "#16170f",
      foreground: "#eeefd8",
      surface: "#1e2015",
    },
  },
  "dragon-fruit": {
    name: "Dragon Fruit",
    light: {
      accent: "#c2127f",
      background: "#fdf6fa",
      foreground: "#300c24",
      surface: "#f8eaf3",
    },
    dark: {
      accent: "#ff3da8",
      background: "#1c0f18",
      foreground: "#f8e0ef",
      surface: "#261420",
    },
  },
  "oat-milk": {
    name: "Oat Milk",
    light: {
      accent: "#86683f",
      background: "#fbf8f3",
      foreground: "#2e2921",
      surface: "#f4efe7",
    },
    dark: {
      accent: "#d9b98c",
      background: "#1d1b18",
      foreground: "#ece6dc",
      surface: "#25221e",
    },
  },
  cinnamon: {
    name: "Cinnamon Roll",
    light: {
      accent: "#a8461f",
      background: "#fcf7f4",
      foreground: "#2e1a12",
      surface: "#f6ece6",
    },
    dark: {
      accent: "#e8804f",
      background: "#1b1411",
      foreground: "#f0e1d8",
      surface: "#241b17",
    },
  },
  "cotton-candy": {
    name: "Cotton Candy",
    light: {
      accent: "#c2378f",
      background: "#fdf8fb",
      foreground: "#2a1830",
      surface: "#f7eef6",
    },
    dark: {
      accent: "#ff8fd0",
      background: "#1a1420",
      foreground: "#f5e4f4",
      surface: "#221a29",
    },
  },
};

const COLOR_THEME_ID_SET: ReadonlySet<string> = new Set(COLOR_THEME_IDS);

export function isColorThemeId(value: unknown): value is ColorThemeId {
  return typeof value === "string" && COLOR_THEME_ID_SET.has(value);
}
