// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import { defineConfig, globalIgnores } from "eslint/config";
import globals from "globals";
import tseslint from "typescript-eslint";

const restrictFeatureImports = (patterns) => [
  "error",
  {
    patterns: patterns.map(({ group, message }) => ({ group, message })),
  },
];

export default defineConfig([
  globalIgnores(["dist", "**/._*"]),
  {
    files: ["**/*.{ts,tsx}"],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    rules: {
      // Allow shadcn ui components to export variants
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true },
      ],
    },
  },
  {
    files: ["src/features/chat/**/*.{ts,tsx}"],
    rules: {
      "no-restricted-imports": restrictFeatureImports([
        {
          group: ["@/features/models", "@/features/models/**"],
          message:
            "Chat must use shared lib/inventory/download modules instead of depending on Hub.",
        },
      ]),
    },
  },
  {
    files: ["src/features/training/**/*.{ts,tsx}"],
    rules: {
      "no-restricted-imports": restrictFeatureImports([
        {
          group: ["@/features/models", "@/features/models/**"],
          message:
            "Training must use shared lib/inventory/download modules instead of depending on Hub.",
        },
      ]),
    },
  },
  {
    files: ["src/features/models/**/*.{ts,tsx}"],
    rules: {
      "no-restricted-imports": restrictFeatureImports([
        {
          group: ["@/features/chat/**", "@/features/training/**"],
          message:
            "Hub may use public feature indexes only; shared primitives belong in lib/inventory/download modules.",
        },
      ]),
    },
  },
]);
