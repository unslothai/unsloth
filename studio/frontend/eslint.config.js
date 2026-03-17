// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import { defineConfig, globalIgnores } from "eslint/config";
import globals from "globals";
import tseslint from "typescript-eslint";

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
      // Import restrictions for architecture enforcement
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            // Prevent cross-feature imports
            {
              group: ["@/features/*/*"],
              message: "Import from feature index only: @/features/[name]",
            },
            // Prevent app layer from importing features internals
            {
              group: ["../features/*/**"],
              message: "Use absolute imports: @/features/[name]",
            },
          ],
        },
      ],
    },
  },
]);
