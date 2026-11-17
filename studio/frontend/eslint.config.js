// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// First, before typescript-eslint loads: it needs the TypeScript 6 API (see the script).
import "./scripts/eslint-typescript6.cjs";

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
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true },
      ],
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              group: ["@/features/*/*"],
              message: "Import from feature index only: @/features/[name]",
            },
            {
              group: ["../features/*/**"],
              message: "Use absolute imports: @/features/[name]",
            },
          ],
        },
      ],
    },
  },
  // The voice domain stays pure: time, settings and messages come in as arguments.
  {
    files: [
      "src/features/chat/voice/{speech-text,tts-chunk-plan,clip-speculation,turn-fuser,heard-cut,voice-instruction,speech-normalize}.ts",
    ],
    rules: {
      "no-restricted-imports": [
        "error",
        {
          paths: [
            { name: "react", message: "Voice domain modules stay free of React." },
            { name: "@/features/auth", message: "Voice domain modules make no requests." },
            { name: "@/lib/api-base", message: "Voice domain modules make no requests." },
          ],
          patterns: [
            {
              group: ["@/features/*/*"],
              message: "Import from feature index only: @/features/[name]",
            },
            {
              group: ["../features/*/**"],
              message: "Use absolute imports: @/features/[name]",
            },
            { group: ["@assistant-ui/*"], message: "Voice domain modules use their own message shape." },
            { group: ["**/stores/**", "**/*-store"], message: "Voice domain modules read no store." },
            { group: ["**/voice-trace"], message: "Voice domain modules do not trace." },
          ],
        },
      ],
      "no-restricted-globals": [
        "error",
        { name: "fetch", message: "Voice domain modules make no requests." },
        { name: "AudioContext", message: "Voice domain modules play nothing." },
      ],
      "no-restricted-properties": [
        "error",
        { object: "performance", property: "now", message: "Pass the time in." },
        { object: "Date", property: "now", message: "Pass the time in." },
      ],
    },
  },
]);
