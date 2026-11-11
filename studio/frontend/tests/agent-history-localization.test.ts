// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

const locales = [
  "ar", "de", "en", "es", "fr", "he", "hi", "it",
  "ja", "ko", "pt-br", "ru", "sv", "zh-CN",
];

test("the virtual-history jump label exists exactly once in every locale", () => {
  for (const locale of locales) {
    const source = readSrc(`i18n/locales/${locale}.ts`);
    const declarations = source.split("\n").filter((line) =>
      line.includes("agentHistoryJumpToLatest:"));
    assert.equal(declarations.length, 1, `${locale}: missing or duplicated jump label`);
    assert.ok(declarations[0].includes('agentHistoryJumpToLatest: "'));
  }
});

test("the visible virtual-history jump button uses the shared translation key", () => {
  const source = readSrc("components/assistant-ui/agent-history-scroll-messages.tsx");
  assert.ok(source.includes('import { useT } from "@/i18n";'));
  assert.ok(source.includes("const t = useT();"));
  assert.ok(source.includes(
    'onClick={backToLatest}>{t("settings.chat.agentHistoryJumpToLatest")}</button>',
  ));
  assert.ok(!source.includes(">Zur neuesten Ausgabe</button>"));
});
