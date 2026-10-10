// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

import { modelDisplayName } from "../src/features/hub/lib/model-identity.ts";
import type {
  ExternalConnectionRef,
  ExternalModelRef,
} from "../src/features/model-picker/components/model-selector/missing-external-model.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { compareModelDisplayName, externalModelLabel } = await import(
  "../src/features/chat/lib/external-model-label.ts"
);
const { missingExternalModel } = await import(
  "../src/features/model-picker/components/model-selector/missing-external-model.ts"
);

const CONNECTION_ID = "6235be0905af4221";
const DROPPED_ID = `external::${CONNECTION_ID}::kimi-k2.5`;

const option = (
  modelId: string,
  overrides: Partial<ExternalModelRef> = {},
): ExternalModelRef => ({
  id: `external::${CONNECTION_ID}::${encodeURIComponent(modelId)}`,
  providerId: CONNECTION_ID,
  providerName: "Ollama",
  providerType: "ollama",
  ...overrides,
});

const connection = (
  overrides: Partial<ExternalConnectionRef> = {},
): ExternalConnectionRef => ({
  id: CONNECTION_ID,
  name: "Ollama",
  providerType: "ollama",
  availableModels: ["llama3.2", "kimi-k2.5"],
  ...overrides,
});

const CATALOG_ONLY_ID = "9f1c33d0a7b24e18";
const CATALOG_ONLY_PICK = `external::${CATALOG_ONLY_ID}::gpt-5.4-mini`;

const catalogOnlyOption = (modelId: string): ExternalModelRef => ({
  id: `external::${CATALOG_ONLY_ID}::${encodeURIComponent(modelId)}`,
  providerId: CATALOG_ONLY_ID,
  providerName: "OpenAI",
  providerType: "openai",
});

const catalogOnlyConnection = (
  overrides: Partial<ExternalConnectionRef> = {},
): ExternalConnectionRef => ({
  id: CATALOG_ONLY_ID,
  name: "OpenAI",
  providerType: "openai",
  availableModels: ["gpt-5.4", "gpt-5.4-mini"],
  ...overrides,
});

test("the generic display helper leaves an external id untouched", () => {
  assert.equal(modelDisplayName(DROPPED_ID), DROPPED_ID);
});

test("a dropped connected model is named, never shown as its raw id", () => {
  const missing = missingExternalModel(
    CATALOG_ONLY_PICK,
    [catalogOnlyOption("gpt-5.4")],
    [catalogOnlyConnection({ availableModels: ["gpt-5.4"] })],
  );
  assert.deepEqual(missing, {
    modelName: "gpt-5.4-mini",
    providerName: "OpenAI",
    providerType: "openai",
    state: "dropped",
  });
  assert.doesNotMatch(missing?.modelName ?? "", /external::/);
});

test("the pick from the report is named, never shown as its raw id", () => {
  const missing = missingExternalModel(
    DROPPED_ID,
    [option("llama3.2")],
    [connection({ availableModels: ["llama3.2"] })],
  );
  assert.equal(missing?.modelName, "kimi-k2.5");
  assert.equal(missing?.providerName, "Ollama");
  assert.equal(missing?.providerType, "ollama");
  assert.doesNotMatch(missing?.modelName ?? "", /external::/);
});

test("a connection that dropped every model still names the model", () => {
  assert.deepEqual(missingExternalModel(DROPPED_ID, []), {
    modelName: "kimi-k2.5",
    providerName: null,
    providerType: null,
    state: "dropped",
  });
});

test("a sibling under a different connection does not lend its name", () => {
  const other = option("gpt-5", {
    id: "external::other::gpt-5",
    providerId: "other",
    providerName: "OpenAI",
    providerType: "openai",
  });
  assert.deepEqual(missingExternalModel(DROPPED_ID, [other]), {
    modelName: "kimi-k2.5",
    providerName: null,
    providerType: null,
    state: "dropped",
  });
});

test("a model the user unticked is reported as disabled, not dropped", () => {
  assert.deepEqual(
    missingExternalModel(DROPPED_ID, [option("llama3.2")], [connection()]),
    {
      modelName: "kimi-k2.5",
      providerName: "Ollama",
      providerType: "ollama",
      state: "disabled",
    },
  );
});

test("unticking every model still names the connection", () => {
  assert.deepEqual(missingExternalModel(DROPPED_ID, [], [connection()]), {
    modelName: "kimi-k2.5",
    providerName: "Ollama",
    providerType: "ollama",
    state: "disabled",
  });
});

test("a connection saved before availableModels existed is not called dropped", () => {
  assert.deepEqual(
    missingExternalModel(
      DROPPED_ID,
      [option("llama3.2")],
      [connection({ availableModels: undefined })],
    ),
    {
      modelName: "kimi-k2.5",
      providerName: "Ollama",
      providerType: "ollama",
      state: "disabled",
    },
  );
});

test("an empty cached catalogue is unknown, not proof of a withdrawal", () => {
  assert.equal(
    missingExternalModel(
      DROPPED_ID,
      [option("llama3.2")],
      [connection({ availableModels: [] })],
    )?.state,
    "disabled",
  );
});

test("another connection's catalogue does not vouch for this one", () => {
  const missing = missingExternalModel(
    CATALOG_ONLY_PICK,
    [catalogOnlyOption("gpt-5.4")],
    [
      catalogOnlyConnection({ availableModels: ["gpt-5.4"] }),
      catalogOnlyConnection({
        id: "other",
        name: "Azure OpenAI",
        availableModels: ["gpt-5.4-mini"],
      }),
    ],
  );
  assert.equal(missing?.state, "dropped");
});

test("a manual model ID the user deleted is not blamed on the provider", () => {
  assert.deepEqual(
    missingExternalModel(
      DROPPED_ID,
      [option("llama3.2")],
      [connection({ availableModels: ["llama3.2"] })],
    ),
    {
      modelName: "kimi-k2.5",
      providerName: "Ollama",
      providerType: "ollama",
      state: "disabled",
    },
  );
});

test("no connection that takes manual model IDs reports a withdrawal", () => {
  for (const providerType of ["ollama", "vllm", "llama_cpp", "custom"]) {
    assert.equal(
      missingExternalModel(
        DROPPED_ID,
        [option("llama3.2", { providerType })],
        [connection({ providerType, availableModels: ["llama3.2"] })],
      )?.state,
      "disabled",
      providerType,
    );
  }
});

test("an OpenRouter model list is a shortlist, never proof of a withdrawal", () => {
  assert.equal(
    missingExternalModel(
      DROPPED_ID,
      [option("llama3.2", { providerType: "openrouter" })],
      [
        connection({
          name: "OpenRouter",
          providerType: "openrouter",
          availableModels: ["openai/gpt-5.4", "anthropic/claude-sonnet-5"],
        }),
      ],
    )?.state,
    "disabled",
  );
});

test("a catalogue-only connection still reports a real withdrawal", () => {
  assert.equal(
    missingExternalModel(
      CATALOG_ONLY_PICK,
      [catalogOnlyOption("gpt-5.4")],
      [catalogOnlyConnection({ availableModels: ["gpt-5.4"] })],
    )?.state,
    "dropped",
  );
  assert.equal(
    missingExternalModel(
      CATALOG_ONLY_PICK,
      [catalogOnlyOption("gpt-5.4")],
      [catalogOnlyConnection()],
    )?.state,
    "disabled",
  );
});

test("a connection with no readable type keeps trusting its catalogue", () => {
  assert.equal(
    missingExternalModel(
      CATALOG_ONLY_PICK,
      [catalogOnlyOption("gpt-5.4")],
      [
        catalogOnlyConnection({
          providerType: undefined,
          availableModels: ["gpt-5.4"],
        }),
      ],
    )?.state,
    "dropped",
  );
});

test("re-ticking the model clears the label entirely", () => {
  const restored = option("kimi-k2.5");
  assert.equal(
    missingExternalModel(restored.id, [restored], [connection()]),
    null,
  );
});

test("a percent-encoded model id is decoded for display", () => {
  const id = "external::conn::openai%2Fgpt-5";
  assert.equal(missingExternalModel(id, [])?.modelName, "openai/gpt-5");
});

test("a model the connection still offers is not treated as missing", () => {
  const listed = option("kimi-k2.5");
  assert.equal(missingExternalModel(listed.id, [listed]), null);
});

test("local and hub selections are left to the generic helper", () => {
  for (const id of [
    "unsloth/gemma-3-4b-it-GGUF",
    "/models/gemma-3-4b-it.gguf",
    "C:\\models\\gemma.gguf",
    "",
    null,
    undefined,
  ]) {
    assert.equal(missingExternalModel(id, []), null);
  }
});

test("compare toasts name the connected model, not its id", () => {
  assert.equal(compareModelDisplayName(DROPPED_ID), "kimi-k2.5");
  assert.equal(
    compareModelDisplayName("external::conn::openai%2Fgpt-5"),
    "gpt-5",
  );
  assert.equal(
    compareModelDisplayName("unsloth/gemma-3-4b-it"),
    "gemma-3-4b-it",
  );
  assert.equal(compareModelDisplayName("gemma-3-4b-it"), "gemma-3-4b-it");
});

test("externalModelLabel yields null for a non-external id", () => {
  assert.equal(externalModelLabel("unsloth/gemma-3-4b-it"), null);
  assert.equal(externalModelLabel(null), null);
  assert.equal(externalModelLabel(DROPPED_ID), "kimi-k2.5");
});

// No DOM renderer here, so the wiring is asserted against source.
const sourceOf = (relative: string, kind: ts.ScriptKind): ts.SourceFile => {
  const path = fileURLToPath(new URL(relative, import.meta.url));
  return ts.createSourceFile(
    path,
    readFileSync(path, "utf8"),
    ts.ScriptTarget.ESNext,
    true,
    kind,
  );
};

function currentModelMemo(): string | null {
  const source = sourceOf(
    "../src/features/model-picker/components/model-selector.tsx",
    ts.ScriptKind.TSX,
  );
  let body: string | null = null;
  const visit = (node: ts.Node): void => {
    if (
      ts.isVariableDeclaration(node) &&
      node.name.getText() === "currentModel" &&
      node.initializer
    ) {
      body = node.initializer.getText();
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  return body;
}

test("the picker trigger resolves a dropped connected model before naming it", () => {
  const memo = currentModelMemo();
  assert.ok(memo, "currentModel not found in model-selector.tsx");
  assert.match(
    memo,
    /missingExternalModel\(\s*selected,\s*externalModels,\s*externalConnections,?\s*\)/,
    "the fallback must consult the connections as well as the enabled options",
  );
  assert.match(memo, /picker\.modelDroppedByProvider/);
  assert.match(memo, /picker\.modelDropped\b/);
  assert.match(memo, /picker\.modelDisabledByProvider/);
  assert.match(memo, /picker\.modelDisabled\b/);
  assert.match(memo, /\[[^\]]*\bexternalModels\b[^\]]*\]\s*\)?\s*$/);
  assert.match(memo, /\[[^\]]*\bexternalConnections\b[^\]]*\]\s*\)?\s*$/);
});

test("the chat page feeds the picker the connections behind the options", () => {
  const page = readSrc("features/chat/chat-page.tsx");
  assert.match(page, /availableModels: provider\.availableModels/);
  assert.ok(
    (page.match(/externalConnections=\{externalConnections\}/g) ?? []).length >=
      5,
    "externalConnections must reach the picker on both the chat and compare paths",
  );
});

test("the compare and audio toasts use the external-aware labels", () => {
  const composer = readSrc("features/chat/shared-composer.tsx");
  assert.match(
    composer,
    /const name1 = model1\?\.id \? compareModelDisplayName\(/,
  );
  assert.match(
    composer,
    /const name2 = model2\?\.id \? compareModelDisplayName\(/,
  );
  assert.doesNotMatch(composer, /function modelDisplayName\(/);

  const audio = readSrc("features/chat/audio-attachment-adapter.ts");
  assert.match(
    audio,
    /activeModel\?\.name \|\|\s*externalModelLabel\(checkpoint\) \|\|/,
  );
});

// check-parity treats "picker." as a required overlay prefix, so a missing string fails CI.
test("the dropped-model strings are translated everywhere", async () => {
  const locales = [
    "ar",
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "it",
    "ja",
    "ko",
    "pt-br",
    "ru",
    "sv",
    "zh-CN",
  ];
  for (const locale of locales) {
    const module = (await import(`../src/i18n/locales/${locale}.ts`)) as Record<
      string,
      { picker?: Record<string, string> }
    >;
    const picker = Object.values(module).find((value) => value?.picker)?.picker;
    assert.ok(picker, `${locale} has no picker section`);
    assert.equal(typeof picker.modelDropped, "string", locale);
    assert.equal(typeof picker.modelDroppedByProvider, "string", locale);
    assert.match(picker.modelDroppedByProvider, /\{provider\}/, locale);
    assert.equal(typeof picker.modelDisabled, "string", locale);
    assert.equal(typeof picker.modelDisabledByProvider, "string", locale);
    assert.match(picker.modelDisabledByProvider, /\{provider\}/, locale);
    assert.notEqual(picker.modelDisabled, picker.modelDropped, locale);
    assert.notEqual(
      picker.modelDisabledByProvider,
      picker.modelDroppedByProvider,
      locale,
    );
  }
});
