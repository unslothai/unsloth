// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #8405: a connected provider that drops the picked model leaves its
// `external::<connectionId>::<modelId>` id in the checkpoint with no option behind it, and
// every generic shortener in the app is an identity function for that id. These cover the
// three surfaces a raw id was verified to reach: the picker trigger, the compare toasts and
// the audio-attachment toast.

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
import { registerBundlerResolver } from "./helpers/kit.ts";

// Both helpers reach across the tree the way vite resolves it: the "@/" alias and an
// extensionless relative import.
registerBundlerResolver();
const { compareModelDisplayName, externalModelLabel } = await import(
  "../src/features/chat/lib/external-model-label.ts"
);
const { missingExternalModel } = await import(
  "../src/features/model-picker/components/model-selector/missing-external-model.ts"
);

// The id from the report, built by buildExternalModelId(providerId, "kimi-k2.5").
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

// A connection whose dialog has no manual model-ID box beside the fetched list: every id it
// can hold came from the catalogue, so the catalogue is a complete record of what it offers.
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

// The premise: the shared helper cannot shorten this id, which is why the picker's
// fallback printed it verbatim.
test("the generic display helper leaves an external id untouched", () => {
  assert.equal(modelDisplayName(DROPPED_ID), DROPPED_ID);
});

test("a dropped connected model is named, never shown as its raw id", () => {
  // Fetch Models replaced both lists and gpt-5.4-mini is in neither. Nothing but the
  // catalogue can put an id in this connection's `models`, so the catalogue is positive
  // evidence that the provider withdrew it.
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

// The connection dialog writes the ticked ids to `models` and the fetched catalogue to
// `availableModels`, and the picker's option list is built from `models` alone. Unticking
// the active model therefore looks exactly like a withdrawal unless the catalogue is
// consulted, and blaming the provider for the user's own edit is what these cover.
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
  // No sibling option survives, so the connection itself is the only source for the name.
  assert.deepEqual(missingExternalModel(DROPPED_ID, [], [connection()]), {
    modelName: "kimi-k2.5",
    providerName: "Ollama",
    providerType: "ollama",
    state: "disabled",
  });
});

test("a connection saved before availableModels existed is not called dropped", () => {
  // Legacy persisted connections carry no catalogue at all. Absent evidence, the claim the
  // provider withdrew the model is exactly the guess that produced the wrong label, so the
  // neutral reading wins: the id is out of `models`, which is all "not enabled" asserts.
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

// Ollama, vLLM, llama.cpp and OpenRouter take typed-in model IDs beside the fetched list,
// and chat-providers-dialog.tsx saves those to `models` only: `modelsToSave` unions the
// ticked ids with the manual ones, while `availableModels` is written as the fetched
// catalogue alone. A catalogue that never carried an id cannot report its withdrawal, so
// deleting the id from the manual box has to read as the user's own edit.
test("a manual model ID the user deleted is not blamed on the provider", () => {
  // The state a save leaves behind: llama3.2 stays ticked, the typed-in kimi-k2.5 is gone
  // from `models`, and the catalogue is untouched because it never held it.
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
  // OpenRouter is curated: the dialog saves `availableModels: []` and the sync fills the
  // gap from the registry's `default_models`, so the list the picker sees is a handful of
  // suggestions rather than the 300-odd models the gateway actually serves.
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
  // The other half of the rule: OpenAI has no manual model-ID box, so every id in `models`
  // came from a fetch and the catalogue's silence is the provider's own answer.
  assert.equal(
    missingExternalModel(
      CATALOG_ONLY_PICK,
      [catalogOnlyOption("gpt-5.4")],
      [catalogOnlyConnection({ availableModels: ["gpt-5.4"] })],
    )?.state,
    "dropped",
  );
  // An id the catalogue still carries is the user's own untick either way.
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

// Compare toasts: reachable because the compare pane headers accept connected models and
// the send path has no external guard.
test("compare toasts name the connected model, not its id", () => {
  assert.equal(compareModelDisplayName(DROPPED_ID), "kimi-k2.5");
  assert.equal(
    compareModelDisplayName("external::conn::openai%2Fgpt-5"),
    "gpt-5",
  );
  // Unchanged for everything else.
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

// No DOM renderer here, so assert the wiring in the source the way
// artifact-frame-network-access.test.ts does: the fix only reaches the user if these three
// call sites route the checkpoint through the helpers above.
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

/** The body of the `currentModel` useMemo in the picker, or null if it moved. */
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
  // A name alone would hide that the model cannot be loaded, so the trigger says so.
  assert.match(memo, /picker\.modelDroppedByProvider/);
  assert.match(memo, /picker\.modelDropped\b/);
  // The enabled list alone cannot tell a disabled model from a withdrawn one, so both
  // readings must be reachable from here.
  assert.match(memo, /picker\.modelDisabledByProvider/);
  assert.match(memo, /picker\.modelDisabled\b/);
  // A memo that reads these must list them, or the label survives a refresh.
  assert.match(memo, /\[[^\]]*\bexternalModels\b[^\]]*\]\s*\)?\s*$/);
  assert.match(memo, /\[[^\]]*\bexternalConnections\b[^\]]*\]\s*\)?\s*$/);
});

// The catalogue only reaches the picker if chat-page builds it from the connections and
// passes it down both the single-chat and compare paths.
test("the chat page feeds the picker the connections behind the options", () => {
  const page = readFileSync(
    fileURLToPath(new URL("../src/features/chat/chat-page.tsx", import.meta.url)),
    "utf8",
  );
  assert.match(page, /availableModels: provider\.availableModels/);
  // Once for the memo's own type, then every hop from chat-page to the picker.
  assert.ok(
    (page.match(/externalConnections=\{externalConnections\}/g) ?? []).length >=
      5,
    "externalConnections must reach the picker on both the chat and compare paths",
  );
});

test("the compare and audio toasts use the external-aware labels", () => {
  const composer = readFileSync(
    fileURLToPath(
      new URL("../src/features/chat/shared-composer.tsx", import.meta.url),
    ),
    "utf8",
  );
  assert.match(
    composer,
    /const name1 = model1\?\.id \? compareModelDisplayName\(/,
  );
  assert.match(
    composer,
    /const name2 = model2\?\.id \? compareModelDisplayName\(/,
  );
  // The local split("/") helper that leaked the id must be gone.
  assert.doesNotMatch(composer, /function modelDisplayName\(/);

  const audio = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/chat/audio-attachment-adapter.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  assert.match(
    audio,
    /activeModel\?\.name \|\|\s*externalModelLabel\(checkpoint\) \|\|/,
  );
});

// The strings the trigger shows must exist in every locale: check-parity treats "picker."
// as a required overlay prefix, so a missing one fails CI rather than falling back.
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
    // The two readings make different claims, so a locale that reuses one string for both
    // puts the withdrawal wording back on the user's own edit.
    assert.notEqual(picker.modelDisabled, picker.modelDropped, locale);
    assert.notEqual(
      picker.modelDisabledByProvider,
      picker.modelDroppedByProvider,
      locale,
    );
  }
});
