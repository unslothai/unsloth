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
import type { ExternalModelRef } from "../src/features/model-picker/components/model-selector/missing-external-model.ts";
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

// The premise: the shared helper cannot shorten this id, which is why the picker's
// fallback printed it verbatim.
test("the generic display helper leaves an external id untouched", () => {
  assert.equal(modelDisplayName(DROPPED_ID), DROPPED_ID);
});

test("a dropped connected model is named, never shown as its raw id", () => {
  // Fetch Models replaced the connection's list and kimi-k2.5 is no longer in it.
  const missing = missingExternalModel(DROPPED_ID, [option("llama3.2")]);
  assert.deepEqual(missing, {
    modelName: "kimi-k2.5",
    providerName: "Ollama",
    providerType: "ollama",
  });
  assert.doesNotMatch(missing?.modelName ?? "", /external::/);
});

test("a connection that dropped every model still names the model", () => {
  assert.deepEqual(missingExternalModel(DROPPED_ID, []), {
    modelName: "kimi-k2.5",
    providerName: null,
    providerType: null,
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
  });
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
    /missingExternalModel\(\s*selected,\s*externalModels,?\s*\)/,
    "the fallback must consult the connected-model list before modelDisplayName",
  );
  // A name alone would hide that the model cannot be loaded, so the trigger says so.
  assert.match(memo, /picker\.modelDroppedByProvider/);
  assert.match(memo, /picker\.modelDropped\b/);
  // A memo that reads externalModels must list it, or the label survives a refresh.
  assert.match(memo, /\[[^\]]*\bexternalModels\b[^\]]*\]\s*\)?\s*$/);
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
  }
});
