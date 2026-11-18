// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "@typescript/typescript6";

const source = ts.createSourceFile(
  "pickers.tsx",
  readFileSync(
    new URL(
      "../src/features/model-picker/components/model-selector/pickers.tsx",
      import.meta.url,
    ),
    "utf8",
  ),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);

function declaration(name: string): string {
  for (const statement of source.statements) {
    if (ts.isFunctionDeclaration(statement) && statement.name?.text === name) {
      return statement.getText(source);
    }
    if (ts.isVariableStatement(statement)) {
      for (const item of statement.declarationList.declarations) {
        if (ts.isIdentifier(item.name) && item.name.text === name) {
          return `const ${item.getText(source)};`;
        }
      }
    }
  }
  throw new Error(`Missing declaration: ${name}`);
}

const compiled = ts.transpileModule(
  `${declaration("formatBytes")}\n${declaration("SoleQuantGgufDeviceRow")}`,
  {
    compilerOptions: { jsx: ts.JsxEmit.React, target: ts.ScriptTarget.ES2022 },
  },
).outputText;

type Node = { type: unknown; props: Record<string, unknown>; children: Node[] };
const React = {
  createElement: (
    type: unknown,
    props: Record<string, unknown> | null,
    ...children: Node[]
  ): Node => ({
    type,
    props: props ?? {},
    children,
  }),
  Fragment: "fragment",
};

const names = [
  "React",
  "memo",
  "useDeviceExpanderCallbacks",
  "useContext",
  "AudioCppCard",
  "ggufVariantExpectedBytes",
  "downloadedRowShellClassName",
  "onDeviceRepoLabel",
  "localPathTooltip",
  "ggufQuantDetailLabel",
  "ggufQuantChipLabel",
  "mediaPageForTask",
  "ggufVariantLookupTableBytes",
  "rovingOptionProps",
  "downloadedRowButtonClassName",
  "ROW_ACTIONS_CLASS",
  "ModelRow",
  "ModelRowMenu",
  "HugeiconsIcon",
  "ChevronDownStandardIcon",
  "ChevronRightStandardIcon",
  "ejectMenuItemsFor",
  "ModelLoadSettingsAction",
];
const modelRow = Symbol("ModelRow");
const values = [
  React,
  (value: unknown) => value,
  () => ({}),
  () => 0,
  {},
  (v: { size_bytes: number }) => v.size_bytes,
  () => "",
  (id: string) => id,
  () => "",
  (label: string) => label,
  (label: string) => label,
  () => null,
  () => 0,
  () => ({}),
  "",
  "",
  modelRow,
  "menu",
  "icon",
  "down",
  "right",
  () => [],
  "settings",
];
const renderRow = new Function(
  ...names,
  `${compiled}\nreturn SoleQuantGgufDeviceRow;`,
)(...values) as (props: unknown) => Node;

function modelProperties(cachedBytes: number): Record<string, unknown> {
  const tree = renderRow({
    repo: {
      repo_id: "example/model-GGUF",
      size_bytes: cachedBytes,
      cache_path: "/cache",
      load_id: "/cache/model",
    },
    sole: {
      variant: {
        quant: "Q4_K_M",
        filename: "model.gguf",
        size_bytes: 1_000_000_000,
        downloaded: true,
      },
    },
    optionKey: "model",
    selected: false,
    loaded: false,
    pinned: false,
    ejectable: false,
    loadedQuants: [],
    expanderOpen: false,
    updateDisabled: false,
    expander: {},
    actions: {},
    optionEvents: {},
    optionId: "model",
    optionActive: false,
    canConfigure: false,
    deleteDisabled: false,
  });
  const row = tree.children[0].children[0].children[0];
  assert.equal(row.type, modelRow);
  return row.props;
}

test("single quant row displays the same cached inventory size as the Hub, retaining weight size for memory", () => {
  const props = modelProperties(1_400_000_000);
  assert.equal(props.meta, "GGUF · 1.4GB");
  assert.equal(
    (props.memory as { sizeBytes: number }).sizeBytes,
    1_000_000_000,
  );
});

test("missing inventory size falls back to the known GGUF size", () => {
  assert.equal(modelProperties(0).meta, "GGUF · 1.0GB");
});
