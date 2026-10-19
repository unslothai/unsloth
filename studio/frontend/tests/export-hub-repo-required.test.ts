// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import * as React from "react";
import * as jsxRuntime from "react/jsx-runtime";
import { renderToStaticMarkup } from "react-dom/server";
import type * as ExportRunPanelModule from "../src/features/export/components/export-run-panel.tsx";
import * as constants from "../src/features/export/constants.ts";
import * as logStyle from "../src/features/export/lib/log-style.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const tag =
  (name: string) =>
  ({
    children,
    asChild: _a,
    variant: _v,
    size: _s,
    ...rest
  }: Record<string, unknown>) =>
    React.createElement(name, rest, children as React.ReactNode);
const none = () => null;

const runState = {
  phase: "idle",
  isExporting: false,
  logLines: [],
  connected: false,
  reconnecting: false,
  stage: null,
  quantIndex: 0,
  quantTotal: 0,
  startedAt: null,
  result: null,
  error: null,
  summary: null,
  requestCancel: async () => {},
};

const { ExportRunPanel } = loadWithStubs<typeof ExportRunPanelModule>(
  new URL(
    "../src/features/export/components/export-run-panel.tsx",
    import.meta.url,
  ),
  {
    react: React,
    "react/jsx-runtime": jsxRuntime,
    "@/components/ui/button": { Button: tag("button") },
    "@/components/ui/input": { Input: tag("input") },
    "@/components/ui/input-group": {
      InputGroup: tag("div"),
      InputGroupAddon: tag("div"),
      InputGroupInput: tag("input"),
    },
    "@/components/ui/progress": { Progress: none },
    "@/components/ui/spinner": { Spinner: none },
    "@/components/ui/switch": { Switch: none },
    "@/components/ui/tooltip": {
      Tooltip: tag("span"),
      TooltipTrigger: tag("span"),
      TooltipContent: none,
    },
    "@/features/model-picker": { FolderBrowser: none },
    "@hugeicons/core-free-icons": {},
    "@hugeicons/react": { HugeiconsIcon: none },
    "lucide-react": { ChevronRightIcon: none },
    "zustand/react/shallow": { useShallow: (fn: unknown) => fn },
    "../constants": constants,
    "../lib/log-style": logStyle,
    "../stores/export-runtime-store": {
      useExportRuntimeStore: (select: (s: typeof runState) => unknown) =>
        select(runState),
      selectExportProgressPercent: () => 0,
    },
  },
);

type Props = React.ComponentProps<typeof ExportRunPanel>;
const props: Props = {
  exportMethod: "gguf",
  quantLevels: ["q4_k_m"],
  checkpoint: "unsloth/Qwen3-0.6B",
  baseModelName: "unsloth/Qwen3-0.6B",
  isAdapter: false,
  destination: "hub",
  onDestinationChange: () => {},
  saveDirectory: "exports/qwen",
  saveDirectoryInput: "",
  defaultSaveDirectory: "exports/qwen",
  saveDirectoryOverridden: false,
  onSaveDirectoryChange: () => {},
  hfUsername: "nilay",
  onHfUsernameChange: () => {},
  modelName: "qwen-gguf",
  onModelNameChange: () => {},
  hfToken: "",
  onHfTokenChange: () => {},
  privateRepo: false,
  onPrivateRepoChange: () => {},
  onStart: () => {},
  onClose: () => {},
};

function startButton(overrides: Partial<Props>): string {
  const html = renderToStaticMarkup(
    React.createElement(ExportRunPanel, { ...props, ...overrides }),
  );
  const button = html.match(/<button\b[^>]*>Start Export<\/button>/)?.[0];
  assert.ok(button, "Start Export is not rendered");
  return button;
}

test("a Hub push cannot start until Username / Org and Model Name are filled", () => {
  for (const blank of [
    { modelName: "" },
    { hfUsername: "" },
    { hfUsername: " ", modelName: " " },
  ]) {
    assert.match(startButton(blank), /disabled=""/, JSON.stringify(blank));
  }
  assert.doesNotMatch(startButton({}), /disabled/);
});

test("a local export starts without the Hub fields", () => {
  assert.doesNotMatch(
    startButton({ destination: "local", hfUsername: "", modelName: "" }),
    /disabled/,
  );
});

test("a decision export, which hides the Hub fields, is not held back by them", () => {
  assert.doesNotMatch(
    startButton({ decisionNote: "Saved to the run folder", modelName: "" }),
    /disabled/,
  );
});
