// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import * as React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import * as jsxRuntime from "react/jsx-runtime";
import type * as ModelSelectorModule from "../src/features/model-picker/components/model-selector.tsx";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const passthrough = ({ children }: { children?: React.ReactNode }) => children ?? null;
const nothing = () => null;

let pickerEject: unknown;

const { ModelSelector } = loadWithStubs<typeof ModelSelectorModule>(
  new URL("../src/features/model-picker/components/model-selector.tsx", import.meta.url),
  {
    react: React,
    "react/jsx-runtime": jsxRuntime,
    "@/components/ui/popover": {
      Popover: passthrough,
      PopoverContent: passthrough,
      PopoverTrigger: passthrough,
    },
    "@/components/ui/spinner": {
      Spinner: ({ label }: { label?: string }) =>
        React.createElement("i", { role: "status", "aria-label": label }),
    },
    "@/components/ui/tooltip": { TooltipProvider: passthrough },
    "@/features/chat/api-provider-logo": { ApiProviderLogo: nothing },
    "@/features/settings/stores/settings-dialog-store": { useSettingsDialogStore: {} },
    "@/features/npu": {
      isNpuModelId: () => false,
      NPU_MODEL_PREFIX: "npu::",
      useNpuStatus: () => [null, () => {}],
    },
    "@/i18n": { useT: () => (key: string) => key },
    "@/lib/chevron-icons": { ChevronDownStandardIcon: {} },
    "@/lib/utils": {
      cn: (...parts: unknown[]) => parts.filter((part) => typeof part === "string").join(" "),
    },
    "@hugeicons/core-free-icons": {},
    "@hugeicons/react": { HugeiconsIcon: nothing },
    "@tanstack/react-router": { useNavigate: () => () => {} },
    "../model-config/model-config-handoff": {
      modelConfigTarget: () => null,
      modelConfigTargetIsResident: () => false,
      modelConfigTargetMatchesSelection: () => false,
    },
    "../model-config/config-signature": { modelConfigInstanceKey: () => "" },
    "../model-config/model-identity": {
      ggufVariantsMatch: () => false,
      modelDisplayName: (id: string) => id,
    },
    "../model-config/per-model-config": { resolveInitialConfig: () => null },
    "./model-config-page": { ModelConfigPage: nothing },
    "./model-selector/missing-external-model": { missingExternalModel: () => null },
    "./model-selector/model-catalog": { artifactForRepoId: () => undefined },
    "./model-selector/pickers": {
      HubModelPicker: ({ onEject }: { onEject?: unknown }) => {
        pickerEject = onEject;
        return null;
      },
      hasDownloadedModels: () => false,
    },
    "./model-selector/hub-section": { resolveHubSection: () => "recommended" },
    "./model-selector/pill-tabs": { PillTabs: nothing },
    "./model-selector/row-meta": { loraOptionLabel: () => "" },
    "./model-selector/source-tabs": { isFineTunedSource: () => false },
  },
);

function render(ejecting: boolean) {
  pickerEject = undefined;
  return renderToStaticMarkup(
    React.createElement(ModelSelector, {
      models: [{ id: "unsloth/Qwen-Image-2.1-GGUF", name: "Qwen-Image-2.1-GGUF" }],
      value: "unsloth/Qwen-Image-2.1-GGUF",
      activeGgufVariant: "Q8_0",
      onEject() {},
      ejecting,
    }),
  );
}

test("a loaded model offers the eject on the pill and in the picker", () => {
  const html = render(false);
  assert.match(html, / data-eject-hit="true"/);
  assert.match(html, /GGUF · Q8_0/);
  assert.doesNotMatch(html, /Unloading/);
  assert.equal(typeof pickerEject, "function");
});

test("an eject in flight shows on the pill and cannot be clicked again", () => {
  const html = render(true);
  assert.match(html, /Qwen-Image-2\.1-GGUF/);
  assert.match(html, /Unloading…/);
  assert.match(html, /role="status"/);
  assert.doesNotMatch(html, / data-eject-hit=/);
  assert.equal(pickerEject, undefined);
});
