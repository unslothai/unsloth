// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs, stubJsxRuntime } from "./helpers/module-stubs.ts";

type Probe = { model_path: string; gguf_variant: string | null };
type DialogProps = {
  repoId: string;
  variant?: string;
  hasLocalGguf?: boolean;
  open: boolean;
  onOpenChange: (open: boolean) => void;
};

function renderInfo(props: DialogProps, online = true): Probe[] {
  const probes: Probe[] = [];
  const react = {
    useRef: (initial: unknown) => ({ current: initial }),
    useState: (initial: unknown) => [
      typeof initial === "function" ? initial() : initial,
      () => {},
    ],
    useEffect: (effect: () => unknown) => effect(),
  };
  const localHook = loadWithStubs(
    new URL(
      "../src/features/model-picker/components/model-selector/use-local-model-meta.ts",
      import.meta.url,
    ),
    {
      react,
      "@/features/chat/api/chat-api": {
        fetchGgufStagedMetadata: async (payload: Probe) => {
          probes.push(payload);
          return {};
        },
      },
    },
  );
  const { ModelInfoDialog } = loadWithStubs<{
    ModelInfoDialog: (props: DialogProps) => unknown;
  }>(
    new URL(
      "../src/features/model-picker/components/model-selector/model-info-dialog.tsx",
      import.meta.url,
    ),
    {
      react,
      "react/jsx-runtime": stubJsxRuntime(),
      "@/components/ui/dialog": {},
      "@/components/ui/spinner": {},
      "@/components/ui/tooltip": {},
      "@/features/hub/hooks/use-online-status": {
        useOnlineStatus: () => online,
      },
      "@/features/hub/hooks/use-selected-model-metadata": {
        useSelectedModelMetadata: () => ({ result: null, error: false }),
      },
      "@/features/hub/stores/external-link-confirm": {},
      "@/features/hub/stores/hf-token-store": {
        useHfTokenStore: () => "",
      },
      "@/lib/hf-endpoint": { useHfEndpoint: () => "https://huggingface.co" },
      "@/lib/utils": { cn: () => "" },
      "@hugeicons/core-free-icons": {},
      "@hugeicons/react": {},
      "../chat-template-editor-dialog": {},
      "./local-model-facts": {},
      "./model-guides": { modelGuide: () => null },
      "./model-info-facts": { metaFromHfResult: () => null },
      "./use-local-model-meta": localHook,
    },
  );
  ModelInfoDialog(props);
  return probes;
}

test("Hub-only info does not run model validation", () => {
  for (const repoId of ["unsloth/Qwen3-8B", "unsloth/Qwen3-8B-GGUF"]) {
    assert.deepEqual(
      renderInfo({ repoId, open: true, onOpenChange: () => {} }),
      [],
    );
  }
});

test("an undownloaded quant does not run model validation", () => {
  assert.deepEqual(
    renderInfo({
      repoId: "unsloth/Qwen3-8B-GGUF",
      variant: "Q8_0",
      hasLocalGguf: false,
      open: true,
      onOpenChange: () => {},
    }),
    [],
  );
});

test("a downloaded GGUF probes its selected quant, including offline", () => {
  for (const online of [true, false]) {
    const probes = renderInfo(
      {
        repoId: "unsloth/Qwen3-8B-GGUF",
        variant: "Q4_K_M",
        hasLocalGguf: true,
        open: true,
        onOpenChange: () => {},
      },
      online,
    );
    assert.equal(probes.length, 1);
    assert.equal(probes[0].model_path, "unsloth/Qwen3-8B-GGUF");
    assert.equal(probes[0].gguf_variant, "Q4_K_M");
  }
});

test("a closed dialog does not probe a downloaded GGUF", () => {
  assert.deepEqual(
    renderInfo({
      repoId: "unsloth/Qwen3-8B-GGUF",
      hasLocalGguf: true,
      open: false,
      onOpenChange: () => {},
    }),
    [],
  );
});
