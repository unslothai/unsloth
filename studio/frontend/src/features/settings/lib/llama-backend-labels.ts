// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";

/** Backend labels shared by the picker and notifications. */
export const LLAMA_BACKEND_LABELS: Record<string, TranslationKey> = {
  auto: "settings.resources.llamaBackend.backends.auto",
  cpu: "settings.resources.llamaBackend.backends.cpu",
  cuda: "settings.resources.llamaBackend.backends.cuda",
  rocm: "settings.resources.llamaBackend.backends.rocm",
  vulkan: "settings.resources.llamaBackend.backends.vulkan",
  metal: "settings.resources.llamaBackend.backends.metal",
};

/** Use the backend identifier when this client has no localized label. */
export function backendDisplayName(
  backend: string | null | undefined,
  t: (key: TranslationKey) => string,
): string {
  if (!backend) {
    return t(LLAMA_BACKEND_LABELS.auto);
  }
  const key = LLAMA_BACKEND_LABELS[backend];
  return key ? t(key) : backend;
}
