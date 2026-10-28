// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Stands in for "@/lib/toast" (sonner + react); tests read `calls` instead.
export const calls = [];

export const toast = Object.assign(
  (title, options) => calls.push({ kind: "default", title, options }),
  {
    info: (title, options) => calls.push({ kind: "info", title, options }),
    error: (title, options) => calls.push({ kind: "error", title, options }),
    warning: (title, options) =>
      calls.push({ kind: "warning", title, options }),
    success: (title, options) =>
      calls.push({ kind: "success", title, options }),
    dismiss: (id) => calls.push({ kind: "dismiss", id }),
  },
);

export function createLoadingToastIcon() {
  return null;
}
