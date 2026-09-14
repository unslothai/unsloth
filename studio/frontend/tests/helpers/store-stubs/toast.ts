// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface RecordedToast {
  kind: "info" | "error" | "warning" | "success" | "loading";
  message: string;
  description?: string;
}

/** Every toast raised since the last clear, for the tests that assert a flow
 * reports itself exactly once. */
export const recordedToasts: RecordedToast[] = [];

function record(kind: RecordedToast["kind"]) {
  return (message?: unknown, options?: { description?: unknown }) => {
    recordedToasts.push({
      kind,
      message: String(message ?? ""),
      description:
        typeof options?.description === "string"
          ? options.description
          : undefined,
    });
  };
}

/** Swallow notifications; the real module pulls in a TSX component. */
export const toast = {
  info: record("info"),
  error: record("error"),
  warning: record("warning"),
  success: record("success"),
  loading: record("loading"),
  dismiss: () => undefined,
};

export function createLoadingToastIcon(): null {
  return null;
}
