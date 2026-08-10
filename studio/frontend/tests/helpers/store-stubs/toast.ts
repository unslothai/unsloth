// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Swallow notifications; the real module pulls in a TSX component. */
export const toast = {
  info: () => undefined,
  error: () => undefined,
  warning: () => undefined,
  success: () => undefined,
  loading: () => undefined,
  dismiss: () => undefined,
};

export function createLoadingToastIcon(): null {
  return null;
}
