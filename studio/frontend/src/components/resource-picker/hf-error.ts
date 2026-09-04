// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const HF_AUTH_ERROR_RE =
  /\b401\b|unauthorized|invalid.*token|invalid.*credential|authentication|forbidden|\b403\b/i;

export function isHfAuthError(message: string | null | undefined): boolean {
  return !!message && HF_AUTH_ERROR_RE.test(message);
}
