// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Fail any unexpected network access. */
export function authFetch(): Promise<Response> {
  throw new Error("authFetch: no network in tests");
}
