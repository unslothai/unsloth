// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The real barrel re-exports a .tsx dialog, which bare node cannot load. */
export function prepareHfTokenForUse(): never {
  throw new Error("prepareHfTokenForUse: no token prompt in tests");
}
