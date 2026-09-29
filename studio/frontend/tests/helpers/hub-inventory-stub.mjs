// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The hub barrel re-exports .tsx panels, which bare node cannot parse. Only the
// cache bump is reachable from the settings API, and it does nothing here.
export function bumpInventoryVersion() {}
