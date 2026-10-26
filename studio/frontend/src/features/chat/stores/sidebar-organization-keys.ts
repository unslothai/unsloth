// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Imports nothing: the store is in an import cycle, so keys there hit the TDZ. Add no imports. */

export const SIDEBAR_ORGANIZATION_STORAGE_KEY = "unsloth_sidebar_organization";
