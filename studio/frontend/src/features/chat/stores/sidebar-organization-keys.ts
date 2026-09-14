// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Storage keys for the sidebar organization store, in a module that imports nothing. The store sits
 *  in an import cycle through the chat barrel, so the key defined there was still in its temporal
 *  dead zone when `general-tab.tsx` read it at module scope, throwing and leaving a white screen
 *  on launch. A module with no imports is always evaluated first, so do not add an import here. */

/** Exported so the preference reset clears the same key the store writes. */
export const SIDEBAR_ORGANIZATION_STORAGE_KEY = "unsloth_sidebar_organization";
