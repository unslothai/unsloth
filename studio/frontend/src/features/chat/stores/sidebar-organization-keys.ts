// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Storage keys for the sidebar organization store, in a module that imports
 * nothing.
 *
 * The store itself imports zustand and sits inside an import cycle that runs
 * through the chat barrel. `general-tab.tsx` reads this key from a top-level
 * `const`, so when the cycle is entered from the settings side the binding is
 * still in its temporal dead zone and the read throws
 * `Cannot access 'SIDEBAR_ORGANIZATION_STORAGE_KEY' before initialization`,
 * which unmounts the app: a white screen on launch.
 *
 * The app only avoided that by accident, because `app-sidebar.tsx` imports the
 * theme toggler above its own chat import and so happened to evaluate the store
 * first. Reordering those two imports reproduced the white screen.
 *
 * A module with no imports of its own is always fully evaluated before the
 * module that imports it, cycle or not, so a value here cannot be read early.
 * Keep it that way: do not add an import to this file.
 */

/** Exported so the preference reset clears the same key the store writes. */
export const SIDEBAR_ORGANIZATION_STORAGE_KEY = "unsloth_sidebar_organization";
