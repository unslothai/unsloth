// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Staged-rollout flag: unregistered custom properties inherit, so writing --sidebar-width or
// --studio-sidebar-live-width on an ancestor restyled the document every drag frame; scope each var
// to its consumers, since @property inherits: false would break descendant consumers.
export const PANEL_RESIZE_SCOPED_VARS_ENABLED = true;
