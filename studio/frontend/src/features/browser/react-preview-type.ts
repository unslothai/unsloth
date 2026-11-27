// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A React preview tab is a file tab with this private type. Only openReactInBrowser() sets it, with
// a key under this prefix; "html:" also makes the tab one of the chat's pages.
export const REACT_PREVIEW_TYPE = "text/x-unsloth-react-preview";
export const REACT_PREVIEW_KEY_PREFIX = "html:react:";
