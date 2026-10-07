// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getClientPlatform } from "@/components/tauri/window-titlebar";
import { isTauri } from "@/lib/api-base";
import { createSettingsSearchIndex } from "./settings-search";

const clientPlatform = getClientPlatform();

export const DIALOG_SETTINGS_SEARCH_INDEX = createSettingsSearchIndex({
  desktop: isTauri,
  closeToTray:
    isTauri &&
    (clientPlatform.startsWith("win") ||
      clientPlatform.includes("windows") ||
      clientPlatform.includes("linux")),
});
