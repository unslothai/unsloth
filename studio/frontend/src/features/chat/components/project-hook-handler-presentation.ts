// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface ProjectHookHandlerPresentation {
  state: string;
  execution: string;
  eligible: boolean;
}

export function projectHookHandlerPresentation({
  trusted,
  enabled,
  active,
  background,
}: {
  trusted: boolean;
  enabled: boolean;
  active: boolean;
  background: boolean;
}): ProjectHookHandlerPresentation {
  const eligible = trusted && enabled && active;
  return {
    state: `${enabled ? "Enabled" : "Disabled"} preference, ${eligible ? "trusted and eligible" : "not eligible"}`,
    execution: background
      ? "background, inactive in this version"
      : "foreground, may control supported pre-events",
    eligible,
  };
}
