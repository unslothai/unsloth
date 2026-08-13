// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type PillSettings = {
  enabled: boolean;
  defaultModel: string | null;
  defaultGgufVariant: string | null;
  autoLoad: boolean;
  excludedApps: string[];
};

export type PillModelOption = {
  id: string;
  label: string;
  source: "exported" | "cached";
};
