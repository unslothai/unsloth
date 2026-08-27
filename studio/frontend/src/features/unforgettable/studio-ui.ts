// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";
import { Bookmark02Icon } from "@hugeicons/core-free-icons";

export const UNFORGETTABLE_SETTINGS_TAB = {
  id: "unforgettable" as const,
  labelKey: "settings.tabs.unforgettable" as const,
  icon: Bookmark02Icon,
  badgeKey: "common.new" as const,
};

export const UNFORGETTABLE_NAV_ITEM_META = {
  icon: Bookmark02Icon,
  labelKey: "shell.navigation.unforgettable" as const,
};

export const UNFORGETTABLE_SEARCH_KEYS = [
  "settings.unforgettable.title",
  "settings.unforgettable.episode.title",
  "settings.unforgettable.episode.planner",
  "settings.unforgettable.episode.plannerModel",
  "settings.unforgettable.episode.filter",
  "settings.unforgettable.episode.filterModel",
  "settings.unforgettable.episode.judgeModel",
  "settings.unforgettable.episode.highStakes",
  "settings.unforgettable.episode.confirmRetry",
  "settings.unforgettable.episode.skipStanding",
  "settings.unforgettable.episode.adapter",
  "settings.unforgettable.episode.testCommand",
  "settings.unforgettable.episode.maxClones",
  "settings.unforgettable.episode.maxSimTurns",
  "settings.unforgettable.episode.twinPlugin",
  "settings.unforgettable.approver.title",
  "settings.unforgettable.approver.voter",
  "settings.unforgettable.approver.voterModel",
  "settings.unforgettable.approver.supervisorUrl",
  "settings.unforgettable.store.title",
  "settings.unforgettable.store.path",
] as const satisfies readonly TranslationKey[];
