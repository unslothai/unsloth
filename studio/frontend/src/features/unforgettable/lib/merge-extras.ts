// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isVirtualModel } from "./virtual-model.ts";

export type UnforgettableEpisodeExtras = {
  planner?: string | null;
  planner_model?: string | null;
  filter?: string | null;
  filter_model?: string | null;
  judge_model?: string | null;
  user_label?: string | null;
  stakes?: string | null;
  confirm_retry?: boolean | null;
  skip_standing?: boolean;
  adapter_id?: string | null;
  test_command?: string | null;
  max_clones?: number | null;
  max_sim_turns?: number | null;
  twin_plugin?: string | null;
  voter_model?: string | null;
};

export function mergeUnforgettableChatExtras(
  model: string | null | undefined,
  extras: UnforgettableEpisodeExtras | null | undefined,
): Record<string, unknown> {
  if (!isVirtualModel(model) || !extras) return {};
  const out: Record<string, unknown> = {};
  if (extras.planner) out.planner = extras.planner;
  if (extras.planner_model) out.planner_model = extras.planner_model;
  if (extras.filter) out.filter = extras.filter;
  if (extras.filter_model) out.filter_model = extras.filter_model;
  if (extras.judge_model) out.judge_model = extras.judge_model;
  if (extras.user_label) out.user_label = extras.user_label;
  if (extras.stakes) out.stakes = extras.stakes;
  if (extras.confirm_retry !== null && extras.confirm_retry !== undefined) {
    out.confirm_retry = extras.confirm_retry;
  }
  if (extras.skip_standing) out.skip_standing = true;
  if (extras.adapter_id) out.adapter_id = extras.adapter_id;
  if (extras.test_command) out.test_command = extras.test_command;
  if (extras.max_clones != null) out.max_clones = extras.max_clones;
  if (extras.max_sim_turns != null) out.max_sim_turns = extras.max_sim_turns;
  if (extras.twin_plugin) out.twin_plugin = extras.twin_plugin;
  if (extras.voter_model) out.voter_model = extras.voter_model;
  return out;
}
