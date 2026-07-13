// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type TrainingPresetRecord,
  createServerTrainingPreset,
  deleteServerTrainingPreset,
  getServerTrainingPreset,
  listServerTrainingPresets,
  updateServerTrainingPreset,
} from "@/features/user-assets";
import type { PortableTrainingConfig } from "../lib/yaml-config";

export type TrainingPreset = TrainingPresetRecord<PortableTrainingConfig>;

export const listTrainingPresets = () =>
  listServerTrainingPresets<PortableTrainingConfig>();

export const getTrainingPreset = (id: string) =>
  getServerTrainingPreset<PortableTrainingConfig>(id);

export const createTrainingPreset = (input: {
  id: string;
  name: string;
  config: PortableTrainingConfig;
}) => createServerTrainingPreset(input);

export const updateTrainingPreset = (input: {
  id: string;
  name: string;
  config: PortableTrainingConfig;
  revision: number;
}) => updateServerTrainingPreset(input);

export const deleteTrainingPreset = (id: string, revision: number) =>
  deleteServerTrainingPreset(id, revision);
