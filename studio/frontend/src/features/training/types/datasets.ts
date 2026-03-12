// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type CheckFormatResponse = {
  requires_manual_mapping: boolean;
  detected_format: string;
  columns: string[];
  suggested_mapping?: Record<string, string> | null;
  detected_image_column?: string | null;
  detected_audio_column?: string | null;
  detected_text_column?: string | null;
  detected_speaker_column?: string | null;
  preview_samples?: Record<string, unknown>[] | null;
  total_rows?: number | null;
  is_image?: boolean;
  is_audio?: boolean;
  multimodal_columns?: string[] | null;
  warning?: string | null;
};

export type UploadDatasetResponse = {
  filename: string;
  stored_path: string;
};

export type LocalDatasetInfo = {
  metadata?: {
    actual_num_records?: number | null;
    target_num_records?: number | null;
    total_num_batches?: number | null;
    num_completed_batches?: number | null;
    columns?: string[] | null;
  } | null;
  id: string;
  label: string;
  path: string;
  rows?: number | null;
  updated_at?: number | null;
};

export type LocalDatasetsResponse = {
  datasets: LocalDatasetInfo[];
};
