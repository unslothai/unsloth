// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface SendTarget {
  id: string;
  workflow: string;
  label: string;
}

export interface StemMixerStem {
  clipId: string;
  role: string;
  label: string;
  src: string | null;
  /** Its audio could not be fetched: left out of playback instead of waiting on it. */
  failed?: boolean;
  durationS: number;
  peaks: readonly number[] | null;
}

export interface StemMixerProps {
  groupId: string;
  title: string;
  subtitle?: string;
  stems: readonly StemMixerStem[];
  autoFocus?: boolean;
  onDownloadStem(clipId: string): void;
  onDownloadAll(): void;
  sendTargets: readonly SendTarget[];
  onSend(target: SendTarget, clipId: string): void;
  active: boolean;
}
