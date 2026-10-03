// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface SendTarget {
  id: string;
  workflow: string;
  label: string;
}

export interface StemMixerStem {
  clipId: string;
  /** The runtime's stem id, e.g. "vocals". */
  role: string;
  label: string;
  /** An object URL once fetched. */
  src: string | null;
  durationS: number;
  peaks: readonly number[] | null;
}

export interface StemMixerProps {
  groupId: string;
  title: string;
  /** Shown beside the title, e.g. the model name. */
  subtitle?: string;
  stems: readonly StemMixerStem[];
  /** Move focus to Play when the mixer mounts (after a run). */
  autoFocus?: boolean;
  onDownloadStem(clipId: string): void;
  onDownloadAll(): void;
  sendTargets: readonly SendTarget[];
  onSend(target: SendTarget, clipId: string): void;
  /** False pauses playback, e.g. when the page is hidden. */
  active: boolean;
}
