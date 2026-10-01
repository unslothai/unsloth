// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const AUDIO_PLAYER_RE = /<audio-player\s+src="([^"]+)"\s*\/>/;

export interface AudioReplyParts {
  audioSrc: string | null;
  text: string;
}

// An older backend's /audio/generate label, not something the assistant said.
const LEGACY_STATUS_LABEL_RE = /^\[Generated audio from: ".*"\]$/s;

export function spokenReplyText(content: string | null | undefined): string {
  const text = (content ?? "").trim();
  return LEGACY_STATUS_LABEL_RE.test(text) ? "" : text;
}

export function splitAudioReply(text: string): AudioReplyParts {
  const match = text.match(AUDIO_PLAYER_RE);
  if (!match) return { audioSrc: null, text };
  return {
    audioSrc: match[1] ?? null,
    text: text.replace(AUDIO_PLAYER_RE, "").trim(),
  };
}
