// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The generated-audio tag the chat adapter writes into a text part. An audio
// model's reply carries this tag followed by the text it spoke; the renderer
// shows the player and then that text, so the transcript half stays readable,
// selectable, and available to a screen reader instead of being discarded.
const AUDIO_PLAYER_RE = /<audio-player\s+src="([^"]+)"\s*\/>/;

export interface AudioReplyParts {
  /** The player's source, or null when the text carries no generated audio. */
  audioSrc: string | null;
  /** The text with the tag removed and trimmed; "" for a clip-only reply. */
  text: string;
}

// What /audio/generate's `choices[0].message.content` used to be before it carried the
// spoken text: a diagnostic label cut at 100 characters. Against an older backend it is
// dropped rather than shown as something the assistant said.
const LEGACY_STATUS_LABEL_RE = /^\[Generated audio from: ".*"\]$/s;

/** The transcript to render under an audio reply's player, or "" when there is none. */
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
