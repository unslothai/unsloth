// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const PAGE_SIZE = 50;
// The list route clamps a page to 200, so asking for more silently gets 200 back.
export const MAX_PAGE_SIZE = 200;
// Mirrors the STT sidecar's own limits, so a recording is stopped at the boundary rather than uploaded and refused.
// The two it mirrors are _MAX_AUDIO_SECONDS and STT_AUDIO_B64_MAX_CHARS.
export const RECORDING_MAX_SECONDS = 30 * 60;
// STT_AUDIO_RAW_MAX_BYTES in utils/upload_limits.py. A larger client cap let a dense codec build
// a recording the raw route then refused with 413.
export const RECORDING_MAX_BYTES = 25 * 1024 * 1024;
export const RECORDING_CHUNK_MS = 1000;
export const TTS_MAX_TOKENS = 8192;
// Max tokens caps the OUTPUT and the prompt's own tokens sit in the same context window, so
// loading at exactly TTS_MAX_TOKENS made the advertised maximum unreachable.
export const TTS_PROMPT_CONTEXT_RESERVE = 2048;
// WAV clips run a few MB a minute; 64 MB keeps a healthy scrollback resident.
export const CLIP_BLOB_BUDGET_BYTES = 64 * 1024 * 1024;
