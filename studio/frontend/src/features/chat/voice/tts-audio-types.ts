// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The TTS codec vocabulary, kept dependency-free so the pure voice helpers (and
// their node:test suites) can read it without pulling in the player hook's React
// and auth imports. use-tts-player re-exports both for its existing importers.

export const TTS_AUDIO_TYPES = new Set(["snac", "csm", "bicodec", "dac"]);

// Codecs the backend voice slot accepts. Mirrors _VOICE_SLOT_AUDIO_TYPES in
// routes/inference.py: the slot is a second llama-server, and llama.cpp decodes
// snac, bicodec and dac. `csm` is in TTS_AUDIO_TYPES but not here -- a CSM GGUF
// loads and then has no decoder, so offering one as a voice is offering a load
// that always fails.
export const VOICE_SLOT_AUDIO_TYPES = new Set(["snac", "bicodec", "dac"]);
