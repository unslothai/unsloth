// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import type {
  MediaGenerationKind,
  MediaGenerationPreset,
  MediaGenerationPresetSettings,
} from "./types";

type OrderedWrite = {
  writer: string;
  sequence: number;
  keepalive?: boolean;
};

function orderedWriteHeaders({ writer, sequence }: OrderedWrite) {
  return {
    "Preset-Writer": writer,
    "Preset-Sequence": String(sequence),
  };
}

async function parseResponse<Result>(response: Response) {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      body && typeof body === "object" && "detail" in body
        ? JSON.stringify(body.detail)
        : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return body as Result;
}

export async function getMediaGenerationPresetSettings<Params, LoadConfig>(
  kind: MediaGenerationKind,
) {
  return parseResponse<MediaGenerationPresetSettings<Params, LoadConfig>>(
    await authFetch(`/api/settings/generation-presets/${kind}`),
  );
}

export async function saveMediaGenerationPresetSettings<Params, LoadConfig>(
  kind: MediaGenerationKind,
  settings: MediaGenerationPresetSettings<Params, LoadConfig>,
  options: OrderedWrite,
) {
  return parseResponse<MediaGenerationPresetSettings<Params, LoadConfig>>(
    await authFetch(`/api/settings/generation-presets/${kind}`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        ...orderedWriteHeaders(options),
      },
      body: JSON.stringify(settings),
      keepalive: options.keepalive,
    }),
  );
}

export async function upsertMediaGenerationPreset<Params, LoadConfig>(
  kind: MediaGenerationKind,
  preset: MediaGenerationPreset<Params, LoadConfig>,
  options: OrderedWrite,
) {
  return parseResponse<{ saved: boolean }>(
    await authFetch(`/api/settings/generation-presets/${kind}/custom`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        ...orderedWriteHeaders(options),
      },
      body: JSON.stringify(preset),
      keepalive: true,
    }),
  );
}

export async function deleteMediaGenerationPreset(
  kind: MediaGenerationKind,
  name: string,
  options: OrderedWrite,
) {
  const query = new URLSearchParams({ name });
  return parseResponse<{ deleted: boolean }>(
    await authFetch(`/api/settings/generation-presets/${kind}/custom?${query}`, {
      method: "DELETE",
      headers: orderedWriteHeaders(options),
      keepalive: true,
    }),
  );
}
