// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Kept out of api.ts, which the eager chat chunk carries: only the Images page calls these.
import { authFetch } from "@/features/auth";
import { type DiffusionControlNetInfo, type GalleryImage, parseJson } from "./api";

export async function listDiffusionControlNets(
  family?: string,
): Promise<DiffusionControlNetInfo[]> {
  const qs = family ? `?family=${encodeURIComponent(family)}` : "";
  const data = await parseJson<{ controlnets: DiffusionControlNetInfo[] }>(
    await authFetch(`/api/models/diffusion-controlnets${qs}`),
  );
  return data.controlnets ?? [];
}

/** Move one image to just after `afterId` (null = front). The server also decides the pin. */
export async function moveGalleryImage(id: string, afterId: string | null): Promise<GalleryImage> {
  return parseJson(
    await authFetch(`/api/inference/images/gallery/${id}/move`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ after_id: afterId }),
    }),
  );
}

export async function addGalleryImageToProject(
  id: string,
  projectId: string,
): Promise<{ path: string; already: boolean }> {
  return parseJson(
    await authFetch(`/api/inference/images/gallery/${id}/project`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_id: projectId }),
    }),
  );
}
