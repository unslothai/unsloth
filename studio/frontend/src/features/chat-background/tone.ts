// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { luminance } from "./contrast";
import { toneOf, type ImageTone } from "./model";

export { glassReadability, type ImageTone } from "./model";

const SAMPLE = 48;

export async function measureTone(url: string): Promise<ImageTone> {
  const image = new Image();
  image.src = url;
  await image.decode();
  const canvas = document.createElement("canvas");
  canvas.width = SAMPLE;
  canvas.height = SAMPLE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("canvas");
  ctx.drawImage(image, 0, 0, SAMPLE, SAMPLE);
  const data = ctx.getImageData(0, 0, SAMPLE, SAMPLE).data;
  const values: number[] = [];
  for (let i = 0; i < data.length; i += 4)
    values.push(luminance([data[i], data[i + 1], data[i + 2]]));
  return toneOf(values);
}
