// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The rendered chart as a standalone SVG or PNG. Its colours are already literal.

export function svgToString(svg: SVGSVGElement): string {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  const box = clone.getAttribute("viewBox")?.split(/\s+/) ?? [];
  if (box.length === 4) {
    clone.setAttribute("width", box[2]);
    clone.setAttribute("height", box[3]);
  }
  return `<?xml version="1.0" encoding="UTF-8"?>\n${new XMLSerializer().serializeToString(clone)}`;
}

export async function svgToPng(svg: SVGSVGElement, scale = 2): Promise<Blob> {
  const box = svg.getAttribute("viewBox")?.split(/\s+/).map(Number) ?? [];
  const width = box.length === 4 ? box[2] : 1000;
  const height = box.length === 4 ? box[3] : 600;
  const img = new Image();
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error("The chart could not be rasterised."));
    img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgToString(svg))}`;
  });
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("No 2D canvas context.");
  ctx.scale(scale, scale);
  ctx.drawImage(img, 0, 0, width, height);
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("PNG encoding failed."))), "image/png");
  });
}
