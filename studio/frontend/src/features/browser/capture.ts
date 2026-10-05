// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The page's sandbox can't open the print dialog, so a copy prints from a frame that can.

import { apiUrl } from "@/lib/api-base";
import { requestFrameSnapshot } from "./page-frame";

const PRINT_TIMEOUT_MS = 20_000;

export async function printFramePage(tabId: string): Promise<boolean> {
  const html = await requestFrameSnapshot(tabId);
  if (!html) return false;
  const frame = document.createElement("iframe");
  frame.setAttribute("sandbox", "allow-scripts allow-modals");
  frame.setAttribute("aria-hidden", "true");
  frame.tabIndex = -1;
  // Kept on screen at a pixel: a frame with no size may never lay out its images.
  frame.style.cssText = "position:fixed;right:0;bottom:0;width:1px;height:1px;border:0;opacity:0;pointer-events:none";
  frame.src = apiUrl("/api/browser/print");
  return new Promise((resolve) => {
    let finished = false;
    const finish = (printed: boolean) => {
      if (finished) return;
      finished = true;
      window.removeEventListener("message", listener);
      clearTimeout(timer);
      // print() returns as the dialog closes; a moment more for engines that spool after it.
      setTimeout(() => frame.remove(), 1000);
      resolve(printed);
    };
    const listener = (event: MessageEvent) => {
      if (event.source !== frame.contentWindow || event.data?.source !== "unsloth-print") return;
      if (event.data.type === "ready") {
        clearTimeout(timer);
        frame.contentWindow?.postMessage({ type: "unsloth:browser-print", html }, "*");
      } else if (event.data.type === "printed") finish(true);
    };
    // Only until the copy arrives: the dialog itself stays open as long as the reader wants.
    const timer = setTimeout(() => finish(false), PRINT_TIMEOUT_MS);
    window.addEventListener("message", listener);
    document.body.appendChild(frame);
  });
}

type CropTargetApi = { fromElement: (element: Element) => Promise<unknown> };
type CroppableTrack = MediaStreamTrack & { cropTo: (target: unknown) => Promise<void> };

function cropTargets(): CropTargetApi | null {
  const api = (globalThis as { CropTarget?: CropTargetApi }).CropTarget;
  return typeof api?.fromElement === "function" ? api : null;
}

export function canScreenshot(): boolean {
  return (
    typeof navigator !== "undefined" &&
    typeof navigator.mediaDevices?.getDisplayMedia === "function" &&
    cropTargets() !== null
  );
}

export class OtherSurfaceError extends Error {}

function nextFrame(video: HTMLVideoElement): Promise<void> {
  return new Promise((resolve) => {
    if (typeof video.requestVideoFrameCallback === "function") video.requestVideoFrameCallback(() => resolve());
    else requestAnimationFrame(() => resolve());
  });
}

 /** A PNG of `element` via this tab's screen capture; rejects if declined, OtherSurfaceError for another surface. */
export async function screenshotElement(element: HTMLElement): Promise<Blob | null> {
  const crop = cropTargets();
  if (!crop) return null;
  const target = await crop.fromElement(element);
  const stream = await navigator.mediaDevices.getDisplayMedia({
    video: { displaySurface: "browser", frameRate: 10 },
    audio: false,
    preferCurrentTab: true,
    selfBrowserSurface: "include",
    surfaceSwitching: "exclude",
    monitorTypeSurfaces: "exclude",
  } as DisplayMediaStreamOptions);
  try {
    const [track] = stream.getVideoTracks();
    if (!track) return null;
    try {
      await (track as CroppableTrack).cropTo(target);
    } catch {
      throw new OtherSurfaceError();
    }
    const video = document.createElement("video");
    video.muted = true;
    video.srcObject = stream;
    await video.play();
    // The picker has to leave the screen, and the sharing bar may resize the tab, before the shot.
    await new Promise((resolve) => setTimeout(resolve, 350));
    await nextFrame(video);
    await nextFrame(video);
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    video.srcObject = null;
    return await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
  } finally {
    for (const track of stream.getTracks()) track.stop();
  }
}
