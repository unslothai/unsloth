// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Print and screenshot for the panel. The desktop app does both natively: a page in a native view
// through its own engine, and anything else as a snapshot of the app's webview cropped to the page
// (src-tauri/src/browser_capture.rs), with no screen-share prompt. The web build prints a copy of a
// framed page and captures through the browser's region capture, where it has one (Chromium).

import { apiUrl, isTauri } from "@/lib/api-base";
import { callNative } from "./native-support";
import { hasNativeView, whenNativeViewShown } from "./native-view";
import { requestFrameSnapshot } from "./page-frame";
import { type BrowserTab, currentEntry } from "./store";

const PRINT_TIMEOUT_MS = 20_000;

const isMac = typeof navigator !== "undefined" && /Mac/.test(navigator.platform);

function nativePageOf(tab: BrowserTab): boolean {
  return currentEntry(tab).kind === "web" && hasNativeView(tab.id);
}

/** Whether a framed page can print. The desktop app's macOS webview drops `print()` from frames. */
export function canPrintFrames(): boolean {
  return !(isTauri && isMac);
}

/** Print the tab's page: the engine's dialog for a native view, else a copy of the framed page. */
export async function printPage(tab: BrowserTab): Promise<boolean> {
  if (!nativePageOf(tab)) return printFramePage(tab.id);
  // The menu that asked hides the view; print what the reader sees once it's back.
  await whenNativeViewShown(tab.id);
  return callNative("browser_view_print", { tabId: tab.id }).then(
    () => true,
    () => false,
  );
}

// The page's sandbox can't open the print dialog, so a copy prints from a frame that can.
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

/** The desktop app always can; the web build needs region capture (Chromium). */
export function canScreenshot(): boolean {
  if (isTauri) return true;
  return (
    typeof navigator !== "undefined" &&
    typeof navigator.mediaDevices?.getDisplayMedia === "function" &&
    cropTargets() !== null
  );
}

export class OtherSurfaceError extends Error {}

/** A PNG of the page area: null if there was nothing to capture or the reader stopped it.
 *  Rejects with NotAllowedError when the share prompt is declined, OtherSurfaceError for another surface. */
export async function screenshotPage(tab: BrowserTab, element: HTMLElement): Promise<Blob | null> {
  if (isTauri) return nativeScreenshot(tab, element);
  return regionScreenshot(element);
}

const png = (bytes: ArrayBuffer) => (bytes.byteLength > 0 ? new Blob([bytes], { type: "image/png" }) : null);

async function nativeScreenshot(tab: BrowserTab, element: HTMLElement): Promise<Blob | null> {
  if (nativePageOf(tab)) {
    // A native view sits over the DOM and hides under the menu: capture it once it's back.
    if (!(await whenNativeViewShown(tab.id))) return null;
    return png(await callNative<ArrayBuffer>("browser_capture", { tabId: tab.id }));
  }
  await clearOfOverlays(element);
  return withChromeHidden(async () => {
    const rect = element.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) return null;
    return png(
      await callNative<ArrayBuffer>("browser_capture", {
        bounds: {
          x: rect.left,
          y: rect.top,
          width: rect.width,
          height: rect.height,
          viewportWidth: window.innerWidth,
        },
      }),
    );
  });
}

async function regionScreenshot(element: HTMLElement): Promise<Blob | null> {
  const crop = cropTargets();
  if (!crop) return null;
  const target = await crop.fromElement(element);
  const stream = await navigator.mediaDevices.getDisplayMedia({
    video: { displaySurface: "browser", frameRate: 30 },
    audio: false,
    preferCurrentTab: true,
    selfBrowserSurface: "include",
    surfaceSwitching: "exclude",
    monitorTypeSurfaces: "exclude",
  } as DisplayMediaStreamOptions);
  const video = document.createElement("video");
  try {
    const [track] = stream.getVideoTracks();
    if (!track) return null;
    try {
      await (track as CroppableTrack).cropTo(target);
    } catch {
      throw new OtherSurfaceError();
    }
    video.muted = true;
    video.srcObject = stream;
    await video.play();
    await clearOfOverlays(element);
    return await withChromeHidden(async () => {
      // The prompt has to leave the screen, and the sharing bar resizes the tab, before the shot.
      if (!(await steadyFrames(video, track, element))) return null;
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d")?.drawImage(video, 0, 0);
      return await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
    });
  } finally {
    video.srcObject = null;
    for (const track of stream.getTracks()) track.stop();
  }
}

function nextFrame(video: HTMLVideoElement): Promise<void> {
  return new Promise((resolve) => {
    // Bounded: a video that stops producing frames would otherwise never answer.
    const timer = setTimeout(resolve, 250);
    const done = () => {
      clearTimeout(timer);
      resolve();
    };
    if (typeof video.requestVideoFrameCallback === "function") video.requestVideoFrameCallback(done);
    else requestAnimationFrame(done);
  });
}

const STEADY_MIN_MS = 250;
const STEADY_MAX_MS = 2000;
const STEADY_FRAMES = 3;

/** Wait until the captured frames and the page keep one size for a few frames (the sharing bar
 *  has settled); false if the reader stopped sharing first. */
async function steadyFrames(video: HTMLVideoElement, track: MediaStreamTrack, element: HTMLElement): Promise<boolean> {
  const started = performance.now();
  let last = "";
  let same = 0;
  while (performance.now() - started < STEADY_MAX_MS) {
    await nextFrame(video);
    if (track.readyState === "ended") return false;
    const rect = element.getBoundingClientRect();
    const size = `${video.videoWidth}x${video.videoHeight}/${Math.round(rect.width)}x${Math.round(rect.height)}`;
    same = size === last && video.videoWidth > 0 ? same + 1 : 0;
    last = size;
    if (same >= STEADY_FRAMES && performance.now() - started >= STEADY_MIN_MS) break;
  }
  return track.readyState !== "ended" && video.videoWidth > 0;
}

// Studio UI the shot shouldn't keep: menus closing, dialogs, popovers.
const OVERLAY_SELECTOR = '[data-radix-popper-content-wrapper], [role="dialog"], [role="alertdialog"], [role="menu"]';

function intersects(a: DOMRect, b: DOMRect): boolean {
  return a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
}

/** Wait (briefly) for menus and dialogs over the page to finish closing, then for a paint. */
async function clearOfOverlays(element: HTMLElement, timeoutMs = 1000): Promise<void> {
  const started = performance.now();
  const frame = () => new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  while (performance.now() - started < timeoutMs) {
    const rect = element.getBoundingClientRect();
    const covered = [...document.querySelectorAll<HTMLElement>(OVERLAY_SELECTOR)].some((overlay) => {
      if (element.contains(overlay)) return false;
      const box = overlay.getBoundingClientRect();
      return box.width > 0 && box.height > 0 && intersects(box, rect);
    });
    if (!covered) break;
    await frame();
  }
  await frame();
  await frame();
}

// Toasts, tooltips and the file toolbar float over the page; hidden only while the shot is taken.
const CAPTURE_STYLE =
  "[data-sonner-toaster],[role=tooltip],[data-radix-popper-content-wrapper]:has([role=tooltip]),.browser-file-toolbar{visibility:hidden!important}";

async function withChromeHidden<T>(capture: () => Promise<T>): Promise<T> {
  const style = document.createElement("style");
  style.textContent = CAPTURE_STYLE;
  document.head.appendChild(style);
  try {
    await new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
    return await capture();
  } finally {
    style.remove();
  }
}
