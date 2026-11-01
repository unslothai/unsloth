// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ResizeLimits } from "./utils/resize-image-file";

// showGreetingSloth stays the on/off switch so older builds and records keep reading it.
export type GreetingStyle = "text" | "image";
export type GreetingIcon = "sloth" | "custom" | "none";
export const GREETING_TEXT_MAX_LENGTH = 100;

// No SVG: uploads are re-encoded through a canvas, and the server and renderer only accept these.
export const GREETING_IMAGE_TYPES = [
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
] as const;
export const GREETING_UPLOAD_MAX_BYTES = 10 * 1024 * 1024;
// Must stay under MAX_GREETING_*_DATA_URL_LENGTH in backend/routes/settings.py.
export const GREETING_ICON_LIMITS: ResizeLimits = {
  maxEdge: 128,
  minEdge: 64,
  maxBytes: 48_000,
};
export const GREETING_IMAGE_LIMITS: ResizeLimits = {
  maxEdge: 960,
  minEdge: 320,
  maxBytes: 240_000,
};

const DATA_URL = /^data:image\/(?:png|jpeg|webp|gif);base64,[A-Za-z0-9+/=]+$/;

export function isGreetingDataUrl(value: unknown): value is string {
  return typeof value === "string" && DATA_URL.test(value);
}

export function validGreetingUpload(file: Pick<File, "type" | "size">) {
  return (
    file.size > 0 &&
    file.size <= GREETING_UPLOAD_MAX_BYTES &&
    (GREETING_IMAGE_TYPES as readonly string[]).includes(file.type)
  );
}

export type GreetingSettings = {
  showGreetingSloth: boolean;
  greetingStyle: GreetingStyle;
  greetingIcon: GreetingIcon;
  greetingIconDataUrl: string | null;
  greetingText: string;
  greetingImageDataUrl: string | null;
};

export const DEFAULT_GREETING: GreetingSettings = {
  showGreetingSloth: true,
  greetingStyle: "text",
  greetingIcon: "sloth",
  greetingIconDataUrl: null,
  greetingText: "",
  greetingImageDataUrl: null,
};

// Remote and persisted values are untrusted; anything unknown falls back to today's greeting.
export function sanitizeGreeting(
  input: Partial<Record<keyof GreetingSettings, unknown>>,
): GreetingSettings {
  return {
    showGreetingSloth: input.showGreetingSloth !== false,
    greetingStyle: input.greetingStyle === "image" ? "image" : "text",
    greetingIcon:
      input.greetingIcon === "custom" || input.greetingIcon === "none"
        ? input.greetingIcon
        : "sloth",
    greetingIconDataUrl: isGreetingDataUrl(input.greetingIconDataUrl)
      ? input.greetingIconDataUrl
      : null,
    greetingText:
      typeof input.greetingText === "string"
        ? input.greetingText.slice(0, GREETING_TEXT_MAX_LENGTH)
        : "",
    greetingImageDataUrl: isGreetingDataUrl(input.greetingImageDataUrl)
      ? input.greetingImageDataUrl
      : null,
  };
}

export type ResolvedGreeting =
  | { kind: "off" }
  | { kind: "image"; src: string }
  | {
      kind: "text";
      icon:
        | { kind: "sloth" }
        | { kind: "none" }
        | { kind: "custom"; src: string };
      // Null keeps the built-in lines that change with the time of day.
      text: string | null;
    };

// A style whose upload is missing falls back to the default rather than an empty greeting.
export function resolveGreeting(settings: GreetingSettings): ResolvedGreeting {
  const s = sanitizeGreeting(settings);
  if (!s.showGreetingSloth) {
    return { kind: "off" };
  }
  if (s.greetingStyle === "image" && s.greetingImageDataUrl) {
    return { kind: "image", src: s.greetingImageDataUrl };
  }
  const text = s.greetingText.trim() || null;
  if (s.greetingIcon === "none") {
    return { kind: "text", icon: { kind: "none" }, text };
  }
  if (s.greetingIcon === "custom" && s.greetingIconDataUrl) {
    return {
      kind: "text",
      icon: { kind: "custom", src: s.greetingIconDataUrl },
      text,
    };
  }
  return { kind: "text", icon: { kind: "sloth" }, text };
}
