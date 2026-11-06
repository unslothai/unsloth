// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Stands in for react-style-singleton (vite.config.ts aliases it), which react-remove-scroll uses to
// lock scrolling under dialogs and modal menus. The original removes its <style> when the last lock
// ends; WebKit then restyles and relays out the whole page, 5 s per close in a 1M-token chat on
// WebKitGTK. Its rules only apply under body[data-scroll-locked], so the tag stays.
import { useEffect } from "react";

export const stylesheetSingleton = () => {
  let tag: HTMLStyleElement | null = null;
  return {
    add(style: string) {
      if (typeof document === "undefined") return;
      if (!tag) {
        tag = document.createElement("style");
        tag.type = "text/css";
        tag.textContent = style;
        document.head.appendChild(tag);
      } else if (tag.textContent !== style) {
        tag.textContent = style;
      }
    },
    remove() {},
  };
};

export const styleHookSingleton = () => {
  const sheet = stylesheetSingleton();
  return (styles: string, isDynamic?: boolean) => {
    // Same dependency as the original: re-adds only when a dynamic sheet changes.
    useEffect(() => {
      sheet.add(styles);
      return () => sheet.remove();
    }, [styles && isDynamic]); // eslint-disable-line react-hooks/exhaustive-deps
  };
};

export const styleSingleton = () => {
  const useStyle = styleHookSingleton();
  return function Sheet({ styles, dynamic }: { styles: string; dynamic?: boolean }) {
    useStyle(styles, dynamic);
    return null;
  };
};
