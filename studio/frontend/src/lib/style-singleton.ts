// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// react-style-singleton (aliased in vite.config.ts) minus removal: removing react-remove-scroll's
// <style> makes WebKit restyle the whole page. Its rules only match body[data-scroll-locked].
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
    // The original's deps: re-adds only when a dynamic sheet changes.
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
