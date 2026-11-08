// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type Ref, type RefCallback, useCallback } from "react";

// Shortcut checks read the elements that register here plus whatever is outside the app root
// (third-party portals), instead of walking a long chat's whole document on every press.

/** Every element that can match MODAL_BACKDROP_SELECTOR or an open dialog's content. */
export const MODAL_SURFACE = "modal";
/** Surfaces keyed by the selector their checks ask for. */
export const COMPOSER_SURFACE = ".aui-composer-input";
export const SIDEBAR_SURFACE = '[data-slot="sidebar"]';
export const FIND_SCOPE_SURFACE = "[data-find-scope]";

const surfaces = new Map<string, Map<Element, number>>();
let appRoot: Element | null = null;

/** Called once by the app entry; without it every check reads the document as before. */
export function trustSurfaceRegistry(root: Element | null): void {
  appRoot = root;
}

/** Refcounted: one element may register twice while a ref hands over. Returns the unregister. */
export function registerSurface(key: string, element: Element): () => void {
  let elements = surfaces.get(key);
  if (!elements) {
    elements = new Map();
    surfaces.set(key, elements);
  }
  const held = elements;
  held.set(element, (held.get(element) ?? 0) + 1);
  let dropped = false;
  return () => {
    if (dropped) return;
    dropped = true;
    const left = (held.get(element) ?? 1) - 1;
    if (left > 0) held.set(element, left);
    else held.delete(element);
  };
}

function pushOutsideRoot(
  root: Element,
  selector: string,
  out: Set<Element>,
  firstOnly: boolean,
): void {
  let node: Element = root;
  for (
    let parent = node.parentElement;
    parent;
    node = parent, parent = parent.parentElement
  ) {
    if (parent.matches(selector)) {
      out.add(parent);
      if (firstOnly) return;
    }
    for (const sibling of parent.children) {
      if (sibling === node) continue;
      if (sibling.matches(selector)) {
        out.add(sibling);
        if (firstOnly) return;
      }
      if (firstOnly) {
        const hit = sibling.querySelector(selector);
        if (hit) {
          out.add(hit);
          return;
        }
      } else {
        for (const hit of sibling.querySelectorAll(selector)) out.add(hit);
      }
    }
  }
}

/** The elements `document.querySelectorAll(selector)` would return, in no particular order, read
 * from the registry under `key` plus the document outside the app root. Null when the registry is
 * not trusted, so the caller queries the document itself. */
export function trackedSurfaces(
  key: string,
  selector: string,
  firstOnly = false,
): Element[] | null {
  if (typeof document === "undefined") return null;
  const root = appRoot;
  if (!root?.isConnected) return null;
  const out = new Set<Element>();
  for (const element of surfaces.get(key)?.keys() ?? []) {
    if (element.isConnected && element.matches(selector)) {
      out.add(element);
      if (firstOnly) return [element];
    }
  }
  pushOutsideRoot(root, selector, out, firstOnly);
  return [...out];
}

/** `document.querySelector(selector) !== null` for a selector only `key`'s producers match. */
export function hasTrackedSurface(key: string, selector: string): boolean {
  if (typeof document === "undefined") return false;
  const found = trackedSurfaces(key, selector, true);
  return found ? found.length > 0 : document.querySelector(selector) !== null;
}

function attachRef<T>(ref: Ref<T> | undefined, value: T): () => void {
  if (typeof ref === "function") {
    const cleanup = ref(value);
    return typeof cleanup === "function" ? cleanup : () => ref(null);
  }
  if (ref && typeof ref === "object") {
    ref.current = value;
    return () => {
      ref.current = null;
    };
  }
  return () => undefined;
}

/** One ref feeding several, so a primitive's own ref and a caller's both see the element. */
export function useMergedRefs<T>(
  ...refs: (Ref<T> | undefined)[]
): RefCallback<T> {
  // biome-ignore lint/correctness/useExhaustiveDependencies: the refs are the dependencies.
  return useCallback((element: T | null) => {
    if (element === null) return;
    const detach = refs.map((ref) => attachRef(ref, element));
    return () => {
      for (const undo of detach) undo();
    };
  }, refs);
}

/** A ref that registers the element under `key` while mounted and still reaches `forwarded`. */
export function useSurfaceRef<T extends Element>(
  key: string,
  forwarded?: Ref<T>,
): RefCallback<T> {
  return useCallback(
    (element: T | null) => {
      if (element === null) return;
      const unregister = registerSurface(key, element);
      const detach = attachRef(forwarded, element);
      return () => {
        unregister();
        detach();
      };
    },
    [key, forwarded],
  );
}
