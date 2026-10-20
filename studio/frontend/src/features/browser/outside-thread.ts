// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const THREAD_VIEWPORT = ".aui-thread-viewport";

export function outsideThread<T extends Element = HTMLElement>(
  selector: string,
  root: Pick<ParentNode, "querySelector" | "querySelectorAll"> = document,
): T[] {
  const thread = root.querySelector(THREAD_VIEWPORT);
  if (!thread) return [...root.querySelectorAll<T>(selector)];
  const found: T[] = [];
  for (let node: Element = thread; node.parentElement; node = node.parentElement) {
    const parent = node.parentElement;
    if (parent.matches(selector)) found.push(parent as unknown as T);
    for (const sibling of parent.children) {
      if (sibling === node) continue;
      if (sibling.matches(selector)) found.push(sibling as T);
      found.push(...sibling.querySelectorAll<T>(selector));
    }
  }
  return found;
}
