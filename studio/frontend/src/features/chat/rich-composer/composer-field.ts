// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const RICH_COMPOSER_ATTRIBUTE = "data-rich-composer";

export interface RichComposerField extends HTMLElement {
  readonly value: string;
  readonly selectionStart: number;
  readonly selectionEnd: number;
  readonly form: HTMLFormElement | null;
  setSelectionRange(start: number, end: number): void;
}

export function isRichComposerField(el: unknown): el is RichComposerField {
  return el instanceof HTMLElement && el.hasAttribute(RICH_COMPOSER_ATTRIBUTE);
}

export function isComposerField(
  el: unknown,
): el is HTMLTextAreaElement | RichComposerField {
  return el instanceof HTMLTextAreaElement || isRichComposerField(el);
}
