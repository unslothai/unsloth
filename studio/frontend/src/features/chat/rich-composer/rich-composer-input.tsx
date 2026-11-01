// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import "../../../vendor/prosemirror-view/prosemirror.css";
import "./rich-composer.css";
import { useAui } from "@assistant-ui/react";
import { type Ref, useCallback, useLayoutEffect, useRef } from "react";
import { isHistoryTransaction } from "../../../vendor/prosemirror-history/index.js";
import {
  type EditorState,
  TextSelection,
  type Transaction,
} from "../../../vendor/prosemirror-state/index.js";
import { EditorView } from "../../../vendor/prosemirror-view/index.js";
import {
  isPlainPasteChord,
  plainPasteStillCounts,
} from "../utils/pasted-text.ts";
import { richPasteMarkdown } from "../utils/rich-paste.ts";
import { RICH_COMPOSER_ATTRIBUTE } from "./composer-field.ts";
import {
  composerPlugins,
  createComposerState,
  inCodeBlock,
  insertMarkdown,
  isEmptyDoc,
} from "./editor.ts";
import {
  docPosToOffset,
  offsetToDocPos,
  serializeMarkdown,
  sliceToMarkdown,
} from "./markdown.ts";

type Handler = (event: unknown) => void;

type RichComposerInputProps = {
  ref?: Ref<HTMLElement>;
  value?: string;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
  minRows?: number;
  maxRows?: number;
  onHeightChange?: (height: number, meta: { rowHeight: number }) => void;
  onChange?: Handler;
  onKeyDown?: Handler;
  onSelect?: Handler;
  onPaste?: Handler;
  [prop: string]: unknown;
};

const DROPPED = new Set([
  "name",
  "autoFocus",
  "style",
  "cacheMeasurements",
  "children",
  "rows",
]);

function reactEvent<E extends Event>(
  native: E,
  field: HTMLElement,
  extra: object = {},
) {
  return Object.assign(
    {
      nativeEvent: native,
      target: field,
      currentTarget: field,
      type: native.type,
      timeStamp: native.timeStamp,
      isTrusted: native.isTrusted,
      get defaultPrevented() {
        return native.defaultPrevented;
      },
      isDefaultPrevented: () => native.defaultPrevented,
      preventDefault: () => native.preventDefault(),
      stopPropagation: () => native.stopPropagation(),
      isPropagationStopped: () => false,
      persist: () => undefined,
    },
    extra,
  );
}

function keyEvent(native: KeyboardEvent, field: HTMLElement) {
  const { key, code, keyCode, shiftKey, ctrlKey, metaKey, altKey, repeat } =
    native;
  return reactEvent(native, field, {
    key,
    code,
    keyCode,
    which: keyCode,
    shiftKey,
    ctrlKey,
    metaKey,
    altKey,
    repeat,
    getModifierState: (name: string) => native.getModifierState(name),
  });
}

// For applySentTextGuard.
function inputTypeOf(tr: Transaction, view: EditorView): string {
  if (isHistoryTransaction(tr)) return "historyUndo";
  const ui = tr.getMeta("uiEvent");
  if (ui === "paste") return "insertFromPaste";
  if (ui === "drop") return "insertFromDrop";
  if (ui === "cut") return "deleteByCut";
  return view.composing ? "insertCompositionText" : "insertText";
}

export default function RichComposerInput(props: RichComposerInputProps) {
  const aui = useAui();
  const latest = useRef(props);
  latest.current = props;
  const hostRef = useRef<HTMLDivElement | null>(null);
  const viewRef = useRef<EditorView | null>(null);
  const syncRef = useRef<(() => void) | null>(null);

  const {
    ref,
    value,
    disabled,
    placeholder,
    className,
    minRows,
    maxRows,
    onHeightChange,
    onChange,
    onKeyDown,
    onSelect,
    onPaste,
    ...rest
  } = props;
  const events: Record<string, unknown> = {};
  const attributes: Record<string, string> = {};
  for (const [key, prop] of Object.entries(rest)) {
    if (
      DROPPED.has(key) ||
      prop === undefined ||
      prop === null ||
      prop === false
    )
      continue;
    if (/^on[A-Z]/.test(key)) events[key] = prop;
    else attributes[key] = String(prop);
  }
  const attributesKey = JSON.stringify(attributes);
  const attributesKeyRef = useRef(attributesKey);
  attributesKeyRef.current = attributesKey;

  const setHost = useCallback(
    (node: HTMLDivElement | null) => {
      hostRef.current = node;
      if (typeof ref === "function") ref(node);
      else if (ref) (ref as { current: HTMLElement | null }).current = node;
    },
    [ref],
  );

  useLayoutEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const composerText = () => {
      const state = aui.composer().getState();
      return state.isEditing ? state.text : "";
    };
    const plugins = composerPlugins(() => latest.current.placeholder ?? "");
    let lastInput: { event: Event; at: number } | null = null;
    let plainPasteAt = 0;
    let pendingSync = false;
    let lastHeight = -1;

    const measure = () => {
      const dom = view.dom;
      const style = getComputedStyle(dom);
      const px = (value: string) => Number.parseFloat(value) || 0;
      const rowHeight = px(style.lineHeight) || 24;
      const frame =
        px(style.paddingTop) +
        px(style.paddingBottom) +
        px(style.borderTopWidth) +
        px(style.borderBottomWidth);
      const last = dom.lastElementChild as HTMLElement | null;
      const content = last
        ? last.offsetTop +
          last.offsetHeight +
          px(getComputedStyle(last).marginBottom) -
          px(style.paddingTop)
        : rowHeight;
      const { minRows: min = 1, maxRows: max } = latest.current;
      const height = Math.min(
        max ? rowHeight * max + frame : Number.POSITIVE_INFINITY,
        Math.max(rowHeight * min + frame, content + frame),
      );
      if (height === lastHeight) return;
      lastHeight = height;
      dom.style.height = `${height}px`;
      latest.current.onHeightChange?.(height, { rowHeight });
    };

    const notifySelect = () => {
      if (!view.composing)
        latest.current.onSelect?.(reactEvent(new Event("select"), view.dom));
    };

    // The composer text is the source of truth, as for a controlled textarea.
    const sync = () => {
      if (view.composing) {
        pendingSync = true;
        return;
      }
      pendingSync = false;
      const text = composerText();
      if (serializeMarkdown(view.state.doc) === text) return;
      view.updateState(createComposerState(text, plugins));
      measure();
      notifySelect();
    };

    const dispatchTransaction = (tr: Transaction) => {
      view.updateState(view.state.apply(tr));
      if (tr.docChanged) {
        const fresh =
          lastInput && performance.now() - lastInput.at < 100
            ? lastInput.event
            : null;
        const native =
          fresh ??
          new InputEvent("input", {
            inputType: inputTypeOf(tr, view),
            isComposing: view.composing,
          });
        latest.current.onChange?.(reactEvent(native, view.dom));
        sync();
        measure();
      } else if (pendingSync) {
        sync();
      }
      if (tr.docChanged || tr.selectionSet) notifySelect();
    };

    const insert = (markdown: string, uiEvent: string) => {
      view.dispatch(
        insertMarkdown(view.state, markdown).setMeta("uiEvent", uiEvent),
      );
    };

    const view: EditorView = new EditorView(
      { mount: host },
      {
        state: createComposerState(composerText(), plugins),
        editable: () => !latest.current.disabled,
        attributes: (state: EditorState) => {
          const own: Record<string, string> = {
            ...JSON.parse(attributesKeyRef.current),
            class: `${latest.current.className ?? ""} unsloth-rich-input`,
            role: "textbox",
            "aria-multiline": "true",
            [RICH_COMPOSER_ATTRIBUTE]: "",
          };
          if (latest.current.placeholder)
            own["aria-placeholder"] = latest.current.placeholder;
          if (latest.current.disabled) own["aria-disabled"] = "true";
          if (isEmptyDoc(state.doc)) own["data-empty"] = "";
          return own;
        },
        dispatchTransaction,
        clipboardTextSerializer: (slice) => sliceToMarkdown(slice),
        handleDOMEvents: {
          // The composer's handlers (send, IME gate, @ picker) run first.
          keydown: (_view, event) => {
            plainPasteAt = isPlainPasteChord(event) ? performance.now() : 0;
            latest.current.onKeyDown?.(keyEvent(event, view.dom));
            return event.defaultPrevented;
          },
          keyup: () => {
            plainPasteAt = 0;
            return false;
          },
          blur: () => {
            plainPasteAt = 0;
            return false;
          },
          beforeinput: (_view, event) => {
            lastInput = { event, at: performance.now() };
            return false;
          },
          compositionend: () => {
            setTimeout(() => {
              if (pendingSync) sync();
            }, 50);
            return false;
          },
          paste: (_view, event) => {
            const plain = plainPasteStillCounts(
              plainPasteAt,
              performance.now(),
            );
            plainPasteAt = 0;
            latest.current.onPaste?.(
              reactEvent(event, view.dom, {
                clipboardData: event.clipboardData,
              }),
            );
            if (event.defaultPrevented) return true;
            const data = event.clipboardData;
            if (!data) return false;
            event.preventDefault();
            const text = data.getData("text/plain").replace(/\r\n?/g, "\n");
            const own = data.getData("text/html").includes("data-pm-slice");
            const markdown =
              inCodeBlock(view.state) || plain || own
                ? null
                : richPasteMarkdown(data);
            if (markdown ?? text) insert(markdown ?? text, "paste");
            return true;
          },
          drop: (_view, event) => (event.dataTransfer?.files.length ?? 0) > 0,
        },
        handleDrop: (_view, event, _slice, moved) => {
          const text = event.dataTransfer?.getData("text/plain");
          if (moved || !text) return false;
          const at = view.posAtCoords({
            left: event.clientX,
            top: event.clientY,
          });
          if (!at) return false;
          view.dispatch(
            view.state.tr.setSelection(
              TextSelection.create(view.state.doc, at.pos),
            ),
          );
          insert(text.replace(/\r\n?/g, "\n"), "drop");
          return true;
        },
      },
    );
    viewRef.current = view;

    const field = view.dom;
    Object.defineProperties(field, {
      value: {
        configurable: true,
        get: () => serializeMarkdown(view.state.doc),
      },
      selectionStart: {
        configurable: true,
        get: () => docPosToOffset(view.state.doc, view.state.selection.from),
      },
      selectionEnd: {
        configurable: true,
        get: () => docPosToOffset(view.state.doc, view.state.selection.to),
      },
      form: { configurable: true, get: () => field.closest("form") },
      setSelectionRange: {
        configurable: true,
        value: (start: number, end: number = start) => {
          const { doc } = view.state;
          const selection = TextSelection.create(
            doc,
            offsetToDocPos(doc, start),
            offsetToDocPos(doc, end),
          );
          view.dispatch(view.state.tr.setSelection(selection));
        },
      },
    });
    syncRef.current = sync;

    measure();
    const observer = new ResizeObserver(() => {
      lastHeight = -1;
      measure();
    });
    observer.observe(field);
    return () => {
      observer.disconnect();
      syncRef.current = null;
      viewRef.current = null;
      for (const key of [
        "value",
        "selectionStart",
        "selectionEnd",
        "form",
        "setSelectionRange",
      ]) {
        Reflect.deleteProperty(field, key);
      }
      view.destroy();
    };
  }, [aui]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: the view reads them through `latest`; these are triggers.
  useLayoutEffect(() => {
    viewRef.current?.setProps({});
  }, [attributesKey, disabled, placeholder, className]);
  // biome-ignore lint/correctness/useExhaustiveDependencies: sync reads the composer itself; value is the trigger.
  useLayoutEffect(() => {
    syncRef.current?.();
  }, [value]);

  return <div ref={setHost} {...events} />;
}
