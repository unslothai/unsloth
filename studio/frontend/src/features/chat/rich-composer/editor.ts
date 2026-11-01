// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { baseKeymap } from "../../../vendor/prosemirror-commands/index.js";
import {
  history,
  redo,
  undo,
} from "../../../vendor/prosemirror-history/index.js";
import {
  inputRules,
  undoInputRule,
} from "../../../vendor/prosemirror-inputrules/index.js";
import { keymap } from "../../../vendor/prosemirror-keymap/index.js";
import type { Node } from "../../../vendor/prosemirror-model/index.js";
import {
  type Command,
  EditorState,
  Plugin,
  Selection,
  TextSelection,
} from "../../../vendor/prosemirror-state/index.js";
import {
  Decoration,
  DecorationSet,
} from "../../../vendor/prosemirror-view/index.js";
import { markerRule } from "./input-rules.ts";
import { composerSchema, markdownSlice, parseMarkdown } from "./markdown.ts";

const {
  paragraph,
  code_block: codeBlock,
  hard_break: hardBreak,
} = composerSchema.nodes;
const { bold, italic, strike, code } = composerSchema.marks;
const OPEN_FENCE = /^```([^\s`]*)$/;

const inCodeBlock = (state: EditorState) =>
  state.selection.$from.parent.type === codeBlock;

// "```lang" alone on a line, caret at its end: the line becomes an empty code block.
const fenceToCodeBlock: Command = (state, dispatch) => {
  const { $from, empty } = state.selection;
  if (!empty || $from.parent.type !== paragraph) return false;
  const para = $from.parent;
  const offset = $from.parentOffset;
  const before = para.textBetween(0, offset, undefined, "\n");
  const lineStart = before.lastIndexOf("\n") + 1;
  const match = OPEN_FENCE.exec(before.slice(lineStart));
  const rest = para.textBetween(offset, para.content.size, undefined, "\n");
  if (!match || (rest && !rest.startsWith("\n"))) return false;
  if (dispatch) {
    const blocks: Node[] = [];
    if (lineStart > 0)
      blocks.push(paragraph.create(null, para.content.cut(0, lineStart - 1)));
    blocks.push(codeBlock.create({ language: match[1] }));
    if (rest) blocks.push(paragraph.create(null, para.content.cut(offset + 1)));
    const tr = state.tr.replaceWith($from.before(), $from.after(), blocks);
    const codeStart =
      $from.before() + (lineStart > 0 ? blocks[0].nodeSize : 0) + 1;
    dispatch(
      tr.setSelection(TextSelection.create(tr.doc, codeStart)).scrollIntoView(),
    );
  }
  return true;
};

// Leaves a code block for the paragraph after it, adding one at the end of the document.
function exitCodeBlock(
  state: EditorState,
  dispatch: Parameters<Command>[1],
  trimEnd: number,
): boolean {
  const { $from } = state.selection;
  if (dispatch) {
    const tr = state.tr;
    if (trimEnd > 0) tr.delete($from.end() - trimEnd, $from.end());
    const after = tr.mapping.map($from.after());
    if (tr.doc.resolve(after).nodeAfter?.type !== paragraph)
      tr.insert(after, paragraph.create());
    dispatch(
      tr.setSelection(TextSelection.create(tr.doc, after + 1)).scrollIntoView(),
    );
  }
  return true;
}

// In a code block, a closing ``` line or a third line break at the end leaves it.
const newlineInCodeBlock: Command = (state, dispatch) => {
  if (!inCodeBlock(state)) return false;
  const { $from, empty } = state.selection;
  if (empty && $from.parentOffset === $from.parent.content.size) {
    const text = $from.parent.textContent;
    if (text === "```" || text.endsWith("\n```"))
      return exitCodeBlock(state, dispatch, text === "```" ? 3 : 4);
    if (text.endsWith("\n\n")) return exitCodeBlock(state, dispatch, 2);
  }
  dispatch?.(state.tr.insertText("\n").scrollIntoView());
  return true;
};

// A line break, never a new paragraph: the composer's text has no paragraphs, only lines.
const insertLineBreak: Command = (state, dispatch) => {
  if (dispatch) {
    const marks = state.storedMarks ?? state.selection.$from.marks();
    const tr = state.tr
      .replaceSelectionWith(hardBreak.create())
      .ensureMarks(marks);
    dispatch(tr.scrollIntoView());
  }
  return true;
};

const newline: Command = (state, dispatch, view) =>
  newlineInCodeBlock(state, dispatch, view) ||
  fenceToCodeBlock(state, dispatch, view) ||
  insertLineBreak(state, dispatch, view);

// A code block first or last in the document has nothing to arrow into, so make the paragraph.
function arrowOutOfCodeBlock(direction: "up" | "down"): Command {
  return (state, dispatch, view) => {
    if (
      !inCodeBlock(state) ||
      !state.selection.empty ||
      !view?.endOfTextblock(direction)
    )
      return false;
    const { $from } = state.selection;
    const index = $from.index(0);
    const edge =
      direction === "down" ? index === state.doc.childCount - 1 : index === 0;
    if (!edge) return false;
    if (dispatch) {
      const at = direction === "down" ? $from.after() : $from.before();
      const tr = state.tr.insert(at, paragraph.create());
      dispatch(
        tr.setSelection(TextSelection.create(tr.doc, at + 1)).scrollIntoView(),
      );
    }
    return true;
  };
}

// Backspace at the start of an empty code block turns it back into a line.
const clearEmptyCodeBlock: Command = (state, dispatch) => {
  const { $from, empty } = state.selection;
  if (
    !empty ||
    $from.parent.type !== codeBlock ||
    $from.parent.content.size > 0
  )
    return false;
  dispatch?.(state.tr.setBlockType($from.before(), $from.after(), paragraph));
  return true;
};

export function placeholderPlugin(text: () => string): Plugin {
  return new Plugin({
    props: {
      decorations(state) {
        const first = state.doc.firstChild;
        if (
          state.doc.childCount !== 1 ||
          !first?.isTextblock ||
          first.content.size > 0
        )
          return null;
        return DecorationSet.create(state.doc, [
          Decoration.node(0, first.nodeSize, {
            class: "is-empty",
            "data-placeholder": text(),
          }),
        ]);
      },
    },
  });
}

export function isEmptyDoc(doc: Node): boolean {
  return (
    doc.childCount === 1 &&
    doc.firstChild?.type === paragraph &&
    doc.firstChild.content.size === 0
  );
}

export function composerPlugins(placeholder: () => string): Plugin[] {
  const mac =
    typeof navigator !== "undefined" &&
    /Mac|iP(?:hone|[ao]d)/.test(navigator.platform);
  return [
    history({ newGroupDelay: 500 }),
    inputRules({
      rules: [
        markerRule("**", bold),
        markerRule("~~", strike),
        markerRule("*", italic),
        markerRule("`", code),
      ],
    }),
    keymap({
      Enter: newline,
      "Shift-Enter": newline,
      "Mod-z": undo,
      "Shift-Mod-z": redo,
      ...(mac ? {} : { "Mod-y": redo }),
      Backspace: (state, dispatch, view) =>
        undoInputRule(state, dispatch, view) ||
        clearEmptyCodeBlock(state, dispatch, view),
      ArrowDown: arrowOutOfCodeBlock("down"),
      ArrowUp: arrowOutOfCodeBlock("up"),
    }),
    keymap(baseKeymap),
    placeholderPlugin(placeholder),
  ];
}

export function createComposerState(
  markdown: string,
  plugins: Plugin[],
): EditorState {
  const doc = parseMarkdown(markdown);
  return EditorState.create({ doc, plugins, selection: Selection.atEnd(doc) });
}

// Pasted or inserted Markdown, styled; inside a code block it stays the text it is.
export function insertMarkdown(state: EditorState, markdown: string) {
  if (inCodeBlock(state)) return state.tr.insertText(markdown).scrollIntoView();
  return state.tr.replaceSelection(markdownSlice(markdown)).scrollIntoView();
}

export { inCodeBlock };
