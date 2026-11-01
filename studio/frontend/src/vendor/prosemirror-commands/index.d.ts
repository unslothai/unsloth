// prosemirror-commands 1.7.2 from npm (sha512-q6Q6szxqdu9Xd6EcdKsqXghu5nQdZTpB4Q9yd04WRc7/jt763e/rT60Owh0L1GYY+T46o5rD+9lEN36dZS43tw==), MIT: see LICENSE. Imports point at the sibling vendored packages.
import { Node, ResolvedPos, NodeType, Attrs, MarkType } from '../prosemirror-model/index.js';
import { Command } from '../prosemirror-state/index.js';

/**
Delete the selection, if there is one.
*/
declare const deleteSelection: Command;
/**
If the selection is empty and at the start of a textblock, try to
reduce the distance between that block and the one before it—if
there's a block directly before it that can be joined, join them.
If not, try to move the selected block closer to the next one in
the document structure by lifting it out of its parent or moving it
into a parent of the previous block. Will use the view for accurate
(bidi-aware) start-of-textblock detection if given.
*/
declare const joinBackward: Command;
/**
A more limited form of [`joinBackward`](https://prosemirror.net/docs/ref/#commands.joinBackward)
that only tries to join the current textblock to the one before
it, if the cursor is at the start of a textblock.
*/
declare const joinTextblockBackward: Command;
/**
A more limited form of [`joinForward`](https://prosemirror.net/docs/ref/#commands.joinForward)
that only tries to join the current textblock to the one after
it, if the cursor is at the end of a textblock.
*/
declare const joinTextblockForward: Command;
/**
When the selection is empty and at the start of a textblock, select
the node before that textblock, if possible. This is intended to be
bound to keys like backspace, after
[`joinBackward`](https://prosemirror.net/docs/ref/#commands.joinBackward) or other deleting
commands, as a fall-back behavior when the schema doesn't allow
deletion at the selected point.
*/
declare const selectNodeBackward: Command;
/**
If the selection is empty and the cursor is at the end of a
textblock, try to reduce or remove the boundary between that block
and the one after it, either by joining them or by moving the other
block closer to this one in the tree structure. Will use the view
for accurate start-of-textblock detection if given.
*/
declare const joinForward: Command;
/**
When the selection is empty and at the end of a textblock, select
the node coming after that textblock, if possible. This is intended
to be bound to keys like delete, after
[`joinForward`](https://prosemirror.net/docs/ref/#commands.joinForward) and similar deleting
commands, to provide a fall-back behavior when the schema doesn't
allow deletion at the selected point.
*/
declare const selectNodeForward: Command;
/**
Join the selected block or, if there is a text selection, the
closest ancestor block of the selection that can be joined, with
the sibling above it.
*/
declare const joinUp: Command;
/**
Join the selected block, or the closest ancestor of the selection
that can be joined, with the sibling after it.
*/
declare const joinDown: Command;
/**
Lift the selected block, or the closest ancestor block of the
selection that can be lifted, out of its parent node.
*/
declare const lift: Command;
/**
If the selection is in a node whose type has a truthy
[`code`](https://prosemirror.net/docs/ref/#model.NodeSpec.code) property in its spec, replace the
selection with a newline character.
*/
declare const newlineInCode: Command;
/**
When the selection is in a node with a truthy
[`code`](https://prosemirror.net/docs/ref/#model.NodeSpec.code) property in its spec, create a
default block after the code block, and move the cursor there.
*/
declare const exitCode: Command;
/**
If a block node is selected, create an empty paragraph before (if
it is its parent's first child) or after it.
*/
declare const createParagraphNear: Command;
/**
If the cursor is in an empty textblock that can be lifted, lift the
block.
*/
declare const liftEmptyBlock: Command;
/**
Create a variant of [`splitBlock`](https://prosemirror.net/docs/ref/#commands.splitBlock) that uses
a custom function to determine the type of the newly split off block.
*/
declare function splitBlockAs(splitNode?: (node: Node, atEnd: boolean, $from: ResolvedPos) => {
    type: NodeType;
    attrs?: Attrs;
} | null): Command;
/**
Split the parent block of the selection. If the selection is a text
selection, also delete its content.
*/
declare const splitBlock: Command;
/**
Acts like [`splitBlock`](https://prosemirror.net/docs/ref/#commands.splitBlock), but without
resetting the set of active marks at the cursor.
*/
declare const splitBlockKeepMarks: Command;
/**
Move the selection to the node wrapping the current selection, if
any. (Will not select the document node.)
*/
declare const selectParentNode: Command;
/**
Select the whole document.
*/
declare const selectAll: Command;
/**
Moves the cursor to the start of current text block.
*/
declare const selectTextblockStart: Command;
/**
Moves the cursor to the end of current text block.
*/
declare const selectTextblockEnd: Command;
/**
Wrap the selection in a node of the given type with the given
attributes.
*/
declare function wrapIn(nodeType: NodeType, attrs?: Attrs | null): Command;
/**
Returns a command that tries to set the selected textblocks to the
given node type with the given attributes.
*/
declare function setBlockType(nodeType: NodeType, attrs?: Attrs | null): Command;
/**
Create a command function that toggles the given mark with the
given attributes. Will return `false` when the current selection
doesn't support that mark. This will remove the mark if any marks
of that type exist in the selection, or add it otherwise. If the
selection is empty, this applies to the [stored
marks](https://prosemirror.net/docs/ref/#state.EditorState.storedMarks) instead of a range of the
document.
*/
declare function toggleMark(markType: MarkType, attrs?: Attrs | null, options?: {
    /**
    Controls whether, when part of the selected range has the mark
    already and part doesn't, the mark is removed (`true`, the
    default) or added (`false`).
    */
    removeWhenPresent?: boolean;
    /**
    When set to false, this will prevent the command from acting on
    the content of inline nodes marked as
    [atoms](https://prosemirror.net/docs/ref/#model.NodeSpec.atom) that are completely covered by a
    selection range.
    */
    enterInlineAtoms?: boolean;
    /**
    By default, this command doesn't apply to leading and trailing
    whitespace in the selection. Set this to `true` to change that.
    */
    includeWhitespace?: boolean;
}): Command;
/**
Wrap a command so that, when it produces a transform that causes
two joinable nodes to end up next to each other, those are joined.
Nodes are considered joinable when they are of the same type and
when the `isJoinable` predicate returns true for them or, if an
array of strings was passed, if their node type name is in that
array.
*/
declare function autoJoin(command: Command, isJoinable: ((before: Node, after: Node) => boolean) | readonly string[]): Command;
/**
Combine a number of command functions into a single function (which
calls them one by one until one returns true).
*/
declare function chainCommands(...commands: readonly Command[]): Command;
/**
A basic keymap containing bindings not specific to any schema.
Binds the following keys (when multiple commands are listed, they
are chained with [`chainCommands`](https://prosemirror.net/docs/ref/#commands.chainCommands)):

* **Enter** to `newlineInCode`, `createParagraphNear`, `liftEmptyBlock`, `splitBlock`
* **Mod-Enter** to `exitCode`
* **Backspace** and **Mod-Backspace** to `deleteSelection`, `joinBackward`, `selectNodeBackward`
* **Delete** and **Mod-Delete** to `deleteSelection`, `joinForward`, `selectNodeForward`
* **Mod-Delete** to `deleteSelection`, `joinForward`, `selectNodeForward`
* **Mod-a** to `selectAll`
*/
declare const pcBaseKeymap: {
    [key: string]: Command;
};
/**
A copy of `pcBaseKeymap` that also binds **Ctrl-h** like Backspace,
**Ctrl-d** like Delete, **Alt-Backspace** like Ctrl-Backspace, and
**Ctrl-Alt-Backspace**, **Alt-Delete**, and **Alt-d** like
Ctrl-Delete.
*/
declare const macBaseKeymap: {
    [key: string]: Command;
};
/**
Depending on the detected platform, this will hold
[`pcBasekeymap`](https://prosemirror.net/docs/ref/#commands.pcBaseKeymap) or
[`macBaseKeymap`](https://prosemirror.net/docs/ref/#commands.macBaseKeymap).
*/
declare const baseKeymap: {
    [key: string]: Command;
};

export { autoJoin, baseKeymap, chainCommands, createParagraphNear, deleteSelection, exitCode, joinBackward, joinDown, joinForward, joinTextblockBackward, joinTextblockForward, joinUp, lift, liftEmptyBlock, macBaseKeymap, newlineInCode, pcBaseKeymap, selectAll, selectNodeBackward, selectNodeForward, selectParentNode, selectTextblockEnd, selectTextblockStart, setBlockType, splitBlock, splitBlockAs, splitBlockKeepMarks, toggleMark, wrapIn };
