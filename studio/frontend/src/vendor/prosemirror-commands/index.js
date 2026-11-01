// prosemirror-commands 1.7.2 from npm (sha512-q6Q6szxqdu9Xd6EcdKsqXghu5nQdZTpB4Q9yd04WRc7/jt763e/rT60Owh0L1GYY+T46o5rD+9lEN36dZS43tw==), MIT: see LICENSE. Imports point at the sibling vendored packages.
import { liftTarget, replaceStep, ReplaceStep, canJoin, joinPoint, canSplit, ReplaceAroundStep, findWrapping } from '../prosemirror-transform/index.js';
import { Slice, Fragment } from '../prosemirror-model/index.js';
import { NodeSelection, Selection, TextSelection, AllSelection, SelectionRange } from '../prosemirror-state/index.js';

/**
Delete the selection, if there is one.
*/
const deleteSelection = (state, dispatch) => {
    if (state.selection.empty)
        return false;
    if (dispatch)
        dispatch(state.tr.deleteSelection().scrollIntoView());
    return true;
};
function atBlockStart(state, view) {
    let { $cursor } = state.selection;
    if (!$cursor || (view ? !view.endOfTextblock("backward", state)
        : $cursor.parentOffset > 0))
        return null;
    return $cursor;
}
/**
If the selection is empty and at the start of a textblock, try to
reduce the distance between that block and the one before it—if
there's a block directly before it that can be joined, join them.
If not, try to move the selected block closer to the next one in
the document structure by lifting it out of its parent or moving it
into a parent of the previous block. Will use the view for accurate
(bidi-aware) start-of-textblock detection if given.
*/
const joinBackward = (state, dispatch, view) => {
    let $cursor = atBlockStart(state, view);
    if (!$cursor)
        return false;
    let $cut = findCutBefore($cursor);
    // If there is no node before this, try to lift
    if (!$cut) {
        let range = $cursor.blockRange(), target = range && liftTarget(range);
        if (target == null)
            return false;
        if (dispatch)
            dispatch(state.tr.lift(range, target).scrollIntoView());
        return true;
    }
    let before = $cut.nodeBefore;
    // Apply the joining algorithm
    if (deleteBarrier(state, $cut, dispatch, -1))
        return true;
    // If the node below has no content and the node above is
    // selectable, delete the node below and select the one above.
    if ($cursor.parent.content.size == 0 &&
        (textblockAt(before, "end") || NodeSelection.isSelectable(before))) {
        for (let depth = $cursor.depth;; depth--) {
            let delStep = replaceStep(state.doc, $cursor.before(depth), $cursor.after(depth), Slice.empty);
            if (delStep && delStep.slice.size < delStep.to - delStep.from) {
                if (dispatch) {
                    let tr = state.tr.step(delStep);
                    tr.setSelection(textblockAt(before, "end")
                        ? Selection.findFrom(tr.doc.resolve(tr.mapping.map($cut.pos, -1)), -1)
                        : NodeSelection.create(tr.doc, $cut.pos - before.nodeSize));
                    dispatch(tr.scrollIntoView());
                }
                return true;
            }
            if (depth == 1 || $cursor.node(depth - 1).childCount > 1)
                break;
        }
    }
    // If the node before is an atom, delete it
    if (before.isAtom && $cut.depth == $cursor.depth - 1) {
        if (dispatch)
            dispatch(state.tr.delete($cut.pos - before.nodeSize, $cut.pos).scrollIntoView());
        return true;
    }
    return false;
};
/**
A more limited form of [`joinBackward`](https://prosemirror.net/docs/ref/#commands.joinBackward)
that only tries to join the current textblock to the one before
it, if the cursor is at the start of a textblock.
*/
const joinTextblockBackward = (state, dispatch, view) => {
    let $cursor = atBlockStart(state, view);
    if (!$cursor)
        return false;
    let $cut = findCutBefore($cursor);
    return $cut ? joinTextblocksAround(state, $cut, dispatch) : false;
};
/**
A more limited form of [`joinForward`](https://prosemirror.net/docs/ref/#commands.joinForward)
that only tries to join the current textblock to the one after
it, if the cursor is at the end of a textblock.
*/
const joinTextblockForward = (state, dispatch, view) => {
    let $cursor = atBlockEnd(state, view);
    if (!$cursor)
        return false;
    let $cut = findCutAfter($cursor);
    return $cut ? joinTextblocksAround(state, $cut, dispatch) : false;
};
function joinTextblocksAround(state, $cut, dispatch) {
    let before = $cut.nodeBefore, beforeText = before, beforePos = $cut.pos - 1;
    for (; !beforeText.isTextblock; beforePos--) {
        if (beforeText.type.spec.isolating)
            return false;
        let child = beforeText.lastChild;
        if (!child)
            return false;
        beforeText = child;
    }
    let after = $cut.nodeAfter, afterText = after, afterPos = $cut.pos + 1;
    for (; !afterText.isTextblock; afterPos++) {
        if (afterText.type.spec.isolating)
            return false;
        let child = afterText.firstChild;
        if (!child)
            return false;
        afterText = child;
    }
    let step = replaceStep(state.doc, beforePos, afterPos, Slice.empty);
    if (!step || step.from != beforePos ||
        step instanceof ReplaceStep && step.slice.size >= afterPos - beforePos)
        return false;
    if (dispatch) {
        let tr = state.tr.step(step);
        tr.setSelection(TextSelection.create(tr.doc, beforePos));
        dispatch(tr.scrollIntoView());
    }
    return true;
}
function textblockAt(node, side, only = false) {
    for (let scan = node; scan; scan = (side == "start" ? scan.firstChild : scan.lastChild)) {
        if (scan.isTextblock)
            return true;
        if (only && scan.childCount != 1)
            return false;
    }
    return false;
}
/**
When the selection is empty and at the start of a textblock, select
the node before that textblock, if possible. This is intended to be
bound to keys like backspace, after
[`joinBackward`](https://prosemirror.net/docs/ref/#commands.joinBackward) or other deleting
commands, as a fall-back behavior when the schema doesn't allow
deletion at the selected point.
*/
const selectNodeBackward = (state, dispatch, view) => {
    let { $head, empty } = state.selection, $cut = $head;
    if (!empty)
        return false;
    if ($head.parent.isTextblock) {
        if (view ? !view.endOfTextblock("backward", state) : $head.parentOffset > 0)
            return false;
        $cut = findCutBefore($head);
    }
    let node = $cut && $cut.nodeBefore;
    if (!node || !NodeSelection.isSelectable(node))
        return false;
    if (dispatch)
        dispatch(state.tr.setSelection(NodeSelection.create(state.doc, $cut.pos - node.nodeSize)).scrollIntoView());
    return true;
};
function findCutBefore($pos) {
    if (!$pos.parent.type.spec.isolating)
        for (let i = $pos.depth - 1; i >= 0; i--) {
            if ($pos.index(i) > 0)
                return $pos.doc.resolve($pos.before(i + 1));
            if ($pos.node(i).type.spec.isolating)
                break;
        }
    return null;
}
function atBlockEnd(state, view) {
    let { $cursor } = state.selection;
    if (!$cursor || (view ? !view.endOfTextblock("forward", state)
        : $cursor.parentOffset < $cursor.parent.content.size))
        return null;
    return $cursor;
}
/**
If the selection is empty and the cursor is at the end of a
textblock, try to reduce or remove the boundary between that block
and the one after it, either by joining them or by moving the other
block closer to this one in the tree structure. Will use the view
for accurate start-of-textblock detection if given.
*/
const joinForward = (state, dispatch, view) => {
    let $cursor = atBlockEnd(state, view);
    if (!$cursor)
        return false;
    let $cut = findCutAfter($cursor);
    // If there is no node after this, there's nothing to do
    if (!$cut)
        return false;
    let after = $cut.nodeAfter;
    // Try the joining algorithm
    if (deleteBarrier(state, $cut, dispatch, 1))
        return true;
    // If the node above has no content and the node below is
    // selectable, delete the node above and select the one below.
    if ($cursor.parent.content.size == 0 &&
        (textblockAt(after, "start") || NodeSelection.isSelectable(after))) {
        let delStep = replaceStep(state.doc, $cursor.before(), $cursor.after(), Slice.empty);
        if (delStep && delStep.slice.size < delStep.to - delStep.from) {
            if (dispatch) {
                let tr = state.tr.step(delStep);
                tr.setSelection(textblockAt(after, "start") ? Selection.findFrom(tr.doc.resolve(tr.mapping.map($cut.pos)), 1)
                    : NodeSelection.create(tr.doc, tr.mapping.map($cut.pos)));
                dispatch(tr.scrollIntoView());
            }
            return true;
        }
    }
    // If the next node is an atom, delete it
    if (after.isAtom && $cut.depth == $cursor.depth - 1) {
        if (dispatch)
            dispatch(state.tr.delete($cut.pos, $cut.pos + after.nodeSize).scrollIntoView());
        return true;
    }
    return false;
};
/**
When the selection is empty and at the end of a textblock, select
the node coming after that textblock, if possible. This is intended
to be bound to keys like delete, after
[`joinForward`](https://prosemirror.net/docs/ref/#commands.joinForward) and similar deleting
commands, to provide a fall-back behavior when the schema doesn't
allow deletion at the selected point.
*/
const selectNodeForward = (state, dispatch, view) => {
    let { $head, empty } = state.selection, $cut = $head;
    if (!empty)
        return false;
    if ($head.parent.isTextblock) {
        if (view ? !view.endOfTextblock("forward", state) : $head.parentOffset < $head.parent.content.size)
            return false;
        $cut = findCutAfter($head);
    }
    let node = $cut && $cut.nodeAfter;
    if (!node || !NodeSelection.isSelectable(node))
        return false;
    if (dispatch)
        dispatch(state.tr.setSelection(NodeSelection.create(state.doc, $cut.pos)).scrollIntoView());
    return true;
};
function findCutAfter($pos) {
    if (!$pos.parent.type.spec.isolating)
        for (let i = $pos.depth - 1; i >= 0; i--) {
            let parent = $pos.node(i);
            if ($pos.index(i) + 1 < parent.childCount)
                return $pos.doc.resolve($pos.after(i + 1));
            if (parent.type.spec.isolating)
                break;
        }
    return null;
}
/**
Join the selected block or, if there is a text selection, the
closest ancestor block of the selection that can be joined, with
the sibling above it.
*/
const joinUp = (state, dispatch) => {
    let sel = state.selection, nodeSel = sel instanceof NodeSelection, point;
    if (nodeSel) {
        if (sel.node.isTextblock || !canJoin(state.doc, sel.from))
            return false;
        point = sel.from;
    }
    else {
        point = joinPoint(state.doc, sel.from, -1);
        if (point == null)
            return false;
    }
    if (dispatch) {
        let tr = state.tr.join(point);
        if (nodeSel)
            tr.setSelection(NodeSelection.create(tr.doc, point - state.doc.resolve(point).nodeBefore.nodeSize));
        dispatch(tr.scrollIntoView());
    }
    return true;
};
/**
Join the selected block, or the closest ancestor of the selection
that can be joined, with the sibling after it.
*/
const joinDown = (state, dispatch) => {
    let sel = state.selection, point;
    if (sel instanceof NodeSelection) {
        if (sel.node.isTextblock || !canJoin(state.doc, sel.to))
            return false;
        point = sel.to;
    }
    else {
        point = joinPoint(state.doc, sel.to, 1);
        if (point == null)
            return false;
    }
    if (dispatch)
        dispatch(state.tr.join(point).scrollIntoView());
    return true;
};
/**
Lift the selected block, or the closest ancestor block of the
selection that can be lifted, out of its parent node.
*/
const lift = (state, dispatch) => {
    let { $from, $to } = state.selection;
    let range = $from.blockRange($to), target = range && liftTarget(range);
    if (target == null)
        return false;
    if (dispatch)
        dispatch(state.tr.lift(range, target).scrollIntoView());
    return true;
};
/**
If the selection is in a node whose type has a truthy
[`code`](https://prosemirror.net/docs/ref/#model.NodeSpec.code) property in its spec, replace the
selection with a newline character.
*/
const newlineInCode = (state, dispatch) => {
    let { $head, $anchor } = state.selection;
    if (!$head.parent.type.spec.code || !$head.sameParent($anchor))
        return false;
    if (dispatch)
        dispatch(state.tr.insertText("\n").scrollIntoView());
    return true;
};
function defaultBlockAt(match) {
    for (let i = 0; i < match.edgeCount; i++) {
        let { type } = match.edge(i);
        if (type.isTextblock && !type.hasRequiredAttrs())
            return type;
    }
    return null;
}
/**
When the selection is in a node with a truthy
[`code`](https://prosemirror.net/docs/ref/#model.NodeSpec.code) property in its spec, create a
default block after the code block, and move the cursor there.
*/
const exitCode = (state, dispatch) => {
    let { $head, $anchor } = state.selection;
    if (!$head.parent.type.spec.code || !$head.sameParent($anchor))
        return false;
    let above = $head.node(-1), after = $head.indexAfter(-1), type = defaultBlockAt(above.contentMatchAt(after));
    if (!type || !above.canReplaceWith(after, after, type))
        return false;
    if (dispatch) {
        let pos = $head.after(), tr = state.tr.replaceWith(pos, pos, type.createAndFill());
        tr.setSelection(Selection.near(tr.doc.resolve(pos), 1));
        dispatch(tr.scrollIntoView());
    }
    return true;
};
/**
If a block node is selected, create an empty paragraph before (if
it is its parent's first child) or after it.
*/
const createParagraphNear = (state, dispatch) => {
    let sel = state.selection, { $from, $to } = sel;
    if (sel instanceof AllSelection || $from.parent.inlineContent || $to.parent.inlineContent)
        return false;
    let type = defaultBlockAt($to.parent.contentMatchAt($to.indexAfter()));
    if (!type || !type.isTextblock)
        return false;
    if (dispatch) {
        let side = (!$from.parentOffset && $to.index() < $to.parent.childCount ? $from : $to).pos;
        let tr = state.tr.insert(side, type.createAndFill());
        tr.setSelection(TextSelection.create(tr.doc, side + 1));
        dispatch(tr.scrollIntoView());
    }
    return true;
};
/**
If the cursor is in an empty textblock that can be lifted, lift the
block.
*/
const liftEmptyBlock = (state, dispatch) => {
    let { $cursor } = state.selection;
    if (!$cursor || $cursor.parent.content.size)
        return false;
    if ($cursor.depth > 1 && $cursor.after() != $cursor.end(-1)) {
        let before = $cursor.before();
        if (canSplit(state.doc, before)) {
            if (dispatch)
                dispatch(state.tr.split(before).scrollIntoView());
            return true;
        }
    }
    let range = $cursor.blockRange(), target = range && liftTarget(range);
    if (target == null)
        return false;
    if (dispatch)
        dispatch(state.tr.lift(range, target).scrollIntoView());
    return true;
};
/**
Create a variant of [`splitBlock`](https://prosemirror.net/docs/ref/#commands.splitBlock) that uses
a custom function to determine the type of the newly split off block.
*/
function splitBlockAs(splitNode) {
    return (state, dispatch) => {
        if (state.selection instanceof NodeSelection && state.selection.node.isBlock) {
            let { $from } = state.selection;
            if (!$from.parentOffset || !canSplit(state.doc, $from.pos))
                return false;
            if (dispatch)
                dispatch(state.tr.split($from.pos).scrollIntoView());
            return true;
        }
        if (!state.selection.$from.depth)
            return false;
        let tr = state.tr;
        if (!state.selection.empty && (state.selection instanceof TextSelection || state.selection instanceof AllSelection))
            tr.deleteSelection();
        let { $from } = tr.selection, mapFrom = tr.steps.length;
        let types = [];
        let splitDepth, deflt, atEnd = false, atStart = false;
        for (let d = $from.depth;; d--) {
            let node = $from.node(d);
            if (node.isBlock) {
                atEnd = $from.end(d) == $from.pos + ($from.depth - d);
                atStart = $from.start(d) == $from.pos - ($from.depth - d);
                deflt = defaultBlockAt($from.node(d - 1).contentMatchAt($from.indexAfter(d - 1)));
                let splitType = splitNode && splitNode($from.parent, atEnd, $from);
                types.unshift(splitType || (atEnd && deflt ? { type: deflt } : null));
                splitDepth = d;
                break;
            }
            else {
                if (d == 1)
                    return false;
                types.unshift(null);
            }
        }
        let splitPos = $from.pos;
        let can = canSplit(tr.doc, splitPos, types.length, types);
        if (!can) {
            types[0] = deflt ? { type: deflt } : null;
            can = canSplit(tr.doc, splitPos, types.length, types);
        }
        if (!can)
            return false;
        tr.split(splitPos, types.length, types);
        if (!atEnd && atStart && $from.node(splitDepth).type != deflt) {
            let mapping = tr.mapping.slice(mapFrom);
            let first = mapping.map($from.before(splitDepth)), $first = tr.doc.resolve(first);
            if (deflt && $from.node(splitDepth - 1).canReplaceWith($first.index(), $first.index() + 1, deflt))
                tr.setNodeMarkup(mapping.map($from.before(splitDepth)), deflt);
        }
        if (dispatch)
            dispatch(tr.scrollIntoView());
        return true;
    };
}
/**
Split the parent block of the selection. If the selection is a text
selection, also delete its content.
*/
const splitBlock = splitBlockAs();
/**
Acts like [`splitBlock`](https://prosemirror.net/docs/ref/#commands.splitBlock), but without
resetting the set of active marks at the cursor.
*/
const splitBlockKeepMarks = (state, dispatch) => {
    return splitBlock(state, dispatch && (tr => {
        let marks = state.storedMarks || (state.selection.$to.parentOffset && state.selection.$from.marks());
        if (marks)
            tr.ensureMarks(marks);
        dispatch(tr);
    }));
};
/**
Move the selection to the node wrapping the current selection, if
any. (Will not select the document node.)
*/
const selectParentNode = (state, dispatch) => {
    let { $from, to } = state.selection, pos;
    let same = $from.sharedDepth(to);
    if (same == 0)
        return false;
    pos = $from.before(same);
    if (dispatch)
        dispatch(state.tr.setSelection(NodeSelection.create(state.doc, pos)));
    return true;
};
/**
Select the whole document.
*/
const selectAll = (state, dispatch) => {
    if (dispatch)
        dispatch(state.tr.setSelection(new AllSelection(state.doc)));
    return true;
};
function joinMaybeClear(state, $pos, dispatch) {
    let before = $pos.nodeBefore, after = $pos.nodeAfter, index = $pos.index();
    if (!before || !after || !before.type.compatibleContent(after.type))
        return false;
    if (!before.content.size && $pos.parent.canReplace(index - 1, index)) {
        if (dispatch)
            dispatch(state.tr.delete($pos.pos - before.nodeSize, $pos.pos).scrollIntoView());
        return true;
    }
    if (!$pos.parent.canReplace(index, index + 1) || !(after.isTextblock || canJoin(state.doc, $pos.pos)))
        return false;
    if (dispatch)
        dispatch(state.tr.join($pos.pos).scrollIntoView());
    return true;
}
function deleteBarrier(state, $cut, dispatch, dir) {
    let before = $cut.nodeBefore, after = $cut.nodeAfter, conn, match;
    let isolated = before.type.spec.isolating || after.type.spec.isolating;
    if (!isolated && joinMaybeClear(state, $cut, dispatch))
        return true;
    let canDelAfter = !isolated && $cut.parent.canReplace($cut.index(), $cut.index() + 1);
    if (canDelAfter &&
        (conn = (match = before.contentMatchAt(before.childCount)).findWrapping(after.type)) &&
        match.matchType(conn[0] || after.type).validEnd) {
        if (dispatch) {
            let end = $cut.pos + after.nodeSize, wrap = Fragment.empty;
            for (let i = conn.length - 1; i >= 0; i--)
                wrap = Fragment.from(conn[i].create(null, wrap));
            wrap = Fragment.from(before.copy(wrap));
            let tr = state.tr.step(new ReplaceAroundStep($cut.pos - 1, end, $cut.pos, end, new Slice(wrap, 1, 0), conn.length, true));
            let $joinAt = tr.doc.resolve(end + 2 * conn.length);
            if ($joinAt.nodeAfter && $joinAt.nodeAfter.type == before.type &&
                canJoin(tr.doc, $joinAt.pos))
                tr.join($joinAt.pos);
            dispatch(tr.scrollIntoView());
        }
        return true;
    }
    let selAfter = after.type.spec.isolating || (dir > 0 && isolated) ? null : Selection.findFrom($cut, 1);
    let range = selAfter && selAfter.$from.blockRange(selAfter.$to), target = range && liftTarget(range);
    if (target != null && target >= $cut.depth) {
        if (dispatch)
            dispatch(state.tr.lift(range, target).scrollIntoView());
        return true;
    }
    if (canDelAfter && textblockAt(after, "start", true) && textblockAt(before, "end")) {
        let at = before, wrap = [];
        for (;;) {
            wrap.push(at);
            if (at.isTextblock)
                break;
            at = at.lastChild;
        }
        let afterText = after, afterDepth = 1;
        for (; !afterText.isTextblock; afterText = afterText.firstChild)
            afterDepth++;
        if (at.canReplace(at.childCount, at.childCount, afterText.content)) {
            if (dispatch) {
                let end = Fragment.empty;
                for (let i = wrap.length - 1; i >= 0; i--)
                    end = Fragment.from(wrap[i].copy(end));
                let tr = state.tr.step(new ReplaceAroundStep($cut.pos - wrap.length, $cut.pos + after.nodeSize, $cut.pos + afterDepth, $cut.pos + after.nodeSize - afterDepth, new Slice(end, wrap.length, 0), 0, true));
                dispatch(tr.scrollIntoView());
            }
            return true;
        }
    }
    return false;
}
function selectTextblockSide(side) {
    return function (state, dispatch) {
        let sel = state.selection, $pos = side < 0 ? sel.$from : sel.$to;
        let depth = $pos.depth;
        while ($pos.node(depth).isInline) {
            if (!depth)
                return false;
            depth--;
        }
        if (!$pos.node(depth).isTextblock)
            return false;
        if (dispatch)
            dispatch(state.tr.setSelection(TextSelection.create(state.doc, side < 0 ? $pos.start(depth) : $pos.end(depth))));
        return true;
    };
}
/**
Moves the cursor to the start of current text block.
*/
const selectTextblockStart = selectTextblockSide(-1);
/**
Moves the cursor to the end of current text block.
*/
const selectTextblockEnd = selectTextblockSide(1);
// Parameterized commands
/**
Wrap the selection in a node of the given type with the given
attributes.
*/
function wrapIn(nodeType, attrs = null) {
    return function (state, dispatch) {
        let { $from, $to } = state.selection;
        let range = $from.blockRange($to), wrapping = range && findWrapping(range, nodeType, attrs);
        if (!wrapping)
            return false;
        if (dispatch)
            dispatch(state.tr.wrap(range, wrapping).scrollIntoView());
        return true;
    };
}
/**
Returns a command that tries to set the selected textblocks to the
given node type with the given attributes.
*/
function setBlockType(nodeType, attrs = null) {
    return function (state, dispatch) {
        let applicable = false;
        for (let i = 0; i < state.selection.ranges.length && !applicable; i++) {
            let { $from: { pos: from }, $to: { pos: to } } = state.selection.ranges[i];
            state.doc.nodesBetween(from, to, (node, pos) => {
                if (applicable)
                    return false;
                if (!node.isTextblock || node.hasMarkup(nodeType, attrs))
                    return;
                if (node.type == nodeType) {
                    applicable = true;
                }
                else {
                    let $pos = state.doc.resolve(pos), index = $pos.index();
                    applicable = $pos.parent.canReplaceWith(index, index + 1, nodeType);
                }
            });
        }
        if (!applicable)
            return false;
        if (dispatch) {
            let tr = state.tr;
            for (let i = 0; i < state.selection.ranges.length; i++) {
                let { $from: { pos: from }, $to: { pos: to } } = state.selection.ranges[i];
                tr.setBlockType(from, to, nodeType, attrs);
            }
            dispatch(tr.scrollIntoView());
        }
        return true;
    };
}
function markApplies(doc, ranges, type, enterAtoms) {
    for (let i = 0; i < ranges.length; i++) {
        let { $from, $to } = ranges[i];
        let can = $from.depth == 0 ? doc.inlineContent && doc.type.allowsMarkType(type) : false;
        doc.nodesBetween($from.pos, $to.pos, (node, pos) => {
            if (can || !enterAtoms && node.isAtom && node.isInline && pos >= $from.pos && pos + node.nodeSize <= $to.pos)
                return false;
            can = node.inlineContent && node.type.allowsMarkType(type);
        });
        if (can)
            return true;
    }
    return false;
}
function removeInlineAtoms(ranges) {
    let result = [];
    for (let i = 0; i < ranges.length; i++) {
        let { $from, $to } = ranges[i];
        $from.doc.nodesBetween($from.pos, $to.pos, (node, pos) => {
            if (node.isAtom && node.content.size && node.isInline && pos >= $from.pos && pos + node.nodeSize <= $to.pos) {
                if (pos + 1 > $from.pos)
                    result.push(new SelectionRange($from, $from.doc.resolve(pos + 1)));
                $from = $from.doc.resolve(pos + 1 + node.content.size);
                return false;
            }
        });
        if ($from.pos < $to.pos)
            result.push(new SelectionRange($from, $to));
    }
    return result;
}
/**
Create a command function that toggles the given mark with the
given attributes. Will return `false` when the current selection
doesn't support that mark. This will remove the mark if any marks
of that type exist in the selection, or add it otherwise. If the
selection is empty, this applies to the [stored
marks](https://prosemirror.net/docs/ref/#state.EditorState.storedMarks) instead of a range of the
document.
*/
function toggleMark(markType, attrs = null, options) {
    let removeWhenPresent = (options && options.removeWhenPresent) !== false;
    let enterAtoms = (options && options.enterInlineAtoms) !== false;
    let dropSpace = !(options && options.includeWhitespace);
    return function (state, dispatch) {
        let { empty, $cursor, ranges } = state.selection;
        if ((empty && !$cursor) || !markApplies(state.doc, ranges, markType, enterAtoms))
            return false;
        if (dispatch) {
            if ($cursor) {
                if (markType.isInSet(state.storedMarks || $cursor.marks()))
                    dispatch(state.tr.removeStoredMark(markType));
                else
                    dispatch(state.tr.addStoredMark(markType.create(attrs)));
            }
            else {
                let add, tr = state.tr;
                if (!enterAtoms)
                    ranges = removeInlineAtoms(ranges);
                if (removeWhenPresent) {
                    add = !ranges.some(r => state.doc.rangeHasMark(r.$from.pos, r.$to.pos, markType));
                }
                else {
                    add = !ranges.every(r => {
                        let missing = false;
                        tr.doc.nodesBetween(r.$from.pos, r.$to.pos, (node, pos, parent) => {
                            if (missing)
                                return false;
                            missing = !markType.isInSet(node.marks) && !!parent && parent.type.allowsMarkType(markType) &&
                                !(node.isText && /^\s*$/.test(node.textBetween(Math.max(0, r.$from.pos - pos), Math.min(node.nodeSize, r.$to.pos - pos))));
                        });
                        return !missing;
                    });
                }
                for (let i = 0; i < ranges.length; i++) {
                    let { $from, $to } = ranges[i];
                    if (!add) {
                        tr.removeMark($from.pos, $to.pos, markType);
                    }
                    else {
                        let from = $from.pos, to = $to.pos, start = $from.nodeAfter, end = $to.nodeBefore;
                        let spaceStart = dropSpace && start && start.isText ? /^\s*/.exec(start.text)[0].length : 0;
                        let spaceEnd = dropSpace && end && end.isText ? /\s*$/.exec(end.text)[0].length : 0;
                        if (from + spaceStart < to) {
                            from += spaceStart;
                            to -= spaceEnd;
                        }
                        tr.addMark(from, to, markType.create(attrs));
                    }
                }
                dispatch(tr.scrollIntoView());
            }
        }
        return true;
    };
}
function wrapDispatchForJoin(dispatch, isJoinable) {
    return (tr) => {
        if (!tr.isGeneric)
            return dispatch(tr);
        let ranges = [];
        for (let i = 0; i < tr.mapping.maps.length; i++) {
            let map = tr.mapping.maps[i];
            for (let j = 0; j < ranges.length; j++)
                ranges[j] = map.map(ranges[j]);
            map.forEach((_s, _e, from, to) => ranges.push(from, to));
        }
        // Figure out which joinable points exist inside those ranges,
        // by checking all node boundaries in their parent nodes.
        let joinable = [];
        for (let i = 0; i < ranges.length; i += 2) {
            let from = ranges[i], to = ranges[i + 1];
            let $from = tr.doc.resolve(from), depth = $from.sharedDepth(to), parent = $from.node(depth);
            for (let index = $from.indexAfter(depth), pos = $from.after(depth + 1); pos <= to; ++index) {
                let after = parent.maybeChild(index);
                if (!after)
                    break;
                if (index && joinable.indexOf(pos) == -1) {
                    let before = parent.child(index - 1);
                    if (before.type == after.type && isJoinable(before, after))
                        joinable.push(pos);
                }
                pos += after.nodeSize;
            }
        }
        // Join the joinable points
        joinable.sort((a, b) => a - b);
        for (let i = joinable.length - 1; i >= 0; i--) {
            if (canJoin(tr.doc, joinable[i]))
                tr.join(joinable[i]);
        }
        dispatch(tr);
    };
}
/**
Wrap a command so that, when it produces a transform that causes
two joinable nodes to end up next to each other, those are joined.
Nodes are considered joinable when they are of the same type and
when the `isJoinable` predicate returns true for them or, if an
array of strings was passed, if their node type name is in that
array.
*/
function autoJoin(command, isJoinable) {
    let canJoin = Array.isArray(isJoinable) ? (node) => isJoinable.indexOf(node.type.name) > -1
        : isJoinable;
    return (state, dispatch, view) => command(state, dispatch && wrapDispatchForJoin(dispatch, canJoin), view);
}
/**
Combine a number of command functions into a single function (which
calls them one by one until one returns true).
*/
function chainCommands(...commands) {
    return function (state, dispatch, view) {
        for (let i = 0; i < commands.length; i++)
            if (commands[i](state, dispatch, view))
                return true;
        return false;
    };
}
let backspace = chainCommands(deleteSelection, joinBackward, selectNodeBackward);
let del = chainCommands(deleteSelection, joinForward, selectNodeForward);
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
const pcBaseKeymap = {
    "Enter": chainCommands(newlineInCode, createParagraphNear, liftEmptyBlock, splitBlock),
    "Mod-Enter": exitCode,
    "Backspace": backspace,
    "Mod-Backspace": backspace,
    "Shift-Backspace": backspace,
    "Delete": del,
    "Mod-Delete": del,
    "Mod-a": selectAll
};
/**
A copy of `pcBaseKeymap` that also binds **Ctrl-h** like Backspace,
**Ctrl-d** like Delete, **Alt-Backspace** like Ctrl-Backspace, and
**Ctrl-Alt-Backspace**, **Alt-Delete**, and **Alt-d** like
Ctrl-Delete.
*/
const macBaseKeymap = {
    "Ctrl-h": pcBaseKeymap["Backspace"],
    "Alt-Backspace": pcBaseKeymap["Mod-Backspace"],
    "Ctrl-d": pcBaseKeymap["Delete"],
    "Ctrl-Alt-Backspace": pcBaseKeymap["Mod-Delete"],
    "Alt-Delete": pcBaseKeymap["Mod-Delete"],
    "Alt-d": pcBaseKeymap["Mod-Delete"],
    "Ctrl-a": selectTextblockStart,
    "Ctrl-e": selectTextblockEnd
};
for (let key in pcBaseKeymap)
    macBaseKeymap[key] = pcBaseKeymap[key];
const mac = typeof navigator != "undefined" ? /Mac|iP(hone|[oa]d)/.test(navigator.platform)
    // @ts-ignore
    : typeof os != "undefined" && os.platform ? os.platform() == "darwin" : false;
/**
Depending on the detected platform, this will hold
[`pcBasekeymap`](https://prosemirror.net/docs/ref/#commands.pcBaseKeymap) or
[`macBaseKeymap`](https://prosemirror.net/docs/ref/#commands.macBaseKeymap).
*/
const baseKeymap = mac ? macBaseKeymap : pcBaseKeymap;

export { autoJoin, baseKeymap, chainCommands, createParagraphNear, deleteSelection, exitCode, joinBackward, joinDown, joinForward, joinTextblockBackward, joinTextblockForward, joinUp, lift, liftEmptyBlock, macBaseKeymap, newlineInCode, pcBaseKeymap, selectAll, selectNodeBackward, selectNodeForward, selectParentNode, selectTextblockEnd, selectTextblockStart, setBlockType, splitBlock, splitBlockAs, splitBlockKeepMarks, toggleMark, wrapIn };
