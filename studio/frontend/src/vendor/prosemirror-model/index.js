// prosemirror-model 1.25.12 from npm (sha512-Ue2gTmXMa7EhpLNhC7J+h4+ykD8ha12K6rrZFFKKJHBForfIStw5gJ6Zrf1mqqAa9NmpmEB4wp9bKmA4eUVjWg==), MIT: see LICENSE. Imports point at the sibling vendored packages.
import OrderedMap from '../orderedmap/index.js';

function findDiffStart(a, b, pos) {
    for (let i = 0;; i++) {
        if (i == a.childCount || i == b.childCount)
            return a.childCount == b.childCount ? null : pos;
        let childA = a.child(i), childB = b.child(i);
        if (childA == childB) {
            pos += childA.nodeSize;
            continue;
        }
        if (!childA.sameMarkup(childB))
            return pos;
        if (childA.isText && childA.text != childB.text) {
            let tA = childA.text, tB = childB.text, j = 0;
            for (; tA[j] == tB[j]; j++)
                pos++;
            if (j && j < tA.length && j < tB.length && surrogateHigh(tA.charCodeAt(j - 1)) && surrogateLow(tA.charCodeAt(j)))
                pos--;
            return pos;
        }
        if (childA.content.size || childB.content.size) {
            let inner = findDiffStart(childA.content, childB.content, pos + 1);
            if (inner != null)
                return inner;
        }
        pos += childA.nodeSize;
    }
}
function findDiffEnd(a, b, posA, posB) {
    for (let iA = a.childCount, iB = b.childCount;;) {
        if (iA == 0 || iB == 0)
            return iA == iB ? null : { a: posA, b: posB };
        let childA = a.child(--iA), childB = b.child(--iB), size = childA.nodeSize;
        if (childA == childB) {
            posA -= size;
            posB -= size;
            continue;
        }
        if (!childA.sameMarkup(childB))
            return { a: posA, b: posB };
        if (childA.isText && childA.text != childB.text) {
            let tA = childA.text, tB = childB.text, iA = tA.length, iB = tB.length;
            while (iA > 0 && iB > 0 && tA[iA - 1] == tB[iB - 1]) {
                iA--;
                iB--;
                posA--;
                posB--;
            }
            if (iA && iB && iA < tA.length && surrogateHigh(tA.charCodeAt(iA - 1)) && surrogateLow(tA.charCodeAt(iA))) {
                posA++;
                posB++;
            }
            return { a: posA, b: posB };
        }
        if (childA.content.size || childB.content.size) {
            let inner = findDiffEnd(childA.content, childB.content, posA - 1, posB - 1);
            if (inner)
                return inner;
        }
        posA -= size;
        posB -= size;
    }
}
function surrogateLow(ch) { return ch >= 0xDC00 && ch < 0xE000; }
function surrogateHigh(ch) { return ch >= 0xD800 && ch < 0xDC00; }

/**
A fragment represents a node's collection of child nodes.

Like nodes, fragments are persistent data structures, and you
should not mutate them or their content. Rather, you create new
instances whenever needed. The API tries to make this easy.
*/
class Fragment {
    /**
    @internal
    */
    constructor(
    /**
    The child nodes in this fragment.
    */
    content, size) {
        this.content = content;
        this.size = size || 0;
        if (size == null)
            for (let i = 0; i < content.length; i++)
                this.size += content[i].nodeSize;
    }
    /**
    Invoke a callback for all descendant nodes between the given two
    positions (relative to start of this fragment). Doesn't descend
    into a node when the callback returns `false`.
    */
    nodesBetween(from, to, f, nodeStart = 0, parent) {
        for (let i = 0, pos = 0; pos < to; i++) {
            let child = this.content[i], end = pos + child.nodeSize;
            if (end > from && f(child, nodeStart + pos, parent || null, i) !== false && child.content.size) {
                let start = pos + 1;
                child.nodesBetween(Math.max(0, from - start), Math.min(child.content.size, to - start), f, nodeStart + start);
            }
            pos = end;
        }
    }
    /**
    Call the given callback for every descendant node. `pos` will be
    relative to the start of the fragment. The callback may return
    `false` to prevent traversal of a given node's children.
    */
    descendants(f) {
        this.nodesBetween(0, this.size, f);
    }
    /**
    Extract the text between `from` and `to`. See the same method on
    [`Node`](https://prosemirror.net/docs/ref/#model.Node.textBetween).
    */
    textBetween(from, to, blockSeparator, leafText) {
        let text = "", first = true;
        this.nodesBetween(from, to, (node, pos) => {
            let nodeText = node.isText ? node.text.slice(Math.max(from, pos) - pos, to - pos)
                : !node.isLeaf ? ""
                    : leafText ? (typeof leafText === "function" ? leafText(node) : leafText)
                        : node.type.spec.leafText ? node.type.spec.leafText(node)
                            : "";
            if (node.isBlock && (node.isLeaf && nodeText || node.isTextblock) && blockSeparator) {
                if (first)
                    first = false;
                else
                    text += blockSeparator;
            }
            text += nodeText;
        }, 0);
        return text;
    }
    /**
    Create a new fragment containing the combined content of this
    fragment and the other.
    */
    append(other) {
        if (!other.size)
            return this;
        if (!this.size)
            return other;
        let last = this.lastChild, first = other.firstChild, content = this.content.slice(), i = 0;
        if (last.isText && last.sameMarkup(first)) {
            content[content.length - 1] = last.withText(last.text + first.text);
            i = 1;
        }
        for (; i < other.content.length; i++)
            content.push(other.content[i]);
        return new Fragment(content, this.size + other.size);
    }
    /**
    Cut out the sub-fragment between the two given positions.
    */
    cut(from, to = this.size) {
        if (from == 0 && to == this.size)
            return this;
        let result = [], size = 0;
        if (to > from)
            for (let i = 0, pos = 0; pos < to; i++) {
                let child = this.content[i], end = pos + child.nodeSize;
                if (end > from) {
                    if (pos < from || end > to) {
                        if (child.isText)
                            child = child.cut(Math.max(0, from - pos), Math.min(child.text.length, to - pos));
                        else
                            child = child.cut(Math.max(0, from - pos - 1), Math.min(child.content.size, to - pos - 1));
                    }
                    result.push(child);
                    size += child.nodeSize;
                }
                pos = end;
            }
        return new Fragment(result, size);
    }
    /**
    @internal
    */
    cutByIndex(from, to) {
        if (from == to)
            return Fragment.empty;
        if (from == 0 && to == this.content.length)
            return this;
        return new Fragment(this.content.slice(from, to));
    }
    /**
    Create a new fragment in which the node at the given index is
    replaced by the given node.
    */
    replaceChild(index, node) {
        let current = this.content[index];
        if (current == node)
            return this;
        let copy = this.content.slice();
        let size = this.size + node.nodeSize - current.nodeSize;
        copy[index] = node;
        return new Fragment(copy, size);
    }
    /**
    Create a new fragment by prepending the given node to this
    fragment.
    */
    addToStart(node) {
        return new Fragment([node].concat(this.content), this.size + node.nodeSize);
    }
    /**
    Create a new fragment by appending the given node to this
    fragment.
    */
    addToEnd(node) {
        return new Fragment(this.content.concat(node), this.size + node.nodeSize);
    }
    /**
    Compare this fragment to another one.
    */
    eq(other) {
        if (this.content.length != other.content.length)
            return false;
        for (let i = 0; i < this.content.length; i++)
            if (!this.content[i].eq(other.content[i]))
                return false;
        return true;
    }
    /**
    The first child of the fragment, or `null` if it is empty.
    */
    get firstChild() { return this.content.length ? this.content[0] : null; }
    /**
    The last child of the fragment, or `null` if it is empty.
    */
    get lastChild() { return this.content.length ? this.content[this.content.length - 1] : null; }
    /**
    The number of child nodes in this fragment.
    */
    get childCount() { return this.content.length; }
    /**
    Get the child node at the given index. Raise an error when the
    index is out of range.
    */
    child(index) {
        let found = this.content[index];
        if (!found)
            throw new RangeError("Index " + index + " out of range for " + this);
        return found;
    }
    /**
    Get the child node at the given index, if it exists.
    */
    maybeChild(index) {
        return this.content[index] || null;
    }
    /**
    Call `f` for every child node, passing the node, its offset
    into this parent node, and its index.
    */
    forEach(f) {
        for (let i = 0, p = 0; i < this.content.length; i++) {
            let child = this.content[i];
            f(child, p, i);
            p += child.nodeSize;
        }
    }
    /**
    Find the first position at which this fragment and another
    fragment differ, or `null` if they are the same.
    */
    findDiffStart(other, pos = 0) {
        return findDiffStart(this, other, pos);
    }
    /**
    Find the first position, searching from the end, at which this
    fragment and the given fragment differ, or `null` if they are
    the same. Since this position will not be the same in both
    nodes, an object with two separate positions is returned.
    */
    findDiffEnd(other, pos = this.size, otherPos = other.size) {
        return findDiffEnd(this, other, pos, otherPos);
    }
    /**
    Find the index and inner offset corresponding to a given relative
    position in this fragment. The result object will be reused
    (overwritten) the next time the function is called. @internal
    */
    findIndex(pos) {
        if (pos == 0)
            return retIndex(0, pos);
        if (pos == this.size)
            return retIndex(this.content.length, pos);
        if (pos > this.size || pos < 0)
            throw new RangeError(`Position ${pos} outside of fragment (${this})`);
        for (let i = 0, curPos = 0;; i++) {
            let cur = this.child(i), end = curPos + cur.nodeSize;
            if (end >= pos) {
                if (end == pos)
                    return retIndex(i + 1, end);
                return retIndex(i, curPos);
            }
            curPos = end;
        }
    }
    /**
    Return a debugging string that describes this fragment.
    */
    toString() { return "<" + this.toStringInner() + ">"; }
    /**
    @internal
    */
    toStringInner() { return this.content.join(", "); }
    /**
    Create a JSON-serializeable representation of this fragment.
    */
    toJSON() {
        return this.content.length ? this.content.map(n => n.toJSON()) : null;
    }
    /**
    Deserialize a fragment from its JSON representation.
    */
    static fromJSON(schema, value) {
        if (!value)
            return Fragment.empty;
        if (!Array.isArray(value))
            throw new RangeError("Invalid input for Fragment.fromJSON");
        return Fragment.fromArray(value.map(schema.nodeFromJSON));
    }
    /**
    Build a fragment from an array of nodes. Ensures that adjacent
    text nodes with the same marks are joined together.
    */
    static fromArray(array) {
        if (!array.length)
            return Fragment.empty;
        let joined, size = 0;
        for (let i = 0; i < array.length; i++) {
            let node = array[i];
            size += node.nodeSize;
            if (i && node.isText && array[i - 1].sameMarkup(node)) {
                if (!joined)
                    joined = array.slice(0, i);
                joined[joined.length - 1] = node
                    .withText(joined[joined.length - 1].text + node.text);
            }
            else if (joined) {
                joined.push(node);
            }
        }
        return new Fragment(joined || array, size);
    }
    /**
    Create a fragment from something that can be interpreted as a
    set of nodes. For `null`, it returns the empty fragment. For a
    fragment, the fragment itself. For a node or array of nodes, a
    fragment containing those nodes.
    */
    static from(nodes) {
        if (!nodes)
            return Fragment.empty;
        if (nodes instanceof Fragment)
            return nodes;
        if (Array.isArray(nodes))
            return this.fromArray(nodes);
        if (nodes.attrs)
            return new Fragment([nodes], nodes.nodeSize);
        throw new RangeError("Can not convert " + nodes + " to a Fragment" +
            (nodes.nodesBetween ? " (looks like multiple versions of prosemirror-model were loaded)" : ""));
    }
}
/**
An empty fragment. Intended to be reused whenever a node doesn't
contain anything (rather than allocating a new empty fragment for
each leaf node).
*/
Fragment.empty = new Fragment([], 0);
const found = { index: 0, offset: 0 };
function retIndex(index, offset) {
    found.index = index;
    found.offset = offset;
    return found;
}

function compareDeep(a, b) {
    if (a === b)
        return true;
    if (!(a && typeof a == "object") ||
        !(b && typeof b == "object"))
        return false;
    let array = Array.isArray(a);
    if (Array.isArray(b) != array)
        return false;
    if (array) {
        if (a.length != b.length)
            return false;
        for (let i = 0; i < a.length; i++)
            if (!compareDeep(a[i], b[i]))
                return false;
    }
    else {
        for (let p in a)
            if (!(p in b) || !compareDeep(a[p], b[p]))
                return false;
        for (let p in b)
            if (!(p in a))
                return false;
    }
    return true;
}

/**
A mark is a piece of information that can be attached to a node,
such as it being emphasized, in code font, or a link. It has a
type and optionally a set of attributes that provide further
information (such as the target of the link). Marks are created
through a `Schema`, which controls which types exist and which
attributes they have.
*/
class Mark {
    /**
    @internal
    */
    constructor(
    /**
    The type of this mark.
    */
    type, 
    /**
    The attributes associated with this mark.
    */
    attrs) {
        this.type = type;
        this.attrs = attrs;
    }
    /**
    Given a set of marks, create a new set which contains this one as
    well, in the right position. If this mark is already in the set,
    the set itself is returned. If any marks that are set to be
    [exclusive](https://prosemirror.net/docs/ref/#model.MarkSpec.excludes) with this mark are present,
    those are replaced by this one.
    */
    addToSet(set) {
        let copy, placed = false;
        for (let i = 0; i < set.length; i++) {
            let other = set[i];
            if (this.eq(other))
                return set;
            if (this.type.excludes(other.type)) {
                if (!copy)
                    copy = set.slice(0, i);
            }
            else if (other.type.excludes(this.type)) {
                return set;
            }
            else {
                if (!placed && other.type.rank > this.type.rank) {
                    if (!copy)
                        copy = set.slice(0, i);
                    copy.push(this);
                    placed = true;
                }
                if (copy)
                    copy.push(other);
            }
        }
        if (!copy)
            copy = set.slice();
        if (!placed)
            copy.push(this);
        return copy;
    }
    /**
    Remove this mark from the given set, returning a new set. If this
    mark is not in the set, the set itself is returned.
    */
    removeFromSet(set) {
        for (let i = 0; i < set.length; i++)
            if (this.eq(set[i]))
                return set.slice(0, i).concat(set.slice(i + 1));
        return set;
    }
    /**
    Test whether this mark is in the given set of marks.
    */
    isInSet(set) {
        for (let i = 0; i < set.length; i++)
            if (this.eq(set[i]))
                return true;
        return false;
    }
    /**
    Test whether this mark has the same type and attributes as
    another mark.
    */
    eq(other) {
        return this == other ||
            (this.type == other.type && compareDeep(this.attrs, other.attrs));
    }
    /**
    Convert this mark to a JSON-serializeable representation.
    */
    toJSON() {
        let obj = { type: this.type.name };
        for (let _ in this.attrs) {
            obj.attrs = this.attrs;
            break;
        }
        return obj;
    }
    /**
    Deserialize a mark from JSON.
    */
    static fromJSON(schema, json) {
        if (!json)
            throw new RangeError("Invalid input for Mark.fromJSON");
        let type = schema.marks[json.type];
        if (!type)
            throw new RangeError(`There is no mark type ${json.type} in this schema`);
        let mark = type.create(json.attrs);
        return mark;
    }
    /**
    Test whether two sets of marks are identical.
    */
    static sameSet(a, b) {
        if (a == b)
            return true;
        if (a.length != b.length)
            return false;
        for (let i = 0; i < a.length; i++)
            if (!a[i].eq(b[i]))
                return false;
        return true;
    }
    /**
    Create a properly sorted mark set from null, a single mark, or an
    unsorted array of marks.
    */
    static setFrom(marks) {
        if (!marks || Array.isArray(marks) && marks.length == 0)
            return Mark.none;
        if (marks instanceof Mark)
            return [marks];
        let copy = marks.slice();
        copy.sort((a, b) => a.type.rank - b.type.rank);
        return copy;
    }
}
/**
The empty set of marks.
*/
Mark.none = [];

/**
Error type raised by [`Node.replace`](https://prosemirror.net/docs/ref/#model.Node.replace) when
given an invalid replacement.
*/
class ReplaceError extends Error {
}
/**
A slice represents a piece cut out of a larger document. It
stores not only a fragment, but also the depth up to which nodes on
both side are ‘open’ (cut through).
*/
class Slice {
    /**
    Create a slice. When specifying a non-zero open depth, you must
    make sure that there are nodes of at least that depth at the
    appropriate side of the fragment—i.e. if the fragment is an
    empty paragraph node, `openStart` and `openEnd` can't be greater
    than 1.
    
    It is not necessary for the content of open nodes to conform to
    the schema's content constraints, though it should be a valid
    start/end/middle for such a node, depending on which sides are
    open.
    */
    constructor(
    /**
    The slice's content.
    */
    content, 
    /**
    The open depth at the start of the fragment.
    */
    openStart, 
    /**
    The open depth at the end.
    */
    openEnd) {
        this.content = content;
        this.openStart = openStart;
        this.openEnd = openEnd;
    }
    /**
    The size this slice would add when inserted into a document.
    */
    get size() {
        return this.content.size - this.openStart - this.openEnd;
    }
    /**
    @internal
    */
    insertAt(pos, fragment) {
        let content = insertInto(this.content, pos + this.openStart, fragment, this.openStart + 1, this.openEnd + 1);
        return content && new Slice(content, this.openStart, this.openEnd);
    }
    /**
    @internal
    */
    removeBetween(from, to) {
        return new Slice(removeRange(this.content, from + this.openStart, to + this.openStart), this.openStart, this.openEnd);
    }
    /**
    Tests whether this slice is equal to another slice.
    */
    eq(other) {
        return this.content.eq(other.content) && this.openStart == other.openStart && this.openEnd == other.openEnd;
    }
    /**
    @internal
    */
    toString() {
        return this.content + "(" + this.openStart + "," + this.openEnd + ")";
    }
    /**
    Convert a slice to a JSON-serializable representation.
    */
    toJSON() {
        if (!this.content.size)
            return null;
        let json = { content: this.content.toJSON() };
        if (this.openStart > 0)
            json.openStart = this.openStart;
        if (this.openEnd > 0)
            json.openEnd = this.openEnd;
        return json;
    }
    /**
    Deserialize a slice from its JSON representation.
    */
    static fromJSON(schema, json) {
        if (!json)
            return Slice.empty;
        let openStart = json.openStart || 0, openEnd = json.openEnd || 0;
        if (typeof openStart != "number" || typeof openEnd != "number")
            throw new RangeError("Invalid input for Slice.fromJSON");
        return new Slice(Fragment.fromJSON(schema, json.content), openStart, openEnd);
    }
    /**
    Create a slice from a fragment by taking the maximum possible
    open value on both side of the fragment.
    */
    static maxOpen(fragment, openIsolating = true) {
        let openStart = 0, openEnd = 0;
        for (let n = fragment.firstChild; n && !n.isLeaf && (openIsolating || !n.type.spec.isolating); n = n.firstChild)
            openStart++;
        for (let n = fragment.lastChild; n && !n.isLeaf && (openIsolating || !n.type.spec.isolating); n = n.lastChild)
            openEnd++;
        return new Slice(fragment, openStart, openEnd);
    }
}
/**
The empty slice.
*/
Slice.empty = new Slice(Fragment.empty, 0, 0);
function removeRange(content, from, to) {
    let { index, offset } = content.findIndex(from), child = content.maybeChild(index);
    let { index: indexTo, offset: offsetTo } = content.findIndex(to);
    if (offset == from || child.isText) {
        if (offsetTo != to && !content.child(indexTo).isText)
            throw new RangeError("Removing non-flat range");
        return content.cut(0, from).append(content.cut(to));
    }
    if (index != indexTo)
        throw new RangeError("Removing non-flat range");
    return content.replaceChild(index, child.copy(removeRange(child.content, from - offset - 1, to - offset - 1)));
}
function insertInto(content, dist, insert, openStart, openEnd, parent) {
    let { index, offset } = content.findIndex(dist), child = content.maybeChild(index);
    if (offset == dist || child.isText) {
        if (parent && openStart <= 0 && openEnd <= 0 && !parent.canReplace(index, index, insert))
            return null;
        return content.cut(0, dist).append(insert).append(content.cut(dist));
    }
    let inner = insertInto(child.content, dist - offset - 1, insert, index == 0 ? openStart - 1 : 0, index == content.childCount - 1 ? openEnd - 1 : 0, child);
    return inner && content.replaceChild(index, child.copy(inner));
}
function replace($from, $to, slice) {
    if (slice.openStart > $from.depth)
        throw new ReplaceError("Inserted content deeper than insertion position");
    if ($from.depth - slice.openStart != $to.depth - slice.openEnd)
        throw new ReplaceError("Inconsistent open depths");
    return replaceOuter($from, $to, slice, 0);
}
function replaceOuter($from, $to, slice, depth) {
    let index = $from.index(depth), node = $from.node(depth);
    if (index == $to.index(depth) && depth < $from.depth - slice.openStart) {
        let inner = replaceOuter($from, $to, slice, depth + 1);
        return node.copy(node.content.replaceChild(index, inner));
    }
    else if (!slice.content.size) {
        return close(node, replaceTwoWay($from, $to, depth));
    }
    else if (!slice.openStart && !slice.openEnd && $from.depth == depth && $to.depth == depth) { // Simple, flat case
        let parent = $from.parent, content = parent.content;
        return close(parent, content.cut(0, $from.parentOffset).append(slice.content).append(content.cut($to.parentOffset)));
    }
    else {
        let { start, end } = prepareSliceForReplace(slice, $from);
        return close(node, replaceThreeWay($from, start, end, $to, depth));
    }
}
function checkJoin(main, sub) {
    if (!sub.type.compatibleContent(main.type))
        throw new ReplaceError("Cannot join " + sub.type.name + " onto " + main.type.name);
}
function joinable($before, $after, depth) {
    let node = $before.node(depth);
    checkJoin(node, $after.node(depth));
    return node;
}
function addNode(child, target) {
    let last = target.length - 1;
    if (last >= 0 && child.isText && child.sameMarkup(target[last]))
        target[last] = child.withText(target[last].text + child.text);
    else
        target.push(child);
}
function addRange($start, $end, depth, target) {
    let node = ($end || $start).node(depth);
    let startIndex = 0, endIndex = $end ? $end.index(depth) : node.childCount;
    if ($start) {
        startIndex = $start.index(depth);
        if ($start.depth > depth) {
            startIndex++;
        }
        else if ($start.textOffset) {
            addNode($start.nodeAfter, target);
            startIndex++;
        }
    }
    for (let i = startIndex; i < endIndex; i++)
        addNode(node.child(i), target);
    if ($end && $end.depth == depth && $end.textOffset)
        addNode($end.nodeBefore, target);
}
function close(node, content) {
    if (!node.type.validContent(content))
        throw new ReplaceError("Invalid content for node " + node.type.name);
    return node.copy(content);
}
function replaceThreeWay($from, $start, $end, $to, depth) {
    let openStart = $from.depth > depth && joinable($from, $start, depth + 1);
    let openEnd = $to.depth > depth && joinable($end, $to, depth + 1);
    let content = [];
    addRange(null, $from, depth, content);
    if (openStart && openEnd && $start.index(depth) == $end.index(depth)) {
        checkJoin(openStart, openEnd);
        addNode(close(openStart, replaceThreeWay($from, $start, $end, $to, depth + 1)), content);
    }
    else {
        if (openStart)
            addNode(close(openStart, replaceTwoWay($from, $start, depth + 1)), content);
        addRange($start, $end, depth, content);
        if (openEnd)
            addNode(close(openEnd, replaceTwoWay($end, $to, depth + 1)), content);
    }
    addRange($to, null, depth, content);
    return new Fragment(content);
}
function replaceTwoWay($from, $to, depth) {
    let content = [];
    addRange(null, $from, depth, content);
    if ($from.depth > depth) {
        let type = joinable($from, $to, depth + 1);
        addNode(close(type, replaceTwoWay($from, $to, depth + 1)), content);
    }
    addRange($to, null, depth, content);
    return new Fragment(content);
}
function prepareSliceForReplace(slice, $along) {
    let extra = $along.depth - slice.openStart, parent = $along.node(extra);
    let node = parent.copy(slice.content);
    for (let i = extra - 1; i >= 0; i--)
        node = $along.node(i).copy(Fragment.from(node));
    return { start: node.resolveNoCache(slice.openStart + extra),
        end: node.resolveNoCache(node.content.size - slice.openEnd - extra) };
}

/**
You can [_resolve_](https://prosemirror.net/docs/ref/#model.Node.resolve) a position to get more
information about it. Objects of this class represent such a
resolved position, providing various pieces of context
information, and some helper methods.

Throughout this interface, methods that take an optional `depth`
parameter will interpret undefined as `this.depth` and negative
numbers as `this.depth + value`.
*/
class ResolvedPos {
    /**
    @internal
    */
    constructor(
    /**
    The position that was resolved.
    */
    pos, 
    /**
    @internal
    */
    path, 
    /**
    The offset this position has into its parent node.
    */
    parentOffset) {
        this.pos = pos;
        this.path = path;
        this.parentOffset = parentOffset;
        this.depth = path.length / 3 - 1;
    }
    /**
    @internal
    */
    resolveDepth(val) {
        if (val == null)
            return this.depth;
        if (val < 0)
            return this.depth + val;
        return val;
    }
    /**
    The parent node that the position points into. Note that even if
    a position points into a text node, that node is not considered
    the parent—text nodes are ‘flat’ in this model, and have no content.
    */
    get parent() { return this.node(this.depth); }
    /**
    The root node in which the position was resolved.
    */
    get doc() { return this.node(0); }
    /**
    The ancestor node at the given level. `p.node(p.depth)` is the
    same as `p.parent`.
    */
    node(depth) { return this.path[this.resolveDepth(depth) * 3]; }
    /**
    The index into the ancestor at the given level. If this points
    at the 3rd node in the 2nd paragraph on the top level, for
    example, `p.index(0)` is 1 and `p.index(1)` is 2.
    */
    index(depth) { return this.path[this.resolveDepth(depth) * 3 + 1]; }
    /**
    The index pointing after this position into the ancestor at the
    given level.
    */
    indexAfter(depth) {
        depth = this.resolveDepth(depth);
        return this.index(depth) + (depth == this.depth && !this.textOffset ? 0 : 1);
    }
    /**
    The (absolute) position at the start of the node at the given
    level.
    */
    start(depth) {
        depth = this.resolveDepth(depth);
        return depth == 0 ? 0 : this.path[depth * 3 - 1] + 1;
    }
    /**
    The (absolute) position at the end of the node at the given
    level.
    */
    end(depth) {
        depth = this.resolveDepth(depth);
        return this.start(depth) + this.node(depth).content.size;
    }
    /**
    The (absolute) position directly before the wrapping node at the
    given level, or, when `depth` is `this.depth + 1`, the original
    position.
    */
    before(depth) {
        depth = this.resolveDepth(depth);
        if (!depth)
            throw new RangeError("There is no position before the top-level node");
        return depth == this.depth + 1 ? this.pos : this.path[depth * 3 - 1];
    }
    /**
    The (absolute) position directly after the wrapping node at the
    given level, or the original position when `depth` is `this.depth + 1`.
    */
    after(depth) {
        depth = this.resolveDepth(depth);
        if (!depth)
            throw new RangeError("There is no position after the top-level node");
        return depth == this.depth + 1 ? this.pos : this.path[depth * 3 - 1] + this.path[depth * 3].nodeSize;
    }
    /**
    When this position points into a text node, this returns the
    distance between the position and the start of the text node.
    Will be zero for positions that point between nodes.
    */
    get textOffset() { return this.pos - this.path[this.path.length - 1]; }
    /**
    Get the node directly after the position, if any. If the position
    points into a text node, only the part of that node after the
    position is returned.
    */
    get nodeAfter() {
        let parent = this.parent, index = this.index(this.depth);
        if (index == parent.childCount)
            return null;
        let dOff = this.pos - this.path[this.path.length - 1], child = parent.child(index);
        return dOff ? parent.child(index).cut(dOff) : child;
    }
    /**
    Get the node directly before the position, if any. If the
    position points into a text node, only the part of that node
    before the position is returned.
    */
    get nodeBefore() {
        let index = this.index(this.depth);
        let dOff = this.pos - this.path[this.path.length - 1];
        if (dOff)
            return this.parent.child(index).cut(0, dOff);
        return index == 0 ? null : this.parent.child(index - 1);
    }
    /**
    Get the position at the given index in the parent node at the
    given depth (which defaults to `this.depth`).
    */
    posAtIndex(index, depth) {
        depth = this.resolveDepth(depth);
        let node = this.path[depth * 3], pos = depth == 0 ? 0 : this.path[depth * 3 - 1] + 1;
        for (let i = 0; i < index; i++)
            pos += node.child(i).nodeSize;
        return pos;
    }
    /**
    Get the marks at this position, factoring in the surrounding
    marks' [`inclusive`](https://prosemirror.net/docs/ref/#model.MarkSpec.inclusive) property. If the
    position is at the start of a non-empty node, the marks of the
    node after it (if any) are returned.
    */
    marks() {
        let parent = this.parent, index = this.index();
        // In an empty parent, return the empty array
        if (parent.content.size == 0)
            return Mark.none;
        // When inside a text node, just return the text node's marks
        if (this.textOffset)
            return parent.child(index).marks;
        let main = parent.maybeChild(index - 1), other = parent.maybeChild(index);
        // If the `after` flag is true of there is no node before, make
        // the node after this position the main reference.
        if (!main) {
            let tmp = main;
            main = other;
            other = tmp;
        }
        // Use all marks in the main node, except those that have
        // `inclusive` set to false and are not present in the other node.
        let marks = main.marks;
        for (var i = 0; i < marks.length; i++)
            if (marks[i].type.spec.inclusive === false && (!other || !marks[i].isInSet(other.marks)))
                marks = marks[i--].removeFromSet(marks);
        return marks;
    }
    /**
    Get the marks after the current position, if any, except those
    that are non-inclusive and not present at position `$end`. This
    is mostly useful for getting the set of marks to preserve after a
    deletion. Will return `null` if this position is at the end of
    its parent node or its parent node isn't a textblock (in which
    case no marks should be preserved).
    */
    marksAcross($end) {
        let after = this.parent.maybeChild(this.index());
        if (!after || !after.isInline)
            return null;
        let marks = after.marks, next = $end.parent.maybeChild($end.index());
        for (var i = 0; i < marks.length; i++)
            if (marks[i].type.spec.inclusive === false && (!next || !marks[i].isInSet(next.marks)))
                marks = marks[i--].removeFromSet(marks);
        return marks;
    }
    /**
    The depth up to which this position and the given (non-resolved)
    position share the same parent nodes.
    */
    sharedDepth(pos) {
        for (let depth = this.depth; depth > 0; depth--)
            if (this.start(depth) <= pos && this.end(depth) >= pos)
                return depth;
        return 0;
    }
    /**
    Returns a range based on the place where this position and the
    given position diverge around block content. If both point into
    the same textblock, for example, a range around that textblock
    will be returned. If they point into different blocks, the range
    around those blocks in their shared ancestor is returned. You can
    pass in an optional predicate that will be called with a parent
    node to see if a range into that parent is acceptable.
    */
    blockRange(other = this, pred) {
        if (other.pos < this.pos)
            return other.blockRange(this);
        for (let d = this.depth - (this.parent.inlineContent || this.pos == other.pos ? 1 : 0); d >= 0; d--)
            if (other.pos <= this.end(d) && (!pred || pred(this.node(d))))
                return new NodeRange(this, other, d);
        return null;
    }
    /**
    Query whether the given position shares the same parent node.
    */
    sameParent(other) {
        return this.pos - this.parentOffset == other.pos - other.parentOffset;
    }
    /**
    Return the greater of this and the given position.
    */
    max(other) {
        return other.pos > this.pos ? other : this;
    }
    /**
    Return the smaller of this and the given position.
    */
    min(other) {
        return other.pos < this.pos ? other : this;
    }
    /**
    @internal
    */
    toString() {
        let str = "";
        for (let i = 1; i <= this.depth; i++)
            str += (str ? "/" : "") + this.node(i).type.name + "_" + this.index(i - 1);
        return str + ":" + this.parentOffset;
    }
    /**
    @internal
    */
    static resolve(doc, pos) {
        if (!(pos >= 0 && pos <= doc.content.size))
            throw new RangeError("Position " + pos + " out of range");
        let path = [];
        let start = 0, parentOffset = pos;
        for (let node = doc;;) {
            let { index, offset } = node.content.findIndex(parentOffset);
            let rem = parentOffset - offset;
            path.push(node, index, start + offset);
            if (!rem)
                break;
            node = node.child(index);
            if (node.isText)
                break;
            parentOffset = rem - 1;
            start += offset + 1;
        }
        return new ResolvedPos(pos, path, parentOffset);
    }
    /**
    @internal
    */
    static resolveCached(doc, pos) {
        let cache = resolveCache.get(doc);
        if (cache) {
            for (let i = 0; i < cache.elts.length; i++) {
                let elt = cache.elts[i];
                if (elt.pos == pos)
                    return elt;
            }
        }
        else {
            resolveCache.set(doc, cache = new ResolveCache);
        }
        let result = cache.elts[cache.i] = ResolvedPos.resolve(doc, pos);
        cache.i = (cache.i + 1) % resolveCacheSize;
        return result;
    }
}
class ResolveCache {
    constructor() {
        this.elts = [];
        this.i = 0;
    }
}
const resolveCacheSize = 12, resolveCache = new WeakMap();
/**
Represents a flat range of content, i.e. one that starts and
ends in the same node.
*/
class NodeRange {
    /**
    Construct a node range. `$from` and `$to` should point into the
    same node until at least the given `depth`, since a node range
    denotes an adjacent set of nodes in a single parent node.
    */
    constructor(
    /**
    A resolved position along the start of the content. May have a
    `depth` greater than this object's `depth` property, since
    these are the positions that were used to compute the range,
    not re-resolved positions directly at its boundaries.
    */
    $from, 
    /**
    A position along the end of the content. See
    caveat for [`$from`](https://prosemirror.net/docs/ref/#model.NodeRange.$from).
    */
    $to, 
    /**
    The depth of the node that this range points into.
    */
    depth) {
        this.$from = $from;
        this.$to = $to;
        this.depth = depth;
    }
    /**
    The position at the start of the range.
    */
    get start() { return this.$from.before(this.depth + 1); }
    /**
    The position at the end of the range.
    */
    get end() { return this.$to.after(this.depth + 1); }
    /**
    The parent node that the range points into.
    */
    get parent() { return this.$from.node(this.depth); }
    /**
    The start index of the range in the parent node.
    */
    get startIndex() { return this.$from.index(this.depth); }
    /**
    The end index of the range in the parent node.
    */
    get endIndex() { return this.$to.indexAfter(this.depth); }
}

const emptyAttrs = Object.create(null);
/**
This class represents a node in the tree that makes up a
ProseMirror document. So a document is an instance of `Node`, with
children that are also instances of `Node`.

Nodes are persistent data structures. Instead of changing them, you
create new ones with the content you want. Old ones keep pointing
at the old document shape. This is made cheaper by sharing
structure between the old and new data as much as possible, which a
tree shape like this (without back pointers) makes easy.

**Do not** directly mutate the properties of a `Node` object. See
[the guide](https://prosemirror.net/docs/guide/#doc) for more information.
*/
class Node {
    /**
    @internal
    */
    constructor(
    /**
    The type of node that this is.
    */
    type, 
    /**
    An object mapping attribute names to values. The kind of
    attributes allowed and required are
    [determined](https://prosemirror.net/docs/ref/#model.NodeSpec.attrs) by the node type.
    */
    attrs, 
    // A fragment holding the node's children.
    content, 
    /**
    The marks (things like whether it is emphasized or part of a
    link) applied to this node.
    */
    marks = Mark.none) {
        this.type = type;
        this.attrs = attrs;
        this.marks = marks;
        this.content = content || Fragment.empty;
    }
    /**
    The array of this node's child nodes.
    */
    get children() { return this.content.content; }
    /**
    The size of this node, as defined by the integer-based [indexing
    scheme](https://prosemirror.net/docs/guide/#doc.indexing). For text nodes, this is the
    amount of characters. For other leaf nodes, it is one. For
    non-leaf nodes, it is the size of the content plus two (the
    start and end token).
    */
    get nodeSize() { return this.isLeaf ? 1 : 2 + this.content.size; }
    /**
    The number of children that the node has.
    */
    get childCount() { return this.content.childCount; }
    /**
    Get the child node at the given index. Raises an error when the
    index is out of range.
    */
    child(index) { return this.content.child(index); }
    /**
    Get the child node at the given index, if it exists.
    */
    maybeChild(index) { return this.content.maybeChild(index); }
    /**
    Call `f` for every child node, passing the node, its offset
    into this parent node, and its index.
    */
    forEach(f) { this.content.forEach(f); }
    /**
    Invoke a callback for all descendant nodes recursively overlapping
    the given two positions that are relative to start of this
    node's content. This includes all ancestors of the nodes
    containing the two positions. The callback is invoked with the
    node, its position relative to the original node (method receiver),
    its parent node, and its child index. When the callback returns
    false for a given node, that node's children will not be
    recursed over. The last parameter can be used to specify a
    starting position to count from.
    */
    nodesBetween(from, to, f, startPos = 0) {
        this.content.nodesBetween(from, to, f, startPos, this);
    }
    /**
    Call the given callback for every descendant node. Doesn't
    descend into a node when the callback returns `false`.
    */
    descendants(f) {
        this.nodesBetween(0, this.content.size, f);
    }
    /**
    Concatenates all the text nodes found in this fragment and its
    children.
    */
    get textContent() {
        return (this.isLeaf && this.type.spec.leafText)
            ? this.type.spec.leafText(this)
            : this.textBetween(0, this.content.size, "");
    }
    /**
    Get all text between positions `from` and `to`. When
    `blockSeparator` is given, it will be inserted to separate text
    from different block nodes. If `leafText` is given, it'll be
    inserted for every non-text leaf node encountered, otherwise
    [`leafText`](https://prosemirror.net/docs/ref/#model.NodeSpec.leafText) will be used.
    */
    textBetween(from, to, blockSeparator, leafText) {
        return this.content.textBetween(from, to, blockSeparator, leafText);
    }
    /**
    Returns this node's first child, or `null` if there are no
    children.
    */
    get firstChild() { return this.content.firstChild; }
    /**
    Returns this node's last child, or `null` if there are no
    children.
    */
    get lastChild() { return this.content.lastChild; }
    /**
    Test whether two nodes represent the same piece of document.
    */
    eq(other) {
        return this == other || (this.sameMarkup(other) && this.content.eq(other.content));
    }
    /**
    Compare the markup (type, attributes, and marks) of this node to
    those of another. Returns `true` if both have the same markup.
    */
    sameMarkup(other) {
        return this.hasMarkup(other.type, other.attrs, other.marks);
    }
    /**
    Check whether this node's markup correspond to the given type,
    attributes, and marks.
    */
    hasMarkup(type, attrs, marks) {
        return this.type == type &&
            compareDeep(this.attrs, attrs || type.defaultAttrs || emptyAttrs) &&
            Mark.sameSet(this.marks, marks || Mark.none);
    }
    /**
    Create a new node with the same markup as this node, containing
    the given content (or empty, if no content is given).
    */
    copy(content = null) {
        if (content == this.content)
            return this;
        return new Node(this.type, this.attrs, content, this.marks);
    }
    /**
    Create a copy of this node, with the given set of marks instead
    of the node's own marks.
    */
    mark(marks) {
        return marks == this.marks ? this : new Node(this.type, this.attrs, this.content, marks);
    }
    /**
    Create a copy of this node with only the content between the
    given positions. If `to` is not given, it defaults to the end of
    the node.
    */
    cut(from, to = this.content.size) {
        if (from == 0 && to == this.content.size)
            return this;
        return this.copy(this.content.cut(from, to));
    }
    /**
    Cut out the part of the document between the given positions, and
    return it as a `Slice` object.
    */
    slice(from, to = this.content.size, includeParents = false) {
        if (from == to)
            return Slice.empty;
        let $from = this.resolve(from), $to = this.resolve(to);
        let depth = includeParents ? 0 : $from.sharedDepth(to);
        let start = $from.start(depth), node = $from.node(depth);
        let content = node.content.cut($from.pos - start, $to.pos - start);
        return new Slice(content, $from.depth - depth, $to.depth - depth);
    }
    /**
    Replace the part of the document between the given positions with
    the given slice. The slice must 'fit', meaning its open sides
    must be able to connect to the surrounding content, and its
    content nodes must be valid children for the node they are placed
    into. If any of this is violated, an error of type
    [`ReplaceError`](https://prosemirror.net/docs/ref/#model.ReplaceError) is thrown.
    */
    replace(from, to, slice) {
        return replace(this.resolve(from), this.resolve(to), slice);
    }
    /**
    Find the node directly after the given position.
    */
    nodeAt(pos) {
        for (let node = this;;) {
            let { index, offset } = node.content.findIndex(pos);
            node = node.maybeChild(index);
            if (!node)
                return null;
            if (offset == pos || node.isText)
                return node;
            pos -= offset + 1;
        }
    }
    /**
    Find the (direct) child node after the given offset, if any,
    and return it along with its index and offset relative to this
    node.
    */
    childAfter(pos) {
        let { index, offset } = this.content.findIndex(pos);
        return { node: this.content.maybeChild(index), index, offset };
    }
    /**
    Find the (direct) child node before the given offset, if any,
    and return it along with its index and offset relative to this
    node.
    */
    childBefore(pos) {
        if (pos == 0)
            return { node: null, index: 0, offset: 0 };
        let { index, offset } = this.content.findIndex(pos);
        if (offset < pos)
            return { node: this.content.child(index), index, offset };
        let node = this.content.child(index - 1);
        return { node, index: index - 1, offset: offset - node.nodeSize };
    }
    /**
    Resolve the given position in the document, returning an
    [object](https://prosemirror.net/docs/ref/#model.ResolvedPos) with information about its context.
    */
    resolve(pos) { return ResolvedPos.resolveCached(this, pos); }
    /**
    @internal
    */
    resolveNoCache(pos) { return ResolvedPos.resolve(this, pos); }
    /**
    Test whether a given mark or mark type occurs in this document
    between the two given positions.
    */
    rangeHasMark(from, to, type) {
        let found = false;
        if (to > from)
            this.nodesBetween(from, to, node => {
                if (type.isInSet(node.marks))
                    found = true;
                return !found;
            });
        return found;
    }
    /**
    True when this is a block (non-inline node)
    */
    get isBlock() { return this.type.isBlock; }
    /**
    True when this is a textblock node, a block node with inline
    content.
    */
    get isTextblock() { return this.type.isTextblock; }
    /**
    True when this node allows inline content.
    */
    get inlineContent() { return this.type.inlineContent; }
    /**
    True when this is an inline node (a text node or a node that can
    appear among text).
    */
    get isInline() { return this.type.isInline; }
    /**
    True when this is a text node.
    */
    get isText() { return this.type.isText; }
    /**
    True when this is a leaf node.
    */
    get isLeaf() { return this.type.isLeaf; }
    /**
    True when this is an atom, i.e. when it does not have directly
    editable content. This is usually the same as `isLeaf`, but can
    be configured with the [`atom` property](https://prosemirror.net/docs/ref/#model.NodeSpec.atom)
    on a node's spec (typically used when the node is displayed as
    an uneditable [node view](https://prosemirror.net/docs/ref/#view.NodeView)).
    */
    get isAtom() { return this.type.isAtom; }
    /**
    Return a string representation of this node for debugging
    purposes.
    */
    toString() {
        if (this.type.spec.toDebugString)
            return this.type.spec.toDebugString(this);
        let name = this.type.name;
        if (this.content.size)
            name += "(" + this.content.toStringInner() + ")";
        return wrapMarks(this.marks, name);
    }
    /**
    Get the content match in this node at the given index.
    */
    contentMatchAt(index) {
        let match = this.type.contentMatch.matchFragment(this.content, 0, index);
        if (!match)
            throw new Error("Called contentMatchAt on a node with invalid content");
        return match;
    }
    /**
    Test whether replacing the range between `from` and `to` (by
    child index) with the given replacement fragment (which defaults
    to the empty fragment) would leave the node's content valid. You
    can optionally pass `start` and `end` indices into the
    replacement fragment.
    */
    canReplace(from, to, replacement = Fragment.empty, start = 0, end = replacement.childCount) {
        let one = this.contentMatchAt(from).matchFragment(replacement, start, end);
        let two = one && one.matchFragment(this.content, to);
        if (!two || !two.validEnd)
            return false;
        for (let i = start; i < end; i++)
            if (!this.type.allowsMarks(replacement.child(i).marks))
                return false;
        return true;
    }
    /**
    Test whether replacing the range `from` to `to` (by index) with
    a node of the given type would leave the node's content valid.
    */
    canReplaceWith(from, to, type, marks) {
        if (marks && !this.type.allowsMarks(marks))
            return false;
        let start = this.contentMatchAt(from).matchType(type);
        let end = start && start.matchFragment(this.content, to);
        return end ? end.validEnd : false;
    }
    /**
    Test whether the given node's content could be appended to this
    node. If that node is empty, this will only return true if there
    is at least one node type that can appear in both nodes (to avoid
    merging completely incompatible nodes).
    */
    canAppend(other) {
        if (other.content.size)
            return this.canReplace(this.childCount, this.childCount, other.content);
        else
            return this.type.compatibleContent(other.type);
    }
    /**
    Check whether this node and its descendants conform to the
    schema, and raise an exception when they do not.
    */
    check() {
        this.type.checkContent(this.content);
        checkAttrs(this.type.attrs, this.attrs, "node", this.type.name);
        let copy = Mark.none;
        for (let i = 0; i < this.marks.length; i++) {
            let mark = this.marks[i];
            checkAttrs(mark.type.attrs, mark.attrs, "mark", mark.type.name);
            copy = mark.addToSet(copy);
        }
        if (!Mark.sameSet(copy, this.marks))
            throw new RangeError(`Invalid collection of marks for node ${this.type.name}: ${this.marks.map(m => m.type.name)}`);
        this.content.forEach(node => node.check());
    }
    /**
    Return a JSON-serializeable representation of this node.
    */
    toJSON() {
        let obj = { type: this.type.name };
        for (let _ in this.attrs) {
            obj.attrs = this.attrs;
            break;
        }
        if (this.content.size)
            obj.content = this.content.toJSON();
        if (this.marks.length)
            obj.marks = this.marks.map(n => n.toJSON());
        return obj;
    }
    /**
    Deserialize a node from its JSON representation.
    */
    static fromJSON(schema, json) {
        if (!json)
            throw new RangeError("Invalid input for Node.fromJSON");
        let marks = undefined;
        if (json.marks) {
            if (!Array.isArray(json.marks))
                throw new RangeError("Invalid mark data for Node.fromJSON");
            marks = json.marks.map(schema.markFromJSON);
        }
        if (json.type == "text") {
            if (typeof json.text != "string")
                throw new RangeError("Invalid text node in JSON");
            return schema.text(json.text, marks);
        }
        let content = Fragment.fromJSON(schema, json.content);
        let node = schema.nodeType(json.type).create(json.attrs, content, marks);
        return node;
    }
}
Node.prototype.text = undefined;
class TextNode extends Node {
    /**
    @internal
    */
    constructor(type, attrs, content, marks) {
        super(type, attrs, null, marks);
        if (!content)
            throw new RangeError("Empty text nodes are not allowed");
        this.text = content;
    }
    toString() {
        if (this.type.spec.toDebugString)
            return this.type.spec.toDebugString(this);
        return wrapMarks(this.marks, JSON.stringify(this.text));
    }
    get textContent() { return this.text; }
    textBetween(from, to) { return this.text.slice(from, to); }
    get nodeSize() { return this.text.length; }
    mark(marks) {
        return marks == this.marks ? this : new TextNode(this.type, this.attrs, this.text, marks);
    }
    withText(text) {
        if (text == this.text)
            return this;
        return new TextNode(this.type, this.attrs, text, this.marks);
    }
    cut(from = 0, to = this.text.length) {
        if (from == 0 && to == this.text.length)
            return this;
        return this.withText(this.text.slice(from, to));
    }
    eq(other) {
        return this.sameMarkup(other) && this.text == other.text;
    }
    toJSON() {
        let base = super.toJSON();
        base.text = this.text;
        return base;
    }
}
function wrapMarks(marks, str) {
    for (let i = marks.length - 1; i >= 0; i--)
        str = marks[i].type.name + "(" + str + ")";
    return str;
}
function checkAttrs(attrs, values, type, name) {
    for (let attr in values)
        if (!(attr in attrs))
            throw new RangeError(`Unsupported attribute ${attr} for ${type} of type ${name}`);
    for (let attr in attrs) {
        if (attrs[attr].validate)
            attrs[attr].validate(values[attr]);
    }
}

/**
Instances of this class represent a match state of a node type's
[content expression](https://prosemirror.net/docs/ref/#model.NodeSpec.content), and can be used to
find out whether further content matches here, and whether a given
position is a valid end of the node.
*/
class ContentMatch {
    /**
    @internal
    */
    constructor(
    /**
    True when this match state represents a valid end of the node.
    */
    validEnd) {
        this.validEnd = validEnd;
        /**
        @internal
        */
        this.next = [];
        /**
        @internal
        */
        this.wrapCache = [];
    }
    /**
    @internal
    */
    static parse(string, nodeTypes) {
        let stream = new TokenStream(string, nodeTypes);
        if (stream.next == null)
            return ContentMatch.empty;
        let expr = parseExpr(stream);
        if (stream.next)
            stream.err("Unexpected trailing text");
        let match = dfa(nfa(expr));
        checkForDeadEnds(match, stream);
        return match;
    }
    /**
    Match a node type, returning a match after that node if
    successful.
    */
    matchType(type) {
        for (let i = 0; i < this.next.length; i++)
            if (this.next[i].type == type)
                return this.next[i].next;
        return null;
    }
    /**
    Try to match a fragment. Returns the resulting match when
    successful.
    */
    matchFragment(frag, start = 0, end = frag.childCount) {
        let cur = this;
        for (let i = start; cur && i < end; i++)
            cur = cur.matchType(frag.child(i).type);
        return cur;
    }
    /**
    @internal
    */
    get inlineContent() {
        return this.next.length != 0 && this.next[0].type.isInline;
    }
    /**
    Get the first matching node type at this match position that can
    be generated.
    */
    get defaultType() {
        for (let i = 0; i < this.next.length; i++) {
            let { type } = this.next[i];
            if (!(type.isText || type.hasRequiredAttrs()))
                return type;
        }
        return null;
    }
    /**
    @internal
    */
    compatible(other) {
        for (let i = 0; i < this.next.length; i++)
            for (let j = 0; j < other.next.length; j++)
                if (this.next[i].type == other.next[j].type)
                    return true;
        return false;
    }
    /**
    Try to match the given fragment, and if that fails, see if it can
    be made to match by inserting nodes in front of it. When
    successful, return a fragment of inserted nodes (which may be
    empty if nothing had to be inserted). When `toEnd` is true, only
    return a fragment if the resulting match goes to the end of the
    content expression.
    */
    fillBefore(after, toEnd = false, startIndex = 0) {
        let seen = [this];
        function search(match, types) {
            let finished = match.matchFragment(after, startIndex);
            if (finished && (!toEnd || finished.validEnd))
                return Fragment.from(types.map(tp => tp.createAndFill()));
            for (let i = 0; i < match.next.length; i++) {
                let { type, next } = match.next[i];
                if (!(type.isText || type.hasRequiredAttrs()) && seen.indexOf(next) == -1) {
                    seen.push(next);
                    let found = search(next, types.concat(type));
                    if (found)
                        return found;
                }
            }
            return null;
        }
        return search(this, []);
    }
    /**
    Find a set of wrapping node types that would allow a node of the
    given type to appear at this position. The result may be empty
    (when it fits directly) and will be null when no such wrapping
    exists.
    */
    findWrapping(target) {
        for (let i = 0; i < this.wrapCache.length; i += 2)
            if (this.wrapCache[i] == target)
                return this.wrapCache[i + 1];
        let computed = this.computeWrapping(target);
        this.wrapCache.push(target, computed);
        return computed;
    }
    /**
    @internal
    */
    computeWrapping(target) {
        let seen = Object.create(null), active = [{ match: this, type: null, via: null }];
        while (active.length) {
            let current = active.shift(), match = current.match;
            if (match.matchType(target)) {
                let result = [];
                for (let obj = current; obj.type; obj = obj.via)
                    result.push(obj.type);
                return result.reverse();
            }
            for (let i = 0; i < match.next.length; i++) {
                let { type, next } = match.next[i];
                if (!type.isLeaf && !type.hasRequiredAttrs() && !(type.name in seen) && (!current.type || next.validEnd)) {
                    active.push({ match: type.contentMatch, type, via: current });
                    seen[type.name] = true;
                }
            }
        }
        return null;
    }
    /**
    The number of outgoing edges this node has in the finite
    automaton that describes the content expression.
    */
    get edgeCount() {
        return this.next.length;
    }
    /**
    Get the _n_​th outgoing edge from this node in the finite
    automaton that describes the content expression.
    */
    edge(n) {
        if (n >= this.next.length)
            throw new RangeError(`There's no ${n}th edge in this content match`);
        return this.next[n];
    }
    /**
    @internal
    */
    toString() {
        let seen = [];
        function scan(m) {
            seen.push(m);
            for (let i = 0; i < m.next.length; i++)
                if (seen.indexOf(m.next[i].next) == -1)
                    scan(m.next[i].next);
        }
        scan(this);
        return seen.map((m, i) => {
            let out = i + (m.validEnd ? "*" : " ") + " ";
            for (let i = 0; i < m.next.length; i++)
                out += (i ? ", " : "") + m.next[i].type.name + "->" + seen.indexOf(m.next[i].next);
            return out;
        }).join("\n");
    }
}
/**
@internal
*/
ContentMatch.empty = new ContentMatch(true);
class TokenStream {
    constructor(string, nodeTypes) {
        this.string = string;
        this.nodeTypes = nodeTypes;
        this.inline = null;
        this.pos = 0;
        this.tokens = string.split(/\s*(?=\b|\W|$)/);
        if (this.tokens[this.tokens.length - 1] == "")
            this.tokens.pop();
        if (this.tokens[0] == "")
            this.tokens.shift();
    }
    get next() { return this.tokens[this.pos]; }
    eat(tok) { return this.next == tok && (this.pos++ || true); }
    err(str) { throw new SyntaxError(str + " (in content expression '" + this.string + "')"); }
}
function parseExpr(stream) {
    let exprs = [];
    do {
        exprs.push(parseExprSeq(stream));
    } while (stream.eat("|"));
    return exprs.length == 1 ? exprs[0] : { type: "choice", exprs };
}
function parseExprSeq(stream) {
    let exprs = [];
    do {
        exprs.push(parseExprSubscript(stream));
    } while (stream.next && stream.next != ")" && stream.next != "|");
    return exprs.length == 1 ? exprs[0] : { type: "seq", exprs };
}
function parseExprSubscript(stream) {
    let expr = parseExprAtom(stream);
    for (;;) {
        if (stream.eat("+"))
            expr = { type: "plus", expr };
        else if (stream.eat("*"))
            expr = { type: "star", expr };
        else if (stream.eat("?"))
            expr = { type: "opt", expr };
        else if (stream.eat("{"))
            expr = parseExprRange(stream, expr);
        else
            break;
    }
    return expr;
}
function parseNum(stream) {
    if (/\D/.test(stream.next))
        stream.err("Expected number, got '" + stream.next + "'");
    let result = Number(stream.next);
    stream.pos++;
    return result;
}
function parseExprRange(stream, expr) {
    let min = parseNum(stream), max = min;
    if (stream.eat(",")) {
        if (stream.next != "}")
            max = parseNum(stream);
        else
            max = -1;
    }
    if (!stream.eat("}"))
        stream.err("Unclosed braced range");
    return { type: "range", min, max, expr };
}
function resolveName(stream, name) {
    let types = stream.nodeTypes, type = types[name];
    if (type)
        return [type];
    let result = [];
    for (let typeName in types) {
        let type = types[typeName];
        if (type.isInGroup(name))
            result.push(type);
    }
    if (result.length == 0)
        stream.err("No node type or group '" + name + "' found");
    return result;
}
function parseExprAtom(stream) {
    if (stream.eat("(")) {
        let expr = parseExpr(stream);
        if (!stream.eat(")"))
            stream.err("Missing closing paren");
        return expr;
    }
    else if (!/\W/.test(stream.next)) {
        let exprs = resolveName(stream, stream.next).map(type => {
            if (stream.inline == null)
                stream.inline = type.isInline;
            else if (stream.inline != type.isInline)
                stream.err("Mixing inline and block content");
            return { type: "name", value: type };
        });
        stream.pos++;
        return exprs.length == 1 ? exprs[0] : { type: "choice", exprs };
    }
    else {
        stream.err("Unexpected token '" + stream.next + "'");
    }
}
// Construct an NFA from an expression as returned by the parser. The
// NFA is represented as an array of states, which are themselves
// arrays of edges, which are `{term, to}` objects. The first state is
// the entry state and the last node is the success state.
//
// Note that unlike typical NFAs, the edge ordering in this one is
// significant, in that it is used to contruct filler content when
// necessary.
function nfa(expr) {
    let nfa = [[]];
    connect(compile(expr, 0), node());
    return nfa;
    function node() { return nfa.push([]) - 1; }
    function edge(from, to, term) {
        let edge = { term, to };
        nfa[from].push(edge);
        return edge;
    }
    function connect(edges, to) {
        edges.forEach(edge => edge.to = to);
    }
    function compile(expr, from) {
        if (expr.type == "choice") {
            return expr.exprs.reduce((out, expr) => out.concat(compile(expr, from)), []);
        }
        else if (expr.type == "seq") {
            for (let i = 0;; i++) {
                let next = compile(expr.exprs[i], from);
                if (i == expr.exprs.length - 1)
                    return next;
                connect(next, from = node());
            }
        }
        else if (expr.type == "star") {
            let loop = node();
            edge(from, loop);
            connect(compile(expr.expr, loop), loop);
            return [edge(loop)];
        }
        else if (expr.type == "plus") {
            let loop = node();
            connect(compile(expr.expr, from), loop);
            connect(compile(expr.expr, loop), loop);
            return [edge(loop)];
        }
        else if (expr.type == "opt") {
            return [edge(from)].concat(compile(expr.expr, from));
        }
        else if (expr.type == "range") {
            let cur = from;
            for (let i = 0; i < expr.min; i++) {
                let next = node();
                connect(compile(expr.expr, cur), next);
                cur = next;
            }
            if (expr.max == -1) {
                connect(compile(expr.expr, cur), cur);
            }
            else {
                for (let i = expr.min; i < expr.max; i++) {
                    let next = node();
                    edge(cur, next);
                    connect(compile(expr.expr, cur), next);
                    cur = next;
                }
            }
            return [edge(cur)];
        }
        else if (expr.type == "name") {
            return [edge(from, undefined, expr.value)];
        }
        else {
            throw new Error("Unknown expr type");
        }
    }
}
function cmp(a, b) { return b - a; }
// Get the set of nodes reachable by null edges from `node`. Omit
// nodes with only a single null-out-edge, since they may lead to
// needless duplicated nodes.
function nullFrom(nfa, node) {
    let result = [];
    scan(node);
    return result.sort(cmp);
    function scan(node) {
        let edges = nfa[node];
        if (edges.length == 1 && !edges[0].term)
            return scan(edges[0].to);
        result.push(node);
        for (let i = 0; i < edges.length; i++) {
            let { term, to } = edges[i];
            if (!term && result.indexOf(to) == -1)
                scan(to);
        }
    }
}
// Compiles an NFA as produced by `nfa` into a DFA, modeled as a set
// of state objects (`ContentMatch` instances) with transitions
// between them.
function dfa(nfa) {
    let labeled = Object.create(null);
    return explore(nullFrom(nfa, 0));
    function explore(states) {
        let out = [];
        states.forEach(node => {
            nfa[node].forEach(({ term, to }) => {
                if (!term)
                    return;
                let set;
                for (let i = 0; i < out.length; i++)
                    if (out[i][0] == term)
                        set = out[i][1];
                nullFrom(nfa, to).forEach(node => {
                    if (!set)
                        out.push([term, set = []]);
                    if (set.indexOf(node) == -1)
                        set.push(node);
                });
            });
        });
        let state = labeled[states.join(",")] = new ContentMatch(states.indexOf(nfa.length - 1) > -1);
        for (let i = 0; i < out.length; i++) {
            let states = out[i][1].sort(cmp);
            state.next.push({ type: out[i][0], next: labeled[states.join(",")] || explore(states) });
        }
        return state;
    }
}
function checkForDeadEnds(match, stream) {
    for (let i = 0, work = [match]; i < work.length; i++) {
        let state = work[i], dead = !state.validEnd, nodes = [];
        for (let j = 0; j < state.next.length; j++) {
            let { type, next } = state.next[j];
            nodes.push(type.name);
            if (dead && !(type.isText || type.hasRequiredAttrs()))
                dead = false;
            if (work.indexOf(next) == -1)
                work.push(next);
        }
        if (dead)
            stream.err("Only non-generatable nodes (" + nodes.join(", ") + ") in a required position (see https://prosemirror.net/docs/guide/#generatable)");
    }
}

// For node types where all attrs have a default value (or which don't
// have any attributes), build up a single reusable default attribute
// object, and use it for all nodes that don't specify specific
// attributes.
function defaultAttrs(attrs) {
    let defaults = Object.create(null);
    for (let attrName in attrs) {
        let attr = attrs[attrName];
        if (!attr.hasDefault)
            return null;
        defaults[attrName] = attr.default;
    }
    return defaults;
}
function computeAttrs(attrs, value) {
    let built = Object.create(null);
    for (let name in attrs) {
        let attr = attrs[name];
        let given = value && value[name];
        if (given === undefined) {
            if (attr.hasDefault)
                given = attr.default;
            else
                throw new RangeError("No value supplied for attribute " + name);
        }
        else if (attr.validate) {
            attr.validate(given);
        }
        built[name] = given;
    }
    return built;
}
function initAttrs(typeName, attrs) {
    let result = Object.create(null);
    if (attrs)
        for (let name in attrs)
            result[name] = new Attribute(typeName, name, attrs[name]);
    return result;
}
/**
Node types are objects allocated once per `Schema` and used to
[tag](https://prosemirror.net/docs/ref/#model.Node.type) `Node` instances. They contain information
about the node type, such as its name and what kind of node it
represents.
*/
class NodeType {
    /**
    @internal
    */
    constructor(
    /**
    The name the node type has in this schema.
    */
    name, 
    /**
    A link back to the `Schema` the node type belongs to.
    */
    schema, 
    /**
    The spec that this type is based on
    */
    spec) {
        this.name = name;
        this.schema = schema;
        this.spec = spec;
        /**
        The set of marks allowed in this node. `null` means all marks
        are allowed.
        */
        this.markSet = null;
        this.groups = spec.group ? spec.group.split(" ") : [];
        this.attrs = initAttrs(name, spec.attrs);
        this.defaultAttrs = defaultAttrs(this.attrs);
        this.contentMatch = null;
        this.inlineContent = null;
        this.isBlock = !(spec.inline || name == "text");
        this.isText = name == "text";
    }
    /**
    True if this is an inline type.
    */
    get isInline() { return !this.isBlock; }
    /**
    True if this is a textblock type, a block that contains inline
    content.
    */
    get isTextblock() { return this.isBlock && this.inlineContent; }
    /**
    True for node types that allow no content.
    */
    get isLeaf() { return this.contentMatch == ContentMatch.empty; }
    /**
    True when this node is an atom, i.e. when it does not have
    directly editable content.
    */
    get isAtom() { return this.isLeaf || !!this.spec.atom; }
    /**
    Return true when this node type is part of the given
    [group](https://prosemirror.net/docs/ref/#model.NodeSpec.group).
    */
    isInGroup(group) {
        return this.groups.indexOf(group) > -1;
    }
    /**
    The node type's [whitespace](https://prosemirror.net/docs/ref/#model.NodeSpec.whitespace) option.
    */
    get whitespace() {
        return this.spec.whitespace || (this.spec.code ? "pre" : "normal");
    }
    /**
    Tells you whether this node type has any required attributes.
    */
    hasRequiredAttrs() {
        for (let n in this.attrs)
            if (this.attrs[n].isRequired)
                return true;
        return false;
    }
    /**
    Indicates whether this node allows some of the same content as
    the given node type.
    */
    compatibleContent(other) {
        return this == other || this.contentMatch.compatible(other.contentMatch);
    }
    /**
    @internal
    */
    computeAttrs(attrs) {
        if (!attrs && this.defaultAttrs)
            return this.defaultAttrs;
        else
            return computeAttrs(this.attrs, attrs);
    }
    /**
    Create a `Node` of this type. The given attributes are
    checked and defaulted (you can pass `null` to use the type's
    defaults entirely, if no required attributes exist). `content`
    may be a `Fragment`, a node, an array of nodes, or
    `null`. Similarly `marks` may be `null` to default to the empty
    set of marks.
    */
    create(attrs = null, content, marks) {
        if (this.isText)
            throw new Error("NodeType.create can't construct text nodes");
        return new Node(this, this.computeAttrs(attrs), Fragment.from(content), Mark.setFrom(marks));
    }
    /**
    Like [`create`](https://prosemirror.net/docs/ref/#model.NodeType.create), but check the given content
    against the node type's content restrictions, and throw an error
    if it doesn't match.
    */
    createChecked(attrs = null, content, marks) {
        content = Fragment.from(content);
        this.checkContent(content);
        return new Node(this, this.computeAttrs(attrs), content, Mark.setFrom(marks));
    }
    /**
    Like [`create`](https://prosemirror.net/docs/ref/#model.NodeType.create), but see if it is
    necessary to add nodes to the start or end of the given fragment
    to make it fit the node. If no fitting wrapping can be found,
    return null. Note that, due to the fact that required nodes can
    always be created, this will always succeed if you pass null or
    `Fragment.empty` as content.
    */
    createAndFill(attrs = null, content, marks) {
        attrs = this.computeAttrs(attrs);
        content = Fragment.from(content);
        if (content.size) {
            let before = this.contentMatch.fillBefore(content);
            if (!before)
                return null;
            content = before.append(content);
        }
        let matched = this.contentMatch.matchFragment(content);
        let after = matched && matched.fillBefore(Fragment.empty, true);
        if (!after)
            return null;
        return new Node(this, attrs, content.append(after), Mark.setFrom(marks));
    }
    /**
    Returns true if the given fragment is valid content for this node
    type.
    */
    validContent(content) {
        let result = this.contentMatch.matchFragment(content);
        if (!result || !result.validEnd)
            return false;
        for (let i = 0; i < content.childCount; i++)
            if (!this.allowsMarks(content.child(i).marks))
                return false;
        return true;
    }
    /**
    Throws a RangeError if the given fragment is not valid content for this
    node type.
    @internal
    */
    checkContent(content) {
        if (!this.validContent(content))
            throw new RangeError(`Invalid content for node ${this.name}: ${content.toString().slice(0, 50)}`);
    }
    /**
    @internal no longer useful, but called by old prosemirror-view versions
    */
    checkAttrs(attrs) { }
    /**
    Check whether the given mark type is allowed in this node.
    */
    allowsMarkType(markType) {
        return this.markSet == null || this.markSet.indexOf(markType) > -1;
    }
    /**
    Test whether the given set of marks are allowed in this node.
    */
    allowsMarks(marks) {
        if (this.markSet == null)
            return true;
        for (let i = 0; i < marks.length; i++)
            if (!this.allowsMarkType(marks[i].type))
                return false;
        return true;
    }
    /**
    Removes the marks that are not allowed in this node from the given set.
    */
    allowedMarks(marks) {
        if (this.markSet == null)
            return marks;
        let copy;
        for (let i = 0; i < marks.length; i++) {
            if (!this.allowsMarkType(marks[i].type)) {
                if (!copy)
                    copy = marks.slice(0, i);
            }
            else if (copy) {
                copy.push(marks[i]);
            }
        }
        return !copy ? marks : copy.length ? copy : Mark.none;
    }
    /**
    @internal
    */
    static compile(nodes, schema) {
        let result = Object.create(null);
        nodes.forEach((name, spec) => result[name] = new NodeType(name, schema, spec));
        let topType = schema.spec.topNode || "doc";
        if (!result[topType])
            throw new RangeError("Schema is missing its top node type ('" + topType + "')");
        if (!result.text)
            throw new RangeError("Every schema needs a 'text' type");
        for (let _ in result.text.attrs)
            throw new RangeError("The text node type should not have attributes");
        return result;
    }
}
function validateType(typeName, attrName, type) {
    let types = type.split("|");
    return (value) => {
        let name = value === null ? "null" : typeof value;
        if (types.indexOf(name) < 0)
            throw new RangeError(`Expected value of type ${types} for attribute ${attrName} on type ${typeName}, got ${name}`);
    };
}
// Attribute descriptors
class Attribute {
    constructor(typeName, attrName, options) {
        this.hasDefault = Object.prototype.hasOwnProperty.call(options, "default");
        this.default = options.default;
        this.validate = typeof options.validate == "string" ? validateType(typeName, attrName, options.validate) : options.validate;
    }
    get isRequired() {
        return !this.hasDefault;
    }
}
// Marks
/**
Like nodes, marks (which are associated with nodes to signify
things like emphasis or being part of a link) are
[tagged](https://prosemirror.net/docs/ref/#model.Mark.type) with type objects, which are
instantiated once per `Schema`.
*/
class MarkType {
    /**
    @internal
    */
    constructor(
    /**
    The name of the mark type.
    */
    name, 
    /**
    @internal
    */
    rank, 
    /**
    The schema that this mark type instance is part of.
    */
    schema, 
    /**
    The spec on which the type is based.
    */
    spec) {
        this.name = name;
        this.rank = rank;
        this.schema = schema;
        this.spec = spec;
        this.attrs = initAttrs(name, spec.attrs);
        this.excluded = null;
        let defaults = defaultAttrs(this.attrs);
        this.instance = defaults ? new Mark(this, defaults) : null;
    }
    /**
    Create a mark of this type. `attrs` may be `null` or an object
    containing only some of the mark's attributes. The others, if
    they have defaults, will be added.
    */
    create(attrs = null) {
        if (!attrs && this.instance)
            return this.instance;
        return new Mark(this, computeAttrs(this.attrs, attrs));
    }
    /**
    @internal
    */
    static compile(marks, schema) {
        let result = Object.create(null), rank = 0;
        marks.forEach((name, spec) => result[name] = new MarkType(name, rank++, schema, spec));
        return result;
    }
    /**
    When there is a mark of this type in the given set, a new set
    without it is returned. Otherwise, the input set is returned.
    */
    removeFromSet(set) {
        for (var i = 0; i < set.length; i++)
            if (set[i].type == this) {
                set = set.slice(0, i).concat(set.slice(i + 1));
                i--;
            }
        return set;
    }
    /**
    Tests whether there is a mark of this type in the given set.
    */
    isInSet(set) {
        for (let i = 0; i < set.length; i++)
            if (set[i].type == this)
                return set[i];
    }
    /**
    Queries whether a given mark type is
    [excluded](https://prosemirror.net/docs/ref/#model.MarkSpec.excludes) by this one.
    */
    excludes(other) {
        return this.excluded.indexOf(other) > -1;
    }
}
/**
A document schema. Holds [node](https://prosemirror.net/docs/ref/#model.NodeType) and [mark
type](https://prosemirror.net/docs/ref/#model.MarkType) objects for the nodes and marks that may
occur in conforming documents, and provides functionality for
creating and deserializing such documents.

When given, the type parameters provide the names of the nodes and
marks in this schema.
*/
class Schema {
    /**
    Construct a schema from a schema [specification](https://prosemirror.net/docs/ref/#model.SchemaSpec).
    */
    constructor(spec) {
        /**
        The [linebreak
        replacement](https://prosemirror.net/docs/ref/#model.NodeSpec.linebreakReplacement) node defined
        in this schema, if any.
        */
        this.linebreakReplacement = null;
        /**
        An object for storing whatever values modules may want to
        compute and cache per schema. (If you want to store something
        in it, try to use property names unlikely to clash.)
        */
        this.cached = Object.create(null);
        let instanceSpec = this.spec = {};
        for (let prop in spec)
            instanceSpec[prop] = spec[prop];
        instanceSpec.nodes = OrderedMap.from(spec.nodes),
            instanceSpec.marks = OrderedMap.from(spec.marks || {}),
            this.nodes = NodeType.compile(this.spec.nodes, this);
        this.marks = MarkType.compile(this.spec.marks, this);
        let contentExprCache = Object.create(null);
        for (let prop in this.nodes) {
            if (prop in this.marks)
                throw new RangeError(prop + " can not be both a node and a mark");
            let type = this.nodes[prop], contentExpr = type.spec.content || "", markExpr = type.spec.marks;
            type.contentMatch = contentExprCache[contentExpr] ||
                (contentExprCache[contentExpr] = ContentMatch.parse(contentExpr, this.nodes));
            type.inlineContent = type.contentMatch.inlineContent;
            if (type.spec.linebreakReplacement) {
                if (this.linebreakReplacement)
                    throw new RangeError("Multiple linebreak nodes defined");
                if (!type.isInline || !type.isLeaf)
                    throw new RangeError("Linebreak replacement nodes must be inline leaf nodes");
                this.linebreakReplacement = type;
            }
            type.markSet = markExpr == "_" ? null :
                markExpr ? gatherMarks(this, markExpr.split(" ")) :
                    markExpr == "" || !type.inlineContent ? [] : null;
        }
        for (let prop in this.marks) {
            let type = this.marks[prop], excl = type.spec.excludes;
            type.excluded = excl == null ? [type] : excl == "" ? [] : gatherMarks(this, excl.split(" "));
        }
        this.nodeFromJSON = json => Node.fromJSON(this, json);
        this.markFromJSON = json => Mark.fromJSON(this, json);
        this.topNodeType = this.nodes[this.spec.topNode || "doc"];
        this.cached.wrappings = Object.create(null);
    }
    /**
    Create a node in this schema. The `type` may be a string or a
    `NodeType` instance. Attributes will be extended with defaults,
    `content` may be a `Fragment`, `null`, a `Node`, or an array of
    nodes.
    */
    node(type, attrs = null, content, marks) {
        if (typeof type == "string")
            type = this.nodeType(type);
        else if (!(type instanceof NodeType))
            throw new RangeError("Invalid node type: " + type);
        else if (type.schema != this)
            throw new RangeError("Node type from different schema used (" + type.name + ")");
        return type.createChecked(attrs, content, marks);
    }
    /**
    Create a text node in the schema. Empty text nodes are not
    allowed.
    */
    text(text, marks) {
        let type = this.nodes.text;
        return new TextNode(type, type.defaultAttrs, text, Mark.setFrom(marks));
    }
    /**
    Create a mark with the given type and attributes.
    */
    mark(type, attrs) {
        if (typeof type == "string")
            type = this.marks[type];
        return type.create(attrs);
    }
    /**
    @internal
    */
    nodeType(name) {
        let found = this.nodes[name];
        if (!found)
            throw new RangeError("Unknown node type: " + name);
        return found;
    }
}
function gatherMarks(schema, marks) {
    let found = [];
    for (let i = 0; i < marks.length; i++) {
        let name = marks[i], mark = schema.marks[name], ok = mark;
        if (mark) {
            found.push(mark);
        }
        else {
            for (let prop in schema.marks) {
                let mark = schema.marks[prop];
                if (name == "_" || (mark.spec.group && mark.spec.group.split(" ").indexOf(name) > -1))
                    found.push(ok = mark);
            }
        }
        if (!ok)
            throw new SyntaxError("Unknown mark type: '" + marks[i] + "'");
    }
    return found;
}

function isTagRule(rule) { return rule.tag != null; }
function isStyleRule(rule) { return rule.style != null; }
/**
A DOM parser represents a strategy for parsing DOM content into a
ProseMirror document conforming to a given schema. Its behavior is
defined by an array of [rules](https://prosemirror.net/docs/ref/#model.ParseRule).
*/
class DOMParser {
    /**
    Create a parser that targets the given schema, using the given
    parsing rules.
    */
    constructor(
    /**
    The schema into which the parser parses.
    */
    schema, 
    /**
    The set of [parse rules](https://prosemirror.net/docs/ref/#model.ParseRule) that the parser
    uses, in order of precedence.
    */
    rules) {
        this.schema = schema;
        this.rules = rules;
        /**
        @internal
        */
        this.tags = [];
        /**
        @internal
        */
        this.styles = [];
        let matchedStyles = this.matchedStyles = [];
        rules.forEach(rule => {
            if (isTagRule(rule)) {
                this.tags.push(rule);
            }
            else if (isStyleRule(rule)) {
                let prop = /[^=]*/.exec(rule.style)[0];
                if (matchedStyles.indexOf(prop) < 0)
                    matchedStyles.push(prop);
                this.styles.push(rule);
            }
        });
        // Only normalize list elements when lists in the schema can't directly contain themselves
        this.normalizeLists = !this.tags.some(r => {
            if (!/^(ul|ol)\b/.test(r.tag) || !r.node)
                return false;
            let node = schema.nodes[r.node];
            return node.contentMatch.matchType(node);
        });
    }
    /**
    Parse a document from the content of a DOM node.
    */
    parse(dom, options = {}) {
        let context = new ParseContext(this, options, false);
        context.addAll(dom, Mark.none, options.from, options.to);
        return context.finish();
    }
    /**
    Parses the content of the given DOM node, like
    [`parse`](https://prosemirror.net/docs/ref/#model.DOMParser.parse), and takes the same set of
    options. But unlike that method, which produces a whole node,
    this one returns a slice that is open at the sides, meaning that
    the schema constraints aren't applied to the start of nodes to
    the left of the input and the end of nodes at the end.
    */
    parseSlice(dom, options = {}) {
        let context = new ParseContext(this, options, true);
        context.addAll(dom, Mark.none, options.from, options.to);
        return Slice.maxOpen(context.finish());
    }
    /**
    @internal
    */
    matchTag(dom, context, after) {
        for (let i = after ? this.tags.indexOf(after) + 1 : 0; i < this.tags.length; i++) {
            let rule = this.tags[i];
            if (matches(dom, rule.tag) &&
                (rule.namespace === undefined || dom.namespaceURI == rule.namespace) &&
                (!rule.context || context.matchesContext(rule.context))) {
                if (rule.getAttrs) {
                    let result = rule.getAttrs(dom);
                    if (result === false)
                        continue;
                    rule.attrs = result || undefined;
                }
                return rule;
            }
        }
    }
    /**
    @internal
    */
    matchStyle(prop, value, context, after) {
        for (let i = after ? this.styles.indexOf(after) + 1 : 0; i < this.styles.length; i++) {
            let rule = this.styles[i], style = rule.style;
            if (style.indexOf(prop) != 0 ||
                rule.context && !context.matchesContext(rule.context) ||
                // Test that the style string either precisely matches the prop,
                // or has an '=' sign after the prop, followed by the given
                // value.
                style.length > prop.length &&
                    (style.charCodeAt(prop.length) != 61 || style.slice(prop.length + 1) != value))
                continue;
            if (rule.getAttrs) {
                let result = rule.getAttrs(value);
                if (result === false)
                    continue;
                rule.attrs = result || undefined;
            }
            return rule;
        }
    }
    /**
    @internal
    */
    static schemaRules(schema) {
        let result = [];
        function insert(rule) {
            let priority = rule.priority == null ? 50 : rule.priority, i = 0;
            for (; i < result.length; i++) {
                let next = result[i], nextPriority = next.priority == null ? 50 : next.priority;
                if (nextPriority < priority)
                    break;
            }
            result.splice(i, 0, rule);
        }
        for (let name in schema.marks) {
            let rules = schema.marks[name].spec.parseDOM;
            if (rules)
                rules.forEach(rule => {
                    insert(rule = copy(rule));
                    if (!(rule.mark || rule.ignore || rule.clearMark))
                        rule.mark = name;
                });
        }
        for (let name in schema.nodes) {
            let rules = schema.nodes[name].spec.parseDOM;
            if (rules)
                rules.forEach(rule => {
                    insert(rule = copy(rule));
                    if (!(rule.node || rule.ignore || rule.mark))
                        rule.node = name;
                });
        }
        return result;
    }
    /**
    Construct a DOM parser using the parsing rules listed in a
    schema's [node specs](https://prosemirror.net/docs/ref/#model.NodeSpec.parseDOM), reordered by
    [priority](https://prosemirror.net/docs/ref/#model.GenericParseRule.priority).
    */
    static fromSchema(schema) {
        return schema.cached.domParser ||
            (schema.cached.domParser = new DOMParser(schema, DOMParser.schemaRules(schema)));
    }
}
const blockTags = {
    address: true, article: true, aside: true, blockquote: true, body: true, canvas: true,
    dd: true, div: true, dl: true, fieldset: true, figcaption: true, figure: true,
    footer: true, form: true, h1: true, h2: true, h3: true, h4: true, h5: true,
    h6: true, header: true, hgroup: true, hr: true, li: true, noscript: true, ol: true,
    output: true, p: true, pre: true, section: true, table: true, tfoot: true, ul: true
};
const ignoreTags = {
    head: true, noscript: true, object: true, script: true, style: true, title: true
};
const listTags = { ol: true, ul: true };
// Using a bitfield for node context options
const OPT_PRESERVE_WS = 1, OPT_PRESERVE_WS_FULL = 2, OPT_OPEN_LEFT = 4;
function wsOptionsFor(type, preserveWhitespace, base) {
    if (preserveWhitespace != null)
        return (preserveWhitespace ? OPT_PRESERVE_WS : 0) |
            (preserveWhitespace === "full" ? OPT_PRESERVE_WS_FULL : 0);
    return type && type.whitespace == "pre" ? OPT_PRESERVE_WS | OPT_PRESERVE_WS_FULL : base & ~OPT_OPEN_LEFT;
}
class NodeContext {
    constructor(type, attrs, marks, solid, match, options) {
        this.type = type;
        this.attrs = attrs;
        this.marks = marks;
        this.solid = solid;
        this.options = options;
        this.content = [];
        // Marks applied to the node's children
        this.activeMarks = Mark.none;
        this.match = match || (options & OPT_OPEN_LEFT ? null : type.contentMatch);
    }
    findWrapping(node) {
        if (!this.match) {
            if (!this.type)
                return [];
            let fill = this.type.contentMatch.fillBefore(Fragment.from(node));
            if (fill) {
                this.match = this.type.contentMatch.matchFragment(fill);
            }
            else {
                let start = this.type.contentMatch, wrap;
                if (wrap = start.findWrapping(node.type)) {
                    this.match = start;
                    return wrap;
                }
                else {
                    return null;
                }
            }
        }
        return this.match.findWrapping(node.type);
    }
    finish(openEnd) {
        if (!(this.options & OPT_PRESERVE_WS)) { // Strip trailing whitespace
            let last = this.content[this.content.length - 1], m;
            if (last && last.isText && (m = /[ \t\r\n\u000c]+$/.exec(last.text))) {
                let text = last;
                if (last.text.length == m[0].length)
                    this.content.pop();
                else
                    this.content[this.content.length - 1] = text.withText(text.text.slice(0, text.text.length - m[0].length));
            }
        }
        let content = Fragment.from(this.content);
        if (!openEnd && this.match)
            content = content.append(this.match.fillBefore(Fragment.empty, true));
        return this.type ? this.type.create(this.attrs, content, this.marks) : content;
    }
    inlineContext(node) {
        if (this.type)
            return this.type.inlineContent;
        if (this.content.length)
            return this.content[0].isInline;
        return node.parentNode && !blockTags.hasOwnProperty(node.parentNode.nodeName.toLowerCase());
    }
}
class ParseContext {
    constructor(
    // The parser we are using.
    parser, 
    // The options passed to this parse.
    options, isOpen) {
        this.parser = parser;
        this.options = options;
        this.isOpen = isOpen;
        this.open = 0;
        this.localPreserveWS = false;
        let topNode = options.topNode, topContext;
        let topOptions = wsOptionsFor(null, options.preserveWhitespace, 0) | (isOpen ? OPT_OPEN_LEFT : 0);
        if (topNode)
            topContext = new NodeContext(topNode.type, topNode.attrs, Mark.none, true, options.topMatch || topNode.type.contentMatch, topOptions);
        else if (isOpen)
            topContext = new NodeContext(null, null, Mark.none, true, null, topOptions);
        else
            topContext = new NodeContext(parser.schema.topNodeType, null, Mark.none, true, null, topOptions);
        this.nodes = [topContext];
        this.find = options.findPositions;
        this.needsBlock = false;
    }
    get top() {
        return this.nodes[this.open];
    }
    // Add a DOM node to the content. Text is inserted as text node,
    // otherwise, the node is passed to `addElement` or, if it has a
    // `style` attribute, `addElementWithStyles`.
    addDOM(dom, marks) {
        if (dom.nodeType == 3)
            this.addTextNode(dom, marks);
        else if (dom.nodeType == 1)
            this.addElement(dom, marks);
    }
    addTextNode(dom, marks) {
        let value = dom.nodeValue;
        let top = this.top, preserveWS = (top.options & OPT_PRESERVE_WS_FULL) ? "full"
            : this.localPreserveWS || (top.options & OPT_PRESERVE_WS) > 0;
        let { schema } = this.parser;
        if (preserveWS === "full" ||
            top.inlineContext(dom) ||
            /[^ \t\r\n\u000c]/.test(value)) {
            if (!preserveWS) {
                value = value.replace(/[ \t\r\n\u000c]+/g, " ");
                // If this starts with whitespace, and there is no node before it, or
                // a hard break, or a text node that ends with whitespace, strip the
                // leading space.
                if (/^[ \t\r\n\u000c]/.test(value) && this.open == this.nodes.length - 1) {
                    let nodeBefore = top.content[top.content.length - 1];
                    let domNodeBefore = dom.previousSibling;
                    if (!nodeBefore ||
                        (domNodeBefore && domNodeBefore.nodeName == 'BR') ||
                        (nodeBefore.isText && /[ \t\r\n\u000c]$/.test(nodeBefore.text)))
                        value = value.slice(1);
                }
            }
            else if (preserveWS === "full") {
                value = value.replace(/\r\n?/g, "\n");
            }
            else if (schema.linebreakReplacement && /[\r\n]/.test(value) && this.top.findWrapping(schema.linebreakReplacement.create())) {
                let lines = value.split(/\r?\n|\r/);
                for (let i = 0; i < lines.length; i++) {
                    if (i)
                        this.insertNode(schema.linebreakReplacement.create(), marks, true);
                    if (lines[i])
                        this.insertNode(schema.text(lines[i]), marks, !/\S/.test(lines[i]));
                }
                value = "";
            }
            else {
                value = value.replace(/\r?\n|\r/g, " ");
            }
            if (value)
                this.insertNode(schema.text(value), marks, !/\S/.test(value));
            this.findInText(dom);
        }
        else {
            this.findInside(dom);
        }
    }
    // Try to find a handler for the given tag and use that to parse. If
    // none is found, the element's content nodes are added directly.
    addElement(dom, marks, matchAfter) {
        let outerWS = this.localPreserveWS, top = this.top;
        if (dom.tagName == "PRE" || /pre/.test(dom.style && dom.style.whiteSpace))
            this.localPreserveWS = true;
        let name = dom.nodeName.toLowerCase(), ruleID;
        if (listTags.hasOwnProperty(name) && this.parser.normalizeLists)
            normalizeList(dom);
        let rule = (this.options.ruleFromNode && this.options.ruleFromNode(dom)) ||
            (ruleID = this.parser.matchTag(dom, this, matchAfter));
        out: if (rule ? rule.ignore : ignoreTags.hasOwnProperty(name)) {
            this.findInside(dom);
            this.ignoreFallback(dom, marks);
        }
        else if (!rule || rule.skip || rule.closeParent) {
            if (rule && rule.closeParent)
                this.open = Math.max(0, this.open - 1);
            else if (rule && rule.skip.nodeType)
                dom = rule.skip;
            let sync, oldNeedsBlock = this.needsBlock;
            if (blockTags.hasOwnProperty(name)) {
                if (top.content.length && top.content[0].isInline && this.open) {
                    this.open--;
                    top = this.top;
                }
                sync = true;
                if (!top.type)
                    this.needsBlock = true;
            }
            else if (!dom.firstChild) {
                this.leafFallback(dom, marks);
                break out;
            }
            let innerMarks = rule && rule.skip ? marks : this.readStyles(dom, marks);
            if (innerMarks)
                this.addAll(dom, innerMarks);
            if (sync)
                this.sync(top);
            this.needsBlock = oldNeedsBlock;
        }
        else {
            let innerMarks = this.readStyles(dom, marks);
            if (innerMarks)
                this.addElementByRule(dom, rule, innerMarks, rule.consuming === false ? ruleID : undefined);
        }
        this.localPreserveWS = outerWS;
    }
    // Called for leaf DOM nodes that would otherwise be ignored
    leafFallback(dom, marks) {
        if (dom.nodeName == "BR" && this.top.type && this.top.type.inlineContent)
            this.addTextNode(dom.ownerDocument.createTextNode("\n"), marks);
    }
    // Called for ignored nodes
    ignoreFallback(dom, marks) {
        // Ignored BR nodes should at least create an inline context
        if (dom.nodeName == "BR" && (!this.top.type || !this.top.type.inlineContent))
            this.findPlace(this.parser.schema.text("-"), marks, true);
    }
    // Run any style parser associated with the node's styles. Either
    // return an updated array of marks, or null to indicate some of the
    // styles had a rule with `ignore` set.
    readStyles(dom, marks) {
        let styles = dom.style;
        // Because many properties will only show up in 'normalized' form
        // in `style.item` (i.e. text-decoration becomes
        // text-decoration-line, text-decoration-color, etc), we directly
        // query the styles mentioned in our rules instead of iterating
        // over the items.
        if (styles && styles.length)
            for (let i = 0; i < this.parser.matchedStyles.length; i++) {
                let name = this.parser.matchedStyles[i], value = styles.getPropertyValue(name);
                if (value)
                    for (let after = undefined;;) {
                        let rule = this.parser.matchStyle(name, value, this, after);
                        if (!rule)
                            break;
                        if (rule.ignore)
                            return null;
                        if (rule.clearMark)
                            marks = marks.filter(m => !rule.clearMark(m));
                        else
                            marks = marks.concat(this.parser.schema.marks[rule.mark].create(rule.attrs));
                        if (rule.consuming === false)
                            after = rule;
                        else
                            break;
                    }
            }
        return marks;
    }
    // Look up a handler for the given node. If none are found, return
    // false. Otherwise, apply it, use its return value to drive the way
    // the node's content is wrapped, and return true.
    addElementByRule(dom, rule, marks, continueAfter) {
        let sync, nodeType;
        if (rule.node) {
            nodeType = this.parser.schema.nodes[rule.node];
            if (!nodeType.isLeaf) {
                let inner = this.enter(nodeType, rule.attrs || null, marks, rule.preserveWhitespace);
                if (inner) {
                    sync = true;
                    marks = inner;
                }
            }
            else if (!this.insertNode(nodeType.create(rule.attrs), marks, dom.nodeName == "BR")) {
                this.leafFallback(dom, marks);
            }
        }
        else {
            let markType = this.parser.schema.marks[rule.mark];
            marks = marks.concat(markType.create(rule.attrs));
        }
        let startIn = this.top;
        if (nodeType && nodeType.isLeaf) {
            this.findInside(dom);
        }
        else if (continueAfter) {
            this.addElement(dom, marks, continueAfter);
        }
        else if (rule.getContent) {
            this.findInside(dom);
            rule.getContent(dom, this.parser.schema).forEach(node => this.insertNode(node, marks, false));
        }
        else {
            let contentDOM = dom;
            if (typeof rule.contentElement == "string")
                contentDOM = dom.querySelector(rule.contentElement);
            else if (typeof rule.contentElement == "function")
                contentDOM = rule.contentElement(dom);
            else if (rule.contentElement)
                contentDOM = rule.contentElement;
            this.findAround(dom, contentDOM, true);
            this.addAll(contentDOM, marks);
            this.findAround(dom, contentDOM, false);
        }
        if (sync && this.sync(startIn))
            this.open--;
    }
    // Add all child nodes between `startIndex` and `endIndex` (or the
    // whole node, if not given). If `sync` is passed, use it to
    // synchronize after every block element.
    addAll(parent, marks, startIndex, endIndex) {
        let index = startIndex || 0;
        for (let dom = startIndex ? parent.childNodes[startIndex] : parent.firstChild, end = endIndex == null ? null : parent.childNodes[endIndex]; dom != end; dom = dom.nextSibling, ++index) {
            this.findAtPoint(parent, index);
            this.addDOM(dom, marks);
        }
        this.findAtPoint(parent, index);
    }
    // Try to find a way to fit the given node type into the current
    // context. May add intermediate wrappers and/or leave non-solid
    // nodes that we're in.
    findPlace(node, marks, cautious) {
        let route, sync;
        for (let depth = this.open, penalty = 0; depth >= 0; depth--) {
            let cx = this.nodes[depth];
            let found = cx.findWrapping(node);
            if (found && (!route || route.length > found.length + penalty)) {
                route = found;
                sync = cx;
                if (!found.length)
                    break;
            }
            if (cx.solid) {
                if (cautious)
                    break;
                penalty += 2;
            }
        }
        if (!route)
            return null;
        this.sync(sync);
        for (let i = 0; i < route.length; i++)
            marks = this.enterInner(route[i], null, marks, false);
        return marks;
    }
    // Try to insert the given node, adjusting the context when needed.
    insertNode(node, marks, cautious) {
        if (node.isInline && this.needsBlock && !this.top.type) {
            let block = this.textblockFromContext();
            if (block)
                marks = this.enterInner(block, null, marks);
        }
        let innerMarks = this.findPlace(node, marks, cautious);
        if (innerMarks) {
            this.closeExtra();
            let top = this.top;
            if (top.match)
                top.match = top.match.matchType(node.type);
            let nodeMarks = Mark.none;
            for (let m of innerMarks.concat(node.marks))
                if (top.type ? top.type.allowsMarkType(m.type) : markMayApply(m.type, node.type))
                    nodeMarks = m.addToSet(nodeMarks);
            top.content.push(node.mark(nodeMarks));
            return true;
        }
        return false;
    }
    // Try to start a node of the given type, adjusting the context when
    // necessary.
    enter(type, attrs, marks, preserveWS) {
        let innerMarks = this.findPlace(type.create(attrs), marks, false);
        if (innerMarks)
            innerMarks = this.enterInner(type, attrs, marks, true, preserveWS);
        return innerMarks;
    }
    // Open a node of the given type
    enterInner(type, attrs, marks, solid = false, preserveWS) {
        this.closeExtra();
        let top = this.top;
        top.match = top.match && top.match.matchType(type);
        let options = wsOptionsFor(type, preserveWS, top.options);
        if ((top.options & OPT_OPEN_LEFT) && top.content.length == 0)
            options |= OPT_OPEN_LEFT;
        let applyMarks = Mark.none;
        marks = marks.filter(m => {
            if (top.type ? top.type.allowsMarkType(m.type) : markMayApply(m.type, type)) {
                applyMarks = m.addToSet(applyMarks);
                return false;
            }
            return true;
        });
        this.nodes.push(new NodeContext(type, attrs, applyMarks, solid, null, options));
        this.open++;
        return marks;
    }
    // Make sure all nodes above this.open are finished and added to
    // their parents
    closeExtra(openEnd = false) {
        let i = this.nodes.length - 1;
        if (i > this.open) {
            for (; i > this.open; i--)
                this.nodes[i - 1].content.push(this.nodes[i].finish(openEnd));
            this.nodes.length = this.open + 1;
        }
    }
    finish() {
        this.open = 0;
        this.closeExtra(this.isOpen);
        return this.nodes[0].finish(!!(this.isOpen || this.options.topOpen));
    }
    sync(to) {
        for (let i = this.open; i >= 0; i--) {
            if (this.nodes[i] == to) {
                this.open = i;
                return true;
            }
            else if (this.localPreserveWS) {
                this.nodes[i].options |= OPT_PRESERVE_WS;
            }
        }
        return false;
    }
    get currentPos() {
        this.closeExtra();
        let pos = 0;
        for (let i = this.open; i >= 0; i--) {
            let content = this.nodes[i].content;
            for (let j = content.length - 1; j >= 0; j--)
                pos += content[j].nodeSize;
            if (i)
                pos++;
        }
        return pos;
    }
    findAtPoint(parent, offset) {
        if (this.find)
            for (let i = 0; i < this.find.length; i++) {
                if (this.find[i].node == parent && this.find[i].offset == offset)
                    this.find[i].pos = this.currentPos;
            }
    }
    findInside(parent) {
        if (this.find)
            for (let i = 0; i < this.find.length; i++) {
                if (this.find[i].pos == null && parent.nodeType == 1 && parent.contains(this.find[i].node))
                    this.find[i].pos = this.currentPos;
            }
    }
    findAround(parent, content, before) {
        if (parent != content && this.find)
            for (let i = 0; i < this.find.length; i++) {
                if (this.find[i].pos == null && parent.nodeType == 1 && parent.contains(this.find[i].node)) {
                    let pos = content.compareDocumentPosition(this.find[i].node);
                    if (pos & (before ? 2 : 4))
                        this.find[i].pos = this.currentPos;
                }
            }
    }
    findInText(textNode) {
        if (this.find)
            for (let i = 0; i < this.find.length; i++) {
                if (this.find[i].node == textNode)
                    this.find[i].pos = this.currentPos - (textNode.nodeValue.length - this.find[i].offset);
            }
    }
    // Determines whether the given context string matches this context.
    matchesContext(context) {
        if (context.indexOf("|") > -1)
            return context.split(/\s*\|\s*/).some(this.matchesContext, this);
        let parts = context.split("/");
        let option = this.options.context;
        let useRoot = !this.isOpen && (!option || option.parent.type == this.nodes[0].type);
        let minDepth = -(option ? option.depth + 1 : 0) + (useRoot ? 0 : 1);
        let match = (i, depth) => {
            for (; i >= 0; i--) {
                let part = parts[i];
                if (part == "") {
                    if (i == parts.length - 1 || i == 0)
                        continue;
                    for (; depth >= minDepth; depth--)
                        if (match(i - 1, depth))
                            return true;
                    return false;
                }
                else {
                    let next = depth > 0 || (depth == 0 && useRoot) ? this.nodes[depth].type
                        : option && depth >= minDepth ? option.node(depth - minDepth).type
                            : null;
                    if (!next || (next.name != part && !next.isInGroup(part)))
                        return false;
                    depth--;
                }
            }
            return true;
        };
        return match(parts.length - 1, this.open);
    }
    textblockFromContext() {
        let $context = this.options.context;
        if ($context)
            for (let d = $context.depth; d >= 0; d--) {
                let deflt = $context.node(d).contentMatchAt($context.indexAfter(d)).defaultType;
                if (deflt && deflt.isTextblock && deflt.defaultAttrs)
                    return deflt;
            }
        for (let name in this.parser.schema.nodes) {
            let type = this.parser.schema.nodes[name];
            if (type.isTextblock && type.defaultAttrs)
                return type;
        }
    }
}
// Kludge to work around directly nested list nodes produced by some
// tools and allowed by browsers to mean that the nested list is
// actually part of the list item above it.
function normalizeList(dom) {
    for (let child = dom.firstChild, prevItem = null; child; child = child.nextSibling) {
        let name = child.nodeType == 1 ? child.nodeName.toLowerCase() : null;
        if (name && listTags.hasOwnProperty(name) && prevItem) {
            prevItem.appendChild(child);
            child = prevItem;
        }
        else if (name == "li") {
            prevItem = child;
        }
        else if (name) {
            prevItem = null;
        }
    }
}
// Apply a CSS selector.
function matches(dom, selector) {
    return (dom.matches || dom.msMatchesSelector || dom.webkitMatchesSelector || dom.mozMatchesSelector).call(dom, selector);
}
function copy(obj) {
    let copy = {};
    for (let prop in obj)
        copy[prop] = obj[prop];
    return copy;
}
// Used when finding a mark at the top level of a fragment parse.
// Checks whether it would be reasonable to apply a given mark type to
// a given node, by looking at the way the mark occurs in the schema.
function markMayApply(markType, nodeType) {
    let nodes = nodeType.schema.nodes;
    for (let name in nodes) {
        let parent = nodes[name];
        if (!parent.allowsMarkType(markType))
            continue;
        let seen = [], scan = (match) => {
            seen.push(match);
            for (let i = 0; i < match.edgeCount; i++) {
                let { type, next } = match.edge(i);
                if (type == nodeType)
                    return true;
                if (seen.indexOf(next) < 0 && scan(next))
                    return true;
            }
        };
        if (scan(parent.contentMatch))
            return true;
    }
}

/**
A DOM serializer knows how to convert ProseMirror nodes and
marks of various types to DOM nodes.
*/
class DOMSerializer {
    /**
    Create a serializer. `nodes` should map node names to functions
    that take a node and return a description of the corresponding
    DOM. `marks` does the same for mark names, but also gets an
    argument that tells it whether the mark's content is block or
    inline content (for typical use, it'll always be inline). A mark
    serializer may be `null` to indicate that marks of that type
    should not be serialized.
    */
    constructor(
    /**
    The node serialization functions.
    */
    nodes, 
    /**
    The mark serialization functions.
    */
    marks) {
        this.nodes = nodes;
        this.marks = marks;
    }
    /**
    Serialize the content of this fragment to a DOM fragment. When
    not in the browser, the `document` option, containing a DOM
    document, should be passed so that the serializer can create
    nodes.
    */
    serializeFragment(fragment, options = {}, target) {
        if (!target)
            target = doc(options).createDocumentFragment();
        let top = target, active = [];
        fragment.forEach(node => {
            if (active.length || node.marks.length) {
                let keep = 0, rendered = 0;
                while (keep < active.length && rendered < node.marks.length) {
                    let next = node.marks[rendered];
                    if (!this.marks[next.type.name]) {
                        rendered++;
                        continue;
                    }
                    if (!next.eq(active[keep][0]) || next.type.spec.spanning === false)
                        break;
                    keep++;
                    rendered++;
                }
                while (keep < active.length)
                    top = active.pop()[1];
                while (rendered < node.marks.length) {
                    let add = node.marks[rendered++];
                    let markDOM = this.serializeMark(add, node.isInline, options);
                    if (markDOM) {
                        active.push([add, top]);
                        top.appendChild(markDOM.dom);
                        top = markDOM.contentDOM || markDOM.dom;
                    }
                }
            }
            top.appendChild(this.serializeNodeInner(node, options));
        });
        return target;
    }
    /**
    @internal
    */
    serializeNodeInner(node, options) {
        if (node.isText)
            return doc(options).createTextNode(node.text);
        let { dom, contentDOM } = renderSpec(doc(options), this.nodes[node.type.name](node), null, node.attrs);
        if (contentDOM) {
            if (node.isLeaf)
                throw new RangeError("Content hole not allowed in a leaf node spec");
            this.serializeFragment(node.content, options, contentDOM);
        }
        return dom;
    }
    /**
    Serialize this node to a DOM node. This can be useful when you
    need to serialize a part of a document, as opposed to the whole
    document. To serialize a whole document, use
    [`serializeFragment`](https://prosemirror.net/docs/ref/#model.DOMSerializer.serializeFragment) on
    its [content](https://prosemirror.net/docs/ref/#model.Node.content).
    */
    serializeNode(node, options = {}) {
        let dom = this.serializeNodeInner(node, options);
        for (let i = node.marks.length - 1; i >= 0; i--) {
            let wrap = this.serializeMark(node.marks[i], node.isInline, options);
            if (wrap) {
                (wrap.contentDOM || wrap.dom).appendChild(dom);
                dom = wrap.dom;
            }
        }
        return dom;
    }
    /**
    @internal
    */
    serializeMark(mark, inline, options = {}) {
        let toDOM = this.marks[mark.type.name];
        return toDOM && renderSpec(doc(options), toDOM(mark, inline), null, mark.attrs);
    }
    static renderSpec(doc, structure, xmlNS = null, blockArraysIn) {
        // Kludge for backwards-compatibility with accidental original behavious
        if (typeof structure == "string")
            return { dom: doc.createTextNode(structure) };
        return renderSpec(doc, structure, xmlNS, blockArraysIn);
    }
    /**
    Build a serializer using the [`toDOM`](https://prosemirror.net/docs/ref/#model.NodeSpec.toDOM)
    properties in a schema's node and mark specs.
    */
    static fromSchema(schema) {
        return schema.cached.domSerializer ||
            (schema.cached.domSerializer = new DOMSerializer(this.nodesFromSchema(schema), this.marksFromSchema(schema)));
    }
    /**
    Gather the serializers in a schema's node specs into an object.
    This can be useful as a base to build a custom serializer from.
    */
    static nodesFromSchema(schema) {
        let result = gatherToDOM(schema.nodes);
        if (!result.text)
            result.text = node => node.text;
        return result;
    }
    /**
    Gather the serializers in a schema's mark specs into an object.
    */
    static marksFromSchema(schema) {
        return gatherToDOM(schema.marks);
    }
}
function gatherToDOM(obj) {
    let result = {};
    for (let name in obj) {
        let toDOM = obj[name].spec.toDOM;
        if (toDOM)
            result[name] = toDOM;
    }
    return result;
}
function doc(options) {
    return options.document || window.document;
}
const suspiciousAttributeCache = new WeakMap();
function suspiciousAttributes(attrs) {
    let value = suspiciousAttributeCache.get(attrs);
    if (value === undefined)
        suspiciousAttributeCache.set(attrs, value = suspiciousAttributesInner(attrs));
    return value;
}
function suspiciousAttributesInner(attrs) {
    let result = null;
    function scan(value) {
        if (value && typeof value == "object") {
            if (Array.isArray(value)) {
                if (typeof value[0] == "string") {
                    if (!result)
                        result = [];
                    result.push(value);
                }
                else {
                    for (let i = 0; i < value.length; i++)
                        scan(value[i]);
                }
            }
            else {
                for (let prop in value)
                    scan(value[prop]);
            }
        }
    }
    scan(attrs);
    return result;
}
function renderSpec(doc, structure, xmlNS, blockArraysIn) {
    if (structure.nodeType == 1)
        return { dom: structure };
    if (structure.dom && structure.dom.nodeType == 1)
        return structure;
    let tagName = structure[0], suspicious;
    if (typeof tagName != "string")
        throw new RangeError("Invalid array passed to renderSpec");
    if (blockArraysIn && (suspicious = suspiciousAttributes(blockArraysIn)) &&
        suspicious.indexOf(structure) > -1)
        throw new RangeError("Using an array from an attribute object as a DOM spec. This may be an attempted cross site scripting attack.");
    let space = tagName.indexOf(" ");
    if (space > 0) {
        xmlNS = tagName.slice(0, space);
        tagName = tagName.slice(space + 1);
    }
    let contentDOM;
    let dom = (xmlNS ? doc.createElementNS(xmlNS, tagName) : doc.createElement(tagName));
    let attrs = structure[1], start = 1;
    if (attrs && typeof attrs == "object" && attrs.nodeType == null && !Array.isArray(attrs)) {
        start = 2;
        for (let name in attrs)
            if (attrs[name] != null) {
                let space = name.indexOf(" ");
                if (space > 0)
                    dom.setAttributeNS(name.slice(0, space), name.slice(space + 1), attrs[name]);
                else if (name == "style" && dom.style)
                    dom.style.cssText = attrs[name];
                else
                    dom.setAttribute(name, attrs[name]);
            }
    }
    for (let i = start; i < structure.length; i++) {
        let child = structure[i];
        if (child === 0) {
            if (i < structure.length - 1 || i > start)
                throw new RangeError("Content hole must be the only child of its parent node");
            return { dom, contentDOM: dom };
        }
        else if (typeof child == "string") {
            dom.appendChild(doc.createTextNode(child));
        }
        else {
            let { dom: inner, contentDOM: innerContent } = renderSpec(doc, child, xmlNS, blockArraysIn);
            dom.appendChild(inner);
            if (innerContent) {
                if (contentDOM)
                    throw new RangeError("Multiple content holes");
                contentDOM = innerContent;
            }
        }
    }
    return { dom, contentDOM };
}

export { ContentMatch, DOMParser, DOMSerializer, Fragment, Mark, MarkType, Node, NodeRange, NodeType, ReplaceError, ResolvedPos, Schema, Slice };
