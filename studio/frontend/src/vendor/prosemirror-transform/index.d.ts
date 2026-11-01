// prosemirror-transform 1.12.4 from npm (sha512-2/WuPum3pn+O/Vi2vzpi8210ICUe2kIPXdLYjQ09Pv5DtT5bcKWMJhAKPBwvpPjpp5iOYUVwVEr0gOM1BY1Mgw==), MIT: see LICENSE. Imports point at the sibling vendored packages.
import { Node, Slice, Schema, Fragment, NodeRange, NodeType, Attrs, Mark, MarkType, ContentMatch } from '../prosemirror-model/index.js';

/**
There are several things that positions can be mapped through.
Such objects conform to this interface.
*/
interface Mappable {
    /**
    Map a position through this object. When given, `assoc` (should
    be -1 or 1, defaults to 1) determines with which side the
    position is associated, which determines in which direction to
    move when a chunk of content is inserted at the mapped position.
    */
    map: (pos: number, assoc?: number) => number;
    /**
    Map a position, and return an object containing additional
    information about the mapping. The result's `deleted` field tells
    you whether the position was deleted (completely enclosed in a
    replaced range) during the mapping. When content on only one side
    is deleted, the position itself is only considered deleted when
    `assoc` points in the direction of the deleted content.
    */
    mapResult: (pos: number, assoc?: number) => MapResult;
}
/**
An object representing a mapped position with extra
information.
*/
declare class MapResult {
    /**
    The mapped version of the position.
    */
    readonly pos: number;
    /**
    Tells you whether the position was deleted, that is, whether the
    step removed the token on the side queried (via the `assoc`)
    argument from the document.
    */
    get deleted(): boolean;
    /**
    Tells you whether the token before the mapped position was deleted.
    */
    get deletedBefore(): boolean;
    /**
    True when the token after the mapped position was deleted.
    */
    get deletedAfter(): boolean;
    /**
    Tells whether any of the steps mapped through deletes across the
    position (including both the token before and after the
    position).
    */
    get deletedAcross(): boolean;
}
/**
A map describing the deletions and insertions made by a step, which
can be used to find the correspondence between positions in the
pre-step version of a document and the same position in the
post-step version.
*/
declare class StepMap implements Mappable {
    /**
    Create a position map. The modifications to the document are
    represented as an array of numbers, in which each group of three
    represents a modified chunk as `[start, oldSize, newSize]`.
    */
    constructor(
    /**
    @internal
    */
    ranges: readonly number[], 
    /**
    @internal
    */
    inverted?: boolean);
    mapResult(pos: number, assoc?: number): MapResult;
    map(pos: number, assoc?: number): number;
    /**
    Calls the given function on each of the changed ranges included in
    this map.
    */
    forEach(f: (oldStart: number, oldEnd: number, newStart: number, newEnd: number) => void): void;
    /**
    Create an inverted version of this map. The result can be used to
    map positions in the post-step document to the pre-step document.
    */
    invert(): StepMap;
    /**
    Create a map that moves all positions by offset `n` (which may be
    negative). This can be useful when applying steps meant for a
    sub-document to a larger document, or vice-versa.
    */
    static offset(n: number): StepMap;
    /**
    A StepMap that contains no changed ranges.
    */
    static empty: StepMap;
}
/**
A mapping represents a pipeline of zero or more [step
maps](https://prosemirror.net/docs/ref/#transform.StepMap). It has special provisions for losslessly
handling mapping positions through a series of steps in which some
steps are inverted versions of earlier steps. (This comes up when
‘[rebasing](https://prosemirror.net/docs/guide/#transform.rebasing)’ steps for
collaboration or history management.)
*/
declare class Mapping implements Mappable {
    /**
    The starting position in the `maps` array, used when `map` or
    `mapResult` is called.
    */
    from: number;
    /**
    The end position in the `maps` array.
    */
    to: number;
    /**
    Create a new mapping with the given position maps.
    */
    constructor(maps?: readonly StepMap[], 
    /**
    @internal
    */
    mirror?: number[] | undefined, 
    /**
    The starting position in the `maps` array, used when `map` or
    `mapResult` is called.
    */
    from?: number, 
    /**
    The end position in the `maps` array.
    */
    to?: number);
    /**
    The step maps in this mapping.
    */
    get maps(): readonly StepMap[];
    private _maps;
    private ownData;
    /**
    Create a mapping that maps only through a part of this one.
    */
    slice(from?: number, to?: number): Mapping;
    /**
    Add a step map to the end of this mapping. If `mirrors` is
    given, it should be the index of the step map that is the mirror
    image of this one.
    */
    appendMap(map: StepMap, mirrors?: number): void;
    /**
    Add all the step maps in a given mapping to this one (preserving
    mirroring information).
    */
    appendMapping(mapping: Mapping): void;
    /**
    Finds the offset of the step map that mirrors the map at the
    given offset, in this mapping (as per the second argument to
    `appendMap`).
    */
    getMirror(n: number): number | undefined;
    /**
    Append the inverse of the given mapping to this one.
    */
    appendMappingInverted(mapping: Mapping): void;
    /**
    Create an inverted version of this mapping.
    */
    invert(): Mapping;
    /**
    Map a position through this mapping.
    */
    map(pos: number, assoc?: number): number;
    /**
    Map a position through this mapping, returning a mapping
    result.
    */
    mapResult(pos: number, assoc?: number): MapResult;
}

/**
A step object represents an atomic change. It generally applies
only to the document it was created for, since the positions
stored in it will only make sense for that document.

New steps are defined by creating classes that extend `Step`,
overriding the `apply`, `invert`, `map`, `getMap` and `fromJSON`
methods, and registering your class with a unique
JSON-serialization identifier using
[`Step.jsonID`](https://prosemirror.net/docs/ref/#transform.Step^jsonID).
*/
declare abstract class Step {
    /**
    Applies this step to the given document, returning a result
    object that either indicates failure, if the step can not be
    applied to this document, or indicates success by containing a
    transformed document.
    */
    abstract apply(doc: Node): StepResult;
    /**
    Get the step map that represents the changes made by this step,
    and which can be used to transform between positions in the old
    and the new document.
    */
    getMap(): StepMap;
    /**
    Create an inverted version of this step. Needs the document as it
    was before the step as argument.
    */
    abstract invert(doc: Node): Step;
    /**
    Map this step through a mappable thing, returning either a
    version of that step with its positions adjusted, or `null` if
    the step was entirely deleted by the mapping.
    */
    abstract map(mapping: Mappable): Step | null;
    /**
    Try to merge this step with another one, to be applied directly
    after it. Returns the merged step when possible, null if the
    steps can't be merged.
    */
    merge(other: Step): Step | null;
    /**
    Create a JSON-serializeable representation of this step. When
    defining this for a custom subclass, make sure the result object
    includes the step type's [JSON id](https://prosemirror.net/docs/ref/#transform.Step^jsonID) under
    the `stepType` property.
    */
    abstract toJSON(): any;
    /**
    Deserialize a step from its JSON representation. Will call
    through to the step class' own implementation of this method.
    */
    static fromJSON(schema: Schema, json: any): Step;
    /**
    To be able to serialize steps to JSON, each step needs a string
    ID to attach to its JSON representation. Use this method to
    register an ID for your step classes. Try to pick something
    that's unlikely to clash with steps from other modules.
    */
    static jsonID(id: string, stepClass: {
        fromJSON(schema: Schema, json: any): Step;
    }): {
        fromJSON(schema: Schema, json: any): Step;
    };
}
/**
The result of [applying](https://prosemirror.net/docs/ref/#transform.Step.apply) a step. Contains either a
new document or a failure value.
*/
declare class StepResult {
    /**
    The transformed document, if successful.
    */
    readonly doc: Node | null;
    /**
    The failure message, if unsuccessful.
    */
    readonly failed: string | null;
    /**
    Create a successful step result.
    */
    static ok(doc: Node): StepResult;
    /**
    Create a failed step result.
    */
    static fail(message: string): StepResult;
    /**
    Call [`Node.replace`](https://prosemirror.net/docs/ref/#model.Node.replace) with the given
    arguments. Create a successful result if it succeeds, and a
    failed one if it throws a `ReplaceError`.
    */
    static fromReplace(doc: Node, from: number, to: number, slice: Slice): StepResult;
}

/**
Abstraction to build up and track an array of
[steps](https://prosemirror.net/docs/ref/#transform.Step) representing a document transformation.

Most transforming methods return the `Transform` object itself, so
that they can be chained.
*/
declare class Transform {
    /**
    The current document (the result of applying the steps in the
    transform).
    */
    doc: Node;
    /**
    The steps in this transform.
    */
    readonly steps: Step[];
    /**
    The documents before each of the steps.
    */
    readonly docs: Node[];
    /**
    A mapping with the maps for each of the steps in this transform.
    */
    readonly mapping: Mapping;
    /**
    Create a transform that starts with the given document.
    */
    constructor(
    /**
    The current document (the result of applying the steps in the
    transform).
    */
    doc: Node);
    /**
    The starting document.
    */
    get before(): Node;
    /**
    Apply a new step in this transform, saving the result. Throws an
    error when the step fails.
    */
    step(step: Step): this;
    /**
    Try to apply a step in this transformation, ignoring it if it
    fails. Returns the step result.
    */
    maybeStep(step: Step): StepResult;
    /**
    True when the document has been changed (when there are any
    steps).
    */
    get docChanged(): boolean;
    /**
    Return a single range, in post-transform document positions,
    that covers all content changed by this transform. Returns null
    if no replacements are made. Note that this will ignore changes
    that add/remove marks without replacing the underlying content.
    */
    changedRange(): {
        from: number;
        to: number;
    } | null;
    /**
    Replace the part of the document between `from` and `to` with the
    given `slice`.
    */
    replace(from: number, to?: number, slice?: Slice): this;
    /**
    Replace the given range with the given content, which may be a
    fragment, node, or array of nodes.
    */
    replaceWith(from: number, to: number, content: Fragment | Node | readonly Node[]): this;
    /**
    Delete the content between the given positions.
    */
    delete(from: number, to: number): this;
    /**
    Insert the given content at the given position.
    */
    insert(pos: number, content: Fragment | Node | readonly Node[]): this;
    /**
    Replace a range of the document with a given slice, using
    `from`, `to`, and the slice's
    [`openStart`](https://prosemirror.net/docs/ref/#model.Slice.openStart) property as hints, rather
    than fixed start and end points. This method may grow the
    replaced area or close open nodes in the slice in order to get a
    fit that is more in line with WYSIWYG expectations, by dropping
    fully covered parent nodes of the replaced region when they are
    marked [non-defining as
    context](https://prosemirror.net/docs/ref/#model.NodeSpec.definingAsContext), or including an
    open parent node from the slice that _is_ marked as [defining
    its content](https://prosemirror.net/docs/ref/#model.NodeSpec.definingForContent).
    
    This is the method, for example, to handle paste. The similar
    [`replace`](https://prosemirror.net/docs/ref/#transform.Transform.replace) method is a more
    primitive tool which will _not_ move the start and end of its given
    range, and is useful in situations where you need more precise
    control over what happens.
    */
    replaceRange(from: number, to: number, slice: Slice): this;
    /**
    Replace the given range with a node, but use `from` and `to` as
    hints, rather than precise positions. When from and to are the same
    and are at the start or end of a parent node in which the given
    node doesn't fit, this method may _move_ them out towards a parent
    that does allow the given node to be placed. When the given range
    completely covers a parent node, this method may completely replace
    that parent node.
    */
    replaceRangeWith(from: number, to: number, node: Node): this;
    /**
    Delete the given range, expanding it to cover fully covered
    parent nodes until a valid replace is found.
    */
    deleteRange(from: number, to: number): this;
    /**
    Split the content in the given range off from its parent, if there
    is sibling content before or after it, and move it up the tree to
    the depth specified by `target`. You'll probably want to use
    [`liftTarget`](https://prosemirror.net/docs/ref/#transform.liftTarget) to compute `target`, to make
    sure the lift is valid.
    */
    lift(range: NodeRange, target: number): this;
    /**
    Join the blocks around the given position. If depth is 2, their
    last and first siblings are also joined, and so on.
    */
    join(pos: number, depth?: number): this;
    /**
    Wrap the given [range](https://prosemirror.net/docs/ref/#model.NodeRange) in the given set of wrappers.
    The wrappers are assumed to be valid in this position, and should
    probably be computed with [`findWrapping`](https://prosemirror.net/docs/ref/#transform.findWrapping).
    */
    wrap(range: NodeRange, wrappers: readonly {
        type: NodeType;
        attrs?: Attrs | null;
    }[]): this;
    /**
    Set the type of all textblocks (partly) between `from` and `to` to
    the given node type with the given attributes.
    */
    setBlockType(from: number, to: number | undefined, type: NodeType, attrs?: Attrs | null | ((oldNode: Node) => Attrs)): this;
    /**
    Change the type, attributes, and/or marks of the node at `pos`.
    When `type` isn't given, the existing node type is preserved,
    */
    setNodeMarkup(pos: number, type?: NodeType | null, attrs?: Attrs | null, marks?: readonly Mark[]): this;
    /**
    Set a single attribute on a given node to a new value.
    The `pos` addresses the document content. Use `setDocAttribute`
    to set attributes on the document itself.
    */
    setNodeAttribute(pos: number, attr: string, value: any): this;
    /**
    Set a single attribute on the document to a new value.
    */
    setDocAttribute(attr: string, value: any): this;
    /**
    Add a mark to the node at position `pos`.
    */
    addNodeMark(pos: number, mark: Mark): this;
    /**
    Remove a mark (or all marks of the given type) from the node at
    position `pos`.
    */
    removeNodeMark(pos: number, mark: Mark | MarkType): this;
    /**
    Split the node at the given position, and optionally, if `depth` is
    greater than one, any number of nodes above that. By default, the
    parts split off will inherit the node type of the original node.
    This can be changed by passing an array of types and attributes to
    use after the split (with the outermost nodes coming first).
    */
    split(pos: number, depth?: number, typesAfter?: (null | {
        type: NodeType;
        attrs?: Attrs | null;
    })[]): this;
    /**
    Add the given mark to the inline content between `from` and `to`.
    */
    addMark(from: number, to: number, mark: Mark): this;
    /**
    Remove marks from inline nodes between `from` and `to`. When
    `mark` is a single mark, remove precisely that mark. When it is
    a mark type, remove all marks of that type. When it is null,
    remove all marks of any type.
    */
    removeMark(from: number, to: number, mark?: Mark | MarkType | null): this;
    /**
    Removes all marks and nodes from the content of the node at
    `pos` that don't match the given new parent node type. Accepts
    an optional starting [content match](https://prosemirror.net/docs/ref/#model.ContentMatch) as
    third argument.
    */
    clearIncompatible(pos: number, parentType: NodeType, match?: ContentMatch): this;
}

/**
Try to find a target depth to which the content in the given range
can be lifted. Will not go across
[isolating](https://prosemirror.net/docs/ref/#model.NodeSpec.isolating) parent nodes.
*/
declare function liftTarget(range: NodeRange): number | null;
/**
Try to find a valid way to wrap the content in the given range in a
node of the given type. May introduce extra nodes around and inside
the wrapper node, if necessary. Returns null if no valid wrapping
could be found. When `innerRange` is given, that range's content is
used as the content to fit into the wrapping, instead of the
content of `range`.
*/
declare function findWrapping(range: NodeRange, nodeType: NodeType, attrs?: Attrs | null, innerRange?: NodeRange): {
    type: NodeType;
    attrs: Attrs | null;
}[] | null;
/**
Check whether splitting at the given position is allowed.
*/
declare function canSplit(doc: Node, pos: number, depth?: number, typesAfter?: (null | {
    type: NodeType;
    attrs?: Attrs | null;
})[]): boolean;
/**
Test whether the blocks before and after a given position can be
joined.
*/
declare function canJoin(doc: Node, pos: number): boolean;
/**
Find an ancestor of the given position that can be joined to the
block before (or after if `dir` is positive). Returns the joinable
point, if any.
*/
declare function joinPoint(doc: Node, pos: number, dir?: number): number | undefined;
/**
Try to find a point where a node of the given type can be inserted
near `pos`, by searching up the node hierarchy when `pos` itself
isn't a valid place but is at the start or end of a node. Return
null if no position was found.
*/
declare function insertPoint(doc: Node, pos: number, nodeType: NodeType): number | null;
/**
Finds a position at or around the given position where the given
slice can be inserted. Will look at parent nodes' nearest boundary
and try there, even if the original position wasn't directly at the
start or end of that node. Returns null when no position was found.
*/
declare function dropPoint(doc: Node, pos: number, slice: Slice): number | null;

/**
Add a mark to all inline content between two positions.
*/
declare class AddMarkStep extends Step {
    /**
    The start of the marked range.
    */
    readonly from: number;
    /**
    The end of the marked range.
    */
    readonly to: number;
    /**
    The mark to add.
    */
    readonly mark: Mark;
    /**
    Create a mark step.
    */
    constructor(
    /**
    The start of the marked range.
    */
    from: number, 
    /**
    The end of the marked range.
    */
    to: number, 
    /**
    The mark to add.
    */
    mark: Mark);
    apply(doc: Node): StepResult;
    invert(doc: Node): Step;
    map(mapping: Mappable): Step | null;
    merge(other: Step): Step | null;
    toJSON(): any;
}
/**
Remove a mark from all inline content between two positions.
*/
declare class RemoveMarkStep extends Step {
    /**
    The start of the unmarked range.
    */
    readonly from: number;
    /**
    The end of the unmarked range.
    */
    readonly to: number;
    /**
    The mark to remove.
    */
    readonly mark: Mark;
    /**
    Create a mark-removing step.
    */
    constructor(
    /**
    The start of the unmarked range.
    */
    from: number, 
    /**
    The end of the unmarked range.
    */
    to: number, 
    /**
    The mark to remove.
    */
    mark: Mark);
    apply(doc: Node): StepResult;
    invert(doc: Node): Step;
    map(mapping: Mappable): Step | null;
    merge(other: Step): Step | null;
    toJSON(): any;
}
/**
Add a mark to a specific node.
*/
declare class AddNodeMarkStep extends Step {
    /**
    The position of the target node.
    */
    readonly pos: number;
    /**
    The mark to add.
    */
    readonly mark: Mark;
    /**
    Create a node mark step.
    */
    constructor(
    /**
    The position of the target node.
    */
    pos: number, 
    /**
    The mark to add.
    */
    mark: Mark);
    apply(doc: Node): StepResult;
    invert(doc: Node): Step;
    map(mapping: Mappable): Step | null;
    toJSON(): any;
}
/**
Remove a mark from a specific node.
*/
declare class RemoveNodeMarkStep extends Step {
    /**
    The position of the target node.
    */
    readonly pos: number;
    /**
    The mark to remove.
    */
    readonly mark: Mark;
    /**
    Create a mark-removing step.
    */
    constructor(
    /**
    The position of the target node.
    */
    pos: number, 
    /**
    The mark to remove.
    */
    mark: Mark);
    apply(doc: Node): StepResult;
    invert(doc: Node): Step;
    map(mapping: Mappable): Step | null;
    toJSON(): any;
}

/**
Replace a part of the document with a slice of new content.
*/
declare class ReplaceStep extends Step {
    /**
    The start position of the replaced range.
    */
    readonly from: number;
    /**
    The end position of the replaced range.
    */
    readonly to: number;
    /**
    The slice to insert.
    */
    readonly slice: Slice;
    /**
    The given `slice` should fit the 'gap' between `from` and
    `to`—the depths must line up, and the surrounding nodes must be
    able to be joined with the open sides of the slice. When
    `structure` is true, the step will fail if the content between
    from and to is not just a sequence of closing and then opening
    tokens (this is to guard against rebased replace steps
    overwriting something they weren't supposed to).
    */
    constructor(
    /**
    The start position of the replaced range.
    */
    from: number, 
    /**
    The end position of the replaced range.
    */
    to: number, 
    /**
    The slice to insert.
    */
    slice: Slice, 
    /**
    @internal
    */
    structure?: boolean);
    apply(doc: Node): StepResult;
    getMap(): StepMap;
    invert(doc: Node): ReplaceStep;
    map(mapping: Mappable): ReplaceStep | null;
    merge(other: Step): ReplaceStep | null;
    toJSON(): any;
    /**
    By default, for backwards compatibility, an inserting step
    mapped over an insertion at that same position fill move after
    the inserted content. In a collaborative editing situation, that
    can make redone insertions appear in unexpected places. You can
    set this to -1 to make such mapping keep the step before the
    insertion instead.
    */
    static MAP_BIAS: -1 | 1;
}
/**
Replace a part of the document with a slice of content, but
preserve a range of the replaced content by moving it into the
slice.
*/
declare class ReplaceAroundStep extends Step {
    /**
    The start position of the replaced range.
    */
    readonly from: number;
    /**
    The end position of the replaced range.
    */
    readonly to: number;
    /**
    The start of preserved range.
    */
    readonly gapFrom: number;
    /**
    The end of preserved range.
    */
    readonly gapTo: number;
    /**
    The slice to insert.
    */
    readonly slice: Slice;
    /**
    The position in the slice where the preserved range should be
    inserted.
    */
    readonly insert: number;
    /**
    Create a replace-around step with the given range and gap.
    `insert` should be the point in the slice into which the content
    of the gap should be moved. `structure` has the same meaning as
    it has in the [`ReplaceStep`](https://prosemirror.net/docs/ref/#transform.ReplaceStep) class.
    */
    constructor(
    /**
    The start position of the replaced range.
    */
    from: number, 
    /**
    The end position of the replaced range.
    */
    to: number, 
    /**
    The start of preserved range.
    */
    gapFrom: number, 
    /**
    The end of preserved range.
    */
    gapTo: number, 
    /**
    The slice to insert.
    */
    slice: Slice, 
    /**
    The position in the slice where the preserved range should be
    inserted.
    */
    insert: number, 
    /**
    @internal
    */
    structure?: boolean);
    apply(doc: Node): StepResult;
    getMap(): StepMap;
    invert(doc: Node): ReplaceAroundStep;
    map(mapping: Mappable): ReplaceAroundStep | null;
    toJSON(): any;
}

/**
Update an attribute in a specific node.
*/
declare class AttrStep extends Step {
    /**
    The position of the target node.
    */
    readonly pos: number;
    /**
    The attribute to set.
    */
    readonly attr: string;
    readonly value: any;
    /**
    Construct an attribute step.
    */
    constructor(
    /**
    The position of the target node.
    */
    pos: number, 
    /**
    The attribute to set.
    */
    attr: string, value: any);
    apply(doc: Node): StepResult;
    getMap(): StepMap;
    invert(doc: Node): AttrStep;
    map(mapping: Mappable): AttrStep | null;
    toJSON(): any;
    static fromJSON(schema: Schema, json: any): AttrStep;
}
/**
Update an attribute in the doc node.
*/
declare class DocAttrStep extends Step {
    /**
    The attribute to set.
    */
    readonly attr: string;
    readonly value: any;
    /**
    Construct an attribute step.
    */
    constructor(
    /**
    The attribute to set.
    */
    attr: string, value: any);
    apply(doc: Node): StepResult;
    getMap(): StepMap;
    invert(doc: Node): DocAttrStep;
    map(mapping: Mappable): this;
    toJSON(): any;
    static fromJSON(schema: Schema, json: any): DocAttrStep;
}

/**
‘Fit’ a slice into a given position in the document, producing a
[step](https://prosemirror.net/docs/ref/#transform.Step) that inserts it. Will return null if
there's no meaningful way to insert the slice here, or inserting it
would be a no-op (an empty slice over an empty range).
*/
declare function replaceStep(doc: Node, from: number, to?: number, slice?: Slice): Step | null;

export { AddMarkStep, AddNodeMarkStep, AttrStep, DocAttrStep, MapResult, Mapping, RemoveMarkStep, RemoveNodeMarkStep, ReplaceAroundStep, ReplaceStep, Step, StepMap, StepResult, Transform, canJoin, canSplit, dropPoint, findWrapping, insertPoint, joinPoint, liftTarget, replaceStep };
export type { Mappable };
