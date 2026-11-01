// prosemirror-model 1.25.12 from npm (sha512-Ue2gTmXMa7EhpLNhC7J+h4+ykD8ha12K6rrZFFKKJHBForfIStw5gJ6Zrf1mqqAa9NmpmEB4wp9bKmA4eUVjWg==), MIT: see LICENSE. Imports point at the sibling vendored packages.
import OrderedMap from '../orderedmap/index.js';

/**
A mark is a piece of information that can be attached to a node,
such as it being emphasized, in code font, or a link. It has a
type and optionally a set of attributes that provide further
information (such as the target of the link). Marks are created
through a `Schema`, which controls which types exist and which
attributes they have.
*/
declare class Mark {
    /**
    The type of this mark.
    */
    readonly type: MarkType;
    /**
    The attributes associated with this mark.
    */
    readonly attrs: Attrs;
    /**
    Given a set of marks, create a new set which contains this one as
    well, in the right position. If this mark is already in the set,
    the set itself is returned. If any marks that are set to be
    [exclusive](https://prosemirror.net/docs/ref/#model.MarkSpec.excludes) with this mark are present,
    those are replaced by this one.
    */
    addToSet(set: readonly Mark[]): readonly Mark[];
    /**
    Remove this mark from the given set, returning a new set. If this
    mark is not in the set, the set itself is returned.
    */
    removeFromSet(set: readonly Mark[]): readonly Mark[];
    /**
    Test whether this mark is in the given set of marks.
    */
    isInSet(set: readonly Mark[]): boolean;
    /**
    Test whether this mark has the same type and attributes as
    another mark.
    */
    eq(other: Mark): boolean;
    /**
    Convert this mark to a JSON-serializeable representation.
    */
    toJSON(): any;
    /**
    Deserialize a mark from JSON.
    */
    static fromJSON(schema: Schema, json: any): Mark;
    /**
    Test whether two sets of marks are identical.
    */
    static sameSet(a: readonly Mark[], b: readonly Mark[]): boolean;
    /**
    Create a properly sorted mark set from null, a single mark, or an
    unsorted array of marks.
    */
    static setFrom(marks?: Mark | readonly Mark[] | null): readonly Mark[];
    /**
    The empty set of marks.
    */
    static none: readonly Mark[];
}

/**
A description of a DOM structure. Can be either a DOM element, a
`{dom, contentDOM}` object, or an array.

An array describes a DOM element. The first value in the array
should be a string—the name of the DOM element, optionally
prefixed by a namespace URL and a space. If the second element is
plain object, it is interpreted as a set of attributes for the
element. Any elements after that (including the 2nd if it's not an
attribute object) are interpreted as children of the DOM elements,
and must either be valid `DOMOutputSpec` values, strings (for text
nodes), or the number zero.

The number zero (pronounced “hole”) is used to indicate the place
where a node's child nodes should be inserted. If it occurs in an
output spec, it should be the only child element in its parent
node.
*/
type DOMOutputSpec = Element | {
    dom: Element;
    contentDOM?: Element;
} | readonly [string, ...any[]];
/**
A DOM serializer knows how to convert ProseMirror nodes and
marks of various types to DOM nodes.
*/
declare class DOMSerializer {
    /**
    The node serialization functions.
    */
    readonly nodes: {
        [node: string]: (node: Node) => DOMOutputSpec;
    };
    /**
    The mark serialization functions.
    */
    readonly marks: {
        [mark: string]: (mark: Mark, inline: boolean) => DOMOutputSpec;
    };
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
    nodes: {
        [node: string]: (node: Node) => DOMOutputSpec;
    }, 
    /**
    The mark serialization functions.
    */
    marks: {
        [mark: string]: (mark: Mark, inline: boolean) => DOMOutputSpec;
    });
    /**
    Serialize the content of this fragment to a DOM fragment. When
    not in the browser, the `document` option, containing a DOM
    document, should be passed so that the serializer can create
    nodes.
    */
    serializeFragment(fragment: Fragment, options?: {
        document?: Document;
    }, target?: HTMLElement | DocumentFragment): DocumentFragment | HTMLElement;
    /**
    Serialize this node to a DOM node. This can be useful when you
    need to serialize a part of a document, as opposed to the whole
    document. To serialize a whole document, use
    [`serializeFragment`](https://prosemirror.net/docs/ref/#model.DOMSerializer.serializeFragment) on
    its [content](https://prosemirror.net/docs/ref/#model.Node.content).
    */
    serializeNode(node: Node, options?: {
        document?: Document;
    }): HTMLElement | Text;
    /**
    Render an [output spec](https://prosemirror.net/docs/ref/#model.DOMOutputSpec) to a DOM node. If
    the spec has a hole (zero) in it, `contentDOM` will point at the
    node with the hole.
    */
    static renderSpec(doc: Document, structure: DOMOutputSpec, xmlNS?: string | null): {
        dom: HTMLElement;
        contentDOM?: HTMLElement;
    };
    /**
    Build a serializer using the [`toDOM`](https://prosemirror.net/docs/ref/#model.NodeSpec.toDOM)
    properties in a schema's node and mark specs.
    */
    static fromSchema(schema: Schema): DOMSerializer;
    /**
    Gather the serializers in a schema's node specs into an object.
    This can be useful as a base to build a custom serializer from.
    */
    static nodesFromSchema(schema: Schema): {
        [node: string]: (node: Node) => DOMOutputSpec;
    };
    /**
    Gather the serializers in a schema's mark specs into an object.
    */
    static marksFromSchema(schema: Schema): {
        [mark: string]: (mark: Mark, inline: boolean) => DOMOutputSpec;
    };
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
declare class ResolvedPos {
    /**
    The position that was resolved.
    */
    readonly pos: number;
    /**
    The offset this position has into its parent node.
    */
    readonly parentOffset: number;
    /**
    The number of levels the parent node is from the root. If this
    position points directly into the root node, it is 0. If it
    points into a top-level paragraph, 1, and so on.
    */
    depth: number;
    /**
    The parent node that the position points into. Note that even if
    a position points into a text node, that node is not considered
    the parent—text nodes are ‘flat’ in this model, and have no content.
    */
    get parent(): Node;
    /**
    The root node in which the position was resolved.
    */
    get doc(): Node;
    /**
    The ancestor node at the given level. `p.node(p.depth)` is the
    same as `p.parent`.
    */
    node(depth?: number | null): Node;
    /**
    The index into the ancestor at the given level. If this points
    at the 3rd node in the 2nd paragraph on the top level, for
    example, `p.index(0)` is 1 and `p.index(1)` is 2.
    */
    index(depth?: number | null): number;
    /**
    The index pointing after this position into the ancestor at the
    given level.
    */
    indexAfter(depth?: number | null): number;
    /**
    The (absolute) position at the start of the node at the given
    level.
    */
    start(depth?: number | null): number;
    /**
    The (absolute) position at the end of the node at the given
    level.
    */
    end(depth?: number | null): number;
    /**
    The (absolute) position directly before the wrapping node at the
    given level, or, when `depth` is `this.depth + 1`, the original
    position.
    */
    before(depth?: number | null): number;
    /**
    The (absolute) position directly after the wrapping node at the
    given level, or the original position when `depth` is `this.depth + 1`.
    */
    after(depth?: number | null): number;
    /**
    When this position points into a text node, this returns the
    distance between the position and the start of the text node.
    Will be zero for positions that point between nodes.
    */
    get textOffset(): number;
    /**
    Get the node directly after the position, if any. If the position
    points into a text node, only the part of that node after the
    position is returned.
    */
    get nodeAfter(): Node | null;
    /**
    Get the node directly before the position, if any. If the
    position points into a text node, only the part of that node
    before the position is returned.
    */
    get nodeBefore(): Node | null;
    /**
    Get the position at the given index in the parent node at the
    given depth (which defaults to `this.depth`).
    */
    posAtIndex(index: number, depth?: number | null): number;
    /**
    Get the marks at this position, factoring in the surrounding
    marks' [`inclusive`](https://prosemirror.net/docs/ref/#model.MarkSpec.inclusive) property. If the
    position is at the start of a non-empty node, the marks of the
    node after it (if any) are returned.
    */
    marks(): readonly Mark[];
    /**
    Get the marks after the current position, if any, except those
    that are non-inclusive and not present at position `$end`. This
    is mostly useful for getting the set of marks to preserve after a
    deletion. Will return `null` if this position is at the end of
    its parent node or its parent node isn't a textblock (in which
    case no marks should be preserved).
    */
    marksAcross($end: ResolvedPos): readonly Mark[] | null;
    /**
    The depth up to which this position and the given (non-resolved)
    position share the same parent nodes.
    */
    sharedDepth(pos: number): number;
    /**
    Returns a range based on the place where this position and the
    given position diverge around block content. If both point into
    the same textblock, for example, a range around that textblock
    will be returned. If they point into different blocks, the range
    around those blocks in their shared ancestor is returned. You can
    pass in an optional predicate that will be called with a parent
    node to see if a range into that parent is acceptable.
    */
    blockRange(other?: ResolvedPos, pred?: (node: Node) => boolean): NodeRange | null;
    /**
    Query whether the given position shares the same parent node.
    */
    sameParent(other: ResolvedPos): boolean;
    /**
    Return the greater of this and the given position.
    */
    max(other: ResolvedPos): ResolvedPos;
    /**
    Return the smaller of this and the given position.
    */
    min(other: ResolvedPos): ResolvedPos;
}
/**
Represents a flat range of content, i.e. one that starts and
ends in the same node.
*/
declare class NodeRange {
    /**
    A resolved position along the start of the content. May have a
    `depth` greater than this object's `depth` property, since
    these are the positions that were used to compute the range,
    not re-resolved positions directly at its boundaries.
    */
    readonly $from: ResolvedPos;
    /**
    A position along the end of the content. See
    caveat for [`$from`](https://prosemirror.net/docs/ref/#model.NodeRange.$from).
    */
    readonly $to: ResolvedPos;
    /**
    The depth of the node that this range points into.
    */
    readonly depth: number;
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
    $from: ResolvedPos, 
    /**
    A position along the end of the content. See
    caveat for [`$from`](https://prosemirror.net/docs/ref/#model.NodeRange.$from).
    */
    $to: ResolvedPos, 
    /**
    The depth of the node that this range points into.
    */
    depth: number);
    /**
    The position at the start of the range.
    */
    get start(): number;
    /**
    The position at the end of the range.
    */
    get end(): number;
    /**
    The parent node that the range points into.
    */
    get parent(): Node;
    /**
    The start index of the range in the parent node.
    */
    get startIndex(): number;
    /**
    The end index of the range in the parent node.
    */
    get endIndex(): number;
}

/**
Error type raised by [`Node.replace`](https://prosemirror.net/docs/ref/#model.Node.replace) when
given an invalid replacement.
*/
declare class ReplaceError extends Error {
}
/**
A slice represents a piece cut out of a larger document. It
stores not only a fragment, but also the depth up to which nodes on
both side are ‘open’ (cut through).
*/
declare class Slice {
    /**
    The slice's content.
    */
    readonly content: Fragment;
    /**
    The open depth at the start of the fragment.
    */
    readonly openStart: number;
    /**
    The open depth at the end.
    */
    readonly openEnd: number;
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
    content: Fragment, 
    /**
    The open depth at the start of the fragment.
    */
    openStart: number, 
    /**
    The open depth at the end.
    */
    openEnd: number);
    /**
    The size this slice would add when inserted into a document.
    */
    get size(): number;
    /**
    Tests whether this slice is equal to another slice.
    */
    eq(other: Slice): boolean;
    /**
    Convert a slice to a JSON-serializable representation.
    */
    toJSON(): any;
    /**
    Deserialize a slice from its JSON representation.
    */
    static fromJSON(schema: Schema, json: any): Slice;
    /**
    Create a slice from a fragment by taking the maximum possible
    open value on both side of the fragment.
    */
    static maxOpen(fragment: Fragment, openIsolating?: boolean): Slice;
    /**
    The empty slice.
    */
    static empty: Slice;
}

type DOMNode = InstanceType<typeof window.Node>;

/**
These are the options recognized by the
[`parse`](https://prosemirror.net/docs/ref/#model.DOMParser.parse) and
[`parseSlice`](https://prosemirror.net/docs/ref/#model.DOMParser.parseSlice) methods.
*/
interface ParseOptions {
    /**
    By default, whitespace is collapsed as per HTML's rules. Pass
    `true` to preserve whitespace, but normalize newlines to
    spaces or, if available, [line break replacements](https://prosemirror.net/docs/ref/#model.NodeSpec.linebreakReplacement),
    and `"full"` to preserve whitespace entirely.
    */
    preserveWhitespace?: boolean | "full";
    /**
    When given, the parser will, beside parsing the content,
    record the document positions of the given DOM positions. It
    will do so by writing to the objects, adding a `pos` property
    that holds the document position. DOM positions that are not
    in the parsed content will not be written to.
    */
    findPositions?: {
        node: DOMNode;
        offset: number;
        pos?: number;
    }[];
    /**
    The child node index to start parsing from.
    */
    from?: number;
    /**
    The child node index to stop parsing at.
    */
    to?: number;
    /**
    By default, the content is parsed into the schema's default
    [top node type](https://prosemirror.net/docs/ref/#model.Schema.topNodeType). You can pass this
    option to use the type and attributes from a different node
    as the top container.
    */
    topNode?: Node;
    /**
    Provide the starting content match that content parsed into the
    top node is matched against.
    */
    topMatch?: ContentMatch;
    /**
    A set of additional nodes to count as
    [context](https://prosemirror.net/docs/ref/#model.GenericParseRule.context) when parsing, above the
    given [top node](https://prosemirror.net/docs/ref/#model.ParseOptions.topNode).
    */
    context?: ResolvedPos;
}
/**
Fields that may be present in both [tag](https://prosemirror.net/docs/ref/#model.TagParseRule) and
[style](https://prosemirror.net/docs/ref/#model.StyleParseRule) parse rules.
*/
interface GenericParseRule {
    /**
    Can be used to change the order in which the parse rules in a
    schema are tried. Those with higher priority come first. Rules
    without a priority are counted as having priority 50. This
    property is only meaningful in a schema—when directly
    constructing a parser, the order of the rule array is used.
    */
    priority?: number;
    /**
    By default, when a rule matches an element or style, no further
    rules get a chance to match it. By setting this to `false`, you
    indicate that even when this rule matches, other rules that come
    after it should also run.
    */
    consuming?: boolean;
    /**
    When given, restricts this rule to only match when the current
    context—the parent nodes into which the content is being
    parsed—matches this expression. Should contain one or more node
    names or node group names followed by single or double slashes.
    For example `"paragraph/"` means the rule only matches when the
    parent node is a paragraph, `"blockquote/paragraph/"` restricts
    it to be in a paragraph that is inside a blockquote, and
    `"section//"` matches any position inside a section—a double
    slash matches any sequence of ancestor nodes. To allow multiple
    different contexts, they can be separated by a pipe (`|`)
    character, as in `"blockquote/|list_item/"`.
    */
    context?: string;
    /**
    The name of the mark type to wrap the matched content in.
    */
    mark?: string;
    /**
    When true, ignore content that matches this rule. Any `<head>`,
    `<noscript>`, `<script>`, `<object>`, `<style>`, or `<title>`
    tags encountered are automatically ignored when no rules match
    them..
    */
    ignore?: boolean;
    /**
    When true, finding an element that matches this rule will close
    the current node.
    */
    closeParent?: boolean;
    /**
    When true, ignore the node that matches this rule, but do parse
    its content.
    */
    skip?: boolean;
    /**
    Attributes for the node or mark created by this rule. When
    `getAttrs` is provided, it takes precedence.
    */
    attrs?: Attrs;
}
/**
Parse rule targeting a DOM element.
*/
interface TagParseRule extends GenericParseRule {
    /**
    A CSS selector describing the kind of DOM elements to match.
    */
    tag: string;
    /**
    The namespace to match. Nodes are only matched when the
    namespace matches or this property is null.
    */
    namespace?: string;
    /**
    The name of the node type to create when this rule matches. Each
    rule should have either a `node`, `mark`, or `ignore` property
    (except when it appears in a [node](https://prosemirror.net/docs/ref/#model.NodeSpec.parseDOM) or
    [mark spec](https://prosemirror.net/docs/ref/#model.MarkSpec.parseDOM), in which case the `node`
    or `mark` property will be derived from its position).
    */
    node?: string;
    /**
    A function used to compute the attributes for the node or mark
    created by this rule. Can also be used to describe further
    conditions the DOM element or style must match. When it returns
    `false`, the rule won't match. When it returns null or undefined,
    that is interpreted as an empty/default set of attributes.
    */
    getAttrs?: (node: HTMLElement) => Attrs | false | null;
    /**
    For rules that produce non-leaf nodes, by default the content of
    the DOM element is parsed as content of the node. If the child
    nodes are in a descendent node, this may be a CSS selector
    string that the parser must use to find the actual content
    element, or a function that returns the actual content element
    to the parser.
    */
    contentElement?: string | HTMLElement | ((node: HTMLElement) => HTMLElement);
    /**
    Can be used to override the content of a matched node. When
    present, instead of parsing the node's child nodes, the result of
    this function is used.
    */
    getContent?: (node: DOMNode, schema: Schema) => Fragment;
    /**
    Controls whether whitespace should be preserved when parsing the
    content inside the matched element. `false` means whitespace may
    be collapsed, `true` means that whitespace should be preserved
    but newlines normalized to spaces, and `"full"` means that
    newlines should also be preserved.
    */
    preserveWhitespace?: boolean | "full";
}
/**
A parse rule targeting a style property.
*/
interface StyleParseRule extends GenericParseRule {
    /**
    A CSS property name to match. This rule will match inline styles
    that list that property. May also have the form
    `"property=value"`, in which case the rule only matches if the
    property's value exactly matches the given value. (For more
    complicated filters, use [`getAttrs`](https://prosemirror.net/docs/ref/#model.StyleParseRule.getAttrs)
    and return false to indicate that the match failed.) Rules
    matching styles may only produce [marks](https://prosemirror.net/docs/ref/#model.GenericParseRule.mark),
    not nodes.
    */
    style: string;
    /**
    Given to make TS see ParseRule as a tagged union @hide
    */
    tag?: undefined;
    /**
    Style rules can remove marks from the set of active marks.
    */
    clearMark?: (mark: Mark) => boolean;
    /**
    A function used to compute the attributes for the node or mark
    created by this rule. Called with the style's value.
    */
    getAttrs?: (node: string) => Attrs | false | null;
}
/**
A value that describes how to parse a given DOM node or inline
style as a ProseMirror node or mark.
*/
type ParseRule = TagParseRule | StyleParseRule;
/**
A DOM parser represents a strategy for parsing DOM content into a
ProseMirror document conforming to a given schema. Its behavior is
defined by an array of [rules](https://prosemirror.net/docs/ref/#model.ParseRule).
*/
declare class DOMParser {
    /**
    The schema into which the parser parses.
    */
    readonly schema: Schema;
    /**
    The set of [parse rules](https://prosemirror.net/docs/ref/#model.ParseRule) that the parser
    uses, in order of precedence.
    */
    readonly rules: readonly ParseRule[];
    /**
    Create a parser that targets the given schema, using the given
    parsing rules.
    */
    constructor(
    /**
    The schema into which the parser parses.
    */
    schema: Schema, 
    /**
    The set of [parse rules](https://prosemirror.net/docs/ref/#model.ParseRule) that the parser
    uses, in order of precedence.
    */
    rules: readonly ParseRule[]);
    /**
    Parse a document from the content of a DOM node.
    */
    parse(dom: DOMNode, options?: ParseOptions): Node;
    /**
    Parses the content of the given DOM node, like
    [`parse`](https://prosemirror.net/docs/ref/#model.DOMParser.parse), and takes the same set of
    options. But unlike that method, which produces a whole node,
    this one returns a slice that is open at the sides, meaning that
    the schema constraints aren't applied to the start of nodes to
    the left of the input and the end of nodes at the end.
    */
    parseSlice(dom: DOMNode, options?: ParseOptions): Slice;
    /**
    Construct a DOM parser using the parsing rules listed in a
    schema's [node specs](https://prosemirror.net/docs/ref/#model.NodeSpec.parseDOM), reordered by
    [priority](https://prosemirror.net/docs/ref/#model.GenericParseRule.priority).
    */
    static fromSchema(schema: Schema): DOMParser;
}

/**
An object holding the attributes of a node.
*/
type Attrs = {
    readonly [attr: string]: any;
};
/**
Node types are objects allocated once per `Schema` and used to
[tag](https://prosemirror.net/docs/ref/#model.Node.type) `Node` instances. They contain information
about the node type, such as its name and what kind of node it
represents.
*/
declare class NodeType {
    /**
    The name the node type has in this schema.
    */
    readonly name: string;
    /**
    A link back to the `Schema` the node type belongs to.
    */
    readonly schema: Schema;
    /**
    The spec that this type is based on
    */
    readonly spec: NodeSpec;
    /**
    True if this node type has inline content.
    */
    inlineContent: boolean;
    /**
    True if this is a block type
    */
    isBlock: boolean;
    /**
    True if this is the text node type.
    */
    isText: boolean;
    /**
    True if this is an inline type.
    */
    get isInline(): boolean;
    /**
    True if this is a textblock type, a block that contains inline
    content.
    */
    get isTextblock(): boolean;
    /**
    True for node types that allow no content.
    */
    get isLeaf(): boolean;
    /**
    True when this node is an atom, i.e. when it does not have
    directly editable content.
    */
    get isAtom(): boolean;
    /**
    Return true when this node type is part of the given
    [group](https://prosemirror.net/docs/ref/#model.NodeSpec.group).
    */
    isInGroup(group: string): boolean;
    /**
    The starting match of the node type's content expression.
    */
    contentMatch: ContentMatch;
    /**
    The set of marks allowed in this node. `null` means all marks
    are allowed.
    */
    markSet: readonly MarkType[] | null;
    /**
    The node type's [whitespace](https://prosemirror.net/docs/ref/#model.NodeSpec.whitespace) option.
    */
    get whitespace(): "pre" | "normal";
    /**
    Tells you whether this node type has any required attributes.
    */
    hasRequiredAttrs(): boolean;
    /**
    Indicates whether this node allows some of the same content as
    the given node type.
    */
    compatibleContent(other: NodeType): boolean;
    /**
    Create a `Node` of this type. The given attributes are
    checked and defaulted (you can pass `null` to use the type's
    defaults entirely, if no required attributes exist). `content`
    may be a `Fragment`, a node, an array of nodes, or
    `null`. Similarly `marks` may be `null` to default to the empty
    set of marks.
    */
    create(attrs?: Attrs | null, content?: Fragment | Node | readonly Node[] | null, marks?: readonly Mark[]): Node;
    /**
    Like [`create`](https://prosemirror.net/docs/ref/#model.NodeType.create), but check the given content
    against the node type's content restrictions, and throw an error
    if it doesn't match.
    */
    createChecked(attrs?: Attrs | null, content?: Fragment | Node | readonly Node[] | null, marks?: readonly Mark[]): Node;
    /**
    Like [`create`](https://prosemirror.net/docs/ref/#model.NodeType.create), but see if it is
    necessary to add nodes to the start or end of the given fragment
    to make it fit the node. If no fitting wrapping can be found,
    return null. Note that, due to the fact that required nodes can
    always be created, this will always succeed if you pass null or
    `Fragment.empty` as content.
    */
    createAndFill(attrs?: Attrs | null, content?: Fragment | Node | readonly Node[] | null, marks?: readonly Mark[]): Node | null;
    /**
    Returns true if the given fragment is valid content for this node
    type.
    */
    validContent(content: Fragment): boolean;
    /**
    Check whether the given mark type is allowed in this node.
    */
    allowsMarkType(markType: MarkType): boolean;
    /**
    Test whether the given set of marks are allowed in this node.
    */
    allowsMarks(marks: readonly Mark[]): boolean;
    /**
    Removes the marks that are not allowed in this node from the given set.
    */
    allowedMarks(marks: readonly Mark[]): readonly Mark[];
}
/**
Like nodes, marks (which are associated with nodes to signify
things like emphasis or being part of a link) are
[tagged](https://prosemirror.net/docs/ref/#model.Mark.type) with type objects, which are
instantiated once per `Schema`.
*/
declare class MarkType {
    /**
    The name of the mark type.
    */
    readonly name: string;
    /**
    The schema that this mark type instance is part of.
    */
    readonly schema: Schema;
    /**
    The spec on which the type is based.
    */
    readonly spec: MarkSpec;
    /**
    Create a mark of this type. `attrs` may be `null` or an object
    containing only some of the mark's attributes. The others, if
    they have defaults, will be added.
    */
    create(attrs?: Attrs | null): Mark;
    /**
    When there is a mark of this type in the given set, a new set
    without it is returned. Otherwise, the input set is returned.
    */
    removeFromSet(set: readonly Mark[]): readonly Mark[];
    /**
    Tests whether there is a mark of this type in the given set.
    */
    isInSet(set: readonly Mark[]): Mark | undefined;
    /**
    Queries whether a given mark type is
    [excluded](https://prosemirror.net/docs/ref/#model.MarkSpec.excludes) by this one.
    */
    excludes(other: MarkType): boolean;
}
/**
An object describing a schema, as passed to the [`Schema`](https://prosemirror.net/docs/ref/#model.Schema)
constructor.
*/
interface SchemaSpec<Nodes extends string = any, Marks extends string = any> {
    /**
    The node types in this schema. Maps names to
    [`NodeSpec`](https://prosemirror.net/docs/ref/#model.NodeSpec) objects that describe the node type
    associated with that name. Their order is significant—it
    determines which [parse rules](https://prosemirror.net/docs/ref/#model.NodeSpec.parseDOM) take
    precedence by default, and which nodes come first in a given
    [group](https://prosemirror.net/docs/ref/#model.NodeSpec.group).
    */
    nodes: {
        [name in Nodes]: NodeSpec;
    } | OrderedMap<NodeSpec>;
    /**
    The mark types that exist in this schema. The order in which they
    are provided determines the order in which [mark
    sets](https://prosemirror.net/docs/ref/#model.Mark.addToSet) are sorted and in which [parse
    rules](https://prosemirror.net/docs/ref/#model.MarkSpec.parseDOM) are tried.
    */
    marks?: {
        [name in Marks]: MarkSpec;
    } | OrderedMap<MarkSpec>;
    /**
    The name of the default top-level node for the schema. Defaults
    to `"doc"`.
    */
    topNode?: string;
}
/**
A description of a node type, used when defining a schema.
*/
interface NodeSpec {
    /**
    The content expression for this node, as described in the [schema
    guide](https://prosemirror.net/docs/guide/#schema.content_expressions). When not given,
    the node does not allow any content.
    */
    content?: string;
    /**
    The marks that are allowed inside of this node. May be a
    space-separated string referring to mark names or groups, `"_"`
    to explicitly allow all marks, or `""` to disallow marks. When
    not given, nodes with inline content default to allowing all
    marks, other nodes default to not allowing marks.
    */
    marks?: string;
    /**
    The group or space-separated groups to which this node belongs,
    which can be referred to in the content expressions for the
    schema.
    */
    group?: string;
    /**
    Should be set to true for inline nodes. (Implied for text nodes.)
    */
    inline?: boolean;
    /**
    Can be set to true to indicate that, though this isn't a [leaf
    node](https://prosemirror.net/docs/ref/#model.NodeType.isLeaf), it doesn't have directly editable
    content and should be treated as a single unit in the view.
    */
    atom?: boolean;
    /**
    The attributes that nodes of this type get.
    */
    attrs?: {
        [name: string]: AttributeSpec;
    };
    /**
    Controls whether nodes of this type can be selected as a [node
    selection](https://prosemirror.net/docs/ref/#state.NodeSelection). Defaults to true for non-text
    nodes.
    */
    selectable?: boolean;
    /**
    Determines whether nodes of this type can be dragged without
    being selected. Defaults to false.
    */
    draggable?: boolean;
    /**
    Can be used to indicate that this node contains code, which
    causes some commands to behave differently.
    */
    code?: boolean;
    /**
    Controls way whitespace in this a node is parsed. The default is
    `"normal"`, which causes the [DOM parser](https://prosemirror.net/docs/ref/#model.DOMParser) to
    collapse whitespace in normal mode, and normalize it (replacing
    newlines and such with spaces) otherwise. `"pre"` causes the
    parser to preserve spaces inside the node. When this option isn't
    given, but [`code`](https://prosemirror.net/docs/ref/#model.NodeSpec.code) is true, `whitespace`
    will default to `"pre"`. Note that this option doesn't influence
    the way the node is rendered—that should be handled by `toDOM`
    and/or styling.
    */
    whitespace?: "pre" | "normal";
    /**
    Determines whether this node is considered an important parent
    node during replace operations (such as paste). Non-defining (the
    default) nodes get dropped when their entire content is replaced,
    whereas defining nodes persist and wrap the inserted content.
    */
    definingAsContext?: boolean;
    /**
    In inserted content the defining parents of the content are
    preserved when possible. Typically, non-default-paragraph
    textblock types, and possibly list items, are marked as defining.
    */
    definingForContent?: boolean;
    /**
    When enabled, enables both
    [`definingAsContext`](https://prosemirror.net/docs/ref/#model.NodeSpec.definingAsContext) and
    [`definingForContent`](https://prosemirror.net/docs/ref/#model.NodeSpec.definingForContent).
    */
    defining?: boolean;
    /**
    When enabled (default is false), the sides of nodes of this type
    count as boundaries that regular editing operations, like
    backspacing or lifting, won't cross. An example of a node that
    should probably have this enabled is a table cell.
    */
    isolating?: boolean;
    /**
    Defines the default way a node of this type should be serialized
    to DOM/HTML (as used by
    [`DOMSerializer.fromSchema`](https://prosemirror.net/docs/ref/#model.DOMSerializer^fromSchema)).
    Should return a DOM node or an [array
    structure](https://prosemirror.net/docs/ref/#model.DOMOutputSpec) that describes one, with an
    optional number zero (“hole”) in it to indicate where the node's
    content should be inserted.
    
    For text nodes, the default is to create a text DOM node. Though
    it is possible to create a serializer where text is rendered
    differently, this is not supported inside the editor, so you
    shouldn't override that in your text node spec.
    */
    toDOM?: (node: Node) => DOMOutputSpec;
    /**
    Associates DOM parser information with this node, which can be
    used by [`DOMParser.fromSchema`](https://prosemirror.net/docs/ref/#model.DOMParser^fromSchema) to
    automatically derive a parser. The `node` field in the rules is
    implied (the name of this node will be filled in automatically).
    If you supply your own parser, you do not need to also specify
    parsing rules in your schema.
    */
    parseDOM?: readonly TagParseRule[];
    /**
    Defines the default way a node of this type should be serialized
    to a string representation for debugging (e.g. in error messages).
    */
    toDebugString?: (node: Node) => string;
    /**
    Defines the default way a [leaf node](https://prosemirror.net/docs/ref/#model.NodeType.isLeaf) of
    this type should be serialized to a string (as used by
    [`Node.textBetween`](https://prosemirror.net/docs/ref/#model.Node.textBetween) and
    [`Node.textContent`](https://prosemirror.net/docs/ref/#model.Node.textContent)).
    */
    leafText?: (node: Node) => string;
    /**
    A single inline node in a schema can be set to be a linebreak
    equivalent. When converting between block types that support the
    node and block types that don't but have
    [`whitespace`](https://prosemirror.net/docs/ref/#model.NodeSpec.whitespace) set to `"pre"`,
    [`setBlockType`](https://prosemirror.net/docs/ref/#transform.Transform.setBlockType) will convert
    between newline characters to or from linebreak nodes as
    appropriate.
    */
    linebreakReplacement?: boolean;
    /**
    Node specs may include arbitrary properties that can be read by
    other code via [`NodeType.spec`](https://prosemirror.net/docs/ref/#model.NodeType.spec).
    */
    [key: string]: any;
}
/**
Used to define marks when creating a schema.
*/
interface MarkSpec {
    /**
    The attributes that marks of this type get.
    */
    attrs?: {
        [name: string]: AttributeSpec;
    };
    /**
    Whether this mark should be active when the cursor is positioned
    at its end (or at its start when that is also the start of the
    parent node). Defaults to true.
    */
    inclusive?: boolean;
    /**
    Determines which other marks this mark can coexist with. Should
    be a space-separated strings naming other marks or groups of marks.
    When a mark is [added](https://prosemirror.net/docs/ref/#model.Mark.addToSet) to a set, all marks
    that it excludes are removed in the process. If the set contains
    any mark that excludes the new mark but is not, itself, excluded
    by the new mark, the mark can not be added an the set. You can
    use the value `"_"` to indicate that the mark excludes all
    marks in the schema.
    
    Defaults to only being exclusive with marks of the same type. You
    can set it to an empty string (or any string not containing the
    mark's own name) to allow multiple marks of a given type to
    coexist (as long as they have different attributes).
    */
    excludes?: string;
    /**
    The group or space-separated groups to which this mark belongs.
    */
    group?: string;
    /**
    Determines whether marks of this type can span multiple adjacent
    nodes when serialized to DOM/HTML. Defaults to true.
    */
    spanning?: boolean;
    /**
    Marks the content of this span as being code, which causes some
    commands and extensions to treat it differently.
    */
    code?: boolean;
    /**
    Defines the default way marks of this type should be serialized
    to DOM/HTML. When the resulting spec contains a hole, that is
    where the marked content is placed. Otherwise, it is appended to
    the top node.
    */
    toDOM?: (mark: Mark, inline: boolean) => DOMOutputSpec;
    /**
    Associates DOM parser information with this mark (see the
    corresponding [node spec field](https://prosemirror.net/docs/ref/#model.NodeSpec.parseDOM)). The
    `mark` field in the rules is implied.
    */
    parseDOM?: readonly ParseRule[];
    /**
    Mark specs can include additional properties that can be
    inspected through [`MarkType.spec`](https://prosemirror.net/docs/ref/#model.MarkType.spec) when
    working with the mark.
    */
    [key: string]: any;
}
/**
Used to [define](https://prosemirror.net/docs/ref/#model.NodeSpec.attrs) attributes on nodes or
marks.
*/
interface AttributeSpec {
    /**
    The default value for this attribute, to use when no explicit
    value is provided. Attributes that have no default must be
    provided whenever a node or mark of a type that has them is
    created.
    */
    default?: any;
    /**
    A function or type name used to validate values of this
    attribute. This will be used when deserializing the attribute
    from JSON, and when running [`Node.check`](https://prosemirror.net/docs/ref/#model.Node.check).
    When a function, it should raise an exception if the value isn't
    of the expected type or shape. When a string, it should be a
    `|`-separated string of primitive types (`"number"`, `"string"`,
    `"boolean"`, `"null"`, and `"undefined"`), and the library will
    raise an error when the value is not one of those types.
    */
    validate?: string | ((value: any) => void);
}
/**
A document schema. Holds [node](https://prosemirror.net/docs/ref/#model.NodeType) and [mark
type](https://prosemirror.net/docs/ref/#model.MarkType) objects for the nodes and marks that may
occur in conforming documents, and provides functionality for
creating and deserializing such documents.

When given, the type parameters provide the names of the nodes and
marks in this schema.
*/
declare class Schema<Nodes extends string = any, Marks extends string = any> {
    /**
    The [spec](https://prosemirror.net/docs/ref/#model.SchemaSpec) on which the schema is based,
    with the added guarantee that its `nodes` and `marks`
    properties are
    [`OrderedMap`](https://code.haverbeke.berlin/marijn/orderedmap) instances
    (not raw objects).
    */
    spec: {
        nodes: OrderedMap<NodeSpec>;
        marks: OrderedMap<MarkSpec>;
        topNode?: string;
    };
    /**
    An object mapping the schema's node names to node type objects.
    */
    nodes: {
        readonly [name in Nodes]: NodeType;
    } & {
        readonly [key: string]: NodeType;
    };
    /**
    A map from mark names to mark type objects.
    */
    marks: {
        readonly [name in Marks]: MarkType;
    } & {
        readonly [key: string]: MarkType;
    };
    /**
    The [linebreak
    replacement](https://prosemirror.net/docs/ref/#model.NodeSpec.linebreakReplacement) node defined
    in this schema, if any.
    */
    linebreakReplacement: NodeType | null;
    /**
    Construct a schema from a schema [specification](https://prosemirror.net/docs/ref/#model.SchemaSpec).
    */
    constructor(spec: SchemaSpec<Nodes, Marks>);
    /**
    The type of the [default top node](https://prosemirror.net/docs/ref/#model.SchemaSpec.topNode)
    for this schema.
    */
    topNodeType: NodeType;
    /**
    An object for storing whatever values modules may want to
    compute and cache per schema. (If you want to store something
    in it, try to use property names unlikely to clash.)
    */
    cached: {
        [key: string]: any;
    };
    /**
    Create a node in this schema. The `type` may be a string or a
    `NodeType` instance. Attributes will be extended with defaults,
    `content` may be a `Fragment`, `null`, a `Node`, or an array of
    nodes.
    */
    node(type: string | NodeType, attrs?: Attrs | null, content?: Fragment | Node | readonly Node[], marks?: readonly Mark[]): Node;
    /**
    Create a text node in the schema. Empty text nodes are not
    allowed.
    */
    text(text: string, marks?: readonly Mark[] | null): Node;
    /**
    Create a mark with the given type and attributes.
    */
    mark(type: string | MarkType, attrs?: Attrs | null): Mark;
    /**
    Deserialize a node from its JSON representation. This method is
    bound.
    */
    nodeFromJSON: (json: any) => Node;
    /**
    Deserialize a mark from its JSON representation. This method is
    bound.
    */
    markFromJSON: (json: any) => Mark;
}

/**
A fragment represents a node's collection of child nodes.

Like nodes, fragments are persistent data structures, and you
should not mutate them or their content. Rather, you create new
instances whenever needed. The API tries to make this easy.
*/
declare class Fragment {
    /**
    The child nodes in this fragment.
    */
    readonly content: readonly Node[];
    /**
    The size of the fragment, which is the total of the size of
    its content nodes.
    */
    readonly size: number;
    /**
    Invoke a callback for all descendant nodes between the given two
    positions (relative to start of this fragment). Doesn't descend
    into a node when the callback returns `false`.
    */
    nodesBetween(from: number, to: number, f: (node: Node, start: number, parent: Node | null, index: number) => boolean | void, nodeStart?: number, parent?: Node): void;
    /**
    Call the given callback for every descendant node. `pos` will be
    relative to the start of the fragment. The callback may return
    `false` to prevent traversal of a given node's children.
    */
    descendants(f: (node: Node, pos: number, parent: Node | null, index: number) => boolean | void): void;
    /**
    Extract the text between `from` and `to`. See the same method on
    [`Node`](https://prosemirror.net/docs/ref/#model.Node.textBetween).
    */
    textBetween(from: number, to: number, blockSeparator?: string | null, leafText?: string | null | ((leafNode: Node) => string)): string;
    /**
    Create a new fragment containing the combined content of this
    fragment and the other.
    */
    append(other: Fragment): Fragment;
    /**
    Cut out the sub-fragment between the two given positions.
    */
    cut(from: number, to?: number): Fragment;
    /**
    Create a new fragment in which the node at the given index is
    replaced by the given node.
    */
    replaceChild(index: number, node: Node): Fragment;
    /**
    Create a new fragment by prepending the given node to this
    fragment.
    */
    addToStart(node: Node): Fragment;
    /**
    Create a new fragment by appending the given node to this
    fragment.
    */
    addToEnd(node: Node): Fragment;
    /**
    Compare this fragment to another one.
    */
    eq(other: Fragment): boolean;
    /**
    The first child of the fragment, or `null` if it is empty.
    */
    get firstChild(): Node | null;
    /**
    The last child of the fragment, or `null` if it is empty.
    */
    get lastChild(): Node | null;
    /**
    The number of child nodes in this fragment.
    */
    get childCount(): number;
    /**
    Get the child node at the given index. Raise an error when the
    index is out of range.
    */
    child(index: number): Node;
    /**
    Get the child node at the given index, if it exists.
    */
    maybeChild(index: number): Node | null;
    /**
    Call `f` for every child node, passing the node, its offset
    into this parent node, and its index.
    */
    forEach(f: (node: Node, offset: number, index: number) => void): void;
    /**
    Find the first position at which this fragment and another
    fragment differ, or `null` if they are the same.
    */
    findDiffStart(other: Fragment, pos?: number): number | null;
    /**
    Find the first position, searching from the end, at which this
    fragment and the given fragment differ, or `null` if they are
    the same. Since this position will not be the same in both
    nodes, an object with two separate positions is returned.
    */
    findDiffEnd(other: Fragment, pos?: number, otherPos?: number): {
        a: number;
        b: number;
    } | null;
    /**
    Return a debugging string that describes this fragment.
    */
    toString(): string;
    /**
    Create a JSON-serializeable representation of this fragment.
    */
    toJSON(): any;
    /**
    Deserialize a fragment from its JSON representation.
    */
    static fromJSON(schema: Schema, value: any): Fragment;
    /**
    Build a fragment from an array of nodes. Ensures that adjacent
    text nodes with the same marks are joined together.
    */
    static fromArray(array: readonly Node[]): Fragment;
    /**
    Create a fragment from something that can be interpreted as a
    set of nodes. For `null`, it returns the empty fragment. For a
    fragment, the fragment itself. For a node or array of nodes, a
    fragment containing those nodes.
    */
    static from(nodes?: Fragment | Node | readonly Node[] | null): Fragment;
    /**
    An empty fragment. Intended to be reused whenever a node doesn't
    contain anything (rather than allocating a new empty fragment for
    each leaf node).
    */
    static empty: Fragment;
}

type MatchEdge = {
    type: NodeType;
    next: ContentMatch;
};
/**
Instances of this class represent a match state of a node type's
[content expression](https://prosemirror.net/docs/ref/#model.NodeSpec.content), and can be used to
find out whether further content matches here, and whether a given
position is a valid end of the node.
*/
declare class ContentMatch {
    /**
    True when this match state represents a valid end of the node.
    */
    readonly validEnd: boolean;
    /**
    Match a node type, returning a match after that node if
    successful.
    */
    matchType(type: NodeType): ContentMatch | null;
    /**
    Try to match a fragment. Returns the resulting match when
    successful.
    */
    matchFragment(frag: Fragment, start?: number, end?: number): ContentMatch | null;
    /**
    Get the first matching node type at this match position that can
    be generated.
    */
    get defaultType(): NodeType | null;
    /**
    Try to match the given fragment, and if that fails, see if it can
    be made to match by inserting nodes in front of it. When
    successful, return a fragment of inserted nodes (which may be
    empty if nothing had to be inserted). When `toEnd` is true, only
    return a fragment if the resulting match goes to the end of the
    content expression.
    */
    fillBefore(after: Fragment, toEnd?: boolean, startIndex?: number): Fragment | null;
    /**
    Find a set of wrapping node types that would allow a node of the
    given type to appear at this position. The result may be empty
    (when it fits directly) and will be null when no such wrapping
    exists.
    */
    findWrapping(target: NodeType): readonly NodeType[] | null;
    /**
    The number of outgoing edges this node has in the finite
    automaton that describes the content expression.
    */
    get edgeCount(): number;
    /**
    Get the _n_​th outgoing edge from this node in the finite
    automaton that describes the content expression.
    */
    edge(n: number): MatchEdge;
}

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
declare class Node {
    /**
    The type of node that this is.
    */
    readonly type: NodeType;
    /**
    An object mapping attribute names to values. The kind of
    attributes allowed and required are
    [determined](https://prosemirror.net/docs/ref/#model.NodeSpec.attrs) by the node type.
    */
    readonly attrs: Attrs;
    /**
    The marks (things like whether it is emphasized or part of a
    link) applied to this node.
    */
    readonly marks: readonly Mark[];
    /**
    A container holding the node's children.
    */
    readonly content: Fragment;
    /**
    The array of this node's child nodes.
    */
    get children(): readonly Node[];
    /**
    For text nodes, this contains the node's text content.
    */
    readonly text: string | undefined;
    /**
    The size of this node, as defined by the integer-based [indexing
    scheme](https://prosemirror.net/docs/guide/#doc.indexing). For text nodes, this is the
    amount of characters. For other leaf nodes, it is one. For
    non-leaf nodes, it is the size of the content plus two (the
    start and end token).
    */
    get nodeSize(): number;
    /**
    The number of children that the node has.
    */
    get childCount(): number;
    /**
    Get the child node at the given index. Raises an error when the
    index is out of range.
    */
    child(index: number): Node;
    /**
    Get the child node at the given index, if it exists.
    */
    maybeChild(index: number): Node | null;
    /**
    Call `f` for every child node, passing the node, its offset
    into this parent node, and its index.
    */
    forEach(f: (node: Node, offset: number, index: number) => void): void;
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
    nodesBetween(from: number, to: number, f: (node: Node, pos: number, parent: Node | null, index: number) => void | boolean, startPos?: number): void;
    /**
    Call the given callback for every descendant node. Doesn't
    descend into a node when the callback returns `false`.
    */
    descendants(f: (node: Node, pos: number, parent: Node | null, index: number) => void | boolean): void;
    /**
    Concatenates all the text nodes found in this fragment and its
    children.
    */
    get textContent(): string;
    /**
    Get all text between positions `from` and `to`. When
    `blockSeparator` is given, it will be inserted to separate text
    from different block nodes. If `leafText` is given, it'll be
    inserted for every non-text leaf node encountered, otherwise
    [`leafText`](https://prosemirror.net/docs/ref/#model.NodeSpec.leafText) will be used.
    */
    textBetween(from: number, to: number, blockSeparator?: string | null, leafText?: null | string | ((leafNode: Node) => string)): string;
    /**
    Returns this node's first child, or `null` if there are no
    children.
    */
    get firstChild(): Node | null;
    /**
    Returns this node's last child, or `null` if there are no
    children.
    */
    get lastChild(): Node | null;
    /**
    Test whether two nodes represent the same piece of document.
    */
    eq(other: Node): boolean;
    /**
    Compare the markup (type, attributes, and marks) of this node to
    those of another. Returns `true` if both have the same markup.
    */
    sameMarkup(other: Node): boolean;
    /**
    Check whether this node's markup correspond to the given type,
    attributes, and marks.
    */
    hasMarkup(type: NodeType, attrs?: Attrs | null, marks?: readonly Mark[]): boolean;
    /**
    Create a new node with the same markup as this node, containing
    the given content (or empty, if no content is given).
    */
    copy(content?: Fragment | null): Node;
    /**
    Create a copy of this node, with the given set of marks instead
    of the node's own marks.
    */
    mark(marks: readonly Mark[]): Node;
    /**
    Create a copy of this node with only the content between the
    given positions. If `to` is not given, it defaults to the end of
    the node.
    */
    cut(from: number, to?: number): Node;
    /**
    Cut out the part of the document between the given positions, and
    return it as a `Slice` object.
    */
    slice(from: number, to?: number, includeParents?: boolean): Slice;
    /**
    Replace the part of the document between the given positions with
    the given slice. The slice must 'fit', meaning its open sides
    must be able to connect to the surrounding content, and its
    content nodes must be valid children for the node they are placed
    into. If any of this is violated, an error of type
    [`ReplaceError`](https://prosemirror.net/docs/ref/#model.ReplaceError) is thrown.
    */
    replace(from: number, to: number, slice: Slice): Node;
    /**
    Find the node directly after the given position.
    */
    nodeAt(pos: number): Node | null;
    /**
    Find the (direct) child node after the given offset, if any,
    and return it along with its index and offset relative to this
    node.
    */
    childAfter(pos: number): {
        node: Node | null;
        index: number;
        offset: number;
    };
    /**
    Find the (direct) child node before the given offset, if any,
    and return it along with its index and offset relative to this
    node.
    */
    childBefore(pos: number): {
        node: Node | null;
        index: number;
        offset: number;
    };
    /**
    Resolve the given position in the document, returning an
    [object](https://prosemirror.net/docs/ref/#model.ResolvedPos) with information about its context.
    */
    resolve(pos: number): ResolvedPos;
    /**
    Test whether a given mark or mark type occurs in this document
    between the two given positions.
    */
    rangeHasMark(from: number, to: number, type: Mark | MarkType): boolean;
    /**
    True when this is a block (non-inline node)
    */
    get isBlock(): boolean;
    /**
    True when this is a textblock node, a block node with inline
    content.
    */
    get isTextblock(): boolean;
    /**
    True when this node allows inline content.
    */
    get inlineContent(): boolean;
    /**
    True when this is an inline node (a text node or a node that can
    appear among text).
    */
    get isInline(): boolean;
    /**
    True when this is a text node.
    */
    get isText(): boolean;
    /**
    True when this is a leaf node.
    */
    get isLeaf(): boolean;
    /**
    True when this is an atom, i.e. when it does not have directly
    editable content. This is usually the same as `isLeaf`, but can
    be configured with the [`atom` property](https://prosemirror.net/docs/ref/#model.NodeSpec.atom)
    on a node's spec (typically used when the node is displayed as
    an uneditable [node view](https://prosemirror.net/docs/ref/#view.NodeView)).
    */
    get isAtom(): boolean;
    /**
    Return a string representation of this node for debugging
    purposes.
    */
    toString(): string;
    /**
    Get the content match in this node at the given index.
    */
    contentMatchAt(index: number): ContentMatch;
    /**
    Test whether replacing the range between `from` and `to` (by
    child index) with the given replacement fragment (which defaults
    to the empty fragment) would leave the node's content valid. You
    can optionally pass `start` and `end` indices into the
    replacement fragment.
    */
    canReplace(from: number, to: number, replacement?: Fragment, start?: number, end?: number): boolean;
    /**
    Test whether replacing the range `from` to `to` (by index) with
    a node of the given type would leave the node's content valid.
    */
    canReplaceWith(from: number, to: number, type: NodeType, marks?: readonly Mark[]): boolean;
    /**
    Test whether the given node's content could be appended to this
    node. If that node is empty, this will only return true if there
    is at least one node type that can appear in both nodes (to avoid
    merging completely incompatible nodes).
    */
    canAppend(other: Node): boolean;
    /**
    Check whether this node and its descendants conform to the
    schema, and raise an exception when they do not.
    */
    check(): void;
    /**
    Return a JSON-serializeable representation of this node.
    */
    toJSON(): any;
    /**
    Deserialize a node from its JSON representation.
    */
    static fromJSON(schema: Schema, json: any): Node;
}

export { type AttributeSpec, type Attrs, ContentMatch, type DOMOutputSpec, DOMParser, DOMSerializer, Fragment, type GenericParseRule, Mark, type MarkSpec, MarkType, Node, NodeRange, type NodeSpec, NodeType, type ParseOptions, type ParseRule, ReplaceError, ResolvedPos, Schema, type SchemaSpec, Slice, type StyleParseRule, type TagParseRule };
