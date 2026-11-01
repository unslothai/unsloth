// prosemirror-state 1.4.4 from npm (sha512-6jiYHH2CIGbCfnxdHbXZ12gySFY/fz/ulZE333G6bPqIZ4F+TXo9ifiR86nAHpWnfoNjOb3o5ESi7J8Uz1jXHw==), MIT: see LICENSE. Imports point at the sibling vendored packages.
import { Schema, Node, Mark, MarkType, Slice, ResolvedPos } from '../prosemirror-model/index.js';
import { Transform, Mappable } from '../prosemirror-transform/index.js';
import { EditorProps, EditorView } from '../prosemirror-view/index.js';

/**
The type of object passed to
[`EditorState.create`](https://prosemirror.net/docs/ref/#state.EditorState^create).
*/
interface EditorStateConfig {
    /**
    The schema to use (only relevant if no `doc` is specified).
    */
    schema?: Schema;
    /**
    The starting document. Either this or `schema` _must_ be
    provided.
    */
    doc?: Node;
    /**
    A valid selection in the document.
    */
    selection?: Selection;
    /**
    The initial set of [stored marks](https://prosemirror.net/docs/ref/#state.EditorState.storedMarks).
    */
    storedMarks?: readonly Mark[] | null;
    /**
    The plugins that should be active in this state.
    */
    plugins?: readonly Plugin[];
}
/**
The state of a ProseMirror editor is represented by an object of
this type. A state is a persistent data structure—it isn't
updated, but rather a new state value is computed from an old one
using the [`apply`](https://prosemirror.net/docs/ref/#state.EditorState.apply) method.

A state holds a number of built-in fields, and plugins can
[define](https://prosemirror.net/docs/ref/#state.PluginSpec.state) additional fields.
*/
declare class EditorState {
    /**
    The current document.
    */
    doc: Node;
    /**
    The selection.
    */
    selection: Selection;
    /**
    A set of marks to apply to the next input. Will be null when
    no explicit marks have been set.
    */
    storedMarks: readonly Mark[] | null;
    /**
    The schema of the state's document.
    */
    get schema(): Schema;
    /**
    The plugins that are active in this state.
    */
    get plugins(): readonly Plugin[];
    /**
    Apply the given transaction to produce a new state.
    */
    apply(tr: Transaction): EditorState;
    /**
    Verbose variant of [`apply`](https://prosemirror.net/docs/ref/#state.EditorState.apply) that
    returns the precise transactions that were applied (which might
    be influenced by the [transaction
    hooks](https://prosemirror.net/docs/ref/#state.PluginSpec.filterTransaction) of
    plugins) along with the new state.
    */
    applyTransaction(rootTr: Transaction): {
        state: EditorState;
        transactions: readonly Transaction[];
    };
    /**
    Accessor that constructs and returns a new [transaction](https://prosemirror.net/docs/ref/#state.Transaction) from this state.
    */
    get tr(): Transaction;
    /**
    Create a new state.
    */
    static create(config: EditorStateConfig): EditorState;
    /**
    Create a new state based on this one, but with an adjusted set
    of active plugins. State fields that exist in both sets of
    plugins are kept unchanged. Those that no longer exist are
    dropped, and those that are new are initialized using their
    [`init`](https://prosemirror.net/docs/ref/#state.StateField.init) method, passing in the new
    configuration object..
    */
    reconfigure(config: {
        /**
        New set of active plugins.
        */
        plugins?: readonly Plugin[];
    }): EditorState;
    /**
    Serialize this state to JSON. If you want to serialize the state
    of plugins, pass an object mapping property names to use in the
    resulting JSON object to plugin objects. The argument may also be
    a string or number, in which case it is ignored, to support the
    way `JSON.stringify` calls `toString` methods.
    */
    toJSON(pluginFields?: {
        [propName: string]: Plugin;
    }): any;
    /**
    Deserialize a JSON representation of a state. `config` should
    have at least a `schema` field, and should contain array of
    plugins to initialize the state with. `pluginFields` can be used
    to deserialize the state of plugins, by associating plugin
    instances with the property names they use in the JSON object.
    */
    static fromJSON(config: {
        /**
        The schema to use.
        */
        schema: Schema;
        /**
        The set of active plugins.
        */
        plugins?: readonly Plugin[];
    }, json: any, pluginFields?: {
        [propName: string]: Plugin;
    }): EditorState;
}

/**
This is the type passed to the [`Plugin`](https://prosemirror.net/docs/ref/#state.Plugin)
constructor. It provides a definition for a plugin.
*/
interface PluginSpec<PluginState> {
    /**
    The [view props](https://prosemirror.net/docs/ref/#view.EditorProps) added by this plugin. Props
    that are functions will be bound to have the plugin instance as
    their `this` binding.
    */
    props?: EditorProps<Plugin<PluginState>>;
    /**
    Allows a plugin to define a [state field](https://prosemirror.net/docs/ref/#state.StateField), an
    extra slot in the state object in which it can keep its own data.
    */
    state?: StateField<PluginState>;
    /**
    Can be used to make this a keyed plugin. You can have only one
    plugin with a given key in a given state, but it is possible to
    access the plugin's configuration and state through the key,
    without having access to the plugin instance object.
    */
    key?: PluginKey;
    /**
    When the plugin needs to interact with the editor view, or
    set something up in the DOM, use this field. The function
    will be called when the plugin's state is associated with an
    editor view.
    */
    view?: (view: EditorView) => PluginView;
    /**
    When present, this will be called before a transaction is
    applied by the state, allowing the plugin to cancel it (by
    returning false).
    */
    filterTransaction?: (tr: Transaction, state: EditorState) => boolean;
    /**
    Allows the plugin to append another transaction to be applied
    after the given array of transactions. When another plugin
    appends a transaction after this was called, it is called again
    with the new state and new transactions—but only the new
    transactions, i.e. it won't be passed transactions that it
    already saw.
    */
    appendTransaction?: (transactions: readonly Transaction[], oldState: EditorState, newState: EditorState) => Transaction | null | undefined;
    /**
    Additional properties are allowed on plugin specs, which can be
    read via [`Plugin.spec`](https://prosemirror.net/docs/ref/#state.Plugin.spec).
    */
    [key: string]: any;
}
/**
A stateful object that can be installed in an editor by a
[plugin](https://prosemirror.net/docs/ref/#state.PluginSpec.view).
*/
type PluginView = {
    /**
    Called whenever the view's state is updated.
    */
    update?: (view: EditorView, prevState: EditorState) => void;
    /**
    Called when the view is destroyed or receives a state
    with different plugins.
    */
    destroy?: () => void;
};
/**
Plugins bundle functionality that can be added to an editor.
They are part of the [editor state](https://prosemirror.net/docs/ref/#state.EditorState) and
may influence that state and the view that contains it.
*/
declare class Plugin<PluginState = any> {
    /**
    The plugin's [spec object](https://prosemirror.net/docs/ref/#state.PluginSpec).
    */
    readonly spec: PluginSpec<PluginState>;
    /**
    Create a plugin.
    */
    constructor(
    /**
    The plugin's [spec object](https://prosemirror.net/docs/ref/#state.PluginSpec).
    */
    spec: PluginSpec<PluginState>);
    /**
    The [props](https://prosemirror.net/docs/ref/#view.EditorProps) exported by this plugin.
    */
    readonly props: EditorProps<Plugin<PluginState>>;
    /**
    Extract the plugin's state field from an editor state.
    */
    getState(state: EditorState): PluginState | undefined;
}
/**
A plugin spec may provide a state field (under its
[`state`](https://prosemirror.net/docs/ref/#state.PluginSpec.state) property) of this type, which
describes the state it wants to keep. Functions provided here are
always called with the plugin instance as their `this` binding.
*/
interface StateField<T> {
    /**
    Initialize the value of the field. `config` will be the object
    passed to [`EditorState.create`](https://prosemirror.net/docs/ref/#state.EditorState^create). Note
    that `instance` is a half-initialized state instance, and will
    not have values for plugin fields initialized after this one.
    */
    init: (config: EditorStateConfig, instance: EditorState) => T;
    /**
    Apply the given transaction to this state field, producing a new
    field value. Note that the `newState` argument is again a partially
    constructed state does not yet contain the state from plugins
    coming after this one.
    */
    apply: (tr: Transaction, value: T, oldState: EditorState, newState: EditorState) => T;
    /**
    Convert this field to JSON. Optional, can be left off to disable
    JSON serialization for the field.
    */
    toJSON?: (value: T) => any;
    /**
    Deserialize the JSON representation of this field. Note that the
    `state` argument is again a half-initialized state.
    */
    fromJSON?: (config: EditorStateConfig, value: any, state: EditorState) => T;
}
/**
A key is used to [tag](https://prosemirror.net/docs/ref/#state.PluginSpec.key) plugins in a way
that makes it possible to find them, given an editor state.
Assigning a key does mean only one plugin of that type can be
active in a state.
*/
declare class PluginKey<PluginState = any> {
    /**
    Create a plugin key.
    */
    constructor(name?: string);
    /**
    Get the active plugin with this key, if any, from an editor
    state.
    */
    get(state: EditorState): Plugin<PluginState> | undefined;
    /**
    Get the plugin's state from an editor state.
    */
    getState(state: EditorState): PluginState | undefined;
}

/**
Commands are functions that take a state and a an optional
transaction dispatch function and...

 - determine whether they apply to this state
 - if not, return false
 - if `dispatch` was passed, perform their effect, possibly by
   passing a transaction to `dispatch`
 - return true

In some cases, the editor view is passed as a third argument.
*/
type Command = (state: EditorState, dispatch?: (tr: Transaction) => void, view?: EditorView) => boolean;
/**
An editor state transaction, which can be applied to a state to
create an updated state. Use
[`EditorState.tr`](https://prosemirror.net/docs/ref/#state.EditorState.tr) to create an instance.

Transactions track changes to the document (they are a subclass of
[`Transform`](https://prosemirror.net/docs/ref/#transform.Transform)), but also other state changes,
like selection updates and adjustments of the set of [stored
marks](https://prosemirror.net/docs/ref/#state.EditorState.storedMarks). In addition, you can store
metadata properties in a transaction, which are extra pieces of
information that client code or plugins can use to describe what a
transaction represents, so that they can update their [own
state](https://prosemirror.net/docs/ref/#state.StateField) accordingly.

The [editor view](https://prosemirror.net/docs/ref/#view.EditorView) uses a few metadata
properties: it will attach a property `"pointer"` with the value
`true` to selection transactions directly caused by mouse or touch
input, a `"composition"` property holding an ID identifying the
composition that caused it to transactions caused by composed DOM
input, and a `"uiEvent"` property of that may be `"paste"`,
`"cut"`, or `"drop"`.
*/
declare class Transaction extends Transform {
    /**
    The timestamp associated with this transaction, in the same
    format as `Date.now()`.
    */
    time: number;
    private curSelection;
    private curSelectionFor;
    private updated;
    private meta;
    /**
    The stored marks set by this transaction, if any.
    */
    storedMarks: readonly Mark[] | null;
    /**
    The transaction's current selection. This defaults to the editor
    selection [mapped](https://prosemirror.net/docs/ref/#state.Selection.map) through the steps in the
    transaction, but can be overwritten with
    [`setSelection`](https://prosemirror.net/docs/ref/#state.Transaction.setSelection).
    */
    get selection(): Selection;
    /**
    Update the transaction's current selection. Will determine the
    selection that the editor gets when the transaction is applied.
    */
    setSelection(selection: Selection): this;
    /**
    Whether the selection was explicitly updated by this transaction.
    */
    get selectionSet(): boolean;
    /**
    Set the current stored marks.
    */
    setStoredMarks(marks: readonly Mark[] | null): this;
    /**
    Make sure the current stored marks or, if that is null, the marks
    at the selection, match the given set of marks. Does nothing if
    this is already the case.
    */
    ensureMarks(marks: readonly Mark[]): this;
    /**
    Add a mark to the set of stored marks.
    */
    addStoredMark(mark: Mark): this;
    /**
    Remove a mark or mark type from the set of stored marks.
    */
    removeStoredMark(mark: Mark | MarkType): this;
    /**
    Whether the stored marks were explicitly set for this transaction.
    */
    get storedMarksSet(): boolean;
    /**
    Update the timestamp for the transaction.
    */
    setTime(time: number): this;
    /**
    Replace the current selection with the given slice.
    */
    replaceSelection(slice: Slice): this;
    /**
    Replace the selection with the given node. When `inheritMarks` is
    true and the content is inline, it inherits the marks from the
    place where it is inserted.
    */
    replaceSelectionWith(node: Node, inheritMarks?: boolean): this;
    /**
    Delete the selection.
    */
    deleteSelection(): this;
    /**
    Replace the given range, or the selection if no range is given,
    with a text node containing the given string.
    */
    insertText(text: string, from?: number, to?: number): this;
    /**
    Store a metadata property in this transaction, keyed either by
    name or by plugin.
    */
    setMeta(key: string | Plugin | PluginKey, value: any): this;
    /**
    Retrieve a metadata property for a given name or plugin.
    */
    getMeta(key: string | Plugin | PluginKey): any;
    /**
    Returns true if this transaction doesn't contain any metadata,
    and can thus safely be extended.
    */
    get isGeneric(): boolean;
    /**
    Indicate that the editor should scroll the selection into view
    when updated to the state produced by this transaction.
    */
    scrollIntoView(): this;
    /**
    True when this transaction has had `scrollIntoView` called on it.
    */
    get scrolledIntoView(): boolean;
}

/**
Superclass for editor selections. Every selection type should
extend this. Should not be instantiated directly.
*/
declare abstract class Selection {
    /**
    The resolved anchor of the selection (the side that stays in
    place when the selection is modified).
    */
    readonly $anchor: ResolvedPos;
    /**
    The resolved head of the selection (the side that moves when
    the selection is modified).
    */
    readonly $head: ResolvedPos;
    /**
    Initialize a selection with the head and anchor and ranges. If no
    ranges are given, constructs a single range across `$anchor` and
    `$head`.
    */
    constructor(
    /**
    The resolved anchor of the selection (the side that stays in
    place when the selection is modified).
    */
    $anchor: ResolvedPos, 
    /**
    The resolved head of the selection (the side that moves when
    the selection is modified).
    */
    $head: ResolvedPos, ranges?: readonly SelectionRange[]);
    /**
    The ranges covered by the selection.
    */
    ranges: readonly SelectionRange[];
    /**
    The selection's anchor, as an unresolved position.
    */
    get anchor(): number;
    /**
    The selection's head.
    */
    get head(): number;
    /**
    The lower bound of the selection's main range.
    */
    get from(): number;
    /**
    The upper bound of the selection's main range.
    */
    get to(): number;
    /**
    The resolved lower  bound of the selection's main range.
    */
    get $from(): ResolvedPos;
    /**
    The resolved upper bound of the selection's main range.
    */
    get $to(): ResolvedPos;
    /**
    Indicates whether the selection contains any content.
    */
    get empty(): boolean;
    /**
    Test whether the selection is the same as another selection.
    */
    abstract eq(selection: Selection): boolean;
    /**
    Map this selection through a [mappable](https://prosemirror.net/docs/ref/#transform.Mappable)
    thing. `doc` should be the new document to which we are mapping.
    */
    abstract map(doc: Node, mapping: Mappable): Selection;
    /**
    Get the content of this selection as a slice.
    */
    content(): Slice;
    /**
    Replace the selection with a slice or, if no slice is given,
    delete the selection. Will append to the given transaction.
    */
    replace(tr: Transaction, content?: Slice): void;
    /**
    Replace the selection with the given node, appending the changes
    to the given transaction.
    */
    replaceWith(tr: Transaction, node: Node): void;
    /**
    Convert the selection to a JSON representation. When implementing
    this for a custom selection class, make sure to give the object a
    `type` property whose value matches the ID under which you
    [registered](https://prosemirror.net/docs/ref/#state.Selection^jsonID) your class.
    */
    abstract toJSON(): any;
    /**
    Find a valid cursor or leaf node selection starting at the given
    position and searching back if `dir` is negative, and forward if
    positive. When `textOnly` is true, only consider cursor
    selections. Will return null when no valid selection position is
    found.
    */
    static findFrom($pos: ResolvedPos, dir: number, textOnly?: boolean): Selection | null;
    /**
    Find a valid cursor or leaf node selection near the given
    position. Searches forward first by default, but if `bias` is
    negative, it will search backwards first.
    */
    static near($pos: ResolvedPos, bias?: number): Selection;
    /**
    Find the cursor or leaf node selection closest to the start of
    the given document. Will return an
    [`AllSelection`](https://prosemirror.net/docs/ref/#state.AllSelection) if no valid position
    exists.
    */
    static atStart(doc: Node): Selection;
    /**
    Find the cursor or leaf node selection closest to the end of the
    given document.
    */
    static atEnd(doc: Node): Selection;
    /**
    Deserialize the JSON representation of a selection. Must be
    implemented for custom classes (as a static class method).
    */
    static fromJSON(doc: Node, json: any): Selection;
    /**
    To be able to deserialize selections from JSON, custom selection
    classes must register themselves with an ID string, so that they
    can be disambiguated. Try to pick something that's unlikely to
    clash with classes from other modules.
    */
    static jsonID(id: string, selectionClass: {
        fromJSON: (doc: Node, json: any) => Selection;
    }): {
        fromJSON: (doc: Node, json: any) => Selection;
    };
    /**
    Get a [bookmark](https://prosemirror.net/docs/ref/#state.SelectionBookmark) for this selection,
    which is a value that can be mapped without having access to a
    current document, and later resolved to a real selection for a
    given document again. (This is used mostly by the history to
    track and restore old selections.) The default implementation of
    this method just converts the selection to a text selection and
    returns the bookmark for that.
    */
    getBookmark(): SelectionBookmark;
    /**
    Controls whether, when a selection of this type is active in the
    browser, the selected range should be visible to the user.
    Defaults to `true`.
    */
    visible: boolean;
}
/**
A lightweight, document-independent representation of a selection.
You can define a custom bookmark type for a custom selection class
to make the history handle it well.
*/
interface SelectionBookmark {
    /**
    Map the bookmark through a set of changes.
    */
    map: (mapping: Mappable) => SelectionBookmark;
    /**
    Resolve the bookmark to a real selection again. This may need to
    do some error checking and may fall back to a default (usually
    [`TextSelection.between`](https://prosemirror.net/docs/ref/#state.TextSelection^between)) if
    mapping made the bookmark invalid.
    */
    resolve: (doc: Node) => Selection;
}
/**
Represents a selected range in a document.
*/
declare class SelectionRange {
    /**
    The lower bound of the range.
    */
    readonly $from: ResolvedPos;
    /**
    The upper bound of the range.
    */
    readonly $to: ResolvedPos;
    /**
    Create a range.
    */
    constructor(
    /**
    The lower bound of the range.
    */
    $from: ResolvedPos, 
    /**
    The upper bound of the range.
    */
    $to: ResolvedPos);
}
/**
A text selection represents a classical editor selection, with a
head (the moving side) and anchor (immobile side), both of which
point into textblock nodes. It can be empty (a regular cursor
position).
*/
declare class TextSelection extends Selection {
    /**
    Construct a text selection between the given points.
    */
    constructor($anchor: ResolvedPos, $head?: ResolvedPos);
    /**
    Returns a resolved position if this is a cursor selection (an
    empty text selection), and null otherwise.
    */
    get $cursor(): ResolvedPos | null;
    map(doc: Node, mapping: Mappable): Selection;
    replace(tr: Transaction, content?: Slice): void;
    eq(other: Selection): boolean;
    getBookmark(): TextBookmark;
    toJSON(): any;
    /**
    Create a text selection from non-resolved positions.
    */
    static create(doc: Node, anchor: number, head?: number): TextSelection;
    /**
    Return a text selection that spans the given positions or, if
    they aren't text positions, find a text selection near them.
    `bias` determines whether the method searches forward (default)
    or backwards (negative number) first. Will fall back to calling
    [`Selection.near`](https://prosemirror.net/docs/ref/#state.Selection^near) when the document
    doesn't contain a valid text position.
    */
    static between($anchor: ResolvedPos, $head: ResolvedPos, bias?: number): Selection;
}
declare class TextBookmark {
    readonly anchor: number;
    readonly head: number;
    constructor(anchor: number, head: number);
    map(mapping: Mappable): TextBookmark;
    resolve(doc: Node): Selection;
}
/**
A node selection is a selection that points at a single node. All
nodes marked [selectable](https://prosemirror.net/docs/ref/#model.NodeSpec.selectable) can be the
target of a node selection. In such a selection, `from` and `to`
point directly before and after the selected node, `anchor` equals
`from`, and `head` equals `to`..
*/
declare class NodeSelection extends Selection {
    /**
    Create a node selection. Does not verify the validity of its
    argument.
    */
    constructor($pos: ResolvedPos);
    /**
    The selected node.
    */
    node: Node;
    map(doc: Node, mapping: Mappable): Selection;
    content(): Slice;
    eq(other: Selection): boolean;
    toJSON(): any;
    getBookmark(): NodeBookmark;
    /**
    Create a node selection from non-resolved positions.
    */
    static create(doc: Node, from: number): NodeSelection;
    /**
    Determines whether the given node may be selected as a node
    selection.
    */
    static isSelectable(node: Node): boolean;
}
declare class NodeBookmark {
    readonly anchor: number;
    constructor(anchor: number);
    map(mapping: Mappable): TextBookmark | NodeBookmark;
    resolve(doc: Node): Selection | NodeSelection;
}
/**
A selection type that represents selecting the whole document
(which can not necessarily be expressed with a text selection, when
there are for example leaf block nodes at the start or end of the
document).
*/
declare class AllSelection extends Selection {
    /**
    Create an all-selection over the given document.
    */
    constructor(doc: Node);
    replace(tr: Transaction, content?: Slice): void;
    toJSON(): any;
    map(doc: Node): AllSelection;
    eq(other: Selection): boolean;
    getBookmark(): {
        map(): any;
        resolve(doc: Node): AllSelection;
    };
}

export { AllSelection, type Command, EditorState, type EditorStateConfig, NodeSelection, Plugin, PluginKey, type PluginSpec, type PluginView, Selection, type SelectionBookmark, SelectionRange, type StateField, TextSelection, Transaction };
