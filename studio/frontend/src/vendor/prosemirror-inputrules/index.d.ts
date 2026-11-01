// prosemirror-inputrules 1.5.1 from npm (sha512-7wj4uMjKaXWAQ1CDgxNzNtR9AlsuwzHfdFH1ygEHA2KHF2DOEaXl1CJfNPAKCg9qNEh4rum975QLaCiQPyY6Fw==), MIT: see LICENSE. Imports point at the sibling vendored packages.
import { EditorState, Transaction, Plugin, Command } from '../prosemirror-state/index.js';
import { NodeType, Attrs, Node } from '../prosemirror-model/index.js';

/**
Input rules are regular expressions describing a piece of text
that, when typed, causes something to happen. This might be
changing two dashes into an emdash, wrapping a paragraph starting
with `"> "` into a blockquote, or something entirely different.
*/
declare class InputRule {
    inCode: boolean | "only";
    inCodeMark: boolean | "only";
    /**
    Create an input rule. The rule applies when the user typed
    something and the text directly in front of the cursor matches
    `match`, which should end with `$`.
    
    The `handler` can be a string, in which case the matched text, or
    the first matched group in the regexp, is replaced by that
    string.
    
    Or a it can be a function, which will be called with the match
    array produced by
    [`RegExp.exec`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/RegExp/exec),
    as well as the start and end of the matched range, and which can
    return a [transaction](https://prosemirror.net/docs/ref/#state.Transaction) that describes the
    rule's effect, or null to indicate the input was not handled.
    */
    constructor(
    /**
    @internal
    */
    match: RegExp, handler: string | ((state: EditorState, match: RegExpMatchArray, start: number, end: number) => Transaction | null), options?: {
        /**
        When set to false,
        [`undoInputRule`](https://prosemirror.net/docs/ref/#inputrules.undoInputRule) doesn't work on
        this rule.
        */
        undoable?: boolean;
        /**
        By default, input rules will not apply inside nodes marked
        as [code](https://prosemirror.net/docs/ref/#model.NodeSpec.code). Set this to true to change
        that, or to `"only"` to _only_ match in such nodes.
        */
        inCode?: boolean | "only";
        /**
        When set to `false`, this rule will not fire inside marks
        marked as [code](https://prosemirror.net/docs/ref/#model.MarkSpec.code). The default is
        `true`.
        */
        inCodeMark?: boolean;
    });
}
type PluginState = {
    transform: Transaction;
    from: number;
    to: number;
    text: string;
} | null;
/**
Create an input rules plugin. When enabled, it will cause text
input that matches any of the given rules to trigger the rule's
action.
*/
declare function inputRules({ rules }: {
    rules: readonly InputRule[];
}): Plugin<PluginState>;
/**
This is a command that will undo an input rule, if applying such a
rule was the last thing that the user did.
*/
declare const undoInputRule: Command;

/**
Converts double dashes to an emdash.
*/
declare const emDash: InputRule;
/**
Converts three dots to an ellipsis character.
*/
declare const ellipsis: InputRule;
/**
“Smart” opening double quotes.
*/
declare const openDoubleQuote: InputRule;
/**
“Smart” closing double quotes.
*/
declare const closeDoubleQuote: InputRule;
/**
“Smart” opening single quotes.
*/
declare const openSingleQuote: InputRule;
/**
“Smart” closing single quotes.
*/
declare const closeSingleQuote: InputRule;
/**
Smart-quote related input rules.
*/
declare const smartQuotes: readonly InputRule[];

/**
Build an input rule for automatically wrapping a textblock when a
given string is typed. The `regexp` argument is
directly passed through to the `InputRule` constructor. You'll
probably want the regexp to start with `^`, so that the pattern can
only occur at the start of a textblock.

`nodeType` is the type of node to wrap in. If it needs attributes,
you can either pass them directly, or pass a function that will
compute them from the regular expression match.

By default, if there's a node with the same type above the newly
wrapped node, the rule will try to [join](https://prosemirror.net/docs/ref/#transform.Transform.join) those
two nodes. You can pass a join predicate, which takes a regular
expression match and the node before the wrapped node, and can
return a boolean to indicate whether a join should happen.
*/
declare function wrappingInputRule(regexp: RegExp, nodeType: NodeType, getAttrs?: Attrs | null | ((matches: RegExpMatchArray) => Attrs | null), joinPredicate?: (match: RegExpMatchArray, node: Node) => boolean): InputRule;
/**
Build an input rule that changes the type of a textblock when the
matched text is typed into it. You'll usually want to start your
regexp with `^` to that it is only matched at the start of a
textblock. The optional `getAttrs` parameter can be used to compute
the new node's attributes, and works the same as in the
`wrappingInputRule` function.
*/
declare function textblockTypeInputRule(regexp: RegExp, nodeType: NodeType, getAttrs?: Attrs | null | ((match: RegExpMatchArray) => Attrs | null)): InputRule;

export { InputRule, closeDoubleQuote, closeSingleQuote, ellipsis, emDash, inputRules, openDoubleQuote, openSingleQuote, smartQuotes, textblockTypeInputRule, undoInputRule, wrappingInputRule };
