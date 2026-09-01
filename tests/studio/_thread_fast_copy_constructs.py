# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The HTML constructs the thread's fast copy path is proven against, one per case.

Split out from playwright_thread_fast_copy.py so the table can be read on its own: it is the
list of things a message can contain that a clipboard might treat differently from
``Selection.toString()``, and every claim in
studio/frontend/src/components/assistant-ui/thread-fast-copy.ts about "what the clipboard does"
was measured by copying these one at a time and pasting the result back.

Grouped by what the driver expects of each:

  ANSWERED   everything not named below. The serialiser must produce the clipboard's own string
             byte for byte.
  REFUSED    the form controls. Chromium emits a control's value AND wraps it in block breaks
             whose shape depends on the control, so the fast path hands the copy back.
  NO CLIPBOARD  content the engine copies nothing at all for, so there is no string to compare.

Adding a case here widens the proof for free; the driver iterates this dict.
"""

from __future__ import annotations

# : A 1x1 transparent GIF, left unclosed so a caller appends its own `alt="..."` attribute.
IMG = '<img src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" '

CONSTRUCTS = {
    "plain": "<p>plain line</p>",
    "text_transform_up": '<p style="text-transform:uppercase">transformed heading</p>',
    "text_transform_low": '<p style="text-transform:lowercase">SHOUTED LINE</p>',
    "text_transform_cap": '<p style="text-transform:capitalize">a capitalised sentence</p>',
    "img_alt": f'<p>{IMG}alt="SVG preview"></p>',
    "img_alt_empty": f'<p>{IMG}alt=""></p>',
    "img_no_alt": f"<p>{IMG}></p>",
    "img_alt_inline": f'<p>before{IMG}alt="Tool result 1">after</p>',
    # clipboard: inserting alt text for one of these ADDS text the clipboard never carried.
    # An image the native iterator SKIPS.
    "img_alt_display_none": f'<p>before {IMG}alt="SVG preview" style="display:none"> after</p>',
    "img_alt_hidden": f'<p>before {IMG}alt="SVG preview" style="visibility:hidden"> after</p>',
    "img_alt_unselectable": f'<p>before {IMG}alt="SVG preview" style="user-select:none"> after</p>',
    "img_alt_invisible_cls": (
        "<style>.invisible{visibility:hidden}</style>"
        f'<p>before {IMG}alt="SVG preview" class="invisible"> after</p>'
    ),
    "input_text": '<p><input value="field value"></p>',
    "input_password": '<p><input type="password" value="hunter2"></p>',
    "input_checkbox": '<p><input type="checkbox" checked></p>',
    "textarea": "<p><textarea>textarea body</textarea></p>",
    "select": (
        "<p><select><option>option one</option><option selected>option two</option></select></p>"
    ),
    "user_select_none": '<p style="user-select:none;-webkit-user-select:none">unselectable</p>',
    "display_none": '<p style="display:none">hidden line</p>',
    "visibility_hidden": '<p style="visibility:hidden">invisible line</p>',
    "generated": '<style>.g::before{content:"GEN "}</style><p class="g">generated host</p>',
    "white_space_pre": '<p style="white-space:pre">  pre   spacing  </p>',
    "br": "<p>line with a<br>break</p>",
    "list": "<ul><li>item one</li><li>item two</li></ul>",
    "table": "<table><tr><td>cell a</td><td>cell b</td></tr></table>",
    "link": '<p>a <a href="https://example.com">link</a> inline</p>',
    "nbsp": "<p>&nbsp;nbsp&nbsp;line</p>",
    "pre_code": '<pre><code>fn main() {\n    println!("hi");\n}</code></pre>',
    "emoji": "<p>emoji \U0001f600 accents éè</p>",
    "blockquote": "<blockquote>quoted</blockquote>",
    "collapse_ws": "<p>trailing   whitespace   collapse</p>",
}

# : The only constructs the serialiser is allowed to refuse on a mapped engine.
MUST_REFUSE = frozenset({"input_text", "input_password", "input_checkbox", "textarea", "select"})

# : Constructs where the engine copies nothing at all, so there is no clipboard to compare with.
NO_COPY = frozenset({"user_select_none", "display_none", "visibility_hidden"})

# : A SELECTION THAT LIES ENTIRELY INSIDE THE TRANSFORMED ELEMENT.
# Its common ancestor is the text : node, so the scope is the transformed element ITSELF, which `querySelectorAll("*")`
# does not : include.
INSIDE_SCOPE = {
    "inside a transformed span": (
        '<p><span style="text-transform:uppercase" id="t">transformed heading</span></p>',
        "document.getElementById('t')",
    ),
    "inside a transformed ancestor": (
        '<div style="text-transform:lowercase"><p><span id="t">SHOUTED LINE</span></p></div>',
        "document.getElementById('t')",
    ),
    "text node inside a transformed div": (
        '<div style="text-transform:capitalize" id="t">a capitalised sentence</div>',
        "document.getElementById('t')",
    ),
}
