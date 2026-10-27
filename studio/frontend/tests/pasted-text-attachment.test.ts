// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

// chat-title.ts reaches the providers store, so it must load through the resolver.
registerBundlerResolver();
installLocalStorageFake();
Object.assign((globalThis.window as { location: object }).location, {
  href: "http://localhost/",
});
const { fallbackTitleFromUserText } = await import(
  "../src/features/chat/utils/chat-title.ts"
);
import {
  PASTED_TEXT_DEFAULT_MIN_CHARS,
  PASTED_TEXT_PREVIEW_MAX_CHARS,
  PASTED_TEXT_THRESHOLD_CHOICES,
  PASTED_TEXT_THRESHOLD_OFF,
  attachmentContentSample,
  attachmentContentText,
  attachmentsPastedText,
  attachmentsSample,
  createPastedTextFile,
  isPastedTextContent,
  isPastedTextFile,
  PLAIN_PASTE_GESTURE_MS,
  isPlainPasteChord,
  pasteLongTextAsFile,
  plainPasteStillCounts,
  pastedTextContentBody,
  pastedTextContentBytes,
  pastedTextContentPreview,
  pastedTextFileName,
  pastedTextOf,
  pastedTextPreview,
  shouldAttachPastedText,
  unwrapPastedTextContent,
} from "../src/features/chat/utils/pasted-text.ts";

import { readSrcAsync } from "./helpers/kit.ts";

type ClipboardStub = {
  readonly files: readonly File[];
  readonly items: readonly { kind: string }[];
  readonly types: readonly string[];
  getData: (type: string) => string;
};

function clipboard(
  text: string,
  files: readonly File[] = [],
  extra: { types?: readonly string[]; data?: Record<string, string> } = {},
): ClipboardStub {
  return {
    files,
    items: files.map(() => ({ kind: "file" })),
    types: extra.types ?? ["text/plain"],
    getData: (type) =>
      type === "text/plain" ? text : (extra.data?.[type] ?? ""),
  };
}

function pasteEvent(clipboardData: ClipboardStub | null) {
  let defaultPrevented = false;
  return {
    clipboardData: clipboardData as unknown as DataTransfer | null,
    get defaultPrevented() {
      return defaultPrevented;
    },
    preventDefault: () => {
      defaultPrevented = true;
    },
  };
}

test("short pastes stay inline", () => {
  assert.equal(shouldAttachPastedText(""), false);
  assert.equal(shouldAttachPastedText("   \n  "), false);
  assert.equal(shouldAttachPastedText("what does this function do?"), false);
  assert.equal(
    shouldAttachPastedText("a".repeat(PASTED_TEXT_DEFAULT_MIN_CHARS - 1)),
    false,
  );
  assert.equal(shouldAttachPastedText("line\n".repeat(200)), false);
});

test("bulk pastes become attachments by length", () => {
  assert.equal(
    shouldAttachPastedText("a".repeat(PASTED_TEXT_DEFAULT_MIN_CHARS)),
    true,
  );
});

test("the threshold is configurable", () => {
  const text = "a".repeat(3000);
  assert.equal(shouldAttachPastedText(text, 2000), true);
  assert.equal(shouldAttachPastedText(text, 4000), false);
  assert.equal(shouldAttachPastedText(text, 16000), false);
});

test("the off choice keeps every paste inline", () => {
  const huge = "a".repeat(5 * 1024 * 1024);
  assert.equal(shouldAttachPastedText(huge, PASTED_TEXT_THRESHOLD_OFF), false);
  assert.equal(shouldAttachPastedText(huge, -1), false);
});

test("the default is one of the offered choices", () => {
  assert.ok(
    PASTED_TEXT_THRESHOLD_CHOICES.includes(PASTED_TEXT_DEFAULT_MIN_CHARS),
  );
  assert.equal(PASTED_TEXT_DEFAULT_MIN_CHARS, 4000);
  assert.deepEqual(
    [...PASTED_TEXT_THRESHOLD_CHOICES],
    [0, 2000, 4000, 8000, 16000],
  );
});

test("the file is named after the opening of the paste", () => {
  assert.equal(
    pastedTextFileName("Introducing Unsloth"),
    "Introducing Unsloth.txt",
  );
  assert.equal(
    pastedTextFileName("\n\n  Release notes  \nbody"),
    "Release notes.txt",
  );
  assert.equal(
    pastedTextFileName(
      "Introducing Unsloth Studio, the fastest way to finetune",
    ),
    "Introducing Unsloth Studio, the.txt",
  );
  assert.equal(
    pastedTextFileName(`${"z".repeat(60)} tail`),
    `${"z".repeat(32)}.txt`,
  );
  assert.equal(
    pastedTextFileName("src/lib\\util:\tmain"),
    "src lib util main.txt",
  );
  assert.equal(pastedTextFileName("\n   \n"), "Pasted text.txt");
  assert.equal(pastedTextFileName("///"), "Pasted text.txt");
  assert.equal(
    pastedTextFileName("b".repeat(4 * 1024 * 1024)),
    `${"b".repeat(32)}.txt`,
  );
  assert.equal(
    pastedTextFileName(`${" ".repeat(300)}${"b".repeat(1000)}`),
    "Pasted text.txt",
  );
  const started = process.hrtime.bigint();
  assert.equal(
    pastedTextFileName(`${"\n".repeat(4 * 1024 * 1024)}Deploy log`),
    "Pasted text.txt",
  );
  assert.ok(Number(process.hrtime.bigint() - started) / 1e6 < 250);
});

test("pasted text files are recognised by identity", () => {
  const file = createPastedTextFile("Deploy log\n".repeat(400));
  assert.equal(file.name, "Deploy log.txt");
  assert.equal(file.type, "text/plain");
  assert.equal(isPastedTextFile(file), true);
  assert.equal(
    isPastedTextFile(
      new File(["hi"], "Deploy log.txt", { type: "text/plain" }),
    ),
    false,
  );
  assert.equal(isPastedTextFile(undefined), false);
});

test("the pasted text is kept beside the file, not re-read from it", () => {
  // Draft autosave runs on a keystroke, so it cannot await File.text().
  const text = `Deploy log\n${"line\n".repeat(200)}`;
  assert.equal(pastedTextOf(createPastedTextFile(text)), text);
  assert.equal(
    pastedTextOf(new File(["hi"], "notes.txt", { type: "text/plain" })),
    undefined,
  );
  assert.equal(pastedTextOf(undefined), undefined);
});

test("a file the paste check accepts always has its body, byte for byte", async () => {
  // Both records are written together, so a passing identity check means the body is there.
  for (const text of [
    "x".repeat(4000),
    `${"é中🚀".repeat(500)}\r\n\ttrailing  `,
    `${"a".repeat(16)}\0${"b".repeat(16)}`,
  ]) {
    const file = createPastedTextFile(text);
    assert.equal(isPastedTextFile(file), true);
    const body = pastedTextOf(file);
    assert.equal(body, text);
    assert.equal(body, await file.text());
  }
});

test("the sent wrapper is what marks a paste after the File is gone", () => {
  const pasted = attachmentContentText("Deploy log.txt", "body", true, 4);
  const attached = attachmentContentText("notes.txt", "body", false, 4);

  assert.equal(
    pasted,
    "<pasted_text name=Deploy log.txt bytes=4>\nbody\n</pasted_text>",
  );
  assert.equal(attached, "<attachment name=notes.txt>\nbody\n</attachment>");
  assert.equal(isPastedTextContent(pasted), true);
  assert.equal(isPastedTextContent(attached), false);
  assert.equal(isPastedTextContent(undefined), false);
  assert.equal(isPastedTextContent("pasted_text elsewhere in the body"), false);
});

test("the chip size is read off the header, never measured", () => {
  const pasted = attachmentContentText("Deploy log.txt", "body", true, 12_483);
  assert.equal(pastedTextContentBytes(pasted), 12_483);
  assert.equal(
    pastedTextContentBytes(
      attachmentContentText("x bytes=5.txt", "body", true, 99),
    ),
    99,
  );
  assert.equal(pastedTextContentBytes(undefined), undefined);
  assert.equal(
    pastedTextContentBytes(attachmentContentText("a.txt", "body", true)),
    undefined,
  );
  assert.equal(
    pastedTextContentBytes(attachmentContentText("a.txt", "body", false, 4)),
    undefined,
  );
});

test("previewing sent content never materialises the body", () => {
  const body = "line one\nline two";
  const pasted = attachmentContentText("a.txt", body, true, 17);
  assert.deepEqual(pastedTextContentPreview(pasted), {
    text: body,
    remaining: 0,
  });
  assert.deepEqual(
    pastedTextContentPreview(attachmentContentText("a.txt", body, false)),
    { text: body, remaining: 0 },
  );

  const huge = "z".repeat(PASTED_TEXT_PREVIEW_MAX_CHARS + 40);
  const cut = pastedTextContentPreview(
    attachmentContentText("a.txt", huge, true, huge.length),
  );
  assert.equal(cut.text.length, PASTED_TEXT_PREVIEW_MAX_CHARS);
  assert.equal(cut.remaining, 40);

  assert.deepEqual(pastedTextContentPreview("bare text"), {
    text: "bare text",
    remaining: 0,
  });
  assert.deepEqual(pastedTextContentPreview("<pasted_text name=a.txt>\nbody"), {
    text: "body",
    remaining: 0,
  });
});

test("a paste-only message still has something to name the thread with", () => {
  const paste = `Fix the retry backoff\n${"detail\n".repeat(500)}`;
  const sent = attachmentContentText(
    pastedTextFileName(paste),
    paste,
    true,
    paste.length,
  );

  assert.equal(
    fallbackTitleFromUserText(attachmentsSample([{ content: [] }])),
    "New Chat",
  );
  assert.equal(
    fallbackTitleFromUserText(
      attachmentsSample([{ content: [{ type: "text", text: sent }] }]),
    ),
    "Fix the retry backoff",
  );
  assert.ok(
    attachmentsSample([{ content: [{ type: "text", text: sent }] }]).length <=
      512,
  );
  assert.equal(
    attachmentsSample([
      { content: [{ type: "image" }] },
      { content: [{ type: "text", text: sent }] },
    ]),
    attachmentContentSample(sent),
  );
  assert.equal(attachmentsSample(undefined), "");
});

test("chat search still finds what the paste moved out of the message", () => {
  const body = `Fix the retry backoff\n${"detail\n".repeat(500)}closing line`;
  const sent = attachmentContentText("Fix.txt", body, true, body.length);

  // The whole body, not the opening: a term late in a paste has to stay
  // findable or the paste is searchable only by its first line.
  assert.equal(pastedTextContentBody(sent), body);
  assert.equal(
    attachmentsPastedText([{ content: [{ type: "text", text: sent }] }]),
    body,
  );

  assert.equal(pastedTextContentBody("[PDF: paper.pdf]\nAbstract"), "");
  assert.equal(
    pastedTextContentBody(attachmentContentText("notes.txt", body, false)),
    "",
  );
  assert.equal(
    attachmentsPastedText([
      { content: [{ type: "text", text: "[PDF: paper.pdf]\nAbstract" }] },
      { content: [{ type: "image" }] },
    ]),
    "",
  );
  assert.equal(attachmentsPastedText(undefined), "");

  const second = attachmentContentText("Log.txt", "second body", true, 11);
  assert.equal(
    attachmentsPastedText([
      { content: [{ type: "text", text: sent }] },
      { content: [{ type: "text", text: second }] },
    ]),
    `${body}\n\nsecond body`,
  );
});

test("copies and exports carry the paste, not its wrapper", () => {
  const body = "Deploy log\nline two";
  const sent = attachmentContentText("Deploy log.txt", body, true, 19);

  assert.equal(unwrapPastedTextContent(sent), body);
  assert.ok(!unwrapPastedTextContent(sent).includes("pasted_text"));

  const attached = attachmentContentText("notes.txt", body, false);
  assert.equal(unwrapPastedTextContent(attached), attached);
  assert.equal(unwrapPastedTextContent("bare text"), "bare text");
  assert.equal(unwrapPastedTextContent(""), "");
});

test("a sample of unwrapped content is left as it is", () => {
  assert.equal(
    attachmentContentSample("[PDF: paper.pdf]\nAbstract"),
    "[PDF: paper.pdf]\nAbstract",
  );
  assert.equal(attachmentContentSample("plain", 3), "pla");
});

test("the preview is capped, and says how much it is holding back", () => {
  const short = pastedTextPreview("all of it");
  assert.equal(short.text, "all of it");
  assert.equal(short.remaining, 0);

  const long = pastedTextPreview(
    "a".repeat(PASTED_TEXT_PREVIEW_MAX_CHARS + 25),
  );
  assert.equal(long.text.length, PASTED_TEXT_PREVIEW_MAX_CHARS);
  assert.equal(long.remaining, 25);
});

test("a long text paste is swallowed and handed over as a file", async () => {
  const text = "a".repeat(PASTED_TEXT_DEFAULT_MIN_CHARS);
  const added: File[] = [];
  const event = pasteEvent(clipboard(text));

  assert.equal(
    pasteLongTextAsFile(event, (file) => {
      added.push(file);
    }),
    true,
  );
  assert.equal(event.defaultPrevented, true);
  assert.equal(added.length, 1);
  assert.equal(await added[0]?.text(), text);
});

test("normal pastes and file pastes fall through untouched", () => {
  const shortPaste = pasteEvent(clipboard("hello"));
  assert.equal(
    pasteLongTextAsFile(shortPaste, () => {}),
    false,
  );
  assert.equal(shortPaste.defaultPrevented, false);

  // A clipboard image goes to the file-paste path even with a long text/plain rendering.
  const withFile = pasteEvent(
    clipboard("a".repeat(PASTED_TEXT_DEFAULT_MIN_CHARS), [
      new File([new Uint8Array([1, 2, 3])], "shot.png", { type: "image/png" }),
    ]),
  );
  assert.equal(
    pasteLongTextAsFile(withFile, () => {}),
    false,
  );
  assert.equal(withFile.defaultPrevented, false);

  assert.equal(
    pasteLongTextAsFile(pasteEvent(null), () => {}),
    false,
  );
});

test("a paste too big to hold inline still attaches", () => {
  const huge = "a".repeat(20 * 1024 * 1024 + 1);
  assert.equal(shouldAttachPastedText(huge), true);
  assert.equal(
    shouldAttachPastedText(" ".repeat(PASTED_TEXT_DEFAULT_MIN_CHARS)),
    true,
  );
  assert.equal(shouldAttachPastedText("   "), false);
});

test("an attachment that throws on the spot reports instead of vanishing", () => {
  let errors = 0;
  const event = pasteEvent(
    clipboard("a".repeat(PASTED_TEXT_DEFAULT_MIN_CHARS)),
  );

  const handled = pasteLongTextAsFile(
    event,
    () => {
      throw new Error("no room for another attachment");
    },
    () => {
      errors += 1;
    },
  );

  // The paste is already swallowed, so the toast is the only signal to the user.
  assert.equal(handled, true);
  assert.equal(event.defaultPrevented, true);
  assert.equal(errors, 1);
});

test("a .txt named like a pasted one keeps the normal tile", () => {
  const pasted = createPastedTextFile("Release notes\nbody");
  const lookalike = new File(["hi"], pasted.name, { type: "text/plain" });
  assert.equal(isPastedTextFile(lookalike), false);
});

test("native image and file payloads stay on the file paste path", () => {
  const text = "a".repeat(PASTED_TEXT_DEFAULT_MIN_CHARS);
  // Tauri advertises native payloads by type only, with nothing in files.
  const svg = pasteEvent(
    clipboard(text, [], { types: ["text/plain", "image/svg+xml"] }),
  );
  assert.equal(
    pasteLongTextAsFile(svg, () => {}),
    false,
  );
  assert.equal(svg.defaultPrevented, false);

  const copiedFiles = pasteEvent(
    clipboard(text, [], {
      types: ["text/plain", "text/uri-list"],
      data: { "text/uri-list": "file:///tmp/notes.txt" },
    }),
  );
  assert.equal(
    pasteLongTextAsFile(copiedFiles, () => {}),
    false,
  );

  const richText = pasteEvent(
    clipboard(text, [], { types: ["text/plain", "text/html"] }),
  );
  assert.equal(
    pasteLongTextAsFile(richText, () => {}),
    true,
  );
});

function keyEvent(
  code: string,
  mods: Partial<{
    metaKey: boolean;
    ctrlKey: boolean;
    shiftKey: boolean;
    altKey: boolean;
  }> = {},
) {
  return {
    code,
    metaKey: false,
    ctrlKey: false,
    shiftKey: false,
    altKey: false,
    ...mods,
  };
}

test("paste without formatting is the chord each platform binds", () => {
  // macOS Paste and Match Style is Option+Shift+Cmd+V.
  assert.ok(
    isPlainPasteChord(
      keyEvent("KeyV", { metaKey: true, shiftKey: true, altKey: true }),
      true,
    ),
  );
  // Web apps bind Shift+Cmd+V, so a host mapping it should reach the field.
  assert.ok(
    isPlainPasteChord(
      keyEvent("KeyV", { metaKey: true, shiftKey: true }),
      true,
    ),
  );
  assert.ok(
    isPlainPasteChord(
      keyEvent("KeyV", { ctrlKey: true, shiftKey: true }),
      false,
    ),
  );
  assert.equal(
    isPlainPasteChord(
      keyEvent("KeyV", { ctrlKey: true, shiftKey: true }),
      true,
    ),
    false,
  );
  assert.equal(
    isPlainPasteChord(
      keyEvent("KeyV", { metaKey: true, shiftKey: true }),
      false,
    ),
    false,
  );
});

test("an ordinary paste is left to the attachment threshold", () => {
  assert.equal(
    isPlainPasteChord(keyEvent("KeyV", { metaKey: true }), true),
    false,
  );
  assert.equal(
    isPlainPasteChord(keyEvent("KeyV", { ctrlKey: true, altKey: true }), false),
    false,
  );
  assert.equal(
    isPlainPasteChord(
      keyEvent("KeyV", { ctrlKey: true, shiftKey: true, altKey: true }),
      false,
    ),
    false,
  );
  assert.equal(
    isPlainPasteChord(
      keyEvent("KeyC", { metaKey: true, shiftKey: true }),
      true,
    ),
    false,
  );
  assert.equal(
    isPlainPasteChord(keyEvent("ShiftLeft", { shiftKey: true }), true),
    false,
  );
});

test("a keyboard reporting no code reads the physical key", () => {
  // The Option chord's `key` is the layout's glyph, so only keyCode carries it.
  const optionChord = {
    code: "",
    key: "\u25ca",
    keyCode: 86,
    metaKey: true,
    ctrlKey: false,
    shiftKey: true,
    altKey: true,
  };
  assert.ok(isPlainPasteChord(optionChord, true));
  assert.equal(isPlainPasteChord({ ...optionChord, keyCode: 67 }, true), false);
  assert.equal(
    isPlainPasteChord({ ...optionChord, key: "v", keyCode: 67 }, true),
    false,
  );
});

test("the chord follows the layout, not the board", () => {
  // Dvorak V sits on the QWERTY period key: code "Period", key "V".
  const dvorak = {
    code: "Period",
    key: "V",
    keyCode: 86,
    metaKey: true,
    ctrlKey: false,
    shiftKey: true,
    altKey: false,
  };
  assert.ok(isPlainPasteChord(dvorak, true));
  assert.equal(
    isPlainPasteChord({ ...dvorak, code: "KeyV", key: "K", keyCode: 75 }, true),
    false,
  );
  // Non-Latin layouts: either remaining signal pointing at V suffices.
  assert.ok(
    isPlainPasteChord(
      { ...dvorak, code: "KeyV", key: "\u041c", keyCode: 86 },
      true,
    ),
  );
  assert.ok(
    isPlainPasteChord(
      { ...dvorak, code: "Period", key: "\u041c", keyCode: 86 },
      true,
    ),
  );
  assert.equal(
    isPlainPasteChord(
      { ...dvorak, code: "Period", key: "\u0411", keyCode: 190 },
      true,
    ),
    false,
  );
});

test("the Option chord follows the layout too", () => {
  // macOS routes the menu chord by letter; Option turns `key` into a glyph, so code and keyCode
  // disagree on Dvorak.
  const optionOnDvorak = {
    code: "Period",
    key: "\u25ca",
    keyCode: 86,
    metaKey: true,
    ctrlKey: false,
    shiftKey: true,
    altKey: true,
  };
  assert.ok(isPlainPasteChord(optionOnDvorak, true));
  assert.ok(isPlainPasteChord({ ...optionOnDvorak, code: "KeyV" }, true));
  // Accepting either signal admits Option+Shift+Cmd+K on Dvorak; harmless, no paste follows.
  assert.ok(
    isPlainPasteChord({ ...optionOnDvorak, code: "KeyV", keyCode: 75 }, true),
  );
  assert.equal(
    isPlainPasteChord({ ...optionOnDvorak, code: "Period", keyCode: 75 }, true),
    false,
  );
});

test("a keyboard reporting no code falls back to the key", () => {
  // Shift rewrites `key` on punctuation but not on a letter, so V stays V.
  assert.ok(
    isPlainPasteChord(
      {
        code: "",
        key: "V",
        metaKey: true,
        ctrlKey: false,
        shiftKey: true,
        altKey: false,
      },
      true,
    ),
  );
  assert.equal(
    isPlainPasteChord(
      {
        code: "",
        key: "",
        metaKey: true,
        ctrlKey: false,
        shiftKey: true,
        altKey: false,
      },
      true,
    ),
    false,
  );
});

test("the composer reads the chord from the keydown and clears it", async () => {
  const { readFile } = await import("node:fs/promises");
  const thread = await readSrcAsync("components/assistant-ui/thread.tsx");
  // A paste event has no modifiers; capture the keydown so inputProps keeps its onKeyDown.
  assert.match(thread, /onKeyDownCapture=\{notePlainPasteChord\}/);
  assert.match(
    thread,
    /isPlainPasteChord\(event\)\n\s*\? performance\.now\(\)/,
  );
  // The paste runs as the keydown's default action, before any key is released.
  assert.match(thread, /onKeyUpCapture=\{endPlainPasteChord\}/);
  assert.match(thread, /onBlurCapture=\{endPlainPasteChord\}/);
  assert.match(
    thread,
    /const endPlainPasteChord = useCallback\(\(\) => \{\n\s*plainPasteAtRef\.current = 0;/,
  );
  const at = thread.indexOf("const handleFilePaste = useCallback(");
  const body = thread.slice(at, thread.indexOf("\n  );", at));
  assert.match(body, /plainPasteStillCounts\(\n\s*plainPasteAtRef\.current,/);
  assert.match(body, /plainPasteAtRef\.current = 0;/);
  assert.match(body, /!overlay &&\n\s*!plainPaste &&\n\s*pasteGoesLast/);
});

test("the chord carries a bulk paste past the threshold, inline", () => {
  const text = "a".repeat(PASTED_TEXT_DEFAULT_MIN_CHARS);
  const decide = (key: ReturnType<typeof keyEvent>) => {
    const plainPaste = isPlainPasteChord(key, true);
    const event = pasteEvent(clipboard(text));
    const attached =
      !plainPaste &&
      pasteLongTextAsFile(event, () => {
        /* attach */
      });
    return { attached, defaultPrevented: event.defaultPrevented };
  };

  const ordinary = decide(keyEvent("KeyV", { metaKey: true }));
  assert.equal(ordinary.attached, true);
  assert.equal(ordinary.defaultPrevented, true);

  const plain = decide(
    keyEvent("KeyV", { metaKey: true, shiftKey: true, altKey: true }),
  );
  assert.equal(plain.attached, false);
  assert.equal(
    plain.defaultPrevented,
    false,
    "the browser still performs the paste it was asked for",
  );
});

const valuesForKey = (tree: unknown, key: string): string[] => {
  const found: string[] = [];
  const walk = (node: unknown) => {
    if (typeof node !== "object" || node === null) return;
    for (const [name, value] of Object.entries(
      node as Record<string, unknown>,
    )) {
      if (name === key && typeof value === "string") found.push(value);
      else walk(value);
    }
  };
  walk(tree);
  return found;
};

test("every locale keeps the shortcut in the threshold description", async () => {
  const { readdir } = await import("node:fs/promises");
  const dir = new URL("../src/i18n/locales/", import.meta.url);
  const files = (await readdir(dir)).filter((name) => name.endsWith(".ts"));
  assert.ok(files.length >= 12, "every shipped locale is read");
  for (const name of files) {
    // Import the resolved value: slicing source text broke when Prettier wrapped a locale.
    const module = (await import(new URL(name, dir).href)) as Record<
      string,
      unknown
    >;
    const tree =
      module[name.replace(/\.ts$/, "").replace(/-/g, "")] ??
      Object.values(module)[0];
    const values = valuesForKey(tree, "pastedTextThresholdDescription");
    assert.equal(
      values.length,
      1,
      `${name} carries the description exactly once`,
    );
    assert.ok(values[0].includes("{shortcut}"), `${name} keeps {shortcut}`);
  }
});

test("the settings label names the chord the composer accepts", async () => {
  const { formatBindingLabel } =
    await import("../src/features/settings/lib/keyboard-shortcuts.ts");
  const { readFile } = await import("node:fs/promises");
  for (const mac of [true, false]) {
    const binding = {
      code: "KeyV",
      mod: true,
      ctrl: false,
      shift: true,
      alt: mac,
    };
    assert.ok(
      isPlainPasteChord(
        keyEvent("KeyV", {
          metaKey: mac,
          ctrlKey: !mac,
          shiftKey: true,
          altKey: mac,
        }),
        mac,
      ),
      `the ${mac ? "macOS" : "other"} label describes an accepted chord`,
    );
    assert.equal(
      formatBindingLabel(binding, mac),
      mac ? "\u2325\u21e7\u2318V" : "Ctrl+Shift+V",
    );
  }
  const tab = await readSrcAsync("features/settings/tabs/chat-tab.tsx");
  assert.match(
    tab,
    /code: "KeyV", mod: true, ctrl: false, shift: true, alt: macPlatform/,
  );
});

test("the chord only stands for the paste it asks for", () => {
  assert.ok(plainPasteStillCounts(1000, 1000));
  assert.ok(plainPasteStillCounts(1000, 1000 + PLAIN_PASTE_GESTURE_MS - 1));
  // Shift+Cmd+V may paste nothing on macOS, and a later menu paste has no keydown to clear it.
  assert.equal(
    plainPasteStillCounts(1000, 1000 + PLAIN_PASTE_GESTURE_MS),
    false,
  );
  assert.equal(plainPasteStillCounts(1000, 30000), false);
  assert.equal(plainPasteStillCounts(0, 0), false);
  assert.equal(plainPasteStillCounts(0, 500), false);
});
