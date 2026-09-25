import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import {
  composerSubmitIntent,
  composerFollowUpBehavior,
  composerShortcutLabels,
  effectiveSendShortcut,
  followUpSubmitIntent,
  normalizeComposerPreferences,
  type ComposerKeyEvent,
} from "../src/features/chat/utils/composer-preferences.ts";
import { installLocalStorageFake } from "./helpers/kit.ts";

const enter: ComposerKeyEvent = {
  key: "Enter",
  metaKey: false,
  ctrlKey: false,
  shiftKey: false,
  altKey: false,
};
for (const mod of ["metaKey", "ctrlKey"] as const) {
  test(`${mod}: configured send, opposite follow-up, and newline chords`, () => {
    assert.equal(composerSubmitIntent(enter, "enter"), "default");
    assert.equal(
      composerSubmitIntent({ ...enter, [mod]: true }, "enter"),
      "opposite",
    );
    assert.equal(
      composerSubmitIntent({ ...enter, shiftKey: true }, "enter"),
      null,
    );
    assert.equal(
      composerSubmitIntent({ ...enter, [mod]: true, shiftKey: true }, "enter"),
      null,
    );
    assert.equal(composerSubmitIntent(enter, "mod-enter"), null);
    assert.equal(
      composerSubmitIntent({ ...enter, shiftKey: true }, "mod-enter"),
      null,
    );
    assert.equal(
      composerSubmitIntent({ ...enter, [mod]: true }, "mod-enter"),
      "default",
    );
    assert.equal(
      composerSubmitIntent(
        { ...enter, [mod]: true, shiftKey: true },
        "mod-enter",
      ),
      "opposite",
    );
  });
}
for (const blocked of [
  { key: "a" },
  { altKey: true },
  { isComposing: true },
  { keyCode: 229 },
  { repeat: true },
  { metaKey: true, ctrlKey: true },
]) {
  test(`do not submit ${JSON.stringify(blocked)}`, () => {
    for (const shortcut of ["enter", "mod-enter-multiline", "mod-enter"] as const) {
      assert.equal(
        composerSubmitIntent({ ...enter, ...blocked }, shortcut),
        null,
      );
      assert.equal(
        composerSubmitIntent({ ...enter, ctrlKey: true, ...blocked }, shortcut),
        null,
      );
    }
  });
}
test("mod-enter-multiline is Enter for one line and mod-enter once the draft has a line break", () => {
  const mod = { ...enter, metaKey: true };
  // One line: Enter sends, Shift+Enter breaks the line.
  assert.equal(composerSubmitIntent(enter, "mod-enter-multiline", "hi"), "default");
  assert.equal(composerSubmitIntent({ ...enter, shiftKey: true }, "mod-enter-multiline", "hi"), null);
  assert.equal(composerSubmitIntent(enter, "mod-enter-multiline"), "default");
  // Several lines: Enter adds another, the modifier sends.
  assert.equal(composerSubmitIntent(enter, "mod-enter-multiline", "a\nb"), null);
  assert.equal(composerSubmitIntent(mod, "mod-enter-multiline", "a\nb"), "default");
  assert.equal(
    composerSubmitIntent({ ...mod, shiftKey: true }, "mod-enter-multiline", "a\nb"),
    "opposite",
  );
  // The other two ignore the draft.
  assert.equal(composerSubmitIntent(enter, "enter", "a\nb"), "default");
  assert.equal(composerSubmitIntent(enter, "mod-enter", "hi"), null);
  assert.equal(effectiveSendShortcut("mod-enter-multiline", "a\nb"), "mod-enter");
  assert.equal(effectiveSendShortcut("mod-enter-multiline", ""), "enter");
  assert.deepEqual(composerShortcutLabels("mod-enter-multiline", true, "a\nb"), {
    send: "⌘Enter",
    opposite: "⇧⌘Enter",
  });
  assert.deepEqual(composerShortcutLabels("mod-enter-multiline", true, "hi"), {
    send: "Enter",
    opposite: "⌘Enter",
  });
  assert.equal(
    normalizeComposerPreferences({ sendShortcut: "mod-enter-multiline" }).sendShortcut,
    "mod-enter-multiline",
  );
});
test("one-message override flips both preferences without changing the default", () => {
  assert.equal(composerFollowUpBehavior("queue", "default"), "queue");
  assert.equal(composerFollowUpBehavior("queue", "opposite"), "steer");
  assert.equal(composerFollowUpBehavior("steer", "default"), "steer");
  assert.equal(composerFollowUpBehavior("steer", "opposite"), "queue");
  assert.deepEqual(composerShortcutLabels("mod-enter", true), {
    send: "⌘Enter",
    opposite: "⇧⌘Enter",
  });
  assert.deepEqual(composerShortcutLabels("mod-enter", false), {
    send: "Ctrl+Enter",
    opposite: "Ctrl+Shift+Enter",
  });
});
// The queue and steer chords name a behavior, where ⌘⏎ only flips the one in
// settings. Both preferences have to reach both behaviors, or a user set to
// steer would find the queue chord steering.
test("the queue and steer chords land on their behavior from either preference", () => {
  for (const preference of ["queue", "steer"] as const) {
    for (const behavior of ["queue", "steer"] as const) {
      assert.equal(
        composerFollowUpBehavior(
          preference,
          followUpSubmitIntent(preference, behavior),
        ),
        behavior,
        `${preference} preference, ${behavior} chord`,
      );
    }
  }
});
// The chords submit the form, and handleSubmit is what reads the intent, so
// the ref has to be set before requestSubmit and cleared after it returns.
test("the queue and steer chords set the intent around the submit", async () => {
  const source = await readFile(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  const body = source.slice(
    source.indexOf("const submitWithFollowUp = useCallback("),
  );
  const call = body.slice(0, body.indexOf("\n  );"));
  assert.match(call, /submitIntentRef\.current = followUpSubmitIntent\(/);
  assert.ok(
    call.indexOf("followUpSubmitIntent(") <
      call.indexOf("formRef.current?.requestSubmit()"),
    "the intent is set after the submit it belongs to",
  );
  assert.match(call, /finally \{\n\s*submitIntentRef\.current = "default";/);
});
test("legacy and malformed saved settings retain usable defaults", () => {
  const defaults = {
    plainTextComposer: true,
    showContextWindowUsage: true,
    sendShortcut: "enter",
    followUpBehavior: "queue",
  };
  for (const value of [
    null,
    undefined,
    {},
    {
      plainTextComposer: "false",
      showContextWindowUsage: 0,
      sendShortcut: "tab",
      followUpBehavior: "other",
    },
  ]) {
    assert.deepEqual(normalizeComposerPreferences(value), defaults);
  }
});
test("the real preference store persists all controls and restores them on reload", async () => {
  const { store } = installLocalStorageFake();
  const { useChatPreferencesStore: prefs } = await import(
    "../src/features/chat/stores/chat-preferences-store.ts"
  );
  prefs.getState().setPlainTextComposer(false);
  prefs.getState().setShowContextWindowUsage(false);
  prefs.getState().setSendShortcut("mod-enter");
  prefs.getState().setFollowUpBehavior("steer");
  const saved = store.get("unsloth_chat_preferences")!;
  const expected = {
    plainTextComposer: false,
    showContextWindowUsage: false,
    sendShortcut: "mod-enter",
    followUpBehavior: "steer",
  };
  assert.deepEqual(
    normalizeComposerPreferences(JSON.parse(saved).state),
    expected,
  );
  prefs.setState(normalizeComposerPreferences(null));
  store.set("unsloth_chat_preferences", saved);
  await prefs.persist.rehydrate();
  assert.deepEqual(normalizeComposerPreferences(prefs.getState()), expected);
  assert.equal(typeof prefs.getState().setFollowUpBehavior, "function");
});
