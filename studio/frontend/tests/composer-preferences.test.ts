import assert from "node:assert/strict";
import test from "node:test";
import {
  composerSubmitIntent,
  composerFollowUpBehavior,
  composerShortcutLabels,
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
    for (const shortcut of ["enter", "mod-enter"] as const) {
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
