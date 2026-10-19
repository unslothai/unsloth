import assert from "node:assert/strict";
import test from "node:test";
import {
  composerKeyEventForImeSubmit,
  composerSubmitIntent,
  imeKeydownBlocksComposerSubmit,
} from "../src/features/chat/utils/composer-preferences.ts";

const imeEnter = {
  key: "Enter",
  metaKey: false,
  ctrlKey: false,
  shiftKey: false,
  altKey: false,
  isComposing: false,
  keyCode: 229,
};

test("idle macOS Pinyin Enter submits (#12137)", () => {
  assert.equal(imeKeydownBlocksComposerSubmit(imeEnter, false, Infinity), false);
  assert.equal(
    composerSubmitIntent(composerKeyEventForImeSubmit(imeEnter), "enter", "nihao"),
    "default",
  );
});

test("IME-owned keydowns stay blocked", () => {
  assert.equal(imeKeydownBlocksComposerSubmit(imeEnter, true, Infinity), true);
  assert.equal(imeKeydownBlocksComposerSubmit(imeEnter, false, 5), true);
  assert.equal(
    imeKeydownBlocksComposerSubmit({ ...imeEnter, metaKey: true }, false, Infinity),
    true,
  );
  assert.equal(
    imeKeydownBlocksComposerSubmit({ ...imeEnter, key: " " }, false, Infinity),
    true,
  );
});
