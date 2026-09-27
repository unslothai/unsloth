import assert from "node:assert/strict";
import test from "node:test";
import {
  composerKeyEventForImeSubmit,
  composerSubmitIntent,
  imeKeydownBlocksComposerSubmit,
} from "../src/features/chat/utils/composer-preferences.ts";

const enter = {
  key: "Enter",
  metaKey: false,
  ctrlKey: false,
  shiftKey: false,
  altKey: false,
};

test("idle macOS Pinyin Enter is not treated as IME-owned", () => {
  assert.equal(
    imeKeydownBlocksComposerSubmit(
      { ...enter, isComposing: true, keyCode: 229 },
      false,
    ),
    false,
  );
  assert.equal(
    composerSubmitIntent(
      composerKeyEventForImeSubmit({
        ...enter,
        isComposing: true,
        keyCode: 229,
      }),
      "enter",
      "nihao",
    ),
    "default",
  );
});

test("modifier Enter during IME composition is still deferred to the IME", () => {
  assert.equal(
    imeKeydownBlocksComposerSubmit(
      { ...enter, isComposing: true, metaKey: true, keyCode: 13 },
      false,
    ),
    true,
  );
});

test("active composition still blocks idle-looking Enter", () => {
  assert.equal(
    imeKeydownBlocksComposerSubmit(
      { ...enter, isComposing: true, keyCode: 229 },
      true,
    ),
    true,
  );
  assert.equal(
    imeKeydownBlocksComposerSubmit({ ...enter, isComposing: false }, true),
    false,
  );
});
