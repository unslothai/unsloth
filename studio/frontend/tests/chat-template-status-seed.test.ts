// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// How /api/inference/status moves the chat-template override pair. The control is what the
// next load or Apply sends and what the Hub settings page presents as the live config; the
// loaded baseline is what the resident server is running and what a rollback resends.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import type {
  ChatTemplateSeed,
  ChatTemplateSeedState,
} from "../src/features/chat/lib/resolve-chat-template-seed.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { resolveChatTemplateSeed } = await import(
  "../src/features/chat/lib/resolve-chat-template-seed.ts"
);

/** The applier must route the pair through the resolver, not re-inline the old guard. */
const DELEGATES_TO_RESOLVER = /resolveChatTemplateSeed\(\{/;

const OLD = "{{ bos_token }}{% for m in messages %}A{% endfor %}";
const NEW = "{{ bos_token }}{% for m in messages %}B{% endfor %}";
const STAGED = "{{ bos_token }}{% for m in messages %}C{% endfor %}";

function seed(
  incoming: string | null | undefined,
  previous: ChatTemplateSeedState,
  options: {
    hydratingExistingModel?: boolean;
    seedLoadParams?: boolean;
  } = {},
): ChatTemplateSeed {
  return resolveChatTemplateSeed({
    incoming,
    previous,
    hydratingExistingModel: options.hydratingExistingModel ?? false,
    seedLoadParams: options.seedLoadParams ?? true,
  });
}

function pair(
  chatTemplateOverride: string | null,
  loadedChatTemplateOverride: string | null,
): ChatTemplateSeedState {
  return { chatTemplateOverride, loadedChatTemplateOverride };
}

test("a same-model reload from another client advances an undirty pair", () => {
  // Another tab, or an OpenAI-compatible caller whose auto-switch load applied this
  // model's saved override, relaunched the same checkpoint and quant on NEW.
  assert.deepEqual(seed(NEW, pair(OLD, OLD)), {
    chatTemplateOverride: NEW,
    loadedChatTemplateOverride: NEW,
  });
});

test("a same-model reload that drops the override is adopted too", () => {
  assert.deepEqual(seed(null, pair(OLD, OLD)), {
    chatTemplateOverride: null,
    loadedChatTemplateOverride: null,
  });
});

test("a genuinely dirty control survives, while its baseline advances", () => {
  // A staged Apply put STAGED on the control and no load is in flight yet.
  const result = seed(NEW, pair(STAGED, OLD));
  assert.equal(result.loadedChatTemplateOverride, NEW);
  assert.ok(
    !("chatTemplateOverride" in result),
    "a staged edit must not be overwritten by a status refresh",
  );
});

test("a control the user blanked is not re-pinned", () => {
  const result = seed(NEW, pair(null, OLD));
  assert.equal(result.loadedChatTemplateOverride, NEW);
  assert.ok(!("chatTemplateOverride" in result));
});

test("a steady poll touches neither field", () => {
  assert.deepEqual(seed(OLD, pair(OLD, OLD)), {});
  assert.deepEqual(seed(OLD, pair(STAGED, OLD)), {});
  // Blank and absent are the same template, so neither is a spurious change.
  assert.deepEqual(seed(null, pair("", "")), {});
  assert.deepEqual(seed("", pair(null, null)), {
    chatTemplateOverride: "",
    loadedChatTemplateOverride: "",
  });
});

test("a load in flight keeps its own params", () => {
  assert.deepEqual(seed(NEW, pair(OLD, OLD), { seedLoadParams: false }), {});
  assert.deepEqual(
    seed(NEW, pair(OLD, OLD), {
      seedLoadParams: false,
      hydratingExistingModel: true,
    }),
    {},
  );
});

test("a model or quant switch adopts the new model's template outright", () => {
  assert.deepEqual(
    seed(null, pair(OLD, OLD), { hydratingExistingModel: true }),
    {
      chatTemplateOverride: null,
      loadedChatTemplateOverride: null,
    },
  );
  assert.deepEqual(
    seed(NEW, pair(STAGED, OLD), { hydratingExistingModel: true }),
    { chatTemplateOverride: NEW, loadedChatTemplateOverride: NEW },
  );
});

test("a fresh store seeds from status, an older backend changes nothing", () => {
  assert.deepEqual(seed(OLD, pair(null, null)), {
    chatTemplateOverride: OLD,
    loadedChatTemplateOverride: OLD,
  });
  assert.deepEqual(seed(undefined, pair(null, null)), {});
  assert.deepEqual(seed(undefined, pair(OLD, OLD)), {});
});

test("the status applier delegates to the seed resolver", () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const source = readFileSync(
    path.join(
      here,
      "../src/features/chat/lib/apply-inference-status-to-store.ts",
    ),
    "utf8",
  );
  assert.match(source, DELEGATES_TO_RESOLVER);
  assert.ok(
    !source.includes("prevState.loadedChatTemplateOverride === null"),
    "the inline chat-template guard must not survive alongside the resolver",
  );
});
