# ADR 0000 — The product is named `Rag Platform`; the vendor name survives only where a licence requires it

* Status: Accepted
* Date: 2026-08-12
* Scope: Faz 0A, `studio/frontend/`, `infra/rag-platform/`, root licence files
* Supersedes: nothing. Superseded by: nothing.

## Context

The frontend is a fork of Unsloth Studio. The upstream vendor name is spread
across the tree in kinds that look identical to `grep` but are not
interchangeable: some are product branding, some are protocol values, and some
are legally required attribution. A blanket search-replace would rename all
three.

### The measured surface

Under `studio/frontend/src`, **1110 occurrences across 256 files**. Broken down
by what the occurrence actually is:

| Kind | Count | Renamable? |
|---|---|---|
| Hugging Face model ids (`unsloth/Qwen-Image-2512`, …) | 158 | No — third-party resource identifiers |
| CSS class and token names (`.unsloth-composer-plus`, `unsloth-green`) | 91 | Yes, but invisible to users |
| Backend environment variable names (`UNSLOTH_MODEL_IDLE_TTL`, …) | 34 | No — read by the Studio backend |
| Upstream URLs (`unsloth.ai/docs`, `github.com/unslothai/…`) | 88 | No — attribution and live upstream docs |
| CLI invocations (`unsloth studio update`) | 13 | No — the installed executable's real name |
| Remainder (UI copy, alt text, aria labels, error strings, comments) | ~726 | **Yes — this is the branding surface** |

Only the last row is what §1.1 of the plan is about. The plan says this
explicitly: "Container içindeki upstream zorunlu path/import/package adları
toplu search-replace ile değiştirilmeyecektir; teknik uyumluluk için korunan bu
değerler kullanıcıya gösterilmez."

### Two document-title sources, one already renamed

`studio/frontend/index.html:7` already reads `<title>Rag Platform</title>`
(commit `a48c6962c`). But the running app overwrites it:

```
studio/frontend/src/app/routes/__root.tsx:146
  const DEFAULT_DOCUMENT_TITLE = "Unsloth";
studio/frontend/src/app/routes/__root.tsx:239-240
  ? `${documentTitle} - ${DEFAULT_DOCUMENT_TITLE}`
  : DEFAULT_DOCUMENT_TITLE;
```

So the static HTML is correct and the live title is not. This is exactly why the
plan requires a *single* branding source rather than per-file fixes: the same
string existing in two places is how a rebrand half-lands.

`studio/frontend/src/i18n/locales/en.ts:49-50` is a third source:

```js
brand: "unsloth",
product: "Unsloth",
```

### What licences require us to keep

**53 tracked files** under `studio/` carry both an SPDX identifier and an
upstream copyright line, e.g.:

```
<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
     See /studio/LICENSE.AGPL-3.0 -->
```

`THIRD_PARTY_NOTICES.md` records the finding that 1231 of 1232 tracked
upstream-derived paths sit under `studio/`, so the effective licence of the
frontend is **AGPL-3.0-only**, not Apache-2.0, and §13's network-use clause
applies to any hosted deployment. Removing or rewriting those headers would be a
licence violation, not a branding change. Two of them were in fact removed by
earlier commits `e9bb2796f` and `2bea3f537`; Faz 0 restored them.

`studio/frontend/src/features/settings/tabs/about-tab.tsx` links upstream docs,
changelog, issues and both licence texts (lines 212–271). Those links are
attribution and must resolve to the real upstream, not to a Rag Platform page.

### Persistence keys are not branding

`localStorage` keys are vendor-prefixed — `unsloth_auth_token`,
`unsloth_chat_only`, `unsloth_user_profile`, `unsloth_training_config_v*` and
~20 more. They are never rendered. Renaming them silently signs every existing
user out and orphans their stored settings.

### Logo assets

`studio/frontend/public/` ships `unsloth-gem.png`, `logotext.png`,
`circle-logo-small.png`, `unsloth.ico`, `rounded-512.png` and a `Sloth emojis`
directory. Exactly one file under `src` references any of them
(`features/studio/training-start-overlay.tsx`). The plan forbids reusing a
recoloured upstream mark: "Geçici logo gerekiyorsa yalnızca tipografik
`Rag Platform` wordmark kullan; upstream logoyu yeniden renklendirip
kullanma."

## Decision

**1. One branding source.** `studio/frontend/src/config/branding.ts` exports the
product name, short name, slug, document title and default metadata. It sits
beside the existing `config/env.ts` and `config/disabled-features.ts`. Nothing
else may hardcode a product name: `__root.tsx`'s `DEFAULT_DOCUMENT_TITLE` and
`en.ts`'s `shell.brand` / `shell.product` both read from it.

**2. The identity values from plan §1.2 are frozen.** Displayed name
`Rag Platform`; slug `rag-platform`; Compose project `rag-platform`; container
`rag-platform-backend`; services `platform-backend-cpu` / `platform-backend-gpu`;
image `rag-platform-backend:<version>`; network `rag-platform-network`;
integration folder `src/integrations/platform-backend`; client `platformRequest`;
error class `PlatformApiError`; env prefix `VITE_RAG_PLATFORM_`. Changing any of
these requires a superseding ADR.

**3. Four categories are renamed; five are not.**

Renamed — everything a user can read: UI copy, document title, login and
onboarding text, `alt` text, `aria-label`s, error and empty-state messages,
About/Settings content, and the `en.ts` brand block.

Not renamed, each for a stated reason:

| Kept | Reason |
|---|---|
| 53 SPDX + copyright headers | Licence obligation |
| `LICENSE`, `studio/LICENSE.AGPL-3.0`, `THIRD_PARTY_NOTICES.md` | Licence obligation |
| `about-tab.tsx` upstream links | Attribution must point at the real upstream |
| Hugging Face model ids, `UNSLOTH_*` env names, `unsloth …` CLI strings | Protocol/identifier values; renaming breaks the call |
| `unsloth_*` `localStorage` keys | Renaming signs users out and orphans settings |

CSS class names are renamable but user-invisible, so they are out of Faz 0
scope: touching 91 selectors is formatting churn with no user-facing effect.

**4. A typographic wordmark, no recoloured upstream mark.** Until a real mark
exists, `Rag Platform` is rendered as text. Upstream image assets stay on disk
untouched — they are AGPL-covered upstream content — and are dereferenced from
UI code rather than edited or recoloured.

**5. The scan is enforced, not trusted.** `scripts/rag-platform/branding-scan.mjs`
fails on any user-visible vendor-name occurrence, with an allowlist naming the
53 header files, `about-tab.tsx`, and the identifier categories above. Each
allowlist entry carries its reason inline. The gate runs at every phase end, so a
later phase cannot reintroduce the name by copying an upstream file.

## Alternatives rejected

* **Tree-wide search-replace of the vendor name** — would rewrite 53 licence
  headers, 158 Hugging Face model ids, 34 backend environment variable names and
  every stored-settings key. Breaks the licence and the running app at once.
* **Rename per file as each is touched** — leaves the name reachable for the
  whole migration and gives no gate. The two-title bug above is what this
  produces.
* **Renaming `localStorage` keys for tidiness with a migration shim** — real
  work, zero user-visible benefit, and it puts an auth-token migration on the
  critical path of a branding task.
* **Recolouring `unsloth-gem.png` as an interim logo** — explicitly forbidden by
  the plan, and it would be a derivative of an AGPL-licensed asset presented as
  our own mark.
* **Dropping `about-tab.tsx`'s upstream links** — would remove attribution the
  AGPL requires while the code stays.

## Consequences

* One file governs the product name; a rebrand or a second white-label is a
  one-file change.
* `grep -i unsloth` will keep returning hits forever. That is correct: the
  remaining hits are licence text, third-party identifiers and storage keys. The
  branding scan — not raw grep — is the release gate.
* The effective frontend licence is **AGPL-3.0-only**. Any hosted Rag Platform
  deployment triggers §13 and must offer corresponding source to its users. This
  is a distribution obligation, recorded here so it is not discovered later.
* `studio/frontend/public/` keeps upstream-branded images that no UI references.
  They are left in place deliberately; deleting AGPL-covered upstream content is
  a separate decision from not displaying it.
* CSS classes and storage keys keep the vendor prefix. Any future phase that
  wants them renamed needs its own migration, not a branding pass.
