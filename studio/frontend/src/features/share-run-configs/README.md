# Share run settings

Shares the model picker's inference configuration through a URL. The Share button opens a preview with a checkbox for each available setting. Generating or copying a link makes no network request.

## Links

Desktop:

```text
unsloth://run?v=1&model=unsloth%2FExample-GGUF&ggufVariant=Q4_K_M&nParallel=2&kvCacheDtype=q8_0
```

Browser:

```text
http://localhost:8888/chat#run?v=1&model=unsloth%2FExample-GGUF&nParallel=2
```

The browser address is the Studio instance to open. A localhost link targets the recipient's own installation and port. Desktop links use the existing registered `unsloth` protocol. Browser payloads use the fragment, which is not included in HTTP requests.

Every parameter is optional. `v` defaults to `1`; unsupported versions are rejected. `model` is a Hugging Face repository ID. If omitted, the current local inference model is used; if none is selected, Studio asks for a model ID. `ggufVariant` uses the existing quantization or relative GGUF file identity. `isGguf` is an optional format hint; otherwise Studio uses the selected model, inventory metadata or its name/variant. Explicit `isGguf=false` does not inherit the selected model's GGUF variant or file access token.

Configuration parameter names match `PerModelConfig`. `fields.ts` defines the complete typed allowlist and uses the editor's existing bounds and enums. Missing fields retain the values the existing editor resolves, including remembered settings. Explicit `null`, `false`, `0`, empty strings and empty argument arrays retain their distinct meanings. Normal backend applicability, device reconciliation, and validation still apply.

`customContextLength` and `maxSeqLength` represent the same context pin on different backends. Supplying either field clears an older value in the other field, matching the editor's context-setting behavior. Supplying neither preserves the existing pin. If both are supplied with non-null values, they must agree; conflicting lengths reject the whole link.

Numbers and booleans use JSON values. Enums accept their plain names or JSON strings. Free text accepts plain text or a JSON string; use a JSON string for literal `"null"` or text starting with a quote. Nullable settings accept `null`.

Extra arguments use a JSON array of individual argument tokens, encoded with `URLSearchParams`. Only the inference options explicitly listed in `extra-args.ts` can be shared. Each value must match its numeric range or enum; switches take no value. Flags and values must be separate tokens; attached `--flag=value` syntax is rejected, matching the existing backend validator. Supported tokens and their order round-trip exactly. No shell parsing or execution occurs when opening a link. For example:

```js
const parameters = new URLSearchParams({
  model: "unsloth/Example-GGUF",
  llamaExtraArgs: JSON.stringify(["--rope-scaling", "yarn", "--yarn-orig-ctx", "32768"]),
});
const link = `unsloth://run?${parameters}`;
```

Opening a link navigates to the existing model configuration editor. It does not load a model, start a download, or save the imported settings. The existing Load/Reload and Remember controls own those actions. A pending import waits for saved settings to hydrate, then overlays only the supplied fields once. A model-only link does not mark the settings as edited. Closing the receiving editor cancels an unfinished import, even when the same model's sidebar remains open. The sidebar reflects the shared draft without consuming pending links. Links received during login remain in memory until authentication completes. Newer native run links, rejected run links and existing Hub intents supersede an older startup link, including during credential loading.

Before handing a Hub model to the editor, the receiver checks the shared local inventory and refreshes expired results using its existing freshness window. Complete cached models carry their local load ID and downloaded status into the existing Load action. GGUF checks use an offline-only listing for the requested variant at the exact load target; partial or different variants and copies in other snapshots do not count as downloaded. Matching active-model imports retain the recipient's existing load path without an inventory lookup. These paths stay local and are never included in shared URLs. Failed availability checks cancel the import with a retry message, and newer links supersede pending checks.

If saved settings fail to load or the editor's wait expires, the import is cancelled with a message and the draft remains untouched. Close the editor and reopen the link after the connection recovers. This prevents a partial link from replacing remembered defaults with temporary fallback values. Consecutive duplicate native events are ignored; an intervening browser run link starts a new intent, including when that browser link is rejected.

Editing a setting, changing Remember, or pressing Reset while an import is waiting cancels that import. Typing also cancels it for numeric fields that commit only when focus leaves the field. The newer user action wins, including edits made through the resident model's sidebar. Ready imports apply after the editor's mount effects settle so React Strict Mode cannot consume a link and then discard its draft during remounting.

## Boundaries

- Shares per-model load settings, including advanced settings and extra arguments. Global server settings such as the VRAM budget, chat sampling presets, training settings, authentication tokens and native file access tokens are not serialized.
- Local filesystem model paths are omitted by the Share dialog. A recipient can apply a model-free link to their own selected local model. Shared extra arguments cannot specify filesystem paths, URLs, credentials, tools, MCP servers, network listeners or remote devices.
- Custom template code cannot be shared, including through extra arguments. `chatTemplateOverride` may only be omitted, `null` or an empty string. Local template editing remains available. Unsupported local values appear disabled and unselected in the Share dialog; other settings can still be shared.
- Links are limited to 16,384 characters. The preview reports oversized configurations; users can omit fields to shorten them. Browser, messaging-client and OS launcher limits can be lower.
- Unknown fields or argument flags, duplicate parameters, invalid encodings and invalid values reject the whole link. Existing argument diagnostics still decide whether a configuration can load on the recipient's backend.
- The receiving installation must include this feature. No hosted redirect service, account, backend route, new dependency, Rust change or global protocol registration change is needed.

## Integration

| File | Role |
| --- | --- |
| `fields.ts` | Field definitions and validation |
| `extra-args.ts` | Explicit inference argument allowlist, arity and value bounds |
| `links.ts` | Pure URL parsing and generation |
| `inbox.ts` | In-memory request lifetime and sparse configuration merging |
| `receive-link.ts` | Link intake and native intent ordering/deduplication |
| `target.ts` | Model identity, format and local capability resolution |
| `cached-target.ts` | Fresh local inventory and exact cached variant resolution |
| `link-handler.tsx` | Browser events, login recovery and existing editor handoff |
| `config-controls.tsx` | One-time draft application and Share entry point |
| `share-dialog.tsx` | Field selection, preview and clipboard |

The root layout mounts the receiver behind the existing credential bootstrap. The app provider supplies the existing Tauri handler with an optional URL callback. The config editor renders `SharedRunConfigControls`. Its popover protects imported forms and an open Share dialog from background focus changes; clicking outside or pressing Escape still dismisses it. Ordinary editors retain their existing focus dismissal. All sharing behavior lives in this folder; the existing editor, model loader and persistence logic stay authoritative.

## Scope of existing-file changes

| Existing file | Shared-links requirement |
| --- | --- |
| `app/provider.tsx` | Connect shared run URLs to the existing desktop listener |
| `app/routes/__root.tsx` | Mount the shared-link receiver after credential bootstrap |
| `features/deep-links/deep-link-handler.tsx` | Offer URLs to the shared-link callback before the existing Hub handler |
| `features/deep-links/deep-link-intent.ts` | Clear the previous Hub duplicate marker after handling a shared link, while preserving the navigation sequence |
| `features/model-picker/components/model-config-page.tsx` | Add Share controls, wait for the receiving editor's saved settings, and cancel pending imports on newer edits |
| `features/model-picker/components/model-selector.tsx` | Protect shared-link editors and an open Share dialog from background focus changes |

This branch contains no general editor fixes, backend changes, model-loader changes, persistence rewrites, dependency updates or unrelated cleanup. Import cancellation does nothing without a matching pending shared link. The resident sidebar uses the existing shared draft; it cannot consume a pending import. The tests include the boundary between normal editing and shared-link behavior. Desktop listener simulations also cover ordinary Hub routing with and without the shared-link callback, mixed Hub/run batches, repeat Hub links after a shared link, delayed startup intents, and listener disposal. These simulations execute the shipped handler and receiver with mocked Tauri APIs; they do not launch an OS protocol handler.

## Local validation

From `studio/frontend`:

```sh
node --experimental-strip-types --test tests/share-run-configs*.test.ts tests/model-config*.test.ts tests/llama-extra-args*.test.ts
npm run typecheck
./node_modules/.bin/eslint src/features/share-run-configs tests/share-run-configs*.test.ts
./node_modules/.bin/biome check src/features/share-run-configs
```

For the browser regression test, start a fresh Vite server on loopback port 5198, then run `node tests/share-run-configs.browser.mjs` with Playwright available. Restart Vite after source edits so the test's direct module imports and the app use the same module instances rather than different hot-reload versions. `PLAYWRIGHT_MODULE` and `PLAYWRIGHT_CHROMIUM_PATH` can point to an existing local Playwright installation and Chromium executable. `SHARE_RUN_BASE_URL` selects another loopback development server. All backend requests are mocked and external requests are blocked. The test does not require a GPU, running backend, or model download.

Set `PLAYWRIGHT_BROWSER` to `chromium` (the default), `firefox`, or `webkit` to run the same suite in each engine. Import-to-Load cases assert that cached and pinned snapshots use their exact local paths without download requests, including Windows paths. Chromium and Firefox check the system clipboard; headless WebKit uses a clipboard test double. Inventory failure and supersession tests cover the asynchronous handoff.

Native OS protocol-launch behavior requires separate Windows, WSL2, macOS and Linux desktop testing. Browser tests exercise the native intent receiver, not the OS launcher.

## Startup size

A same-toolchain production build of `origin/main` at `606d87e76` measures 5,495.3 KiB of eager JavaScript and 1,634.5 KiB transferred. With this feature, the measurements are 5,513.1 KiB and 1,640.3 KiB, with the same 85 eager chunks. Both builds use the dependencies from that commit's lockfile. The feature fits upstream's unchanged raw and transfer budgets, with 82.6 KiB and 34.5 KiB remaining respectively.

Run `npm run build` and `npm run bundle:check` to remeasure with the installed toolchain.

## Security boundary

Treat every incoming link as untrusted. The URL length is checked before URL parsing. Raw control characters and malformed percent/UTF-8 encodings are rejected. The query is decoded once into values; encoded delimiters inside a value are never reparsed as parameters, paths or arguments. Duplicate keys are checked after decoding. JSON supplies primitive values and bounded flat arrays only. Prototype keys and unrelated application settings cannot enter the draft.

Model IDs must be Hugging Face repository IDs. GGUF identities accept conservative relative name segments, excluding traversal, absolute paths, Windows drive/stream syntax, reserved device names (including spaces before their extension), trailing dots/spaces, wildcards and encoded path separators. Canonical desktop addresses prevent URL normalization from hiding extra path segments.

Argument validation uses a static allowlist rather than the installed binary's flag catalog or a denylist. New upstream flags remain unavailable to shared links until reviewed. No free-form argument values, template expressions, grammars, file options or tool options are accepted. These restrictions apply when generating and receiving links. A disallowed value rejects the entire received configuration; it is never silently stripped and applied partially. Rejected newer links cancel pending imports. Text is rendered through React text nodes and form values, never HTML, JavaScript, shell code or a URL destination. Supplied text is not recursively decoded.

Opening a valid link only imports into the existing editor for review. The editor can make its normal metadata and saved-setting reads, including POSTs to the read-only inference validation and memory estimation endpoints. Normal application startup can also migrate existing chat preferences independently of the link. Loading, downloading and saving the imported configuration still require the existing user actions. Omitted fields retain existing local defaults, including locally configured arguments and templates; links do not certify those defaults as safe. Prompt text can influence subsequent model output and must be reviewed before loading.

Regression tests cover encoded and double-encoded attacks, duplicate and prototype keys, Windows/Unix path tricks, option/value smuggling, malformed Unicode, HTML injection, template expressions, size bounds, non-finite values before serialization, context-pin conflicts and merging, and 10,000 deterministic malformed inputs. Browser tests check clipboard sharing, login recovery, mixed browser/native intent ordering, browser hash changes, model identity, editor dismissal, saved-settings failures and timeouts, recovery, newer edits during hydration, context pins across backend fields, native-model imports under Strict Mode, inert text, unchanged drafts after rejection, cancelled pending imports, disabled unsupported sharing fields, and absence of load/download/save calls. The backend is mocked in these UI tests.

These checks are not a guarantee of zero vulnerabilities. They do not sandbox the model runtime, audit model files, remove existing tool permissions, or prove OS protocol handlers safe. A user can still choose resource-intensive settings or explicitly load an untrusted model. Backend validation, runtime updates and platform testing remain necessary.
