# i18n Contribution Guide

- `locales/en.ts` is the complete baseline message file.
- Non-English locale files may be partial. Missing keys must fall back to English at runtime.
- Use BCP 47 locale tags for new languages, for example `zh-CN`, `pt-BR`, `ja-JP`, and `ko-KR`.
- Do not change fallback logic to hide missing translations.
- Do not add automatic DOM translation, MutationObserver text replacement, or runtime guess-based translation.
- Preserve interpolation variables exactly, for example `{count}`, `{model}`, and `{provider}`.
- Keep product and technical names unchanged unless there is an established localized name, for example `Unsloth`, `LoRA`, `GGUF`, and `Hugging Face`.
- Keep translation changes small and reviewable. Prefer separate commits for runtime changes, UI migration, and locale text.
- When adding user-facing Unsloth UI text, add the English message key first and add non-English overrides only when the translation is clear.
- Run `npm run i18n:check` before committing to ensure there are no shape mismatches or placeholder discrepancies in the non-English overlays.
- CI runs `npm run i18n:check:strict`, which also fails on missing keys. The runtime fallback stays as a safety net, but a key that only exists in `en.ts` renders English inside a translated UI, so add every new key to all overlays in the same change.
