# i18n Contribution Guide

- `locales/en.ts` is the complete baseline message file.
- Non-English locale files may be partial. Missing keys must fall back to English at runtime.
- Use BCP 47 locale tags for new languages, for example `zh-CN`, `pt-BR`, `ja-JP`, and `ko-KR`.
- Do not change fallback logic to hide missing translations.
- Do not add automatic DOM translation, MutationObserver text replacement, or runtime guess-based translation.
- Preserve interpolation variables exactly, for example `{count}`, `{model}`, and `{provider}`.
- Keep product and technical names unchanged unless there is an established localized name, for example `Unsloth Studio`, `LoRA`, `GGUF`, and `Hugging Face`.
- Keep translation changes small and reviewable. Prefer separate commits for runtime changes, UI migration, and locale text.
- When adding user-facing Studio UI text, add the English message key first and add non-English overrides only when the translation is clear.
- Run `npx tsx src/i18n/check-parity.ts` before committing to ensure there are no shape mismatches or placeholder discrepancies in the non-English overlays.
