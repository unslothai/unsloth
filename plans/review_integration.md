# PR #7074 review — merge integration + i18n/edge (head a3e67ef)

Task: review the a3e67ef merge (main's settings-search feature × this PR's Voice tab) for integration
bugs, dropped/duplicated merge content, and invalid/missing i18n keys. Review only, no edits.

## Verdict: no genuine new bugs. Integration is correct.

Checks that pass:
- `tsc -b` EXIT 0 on the full merged tree.
- `npm run i18n:check` -> "All locale overlays pass parity" (no missing/invalid keys).
- `biome lint en.ts` -> no duplicate object keys (the two `voice:` matches are distinct keys:
  `settings.tabs.voice` label vs the `settings.voice` block at en.ts:105).
- Merge preserved the PR's wiring:
  - runtime-provider.tsx: `adapters = { history, dictation, speech, attachments }` intact (read-aloud
    adapter still registered).
  - thread.tsx: the DeleteMessageButton `isSpeaking` guard (aefc743 fix) survived the merge
    (thread.tsx:3816/3821).
- Search-scroll works for Voice rows: `SettingsRow` emits `data-settings-label={label}`
  (settings-row.tsx:27) and `SettingsSection` emits `data-settings-label={title}`
  (settings-section.tsx:16); the Voice tab uses both, and every search-index key uses the exact same
  translation key as the row it targets, so `openResult` -> `data-settings-label === t(key)` matches.
- settings-dialog.tsx search-results icon branch renders `tab.iconComponent` (MicIcon) for the Voice
  tab correctly (type-checked; MicIcon is `FC<{className?}>`).

## Low-severity observations (NOT recommending changes)

| Sev | File:line | Note |
|-----|-----------|------|
| very-low | settings-search.ts (voice: voiceLabel, pitchLabel) | These two rows render only when `effectiveEngine === "system"`. Searching them while the engine is "studio" (or system voices unavailable) opens the Voice tab but the scroll no-ops (retry-then-give-up in settings-dialog.tsx:193). This is graceful and matches main's behavior for any conditionally-rendered row, so keeping them indexed (searchable when visible) is the right call. |
| very-low | settings-search.ts (voice) | The studio-engine `readAloud.modelLabel` row is not indexed; minor completeness gap only. Tab is still reachable by name and by every other indexed row. |

Neither observation is a correctness bug or a regression; both are cosmetic search-UX under the strict
bar and do not warrant a code change.

## Already-settled items (excluded per directive, not re-reported)
start(audioTrack) gating (unfixable platform limit), backend Studio-TTS GPU cancellation (out-of-scope
backend), dictionary spaced-alias matching (deferred design).
