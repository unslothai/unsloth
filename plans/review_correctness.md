# PR #7074 correctness review (fork: correctness) — head a3e67ef

Scope: STT/TTS adapters, voice-settings-store, and the merge integration
(settings-search voice entry + settings-dialog search-results icon). Excludes
the 3 settled items (start(audioTrack) gating, backend TTS cancel, dictionary
spaced aliases).

## Verdict: no genuine NEW correctness bugs found.

The frontend logic is mature (many rounds of fixes already landed). I verified
the following paths against source and they are correct:

- **useDictation (shared-composer.tsx:207-366)**: stop() nulls `recognitionRef`
  before onend, so onend's `recognitionRef.current === recognition` guard
  correctly skips a double teardown while still recording `sessionTranscript`.
  stop()+restart installs a new recognizer; the old onend is a no-op for shared
  refs. onresult iterates `resultIndex..length` and joins final chunks with a
  space. onerror toasts non-abort errors. No double-record; stream released on
  every exit path. Correct.
- **studio-speech-synthesis-adapter.ts**: `handleEnd` toast filter excludes
  `interrupted`/`canceled` (system) and always toasts studio Error objects;
  cancel() during playback is guarded by `cancelled` so the audio error/ended
  listeners cannot double-fire `handleEnd`. Audio-only fallback routes to the
  studio model. Correct.
- **voice-settings-store.ts**: recent-dictation text is length-capped on save
  and hydration; count cap and dictionary caps intact; `applyDictationDictionary`
  regex failures are swallowed per-entry. Correct.

## Merge-integration checks (my a3e67ef changes) — all correct

- **settings-search.ts `voice` entry**: all 14 keys are valid `TranslationKey`s
  (tsc EXIT 0 on the exhaustive `Record<SettingsTab, ...>`). Row-label keys
  (`microphoneLabel`, `languageLabel`, `readAloud.*Label`) map to VoiceTab
  `SettingsRow`s, which emit `data-settings-label`, so search-jump resolves.
  Section-title keys follow the exact pattern the existing tabs (general,
  resources, chat) already use — consistent, not a new defect.
- **settings-dialog.tsx search-results icon**: `tab.iconComponent ? ... :
  tab.icon ? ... : null` mirrors the nav renderer and handles the Voice tab
  (icon undefined, iconComponent=MicIcon). No unguarded `HugeiconsIcon icon={undefined}`.

## Minor observations (NOT bugs, no action)

- Search results for a section-title entry (e.g. "Dictation") navigate to the
  Voice tab but will not smooth-scroll/flash a specific row, because section
  titles are rendered by `SettingsSection` (no `data-settings-label`). This is
  pre-existing behavior shared by every tab's section-title index entries, not
  introduced here; the retry loop fails gracefully (`setPendingScroll(null)`).
