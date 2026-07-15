# PR #7074 — Lifecycle / State / Resource review (fork: lifecycle)

Head a3e67ef. Scope: React effects, cleanup, listeners, mic/stream/audio resources,
persistence, races, stale closures. Settled items (start(audioTrack) gating, backend
TTS GPU cancel, dictionary aliases) excluded per directive.

## Verdict: no NEW clearly-reachable lifecycle/resource bug. Guards are thorough.

The dictation/preview/read-aloud resource handling is well-defended. Verified clean:

- **shared-composer `useDictation`** (shared-composer.tsx:207-366): `disposedRef` guards the
  `getUserMedia` await so a mic opened after unmount is released (275-279); `onend` teardown is
  gated on `recognitionRef.current === recognition` so a stop()+restart cannot cross-tear-down
  (314); `onerror` defers stream release to `onend` (which always follows `error` per spec);
  synchronous `start()` throw path resets `startingRef` + releases stream (337-340); unmount
  effect aborts recognition and stops tracks (356-366). No leak found.
- **voice-tab DictationTest + preview**: `disposedRef`/`startingRef` guards, `ownsSystemPreviewRef`
  so a studio preview never cancels an unrelated chat read-aloud, `releasePreviewAudio` on
  ended/error/catch, and `useEffect(() => stopPreview, ...)` on unmount. Listeners in
  `useAudioInputDevices` (devicechange) and `useSystemVoices` (voiceschanged) both remove on cleanup.
- **thread.tsx DeleteMessageButton guard**: `isSpeaking = message.speech != null` exactly matches
  the runtime's `thread.speech?.messageId === id` throw condition, so `stopSpeaking()` is only
  called when safe. Correct.
- **voice-settings-store recent-dictation cap**: `MAX_RECENT_DICTATION_LENGTH = 2000` applied on
  save (line 116) and hydration (line 165); bounds the persisted blob so the quota-throw vector in
  the cleanup path is closed. Correct.
- **runtime-provider speech adapter**: stateless `StudioSpeechSynthesisAdapter` created via stable
  `useMemo([])`; each `speak()` owns its own audio/utterance and cleans up on ended/error/cancel.
  System-path `handleEnd` is idempotent (guarded by `res.status.type === "ended"`), so the
  cancel()->speechSynthesis.cancel()->end-event sequence does not double-handle.

## LOW / debatable (1)

- **thread TTS may keep playing after switching threads / closing chat**
  (studio-speech-synthesis-adapter via runtime-provider). The adapter's `cancel()` is only invoked
  by assistant-ui when the runtime decides to stop speech. `base-thread-runtime-core` calls
  `_stopSpeaking?.()` inside `speak()` (to replace current speech) but I could not confirm a
  thread-switch/dispose path cancels an in-flight utterance. If it does not, a system utterance
  (global `speechSynthesis`) or a Studio `HTMLAudioElement` could outlive a thread switch with no
  in-UI Stop control — the same class the delete-guard fixed, but one layer up.
  Failure scenario: start read-aloud on message A, switch to another thread mid-playback; audio
  may continue. Confidence LOW — likely assistant-ui runtime responsibility, not clearly this PR's,
  and unverified. Suggested next step (only if reproduced): stop speech on thread switch, mirroring
  the delete guard.
