# Native parallel browser probe

This fixture runs the production browser stores, executor, page agent, native view manager and capture code in a Tauri debug app. Both page webviews and the Rust bridge are real. A loopback server provides two pages with independent document tokens and receives browser-action results; no model is required.

It checks background input without stealing foreground focus, independent page state, overlapping A/A/B screenshot requests, page retention after replies finish and the panel closes, chat-scoped Stop, foreign-target rejection, and release when the owner tab closes. These native checks complement the production Studio two-chat UI scenario; the fixture shell does not exercise chat creation or provider streaming.

Use a disposable checkout: preparation replaces the frontend entrypoint for this test build. Install the platform's Tauri prerequisites and a Studio backend normally, then run from the checkout root:

```sh
python studio/frontend/tests/native_browser_parallel/runner.py --prepare .
cd studio
npm install --save-dev @tauri-apps/cli@2.10.1
npx tauri build --debug --no-bundle
cd ..
python studio/frontend/tests/native_browser_parallel/runner.py --bin studio/src-tauri/target/debug/unsloth-studio --out native-evidence
```

Windows uses `unsloth-studio.exe`. Linux without a display runs the final command under `xvfb-run -a -s "-screen 0 1440x900x24"`. If `CARGO_TARGET_DIR` is set, use that directory's debug binary instead. Preparation must be followed by the normal frontend build; the Tauri build hook performs it.

Run the same fixture against the PR merge base and head, sequentially on each OS. Inspect `result.json`, `progress.jsonl`, native capture PNGs and `app.log`; a missing result is infrastructure failure, not a passing test. The probe returns 0 only when its native assertions pass, 1 for an observed assertion failure, and a nonzero exception exit for setup/runtime failure. Keep the original source refs with the results.
