# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Exercise the real chat recorder and attachment adapter with Chromium's test microphone."""

import json
import os

from playwright.sync_api import sync_playwright

from _playwright_robust import start_vite, stop_process, wait_for_smoke_page


def main():
    """Check attachment, cancellation and microphone cleanup without a model or backend."""
    port = int(os.environ.get("SMOKE_PORT", "5483"))
    server = start_vite(port)
    url = f"http://127.0.0.1:{port}/smoke-chat-audio-recording.html"
    try:
        wait_for_smoke_page(url, "/smoke-chat-audio-recording-main.tsx", proc = server)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                channel = os.environ.get("STUDIO_PLAYWRIGHT_CHANNEL") or None,
                args = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
            )
            try:
                page = browser.new_page()
                page.set_default_timeout(120_000)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                # Observe the real browser tracks; Chromium supplies a synthetic audio signal.
                page.add_init_script("""(() => {
                  const open = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
                  window.recordingStreams = [];
                  navigator.mediaDevices.getUserMedia = async (constraints) => {
                    const stream = await open(constraints);
                    window.recordingStreams.push(stream);
                    if (window.holdMic) {
                      await new Promise(resolve => { window.releaseMic = resolve; });
                    }
                    return stream;
                  };
                })()""")
                page.goto(url, wait_until = "domcontentloaded", timeout = 120_000)

                def start_recording():
                    """Start a recording and wait until the microphone is active."""
                    page.get_by_role("button", name = "Record audio for model").click()
                    page.get_by_role("button", name = "Start recording").click()
                    page.get_by_text("Recording…", exact = True).wait_for()

                def tracks_stopped():
                    """Assert every microphone stream acquired by this page has ended."""
                    assert page.evaluate(
                        "recordingStreams.length > 0 && recordingStreams.every(s => "
                        "s.getTracks().every(t => t.readyState === 'ended'))"
                    )

                start_recording()
                page.wait_for_timeout(1200)
                page.get_by_role("button", name = "Stop and attach").click()
                page.get_by_role("dialog").wait_for(state = "hidden")
                attachments = json.loads(page.get_by_label("Attachments").inner_text())
                assert len(attachments) == 1
                assert attachments[0]["name"] == "Recording.wav"
                assert attachments[0]["type"] == "audio/wav"
                assert attachments[0]["size"] > 44
                tracks_stopped()

                start_recording()
                page.get_by_role("button", name = "Cancel", exact = True).click()
                tracks_stopped()
                assert len(json.loads(page.get_by_label("Attachments").inner_text())) == 1

                # Cancel while the permission request is still in flight; the late stream must close.
                page.evaluate("window.holdMic = true")
                page.get_by_role("button", name = "Record audio for model").click()
                page.get_by_role("button", name = "Start recording").click()
                page.get_by_text("Waiting for microphone…", exact = True).wait_for()
                page.wait_for_function("typeof window.releaseMic === 'function'")
                page.get_by_role("button", name = "Cancel", exact = True).click()
                page.evaluate("window.holdMic = false; window.releaseMic()")
                page.wait_for_function(
                    "recordingStreams.every(s => s.getTracks().every(t => t.readyState === 'ended'))"
                )
                assert len(json.loads(page.get_by_label("Attachments").inner_text())) == 1

                start_recording()
                page.keyboard.press("Escape")
                tracks_stopped()
                assert len(json.loads(page.get_by_label("Attachments").inner_text())) == 1

                start_recording()
                # The app unmounts this control when its composer identity changes.
                page.evaluate("document.querySelector('button').click()")
                page.get_by_role("dialog").wait_for(state = "hidden")
                tracks_stopped()
                assert len(json.loads(page.get_by_label("Attachments").inner_text())) == 1
                assert not errors, errors
                print("Audio recording, cancellation, Escape and chat-switch cleanup passed")
            finally:
                browser.close()
    finally:
        stop_process(server)


if __name__ == "__main__":
    main()
