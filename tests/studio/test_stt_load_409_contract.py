"""Wording contract behind the frontend's split of /audio/stt/load 409s.

The route maps two different failures onto 409: a model that was never
downloaded, and a load cancelled so training could start. Only the first is
final -- /transcribe/raw reloads on CPU after the second -- so the dictation
adapter tells them apart by the backend's message. These tests pin both ends of
that coupling, so rewording either message fails here instead of silently
discarding a recording.
"""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROUTES = REPO / "studio/backend/routes/inference.py"
SIDECAR = REPO / "studio/backend/core/inference/stt_sidecar.py"
GGML_SIDECAR = REPO / "studio/backend/core/inference/stt_ggml_sidecar.py"
PREDICATE = REPO / "studio/frontend/src/features/chat/adapters/stt-load-error.ts"

NOT_DOWNLOADED = "is not downloaded"


def test_both_failures_still_share_the_409_the_frontend_has_to_split():
    source = ROUTES.read_text(encoding = "utf-8")
    for name in ("SttModelNotDownloadedError", "SttLoadCancelledError"):
        handler = source.split(f"except {name} as e:", 1)
        assert len(handler) == 2, f"{name} no longer handled in stt_load"
        assert "status_code = 409" in handler[1].split("except", 1)[0], (
            f"{name} no longer maps to 409; revisit isUnrecoverableSttLoadError"
        )


def test_a_missing_model_says_it_is_not_downloaded():
    for path in (SIDECAR, GGML_SIDECAR):
        source = path.read_text(encoding = "utf-8")
        raises = source.count("raise SttModelNotDownloadedError(")
        assert raises, f"{path.name} raises SttModelNotDownloadedError nowhere"
        # Every raise site must carry the wording, not just one of them.
        assert source.count(NOT_DOWNLOADED) >= raises, (
            f"{path.name}: a SttModelNotDownloadedError lost the "
            f"'{NOT_DOWNLOADED}' wording the frontend matches on"
        )


def test_a_cancelled_load_does_not_borrow_that_wording():
    for path in (SIDECAR, GGML_SIDECAR):
        source = path.read_text(encoding = "utf-8")
        for chunk in source.split("raise SttLoadCancelledError(")[1:]:
            message = chunk.split(")", 1)[0]
            assert NOT_DOWNLOADED not in message, (
                f"{path.name}: a cancelled load now reads as not-downloaded, "
                "which would end the recording instead of retrying"
            )


def test_the_frontend_matches_that_wording():
    source = PREDICATE.read_text(encoding = "utf-8")
    assert NOT_DOWNLOADED in source
    assert "status === 501" in source
