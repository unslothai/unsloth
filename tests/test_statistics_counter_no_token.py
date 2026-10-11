"""The download counter must fetch its public repos without the user's Hugging Face token (#8900)."""

import huggingface_hub
import unsloth_zoo.rl_environments as rl_environments

from unsloth.models import _utils


def test_statistics_fetch_sends_no_token(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("HF_TOKEN", "hf_should_never_be_sent")
    monkeypatch.setattr(_utils, "has_internet", lambda *a, **k: True)
    monkeypatch.setattr(rl_environments, "execute_with_time_limit", lambda seconds: (lambda fn: fn))
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo_id, **kwargs: calls.append((repo_id, kwargs)),
    )

    _utils._get_statistics("other")

    assert [repo for repo, _ in calls] == ["unslothai/other"]
    assert calls[0][1]["token"] is False
    assert (
        huggingface_hub.utils.build_hf_headers(token = calls[0][1]["token"]).get("authorization")
        is None
    )
