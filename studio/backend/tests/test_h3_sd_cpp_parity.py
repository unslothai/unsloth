"""MiniMax-H3 on stable-diffusion.cpp: the reference sampler / schedule and the uncompressed result container."""

import numpy as np
import pytest

from core.inference import sd_cpp_args as args
from core.inference.sd_cpp_args import (
    SdCppModelFiles,
    SdCppVideoGenParams,
    build_sd_cpp_video_command,
    build_vid_gen_request,
    h3_sd_cpp_sampler,
    h3_server_output_format,
)
from core.inference.video_minimax_h3 import _full_scale_constant

_FILES = SdCppModelFiles(
    diffusion_model = "/m/h3.gguf",
    vae = "/m/vae.safetensors",
    audio_vae = "/m/audio.safetensors",
    llm = "/m/qwen.gguf",
)


def _params(**kwargs):
    base = dict(prompt = "a dog barks", width = 1344, height = 768, num_frames = 124, steps = 20, seed = 7)
    base.update(kwargs)
    return SdCppVideoGenParams(**base)


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1] if name in cmd else None


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(args.H3_SAMPLER_ENV, raising = False)
    monkeypatch.delenv(args.H3_LOSSLESS_ENV, raising = False)


def test_reference_sampler_and_schedule_for_regular_checkpoints():
    assert h3_sd_cpp_sampler(False) == ("res_multistep", "simple")
    # Distilled / PDD checkpoints keep Euler (their recipe) on the same schedule.
    assert h3_sd_cpp_sampler(True) == ("euler", "simple")


def test_legacy_sampler_kill_switch(monkeypatch):
    monkeypatch.setenv(args.H3_SAMPLER_ENV, "legacy")
    assert h3_sd_cpp_sampler(False) == ("euler", "discrete")


def test_cli_argv_carries_sampler_scheduler_and_lossless_flag():
    method, scheduler = h3_sd_cpp_sampler(False)
    cmd = build_sd_cpp_video_command(
        "/bin/sd-cli",
        _FILES,
        _params(sample_method = method, scheduler = scheduler, lossless_output = True),
        output_path = "/tmp/out.avi",
    )
    assert _flag(cmd, "--sampling-method") == "res_multistep"
    assert _flag(cmd, "--scheduler") == "simple"
    assert "--lossless-video" in cmd
    assert cmd[cmd.index("--output") + 1] == "/tmp/out.avi"


def test_cli_argv_unchanged_without_the_new_fields():
    cmd = build_sd_cpp_video_command("/bin/sd-cli", _FILES, _params(), output_path = "/tmp/out.webm")
    assert (
        "--sampling-method" not in cmd
        and "--scheduler" not in cmd
        and "--lossless-video" not in cmd
    )


def test_server_request_mirrors_the_cli_sampler():
    req = build_vid_gen_request(_params(sample_method = "res_multistep", scheduler = "simple"))
    assert req["sample_params"]["sample_method"] == "res_multistep"
    assert req["sample_params"]["scheduler"] == "simple"
    assert req["output_format"] == "avi"
    assert build_vid_gen_request(_params(lossless_output = True))["output_format"] == "avi_raw"


def test_server_output_format_follows_the_capabilities(monkeypatch):
    assert h3_server_output_format(("avi", "avi_raw")) == "avi_raw"
    assert h3_server_output_format(("avi",)) == "avi"
    assert h3_server_output_format(()) == "avi"
    monkeypatch.setenv(args.H3_LOSSLESS_ENV, "0")
    assert h3_server_output_format(("avi", "avi_raw")) == "avi"


def test_lossless_probe_reads_help(monkeypatch, tmp_path):
    binary = tmp_path / "sd-cli"
    binary.write_bytes(b"x")
    monkeypatch.setattr(
        "core.inference.sd_cpp_backend._sd_cpp_probe_output",
        lambda b, flag: "  --lossless-video   write .avi video results uncompressed",
    )
    args._lossless_video_in_help.cache_clear()
    assert args.sd_cpp_supports_lossless_video(str(binary))
    monkeypatch.setattr(
        "core.inference.sd_cpp_backend._sd_cpp_probe_output", lambda b, flag: "  --video-frames"
    )
    args._lossless_video_in_help.cache_clear()
    assert not args.sd_cpp_supports_lossless_video(str(binary))
    assert not args.sd_cpp_supports_lossless_video(None)


def test_nan_float_pcm_counts_as_a_full_scale_level():
    # The uncompressed AVI carries float PCM, where a NaN render's audio stays NaN instead of a clamped level.
    nan = np.full((1, 64), np.nan, dtype = np.float32)
    assert _full_scale_constant(nan) == float("inf")
    assert _full_scale_constant(np.full((1, 64), 0.25, dtype = np.float32)) is None


def test_server_lists_its_video_output_formats_once():
    from core.inference.sd_cpp_server import SdCppServer

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "features_by_mode": {"vid_gen": {"ref_videos": True}},
                "output_formats_by_mode": {"vid_gen": ["avi", "avi_raw"]},
            }

    class _Client:
        def __init__(self):
            self.urls = []

        def get(
            self,
            url,
            timeout = None,
        ):
            self.urls.append(url)
            return _Resp()

    s = SdCppServer.__new__(SdCppServer)
    s.host, s.port = "127.0.0.1", 1234
    s._client = _Client()
    s._vid_gen_features = None
    s._vid_gen_output_formats = ()
    assert s.vid_gen_output_formats() == ("avi", "avi_raw")
    assert s.vid_gen_accepts(("ref_videos",))
    assert len(s._client.urls) == 1


def _bare_server(get):
    from core.inference.sd_cpp_server import SdCppServer

    class _Client:
        def __init__(self):
            self.urls = []

        def get(
            self,
            url,
            timeout = None,
        ):
            self.urls.append(url)
            return get(url)

    s = SdCppServer.__new__(SdCppServer)
    s.host, s.port = "127.0.0.1", 1234
    s._client = _Client()
    return s


def test_uncompressed_result_is_downloaded_from_media_url():
    class _Resp:
        status_code = 200
        content = b"RIFF....AVI raw"

    s = _bare_server(lambda url: _Resp())
    job = {
        "result": {
            "b64_json": "",
            "media_url": "/sdcpp/v1/jobs/j1/media",
            "media_size": len(_Resp.content),
        }
    }
    assert s._fetch_video(job) == _Resp.content
    assert s._client.urls == ["http://127.0.0.1:1234/sdcpp/v1/jobs/j1/media"]


def test_short_media_download_is_an_error():
    class _Resp:
        status_code = 200
        content = b"RIFF"

    s = _bare_server(lambda url: _Resp())
    with pytest.raises(RuntimeError, match = "download failed"):
        s._fetch_video({"result": {"media_url": "/sdcpp/v1/jobs/j1/media", "media_size": 99}})


def test_base64_results_still_decode_without_a_media_url():
    import base64
    s = _bare_server(lambda url: pytest.fail("no download expected"))
    assert s._fetch_video({"result": {"b64_json": base64.b64encode(b"AVI").decode()}}) == b"AVI"


def _refs_with_gap():
    from PIL import Image

    from core.inference import video_minimax_h3 as h3

    frames = [Image.new("RGB", (16, 16), (i, 0, 0)) for i in range(3)]
    wave = np.zeros((3200, 2), dtype = np.float32)
    # A silent video, then one with a soundtrack.
    return h3, h3.MiniMaxH3References(videos = ((frames, None, None), (frames, wave, 32000)))


def test_silent_video_before_a_soundtrack_is_refused_without_explicit_pairing(tmp_path):
    h3, refs = _refs_with_gap()
    with pytest.raises(ValueError, match = "leading"):
        h3.stage_h3_references(refs, tmp_path)


def test_silent_video_before_a_soundtrack_stages_with_explicit_pairing(tmp_path):
    h3, refs = _refs_with_gap()
    staged = h3.stage_h3_references(refs, tmp_path, explicit_pairing = True)
    assert len(staged.videos) == 2 and len(staged.video_audios) == 1
    assert staged.video_audio_indices == (1,)
    params = _params(
        ref_videos = staged.videos,
        ref_video_audios = staged.video_audios,
        ref_video_audio_indices = staged.video_audio_indices,
    )
    cmd = build_sd_cpp_video_command("/bin/sd-cli", _FILES, params, output_path = "/tmp/o.avi")
    assert _flag(cmd, "--ref-video-audio-index") == "1"
    req = build_vid_gen_request(
        params,
        ref_videos_b64 = [["f"], ["f"]],
        ref_video_audios_b64 = ["w"],
    )
    assert req["ref_video_audio_indices"] == [1]


def test_positional_soundtracks_keep_the_old_argv(tmp_path):
    from PIL import Image

    from core.inference import video_minimax_h3 as h3

    frames = [Image.new("RGB", (16, 16), (i, 0, 0)) for i in range(3)]
    wave = np.zeros((3200, 2), dtype = np.float32)
    refs = h3.MiniMaxH3References(videos = ((frames, wave, 32000), (frames, None, None)))
    staged = h3.stage_h3_references(refs, tmp_path, explicit_pairing = True)
    assert staged.video_audio_indices == ()
    cmd = build_sd_cpp_video_command(
        "/bin/sd-cli",
        _FILES,
        _params(ref_videos = staged.videos, ref_video_audios = staged.video_audios),
        output_path = "/tmp/o.avi",
    )
    assert "--ref-video-audio-index" not in cmd


@pytest.mark.parametrize(
    "indices,audios",
    [((0, 0), ("a.wav", "b.wav")), ((2,), ("a.wav",)), ((0,), ("a.wav", "b.wav"))],
    ids = ["duplicate", "out-of-range", "count-mismatch"],
)
def test_bad_soundtrack_indices_are_refused(indices, audios):
    params = _params(
        ref_videos = ("v0", "v1"), ref_video_audios = audios, ref_video_audio_indices = indices
    )
    with pytest.raises(ValueError):
        build_sd_cpp_video_command("/bin/sd-cli", _FILES, params, output_path = "/tmp/o.avi")
