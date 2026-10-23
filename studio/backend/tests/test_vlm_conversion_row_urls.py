# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""VLM conversion must download row image URLs through the SSRF guard, never from private addresses."""

import io
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from datasets import Dataset
from PIL import Image

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from utils.datasets import format_conversion, remote_images  # noqa: E402
from utils.datasets.format_conversion import (  # noqa: E402
    convert_sharegpt_with_images_to_vlm_format,
    convert_to_vlm_format,
)


def _png() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(out, format = "PNG")
    return out.getvalue()


@pytest.fixture
def loopback_server():
    hits = []
    body = _png()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", hits
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse = True)
def no_llm_warning(monkeypatch):
    from utils.datasets import llm_assist
    def refuse(*args, **kwargs):
        raise RuntimeError("no LLM in tests")

    monkeypatch.setattr(llm_assist, "llm_generate_dataset_warning", refuse)


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ValueError:
        return []


def test_simple_vlm_conversion_does_not_fetch_loopback(loopback_server):
    base, hits = loopback_server
    ds = Dataset.from_list([{"image": f"{base}/a.png", "caption": "a red square"}])

    converted = _run(convert_to_vlm_format, ds, instruction = "Describe.", text_column = "caption")

    assert hits == []
    assert converted == []


def test_sharegpt_vlm_conversion_does_not_fetch_loopback(loopback_server):
    base, hits = loopback_server
    ds = Dataset.from_list(
        [
            {
                "image": f"{base}/b.png",
                "conversations": [
                    {"from": "human", "value": "<image> What is this?"},
                    {"from": "gpt", "value": "A red square."},
                ],
            }
        ]
    )

    converted = _run(convert_sharegpt_with_images_to_vlm_format, ds)

    assert hits == []
    assert converted == []


def test_public_row_urls_go_through_the_guarded_fetch(monkeypatch):
    from core.inference import tools

    calls = []
    body = _png()

    def fake_fetch(url, **kwargs):
        calls.append((url, kwargs.get("raw_bytes_max")))
        return None, body, "image/png"

    monkeypatch.setattr(tools, "_fetch_url_raw", fake_fetch)
    ds = Dataset.from_list([{"image": "https://example.com/c.png", "caption": "a red square"}])

    converted = convert_to_vlm_format(ds, instruction = "Describe.", text_column = "caption")

    assert [url for url, _ in calls] == ["https://example.com/c.png"]
    assert calls[0][1] == remote_images._MAX_IMAGE_BYTES
    image = converted[0]["messages"][0]["content"][1]["image"]
    assert image.size == (4, 4)


def test_guard_refusals_skip_the_row(monkeypatch):
    from core.inference import tools

    monkeypatch.setattr(
        tools, "_fetch_url_raw", lambda url, **kwargs: ("Blocked: private address.", "", "")
    )
    ds = Dataset.from_list(
        [
            {"image": "https://example.com/ok.png", "caption": "x"},
        ]
    )

    with pytest.raises(ValueError):
        convert_to_vlm_format(ds, instruction = "Describe.", text_column = "caption")


def _typed_image_dataset(url):
    from datasets import Image as ImageFeature
    ds = Dataset.from_dict({"image": [url], "text": ["a red square"]})
    return ds.cast_column("image", ImageFeature())


def _recording_fetch(monkeypatch, body = None):
    from core.inference import tools

    calls = []

    def fake_fetch(url, **kwargs):
        calls.append(url)
        if body is None:
            return "Blocked: refusing to fetch non-public address 10.0.0.1.", b"", ""
        return None, body, "image/png"

    monkeypatch.setattr(tools, "_fetch_url_raw", fake_fetch)
    return calls


def test_image_typed_url_rows_do_not_fetch_loopback(loopback_server):
    # datasets decodes an Image()-typed cell holding only a URL by fetching it on access.
    base, hits = loopback_server
    ds = _typed_image_dataset(f"{base}/typed.png")

    with pytest.raises((OSError, ValueError)):
        convert_to_vlm_format(ds, instruction = "Describe.")
    with pytest.raises(OSError):
        ds[0]
    assert hits == []


def test_image_typed_url_rows_use_the_guarded_fetch(monkeypatch):
    calls = _recording_fetch(monkeypatch, _png())
    ds = _typed_image_dataset("https://example.com/typed.png")

    simple = convert_to_vlm_format(ds, instruction = "Describe.")
    sharegpt = convert_sharegpt_with_images_to_vlm_format(
        ds.add_column(
            "conversations",
            [[{"from": "human", "value": "<image> hi"}, {"from": "gpt", "value": "ok"}]],
        )
    )

    assert set(calls) == {"https://example.com/typed.png"}
    assert isinstance(simple[0]["messages"][0]["content"][1]["image"], Image.Image)
    assert len(sharegpt) == 1


def test_nested_and_streamed_image_features_are_guarded(loopback_server):
    from datasets import Image as ImageFeature
    from datasets import Sequence

    base, hits = loopback_server
    nested = Dataset.from_dict({"images": [[f"{base}/n.png"]]}).cast_column(
        "images", Sequence(ImageFeature())
    )
    with pytest.raises(OSError):
        nested[0]
    with pytest.raises(OSError):
        next(iter(_typed_image_dataset(f"{base}/s.png").to_iterable_dataset()))
    assert hits == []


def test_embedded_and_hub_images_keep_datasets_own_decode(monkeypatch):
    calls = _recording_fetch(monkeypatch)
    from datasets import Image as ImageFeature

    ds = Dataset.from_dict({"image": [{"bytes": _png(), "path": None}], "text": ["x"]})
    ds = ds.cast_column("image", ImageFeature())
    out = convert_to_vlm_format(ds, instruction = "Describe.")

    assert isinstance(out[0]["messages"][0]["content"][1]["image"], Image.Image)
    assert calls == []


def test_detection_does_not_decode_typed_url_rows(loopback_server):
    from utils.datasets.format_detection import detect_vlm_dataset_structure

    base, hits = loopback_server
    with pytest.raises(OSError):
        detect_vlm_dataset_structure(_typed_image_dataset(f"{base}/detect.png"))
    assert hits == []


def test_typed_url_rows_keep_the_exif_orientation_datasets_applies(monkeypatch):
    rotated = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    Image.new("RGB", (4, 2), "red").save(rotated, format = "JPEG", exif = exif)
    _recording_fetch(monkeypatch, rotated.getvalue())

    out = convert_to_vlm_format(
        _typed_image_dataset("https://example.com/r.jpg"), instruction = "Describe."
    )
    assert out[0]["messages"][0]["content"][1]["image"].size == (2, 4)


def test_guarded_decode_is_installed_once():
    from datasets.features import image as hf_image

    from utils.datasets import remote_images

    before = hf_image.Image.decode_example
    remote_images.install_guarded_image_decode()
    assert hf_image.Image.decode_example is before
    assert getattr(before, "_unsloth_guarded", False)


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co.attacker.example/a.png",
        "https://huggingface.co@attacker.example/a.png",
    ],
)
def test_lookalike_hub_hosts_still_go_through_the_guard(monkeypatch, url):
    calls = _recording_fetch(monkeypatch)
    with pytest.raises(OSError):
        _typed_image_dataset(url)[0]
    assert calls == [url]


@pytest.mark.parametrize(
    "url", ["ftp://127.0.0.1/a.png", "sftp://10.0.0.1/a.png", "smb://host/share/a.png"]
)
def test_other_remote_schemes_are_refused_before_fsspec(monkeypatch, url):
    import fsspec

    def no_open(*args, **kwargs):
        raise AssertionError("refused scheme reached fsspec")

    monkeypatch.setattr(fsspec, "open", no_open)
    with pytest.raises(OSError, match = "unsupported scheme"):
        _typed_image_dataset(url)[0]


def _decode_without_io(monkeypatch, path):
    """Decode one bytes-less cell; any open that slips past the guard raises AssertionError."""
    from datasets import Image as ImageFeature
    from datasets.features import image as hf_image

    def no_open(*args, **kwargs):
        raise AssertionError(f"reached fsspec / PIL with {args[:1]}")

    monkeypatch.setattr(hf_image, "xopen", no_open)
    import PIL.Image

    monkeypatch.setattr(PIL.Image, "open", no_open)
    return ImageFeature().decode_example({"path": path, "bytes": None})


@pytest.mark.parametrize(
    "path",
    [
        "ftp://10.0.0.5/image.png::/tmp/x",
        "zip://a.png::ftp://10.0.0.5/x.zip",
        "zip://a.png::smb://host/share/x.zip",
    ],
)
def test_every_chained_hop_is_checked(monkeypatch, path):
    with pytest.raises(OSError, match = "unsupported scheme"):
        _decode_without_io(monkeypatch, path)


@pytest.mark.parametrize(
    "path",
    ["https://attacker.example/x.zip::/tmp/x", "zip://a.png::https://attacker.example/x.zip"],
)
def test_chained_non_hub_urls_are_refused(monkeypatch, path):
    calls = _recording_fetch(monkeypatch, _png())
    with pytest.raises(OSError, match = "remote archive"):
        _decode_without_io(monkeypatch, path)
    assert calls == []


@pytest.mark.parametrize(
    "path",
    [
        "\\\\attacker\\share\\a.png",
        "//attacker/share/a.png",
        "file:////attacker/share/a.png",
        "file://attacker/share/a.png",
        "\\\\?\\UNC\\attacker\\share\\a.png",
        "zip://a.png::\\\\attacker\\share\\x.zip",
    ],
)
def test_network_share_paths_are_refused(monkeypatch, path):
    with pytest.raises(OSError, match = "network share"):
        _decode_without_io(monkeypatch, path)


@pytest.mark.parametrize(
    "path",
    [
        "/data/images/a.png",
        "C:\\data\\a.png",
        "file:///data/a.png",
        "file://localhost/data/a.png",
        "zip://a.png::https://huggingface.co/datasets/o/r/resolve/main/a.zip",
        "gzip://a.png::hf://datasets/o/r/a.png.gz",
    ],
)
def test_local_and_hub_paths_still_reach_datasets(monkeypatch, path):
    with pytest.raises(AssertionError, match = "reached"):
        _decode_without_io(monkeypatch, path)


def test_string_rows_naming_a_network_share_never_touch_it(monkeypatch):
    import PIL.Image

    from utils.datasets import format_conversion

    def no_touch(*args, **kwargs):
        raise AssertionError(f"touched {args[:1]}")

    monkeypatch.setattr(PIL.Image, "open", no_touch)
    monkeypatch.setattr(format_conversion.os.path, "exists", no_touch)
    ds = Dataset.from_list([{"image": "\\\\attacker\\share\\a.png", "caption": "x"}])

    with pytest.raises((OSError, ValueError)) as err:
        convert_to_vlm_format(ds, instruction = "Describe.", text_column = "caption")
    assert "touched" not in str(err.value)


def test_guard_applies_when_datasets_loads_after_this_module():
    """Startup imports this module without datasets; the guard must still land on first use."""
    import subprocess
    import sys
    from pathlib import Path

    snippet = (
        "import sys\n"
        "import utils.datasets.remote_images\n"
        "assert 'datasets' not in sys.modules, 'importing remote_images loaded datasets'\n"
        "from datasets.features import image\n"
        "assert getattr(image.Image.decode_example, '_unsloth_guarded', False)\n"
        "print('GUARDED_LAZILY')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd = Path(__file__).resolve().parents[1],
        capture_output = True,
        text = True,
        timeout = 300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "GUARDED_LAZILY" in result.stdout
