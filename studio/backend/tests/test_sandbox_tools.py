# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for the sandboxed-Python AST policy in core/inference/tools.py."""

import os
import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference.tools import _check_code_safety


def _ok(code: str):
    assert _check_code_safety(code) is None, code


def _blocked(code: str, *, expect_phrase: str):
    msg = _check_code_safety(code)
    assert msg is not None, code
    assert expect_phrase in msg, (expect_phrase, msg)


class TestMetadataHostDenylist:
    def test_aws_imds_literal_blocked(self):
        _blocked(
            'import requests; requests.get("http://169.254.169.254/latest/meta-data/")',
            expect_phrase = "Blocked: cloud-metadata host",
        )

    def test_gcp_metadata_dns_blocked(self):
        _blocked(
            'import requests; requests.get("http://metadata.google.internal/")',
            expect_phrase = "Blocked: cloud-metadata host",
        )

    def test_alibaba_ecs_literal_blocked(self):
        _blocked(
            'import socket; s=socket.socket(); s.connect(("100.100.100.200", 80))',
            expect_phrase = "Blocked: cloud-metadata host",
        )

    def test_ipv6_imds_literal_blocked(self):
        _blocked(
            'import urllib.request; urllib.request.urlopen("http://[fd00:ec2::254]/")',
            expect_phrase = "Blocked: cloud-metadata host",
        )

    def test_metadata_link_local_prefix_blocked(self):
        _blocked(
            'import requests; requests.get("http://169.254.170.2/v3/")',
            expect_phrase = "Blocked: cloud-metadata host",
        )


class TestTrustedHostAllowlist:
    @pytest.mark.parametrize(
        "url",
        [
            "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "https://fr.wikipedia.org/wiki/Python_(langage)",
            "https://www.google.com/search?q=foo",
            "https://duckduckgo.com/?q=foo",
            "https://huggingface.co/unsloth",
            "https://cdn-lfs.huggingface.co/repos/abc/def/file.bin",
            "https://raw.githubusercontent.com/foo/bar/main/README.md",
            "https://api.github.com/repos/foo/bar",
            "https://arxiv.org/abs/2401.12345",
            "https://export.arxiv.org/abs/2401.12345",
            "https://stackoverflow.com/questions/12345",
            "https://math.stackexchange.com/questions/12345",
            "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
            "https://docs.python.org/3/library/asyncio.html",
            "https://pypi.org/project/requests/",
            "https://files.pythonhosted.org/packages/foo/bar.whl",
            "https://www.bbc.com/news",
            "https://api.weather.gov/points/40,-90",
            "https://numpy.org/doc/stable/",
            "https://pytorch.org/docs/stable/index.html",
        ],
    )
    def test_trusted_host_passes(self, url):
        _ok(f"import requests; requests.get({url!r})")

    def test_wikipedia_subdomain_passes(self):
        _ok(
            'import urllib.request; urllib.request.urlopen("https://m.en.wikipedia.org/wiki/Foo")'
        )

    def test_hf_co_short_form_passes(self):
        _ok('import requests; requests.get("https://hf.co/unsloth/Qwen3.5-4B-GGUF")')

    def test_github_io_pages_pass(self):
        _ok('import requests; requests.get("https://unslothai.github.io/")')


class TestUntrustedHostBlock:
    def test_example_com_blocked(self):
        _blocked(
            'import requests; requests.get("https://example.com/")',
            expect_phrase = "Blocked: host not in sandbox allowlist",
        )

    def test_random_blog_blocked(self):
        _blocked(
            'import urllib.request; urllib.request.urlopen("https://random-blog-host.example/")',
            expect_phrase = "Blocked: host not in sandbox allowlist",
        )

    def test_socket_connect_random_host_blocked(self):
        _blocked(
            'import socket; s=socket.socket(); s.connect(("evil.example", 80))',
            expect_phrase = "Blocked: host not in sandbox allowlist",
        )

    def test_simple_variable_url_resolved_and_blocked(self):
        # Static AST resolves ``u = "..."; requests.get(u)`` by following the
        # assignment, so metadata / untrusted hosts are caught even when the
        # URL is staged into a local. Before this hardening pass the
        # variable-URL path was treated as opaque and slipped through.
        _blocked(
            "import requests\n" 'url = "https://example.com/"\n' "requests.get(url)",
            expect_phrase = "Blocked: host not in sandbox allowlist",
        )
        _blocked(
            "import requests\n"
            'url = "http://169.254.169.254/latest/meta-data/"\n'
            "requests.get(url)",
            expect_phrase = "Blocked: cloud-metadata host",
        )

    def test_constant_fstring_url_resolved(self):
        # f-string URLs that fold to a constant should still be checked.
        _blocked(
            "import requests\n"
            'host = "169.254.169.254"\n'
            'requests.get(f"http://{host}/latest/")',
            expect_phrase = "Blocked: cloud-metadata host",
        )

    def test_truly_dynamic_url_marked_opaque(self):
        # Genuinely runtime-computed URLs (input, env var, network) are
        # reported as opaque so the static checker stays honest -- the
        # bash blocklist + cloud-metadata IP block at OS layer cover the
        # rest, but the AST can no longer say "looks fine to me".
        _blocked(
            "import os, requests\n" 'requests.get(os.environ["WEBHOOK"])',
            expect_phrase = "network call target is computed at runtime",
        )


class TestHostNormalization:
    def test_trailing_dot_treated_same(self):
        _ok('import requests; requests.get("https://wikipedia.org./")')

    def test_explicit_port_does_not_unblock_or_misblock(self):
        _ok('import requests; requests.get("https://en.wikipedia.org:443/wiki/Foo")')
        _blocked(
            'import requests; requests.get("https://example.com:8080/")',
            expect_phrase = "Blocked: host not in sandbox allowlist",
        )

    def test_userinfo_at_does_not_smuggle_metadata_host(self):
        _blocked(
            'import requests; requests.get("https://wikipedia.org@169.254.169.254/latest/")',
            expect_phrase = "Blocked: cloud-metadata host",
        )

    def test_uppercase_host_normalised(self):
        _ok('import requests; requests.get("https://EN.WIKIPEDIA.ORG/wiki/Foo")')


class TestUploadDenylist:
    def test_requests_post_files_blocked(self):
        _blocked(
            (
                "import requests\n"
                'requests.post("https://huggingface.co/api/repos/upload", '
                'files={"f": open("x.bin", "rb")})'
            ),
            expect_phrase = "Blocked: file upload disallowed in sandbox",
        )

    def test_requests_put_data_bytes_blocked(self):
        _blocked(
            (
                "import requests\n"
                'requests.put("https://huggingface.co/api/repos/upload", '
                'data=b"\\x00\\x01\\x02")'
            ),
            expect_phrase = "Blocked: file upload disallowed in sandbox",
        )

    def test_requests_post_data_open_handle_blocked(self):
        _blocked(
            (
                "import requests\n"
                'requests.post("https://huggingface.co/api/repos/upload", '
                'data=open("x.bin", "rb"))'
            ),
            expect_phrase = "Blocked: file upload disallowed in sandbox",
        )

    def test_httpx_post_files_blocked(self):
        _blocked(
            (
                "import httpx\n"
                'httpx.post("https://huggingface.co/api/repos/upload", '
                'files={"f": open("x.bin", "rb")})'
            ),
            expect_phrase = "Blocked: file upload disallowed in sandbox",
        )

    def test_hf_api_upload_file_blocked(self):
        _blocked(
            (
                "from huggingface_hub import HfApi\n"
                'HfApi().upload_file(path_or_fileobj="x.bin", '
                'path_in_repo="x.bin", repo_id="foo/bar")'
            ),
            expect_phrase = "Blocked: file upload disallowed in sandbox",
        )

    def test_hf_module_upload_folder_blocked(self):
        _blocked(
            (
                "import huggingface_hub\n"
                'huggingface_hub.upload_folder(folder_path="./", repo_id="foo/bar")'
            ),
            expect_phrase = "Blocked: file upload disallowed in sandbox",
        )

    def test_hf_create_commit_method_blocked(self):
        _blocked(
            (
                "import huggingface_hub\n"
                "api = huggingface_hub.HfApi()\n"
                'api.create_commit(repo_id="foo/bar", operations=[])'
            ),
            expect_phrase = "Blocked: file upload disallowed in sandbox",
        )

    def test_plain_post_json_not_blocked(self):
        _ok(
            "import requests\n"
            'requests.post("https://api.weather.gov/lookup", json={"k": "v"})'
        )


class TestImportAliasResolution:
    """Aliased / from-imported network APIs must obey the same policy.

    Pre-hardening, ``import requests as r`` and ``from requests import get``
    bypassed the prefix check because the visitor matched on the literal
    "requests.<method>" FQ at the call site. The new visitor tracks
    aliases at import time and rewrites the call's FQ before policy eval.
    """

    def test_module_alias_metadata_blocked(self):
        _blocked(
            'import requests as r; r.get("http://169.254.169.254/latest/")',
            expect_phrase = "Blocked: cloud-metadata host",
        )

    def test_module_alias_untrusted_blocked(self):
        _blocked(
            'import requests as r; r.get("https://example.com/")',
            expect_phrase = "Blocked: host not in sandbox allowlist",
        )

    def test_module_alias_trusted_passes(self):
        _ok('import requests as r; r.get("https://en.wikipedia.org/wiki/Foo")')

    def test_from_import_metadata_blocked(self):
        _blocked(
            'from requests import get\nget("http://metadata.google.internal/")',
            expect_phrase = "Blocked: cloud-metadata host",
        )

    def test_from_import_aliased_blocked(self):
        _blocked(
            "from requests import get as fetch\n"
            'fetch("http://169.254.169.254/latest/")',
            expect_phrase = "Blocked: cloud-metadata host",
        )

    def test_from_import_trusted_passes(self):
        _ok(
            "from urllib.request import urlopen\n"
            'urlopen("https://en.wikipedia.org/wiki/Foo")'
        )

    def test_nested_module_alias_blocked(self):
        _blocked(
            "import urllib.request as ur\n"
            'ur.urlopen("http://169.254.169.254/latest/")',
            expect_phrase = "Blocked: cloud-metadata host",
        )


class TestSessionObjectMethods:
    """``s = requests.Session(); s.get(url)`` must obey the host policy.

    The visitor tracks session-shaped constructor assignments so method
    calls on the bound variable become egress-equivalent.
    """

    def test_requests_session_get_metadata_blocked(self):
        _blocked(
            "import requests\n"
            "s = requests.Session()\n"
            's.get("http://169.254.169.254/latest/")',
            expect_phrase = "Blocked: cloud-metadata host",
        )

    def test_requests_session_get_untrusted_blocked(self):
        _blocked(
            "import requests\n"
            "s = requests.Session()\n"
            's.get("https://example.com/")',
            expect_phrase = "Blocked: host not in sandbox allowlist",
        )

    def test_requests_session_post_upload_blocked(self):
        _blocked(
            "import requests\n"
            "s = requests.Session()\n"
            's.post("https://huggingface.co/api/repos/upload", '
            'files={"f": open("x.bin", "rb")})',
            expect_phrase = "Blocked: file upload disallowed in sandbox",
        )

    def test_requests_session_trusted_passes(self):
        _ok(
            "import requests\n"
            "s = requests.Session()\n"
            's.get("https://en.wikipedia.org/wiki/Foo")'
        )

    def test_httpx_client_metadata_blocked(self):
        _blocked(
            "import httpx\n"
            "c = httpx.Client()\n"
            'c.get("http://169.254.169.254/latest/")',
            expect_phrase = "Blocked: cloud-metadata host",
        )


class TestSandboxCpuRlimitDefault:
    """Pin the default so a regression below 600s without opt-in is caught."""

    def test_default_cpu_s_is_600(self):
        src = (_BACKEND_ROOT / "core" / "inference" / "tools.py").read_text()
        assert 'UNSLOTH_STUDIO_SANDBOX_CPU_S", "600"' in src

    def test_clone_newnet_removed(self):
        src = (_BACKEND_ROOT / "core" / "inference" / "tools.py").read_text()
        assert "_libc.unshare(0x40000000)" not in src
        # Explanatory comment retained.
        assert "CLONE_NEWNET" in src


class TestMaxBodyDefault:
    def test_default_is_500_mb(self):
        src = (_BACKEND_ROOT / "main.py").read_text()
        assert 'UNSLOTH_STUDIO_MAX_BODY_MB", "500"' in src
