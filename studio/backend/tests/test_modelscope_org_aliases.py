# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A model of an org-aliased Hugging Face org that maps to another ModelScope id must, once mirrored there,
have loaded files that match Hugging Face's, and ModelScope must not host the model under its own id."""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from hub.modelscope import upstream

HF = "https://huggingface.co"
# Weights, configs, remote code and tokenizers; anything else may differ. ModelScope adds configuration.json.
LOADED = re.compile(
    r"\.(safetensors|sft|bin|gguf|pt|pth|ckpt|onnx|nemo|msgpack|h5|json|ya?ml|py|model|tiktoken|jinja|vocab)$"
    r"|(^|/)(vocab|merges|tokenizer|spiece|sentencepiece)[^/]*\.txt$",
    re.I,
)


def loaded(path: str) -> bool:
    return (
        bool(LOADED.search(path))
        and not re.search(r"(^|/)\.", path)
        and path != "configuration.json"
    )


INCONCLUSIVE = object()


def _get(
    http: httpx.Client,
    url: str,
    params = None,
    *,
    definite = (200, 404),
    headers = None,
):
    """The response once its status is definite; INCONCLUSIVE after repeated throttling or errors."""
    wait = 0.0
    for attempt in range(5):
        time.sleep(wait)
        wait = 2.0 ** (attempt + 1)
        try:
            response = http.get(url, params = params, headers = headers)
            if response.status_code in definite:
                return response
            retry_after = response.headers.get("retry-after", "")
            if retry_after.isdigit():
                wait = max(wait, min(float(retry_after), 60.0))
        except httpx.HTTPError:
            pass
    return INCONCLUSIVE


def _hf_get(
    http: httpx.Client,
    url: str,
    params = None,
    *,
    definite,
):
    # The token only lifts Hugging Face's anonymous rate limit; it must never reach ModelScope.
    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else None
    return _get(http, url, params, definite = definite, headers = headers)


def _hf_models(http: httpx.Client, org: str) -> list[str]:
    url, params, repos = f"{HF}/api/models", {"author": org, "limit": 1000}, []
    while url:
        response = _hf_get(http, url, params, definite = (200,))
        assert response is not INCONCLUSIVE, f"Hugging Face would not list {org}'s models"
        repos += [entry["id"] for entry in response.json()]
        url, params = response.links.get("next", {}).get("url"), None
    return repos


def _hf_files(http: httpx.Client, repo: str):
    """File -> (size, LFS sha256 or ""); None when Hugging Face refuses the listing."""
    url, params, files = f"{HF}/api/models/{repo}/tree/main", {"recursive": "true"}, {}
    while url:
        response = _hf_get(http, url, params, definite = (200, 401, 403, 404))
        if response is INCONCLUSIVE:
            return response
        if response.status_code != 200:
            return None
        for entry in response.json():
            if entry.get("type") == "file":
                files[entry["path"]] = (
                    entry.get("size", 0),
                    (entry.get("lfs") or {}).get("oid", ""),
                )
        url, params = response.links.get("next", {}).get("url"), None
    return files


def _ms_files(http: httpx.Client, repo: str):
    """File -> (size, sha256); None when ModelScope has no such repo."""
    response = _get(
        http, f"{upstream.MODELSCOPE}/api/v1/models/{repo}/repo/files", {"Recursive": "true"}
    )
    if response is INCONCLUSIVE:
        return response
    if response.status_code == 404:
        return None
    entries = (response.json().get("Data") or {}).get("Files") or []
    return {
        e["Path"]: (int(e.get("Size") or 0), e.get("Sha256") or "")
        for e in entries
        if e.get("Type") != "tree" and isinstance(e.get("Path"), str)
    }


def mismatch(hf: dict, ms: dict) -> list[str]:
    """Loaded files missing, different (non-LFS by size) or extra on ModelScope."""
    bad = []
    for path, (size, sha) in hf.items():
        if not loaded(path):
            continue
        if path not in ms:
            bad.append(f"missing {path}")
        elif (sha and ms[path][1] and sha != ms[path][1]) or (
            not (sha and ms[path][1]) and size != ms[path][0]
        ):
            bad.append(f"differs {path}")
    bad += [f"extra {path}" for path in ms if path not in hf and loaded(path)]
    return bad


def _check(http: httpx.Client, repo: str, target: str):
    same_id = _ms_files(http, repo)
    if same_id is INCONCLUSIVE:
        return same_id
    if same_id is not None:
        return f"ModelScope now has {repo} itself"
    ms = _ms_files(http, target)
    if ms is None:
        return None  # not mirrored yet: the alias answers 404 just as the id would
    hf = _hf_files(http, repo)
    if ms is INCONCLUSIVE or hf is INCONCLUSIVE:
        return INCONCLUSIVE
    if hf is None:
        return "Hugging Face has no file listing for it (gated or deleted)"
    bad = mismatch(hf, ms)
    return f"{target}: {', '.join(bad[:5])}" if bad else None


def org_alias_failures(orgs) -> tuple[dict[str, str], list[str], int]:
    """Failing models, inconclusive models, and how many models were checked (kept ones are not)."""
    with httpx.Client(timeout = 60, follow_redirects = True) as http:
        listed = [repo for org in orgs for repo in _hf_models(http, org)]
        ruled = {org.casefold() for org in orgs}
        known = {repo.casefold() for repo in listed}
        listed += [  # an exact alias stays checked even when the org listing leaves its model out
            repo
            for repo in upstream._ALIASES["model"]
            if repo.split("/", 1)[0].casefold() in ruled and repo.casefold() not in known
        ]
        mapped = {repo: upstream.upstream_id("model", repo) for repo in listed}
        mapped = {
            repo: target for repo, target in mapped.items() if target.casefold() != repo.casefold()
        }
        with ThreadPoolExecutor(8) as pool:
            results = dict(
                zip(mapped, pool.map(lambda repo: _check(http, repo, mapped[repo]), mapped))
            )
    failures = {repo: why for repo, why in results.items() if isinstance(why, str)}
    return failures, [repo for repo, why in results.items() if why is INCONCLUSIVE], len(results)


def remedy(failures: dict[str, str]) -> str:
    exact = {repo.casefold() for repo in upstream._ALIASES["model"]}
    lines = {True: [], False: []}
    for repo, why in sorted(failures.items(), key = lambda kv: kv[0].casefold()):
        lines[repo.casefold() in exact].append(f'    "{repo}",  # {why}')
    parts = []
    if lines[False]:
        parts.append(
            "Add these to _ORG_KEPT in hub/modelscope/upstream.py:\n" + "\n".join(lines[False])
        )
    if lines[True]:
        parts.append(
            "Point these _ALIASES entries at a matching copy, or move them to _ORG_KEPT:\n"
            + "\n".join(lines[True])
        )
    return "\n".join(parts)


def test_mismatch_follows_the_file_rule():
    hf = {
        "model.safetensors": (9, "a"),
        "vocab.txt": (3, ""),
        "README.md": (1, ""),
        "LICENSE": (1, ""),
    }
    assert (
        mismatch(hf, {"model.safetensors": (9, "a"), "vocab.txt": (3, ""), "notes.md": (2, "")})
        == []
    )
    assert mismatch(hf, {"model.safetensors": (9, "b"), "x.py": (1, "")}) == [
        "differs model.safetensors",
        "missing vocab.txt",
        "extra x.py",
    ]
    assert mismatch({"a.bin": (5, "")}, {"a.bin": (6, "s")}) == ["differs a.bin"]
    bookkeeping = [
        "gitattributes",
        "sub/.gitattributes",
        "configuration.json",
        ".cache/huggingface/x.lock",
        "img/a.PNG",
        "uv.lock",
        "md5",
        "ckpt/.zarray",
        "ckpt/27.0",
        "train.log",
    ]
    assert mismatch({"a.bin": (1, "")}, dict.fromkeys(["a.bin", *bookkeeping], (1, ""))) == []
    assert mismatch({"a.bin": (1, "")}, {"a.bin": (1, ""), "sub/configuration.json": (1, "")}) == [
        "extra sub/configuration.json"
    ]
    assert mismatch({"README.md": (1, ""), ".hidden/config.json": (1, "")}, {}) == []
    assert mismatch({"vae.sft": (1, "a")}, {"vae.sft": (1, "b")}) == ["differs vae.sft"]
    assert mismatch({"merges.txt": (1, ""), "cfg/model.yaml": (1, "")}, {}) == [
        "missing merges.txt",
        "missing cfg/model.yaml",
    ]
    assert (
        mismatch(
            {"a.bin": (1, ""), "README": (1, ""), "docs/README_CN.rst": (1, "")}, {"a.bin": (1, "")}
        )
        == []
    )
    assert mismatch({"a.bin": (1, ""), "README.gguf": (1, "")}, {"a.bin": (1, "")}) == [
        "missing README.gguf"
    ]


def test_the_hugging_face_token_never_reaches_modelscope(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    seen = {}

    def answer(request):
        seen[request.url.host] = request.headers.get("authorization")
        return httpx.Response(404)

    with httpx.Client(transport = httpx.MockTransport(answer)) as http:
        assert _ms_files(http, "a/b") is None
        assert _hf_files(http, "a/b") is None
    assert seen == {"www.modelscope.cn": None, "huggingface.co": "Bearer t"}


def test_a_failing_exact_alias_is_reported_as_an_alias_to_fix():
    report = remedy({"tencent/Hy3-preview": "x", "nvidia/New-Model": "y"})
    assert report == (
        'Add these to _ORG_KEPT in hub/modelscope/upstream.py:\n    "nvidia/New-Model",  # y\n'
        "Point these _ALIASES entries at a matching copy, or move them to _ORG_KEPT:\n"
        '    "tencent/Hy3-preview",  # x'
    )


@pytest.mark.allow_network
@pytest.mark.skipif(
    os.environ.get("UNSLOTH_MODELSCOPE_ORG_ALIAS_NETWORK") != "1",
    reason = "set UNSLOTH_MODELSCOPE_ORG_ALIAS_NETWORK=1 to check the org aliases against ModelScope",
)
def test_every_org_aliased_model_matches_its_modelscope_copy():
    failures, inconclusive, checked = org_alias_failures(upstream._ORG_ALIASES)
    assert not failures, remedy(failures)
    assert (
        len(inconclusive) * 20 <= checked
    ), f"{len(inconclusive)} of {checked} models could not be checked: {inconclusive[:20]}"
