# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load-path parity for /api/models/gguf-variants local resolution."""

import asyncio
import os
from pathlib import Path

import pytest

from hub.services.models.gguf_variants import get_gguf_variants_response


def _variants(repo_id: str, **kwargs):
    return asyncio.run(get_gguf_variants_response(repo_id, **kwargs))


@pytest.fixture()
def in_tmp_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_markerless_relative_dir_resolves_locally(in_tmp_cwd):
    gguf_dir = in_tmp_cwd / "models" / "qwen"
    gguf_dir.mkdir(parents = True)
    (gguf_dir / "qwen-Q4_K_M.gguf").write_bytes(b"GGUF")

    response = _variants("models/qwen")
    assert [v.quant for v in response.variants] == ["Q4_K_M"]
    # Lets the CLI gate match against the local resolver's labels.
    assert response.resolved_locally is True


def test_direct_gguf_file_is_a_loadable_variant(in_tmp_cwd):
    gguf = in_tmp_cwd / "foo-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")

    response = _variants(os.fspath(gguf))
    assert [v.filename for v in response.variants] == ["foo-Q4_K_M.gguf"]
    assert response.variants[0].quant == "Q4_K_M"
    # The file is the model; the shard scan's empty answer must not mark the only row partial.
    assert response.variants[0].downloaded is True
    assert response.variants[0].partial is False


def test_markerless_relative_gguf_file_resolves_locally(in_tmp_cwd):
    (in_tmp_cwd / "models").mkdir()
    (in_tmp_cwd / "models" / "foo.gguf").write_bytes(b"GGUF")

    response = _variants("models/foo.gguf")
    assert [v.filename for v in response.variants] == ["foo.gguf"]


def test_nonexistent_local_syntax_path_still_returns_empty(in_tmp_cwd):
    response = _variants(os.fspath(in_tmp_cwd / "missing-dir"))
    assert response.variants == []


def test_direct_gguf_file_in_marked_dir_still_lists_siblings(in_tmp_cwd):
    # The load resolves a .gguf in a marked directory to the directory, so the listing keeps
    # sibling quants and the vision flag.
    (in_tmp_cwd / "config.json").write_text("{}")
    (in_tmp_cwd / "model-Q4_K_M.gguf").write_bytes(b"GGUF")
    (in_tmp_cwd / "model-Q8_0.gguf").write_bytes(b"GGUF" * 2)
    (in_tmp_cwd / "mmproj-F16.gguf").write_bytes(b"GGUF")

    response = _variants(os.fspath(in_tmp_cwd / "model-Q4_K_M.gguf"))
    assert sorted(v.quant for v in response.variants) == ["Q4_K_M", "Q8_0"]
    assert response.has_vision is True
    # A marked parent is still scanned for completeness.
    assert all(v.downloaded for v in response.variants)


def test_an_audio_only_projector_does_not_flag_vision(in_tmp_cwd):
    """The picker badge reads this flag, and ultravox / Voxtral / Qwen3-ASR ship a
    projector for audio input only."""
    import struct

    (in_tmp_cwd / "config.json").write_text("{}")
    (in_tmp_cwd / "model-Q4_K_M.gguf").write_bytes(b"GGUF")
    key = "clip.has_audio_encoder"
    (in_tmp_cwd / "mmproj-F16.gguf").write_bytes(
        struct.pack("<IIQQ", 0x46554747, 3, 0, 1)
        + struct.pack("<Q", len(key))
        + key.encode()
        + struct.pack("<I", 7)
        + struct.pack("<?", True)
    )

    response = _variants(os.fspath(in_tmp_cwd / "model-Q4_K_M.gguf"))
    assert [v.quant for v in response.variants] == ["Q4_K_M"]
    assert response.has_vision is False


def test_online_only_projector_is_not_opened(in_tmp_cwd, monkeypatch):
    """Projector discovery skips content only when metadata marks it unhydrated."""
    from hub.utils import gguf

    (in_tmp_cwd / "config.json").write_text("{}")
    weight = in_tmp_cwd / "model-Q4_K_M.gguf"
    weight.write_bytes(b"GGUF")
    projector = in_tmp_cwd / "mmproj-F16.gguf"
    projector.write_bytes(b"online-only placeholder")

    monkeypatch.setattr(
        gguf,
        "file_contents_available_locally",
        lambda path, stat_result = None: Path(path) != projector,
    )

    def forbidden(_path):
        raise AssertionError("online-only projector was opened")

    monkeypatch.setattr(gguf, "mmproj_accepts_image", forbidden)
    response = _variants(os.fspath(weight))

    assert [variant.quant for variant in response.variants] == ["Q4_K_M"]
    assert response.has_vision is False


@pytest.mark.parametrize(
    "relpath",
    [
        "mmproj-F16.gguf",
        "mtp-model-Q4_K_M.gguf",
        "MTP/model-Q8_0-MTP.gguf",
        "dspark/dspark-model-Q8_0.gguf",
        "stories260K-be.gguf",
    ],
)
def test_direct_auxiliary_gguf_file_is_not_a_variant(in_tmp_cwd, relpath):
    # detect_gguf_model refuses companions and big-endian builds, so a row would offer a
    # load that cannot happen.
    from utils.models.model_config import detect_gguf_model

    target = in_tmp_cwd / relpath
    target.parent.mkdir(parents = True, exist_ok = True)
    target.write_bytes(b"GGUF")

    assert detect_gguf_model(os.fspath(target)) is None
    assert _variants(os.fspath(target)).variants == []


def test_direct_gguf_file_quant_round_trips_through_the_load_path(in_tmp_cwd):
    # Clients echo the quant back as gguf_variant, so it must resolve for the same identifier.
    from utils.models.model_config import ModelConfig

    gguf = in_tmp_cwd / "foo-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")

    quant = _variants(os.fspath(gguf)).variants[0].quant
    config = ModelConfig.from_identifier(os.fspath(gguf), gguf_variant = quant)
    assert config is not None and config.is_gguf
    assert config.gguf_file == os.fspath(gguf)
    # from_identifier consults the quant only for a directory, so the listing must not be stricter.
    assert ModelConfig.from_identifier(os.fspath(gguf), gguf_variant = "Q8_0").is_gguf is True
    assert _variants(os.fspath(gguf)).loadable_variants is None


def test_direct_gguf_file_quant_round_trips_case_insensitively(in_tmp_cwd):
    # llama.cpp matches quant labels case-insensitively, so a lowercase --gguf-variant must
    # resolve here too, or the load evicts the resident model before failing.
    from utils.models.model_config import ModelConfig

    gguf = in_tmp_cwd / "foo-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")

    config = ModelConfig.from_identifier(os.fspath(gguf), gguf_variant = "q4_k_m")
    assert config is not None and config.is_gguf
    assert config.gguf_file == os.fspath(gguf)
    # A directory of the same weights answers the same spelling.
    marked = in_tmp_cwd / "marked"
    marked.mkdir()
    (marked / "config.json").write_text("{}")
    (marked / "foo-Q4_K_M.gguf").write_bytes(b"GGUF")
    dir_config = ModelConfig.from_identifier(os.fspath(marked), gguf_variant = "q4_k_m")
    assert dir_config is not None and dir_config.is_gguf


def test_direct_gguf_label_is_the_load_resolvers_label(in_tmp_cwd):
    # The extractors disagree on F16-checkpoint-Q4_K_M; the advertised quant must be the
    # one the echoed load resolves.
    from utils.models.model_config import ModelConfig

    gguf = in_tmp_cwd / "F16-checkpoint-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")

    quant = _variants(os.fspath(gguf)).variants[0].quant
    config = ModelConfig.from_identifier(os.fspath(gguf), gguf_variant = quant)
    assert config is not None and config.is_gguf
    assert config.gguf_file == os.fspath(gguf)


def test_marked_dir_resolves_the_bpw_stripped_label(in_tmp_cwd):
    # Listings advertise the hub-style stripped spelling, which the directory resolver takes too.
    from utils.models.model_config import ModelConfig

    marked = in_tmp_cwd / "m"
    marked.mkdir()
    (marked / "config.json").write_text("{}")
    (marked / "model-IQ4_XS-3.53bpw.gguf").write_bytes(b"GGUF")

    config = ModelConfig.from_identifier(os.fspath(marked), gguf_variant = "IQ4_XS")
    assert config is not None and config.is_gguf


def test_direct_big_endian_check_uses_the_load_extractor(in_tmp_cwd):
    # detect_gguf_model refuses this shape (F16 reads before the be marker), so it must
    # not be advertised as loadable.
    from utils.models.model_config import detect_gguf_model

    gguf = in_tmp_cwd / "F16-be-checkpoint-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")

    assert detect_gguf_model(os.fspath(gguf)) is None
    assert _variants(os.fspath(gguf)).variants == []


def test_variantless_pick_prefers_a_complete_candidate(in_tmp_cwd):
    # detect_gguf_model sorts by size and a torn split's lone shard can be the largest, so
    # the load must prefer a candidate it can open.
    from utils.models.model_config import detect_gguf_model

    (in_tmp_cwd / "config.json").write_text("{}")
    (in_tmp_cwd / "model-Q8_0.gguf").write_bytes(b"GGUF")
    (in_tmp_cwd / "model-Q4_K_M-00001-of-00002.gguf").write_bytes(b"GGUF" * 100)

    assert detect_gguf_model(os.fspath(in_tmp_cwd)).endswith("model-Q8_0.gguf")


def test_parent_quant_does_not_resolve_a_different_basename_quant(in_tmp_cwd):
    # Q8_0/model-Q4_K_M.gguf IS the Q4_K_M file; matching it for Q8_0 serves wrong weights.
    from utils.models.model_config import ModelConfig, _find_local_gguf_by_variant

    marked = in_tmp_cwd / "m"
    (marked / "Q8_0").mkdir(parents = True)
    (marked / "config.json").write_text("{}")
    (marked / "Q8_0" / "model-Q4_K_M.gguf").write_bytes(b"GGUF")

    assert _find_local_gguf_by_variant(os.fspath(marked), "Q8_0") is None
    assert ModelConfig.from_identifier(os.fspath(marked), gguf_variant = "Q8_0").is_gguf is False
    # The file's own label still resolves it.
    assert _find_local_gguf_by_variant(os.fspath(marked), "Q4_K_M") is not None


def test_direct_file_default_variant_resolves(in_tmp_cwd):
    # Clients load the advertised default directly, so it must resolve for the same file.
    from utils.models.model_config import ModelConfig

    gguf = in_tmp_cwd / "F16-checkpoint-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")

    response = _variants(os.fspath(gguf))
    config = ModelConfig.from_identifier(os.fspath(gguf), gguf_variant = response.default_variant)
    assert config is not None and config.is_gguf
    assert config.gguf_file == os.fspath(gguf)


def test_dir_resolver_accepts_the_advertised_hub_style_label(in_tmp_cwd):
    # The listing labels F16-checkpoint-Q4_K_M.gguf as Q4_K_M; echoing it back must resolve it.
    from utils.models.model_config import ModelConfig

    marked = in_tmp_cwd / "m"
    marked.mkdir()
    (marked / "config.json").write_text("{}")
    (marked / "F16-checkpoint-Q4_K_M.gguf").write_bytes(b"GGUF")

    quant = _variants(os.fspath(marked)).variants[0].quant
    config = ModelConfig.from_identifier(os.fspath(marked), gguf_variant = quant)
    assert config is not None and config.is_gguf


def test_local_listing_filters_what_the_local_detector_refuses(in_tmp_cwd):
    # The local detector reads F16 before the be marker and refuses the file, so the
    # listing must not advertise a row for it.
    from utils.models.model_config import detect_gguf_model

    (in_tmp_cwd / "config.json").write_text("{}")
    (in_tmp_cwd / "F16-be-checkpoint-Q4_K_M.gguf").write_bytes(b"GGUF")

    assert detect_gguf_model(os.fspath(in_tmp_cwd)) is None
    assert _variants(os.fspath(in_tmp_cwd)).variants == []


def test_short_shard_like_name_is_not_a_torn_split(in_tmp_cwd):
    # The split grammar is five digits exactly, so a -001-of-002 file loads on its own.
    lone = in_tmp_cwd / "model-Q4_K_M-001-of-002.gguf"
    lone.write_bytes(b"GGUF")

    row = _variants(os.fspath(lone)).variants[0]
    assert row.downloaded is True and row.partial is False


def test_remote_listing_filters_what_the_remote_detector_refuses(monkeypatch):
    # The remote detector extracts F16 and refuses the be marker, so no row for that sibling.
    from types import SimpleNamespace

    from hub.utils.gguf import list_gguf_variants

    info = SimpleNamespace(
        siblings = [
            SimpleNamespace(rfilename = "F16-be-checkpoint-Q4_K_M.gguf", size = 100),
            SimpleNamespace(rfilename = "model-Q8_0.gguf", size = 10),
        ]
    )
    api = SimpleNamespace(model_info = lambda *a, **k: info)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token = None: api)

    variants, _, _ = list_gguf_variants("owner/repo")
    assert [v.quant for v in variants] == ["Q8_0"]


def test_direct_gguf_bpw_label_round_trips_through_the_load_path(in_tmp_cwd):
    # The hub extractor drops the bpw modifier, so the shorter advertised label must still
    # resolve this file.
    from utils.models.model_config import ModelConfig

    gguf = in_tmp_cwd / "model-IQ4_XS-3.53bpw.gguf"
    gguf.write_bytes(b"GGUF")

    quant = _variants(os.fspath(gguf)).variants[0].quant
    config = ModelConfig.from_identifier(os.fspath(gguf), gguf_variant = quant)
    assert config is not None and config.is_gguf
    assert config.gguf_file == os.fspath(gguf)


def test_torn_direct_split_is_not_offered_as_downloaded(in_tmp_cwd):
    # llama.cpp resolves siblings from the main shard's directory, so a lone shard fails
    # after teardown and must not be called ready.
    shard = in_tmp_cwd / "model-Q4_K_M-00001-of-00002.gguf"
    shard.write_bytes(b"GGUF")

    row = _variants(os.fspath(shard)).variants[0]
    assert row.quant == "Q4_K_M"
    assert row.downloaded is False and row.partial is True

    # The whole set beside it is ready, and an unsplit file is untouched.
    (in_tmp_cwd / "model-Q4_K_M-00002-of-00002.gguf").write_bytes(b"GGUF")
    whole = _variants(os.fspath(shard)).variants[0]
    assert whole.downloaded is True and whole.partial is False


def test_local_answers_report_what_a_load_would_serve(in_tmp_cwd):
    # The gate's real question: the loader, plus the two ways a resolvable path still
    # fails llama-server (empty bytes, torn split).
    def _mk(name, *files):
        d = in_tmp_cwd / name
        d.mkdir()
        (d / "config.json").write_text("{}")
        for rel, data in files:
            target = d / rel
            target.parent.mkdir(parents = True, exist_ok = True)
            target.write_bytes(data)
        return _variants(os.fspath(d))

    ready = _mk("ready", ("m-Q4_K_M.gguf", b"GGUF"))
    assert ready.loadable is True and "Q4_K_M" in ready.loadable_variants

    empty = _mk("empty", ("m-Q4_K_M.gguf", b""))
    assert empty.loadable is False and empty.loadable_variants == []

    torn = _mk("torn", ("m-Q8_0-00001-of-00002.gguf", b"GGUF"))
    assert torn.loadable is False and torn.loadable_variants == []

    whole = _mk(
        "whole",
        ("m-Q8_0-00001-of-00002.gguf", b"GGUF"),
        ("m-Q8_0-00002-of-00002.gguf", b"GGUF"),
    )
    assert whole.loadable is True and "Q8_0" in whole.loadable_variants

    # Weights only under a quant subdirectory: variantless finds nothing, the quant resolves.
    nested = _mk("nested", ("BF16/model.gguf", b"GGUF"))
    assert nested.loadable is False and "BF16" in nested.loadable_variants

    # Every spelling the resolver accepts is listed, so echoing the default is never rejected.
    aliased = _mk("aliased", ("F16-checkpoint-Q4_K_M.gguf", b"GGUF"))
    assert aliased.default_variant in aliased.loadable_variants


def test_symlink_outside_the_tree_keeps_its_relative_alias(in_tmp_cwd):
    # The resolver reads the snapshot-relative spelling, so a link out of the tree still
    # answers BF16/model; resolving first loses that.
    from utils.models.model_config import _find_local_gguf_by_variant

    pool = in_tmp_cwd / "pool"
    pool.mkdir()
    (pool / "real.gguf").write_bytes(b"GGUF")
    model = in_tmp_cwd / "m"
    (model / "BF16").mkdir(parents = True)
    (model / "config.json").write_text("{}")
    (model / "BF16" / "model.gguf").symlink_to(pool / "real.gguf")

    offered = _variants(os.fspath(model)).loadable_variants
    assert "BF16/model" in offered
    for spelling in offered:
        assert _find_local_gguf_by_variant(os.fspath(model), spelling)


def test_a_missing_direct_path_is_not_loadable(in_tmp_cwd):
    # The extension is authoritative, so the resolver answers for paths that do not exist
    # and absence must be caught before the gate trusts it.
    response = _variants(os.fspath(in_tmp_cwd / "typo-Q4_K_M.gguf"))
    assert response.loadable is False

    # A present file still serves, and a direct file leaves the list unanswered because
    # the load ignores the quant for one.
    real = in_tmp_cwd / "real-Q8_0.gguf"
    real.write_bytes(b"GGUF")
    served = _variants(os.fspath(real))
    assert served.loadable is True and served.loadable_variants is None


def test_relative_identifiers_keep_their_relative_alias(in_tmp_cwd):
    # The resolver returns an absolute path, so a relative identifier must be resolved the
    # same way or its spelling is lost.
    from utils.models.model_config import _find_local_gguf_by_variant

    (in_tmp_cwd / "models" / "qwen" / "BF16").mkdir(parents = True)
    (in_tmp_cwd / "models" / "qwen" / "config.json").write_text("{}")
    (in_tmp_cwd / "models" / "qwen" / "BF16" / "model.gguf").write_bytes(b"GGUF")

    offered = _variants("models/qwen").loadable_variants
    assert "BF16/model" in offered
    for spelling in offered:
        assert _find_local_gguf_by_variant("models/qwen", spelling)


def test_loadable_variants_include_the_relative_fallback_label(in_tmp_cwd):
    # The resolver accepts the snapshot-relative stem, so the answer must not omit it.
    from utils.models.model_config import _find_local_gguf_by_variant

    (in_tmp_cwd / "config.json").write_text("{}")
    (in_tmp_cwd / "BF16").mkdir()
    (in_tmp_cwd / "BF16" / "model.gguf").write_bytes(b"GGUF")

    offered = _variants(os.fspath(in_tmp_cwd)).loadable_variants
    assert "BF16/model" in offered
    for spelling in offered:
        assert _find_local_gguf_by_variant(os.fspath(in_tmp_cwd), spelling)


def test_a_torn_split_keeps_its_quant_partial(in_tmp_cwd):
    # A short shard-like name is ready alone, but not when the file the resolver binds for
    # that quant is an earlier torn five-digit split.
    (in_tmp_cwd / "config.json").write_text("{}")
    (in_tmp_cwd / "a-Q4_K_M-00001-of-00002.gguf").write_bytes(b"GGUF")
    (in_tmp_cwd / "z-Q4_K_M-001-of-002.gguf").write_bytes(b"GGUF")

    response = _variants(os.fspath(in_tmp_cwd))
    assert response.variants[0].partial is True
    assert response.loadable_variants == []


def test_short_shard_like_name_in_a_directory_reads_ready(in_tmp_cwd):
    # The cache scan's looser grammar calls this torn, but the load opens a -001-of-002
    # name as an ordinary file.
    from utils.models.model_config import detect_gguf_model

    (in_tmp_cwd / "config.json").write_text("{}")
    (in_tmp_cwd / "model-Q4_K_M-001-of-002.gguf").write_bytes(b"GGUF")

    assert detect_gguf_model(os.fspath(in_tmp_cwd)) is not None
    row = _variants(os.fspath(in_tmp_cwd)).variants[0]
    assert row.downloaded is True and row.partial is False

    # A real five-digit split missing a shard is still partial.
    torn = in_tmp_cwd / "torn"
    torn.mkdir()
    (torn / "config.json").write_text("{}")
    (torn / "m-Q8_0-00001-of-00002.gguf").write_bytes(b"GGUF")
    torn_row = _variants(os.fspath(torn)).variants[0]
    assert torn_row.downloaded is False and torn_row.partial is True


def test_parent_quant_short_shard_reads_ready(in_tmp_cwd):
    # The label comes from the snapshot-relative path, so a parent directory's quant is
    # honored; a zero-byte file stays partial either way.
    (in_tmp_cwd / "config.json").write_text("{}")
    (in_tmp_cwd / "Q4_K_M").mkdir()
    (in_tmp_cwd / "Q4_K_M" / "model-001-of-002.gguf").write_bytes(b"GGUF")

    row = _variants(os.fspath(in_tmp_cwd)).variants[0]
    assert row.quant == "Q4_K_M"
    assert row.downloaded is True and row.partial is False

    empty = in_tmp_cwd / "empty"
    empty.mkdir()
    (empty / "config.json").write_text("{}")
    (empty / "model-Q8_0-001-of-002.gguf").write_bytes(b"")
    empty_row = _variants(os.fspath(empty)).variants[0]
    assert empty_row.downloaded is False and empty_row.partial is True


def test_symlinked_split_target_with_a_different_total_is_checked(in_tmp_cwd):
    # The load launches the set the target declares, so a torn target is torn however the
    # alias is spelled.
    real = in_tmp_cwd / "real"
    real.mkdir()
    (real / "m-Q4_K_M-00001-of-00003.gguf").write_bytes(b"GGUF")
    links = in_tmp_cwd / "links"
    links.mkdir()
    alias = links / "alias-Q4_K_M-00001-of-00002.gguf"
    alias.symlink_to(real / "m-Q4_K_M-00001-of-00003.gguf")

    torn = _variants(os.fspath(alias)).variants[0]
    assert torn.downloaded is False and torn.partial is True

    # Completing the target's own set makes it ready.
    (real / "m-Q4_K_M-00002-of-00003.gguf").write_bytes(b"GGUF")
    (real / "m-Q4_K_M-00003-of-00003.gguf").write_bytes(b"GGUF")
    whole = _variants(os.fspath(alias)).variants[0]
    assert whole.downloaded is True and whole.partial is False


def test_variantless_pick_keeps_a_symlinked_whole_split(in_tmp_cwd):
    # A shard symlink with a complete target set is loadable, so it stays a candidate.
    from utils.models.model_config import detect_gguf_model

    real = in_tmp_cwd / "real"
    real.mkdir()
    (real / "big-Q8_0-00001-of-00002.gguf").write_bytes(b"GGUF" * 100)
    (real / "big-Q8_0-00002-of-00002.gguf").write_bytes(b"GGUF" * 100)
    marked = in_tmp_cwd / "m"
    marked.mkdir()
    (marked / "config.json").write_text("{}")
    (marked / "small-Q4_K_M.gguf").write_bytes(b"GGUF")
    (marked / "big-Q8_0-00001-of-00002.gguf").symlink_to(real / "big-Q8_0-00001-of-00002.gguf")

    assert detect_gguf_model(os.fspath(marked)).endswith("big-Q8_0-00001-of-00002.gguf")


def test_split_named_symlink_to_a_plain_target_is_ready(in_tmp_cwd):
    # The load launches the ordinary target, so the alias's split-shaped name means nothing.
    real = in_tmp_cwd / "real-Q4_K_M.gguf"
    real.write_bytes(b"GGUF")
    alias = in_tmp_cwd / "alias-Q4_K_M-00001-of-00002.gguf"
    alias.symlink_to(real)

    row = _variants(os.fspath(alias)).variants[0]
    assert row.downloaded is True and row.partial is False


def test_high_count_split_is_still_checked(in_tmp_cwd):
    # A declared shard count above the old cap is a real split, not a pass.
    lone = in_tmp_cwd / "m-Q4_K_M-00001-of-01001.gguf"
    lone.write_bytes(b"GGUF")

    row = _variants(os.fspath(lone)).variants[0]
    assert row.downloaded is False and row.partial is True


def test_aliased_split_symlink_uses_the_target_name(in_tmp_cwd):
    # _local_gguf_load_path names siblings from the target, so a differing stem still loads it.
    real = in_tmp_cwd / "real"
    real.mkdir()
    (real / "model-Q4_K_M-00001-of-00002.gguf").write_bytes(b"GGUF")
    (real / "model-Q4_K_M-00002-of-00002.gguf").write_bytes(b"GGUF")
    links = in_tmp_cwd / "links"
    links.mkdir()
    alias = links / "alias-Q4_K_M-00001-of-00002.gguf"
    alias.symlink_to(real / "model-Q4_K_M-00001-of-00002.gguf")

    row = _variants(os.fspath(alias)).variants[0]
    assert row.downloaded is True and row.partial is False


def test_symlinked_split_follows_its_target_set(in_tmp_cwd):
    # The load resolves a symlinked shard to its target's set, so a link with no siblings
    # beside it is ready when that set is whole.
    real = in_tmp_cwd / "real"
    real.mkdir()
    (real / "m-Q4_K_M-00001-of-00002.gguf").write_bytes(b"GGUF")
    (real / "m-Q4_K_M-00002-of-00002.gguf").write_bytes(b"GGUF")
    links = in_tmp_cwd / "links"
    links.mkdir()
    link = links / "m-Q4_K_M-00001-of-00002.gguf"
    link.symlink_to(real / "m-Q4_K_M-00001-of-00002.gguf")

    row = _variants(os.fspath(link)).variants[0]
    assert row.downloaded is True and row.partial is False

    # A torn target set is still torn.
    (real / "m-Q4_K_M-00002-of-00002.gguf").unlink()
    torn = _variants(os.fspath(link)).variants[0]
    assert torn.downloaded is False and torn.partial is True


def test_stray_over_indexed_shard_does_not_complete_a_split(in_tmp_cwd):
    # Completeness is the declared index set, not a count: a stray 00003-of-00002 is not shard 2.
    shard = in_tmp_cwd / "model-Q4_K_M-00001-of-00002.gguf"
    shard.write_bytes(b"GGUF")
    (in_tmp_cwd / "model-Q4_K_M-00003-of-00002.gguf").write_bytes(b"GGUF")

    row = _variants(os.fspath(shard)).variants[0]
    assert row.downloaded is False and row.partial is True


def test_zero_byte_split_sibling_does_not_complete_a_split(in_tmp_cwd):
    # An empty sibling is an interrupted copy; the name alone must not count as the shard.
    shard = in_tmp_cwd / "model-Q4_K_M-00001-of-00002.gguf"
    shard.write_bytes(b"GGUF")
    (in_tmp_cwd / "model-Q4_K_M-00002-of-00002.gguf").write_bytes(b"")

    row = _variants(os.fspath(shard)).variants[0]
    assert row.downloaded is False and row.partial is True


def test_zero_byte_direct_gguf_is_partial(in_tmp_cwd):
    # The directory scan calls an empty gguf incomplete, so the direct-file fallback must too.
    empty = in_tmp_cwd / "foo-Q4_K_M.gguf"
    empty.write_bytes(b"")

    row = _variants(os.fspath(empty)).variants[0]
    assert row.downloaded is False and row.partial is True


def test_local_dir_answer_ignores_the_hub_cache_of_the_same_name(in_tmp_cwd, monkeypatch):
    # A repo-shaped id existing as a directory resolves existence-first, so it must not gain
    # rows from the same-named HF cache: the attach gate reads any row as a GGUF model.
    from types import SimpleNamespace

    hub_cache = in_tmp_cwd / "hub"
    (hub_cache / "models--unsloth--foo" / "snapshots" / "rev" / "Q4_K_M").mkdir(parents = True)
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(
            hub_cache = hub_cache,
            hf_home = in_tmp_cwd,
            source = "studio",
            cache_home = in_tmp_cwd,
        ),
    )
    (in_tmp_cwd / "unsloth" / "foo").mkdir(parents = True)
    (in_tmp_cwd / "unsloth" / "foo" / "config.json").write_text("{}")

    from utils.models.model_config import detect_gguf_model

    assert detect_gguf_model("unsloth/foo") is None
    assert _variants("unsloth/foo").variants == []


def test_wsl_drive_path_is_normalized_like_the_load(in_tmp_cwd, monkeypatch):
    # from_identifier normalizes first, so under WSL "C:\models\qwen" is served from the
    # mapped path and probing the raw spelling would call a working model unloadable.
    #
    # Which mapping is the live question: hub.utils.paths honours [automount] root while the
    # loader hardcodes /mnt, so on a custom-root host only the loader's answer predicts the
    # load. Its real root is unwritable from a test, so it is stood in for in the tmp tree
    # (the binding is pinned below) while the hub root points at a decoy of another quant.
    from hub.utils import paths as hub_paths
    from hub.services.models import gguf_variants
    from utils.paths import normalize_path as loader_normalize_path

    assert gguf_variants._loader_normalize_path is loader_normalize_path

    monkeypatch.setattr(hub_paths, "_IS_WSL", True)
    monkeypatch.setattr(hub_paths, "_WSL_AUTOMOUNT_ROOT", f"{in_tmp_cwd}/custom/")
    monkeypatch.setattr(
        gguf_variants,
        "_loader_normalize_path",
        lambda path: f"{in_tmp_cwd}/mnt/{path[0].lower()}/{path[3:].replace(chr(92), '/')}"
        if len(path) >= 3 and path[1] == ":"
        else path,
    )
    gguf_dir = in_tmp_cwd / "mnt" / "c" / "models" / "qwen"
    gguf_dir.mkdir(parents = True)
    (gguf_dir / "qwen-Q4_K_M.gguf").write_bytes(b"GGUF")
    decoy = in_tmp_cwd / "custom" / "c" / "models" / "qwen"
    decoy.mkdir(parents = True)
    (decoy / "qwen-Q8_0.gguf").write_bytes(b"GGUF")

    response = _variants(r"C:\models\qwen")

    assert response.resolved_locally is True
    assert [v.quant for v in response.variants] == ["Q4_K_M"]
    assert "Q4_K_M" in (response.loadable_variants or [])
    assert response.loadable is True


@pytest.mark.parametrize(
    "error, serves",
    [
        (FileNotFoundError(2, "No such file"), False),
        (PermissionError(13, "Permission denied"), True),
        (OSError(11, "Resource temporarily unavailable"), True),
    ],
)
def test_will_serve_only_treats_definite_absence_as_unloadable(
    in_tmp_cwd, monkeypatch, error, serves
):
    # Only "no such file" is definite; a read error (Windows sharing violation) must stay
    # unknown and serve. exists() cannot express that: on 3.14 it swallows every OSError.
    from hub.services.models import gguf_variants

    gguf = in_tmp_cwd / "foo-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")

    def raising_stat(self, *args, **kwargs):
        raise error

    monkeypatch.setattr(gguf_variants.Path, "stat", raising_stat)
    # Stand in for 3.14, where exists() swallows the error and calls all of these absent.
    monkeypatch.setattr(gguf_variants.Path, "exists", lambda self: False)
    assert gguf_variants._will_serve(os.fspath(gguf)) is serves


def test_loadable_variants_stays_unanswered_for_an_unstatable_direct_file(in_tmp_cwd, monkeypatch):
    # An empty list is authoritative at the attach gate, so a locked direct file must stay
    # unanswered: from_identifier ignores the variant for a file and loads it regardless.
    from hub.services.models import gguf_variants
    from hub.utils.gguf import list_local_gguf_variants

    gguf = in_tmp_cwd / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")
    variants, _ = list_local_gguf_variants(os.fspath(gguf))
    assert gguf_variants._loadable_variants(os.fspath(gguf), variants) is None

    def locked(self, *args, **kwargs):
        raise PermissionError(13, "sharing violation")

    monkeypatch.setattr(gguf_variants.Path, "stat", locked)
    # is_file() cannot express this: it raises here, and answers False from 3.14.
    monkeypatch.setattr(gguf_variants.Path, "is_file", lambda self: False)
    assert gguf_variants._loadable_variants(os.fspath(gguf), variants) is None
