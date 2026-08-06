# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load-path parity for /api/models/gguf-variants local resolution."""

import asyncio
import os

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
    # Says it resolved locally, so the CLI gate can match it against the local resolver's labels.
    assert response.resolved_locally is True


def test_direct_gguf_file_is_a_loadable_variant(in_tmp_cwd):
    gguf = in_tmp_cwd / "foo-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")

    response = _variants(os.fspath(gguf))
    assert [v.filename for v in response.variants] == ["foo-Q4_K_M.gguf"]
    assert response.variants[0].quant == "Q4_K_M"
    # The file is the model; the shard scan's empty answer must not mark the only row
    # partial (the picker disables partial local rows).
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
    # The load resolves a .gguf in a marked directory to the whole directory, so the
    # listing keeps sibling quants and the vision flag.
    (in_tmp_cwd / "config.json").write_text("{}")
    (in_tmp_cwd / "model-Q4_K_M.gguf").write_bytes(b"GGUF")
    (in_tmp_cwd / "model-Q8_0.gguf").write_bytes(b"GGUF" * 2)
    (in_tmp_cwd / "mmproj-F16.gguf").write_bytes(b"GGUF")

    response = _variants(os.fspath(in_tmp_cwd / "model-Q4_K_M.gguf"))
    assert sorted(v.quant for v in response.variants) == ["Q4_K_M", "Q8_0"]
    assert response.has_vision is True
    # A marked parent is still scanned for completeness.
    assert all(v.downloaded for v in response.variants)


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
    # detect_gguf_model refuses companions and big-endian builds; a row for one would
    # offer a load that cannot happen.
    from utils.models.model_config import detect_gguf_model

    target = in_tmp_cwd / relpath
    target.parent.mkdir(parents = True, exist_ok = True)
    target.write_bytes(b"GGUF")

    assert detect_gguf_model(os.fspath(target)) is None
    assert _variants(os.fspath(target)).variants == []


def test_direct_gguf_file_quant_round_trips_through_the_load_path(in_tmp_cwd):
    # Clients echo the selected quant back as gguf_variant, so the advertised quant must
    # resolve for the same identifier, or the file loads without it and fails.
    from utils.models.model_config import ModelConfig

    gguf = in_tmp_cwd / "foo-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")

    quant = _variants(os.fspath(gguf)).variants[0].quant
    config = ModelConfig.from_identifier(os.fspath(gguf), gguf_variant = quant)
    assert config is not None and config.is_gguf
    assert config.gguf_file == os.fspath(gguf)
    # from_identifier only consults the quant for a directory; a direct file loads itself
    # regardless, so the listing must not be stricter.
    assert ModelConfig.from_identifier(os.fspath(gguf), gguf_variant = "Q8_0").is_gguf is True
    assert _variants(os.fspath(gguf)).loadable_variants is None


def test_direct_gguf_file_quant_round_trips_case_insensitively(in_tmp_cwd):
    # llama.cpp (and the CLI's pre-load gate) matches quant labels case-insensitively, so
    # a lowercase --gguf-variant must resolve here too, or the load evicts the resident
    # model on the transformers path before failing.
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
    # The two extractors disagree on shapes like F16-checkpoint-Q4_K_M; the advertised
    # quant must be the one the echoed load resolves.
    from utils.models.model_config import ModelConfig

    gguf = in_tmp_cwd / "F16-checkpoint-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")

    quant = _variants(os.fspath(gguf)).variants[0].quant
    config = ModelConfig.from_identifier(os.fspath(gguf), gguf_variant = quant)
    assert config is not None and config.is_gguf
    assert config.gguf_file == os.fspath(gguf)


def test_marked_dir_resolves_the_bpw_stripped_label(in_tmp_cwd):
    # Listings advertise the hub-style stripped spelling; the directory
    # resolver accepts it like the direct-file resolver does.
    from utils.models.model_config import ModelConfig

    marked = in_tmp_cwd / "m"
    marked.mkdir()
    (marked / "config.json").write_text("{}")
    (marked / "model-IQ4_XS-3.53bpw.gguf").write_bytes(b"GGUF")

    config = ModelConfig.from_identifier(os.fspath(marked), gguf_variant = "IQ4_XS")
    assert config is not None and config.is_gguf


def test_direct_big_endian_check_uses_the_load_extractor(in_tmp_cwd):
    # detect_gguf_model refuses this shape (its extractor reads F16 before the
    # be marker), so the endpoint must not advertise it as loadable.
    from utils.models.model_config import detect_gguf_model

    gguf = in_tmp_cwd / "F16-be-checkpoint-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")

    assert detect_gguf_model(os.fspath(gguf)) is None
    assert _variants(os.fspath(gguf)).variants == []


def test_variantless_pick_prefers_a_complete_candidate(in_tmp_cwd):
    # detect_gguf_model sorts by size, and a torn split's lone shard can be
    # the largest file; the load must prefer a candidate it can actually open.
    from utils.models.model_config import detect_gguf_model

    (in_tmp_cwd / "config.json").write_text("{}")
    (in_tmp_cwd / "model-Q8_0.gguf").write_bytes(b"GGUF")
    (in_tmp_cwd / "model-Q4_K_M-00001-of-00002.gguf").write_bytes(b"GGUF" * 100)

    assert detect_gguf_model(os.fspath(in_tmp_cwd)).endswith("model-Q8_0.gguf")


def test_parent_quant_does_not_resolve_a_different_basename_quant(in_tmp_cwd):
    # Q8_0/model-Q4_K_M.gguf IS the Q4_K_M file; matching it for Q8_0 would
    # serve the wrong weights under the requested name.
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
    # Clients can load the advertised default directly, so the label the
    # listing recommends has to resolve for the same loose file.
    from utils.models.model_config import ModelConfig

    gguf = in_tmp_cwd / "F16-checkpoint-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")

    response = _variants(os.fspath(gguf))
    config = ModelConfig.from_identifier(os.fspath(gguf), gguf_variant = response.default_variant)
    assert config is not None and config.is_gguf
    assert config.gguf_file == os.fspath(gguf)


def test_dir_resolver_accepts_the_advertised_hub_style_label(in_tmp_cwd):
    # The listing labels F16-checkpoint-Q4_K_M.gguf as Q4_K_M; echoing that
    # label back must resolve the same file.
    from utils.models.model_config import ModelConfig

    marked = in_tmp_cwd / "m"
    marked.mkdir()
    (marked / "config.json").write_text("{}")
    (marked / "F16-checkpoint-Q4_K_M.gguf").write_bytes(b"GGUF")

    quant = _variants(os.fspath(marked)).variants[0].quant
    config = ModelConfig.from_identifier(os.fspath(marked), gguf_variant = quant)
    assert config is not None and config.is_gguf


def test_local_listing_filters_what_the_local_detector_refuses(in_tmp_cwd):
    # The local detector reads F16 before the be marker and refuses the file;
    # the listing must not advertise a row the load cannot serve.
    from utils.models.model_config import detect_gguf_model

    (in_tmp_cwd / "config.json").write_text("{}")
    (in_tmp_cwd / "F16-be-checkpoint-Q4_K_M.gguf").write_bytes(b"GGUF")

    assert detect_gguf_model(os.fspath(in_tmp_cwd)) is None
    assert _variants(os.fspath(in_tmp_cwd)).variants == []


def test_short_shard_like_name_is_not_a_torn_split(in_tmp_cwd):
    # Five digits exactly is the load path's split grammar; a -001-of-002 file
    # loads on its own, so it must be offered as ready.
    lone = in_tmp_cwd / "model-Q4_K_M-001-of-002.gguf"
    lone.write_bytes(b"GGUF")

    row = _variants(os.fspath(lone)).variants[0]
    assert row.downloaded is True and row.partial is False


def test_remote_listing_filters_what_the_remote_detector_refuses(monkeypatch):
    # The remote detector extracts F16 and refuses the be marker; the listing
    # must not advertise a row for the same sibling.
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
    # The hub-side extractor drops the bpw modifier, so the advertised quant is
    # the shorter label; echoing it back must still resolve this same file.
    from utils.models.model_config import ModelConfig

    gguf = in_tmp_cwd / "model-IQ4_XS-3.53bpw.gguf"
    gguf.write_bytes(b"GGUF")

    quant = _variants(os.fspath(gguf)).variants[0].quant
    config = ModelConfig.from_identifier(os.fspath(gguf), gguf_variant = quant)
    assert config is not None and config.is_gguf
    assert config.gguf_file == os.fspath(gguf)


def test_torn_direct_split_is_not_offered_as_downloaded(in_tmp_cwd):
    # llama.cpp resolves a split's siblings from the main shard's directory, so a lone
    # shard fails after teardown; the direct-file fallback must not call it ready.
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
    # The gate's real question, answered by the loader plus the two ways a
    # resolvable path still fails llama-server: empty bytes and a torn split.
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

    # Weights only under a quant subdirectory: the variantless pick finds
    # nothing, but naming the quant resolves them.
    nested = _mk("nested", ("BF16/model.gguf", b"GGUF"))
    assert nested.loadable is False and "BF16" in nested.loadable_variants

    # Every spelling the resolver accepts is listed, so a client echoing the
    # advertised default is never rejected by a label the answer omitted.
    aliased = _mk("aliased", ("F16-checkpoint-Q4_K_M.gguf", b"GGUF"))
    assert aliased.default_variant in aliased.loadable_variants


def test_symlink_outside_the_tree_keeps_its_relative_alias(in_tmp_cwd):
    # The resolver reads the snapshot-relative spelling, so a link pointing out
    # of the tree still answers BF16/model; resolving it first loses that.
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
    # The resolver answers for paths that do not exist (the extension is
    # authoritative), so absence has to be caught before the gate trusts it.
    response = _variants(os.fspath(in_tmp_cwd / "typo-Q4_K_M.gguf"))
    assert response.loadable is False

    # A file that is there still serves, and a direct file leaves the label
    # list unanswered because the load ignores the quant for one.
    real = in_tmp_cwd / "real-Q8_0.gguf"
    real.write_bytes(b"GGUF")
    served = _variants(os.fspath(real))
    assert served.loadable is True and served.loadable_variants is None


def test_relative_identifiers_keep_their_relative_alias(in_tmp_cwd):
    # The resolver returns an absolute path, so a relative identifier has to be
    # resolved the same way or the relative spelling is lost from the answer.
    from utils.models.model_config import _find_local_gguf_by_variant

    (in_tmp_cwd / "models" / "qwen" / "BF16").mkdir(parents = True)
    (in_tmp_cwd / "models" / "qwen" / "config.json").write_text("{}")
    (in_tmp_cwd / "models" / "qwen" / "BF16" / "model.gguf").write_bytes(b"GGUF")

    offered = _variants("models/qwen").loadable_variants
    assert "BF16/model" in offered
    for spelling in offered:
        assert _find_local_gguf_by_variant("models/qwen", spelling)


def test_loadable_variants_include_the_relative_fallback_label(in_tmp_cwd):
    # The resolver accepts the snapshot-relative stem, so a client sending that
    # spelling must not be rejected by an answer that omitted it.
    from utils.models.model_config import _find_local_gguf_by_variant

    (in_tmp_cwd / "config.json").write_text("{}")
    (in_tmp_cwd / "BF16").mkdir()
    (in_tmp_cwd / "BF16" / "model.gguf").write_bytes(b"GGUF")

    offered = _variants(os.fspath(in_tmp_cwd)).loadable_variants
    assert "BF16/model" in offered
    for spelling in offered:
        assert _find_local_gguf_by_variant(os.fspath(in_tmp_cwd), spelling)


def test_a_torn_split_keeps_its_quant_partial(in_tmp_cwd):
    # A short shard-like name is ready on its own, but not when the file the
    # resolver binds for that quant is an earlier torn five-digit split.
    (in_tmp_cwd / "config.json").write_text("{}")
    (in_tmp_cwd / "a-Q4_K_M-00001-of-00002.gguf").write_bytes(b"GGUF")
    (in_tmp_cwd / "z-Q4_K_M-001-of-002.gguf").write_bytes(b"GGUF")

    response = _variants(os.fspath(in_tmp_cwd))
    assert response.variants[0].partial is True
    assert response.loadable_variants == []


def test_short_shard_like_name_in_a_directory_reads_ready(in_tmp_cwd):
    # The cache scan's looser split grammar would call this a torn set, but the
    # load treats a -001-of-002 name as an ordinary file and opens it.
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
    # The label comes from the snapshot-relative path, so a quant named by the
    # parent directory is honored; a zero-byte file stays partial either way.
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
    # The target declares its own grammar and total; the load launches that
    # set, so a torn target is torn however the alias is spelled.
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
    # A shard symlink whose target set is complete is loadable, so it must stay
    # a candidate rather than being dropped for another quant.
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
    # The load follows the link and launches the ordinary target, so the
    # alias's split-shaped name completes nothing.
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
    # _local_gguf_load_path names the siblings from the target, so an alias
    # whose stem differs still loads the real set.
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
    # The load resolves a symlinked shard to its target's colocated set, so a
    # link beside no siblings is still ready when the target set is whole.
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
    # Completeness is the declared index set, not a file count: a stray
    # 00003-of-00002 must not stand in for the missing shard 2.
    shard = in_tmp_cwd / "model-Q4_K_M-00001-of-00002.gguf"
    shard.write_bytes(b"GGUF")
    (in_tmp_cwd / "model-Q4_K_M-00003-of-00002.gguf").write_bytes(b"GGUF")

    row = _variants(os.fspath(shard)).variants[0]
    assert row.downloaded is False and row.partial is True


def test_zero_byte_split_sibling_does_not_complete_a_split(in_tmp_cwd):
    # The directory scan marks a torn split partial when a sibling is an empty
    # interrupted copy; the name alone must not count as the shard.
    shard = in_tmp_cwd / "model-Q4_K_M-00001-of-00002.gguf"
    shard.write_bytes(b"GGUF")
    (in_tmp_cwd / "model-Q4_K_M-00002-of-00002.gguf").write_bytes(b"")

    row = _variants(os.fspath(shard)).variants[0]
    assert row.downloaded is False and row.partial is True


def test_zero_byte_direct_gguf_is_partial(in_tmp_cwd):
    # The directory scan treats an empty gguf as incomplete (an interrupted
    # copy), so the direct-file fallback must not call the same bytes ready.
    empty = in_tmp_cwd / "foo-Q4_K_M.gguf"
    empty.write_bytes(b"")

    row = _variants(os.fspath(empty)).variants[0]
    assert row.downloaded is False and row.partial is True


def test_local_dir_answer_ignores_the_hub_cache_of_the_same_name(in_tmp_cwd, monkeypatch):
    # A repo-shaped id that exists as a directory is resolved existence-first, so this
    # answer must not gain rows from the HF cache of the identically named repo -- the
    # CLI's attach gate reads any row as a GGUF model and would evict the resident one.
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
