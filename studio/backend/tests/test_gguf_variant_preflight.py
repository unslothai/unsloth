"""An explicit --gguf-variant that exists nowhere must be rejected by
ModelConfig.from_identifier, before the load path unloads the resident model."""

from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

import core.inference.llama_cpp as llama_cpp
import utils.hf_cache_settings as hf_cache_settings
import utils.models.model_config as mc
from utils.models.model_config import GgufVariantInfo, ModelConfig

REPO = "unsloth/Llama-3.2-1B-Instruct-GGUF"

REPO_FILES = [
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "Llama-3.2-1B-Instruct-Q8_0.gguf",
]

VARIANTS = [
    GgufVariantInfo(filename = "Llama-3.2-1B-Instruct-Q4_K_M.gguf", quant = "Q4_K_M", size_bytes = 100),
    GgufVariantInfo(filename = "Llama-3.2-1B-Instruct-Q8_0.gguf", quant = "Q8_0", size_bytes = 200),
]


@pytest.fixture
def remote_gguf_repo(monkeypatch):
    monkeypatch.setattr(
        mc, "detect_gguf_model_remote", lambda identifier, hf_token = None: VARIANTS[0].filename
    )
    monkeypatch.setattr(
        mc, "list_gguf_variants", lambda identifier, hf_token = None: (list(VARIANTS), False)
    )
    monkeypatch.setattr(
        huggingface_hub, "list_repo_files", lambda repo_id, token = None: list(REPO_FILES)
    )
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend,
        "_find_llama_server_binary",
        staticmethod(lambda *, include_denied = False: "/fake/llama-server"),
    )
    monkeypatch.setattr(llama_cpp, "cached_gguf_for_load", lambda repo, variant, **kwargs: None)


@pytest.fixture
def hub_cache(tmp_path, monkeypatch):
    """Point the active Hub cache at a temp root."""
    monkeypatch.setattr(
        hf_cache_settings, "get_hf_cache_paths", lambda: SimpleNamespace(hub_cache = tmp_path)
    )
    return tmp_path


def _cached(
    root: Path,
    name: str,
    payload: bytes = b"cached",
) -> Path:
    """Write a GGUF into a snapshot of ``REPO`` under the active Hub cache."""
    snapshot = root / f"models--{REPO.replace('/', '--')}" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents = True, exist_ok = True)
    path = snapshot / name
    path.write_bytes(payload)
    return path


def test_unknown_variant_rejected_before_load(remote_gguf_repo):
    with pytest.raises(ValueError) as exc_info:
        ModelConfig.from_identifier(REPO, gguf_variant = "NOPE_Q9")
    msg = str(exc_info.value)
    assert "NOPE_Q9" in msg
    assert "Q4_K_M" in msg and "Q8_0" in msg


def test_known_variant_accepted(remote_gguf_repo):
    config = ModelConfig.from_identifier(REPO, gguf_variant = "Q8_0")
    assert config is not None
    assert config.is_gguf
    assert config.gguf_variant == "Q8_0"


def test_variant_match_is_case_insensitive(remote_gguf_repo):
    config = ModelConfig.from_identifier(REPO, gguf_variant = "q4_k_m")
    assert config is not None
    assert config.gguf_variant == "q4_k_m"


def test_variant_only_in_verified_local_cache_accepted(remote_gguf_repo, monkeypatch, tmp_path):
    cached = tmp_path / "Llama-3.2-1B-Instruct-OLD_Q2.gguf"
    cached.write_bytes(b"cached")
    monkeypatch.setattr(
        llama_cpp,
        "cached_gguf_for_load",
        lambda repo, variant, **kwargs: str(cached),
    )
    config = ModelConfig.from_identifier(REPO, gguf_variant = "OLD_Q2")
    assert config is not None
    assert config.gguf_variant == "OLD_Q2"


def test_cache_escape_uses_size_verified_predicate(remote_gguf_repo, monkeypatch):
    calls = {}

    def fake_cached_gguf_for_load(repo, variant, **kwargs):
        calls["repo"] = repo
        calls["kwargs"] = kwargs
        return None

    monkeypatch.setattr(llama_cpp, "cached_gguf_for_load", fake_cached_gguf_for_load)
    with pytest.raises(ValueError):
        ModelConfig.from_identifier(REPO, gguf_variant = "OLD_Q2")
    assert calls["repo"] == REPO
    assert calls["kwargs"].get("verify_sizes") is True


def test_variant_matching_uncollapsed_repo_file_accepted(remote_gguf_repo, monkeypatch):
    # Same quant label across shards collapses to one GgufVariantInfo; the
    # preflight must still match against every repo file, like the load path.
    monkeypatch.setattr(
        huggingface_hub,
        "list_repo_files",
        lambda repo_id, token = None: [
            "Llama-3.2-1B-Instruct-Q4_K_M-00001-of-00002.gguf",
            "Llama-3.2-1B-Instruct-Q4_K_M-00002-of-00002.gguf",
        ],
    )
    config = ModelConfig.from_identifier(REPO, gguf_variant = "00002-of-00002")
    assert config is not None
    assert config.gguf_variant == "00002-of-00002"


def test_listing_unavailable_skips_check(remote_gguf_repo, monkeypatch):
    def boom(repo_id, token = None):
        raise ConnectionError("hub unreachable")

    monkeypatch.setattr(huggingface_hub, "list_repo_files", boom)
    config = ModelConfig.from_identifier(REPO, gguf_variant = "NOPE_Q9")
    assert config is not None
    assert config.gguf_variant == "NOPE_Q9"


def test_no_variant_still_autoselects(remote_gguf_repo):
    config = ModelConfig.from_identifier(REPO)
    assert config is not None
    assert config.gguf_variant in {"Q4_K_M", "Q8_0"}


def test_a_verified_cached_copy_is_carried_to_the_load(remote_gguf_repo, monkeypatch, hub_cache):
    """Config carries the cached file it already verified."""
    name = "Llama-3.2-1B-Instruct-Q8_0.gguf"
    cached = _cached(hub_cache, name)
    monkeypatch.setattr(llama_cpp, "cached_gguf_for_load", lambda repo, variant, **kw: str(cached))

    config = ModelConfig.from_identifier(REPO, gguf_variant = "Q8_0")
    assert config.gguf_verified == (
        REPO,
        "Q8_0",
        str(cached),
        ((name, cached.stat().st_size),),
    )


def test_a_verified_cached_copy_uses_a_projector_beside_it(
    remote_gguf_repo, monkeypatch, hub_cache
):
    """#9286: the repo publishes none, the user dropped one next to the weight, and
    this branch took the listing's word for it."""
    name = "Llama-3.2-1B-Instruct-Q8_0.gguf"
    cached = _cached(hub_cache, name)
    projector = cached.parent / "mmproj-F16.gguf"
    projector.write_bytes(b"mmproj")
    monkeypatch.setattr(llama_cpp, "cached_gguf_for_load", lambda repo, variant, **kw: str(cached))

    config = ModelConfig.from_identifier(REPO, gguf_variant = "Q8_0")

    assert config.is_vision is True
    # Carried for the training guard only: the launch resolves its own beside the weight.
    assert config.gguf_local_mmproj_file == str(projector.resolve())
    assert config.gguf_mmproj_file is None


def test_a_verified_cached_copy_uses_its_repo_root_projector(
    remote_gguf_repo, monkeypatch, hub_cache
):
    """The snapshot dir is a hex sha, so a user browsing to "the model folder" often
    stops at models--<repo> instead."""
    name = "Llama-3.2-1B-Instruct-Q8_0.gguf"
    cached = _cached(hub_cache, name)
    projector = cached.parent.parent.parent / "mmproj-F16.gguf"
    projector.write_bytes(b"mmproj")
    monkeypatch.setattr(llama_cpp, "cached_gguf_for_load", lambda repo, variant, **kw: str(cached))

    config = ModelConfig.from_identifier(REPO, gguf_variant = "Q8_0")

    assert config.is_vision is True
    assert config.gguf_local_mmproj_file == str(projector.resolve())
    assert config.gguf_mmproj_file is None


def test_audio_only_repo_root_projector_still_triggers_companion_loading(
    remote_gguf_repo, monkeypatch, hub_cache
):
    """load_model resolves a projector only when the config says vision, and the
    metadata that separates an audio encoder from an image tower is read later. So
    an audio-only file at the repo root has to flip the flag too."""
    name = "Llama-3.2-1B-Instruct-Q8_0.gguf"
    cached = _cached(hub_cache, name)
    projector = cached.parent.parent.parent / "mmproj-F16.gguf"
    projector.write_bytes(b"audio")
    monkeypatch.setattr(llama_cpp, "cached_gguf_for_load", lambda repo, variant, **kw: str(cached))
    monkeypatch.setattr(mc, "mmproj_accepts_image", lambda _path: False)

    config = ModelConfig.from_identifier(REPO, gguf_variant = "Q8_0")

    assert config.is_vision is True
    # The remote branch launches with -hf, so the projector stays the loader's to
    # resolve beside the weight it downloads.
    assert config.gguf_mmproj_file is None


def test_a_sibling_repos_projector_is_not_borrowed(remote_gguf_repo, monkeypatch, hub_cache):
    """The walk stops at the repo the weight came out of, so the cache root and every
    other repo in it stay invisible."""
    name = "Llama-3.2-1B-Instruct-Q8_0.gguf"
    cached = _cached(hub_cache, name)
    sibling = hub_cache / "models--someone--Other-GGUF"
    sibling.mkdir()
    (sibling / "mmproj-F16.gguf").write_bytes(b"mmproj")
    (hub_cache / "mmproj-F16.gguf").write_bytes(b"mmproj")
    monkeypatch.setattr(llama_cpp, "cached_gguf_for_load", lambda repo, variant, **kw: str(cached))

    config = ModelConfig.from_identifier(REPO, gguf_variant = "Q8_0")

    assert config.gguf_local_mmproj_file is None
    assert config.is_vision is False


def test_nothing_is_named_before_a_quant_of_the_repo_is_cached(
    remote_gguf_repo, monkeypatch, hub_cache
):
    """No weight to pair against yet. The first Apply downloads and launches without
    the hand-added projector; the Apply after it finds one."""
    repo_root = hub_cache / f"models--{REPO.replace('/', '--')}"
    repo_root.mkdir()
    (repo_root / "mmproj-F16.gguf").write_bytes(b"mmproj")
    monkeypatch.setattr(llama_cpp, "cached_gguf_for_load", lambda repo, variant, **kw: None)

    config = ModelConfig.from_identifier(REPO, gguf_variant = "Q8_0")

    assert config.gguf_verified is None
    assert config.gguf_local_mmproj_file is None
    assert config.is_vision is False


def test_a_published_projector_does_not_reach_the_local_lookup(
    remote_gguf_repo, monkeypatch, hub_cache
):
    """The listing already answered, so the widened walk never runs and a repo that
    ships one resolves exactly as it did before."""
    name = "Llama-3.2-1B-Instruct-Q8_0.gguf"
    cached = _cached(hub_cache, name)
    (cached.parent.parent.parent / "mmproj-F16.gguf").write_bytes(b"mmproj")
    monkeypatch.setattr(llama_cpp, "cached_gguf_for_load", lambda repo, variant, **kw: str(cached))
    monkeypatch.setattr(
        mc, "list_gguf_variants", lambda identifier, hf_token = None: (list(VARIANTS), True)
    )

    config = ModelConfig.from_identifier(REPO, gguf_variant = "Q8_0")

    assert config.is_vision is True
    assert config.gguf_local_mmproj_file is None


def test_every_shard_of_a_verified_cached_copy_is_measured(
    remote_gguf_repo, monkeypatch, hub_cache
):
    """A split copy carries all of its shards, not just the one the load opens."""
    shards = {
        f"Llama-3.2-1B-Instruct-Q8_0-0000{n}-of-00003.gguf": b"x" * (10 + n) for n in (1, 2, 3)
    }
    written = {name: _cached(hub_cache, name, payload) for name, payload in shards.items()}
    main = written["Llama-3.2-1B-Instruct-Q8_0-00001-of-00003.gguf"]
    monkeypatch.setattr(llama_cpp, "cached_gguf_for_load", lambda repo, variant, **kw: str(main))

    config = ModelConfig.from_identifier(REPO, gguf_variant = "Q8_0")
    assert config.gguf_verified == (
        REPO,
        "Q8_0",
        str(main),
        tuple(sorted((name, len(payload)) for name, payload in shards.items())),
    )


def test_a_cached_copy_outside_the_variant_set_is_not_carried(
    remote_gguf_repo, monkeypatch, hub_cache
):
    """Nothing is carried when the shard set cannot be derived for the path."""
    loose = hub_cache / "Llama-3.2-1B-Instruct-Q8_0.gguf"
    loose.write_bytes(b"cached")
    monkeypatch.setattr(llama_cpp, "cached_gguf_for_load", lambda repo, variant, **kw: str(loose))

    assert ModelConfig.from_identifier(REPO, gguf_variant = "Q8_0").gguf_verified is None


def test_a_verified_cached_copy_settles_the_variant_without_a_second_listing(
    remote_gguf_repo, monkeypatch, tmp_path
):
    """A verified cached file proves the requested variant exists."""
    listings = []
    cached = tmp_path / "old.gguf"
    cached.write_bytes(b"cached")

    def counting_list_repo_files(repo_id, token = None):
        listings.append(repo_id)
        return list(REPO_FILES)

    monkeypatch.setattr(huggingface_hub, "list_repo_files", counting_list_repo_files)
    monkeypatch.setattr(llama_cpp, "cached_gguf_for_load", lambda repo, variant, **kw: str(cached))

    config = ModelConfig.from_identifier(REPO, gguf_variant = "OLD_Q2")
    assert config.gguf_variant == "OLD_Q2"
    assert listings == [], "a verified cached file still cost a repo listing"

    # Nothing cached: the listing runs and still rejects a variant it does not name.
    listings.clear()
    monkeypatch.setattr(llama_cpp, "cached_gguf_for_load", lambda repo, variant, **kw: None)
    with pytest.raises(ValueError, match = "OLD_Q2"):
        ModelConfig.from_identifier(REPO, gguf_variant = "OLD_Q2")
    assert listings == [REPO]


def test_the_auto_selected_variant_carries_its_own_verified_copy(
    remote_gguf_repo, monkeypatch, hub_cache
):
    """No explicit variant means config picks one, and that pick is what the load asks for."""
    asked = []

    def fake_cached_gguf_for_load(repo, variant, **kw):
        asked.append((repo, variant, kw.get("verify_sizes")))
        return str(_cached(hub_cache, f"Llama-3.2-1B-Instruct-{variant}.gguf"))

    monkeypatch.setattr(llama_cpp, "cached_gguf_for_load", fake_cached_gguf_for_load)

    config = ModelConfig.from_identifier(REPO)
    assert config.gguf_variant in {"Q4_K_M", "Q8_0"}
    assert asked == [(REPO, config.gguf_variant, True)]
    name = f"Llama-3.2-1B-Instruct-{config.gguf_variant}.gguf"
    cached = hub_cache / f"models--{REPO.replace('/', '--')}" / "snapshots" / ("a" * 40) / name
    assert config.gguf_verified == (
        REPO,
        config.gguf_variant,
        str(cached),
        ((name, cached.stat().st_size),),
    )


def test_nothing_is_carried_when_no_cached_copy_verifies(remote_gguf_repo):
    """A first-time download has no cached path to carry."""
    assert ModelConfig.from_identifier(REPO, gguf_variant = "Q8_0").gguf_verified is None
    assert ModelConfig.from_identifier(REPO).gguf_verified is None
