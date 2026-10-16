"""Mirror targets that are also a model's own artifact repo (GGUF / FP8 checkpoints at the root, the official
diffusers folders beside them): the opt-out must not rewrite them, and pipeline sizing must not count the quants."""

from __future__ import annotations

from pathlib import Path

from core.inference import diffusion_families as df
from core.inference.diffusion import DiffusionBackend


def test_the_new_targets_are_artifact_mirrors_and_the_old_full_mirrors_are_not():
    for repo in ("unsloth/Z-Image-GGUF", "unsloth/Qwen-Image-Edit-GGUF", "unsloth/Qwen-Image-Edit-2509-GGUF",
                 "unsloth/Qwen-Image-Layered-GGUF", "unsloth/Qwen-Image-2.1-Turbo-FP8"):
        assert df.mirror_hosts_artifacts(repo), repo
        assert df.mirror_hosts_artifacts(repo.lower()), repo
    for repo in ("unsloth/Z-Image-Turbo", "unsloth/FLUX.1-dev", "unsloth/Qwen-Image", "unsloth/Z-Image-Turbo-GGUF"):
        assert not df.mirror_hosts_artifacts(repo), repo


def test_the_opt_out_keeps_an_artifact_repo_and_still_unmirrors_a_full_mirror(monkeypatch):
    """Under UNSLOTH_DIFFUSION_NO_MIRROR a pure mirror maps back to its upstream, but unsloth/Z-Image-GGUF holds the
    GGUF being loaded, which Tongyi-MAI/Z-Image does not have: rewriting it 404s the denoiser fetch."""
    monkeypatch.setenv("UNSLOTH_DIFFUSION_NO_MIRROR", "1")
    assert df.prefer_ungated_mirror("unsloth/Z-Image-GGUF") == "unsloth/Z-Image-GGUF"
    assert df.prefer_ungated_mirror("unsloth/Qwen-Image-2.1-Turbo-FP8") == "unsloth/Qwen-Image-2.1-Turbo-FP8"
    assert df.prefer_ungated_mirror("unsloth/Z-Image-Turbo") == "Tongyi-MAI/Z-Image-Turbo"
    assert df.prefer_ungated_mirror("Tongyi-MAI/Z-Image") == "Tongyi-MAI/Z-Image"


def _snapshot(root: Path, repo: str, files: dict[str, int]) -> None:
    repo_dir = root / f"models--{repo.replace('/', '--')}"
    snap = repo_dir / "snapshots" / "rev1"
    for name, size in files.items():
        p = snap / name
        p.parent.mkdir(parents = True, exist_ok = True)
        p.write_bytes(b"\0" * size)
    (repo_dir / "refs").mkdir(parents = True, exist_ok = True)
    (repo_dir / "refs" / "main").write_text("rev1")


def test_pipeline_sizing_of_an_artifact_repo_counts_only_the_diffusers_part(monkeypatch, tmp_path):
    _snapshot(tmp_path, "unsloth/Z-Image-GGUF", {
        "z-image-Q4_K_M.gguf": 5000, "z-image-Q8_0.gguf": 8000, "README.md": 10,
        "model_index.json": 7, "vae/diffusion_pytorch_model.safetensors": 300, "text_encoder/model.safetensors": 600,
    })
    monkeypatch.setattr(DiffusionBackend, "_hub_cache_repo_dirs",
                        staticmethod(lambda repo_id: [tmp_path / f"models--{repo_id.replace('/', '--')}"]))
    assert DiffusionBackend._cache_bytes("unsloth/Z-Image-GGUF") == 13917
    assert DiffusionBackend._cache_bytes("unsloth/Z-Image-GGUF", pipeline_only = True) == 907


def test_a_direct_pipeline_pick_of_an_artifact_repo_is_still_refused():
    """The official folders inside unsloth/Qwen-Image-2.1-Turbo-FP8 are what a pick of Qwen/Qwen-Image-2.1-Turbo
    fetches; naming the FP8 repo itself as a pipeline stays refused, as it was before the folders arrived. The GGUF
    repos keep their own -GGUF refusal."""
    import pytest

    assert "unsloth/qwen-image-2.1-turbo-fp8" in df.prequant_only_repo_ids()
    backend = DiffusionBackend()
    with pytest.raises(ValueError, match = "pre-quantised"):
        backend.validate_load_request("unsloth/Qwen-Image-2.1-Turbo-FP8", model_kind = "pipeline")
    for repo in ("unsloth/Z-Image-GGUF", "unsloth/Qwen-Image-Edit-GGUF", "unsloth/Qwen-Image-Edit-2509-GGUF",
                 "unsloth/Qwen-Image-Layered-GGUF"):
        with pytest.raises(ValueError):
            backend.validate_load_request(repo, model_kind = "pipeline")
    # and the base pick itself still validates
    assert backend.validate_load_request("Qwen/Qwen-Image-2.1-Turbo", model_kind = "pipeline").name == "qwen-image-2.1"
