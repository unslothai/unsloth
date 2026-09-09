"""The transformers-5 config fix, demonstrated against a real transformers 5.

transformers 5.x turns `PretrainedConfig` subclasses into dataclasses. vLLM's
`configs/deepseek_vl2.py` declares `vision_config: VisionEncoderConfig` with no
default, and a dataclass will not accept a non-default field after an inherited
default one ("TypeError: non-default argument 'vision_config' follows default
argument"). That fires while importing `vllm.transformers_utils.configs`, taking
down `import vllm` and with it `import unsloth`.

The other tests for this fix assert on source text; this one reproduces the
failing shape and checks the outcome, so it catches the fix silently ceasing to
work. No vLLM install needed: the config class above IS the reproduction. Skips
on transformers 4.x, where configs are not dataclasses.
"""

import pytest

transformers = pytest.importorskip("transformers")

from packaging.version import Version  # noqa: E402

pytestmark = pytest.mark.skipif(
    Version(transformers.__version__) < Version("5.0.0"),
    reason = "transformers 4.x does not convert config subclasses to dataclasses",
)


def _build(tag):
    """A vLLM-shaped config pair: a bare annotation with no default."""
    from transformers.configuration_utils import PretrainedConfig

    class VisionEncoderConfig(PretrainedConfig):
        model_type = f"vision_{tag}"

    class DeepseekVL2Config(PretrainedConfig):
        model_type = f"deepseek_vl_v2_{tag}"
        vision_config: VisionEncoderConfig  # no default: the trigger

    return DeepseekVL2Config


@pytest.fixture
def unpatched():
    """Remove the patch so the failure can be observed, then restore it.
    Imports unsloth first: run alone, nothing would have installed it yet."""
    import unsloth  # noqa: F401 - installs the patch we are about to remove

    from transformers.configuration_utils import PretrainedConfig

    saved = PretrainedConfig.__dict__.get("__init_subclass__")
    flag = getattr(PretrainedConfig, "_unsloth_patched_init_subclass", False)
    inner = getattr(saved, "__func__", saved)
    original = getattr(inner, "__wrapped__", None)
    if flag and original is not None:
        PretrainedConfig.__init_subclass__ = classmethod(original)
        PretrainedConfig._unsloth_patched_init_subclass = False
    yield
    if saved is not None:
        PretrainedConfig.__init_subclass__ = saved
    PretrainedConfig._unsloth_patched_init_subclass = flag


def test_the_failure_is_real_without_the_fix(unpatched):
    """Guards the premise: if this stops raising, the fix tests nothing."""
    from unsloth.import_fixes import (
        _transformers_configs_are_kw_only,
        _transformers_needs_bare_annotation_fix,
        fix_transformers5_bare_annotation_configs,
    )
    from transformers.configuration_utils import PretrainedConfig

    if getattr(PretrainedConfig, "_unsloth_patched_init_subclass", False):
        pytest.skip("could not unpatch; the wrapped original was not reachable")
    if _transformers_configs_are_kw_only(PretrainedConfig):
        pytest.skip(
            f"transformers {transformers.__version__} passes kw_only=True "
            f"(5.5.1+), so the ordering rule this fix works around is gone"
        )
    # The ordering rule only exists between 5.4.0 and 5.5.0: 5.0.0 to 5.3.x are 5.x but do not dataclass-ify configs at
    # all (no `__init_subclass__`), so nothing raises there and the premise below does not apply.
    if not _transformers_needs_bare_annotation_fix():
        pytest.skip(
            f"transformers {transformers.__version__} does not apply the "
            f"dataclass ordering rule to config subclasses (pre-5.4.0)"
        )
    with pytest.raises(TypeError, match = "non-default argument"):
        _build("unpatched")


def test_the_fix_stands_down_when_transformers_handles_it():
    """kw_only=True fixed this upstream, so patching anyway would be an untested
    monkey patch. >= 5.5.1 covers both branches (5.5.1 on 5.5, 5.6.0 on main)."""
    from unsloth.import_fixes import (
        _transformers_configs_are_kw_only,
        fix_transformers5_bare_annotation_configs,
    )
    from transformers.configuration_utils import PretrainedConfig

    kw_only = _transformers_configs_are_kw_only(PretrainedConfig)
    expected = Version(transformers.__version__) >= Version("5.5.1")
    assert (
        kw_only == expected
    ), f"transformers {transformers.__version__}: probe says kw_only={kw_only}"
    if not kw_only:
        pytest.skip("this transformers still needs the fix")

    PretrainedConfig._unsloth_patched_init_subclass = False
    fix_transformers5_bare_annotation_configs()
    assert not getattr(PretrainedConfig, "_unsloth_patched_init_subclass", False)


def test_the_fix_lets_it_import():
    from unsloth.import_fixes import fix_transformers5_bare_annotation_configs

    fix_transformers5_bare_annotation_configs()
    cls = _build("patched")
    assert cls.__name__ == "DeepseekVL2Config"


def test_applying_twice_is_a_no_op():
    from unsloth.import_fixes import fix_transformers5_bare_annotation_configs
    from transformers.configuration_utils import PretrainedConfig

    fix_transformers5_bare_annotation_configs()
    first = PretrainedConfig.__dict__.get("__init_subclass__")
    fix_transformers5_bare_annotation_configs()
    assert PretrainedConfig.__dict__.get("__init_subclass__") is first


def test_ordinary_configs_are_unaffected():
    """The patch runs for EVERY config subclass, so it must disturb none."""
    from unsloth.import_fixes import fix_transformers5_bare_annotation_configs
    from transformers.configuration_utils import PretrainedConfig

    fix_transformers5_bare_annotation_configs()

    class Ordinary(PretrainedConfig):
        model_type = "ordinary_probe"

        def __init__(
            self,
            hidden_size = 16,
            **kwargs,
        ):
            self.hidden_size = hidden_size
            super().__init__(**kwargs)

    cfg = Ordinary(hidden_size = 32)
    assert cfg.hidden_size == 32
    assert cfg.model_type == "ordinary_probe"


def test_a_real_model_config_still_loads():
    from unsloth.import_fixes import fix_transformers5_bare_annotation_configs

    fix_transformers5_bare_annotation_configs()
    from transformers import LlamaConfig

    cfg = LlamaConfig(hidden_size = 64, num_hidden_layers = 2)
    assert cfg.hidden_size == 64


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
