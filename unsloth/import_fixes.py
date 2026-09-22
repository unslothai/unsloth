# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
from importlib.metadata import distribution as importlib_distribution
from importlib.metadata import version as importlib_version
from importlib.metadata import PackageNotFoundError
from packaging.version import Version as TrueVersion
import re
import logging
import textwrap
import warnings
import sys
import threading
import functools
import inspect

# We cannot do from unsloth_zoo.log import logger since FBGEMM might cause seg faults.
UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") in (
    "1",
    "True",
    "true",
)
logger = logging.getLogger(__name__)
if UNSLOTH_ENABLE_LOGGING:
    logging.basicConfig(level = logging.INFO, format = "[%(name)s|%(levelname)s]%(message)s")
    logger.setLevel(logging.INFO)
else:
    logging.basicConfig(level = logging.WARNING, format = "[%(name)s|%(levelname)s]%(message)s")
    logger.setLevel(logging.WARNING)

_AMDGPU_IDS_MISSING_TEXT = "amdgpu.ids: No such file or directory"


def Version(version):
    try:
        new_version = str(version)
        new_version = re.match(r"[0-9\.]{1,}", new_version)
        if new_version is None:
            raise ValueError(f"Could not parse version: {version}")
        new_version = new_version.group(0).rstrip(".")
        if new_version != version:
            new_version += ".1"
        return TrueVersion(new_version)
    except:
        from inspect import getframeinfo, stack
        caller = getframeinfo(stack()[1][0])
        raise RuntimeError(
            f"Unsloth: Could not get version for `{version}`\n"
            f"File name = [{caller.filename}] Line number = [{caller.lineno}]"
        )


# Ignore logging messages
class HideLoggingMessage(logging.Filter):
    __slots__ = ("text",)

    def __init__(self, text):
        self.text = text

    def filter(self, x):
        return not (self.text in x.getMessage())


class HidePrintMessage:
    def __init__(self, original_stream):
        self._original_stream = original_stream
        self._hidden_texts = []

    def add_filter(self, text):
        self._hidden_texts.append(text)

    def write(self, message):
        if not any(text in message for text in self._hidden_texts):
            self._original_stream.write(message)

    def flush(self):
        self._original_stream.flush()

    def __getattr__(self, name):
        return getattr(self._original_stream, name)


import contextlib
import ctypes

try:
    _libc = ctypes.CDLL(None)
except Exception:
    _libc = None


@contextlib.contextmanager
def suppress_cuda_printf():
    """Suppress CUDA device-side printf by redirecting stdout/stderr fds to /dev/null.

    CUDA device printf (e.g. CUTLASS "Arch conditional MMA" errors on Blackwell)
    writes to fd 1 at the C level, bypassing Python's sys.stdout, so the
    HidePrintMessage filter can't catch it. Redirect fd 1 and 2 at the OS level,
    sync CUDA, then restore.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved_fds = {}
    try:
        for fd in (1, 2):
            saved_fds[fd] = os.dup(fd)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, fd)
            os.close(devnull)
        yield
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        if _libc is not None:
            try:
                _libc.fflush(None)
            except Exception:
                pass
        for fd, saved in saved_fds.items():
            os.dup2(saved, fd)
            os.close(saved)


if not UNSLOTH_ENABLE_LOGGING:
    import sys

    # Applied to stderr for FBGEMM (pytorch/FBGEMM utils.py#L43-L52) and CUTLASS SM90-vs-SM100 MMA / arch / TMA errors.
    sys.stderr = HidePrintMessage(sys.stderr)
    # https://github.com/pytorch/FBGEMM/blob/d99cd96490ec4aabac2ee95b1e76ea4dcfcfa628/fbgemm_gpu/experimental/ge
    # mm/triton_gemm/utils.py#L43-L52
    sys.stderr.add_filter("TMA benchmarks will be running")
    # CUTLASS/FBGEMM MMA instruction error on SM90 vs SM100 (Blackwell) GPUs
    # https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp
    sys.stderr.add_filter("Arch conditional MMA instruction used without targeting")
    # CUTLASS arch conditional errors for various architectures
    sys.stderr.add_filter("CUTE_INVALID_CONTROL_PATH")
    # CUTLASS TMA-related errors when not targeting correct architecture
    sys.stderr.add_filter("Trying to use tma without CUTE_ARCH_TMA")
    # torchao logs a cosmetic "Skipping import of cpp extensions" WARNING on torch < 2.11. The
    # bnb-4bit / Unsloth paths do not use its cpp kernels, so drop that one record rather than
    # raising the whole torchao logger to ERROR.
    logging.getLogger("torchao").addFilter(
        HideLoggingMessage("Skipping import of cpp extensions due to incompatible torch version")
    )
    # torch >= 2.11: torchao dlopens each prebuilt _C*.so and logs a failure on an ABI tag
    # mismatch (a cp310 .so under cp312) or a missing arch kernel, then falls back to non-cpp
    # paths Unsloth does not use.
    logging.getLogger("torchao").addFilter(HideLoggingMessage("Failed to load "))
    # SyntaxWarning: invalid escape sequence '\.'
    warnings.filterwarnings("ignore", message = "invalid escape sequence", category = SyntaxWarning)
    # PYTORCH_CUDA_ALLOC_CONF is deprecated warning from torch
    warnings.filterwarnings("ignore", message = "PYTORCH_CUDA_ALLOC_CONF is deprecated")
    # TF32 precision deprecation warning from torch
    warnings.filterwarnings("ignore", message = "Please use the new API settings to control TF32")
    # Deprecation warnings from torchao
    warnings.filterwarnings("ignore", message = "`int4_weight_only` is deprecated")
    warnings.filterwarnings("ignore", message = "`int8_weight_only` is deprecated")
    # torch._check_is_size FutureWarning (called by bitsandbytes 4-bit dequant)
    warnings.filterwarnings(
        "ignore", message = r"_check_is_size will be removed", category = FutureWarning
    )

    # TorchAO deprecated import paths (https://github.com/pytorch/ao/issues/2752)
    warnings.filterwarnings(
        "ignore",
        message = r"Importing.*from torchao\.dtypes.*is deprecated",
        category = DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message = r"Importing BlockSparseLayout from torchao\.dtypes is deprecated",
        category = DeprecationWarning,
    )

    warnings.filterwarnings(
        "ignore",
        message = r"builtin type Swig.*has no __module__ attribute",
        category = DeprecationWarning,
    )

    # Triton autotuner deprecation (https://github.com/triton-lang/triton/pull/4496)
    warnings.filterwarnings(
        "ignore",
        message = r"warmup, rep, and use_cuda_graph parameters are deprecated",
        category = DeprecationWarning,
    )

    # Python 3.12+ multiprocessing fork warning in multi-threaded processes
    warnings.filterwarnings(
        "ignore",
        message = r".*multi-threaded.*use of fork\(\) may lead to deadlocks",
        category = DeprecationWarning,
    )

    # Resource warnings from internal socket/file operations
    warnings.filterwarnings("ignore", message = r"unclosed.*socket", category = ResourceWarning)
    warnings.filterwarnings("ignore", message = r"unclosed file.*dev/null", category = ResourceWarning)

    # torch 2.9+ pin_memory/is_pinned device arg deprecation
    warnings.filterwarnings(
        "ignore",
        message = r"The `device` argument is deprecated",
        category = DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message = r".*pin_memory.*device.*deprecated",
        category = DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message = r".*is_pinned.*device.*deprecated",
        category = DeprecationWarning,
    )

    # vllm "Level is deprecated" stderr noise
    sys.stderr.add_filter("Level is deprecated")

    # PydanticSerializationUnexpectedValue warning
    warnings.filterwarnings(
        "ignore",
        message = r".*PydanticSerializationUnexpectedValue",
    )
    warnings.filterwarnings(
        "ignore",
        message = r"Expected.*but got.*with value.*is not.*subclass",
    )

    # Triton "df: No such file or directory" stderr noise
    sys.stderr.add_filter("df: No such file")
    # ROCm/libdrm missing ids table stderr noise on some AMD setups
    sys.stderr.add_filter(_AMDGPU_IDS_MISSING_TEXT)
    # Apex ROCm fused RoPE backend selection warning when Aiter is enabled.
    warnings.filterwarnings(
        "ignore",
        message = r"^Aiter backend is selected for fused RoPE\.?",
        category = UserWarning,
        module = r"^apex\.transformer\.functional\.fused_rope$",
    )


def fix_torch_check_is_size():
    """Shim torch._check_is_size if a future torch removes it (bitsandbytes 4-bit
    dequant calls it). The FutureWarning is silenced in suppress_cuda_printf."""
    try:
        import torch

        if hasattr(torch, "_check_is_size"):
            return

        def _check_is_size(
            i,
            message = None,
            *,
            max = None,
        ):
            torch._check(i >= 0, message)
            if max is not None:
                torch._check(i <= max, message)

        torch._check_is_size = _check_is_size
    except Exception:
        return


# Fix "AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'" first, mainly
# because tensorflow causes issues.
def fix_message_factory_issue():
    try:
        import google.protobuf.message_factory

        class MessageFactory:
            def CreatePrototype(self, *args, **kwargs):
                return

            def GetMessages(self, *args, **kwargs):
                return

            def GetPrototype(self, *args, **kwargs):
                return

        if not hasattr(google.protobuf.message_factory, "MessageFactory"):
            logger.info("Unsloth: Patching protobuf.MessageFactory as it doesn't exist")
            google.protobuf.message_factory.MessageFactory = MessageFactory
        elif (
            hasattr(google.protobuf.message_factory, "MessageFactory")
            and not hasattr(google.protobuf.message_factory.MessageFactory, "GetPrototype")
            and not hasattr(google.protobuf.message_factory, "GetMessageClass")
        ):
            google.protobuf.message_factory.MessageFactory = MessageFactory
            logger.info("Unsloth: Patching protobuf.MessageFactory as it doesn't exist")
        elif (
            hasattr(google.protobuf.message_factory, "MessageFactory")
            and not hasattr(google.protobuf.message_factory.MessageFactory, "GetPrototype")
            and hasattr(google.protobuf.message_factory, "GetMessageClass")
        ):
            GetMessageClass = google.protobuf.message_factory.GetMessageClass

            def GetPrototype(self, descriptor):
                return GetMessageClass(descriptor)

            google.protobuf.message_factory.MessageFactory.GetPrototype = GetPrototype
            logger.info("Unsloth: Patching protobuf.MessageFactory.GetPrototype")
        pass
    except:
        pass


# Fix xformers performance issues since 0.0.25; see facebookresearch/xformers#1176 (comment 2545829591).
def fix_xformers_performance_issue():
    spec = importlib.util.find_spec("xformers")
    if spec is None:
        return
    xformers_version = importlib_version("xformers")
    if Version(xformers_version) < Version("0.0.29"):
        xformers_location = spec.origin
        if xformers_location is None:
            xformers_location = spec.submodule_search_locations[0]
        else:
            xformers_location = os.path.split(xformers_location)[0]
        cutlass = Path(xformers_location) / "ops" / "fmha" / "cutlass.py"
        try:
            if cutlass.exists():
                with open(cutlass, "r+", encoding = "utf-8") as f:
                    text = f.read()
                    # See https://github.com/facebookresearch/xformers/issues/1176#issuecomment-2545829591
                    if "num_splits_key=-1," in text:
                        text = text.replace(
                            "num_splits_key=-1,",
                            "num_splits_key=None,",
                        )
                        f.seek(0)
                        f.write(text)
                        f.truncate()
                        logger.info("Unsloth: Patching Xformers to fix some performance issues.")
        except Exception as e:
            logger.info(f"Unsloth: Failed patching Xformers with error = {str(e)}")


# flash-attn 4 ships flash_attn/cute/ with no __init__.py, so flash_attn is a namespace package
# without flash_attn.flash_attn_interface. xformers gates on find_spec then imports that submodule
# unguarded, so `import xformers.ops` raises, _utils.py swallows it into xformers = None,
# HAS_XFORMERS goes False and every fast-path model drops to SDPA: measured on a B200 at seq_len
# 8192, 547 ms/step and 2.69 GB peak against 2154 ms/step and 19.02 GB on SDPA. The repair imports
# xformers ONCE with flash_attn hidden from find_spec, so it takes the next branch of its own elif
# chain and the working module is cached in sys.modules. Nothing is written to any third-party
# package.
_FLASH_ATTN_INTERFACE_NAME = "flash_attn_interface"
_FLASH_ATTN_INTERFACE_MODULE = "flash_attn." + _FLASH_ATTN_INTERFACE_NAME
_FA4_NAMESPACE_WARNED = [False]


def _flash_attn_submodule_exists(name):
    """True iff `flash_attn.<name>` exists on disk, WITHOUT importing `flash_attn`.

    `importlib.util.find_spec("flash_attn.x")` looks cheap but is not: resolving a dotted name
    IMPORTS the parent package first, so on every machine with a real flash-attn 2 it would run
    `flash_attn/__init__.py` (and load `flash_attn_2_cuda`) during `import unsloth`, for users
    who never asked for flash attention. Probing the package's own search locations answers the
    same question with a stat() and no side effects.
    """
    try:
        spec = importlib.util.find_spec("flash_attn")
        if spec is None:
            return False
        locations = list(spec.submodule_search_locations or ())
    except Exception:
        return False
    for location in locations:
        base = os.path.join(location, name)
        if os.path.isdir(base):
            return True
        for suffix in importlib.machinery.all_suffixes():
            if os.path.isfile(base + suffix):
                return True
    return False


def _flash_attn_layout():
    """Classify the installed `flash_attn` module tree.

    Returns ``"absent"`` (no `flash_attn` at all), ``"flash_attn_2"`` (a real flash-attn 2/3
    layout -- `flash_attn.flash_attn_interface` is present, which is precisely what xformers
    imports, so it works whether or not flash-attn 4 is installed ALONGSIDE it), or
    ``"flash_attn_4_only"`` (importable `flash_attn` with no FA2 entry points).

    Never imports `flash_attn`. A real flash-attn 2 whose extension fails to load is still
    classified as FA2 here and is left completely alone -- that breakage is separate and
    already reported by `models/_utils.py`.
    """
    try:
        if importlib.util.find_spec("flash_attn") is None:
            return "absent"
    except Exception:
        # A parent package that explodes on import is not something to second-guess.
        return "absent"
    if _flash_attn_submodule_exists(_FLASH_ATTN_INTERFACE_NAME):
        return "flash_attn_2"
    return "flash_attn_4_only"


def _flash_attn_4_present():
    return _flash_attn_submodule_exists("cute")


def _warn_flash_attn_4_shadow_once(detail):
    if _FA4_NAMESPACE_WARNED[0]:
        return
    _FA4_NAMESPACE_WARNED[0] = True
    if _flash_attn_4_present():
        head = (
            "Unsloth: flash-attn 4 is installed as the namespace package `flash_attn` (only "
            "`flash_attn.cute`), which has no `flash_attn_func` / `flash_attn_varlen_func` and "
            "no `flash_attn.flash_attn_interface`."
        )
    else:
        head = (
            "Unsloth: `flash_attn` resolves to a namespace package with no "
            "`flash_attn.flash_attn_interface` (a partial or shadowed flash-attn install)."
        )
    logger.warning(
        head + "\n"
        f"xFormers imports that module unconditionally, so it is unusable here ({detail}), and "
        "Unsloth has fallen back to PyTorch SDPA. Measured cost on a B200 at seq_len 8192 "
        "(Qwen3-0.6B + LoRA): 547 ms/step -> 2154 ms/step (3.9x slower) and 2.69 GB -> 19.02 GB "
        "peak (7x more memory).\n"
        "Unsloth cannot use flash-attn 4 either: its entry point is "
        "`from flash_attn.cute import flash_attn_func` and it returns a (out, softmax_lse) "
        "tuple rather than a tensor.\n"
        "To get the fast path back, install a flash-attn 2 that xFormers accepts (it enforces "
        '>=2.7.1: `pip install --no-build-isolation "flash-attn>=2.7.1"`) or uninstall '
        "flash-attn 4 (`pip uninstall flash-attn-4`) so xFormers can load."
    )


def fix_flash_attn_4_namespace_shadow():
    """Keep xFormers importable when only flash-attn 4 is installed."""
    if _flash_attn_layout() != "flash_attn_4_only":
        return
    if importlib.util.find_spec("xformers") is None:
        # Nothing to protect: no xformers means SDPA anyway, and is_flash_attn_2_available() is False
        # here, since the distribution is flash-attn-4 and the flash_attn metadata lookup misses.
        return
    if "xformers.ops.fmha.flash" in sys.modules:
        # Already imported successfully. A FAILED import leaves nothing in sys.modules, so this does
        # not mask the case we are here to fix.
        return

    real_find_spec = importlib.util.find_spec

    def _find_spec_without_flash_attn(name, package = None):
        if name == "flash_attn" or name.startswith("flash_attn."):
            return None
        return real_find_spec(name, package)

    # Process-global swap, scoped to this one import and restored in `finally`. Another thread
    # calling find_spec("flash_attn*") in the ~1.0s window is told it is absent, which is honest
    # for the FA2 namespace xformers asks about.
    importlib.util.find_spec = _find_spec_without_flash_attn
    try:
        import xformers.ops  # noqa: F401
    except Exception as error:
        _warn_flash_attn_4_shadow_once(f"xFormers still failed to import: {error}")
        return
    finally:
        importlib.util.find_spec = real_find_spec

    logger.info(
        "Unsloth: Hid the flash-attn 4 namespace package from xFormers' import so xFormers "
        "keeps working. Unsloth cannot use flash-attn 4 itself."
    )


def patch_vllm_for_notebooks():
    import sys

    ipython = None
    try:
        from IPython import get_ipython as _get_ipython
    except Exception:
        _get_ipython = None

    if _get_ipython is not None:
        try:
            ipython = _get_ipython()
        except Exception:
            ipython = None

    if ipython is None:
        try:
            import builtins
            _get_ipython = getattr(builtins, "get_ipython", None)
            if callable(_get_ipython):
                ipython = _get_ipython()
        except Exception:
            ipython = None

    if ipython is None:
        return

    try:
        shell = ipython.__class__.__name__
        is_notebook = shell == "ZMQInteractiveShell" or "google.colab" in str(type(ipython))
    except Exception:
        return

    if not is_notebook:
        return

    if not hasattr(sys.stdout, "fileno"):
        return

    needs_patch = False
    try:
        fd = sys.stdout.fileno()
        if not isinstance(fd, int) or fd < 0:
            needs_patch = True
    except Exception:
        needs_patch = True

    if not needs_patch:
        return

    logger.info(
        "Unsloth: Notebook detected - Patching sys.stdout.fileno for newer `vllm>=0.12.0` versions"
    )
    sys.stdout.fileno = lambda: 1


# TypeError: non-default argument 'vision_config' follows default argument
_UNSLOTH_DC_BACKFILL_FLAG = "__unsloth_dc_defaults_backfilled__"


def _backfill_dataclass_defaults(cls):
    """Give class-local bare annotations a ``None`` default.

    transformers 5.x dataclass-ifies every ``PretrainedConfig`` subclass, so a
    bare annotation after an inherited default trips "non-default argument
    follows default argument". Skips names resolving anywhere in the MRO, and
    ``ClassVar`` / ``InitVar``, matched textually as annotations may be strings.
    """
    if cls.__dict__.get(_UNSLOTH_DC_BACKFILL_FLAG):
        return []
    own_annotations = cls.__dict__.get("__annotations__") or {}
    backfilled = []
    for name, annotation in own_annotations.items():
        text = (
            annotation
            if isinstance(annotation, str)
            else (getattr(annotation, "__name__", "") or repr(annotation))
        )
        if "ClassVar" in text or "InitVar" in text:
            continue
        # dataclass reads defaults with getattr, so the MRO already counts.
        if hasattr(cls, name):
            continue
        try:
            setattr(cls, name, None)
            backfilled.append(name)
        except Exception:
            pass
    try:
        setattr(cls, _UNSLOTH_DC_BACKFILL_FLAG, True)
    except Exception:
        pass
    return backfilled


def _transformers_configs_are_kw_only(PretrainedConfig):
    """Does this transformers already build config dataclasses `kw_only`?

    `kw_only=True` (5.5.1 on the 5.5 branch, 5.6.0 on main) removes the ordering
    rule this fix exists for. Read the source, not the version, since it was
    backported; where the source is unreadable (stripped or frozen installs),
    probe instead, as a wrong False gives required fields a `None` default.
    """
    import inspect

    hook = PretrainedConfig.__dict__.get("__init_subclass__")
    hook = getattr(hook, "__func__", hook)
    if hook is None:
        return False
    try:
        return re.search(r"\bkw_only\s*=\s*True", inspect.getsource(hook)) is not None
    except Exception:
        pass
    return not _transformers_needs_bare_annotation_fix()


def _transformers_needs_bare_annotation_fix():
    """Does defining a bare-annotation config subclass actually raise here?

    Answered by trying it: the rule (5.4.0) and `kw_only=True` (5.5.1) were both
    backported, so a version window mislabels distros carrying them early or
    late. Absent positive evidence answer False: a loud `TypeError` beats a
    config quietly accepting a missing required field.
    """
    try:
        from transformers.configuration_utils import PretrainedConfig
    except Exception:
        return False
    try:
        # Exactly the shape vLLM defines: a default, then a bare annotation.
        type(
            "_UnslothProbeConfig",
            (PretrainedConfig,),
            {"__annotations__": {"a": int, "b": int}, "a": 0},
        )
    except TypeError:
        return True
    except Exception:
        return False
    return False


def fix_transformers5_bare_annotation_configs():
    """Stop transformers 5.x from breaking third-party config classes.

    vLLM's ``configs/deepseek_vl2.py`` declares a bare ``vision_config``, which
    raises ``TypeError`` while importing ``vllm.transformers_utils.configs`` and
    takes ``import unsloth`` down with it. Patching
    ``PretrainedConfig.__init_subclass__`` rather than vLLM's source covers every
    affected class in any vLLM version. No-ops outside the 5.4.0 to 5.5.0
    window: below it configs are not dataclasses, above ``kw_only=True`` leaves
    no ordering rule to break.
    """
    try:
        import transformers
        if Version(transformers.__version__) < Version("5.0.0"):
            return
        from transformers.configuration_utils import PretrainedConfig
    except Exception as e:
        logger.info(f"Unsloth: Skipping transformers-5 config fix ({e})")
        return

    if _transformers_configs_are_kw_only(PretrainedConfig):
        return

    if getattr(PretrainedConfig, "_unsloth_patched_init_subclass", False):
        return

    original = PretrainedConfig.__dict__.get("__init_subclass__")
    if original is None:
        return
    # __init_subclass__ is an implicit classmethod; unwrap to the function.
    original_func = getattr(original, "__func__", original)

    def __init_subclass__(cls, *args, **kwargs):
        try:
            _backfill_dataclass_defaults(cls)
        except Exception as e:
            logger.info(f"Unsloth: dataclass default backfill skipped ({e})")
        return original_func(cls, *args, **kwargs)

    # Keep the original reachable, so the patch can be tested and undone.
    __init_subclass__.__wrapped__ = original_func
    try:
        PretrainedConfig.__init_subclass__ = classmethod(__init_subclass__)
        PretrainedConfig._unsloth_patched_init_subclass = True
        logger.info(
            "Unsloth: Patching transformers `PretrainedConfig.__init_subclass__` "
            "so vLLM config classes with bare annotations still import"
        )
    except Exception as e:
        logger.info(f"Unsloth: Failed patching PretrainedConfig ({e})")


_SDPA_MASK_PATCH_FLAG = "_unsloth_patched_sdpa_mask"


def _sdpa_mask_is_patched(masking_utils):
    """Are the live bindings ours, right now?

    Asked of the FUNCTIONS rather than of a flag on the module. A module-level
    flag outlives what it describes: `importlib.reload(masking_utils)` re-runs
    the module body in the existing namespace, so `sdpa_mask` and the registry
    entry go back to upstream while any attribute we added survives. Gating on
    that flag would then refuse to re-patch a build that is once again
    vulnerable, which is the opposite of what an idempotence guard is for.

    Both bindings must carry the mark, so a half-installed state re-runs.
    """
    flagged = lambda fn: bool(getattr(fn, _SDPA_MASK_PATCH_FLAG, False))
    if not flagged(getattr(masking_utils, "sdpa_mask", None)):
        return False
    interface = getattr(masking_utils, "ALL_MASK_ATTENTION_FUNCTIONS", None)
    if interface is None:
        return True
    try:
        return flagged(interface["sdpa"])
    except Exception:
        return False


def _unmask_rows_attending_to_nothing(mask):
    """Give a query row that attends to no key uniform attention instead.

    Separated from the wrapper so it can be tested directly on every dtype the
    mask can arrive as, rather than only through a live transformers.

    `None` means transformers chose `is_causal` over a materialised mask. A
    FLOATING mask is the eager path, which converts to `finfo(dtype).min` and
    survives softmax's max-subtraction as uniform attention already, so it
    neither needs this nor would tolerate a boolean `|`. Bool is the normal sdpa
    result; int is what a caller who passed an int `attention_mask` gets back,
    and the correction is right for both.
    """
    if mask is None or mask.is_floating_point():
        return mask
    return mask | ~mask.any(dim = -1, keepdim = True)


def _left_padded_probe_mask(torch):
    """A 2-token batch whose first row is one pad then one real token.

    Row 0's query at position 0 is a pad: causal masking allows it only key 0,
    and the padding mask then removes key 0, so nothing is left. That is the
    row this whole fix is about, built as small as it can be.

    Bool, not int: `sdpa_mask` returns whatever dtype it was handed, and the
    real callers hand it a bool. Probing with an int mask gets an int mask back
    and a `dtype == torch.bool` check then answers "not affected" for a reason
    that has nothing to do with the bug.

    CPU explicitly, never the ambient default: under a `torch.set_default_device`
    of "meta" this lands on meta, where `sdpa_mask` builds its index tensors on
    CPU and raises, or returns a meta mask whose truth value cannot be read. The
    first answers "not affected" for the wrong reason; the second aborts the
    import. The probe is two elements, so pinning it to CPU costs nothing.
    """
    return torch.tensor(
        [[False, True], [True, True]],
        dtype = torch.bool,
        device = "cpu",
    )


def _sdpa_mask_leaves_rows_fully_masked():
    """Does THIS build hand SDPA a query row that attends to nothing?

    Answered by building one, not by comparing versions. `allow_torch_fix` was
    the correction and is now documented "Deprecated and has no effect", but it
    was a live parameter in 4.x and the retirement landed mid-5.x, so a version
    window would mislabel builds carrying it early or late. Absent positive
    evidence answer False and leave transformers alone.
    """
    try:
        import torch
        from transformers import masking_utils
    except Exception:
        return False
    sdpa_mask = getattr(masking_utils, "sdpa_mask", None)
    if sdpa_mask is None:
        return False
    # Unwrap first: a second call must probe the ORIGINAL, or the installed patch reports the bug
    # as fixed and a reload silently drops it.
    sdpa_mask = getattr(sdpa_mask, "__wrapped__", sdpa_mask)
    # The query axis is named differently per version: 5.x takes q_length, while 4.57.6 binds
    # sdpa_mask to sdpa_mask_recent_torch, which takes cache_position. Ask the signature, since a
    # probe that raises TypeError answers "no bug" for an unrelated reason.
    kwargs = {
        "batch_size": 2,
        "kv_length": 2,
        "attention_mask": _left_padded_probe_mask(torch),
        "allow_is_causal_skip": False,
    }
    try:
        params = inspect.signature(sdpa_mask).parameters
    except Exception:
        return False
    if "q_length" in params:
        kwargs["q_length"] = 2
    elif "cache_position" in params:
        kwargs["cache_position"] = torch.arange(2, device = "cpu")
    else:
        return False
    if "device" in params:
        kwargs["device"] = torch.device("cpu")
    try:
        mask = sdpa_mask(**kwargs)
        if mask is None or mask.is_floating_point():
            return False
        # Inside the guard too: reading a mask's truth value is what fails on a meta or unmaterialised
        # tensor, and a probe that raises would take `import unsloth` down with it.
        return bool((~mask.any(dim = -1)).any())
    except Exception:
        # Signature moved under us. Patching blind would be worse than not.
        return False


def fix_transformers_fully_masked_rows():
    """Stop a left-padded batch returning NaN logits, and empty generations.

    `masking_utils.sdpa_mask` returns a boolean mask with no correction for
    query rows that attend to nothing, and the parameter that used to make that
    correction, `allow_torch_fix`, is now "Deprecated and has no effect. Will be
    removed in version 5.18.0." In 4.57.6 the same function still carried it,
    guarded on `not _is_torch_greater_or_equal_than_2_5`, on the belief that
    torch 2.5 had made it unnecessary.

    It has not. Measured on a B200, torch 2.13.0+cu130, transformers 5.15.1,
    unquantized fp16 `google/gemma-4-E2B-it`, with no unsloth in the process: a
    SINGLE forward pass returns NaN logits on exactly the rows that received a
    left pad token and finite logits on every row that did not -- 16 of 16 rows
    across batch sizes 2, 4 and 8 -- and under `generate` those rows decode to
    the empty string. bfloat16 produces no NaNs. Three repeats per batch size
    are byte-identical, so it is deterministic. Reported as unsloth #9708.

    A fully masked row makes SDPA produce NaN, and the NaN then spreads along
    the row through `0 * NaN` where the next layer mixes values. Restoring the
    correction gives those rows uniform attention instead. They are pad
    positions whose outputs are discarded, so this does not change any result
    that is read -- which is what transformers' own docstring said when it did
    this: "in order to avoid `nan` propagation (this does not change the final
    result)".

    Self-neutralising: `mask | ~mask.any(-1)` contributes nothing once a row
    attends to something, so if a future transformers restores the guard this
    wrapper becomes a no-op rather than permanent damage. The eager path is
    unaffected either way -- `eager_mask` converts to `finfo(dtype).min` and a
    uniform-min row survives softmax's max-subtraction as uniform attention.
    """
    if not _sdpa_mask_leaves_rows_fully_masked():
        return
    try:
        import torch
        from transformers import masking_utils
    except Exception as e:
        logger.info(f"Unsloth: Skipping the fully-masked-row fix ({e})")
        return

    if _sdpa_mask_is_patched(masking_utils):
        return

    original = masking_utils.sdpa_mask
    original = getattr(original, "__wrapped__", original)

    @functools.wraps(original)
    def sdpa_mask(*args, **kwargs):
        return _unmask_rows_attending_to_nothing(original(*args, **kwargs))

    # functools.wraps sets __wrapped__, but set it explicitly: the probe and the tests both read
    # it, and a wraps-less edit must not make the patch un-probeable and un-undoable.
    sdpa_mask.__wrapped__ = original
    # The mark travels ON the wrapper, so the guard above reads the live binding and a reload that
    # drops it is re-patched rather than skipped.
    setattr(sdpa_mask, _SDPA_MASK_PATCH_FLAG, True)

    try:
        # BOTH bindings, which are not the same object: eager_mask calls the module global by name at
        # call time, while the interface captured the original in _global_mapping at class-body time.
        masking_utils.sdpa_mask = sdpa_mask
        interface = getattr(masking_utils, "ALL_MASK_ATTENTION_FUNCTIONS", None)
        if interface is not None:
            interface.register("sdpa", sdpa_mask)
        setattr(masking_utils, _SDPA_MASK_PATCH_FLAG, True)
        logger.info(
            "Unsloth: Patching transformers `sdpa_mask` so a left-padded row "
            "that attends to nothing cannot return NaN (unsloth #9708)"
        )
    except Exception as e:
        logger.info(f"Unsloth: Failed patching sdpa_mask ({e})")


_COMPOSITE_PREFIX_RENAMING_FLAG = "_unsloth_patched_composite_prefix_renaming"

# unsloth_zoo marks its own copy of this repair with this. Spelled as a literal rather than
# imported, so a zoo too old to define it cannot turn a skipped repair into an ImportError.
_ZOO_COMPOSITE_PREFIX_RENAMING_FLAG = "_unsloth_zoo_patched_composite_prefix_renaming"


def _zoo_composite_prefix_renaming_installed():
    """Has unsloth_zoo's copy of this repair already wrapped the function?

    The whole `__wrapped__` chain, because zoo also wraps the same function in
    `temporary_patches/moe_utils_bnb4bit.py` without setting `__wrapped__`, so the repair can
    sit under another wrapper. Bounded, so a malformed chain cannot spin.
    """
    try:
        from transformers import conversion_mapping
    except Exception:
        return False
    function = getattr(conversion_mapping, "get_model_conversion_mapping", None)
    seen = 0
    while function is not None and seen < 8:
        if getattr(function, _ZOO_COMPOSITE_PREFIX_RENAMING_FLAG, False):
            return True
        function = getattr(function, "__wrapped__", None)
        seen += 1
    return False


# How many of a submodule's own parameter names to try a renaming against. A prefix renaming
# either matches every name under the submodule or none of them, so one would do; eight costs
# nothing and covers a mapping that only rewrites some leaf names.
_COMPOSITE_RENAMING_SAMPLE = 8


def _renaming_signature(conversion):
    """Identity of a renaming by VALUE, because the mapping is handed out as deep copies.

    `get_checkpoint_conversion_mapping` returns `deepcopy(...)`, so the object this module
    collects from a submodule is never the object that ended up in the parent's list, and
    `is` can never match them. The patterns are what a renaming is, so compare those. The
    unprocessed ones when they are kept (5.5+), since `__post_init__` rewrites the live ones.
    """
    source = getattr(conversion, "_original_source_patterns", None) or conversion.source_patterns
    target = getattr(conversion, "_original_target_patterns", None) or conversion.target_patterns
    return (type(conversion).__name__, tuple(source), tuple(target))


def _sample_submodule_keys(submodule, prefix):
    """A few of the submodule's real parameter names, spelled as the PARENT spells them."""
    keys = []
    try:
        for name, _ in submodule.named_parameters(recurse = True):
            keys.append(f"{prefix}.{name}")
            if len(keys) >= _COMPOSITE_RENAMING_SAMPLE:
                return keys
        for name, _ in submodule.named_buffers(recurse = True):
            keys.append(f"{prefix}.{name}")
            if len(keys) >= _COMPOSITE_RENAMING_SAMPLE:
                break
    except Exception:
        return []
    return keys


def _renaming_destroys_keys(conversion, sample_keys, model_keys):
    """Does this renaming rewrite the model's OWN parameter names into names it does not have?

    This is the whole discriminator, and it is a behaviour, not a name. A renaming exists to
    map CHECKPOINT keys onto MODEL keys, so one that fires on a key the model really has and
    produces one it does not is wrong wherever it came from: the weight it renames is then
    dropped as unexpected, or rescued with no converter attached, which is how a pre-quantized
    checkpoint loses the bitsandbytes `absmax` / `quant_map` / `quant_state` sidecars.

    It also separates the two directions the same entry serves. The standalone text model
    registers `^model.language_model.` -> `^model.` because its checkpoint carries the longer
    keys and the model carries the shorter ones -- fired against that model's own names it
    matches nothing, and answers False here. Merged into the composite model, whose names ARE
    the longer ones, it matches all of them and lands off the map. One rule, both cases.

    A single rename onto a name the model really has is enough to answer False: that is a
    mapping doing its job, and no amount of other evidence should override it.
    """
    destroys = False
    for key in sample_keys:
        try:
            renamed, matched = conversion.rename_source_key(key)
        except Exception:
            return False
        if matched is None or renamed == key:
            continue
        if renamed in model_keys:
            return False
        destroys = True
    return destroys


def _prefixed_pattern(pattern, prefix):
    """Push a pattern down into `prefix`, keeping a start anchor anchored."""
    if pattern.startswith("^"):
        return f"^{prefix}.{pattern[1:]}"
    return f"{prefix}.{pattern}"


def _rescoped_renaming(conversion, prefix, sample_keys, model_keys):
    """The same renaming, scoped to the submodule it came from, or None if that cannot be built.

    This is what upstream's `PrefixChange.with_submodel_prefix` produces, spelled for a
    transformers that has no `PrefixChange`: `^model.language_model.` -> `^model.` collected
    from the submodule at `model.language_model` becomes
    `^model.language_model.model.language_model.` -> `model.language_model.model.`, which can
    only fire on a genuinely doubled prefix and therefore leaves every real key alone. Upstream
    5.6.2 builds the same pair for the same model.

    Only a renaming anchored at the start of the key can be scoped this way, which is the same
    restriction upstream has: a `PrefixChange` is anchored by construction. An unanchored
    pattern says "wherever this appears", and pushing a prefix in front of it produces a
    pattern that means neither thing, so this answers None and lets the caller drop the entry
    instead of pretending to have placed it. Nothing legitimate is lost -- the caller only
    reaches here for an entry already shown to rewrite real weight names off the map.

    Returned only after checking that it really is inert on this model's own names. A renaming
    that still rewrites real keys into nonexistent ones is worse than no renaming at all, so
    the caller drops it instead.
    """
    from transformers.core_model_loading import WeightRenaming

    source = getattr(conversion, "_original_source_patterns", None) or conversion.source_patterns
    target = getattr(conversion, "_original_target_patterns", None) or conversion.target_patterns
    if not all(pattern.startswith("^") for pattern in source):
        return None

    patterns = {
        "source_patterns": [_prefixed_pattern(p, prefix) for p in source],
        "target_patterns": [_prefixed_pattern(p, prefix) for p in target],
    }
    try:
        rescoped = type(conversion)(**patterns)
    except Exception:
        # A subclass whose __init__ takes something else entirely -- upstream's `PrefixChange`
        # takes prefixes, not patterns. It is still a WeightRenaming, and a renaming is all
        # this produces, so fall back to the base class rather than giving up.
        try:
            rescoped = WeightRenaming(**patterns)
        except Exception:
            return None
    for key in sample_keys:
        try:
            renamed, matched = rescoped.rename_source_key(key)
        except Exception:
            return None
        if matched is not None and renamed != key and renamed not in model_keys:
            return None
    return rescoped


def _leaked_submodule_prefix_renamings(model):
    """Which renamings did the recursion merge into `model`'s mapping where they do not belong?

    Walks the submodules the way `get_model_conversion_mapping` walks them -- same order, same
    "first model type wins" rule -- so the entries found here are the entries it collected.
    A renaming that the parent registers for ITSELF is never considered, even if a submodule
    registers the same one: re-scoping something the parent asked for by name would be a
    different change from the one upstream made.

    Returns `(signature -> (prefix, sample_keys), model_keys)`, empty when nothing leaked.
    """
    from transformers.conversion_mapping import extract_weight_conversions_for_model
    from transformers.core_model_loading import WeightRenaming
    from transformers.modeling_utils import PreTrainedModel

    def extract(module, prefix):
        # 5.6 to 5.9 take the submodule's dotted path as a second argument, everything else
        # takes the module alone. Asked of the function rather than of a version.
        try:
            return extract_weight_conversions_for_model(module, prefix)
        except TypeError:
            return extract_weight_conversions_for_model(module)

    model_keys = set()
    for name, _ in model.named_parameters(remove_duplicate = False):
        model_keys.add(name)
    for name, _ in model.named_buffers(remove_duplicate = False):
        model_keys.add(name)

    own = set()
    seen_model_types = set()
    own_conversions = extract(model, "")
    if own_conversions is not None:
        seen_model_types.add(getattr(model.config, "model_type", None))
        own.update(_renaming_signature(c) for c in own_conversions)

    leaked = {}
    for name, submodule in model.named_modules():
        if submodule is model or not name or not isinstance(submodule, PreTrainedModel):
            continue
        model_type = getattr(getattr(submodule, "config", None), "model_type", None)
        if model_type is None or model_type in seen_model_types:
            continue
        conversions = extract(submodule, name)
        if conversions is None:
            continue
        seen_model_types.add(model_type)
        sample_keys = _sample_submodule_keys(submodule, name)
        if not sample_keys:
            continue
        for conversion in conversions:
            if not isinstance(conversion, WeightRenaming):
                continue
            signature = _renaming_signature(conversion)
            if signature in own or signature in leaked:
                continue
            if _renaming_destroys_keys(conversion, sample_keys, model_keys):
                leaked[signature] = (name, sample_keys)
    return leaked, model_keys


def _rescope_conversions(model, conversions):
    """Replace every leaked renaming in `conversions` with its scoped form, or drop it."""
    from transformers.core_model_loading import WeightRenaming

    leaked, model_keys = _leaked_submodule_prefix_renamings(model)
    if not leaked:
        return conversions

    rescoped_conversions = []
    fixed = 0
    dropped = 0
    for conversion in conversions:
        scoped_by_transformers = isinstance(conversion, WeightRenaming) and (
            getattr(conversion, "scope_prefix", None) is not None
        )
        entry = (
            leaked.get(_renaming_signature(conversion))
            if isinstance(conversion, WeightRenaming) and not scoped_by_transformers
            else None
        )
        if entry is None:
            rescoped_conversions.append(conversion)
            continue
        prefix, sample_keys = entry
        replacement = _rescoped_renaming(conversion, prefix, sample_keys, model_keys)
        if replacement is None:
            dropped += 1
            continue
        rescoped_conversions.append(replacement)
        fixed += 1

    if fixed or dropped:
        logger.info(
            f"Unsloth: re-scoped {fixed} and dropped {dropped} checkpoint renaming(s) that "
            f"transformers merged into {type(model).__name__}'s conversion mapping from a "
            f"submodule, where they rewrite the model's own weight names into names it does "
            f"not have (transformers PR #45567, released in 5.6.0)"
        )
    return rescoped_conversions


def fix_transformers_composite_prefix_renaming():
    """Stop a pre-quantized multimodal checkpoint loading with no quantization metadata.

    transformers 5.4.0 (PR #44300) made `get_model_conversion_mapping` recurse into
    `PreTrainedModel` submodules and merge their conversion mappings into the parent's. The
    standalone Qwen3.5 / Qwen3.5-MoE / Gemma 3n text models register
    `^model.language_model.` -> `^model.` to strip a prefix their own checkpoints carry, and
    the recursion hands that renaming to the COMPOSITE model, whose weights really are named
    `model.language_model....`. Renamings run before the bitsandbytes converter, so every
    checkpoint key loses the `language_model.` segment: the packed weight is rescued by
    original key with no converter attached and loads as a bare uint8 parameter, while the
    `absmax`, `quant_map`, `nested_absmax`, `nested_quant_map` and
    `quant_state.bitsandbytes__nf4` sidecars match nothing and are discarded as unexpected.

    Measured with plain transformers and no unsloth in the process, on
    `unsloth/qwen3.8-27b-unsloth-bnb-4bit`: 352 of 352 `Linear4bit` modules come back with
    `quant_state is None` on 5.4.0 and on 5.5.4, and the first forward dies with
    `ValueError: quant_state is required` (`RuntimeError: mat1 and mat2 shapes cannot be
    multiplied (8x5120 and 1x15728640)` through Unsloth's own Linear4bit path). Gemma 3n is a
    second affected family (439 of 439). The published checkpoints are correct and must not be
    regenerated. Fixed upstream in 5.6.0 by PR #45567, so the window is 5.4.0 to 5.5.4 -- but
    excluding those releases does not help anyone whose environment pins one of them for
    another reason, and unsloth-zoo caps transformers at 5.5.0 on Apple Silicon, which puts
    every Mac user inside the window by construction. Hence a runtime fix.

    Two gates, neither of them a version number. The API probe above declines to install on a
    transformers that already re-scopes, and the wrapper itself only touches a renaming that
    demonstrably rewrites THIS model's own parameter names into names it does not have, so it
    is inert for a model with no submodel mapping, for a non-composite model whose mapping
    fires on nothing, and for the standalone text model the entry was written for.
    """
    if _transformers_rescopes_submodule_prefix_renamings():
        return
    # unsloth_zoo carries the same repair, in temporary_patches/conversion_mapping_rescope.py,
    # because it owns the bitsandbytes Linear4bit patch that reports this failure and is
    # importable without unsloth. When it installed first there is nothing left to do: a second
    # wrapper is measurably inert -- the first pass leaves no leaked signature for the second to
    # match -- but it is still a wrapper nobody needs, and one of the two has to yield. This one
    # does, so the newer implementation wins on a stack where the two versions disagree.
    if _zoo_composite_prefix_renaming_installed():
        return
    try:
        from transformers import conversion_mapping
    except Exception as e:
        logger.info(f"Unsloth: Skipping the composite renaming fix ({e})")
        return

    original = getattr(conversion_mapping, "get_model_conversion_mapping", None)
    if original is None:
        return
    # The mark travels on the wrapper, so a reload of conversion_mapping -- which puts the
    # upstream function back in a namespace any flag we set would survive -- re-patches rather
    # than being skipped.
    if getattr(original, _COMPOSITE_PREFIX_RENAMING_FLAG, False):
        return
    # Probe and wrap the ORIGINAL, never a wrapper of ours that lost its mark.
    original = getattr(original, "__wrapped__", original)

    @functools.wraps(original)
    def get_model_conversion_mapping(*args, **kwargs):
        conversions = original(*args, **kwargs)
        try:
            model = kwargs["model"] if "model" in kwargs else (args[0] if args else None)
            if model is None or not conversions:
                return conversions
            return _rescope_conversions(model, conversions)
        except Exception as e:
            # A mapping we could not reason about is still the mapping transformers built.
            logger.info(f"Unsloth: Could not re-scope the conversion mapping ({e})")
            return conversions

    # functools.wraps sets __wrapped__, but set it explicitly: the probe and the tests both
    # read it, and a wraps-less edit must not make the patch un-probeable and un-undoable.
    get_model_conversion_mapping.__wrapped__ = original
    setattr(get_model_conversion_mapping, _COMPOSITE_PREFIX_RENAMING_FLAG, True)

    try:
        conversion_mapping.get_model_conversion_mapping = get_model_conversion_mapping
        # Every module that already did `from .conversion_mapping import
        # get_model_conversion_mapping` holds the original object, not the attribute:
        # transformers.modeling_utils and transformers.integrations.peft do, and so does peft
        # itself. Modules that import it later pick the patched one up from the module above.
        # Only the packages that import this name from transformers. The binding test
        # below cannot be an identity test (see the comment on it), so without a module
        # restriction the sweep would replace ANY callable called
        # `get_model_conversion_mapping`, including one a notebook or a plugin defined
        # for itself before importing unsloth.
        owning_packages = ("transformers", "peft", "unsloth_zoo", "unsloth")
        upstream_module = getattr(conversion_mapping, "__name__", "transformers.conversion_mapping")
        for module_name, module in list(sys.modules.items()):
            if module is None or module is conversion_mapping:
                continue
            root = module_name.partition(".")[0]
            if root not in owning_packages:
                continue
            namespace = getattr(module, "__dict__", None)
            if not isinstance(namespace, dict):
                continue
            # `__dict__`, never `getattr`: transformers' lazy modules answer ANY attribute
            # name through `__getattr__`, which imports a submodule, prints a deprecation
            # notice and hands back an alias. Sweeping with getattr walks the whole model
            # zoo and says "Accessing get_model_conversion_mapping from
            # .models.vitmatte.image_processing_vitmatte" several hundred times. Only a
            # module that really did `from ... import get_model_conversion_mapping` has the
            # name in its own namespace, and those are the only ones that need rebinding.
            # Three things get rebound, and nothing else: `original` (the object we wrapped),
            # a function that IS upstream's -- it answers to `transformers.conversion_mapping`
            # as its module -- and a wrapper that carries unsloth_zoo's own marker.
            #
            # Not a plain identity test against `original`, because unsloth_zoo patches the
            # same function without functools.wraps and without `__wrapped__`
            # (temporary_patches/moe_utils_bnb4bit.py), so when zoo goes first `original` is
            # zoo's wrapper and the unwrap above cannot see past it. A module that imported
            # the name before zoo ran still holds the UNDERLYING upstream function and would
            # match neither object, leaving it bound to the unscoped mapping.
            # transformers.integrations.peft is exactly that module, imported during
            # modeling_utils init, and PeftAdapterMixin.load_adapter() then renames Qwen3.5
            # and Gemma 3n adapter keys away from their real `model.language_model.*` modules.
            #
            # And not "any callable under this name" either: these packages are large enough
            # that one of them, or a plugin living inside their namespace, may define its own
            # helper with the same name, and replacing that would be a bug of our own making.
            #
            # Rebinding zoo's own wrapper to ours does not drop zoo's fix: ours wraps
            # `original`, which IS zoo's wrapper when zoo went first, so its work still runs.
            try:
                bound = namespace.get("get_model_conversion_mapping", None)
                if not callable(bound) or bound is get_model_conversion_mapping:
                    continue
                is_alias = (
                    bound is original
                    or getattr(bound, "__module__", None) == upstream_module
                    or getattr(bound, "_unsloth_moe_patched", False)
                )
                if is_alias:
                    module.get_model_conversion_mapping = get_model_conversion_mapping
            except Exception:
                continue
        logger.info(
            "Unsloth: Patching transformers `get_model_conversion_mapping` so a pre-quantized "
            "multimodal checkpoint keeps its bitsandbytes quant_state (transformers PR #45567)"
        )
    except Exception as e:
        logger.info(f"Unsloth: Failed patching get_model_conversion_mapping ({e})")


_ROPE_SCALING_PATCH_FLAG = "_unsloth_patched_rope_scaling_setter"


def _rope_scaling_property_owner():
    """The class that DEFINES a writable ``rope_scaling`` property, or ``None``.

    transformers 5 keeps ``rope_scaling`` as an alias for ``rope_parameters``; 4.x has a
    plain attribute, so ``None`` there. The MRO is walked because the base config is
    spelled ``PretrainedConfig`` on 4.x and ``PreTrainedConfig`` on 5.x.
    """
    try:
        from transformers.configuration_utils import PretrainedConfig
    except Exception:
        return None
    for klass in getattr(PretrainedConfig, "__mro__", ()):
        prop = klass.__dict__.get("rope_scaling", None)
        if isinstance(prop, property) and prop.fset is not None:
            return klass
    return None


def _rope_probe_inherits(owner):
    """Does the config class the probe measures still descend from the LIVE owner?

    False only after a reload has replaced the owner underneath a cached config module,
    which is exactly when the probe's verdict describes a class nobody will use again.
    """
    try:
        from transformers import LlamaConfig
    except Exception:
        return False
    return isinstance(owner, type) and issubclass(LlamaConfig, owner)


def _rope_scaling_setter_is_patched(owner = None):
    """Is the LIVE ``rope_scaling`` setter ours, right now?

    Asked of the descriptor, not of a class flag: a reload of
    ``transformers.configuration_utils`` puts the upstream property back while a class
    attribute survives, so a flag would refuse to re-patch a vulnerable build.
    """
    if owner is None:
        owner = _rope_scaling_property_owner()
    if owner is None:
        return False
    prop = owner.__dict__.get("rope_scaling", None)
    if not isinstance(prop, property):
        return False
    return bool(getattr(prop.fset, _ROPE_SCALING_PATCH_FLAG, False))


_ROPE_LABELS_UNSET = object()


def _rope_nesting_labels(config):
    """Where ``rope_parameters`` nests: ``standardize_rope_params`` reads ``_rope_type_labels``,
    falling back to ``layer_types``. Different axes -- DeepseekV4 keys rope by main/compress."""
    labels = getattr(config, "_rope_type_labels", _ROPE_LABELS_UNSET)
    if labels is _ROPE_LABELS_UNSET:
        labels = getattr(config, "layer_types", None)
    return labels


def _rope_parameters_are_per_layer(config, parameters):
    """Nested per-label rope dict or one global one? Asked first: a global key inside a nested
    dict makes transformers read it flat. Its test is ``isdisjoint``, NOT ``issubset`` -- a
    nested dict may name labels this config's ``layer_types`` omits."""
    labels = _rope_nesting_labels(config)
    if not labels or not isinstance(parameters, dict) or not parameters:
        return False
    try:
        return not set(parameters.keys()).isdisjoint(set(labels))
    except Exception:
        return False


def _rope_theta_snapshot(config):
    """The base(s) ``rope_parameters`` holds right now, in the shape it holds them.

    Flat -> one base; nested -> a ``{label: base}`` mapping, valid only one level down.
    """
    parameters = getattr(config, "rope_parameters", None)
    if not isinstance(parameters, dict):
        return None
    if not _rope_parameters_are_per_layer(config, parameters):
        return parameters.get("rope_theta", None)
    snapshot = {}
    for layer_type, entry in parameters.items():
        if isinstance(entry, dict) and entry.get("rope_theta", None) is not None:
            snapshot[layer_type] = entry["rope_theta"]
    return snapshot or None


def _carry_per_layer_rope_theta(config, parameters, carried):
    """Put each layer type's base back into a per-layer replacement that lost it.

    A top-level ``rope_theta`` would make the dict read as flat, so the base goes back
    into each nested entry. Copies throughout, never the caller's dicts in place. An
    entry naming its own base is left alone. ``None`` means nothing was written -- no nested
    snapshot, OR every entry already named its base, OR refused; never "fall through as scalar".
    """
    if not isinstance(carried, dict) or not carried:
        return None
    replacement = {}
    changed = False
    for layer_type, entry in parameters.items():
        if (
            isinstance(entry, dict)
            and entry.get("rope_theta", None) is None
            and carried.get(layer_type, None) is not None
        ):
            entry = dict(entry)
            entry["rope_theta"] = carried[layer_type]
            changed = True
        replacement[layer_type] = entry
    if not changed:
        return None
    try:
        config.rope_parameters = replacement
    except Exception:
        return None
    return carried


def _carry_rope_theta_across_assignment(config, carried):
    """Keep the RoPE base frequency across a ``rope_scaling`` replacement.

    Separated from the wrapper so every parameter shape is testable without a live
    transformers.

    Precedence, so nothing a caller said is overwritten: the NEW parameters' base, then
    the config's own, then ``carried`` (the base held before the assignment). That order
    matters because ``unsloth_zoo/empty_model.py`` sets Gemma's local rotary base and
    then replaces the scaling, and carrying the global base over it would be wrong.

    The base is written to ``rope_parameters`` where 5.x keeps it, so the answer does
    not depend on ``standardize_rope_params`` running first and ``validate_rope`` and
    ``save_pretrained`` see a complete dict. A nested per-label dict is left alone: a
    ``{label: base}`` ``carried`` is restored one level down or dropped, never made a scalar.

    The config's ``rope_theta`` attribute (the 4.x slot) is written only when it is the
    only slot that can hold the base, which is the non-dict replacement of #2405, or
    when it already exists and would otherwise go stale. Otherwise a saved config gains
    no key it did not have. Writes nothing when the new parameters name their own base
    and the config states none, so this is a no-op on a build that keeps the base.
    """
    parameters = getattr(config, "rope_parameters", None)
    per_layer = _rope_parameters_are_per_layer(config, parameters)
    if per_layer:
        # Per-layer bases go back one level down. Falls through to the global attribute
        # below only when there is no nested snapshot to restore.
        restored = _carry_per_layer_rope_theta(config, parameters, carried)
        if restored is not None:
            return restored
        if isinstance(carried, dict):
            # Nothing restored = no-op or refused; a scalar fall-through put the MAPPING here.
            return None
    elif isinstance(carried, dict):
        # Per-layer parameters replaced by a FLAT dict. Passing the mapping through would
        # write a dict where a number belongs. One base stands for it only when every
        # layer type agreed; otherwise carry nothing rather than invent a scalar.
        bases = set(carried.values())
        carried = bases.pop() if len(bases) == 1 else None
    is_flat_dict = isinstance(parameters, dict) and not per_layer
    current = parameters.get("rope_theta", None) if is_flat_dict else None
    stated = getattr(config, "rope_theta", None)

    base = current
    if base is None:
        base = stated
    if base is None:
        base = carried
    if base is None:
        return None
    if isinstance(base, dict):
        # A base is a number. Also reachable from a config SAVED by the #11037 release,
        # whose own rope_theta holds a mapping: refuse, it needs fixing at rest instead.
        return None

    readable = current is not None
    if is_flat_dict and not readable:
        try:
            # A COPY, never the dict in place: transformers 5 stores the caller's object
            # verbatim, so one scaling dict reused across a loop of models would carry the
            # first config's base into the second. Measured on 5.0.0, 5.5.4 and 5.17.0: a
            # config declaring 10000.0 came out with 500000.0, silently.
            replacement = dict(parameters)
            replacement["rope_theta"] = base
            config.rope_parameters = replacement
            parameters = replacement
            readable = True
        except Exception:
            pass
    if stated is not None or not readable:
        try:
            config.rope_theta = base
        except Exception:
            return None
    return base


def _transformers_rope_scaling_assignment_drops_theta():
    """Does replacing ``config.rope_scaling`` on THIS build lose the base frequency?

    Measured, not inferred from a version: the move into ``rope_parameters``, the alias
    property and the strict validation landed in different 5.x releases and have been
    backported. Two configs, because ``ROPE_INIT_FUNCTIONS`` mutates the one it is
    handed. Absent positive evidence answer ``False`` and leave transformers alone.
    """
    try:
        import torch
        from transformers import LlamaConfig
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    except Exception:
        return False
    try:
        rope_init_fn = ROPE_INIT_FUNCTIONS["linear"]
    except Exception:
        return False
    scaling = {"rope_type": "linear", "factor": 4.0}

    def build():
        # Tiny, and a base far from the 10000.0 default so a silent fallback shows up
        # as a difference rather than a match.
        return LlamaConfig(
            hidden_size = 64,
            num_attention_heads = 2,
            num_key_value_heads = 2,
            head_dim = 32,
            rope_theta = 500000.0,
            max_position_embeddings = 2048,
            rope_scaling = dict(scaling),
        )

    # CPU explicitly: under a "meta" default device the comparison cannot be read, which
    # would answer "not affected" for a reason unrelated to the bug.
    device = torch.device("cpu")
    try:
        reference, _attention_factor = rope_init_fn(build(), device)
        subject = build()
    except Exception:
        return False
    try:
        subject.rope_scaling = dict(scaling)
    except Exception:
        return False
    try:
        replaced, _attention_factor = rope_init_fn(subject, device)
    except Exception:
        # The base came out None and `base ** positions` raised. This is the defect.
        return True
    try:
        return not bool(torch.allclose(replaced.float().cpu(), reference.float().cpu()))
    except Exception:
        return False


def fix_transformers_rope_scaling_drops_theta():
    """Stop a replaced ``rope_scaling`` silently unscaling RoPE (issue #2405).

    transformers 5 moved ``rope_theta`` into ``config.rope_parameters`` and kept
    ``rope_scaling`` as an alias that REPLACES that whole dict, so assigning a
    normalized scaling dict takes the base with it and
    ``_compute_linear_scaling_rope_parameters`` raises ``TypeError: unsupported operand
    type(s) for ** or pow(): 'NoneType' and 'Tensor'``. That assignment is what
    ``models/llama.py``'s ``_compute_config_rope_inv_freq`` retry does, and unsloth
    catches the raise and falls back to UNSCALED RoPE, which is #2405's gibberish past
    the original context.

    Measured here, one venv per transformers minor on torch 2.11.0:
    ``tests/utils/test_rope_scaling_drift.py`` fails on 5.0.0 through 5.3.0 and passes
    on 4.57.6, where ``rope_theta`` is an attribute the assignment cannot touch. From
    5.4.0 a config refuses a non-dict ``rope_parameters``, so the object route is
    unreachable, but any dict assignment still drops the base and ``save_pretrained``
    raises ``KeyError: Missing required keys ... {'rope_theta'}`` instead. The probe
    therefore fires on every 5.x.

    The alias setter is the only site where the outgoing and incoming parameters are
    both visible. Nothing is normalized or rejected on the way in; the wrapper only puts
    the base back where it would otherwise be gone. Assignments to ``rope_parameters``
    itself need no cover: a caller writing that field writes what transformers reads.
    """
    owner = _rope_scaling_property_owner()
    if owner is not None and _rope_scaling_setter_is_patched(owner):
        return
    # The probe answers through the cached `transformers.LlamaConfig`. After a reload of
    # configuration_utils that class still inherits the OLD patched owner while the live
    # owner is new and unpatched, so the probe would report the base survives and leave
    # the live owner to lose it. A probe that cannot see the owner is no evidence and
    # does not get to veto; installing anyway is safe, since the wrapper is a no-op on a
    # build that keeps the base.
    probe_can_see_owner = owner is None or _rope_probe_inherits(owner)
    if probe_can_see_owner and not _transformers_rope_scaling_assignment_drops_theta():
        return
    if owner is None:
        logger.info("Unsloth: Skipping the rope_scaling base-frequency fix (no alias property)")
        return

    prop = owner.__dict__["rope_scaling"]
    # Unwrap first, so a reload that restored upstream's property is re-patched rather
    # than wrapped on top of a wrapper.
    original = getattr(prop.fset, "__wrapped__", prop.fset)

    @functools.wraps(original)
    def rope_scaling(self, value):
        # Both halves are guarded, since this runs on every model load and an unexpected
        # config shape must not turn an assignment into a traceback. The assignment
        # itself is not, since swallowing that would be a silent behaviour change.
        try:
            carried = _rope_theta_snapshot(self)
        except Exception:
            carried = None
        result = original(self, value)
        try:
            _carry_rope_theta_across_assignment(self, carried)
        except Exception:
            pass
        return result

    # Set explicitly as well as by wraps: the unwrap above and the tests read it, so a
    # wraps-less edit must not make the patch un-probeable.
    rope_scaling.__wrapped__ = original
    # The mark travels ON the setter, so a reload that drops it is re-patched.
    setattr(rope_scaling, _ROPE_SCALING_PATCH_FLAG, True)

    try:
        setattr(owner, "rope_scaling", property(prop.fget, rope_scaling, prop.fdel, prop.__doc__))
        logger.info(
            "Unsloth: Patching transformers `rope_scaling` so replacing it keeps the "
            "RoPE base frequency (unsloth #2405)"
        )
    except Exception as e:
        logger.info(f"Unsloth: Failed patching rope_scaling ({e})")


# ValueError: 'aimv2' is already used by a Transformers config, pick another name.
def fix_vllm_aimv2_issue():
    spec = importlib.util.find_spec("vllm")
    if spec is None:
        return
    # A findable spec with unreadable dist metadata (broken/partial vllm install) must not crash `import unsloth`.
    try:
        vllm_version = importlib_version("vllm")
    except Exception as e:
        logger.info(f"Unsloth: Skipping vLLM aimv2 fix -- vLLM version unreadable ({e})")
        return
    if Version(vllm_version) < Version("0.10.1"):
        vllm_location = spec.origin
        if vllm_location is None:
            vllm_location = spec.submodule_search_locations[0]
        else:
            vllm_location = os.path.split(vllm_location)[0]
        ovis_config = Path(vllm_location) / "transformers_utils" / "configs" / "ovis.py"
        try:
            if ovis_config.exists():
                with open(ovis_config, "r+", encoding = "utf-8") as f:
                    text = f.read()
                    # See https://github.com/vllm-project/vllm-ascend/issues/2046
                    if 'AutoConfig.register("aimv2", AIMv2Config)' in text:
                        text = text.replace(
                            'AutoConfig.register("aimv2", AIMv2Config)',
                            "",
                        )
                        text = text.replace(
                            """backbone_config.pop('model_type')
                backbone_config = AutoConfig.for_model(model_type,
                                                       **backbone_config)""",
                            """if model_type != "aimv2":
                    backbone_config.pop('model_type')
                    backbone_config = AutoConfig.for_model(model_type, **backbone_config)
                else:
                    backbone_config = AIMv2Config(**backbone_config)""",
                        )
                        f.seek(0)
                        f.write(text)
                        f.truncate()
                        logger.info(
                            "Unsloth: Patching vLLM to fix `'aimv2' is already used by a Transformers config, pick another name.`"
                        )
        except Exception as e:
            logger.info(f"Unsloth: Failed patching vLLM with error = {str(e)}")


# vLLM >= 0.22 (PR #35024) deleted vllm.transformers_utils.tokenizer, which an older
# unsloth_zoo still imports unguarded (#6385). Stub it with a meta path finder appended AFTER
# the real ones, so it only fires when vLLM no longer ships the module.
_VLLM_LORA_TOKENIZER_MODULE = "vllm.transformers_utils.tokenizer"
_VLLM_TOKENIZER_STUB_SENTINEL = "__unsloth_vllm_tokenizer_stub__"


def _unsloth_return_no_lora_tokenizer(*args, **kwargs):
    # None -> vLLM uses the base tokenizer for LoRA (matches unsloth_zoo).
    return None


class _VllmLoraTokenizerStubLoader(importlib.abc.Loader):
    __slots__ = ("module_name",)

    def __init__(self, module_name):
        self.module_name = module_name

    def create_module(self, spec):
        import types

        module = types.ModuleType(self.module_name)
        module.__file__ = f"<unsloth stub: {self.module_name}>"
        module.__package__ = self.module_name.rpartition(".")[0]
        setattr(module, _VLLM_TOKENIZER_STUB_SENTINEL, True)
        module.get_lora_tokenizer = _unsloth_return_no_lora_tokenizer
        module.get_lora_tokenizer_async = _unsloth_return_no_lora_tokenizer
        return module

    def exec_module(self, module):
        return None


class _VllmLoraTokenizerStubFinder(importlib.abc.MetaPathFinder):
    __slots__ = (_VLLM_TOKENIZER_STUB_SENTINEL,)

    def __init__(self):
        setattr(self, _VLLM_TOKENIZER_STUB_SENTINEL, True)

    def find_spec(
        self,
        fullname,
        path = None,
        target = None,
    ):
        if fullname != _VLLM_LORA_TOKENIZER_MODULE:
            return None
        return importlib.machinery.ModuleSpec(
            name = fullname,
            loader = _VllmLoraTokenizerStubLoader(fullname),
            is_package = False,
        )


def fix_vllm_lora_tokenizer_module():
    if importlib.util.find_spec("vllm") is None:
        return
    for finder in sys.meta_path:
        if getattr(finder, _VLLM_TOKENIZER_STUB_SENTINEL, False):
            return
    # Appended, not inserted at 0, so a real module on older vLLM always wins.
    sys.meta_path.append(_VllmLoraTokenizerStubFinder())
    logger.info(
        "Unsloth: Installed `vllm.transformers_utils.tokenizer` compatibility "
        "stub for newer vLLM versions"
    )


def fix_vllm_guided_decoding_params():
    def _maybe_raise_vllm_transformers_mismatch(error):
        error_text = str(error)
        if "ALLOWED_LAYER_TYPES" in error_text or "transformers.configuration_utils" in error_text:
            try:
                vllm_version = importlib_version("vllm")
            except Exception:
                vllm_version = "unknown"
            raise RuntimeError(
                "Unsloth: vLLM with version "
                f"{vllm_version} does not yet support transformers>=5.0.0. "
                "Please downgrade to transformers==4.57.3 via "
                'pip install --force-reinstall "transformers==4.57.3". '
                f"Original error: {error}"
            ) from error

    if importlib.util.find_spec("vllm") is None:
        return
    # GuidedDecodingParams was renamed to StructuredOutputsParams in vLLM (vllm#22772); trl still wants the old name.
    try:
        import vllm
    except (ImportError, OSError) as e:
        _maybe_raise_vllm_transformers_mismatch(e)
        if disable_broken_vllm(e):
            return
        raise

    try:
        from vllm.sampling_params import GuidedDecodingParams
    except (ImportError, OSError) as e:
        _maybe_raise_vllm_transformers_mismatch(e)
        if disable_broken_vllm(e):
            return
        if not hasattr(vllm, "sampling_params") or not hasattr(
            vllm.sampling_params, "StructuredOutputsParams"
        ):
            raise
        vllm.sampling_params.GuidedDecodingParams = vllm.sampling_params.StructuredOutputsParams


def fix_trl_vllm_ascend():
    # transformers >= 4.48's _is_package_available returns (bool, version_or_None), which TRL
    # caches and returns directly. A non-empty tuple is truthy, so `if is_X_available():` fires
    # for an absent X and triggers a failing import -- vllm_ascend blocked `from trl import
    # GRPOConfig` off Ascend hosts. Coerce every tuple-cached flag to bool.
    if importlib.util.find_spec("trl") is None:
        return
    try:
        import trl.import_utils as tiu
    except Exception:
        return
    for attr in list(vars(tiu)):
        if not (attr.startswith("_") and attr.endswith("_available")):
            continue
        cached = getattr(tiu, attr)
        if isinstance(cached, tuple):
            setattr(tiu, attr, bool(cached and cached[0]))


def ignore_logger_messages():
    try:
        from huggingface_hub._login import logger as huggingface_hub_logger
        huggingface_hub_logger.addFilter(HideLoggingMessage("`HF_TOKEN`"))
        del huggingface_hub_logger
    except:
        pass


# huggingface_hub routes to Xet on `is_xet_available()`, which asks importlib.metadata whether the
# hf_xet DISTRIBUTION is installed and never whether it IMPORTS, so an unloadable hf_xet is still
# routed to and dies as transformers' misleading "you need to install the hf_xet package".
# HF_HUB_DISABLE_XET=1 is the Hub's own way off that path, and constants.py freezes it at import
# time, which is why this is wired into the FIRST block of _gpu_init.py, not the late fixes.
# unsloth_zoo.hf_xet_fallback does not cover this: it is a stall watchdog around its own wrappers,
# and an ImportError at import time never reaches it.
_CPU_FAMILY_BY_MACHINE = {
    "amd64": "x86_64",
    "x86_64": "x86_64",
    "x64": "x86_64",
    "arm64": "arm64",
    "aarch64": "arm64",
    "i386": "x86",
    "i686": "x86",
    "x86": "x86",
    "armv7l": "armv7l",
    "ppc64le": "ppc64le",
    "s390x": "s390x",
}
# Windows tags name the cpu outright; every other OS carries it as the tag's last token.
_WINDOWS_PLATFORM_TAG_CPU = {
    "win_amd64": "x86_64",
    "win_arm64": "arm64",
    "win32": "x86",
}
_MULTI_ARCH_PLATFORM_TAG_PREFIXES = ("macosx_", "manylinux", "musllinux", "linux_")
# Header machine ids, for installs whose WHEEL metadata cannot be read. A family absent here reads
# as "cannot tell", which is never a mismatch.
_ELF_MACHINE_CPU = {
    0x03: "x86",
    0x15: "ppc64le",
    0x16: "s390x",
    0x28: "armv7l",
    0x3E: "x86_64",
    0xB7: "arm64",
}
_PE_MACHINE_CPU = {0x014C: "x86", 0x01C4: "armv7l", 0x8664: "x86_64", 0xAA64: "arm64"}
# huggingface_hub.constants.ENV_VARS_TRUE_VALUES, matched after .upper(), so an explicit disable
# is read exactly the way the Hub itself would have read it.
_HF_HUB_ENV_TRUE_VALUES = ("1", "ON", "TRUE", "YES")
_MACHO_CPU_TYPE_CPU = {7: "x86", 12: "armv7l", 0x01000007: "x86_64", 0x0100000C: "arm64"}


def _host_cpu_family():
    """CPU family of THIS interpreter, or None when it cannot be told.

    sysconfig first: it reports what the interpreter was BUILT for, which is what a wheel tag must
    match. platform.machine() is WRONG on Windows, where CPython asks WMI for the PHYSICAL cpu, so
    an emulated x86-64 process on Windows on ARM answers "ARM64" while being win-amd64. It stays
    only as the fallback for platforms sysconfig cannot name a cpu for, chiefly macOS universal2.
    """
    import platform
    import sysconfig

    try:
        host_platform = sysconfig.get_platform().strip().lower()
    except Exception:
        host_platform = ""
    if host_platform:
        family = _WINDOWS_PLATFORM_TAG_CPU.get(host_platform.replace("-", "_"))
        if family is None:
            # linux-x86_64, macosx-11.0-arm64: cpu is the last token.
            family = _CPU_FAMILY_BY_MACHINE.get(host_platform.rsplit("-", 1)[-1])
        if family is not None:
            return family
    try:
        return _CPU_FAMILY_BY_MACHINE.get(platform.machine().strip().lower())
    except Exception:
        return None


def _cpu_family_from_compiled_object(path):
    """CPU family an ELF / PE / Mach-O file targets, read from its header, or None."""
    import struct

    try:
        with open(path, "rb") as binary:
            head = binary.read(4096)
    except Exception:
        return None
    try:
        if head[:4] == b"\x7fELF":
            endian = "<" if head[5] == 1 else ">"
            return _ELF_MACHINE_CPU.get(struct.unpack_from(endian + "H", head, 18)[0])
        if head[:2] == b"MZ":
            pe_offset = struct.unpack_from("<I", head, 0x3C)[0]
            if head[pe_offset : pe_offset + 4] != b"PE\0\0":
                return None
            return _PE_MACHINE_CPU.get(struct.unpack_from("<H", head, pe_offset + 4)[0])
        magic = struct.unpack_from("<I", head, 0)[0]
        if magic in (0xFEEDFACE, 0xFEEDFACF):  # little endian Mach-O, 32 and 64 bit
            return _MACHO_CPU_TYPE_CPU.get(struct.unpack_from("<i", head, 4)[0] & 0xFFFFFFFF)
        if magic in (0xCEFAEDFE, 0xCFFAEDFE):  # byte swapped
            return _MACHO_CPU_TYPE_CPU.get(struct.unpack_from(">i", head, 4)[0] & 0xFFFFFFFF)
    except Exception:
        return None
    return None  # Mach-O universal binaries carry several cpus: unreadable on purpose.


def _hf_xet_extension_cpu_families(spec):
    """CPU families of hf_xet's compiled extensions, or () when unreadable. The fallback when the
    WHEEL metadata is gone: the .pyd/.so still says what it was built for."""
    families = set()
    for location in list(getattr(spec, "submodule_search_locations", None) or []):
        try:
            names = os.listdir(location)
        except Exception:
            continue
        for name in names:
            if not name.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
                continue
            family = _cpu_family_from_compiled_object(os.path.join(location, name))
            if family is None:
                return ()
            families.add(family)
    return tuple(sorted(families))


def _cpu_family_from_platform_tag(platform_tag):
    """CPU family a wheel platform tag targets, or None when it cannot be told: a tag we cannot
    read (`any`, macOS universal2) must never be reported as a mismatch."""
    platform_tag = platform_tag.strip().lower()
    if platform_tag in _WINDOWS_PLATFORM_TAG_CPU:
        return _WINDOWS_PLATFORM_TAG_CPU[platform_tag]
    if not platform_tag.startswith(_MULTI_ARCH_PLATFORM_TAG_PREFIXES):
        return None
    for machine, family in _CPU_FAMILY_BY_MACHINE.items():
        if platform_tag.endswith("_" + machine):
            return family
    return None


def _hf_xet_distribution_is_installed():
    """True when importlib.metadata sees an hf_xet distribution, the only question
    huggingface_hub's is_xet_available() asks before routing a download to Xet."""
    try:
        importlib_version("hf_xet")
    except Exception:
        return False
    return True


def _hf_xet_wheel_platform_tags():
    """Platform components of the installed hf_xet's WHEEL tags, or () when unreadable."""
    try:
        wheel_metadata = importlib_distribution("hf_xet").read_text("WHEEL")
    except Exception:
        return ()
    if not wheel_metadata:
        return ()
    platform_tags = []
    for line in wheel_metadata.splitlines():
        if not line.lower().startswith("tag:"):
            continue
        tag = line.split(":", 1)[1].strip()
        parts = tag.rsplit("-", 1)
        if len(parts) != 2 or not parts[1]:
            continue
        # compressed tag sets join platforms with a dot
        platform_tags.extend(parts[1].split("."))
    return tuple(platform_tags)


def _hf_xet_architecture_mismatch(spec = None):
    """True if hf_xet targets a CPU family this interpreter is not, False if it matches, None when
    either side cannot be read. Metadata and file headers only: nothing is imported here."""
    host_family = _host_cpu_family()
    if host_family is None:
        return None
    wheel_families = set()
    for platform_tag in _hf_xet_wheel_platform_tags():
        family = _cpu_family_from_platform_tag(platform_tag)
        if family is None:
            return None  # one unreadable tag makes the whole verdict unsafe
        wheel_families.add(family)
    if not wheel_families:
        # No readable WHEEL metadata. The extension's own header is the better source anyway.
        wheel_families = set(_hf_xet_extension_cpu_families(spec))
    if not wheel_families:
        return None
    return host_family not in wheel_families


def _hf_xet_extension_is_missing(spec):
    """True only when hf_xet's package directory holds no compiled extension at all."""
    locations = list(getattr(spec, "submodule_search_locations", None) or [])
    if not locations:
        return False
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    for location in locations:
        try:
            names = os.listdir(location)
        except Exception:
            return False
        if any(name.endswith(suffixes) for name in names):
            return False
    return True


def _disable_xet_on_already_imported_huggingface_hub():
    """Turn Xet off in an ALREADY imported huggingface_hub, returning the bindings changed.

    The variable alone is not enough: >= 0.34 freezes it into `constants.HF_HUB_DISABLE_XET` at
    import time. Readers go through the module attribute, never a local copy, so rebinding does
    take effect. Every module is scanned, not constants.py alone, so a future
    `from .constants import HF_HUB_DISABLE_XET` is covered; versions without the attribute are
    left alone rather than given one.
    """
    if "huggingface_hub" not in sys.modules:
        # Nothing frozen yet, and importing the Hub just to patch it would cost every user.
        return ()

    patched = []
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        if name != "huggingface_hub" and not name.startswith("huggingface_hub."):
            continue
        try:
            current = getattr(module, "HF_HUB_DISABLE_XET")
        except Exception:
            continue  # no binding here to fix, and we must not invent one
        if current is True:
            continue
        try:
            setattr(module, "HF_HUB_DISABLE_XET", True)
        except Exception:
            continue
        patched.append(name)
    return tuple(patched)


def _hf_xet_module_is_real(module):
    """False for the empty namespace package an incomplete hf_xet directory produces.

    A namespace package imports cleanly and defines nothing, so it satisfies neither
    `import hf_xet` nor huggingface_hub's `from hf_xet import PyXetDownloadInfo, download_files`.
    Its __file__ is None, which every genuinely installed package (extension modules included)
    sets, so that is the whole test.
    """
    return getattr(module, "__file__", None) is not None


def fix_broken_hf_xet_wheel():
    """Route Hugging Face downloads over plain HTTPS when hf_xet is installed but unimportable."""
    explicit = os.environ.get("HF_HUB_DISABLE_XET", "").strip()
    if explicit != "":
        # Set explicitly, in either direction: the user's choice outranks ours. A TRUTHY value set
        # after the Hub was already imported is read by nobody, though, so carry it to the frozen
        # constant. Falsey values are left alone, so an explicit opt back IN to Xet still wins.
        if explicit.upper() in _HF_HUB_ENV_TRUE_VALUES:
            _disable_xet_on_already_imported_huggingface_hub()
        return
    cached = sys.modules.get("hf_xet", None)
    if cached is not None:
        # Already imported, so it works, and repeat calls stay free. Unless it is the namespace
        # shell: huggingface_hub's own `from hf_xet import ...` leaves that cached even though the
        # symbols it wanted were missing, so a cached module is not proof of a working one.
        if _hf_xet_module_is_real(cached):
            return
        sys.modules.pop("hf_xet", None)
    try:
        if importlib.util.find_spec("huggingface_hub") is None:
            return
        spec = importlib.util.find_spec("hf_xet")
    except Exception:
        return
    if spec is None:
        # find_spec cannot see it, but huggingface_hub never asks find_spec: a leftover
        # hf_xet-*.dist-info with no package beside it still reports Xet available and still dies
        # in `from hf_xet import XetFileInfo`. A genuinely absent hf_xet has no metadata either
        # and is left alone, since huggingface_hub handles that case correctly.
        if not _hf_xet_distribution_is_installed():
            return
        suspicion = (
            "its distribution metadata is installed but the package itself is not importable"
        )
    elif _hf_xet_architecture_mismatch(spec) is True:
        suspicion = "it is built for a different CPU architecture than this interpreter"
    elif _hf_xet_extension_is_missing(spec):
        suspicion = "its compiled extension module is missing from the installed package"
    else:
        return

    # Confirm the suspicion by importing. If it imports after all, we misread the metadata and Xet
    # stays on. Only reached for an already-suspect install, so a healthy one never pays for it.
    #
    # Importing the package is not enough on its own: an hf_xet/ directory with no __init__.py and
    # no extension in it is a NAMESPACE package, which imports fine and defines nothing, while
    # huggingface_hub does `from hf_xet import PyXetDownloadInfo, download_files` and fails. So the
    # module has to carry real code, which a namespace package never does (its __file__ is None).
    try:
        module = importlib.import_module("hf_xet")
        if _hf_xet_module_is_real(module):
            return
        sys.modules.pop("hf_xet", None)
        failure = "ImportError: hf_xet resolves to an empty namespace package, which defines none of the symbols huggingface_hub imports"
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"

    # The variable is what child processes inherit; the helper is what fixes THIS process when
    # something already imported the Hub and froze the constant.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    _disable_xet_on_already_imported_huggingface_hub()
    logger.warning(
        f"Unsloth: `hf_xet` is installed but cannot be imported ({failure}) because {suspicion} "
        f"(wheel platform tag(s): {', '.join(_hf_xet_wheel_platform_tags()) or 'unknown'}). "
        "Set HF_HUB_DISABLE_XET=1, so Hugging Face downloads use plain HTTPS instead: same files, "
        "same cache, only the Xet transport is skipped. Note that without this, transformers "
        "reports 'you need to install the hf_xet package', which is misleading: hf_xet IS "
        "installed, it is simply not loadable by this Python, and reinstalling the same wheel "
        "cannot fix it. To get Xet back, install an hf_xet built for this interpreter, for example "
        "`pip install --force-reinstall --no-cache-dir hf_xet`."
    )


def patch_ipykernel_hf_xet():
    # HF-XET 1.1.10 with ipykernel 7.0.0 / 7.0.1 raises LookupError on ContextVar 'shell_parent';
    # see huggingface/xet-core#526.
    if importlib.util.find_spec("hf_xet") is None:
        return
    if importlib.util.find_spec("ipykernel") is None:
        return
    if importlib.util.find_spec("huggingface_hub") is None:
        return

    ipykernel_version = Version(importlib_version("ipykernel"))
    if (Version(importlib_version("hf_xet")) == Version("1.1.10")) and (
        (ipykernel_version == Version("7.0.0")) or (ipykernel_version == Version("7.0.1"))
    ):
        print(
            "#### Unsloth: `hf_xet==1.1.10` and `ipykernel==7.0.0` or `ipykernel==7.0.1` breaks progress bars. Using ASCII progress bars.\n"
            "#### Unsloth: To re-enable progress bars, please upgrade to `ipykernel>=7.1.0` or wait for a fix to\n"
            "https://github.com/huggingface/xet-core/issues/526"
        )
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()


def patch_trackio():
    # Customize the Trackio dashboard for experiment tracking; see unslothai/notebooks#110.
    os.environ["TRACKIO_LOGO_LIGHT_URL"] = (
        "https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20black%20text.png"
    )
    os.environ["TRACKIO_LOGO_DARK_URL"] = (
        "https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20white%20text.png"
    )
    os.environ["TRACKIO_PLOT_ORDER"] = "train/reward"


def patch_datasets():
    # Datasets 4.4.0 and 4.4.1 weirdly have some weird `_thread.RLock_recursion_count` issues
    if importlib.util.find_spec("datasets") is None:
        return

    datasets_version = Version(importlib_version("datasets"))
    if (datasets_version <= Version("4.5.0")) and (datasets_version >= Version("4.4.0")):
        raise NotImplementedError(
            f"#### Unsloth: Using `datasets = {str(datasets_version)}` will cause recursion errors.\n"
            "Please downgrade datasets to `datasets==4.3.0"
        )


# psutil divides the pmgr voltage-states tables by 1e6 for MHz, but Apple switched them from
# Hz to kHz on M4, so psutil <= 7.2.2 reads a 4.5 GHz M4 Pro as 4 MHz (#8519). Upstream fix
# giampaolo/psutil#2824 is merged and unreleased; mirrored here and in
# studio/backend/utils/hardware/hardware.py. Keep both in sync; delete once psutil is fixed.
# Apple clocks are 0.6-4.6 GHz, so a raw Hz entry sits above 1e8 and kHz below.
_APPLE_CPU_FREQ_UNIT_THRESHOLD = 100_000_000
_APPLE_MIN_PLAUSIBLE_CPU_MHZ = 500
_APPLE_MAX_PLAUSIBLE_CPU_MHZ = 20000
# Below this a table is a GPU/NPU rail: above every Apple GPU peak so far, under the slowest
# CPU cluster shipped (M1 E-core, 2064 MHz).
_APPLE_CPU_CLUSTER_MIN_PEAK_MHZ = 2000
_APPLE_VOLTAGE_STATES_KEY = re.compile(r"^voltage-states\d+-sram$")
# Fixed for the life of the host, so probe once. The sentinel separates "not probed yet" from "probed, unavailable".
_apple_cpu_freq_range = "unprobed"
# At import, not on first use: two threads reaching a lazy initialiser build a lock each and then exclude nothing.
_apple_cpu_freq_lock = threading.Lock()


def _apple_voltage_state_freqs_mhz(blob):
    """Plausible MHz from one voltage-statesN-sram blob.

    Entries are 8 bytes: little-endian uint32 frequency, then uint32 voltage.
    """
    freqs = []
    for offset in range(0, len(blob) - 7, 8):
        raw = int.from_bytes(blob[offset : offset + 4], "little")
        if raw == 0:
            continue
        mhz = raw / 1e6 if raw > _APPLE_CPU_FREQ_UNIT_THRESHOLD else raw / 1e3
        if _APPLE_MIN_PLAUSIBLE_CPU_MHZ <= mhz <= _APPLE_MAX_PLAUSIBLE_CPU_MHZ:
            freqs.append(mhz)
    return freqs


def _apple_cpu_freq_range_from_ioreg_entries(entries):
    """(min_mhz, max_mhz) across the CPU-cluster tables, or None."""
    lows, peaks = [], []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            if not isinstance(value, (bytes, bytearray)):
                continue
            if not _APPLE_VOLTAGE_STATES_KEY.match(str(key)):
                continue
            freqs = _apple_voltage_state_freqs_mhz(bytes(value))
            # M5 renumbered the indexes, so classify by peak, not by index.
            if freqs and max(freqs) >= _APPLE_CPU_CLUSTER_MIN_PEAK_MHZ:
                lows.append(min(freqs))
                peaks.append(max(freqs))
    if not peaks:
        return None
    return (min(lows), max(peaks))


def _apple_cpu_freq_range_mhz():
    """Read (min, max) CPU MHz from the pmgr IORegistry node, cached."""
    global _apple_cpu_freq_range
    if _apple_cpu_freq_range != "unprobed":
        return _apple_cpu_freq_range

    # Unlocked, every thread spawns its own ioreg and a slow failing probe landing last overwrites
    # a good reading with None.
    with _apple_cpu_freq_lock:
        if _apple_cpu_freq_range != "unprobed":
            return _apple_cpu_freq_range

        freq_range = None
        try:
            import plistlib
            import subprocess

            result = subprocess.run(
                ["ioreg", "-a", "-r", "-c", "AppleARMIODevice", "-d", "1"],
                capture_output = True,
                timeout = 2,
            )
            entries = plistlib.loads(result.stdout) if result.stdout else []
            if isinstance(entries, dict):
                entries = [entries]
            freq_range = _apple_cpu_freq_range_from_ioreg_entries(entries)
        except Exception as exception:
            logger.info("Unsloth: could not read Apple CPU frequencies from ioreg (%s)", exception)

        _apple_cpu_freq_range = freq_range
        return freq_range


def _corrected_apple_cpu_freq(sample):
    """Rescale one psutil scpufreq sample that was read in the wrong unit."""
    current = getattr(sample, "current", None)
    usable = isinstance(current, (int, float)) and current == current and current > 0
    if usable and current >= _APPLE_MIN_PLAUSIBLE_CPU_MHZ:
        return sample  # already plausible: a fixed psutil, or not an affected chip

    freq_range = _apple_cpu_freq_range_mhz()
    if freq_range is not None:
        low, peak = freq_range
        # macOS has no per-instant clock, so psutil reports the peak as `current`.
        return sample._replace(current = peak, min = low, max = peak)
    if not usable:
        return sample
    # No tables: recover the magnitude instead. psutil truncates in integer arithmetic, so this
    # lands on the GHz step (4 -> 4000 MHz), not the peak.
    return sample._replace(
        current = current * 1000,
        min = getattr(sample, "min", 0.0) * 1000,
        max = getattr(sample, "max", 0.0) * 1000,
    )


def patch_psutil_cpu_freq():
    """Fix psutil.cpu_freq() reporting kHz-derived MHz on Apple Silicon M4+."""
    if sys.platform != "darwin":
        return
    import platform as _platform

    if _platform.machine() != "arm64":
        return  # Rosetta / Intel Macs read a different, unaffected code path
    if importlib.util.find_spec("psutil") is None:
        return
    try:
        import psutil
    except Exception:
        return

    original_cpu_freq = getattr(psutil, "cpu_freq", None)
    if original_cpu_freq is None or getattr(original_cpu_freq, "__unsloth_patched__", False):
        return

    def _percpu_requested(args, kwargs):
        return bool(args[0]) if args else bool(kwargs.get("percpu", False))

    def _scpufreq_type():
        # psutil 7.x keeps the namedtuple in _ntuples, older releases in _common.
        for module_name in ("_ntuples", "_common"):
            namedtuple_type = getattr(getattr(psutil, module_name, None), "scpufreq", None)
            if namedtuple_type is not None:
                return namedtuple_type
        return None

    def _from_tables(percpu):
        """The IORegistry reading in psutil's own return shape, or None.

        macOS has no per-core clock, so psutil's percpu answer is a one-element
        list of the same sample; matching that keeps percpu callers whole.
        """
        namedtuple_type = _scpufreq_type()
        if namedtuple_type is None:
            return None
        freq_range = _apple_cpu_freq_range_mhz()
        if freq_range is None:
            return None
        low, peak = freq_range
        sample = namedtuple_type(peak, low, peak)
        return [sample] if percpu else sample

    @functools.wraps(original_cpu_freq)
    def cpu_freq(*args, **kwargs):
        try:
            result = original_cpu_freq(*args, **kwargs)
        except TypeError:
            # Arguments psutil does not take are the caller's mistake, not psutil declining to answer.
            # This replaces psutil's function globally, so swallowing would hide the error everywhere.
            raise
        except Exception:
            # psutil raises on M5, whose renumbered tables are not at the indexes it hardcodes. With no
            # tables either, its error is the honest answer.
            stand_in = _from_tables(_percpu_requested(args, kwargs))
            if stand_in is None:
                raise
            return stand_in
        try:
            # percpu = True returns a list of samples, otherwise a single one.
            if isinstance(result, list):
                # Empty is giampaolo/psutil#2382's "undeterminable".
                if not result:
                    return _from_tables(percpu = True) or result
                return [_corrected_apple_cpu_freq(sample) for sample in result]
            if result is None:
                stand_in = _from_tables(percpu = False)
                return result if stand_in is None else stand_in
            return _corrected_apple_cpu_freq(result)
        except Exception:
            return result  # never let a cosmetic fix break a caller

    cpu_freq.__unsloth_patched__ = True
    psutil.cpu_freq = cpu_freq


def check_fbgemm_gpu_version():
    if importlib.util.find_spec("fbgemm_gpu") is None:
        return
    try:
        fbgemm_gpu_version = importlib_version("fbgemm_gpu_genai")
    except:
        return
    # Lower fbgemm_gpu versions segfault or bad-alloc, so disable FBGEMM and fall back to Triton
    # rather than raise.
    if Version(fbgemm_gpu_version) < Version("1.4.0"):
        os.environ["UNSLOTH_HAS_FBGEMM"] = "0"
        logger.info(
            f"Unsloth: fbgemm_gpu_genai=={fbgemm_gpu_version} is old and may cause issues. "
            f"Disabling FBGEMM - using Triton kernels instead."
        )
        return

    logger.info(f"Unsloth: fbgemm_gpu_genai=={fbgemm_gpu_version} detected.")


def patch_enable_input_require_grads():
    """Patch PreTrainedModel.enable_input_require_grads to tolerate vision models
    that raise NotImplementedError from get_input_embeddings()."""
    import inspect
    from transformers import PreTrainedModel

    # Only patch the new variant that iterates over self.modules(); see huggingface/transformers#41993.
    try:
        original_source = inspect.getsource(PreTrainedModel.enable_input_require_grads)
    except:
        return

    if "for module in self.modules()" not in original_source:
        return

    def _patched_enable_input_require_grads(self):
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        hooks = []
        seen_modules = set()

        for module in self.modules():
            if not (
                isinstance(module, PreTrainedModel) and hasattr(module, "get_input_embeddings")
            ):
                continue

            try:
                input_embeddings = module.get_input_embeddings()
            except NotImplementedError:
                # Vision models may not implement get_input_embeddings (GLM V4.6 skips only self.visual).
                continue

            if input_embeddings is None:
                continue

            embedding_id = id(input_embeddings)
            if embedding_id in seen_modules:
                continue

            seen_modules.add(embedding_id)
            hooks.append(input_embeddings.register_forward_hook(make_inputs_require_grads))

        self._require_grads_hooks = hooks
        if hooks:
            self._require_grads_hook = hooks[0]

    PreTrainedModel.enable_input_require_grads = _patched_enable_input_require_grads

    logger.info("Unsloth: Patched enable_input_require_grads for vision model compatibility")


def patch_unsafe_trainer_rng_load():
    """Harden Trainer._load_rng_state against CVE-2026-1839 (RCE from a malicious
    rng_state.pth on resume). Hardens only the rng torch.load, via a thread-local
    flag, so it forces weights_only=True (defeats TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD)
    and refuses torch < 2.6 (CVE-2025-32434), while rng-less resumes and unrelated
    torch.load calls are untouched. No-op if transformers is absent or already
    guards the load (>= 5.0.0rc3)."""
    if importlib.util.find_spec("transformers") is None:
        return
    try:
        from transformers.trainer import Trainer
    except Exception:
        return
    load_rng_state = getattr(Trainer, "_load_rng_state", None)
    if load_rng_state is None or getattr(load_rng_state, "_unsloth_safe_rng_load", False):
        return
    try:
        source = inspect.getsource(load_rng_state)
    except Exception:
        return
    if "torch.load" not in source or "check_torch_load_is_safe" in source:
        return

    import threading, torch

    try:
        # Older supported transformers (>= 4.51.3) may not export the helper.
        from transformers.utils.import_utils import check_torch_load_is_safe
    except Exception:

        def check_torch_load_is_safe():
            if TrueVersion(torch.__version__.split("+")[0]) < TrueVersion("2.6"):
                raise RuntimeError(
                    "Unsloth: refusing to load checkpoint RNG state on torch < 2.6 "
                    "(CVE-2026-1839 / CVE-2025-32434); upgrade to torch >= 2.6."
                )

    # One process-wide torch.load shim, inert unless the calling thread is inside _load_rng_state,
    # so the gate applies at the real rng load with no global-swap race.
    if not getattr(torch.load, "_unsloth_rng_guard", False):
        _orig_load = torch.load
        _rng_active = threading.local()

        @functools.wraps(_orig_load)
        def _guarded_torch_load(*args, **kwargs):
            if getattr(_rng_active, "on", False):
                check_torch_load_is_safe()  # raises on torch < 2.6 (CVE-2025-32434)
                kwargs.setdefault("weights_only", True)
            return _orig_load(*args, **kwargs)

        _guarded_torch_load._unsloth_rng_guard = True
        _guarded_torch_load._unsloth_rng_flag = _rng_active
        torch.load = _guarded_torch_load
    _rng_active = torch.load._unsloth_rng_flag

    @functools.wraps(load_rng_state)
    def _unsloth_safe_load_rng_state(self, checkpoint):
        _rng_active.on = True
        try:
            return load_rng_state(self, checkpoint)
        finally:
            _rng_active.on = False

    _unsloth_safe_load_rng_state._unsloth_safe_rng_load = True
    Trainer._load_rng_state = _unsloth_safe_load_rng_state
    logger.info("Unsloth: Hardened Trainer._load_rng_state rng loading (CVE-2026-1839).")


def _is_custom_torch_build(raw_version_str):
    """Check if a raw version string indicates a custom or source build.

    Operates on the raw importlib_version() string (our Version() strips local
    identifiers). Standard releases use +cu124/+rocm6.3/+cpu/+xpu; custom builds
    use +gitXXXX or other suffixes.
    """
    if "+" not in raw_version_str:
        return False
    local = raw_version_str.split("+", 1)[1]
    if not local:
        return False
    # fullmatch, so the whole local identifier matches: cu/rocm need a trailing digit (cu124,
    # rocm6.3), cpu/xpu are exact, and case-insensitive since some builds use uppercase.
    return not re.fullmatch(r"cu\d[\d.]*|rocm\d[\d.]*|cpu|xpu", local, re.IGNORECASE)


def _infer_required_torchvision(torch_major, torch_minor):
    """Infer the minimum required torchvision minor version from torch version.

    The torch -> torchvision minor version mapping follows a consistent formula:
      torch 1.x  ->  torchvision 0.(x + 1)   (verified: torch 1.7 through 1.13)
      torch 2.x  ->  torchvision 0.(x + 15)  (verified: torch 2.0 through 2.9)

    Returns (tv_major, tv_minor) or None if the major version is unrecognized.
    """
    if torch_major == 1 and torch_minor >= 7:
        return (0, torch_minor + 1)
    if torch_major == 2:
        return (0, torch_minor + 15)
    return None


# Unambiguous on their own: only a torchvision/torch mismatch produces these.
_TORCHVISION_ABI_MARKERS = (
    "torchvision::",
    "torchvision.io.video",
    "torchvision.io._video",
)
# A loader failure means torchvision only when it names torchvision or the torch libraries it
# links: the probe below imports torchvision where nothing used to, so a box broken for an
# unrelated reason must keep importing unsloth, not get "reinstall torchvision".
_LOADER_FAILURE_MARKERS = ("undefined symbol", "cannot open shared object file")
_TORCH_LIBRARY_MARKERS = ("torchvision", "libtorch", "libc10", "_C.so", "c10::")
# A lazily-imported torchvision with a dead extension surfaces as this, not a loader error.
# "partially initialized" is load-bearing: without it a typo on a healthy torchvision
# ("module 'torchvision' has no attribute 'nms'") would be answered with "reinstall". Nothing
# AFTER the module name is, because CPython words the rest four ways for the one fault:
#   3.9 - 3.12   partially initialized module 'torchvision' has no attribute 'extension'
#   3.13, 3.14   partially initialized module 'torchvision' from '<file>' has no attribute
#   from-import  cannot import name 'extension' from partially initialized module 'torchvision'
#   submodule    cannot access submodule 'ops' of module 'torchvision'
# The first three share "partially initialized module 'torchvision'"; the fourth names it as
# the parent instead, so it gets its own alternative.
_TORCHVISION_ATTRIBUTE_RE = re.compile(
    r"partially initialized module 'torchvision(?:\.[\w.]+)?'"
    r"|cannot access submodule '[\w.]+' of module 'torchvision(?:\.[\w.]+)?'"
)


def _shadowing_torchvision_path():
    """What is standing in for torchvision, if something is, else None.

    A local `torchvision` earlier on sys.path raises the same "partially initialized" words
    while the metadata still describes the installed distribution, so the binary check would
    answer it with "reinstall", which cannot change which sys.path entry is searched first.

    Identity, not shape: a `torchvision/` directory is a package exactly like the real one, so
    only asking whether the resolved file is the one the metadata installed separates them.

    find_spec, not sys.modules: the caller is in the `except` of the import that failed, and
    CPython has already removed the half-built module by then.
    """
    try:
        spec = importlib.util.find_spec("torchvision")
    except Exception:
        # A shadow can break find_spec itself. No answer is not evidence of shadowing.
        return None
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not origin:
        return None
    try:
        installed = os.fspath(
            importlib_distribution("torchvision").locate_file("torchvision/__init__.py")
        )
    except Exception:
        return None
    if not os.path.exists(installed):
        # Editable install, usually. Cannot tell, so do not send anyone hunting a file.
        return None
    try:
        if os.path.samefile(origin, installed):
            return None
    except OSError:
        if os.path.realpath(origin) == os.path.realpath(installed):
            return None
    return origin


def _is_broken_torchvision_error(error) -> bool:
    checked = set()
    current = error
    while current is not None and id(current) not in checked:
        checked.add(id(current))
        message = str(current)
        if any(marker in message for marker in _TORCHVISION_ABI_MARKERS):
            return True
        if _TORCHVISION_ATTRIBUTE_RE.search(message) and _shadowing_torchvision_path() is None:
            return True
        if any(m in message for m in _LOADER_FAILURE_MARKERS) and any(
            m in message for m in _TORCH_LIBRARY_MARKERS
        ):
            return True
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return False


# PyPI carries one torchvision build per release, for one CUDA family; every other build lives
# on download.pytorch.org under an index named after torch's local tag. With --no-deps an
# unqualified pin swaps a working wheel for PyPI's and raises the very
# "operator torchvision::nms does not exist" it was handed out to clear.
_TORCH_BACKEND_INDEX = re.compile(r"cpu|xpu|cu\d+|rocm\d+(?:\.\d+)*", re.IGNORECASE)
# The same families, matched as a PREFIX. A vendor build carries more than the family in its
# local tag: this repo's AMD extras install torch-2.10.0+rocm7.2.0.lw.gitb6ee5fde and
# triton-3.6.0+rocm7.2.0.gitba5c1517 straight from repo.radeon.com, and neither tag is an
# index name. Those are not PyPI builds, so advice that treats them as PyPI ones would swap a
# working Radeon wheel for the default CUDA build.
_TORCH_BACKEND_FAMILY = re.compile(r"(cpu|xpu|cu\d+|rocm[\d.]*)", re.IGNORECASE)
# conda keeps the backend in the build string (py3.12_cuda12.4_cudnn9_0) and leaves the version
# plain, so a conda torch looks like a PyPI one by version alone; its
# conda-meta/<name>-<version>-<build>.json tells them apart.
_CONDA_TORCH_PACKAGES = ("pytorch", "pytorch-cpu", "pytorch-gpu", "libtorch")


def _torch_local_tag(torch_version_raw):
    if not torch_version_raw or "+" not in torch_version_raw:
        return ""
    return torch_version_raw.split("+", 1)[1]


def _is_conda_managed(names, version_raw):
    """Did conda install one of `names` at this version, rather than pip?"""
    conda_meta = os.path.join(sys.prefix, "conda-meta")
    # A conda version never carries the +tag, but strip it so a pip install inside a conda prefix is
    # still matched on its release numbers.
    version = (version_raw or "").split("+", 1)[0]
    if not version or not os.path.isdir(conda_meta):
        return False
    # Pinned to this exact version, so an unrelated pytorch-lightning-*.json cannot answer for torch.
    prefixes = tuple(f"{name}-{version}-" for name in names)
    try:
        entries = os.listdir(conda_meta)
    except OSError:
        return False
    return any(e.endswith(".json") and e.startswith(prefixes) for e in entries)


def _torch_is_conda_managed(torch_version_raw):
    """Did conda install this torch, rather than pip?"""
    return _is_conda_managed(_CONDA_TORCH_PACKAGES, torch_version_raw)


def _has_no_matching_public_wheel(torch_version_raw):
    try:
        # .is_prerelease, not a substring list: 2.11.0a1 and 2.11.0b2 are prereleases no a0/b0 match would catch.
        if TrueVersion(torch_version_raw).is_prerelease:
            return True
    except Exception:
        return True
    local = _torch_local_tag(torch_version_raw)
    if not local:
        # An absent tag means PyPI for a pip install, but conda never writes one either, and its torch
        # may be CPU, ROCm or a CUDA family PyPI does not ship.
        return _torch_is_conda_managed(torch_version_raw)
    return not _TORCH_BACKEND_INDEX.fullmatch(local)


def _torch_wheel_index(torch_version_raw):
    """The ` --index-url ...` fragment that keeps an install on torch's own accelerator family.

    pip's `--index-url` defaults to https://pypi.org/simple, which carries exactly one build
    per release: the default CUDA one. A ROCm or XPU or CPU torch records its family in the
    local version tag (2.9.0+rocm6.4, 2.9.0+xpu, 2.9.0+cpu) and every one of those builds is
    published only under https://download.pytorch.org/whl/<tag>, so an unqualified command
    handed to such a user swaps their working torch for an incompatible default build. An
    absent or non-family tag means PyPI, where the unqualified command is already right.
    """
    local = _torch_local_tag(torch_version_raw)
    if local and _TORCH_BACKEND_INDEX.fullmatch(local):
        return f" --index-url https://download.pytorch.org/whl/{local.lower()}"
    return ""


def _torch_companion_specs():
    """The packages that must move WITH torch, as a `"pkg" ` prefix for the upgrade command.

    `pip install --upgrade` upgrades the packages it is given and nothing else, so naming
    only torch moves torch and leaves the companion that was built against the old one. Every
    torchvision wheel requires an exact `torch==X.Y.Z`, which is why `_torchvision_repair_command`
    above pins the pair; a torch upgrade on its own is how the mismatch it repairs gets created
    in the first place ("operator torchvision::nms does not exist"). Only packages that are
    actually installed are named, so an install without torchvision gets the plain command.
    """
    return "".join(
        f' "{name}"'
        for name in ("torchvision", "torchaudio")
        if _installed_version(name) != "unknown"
    )


def _torch_accelerator_note(torch_version_raw):
    """The sentence that keeps an UPGRADE on the right accelerator, without naming an index.

    A repair reinstalls the release that is already here, so the installed torch's own index
    is guaranteed to carry it and `_torch_wheel_index` can hand over a ready command. An
    upgrade moves to a release that index may never have had: each
    download.pytorch.org/whl/<backend> holds only what was built for that backend, and the
    CUDA families retire (cu124's newest torch is 2.6.0, while torch 2.7 shipped on cu126 and
    cu128), so pinning the installed tag would hand the reported torch 2.6.0+cu124 user a
    command with no candidate at all. Naming the family and the selector is the part that is
    true for every release.
    """
    local = _torch_local_tag(torch_version_raw)
    if not local:
        # conda never writes a local tag, so an untagged version alone reads like a PyPI
        # install. `pip install --upgrade torch` into a conda environment overlays the
        # conda-managed files with a PyPI wheel, and the backend it lands on is whatever
        # PyPI ships by default rather than the one the channel built.
        if _torch_is_conda_managed(torch_version_raw):
            return (
                "    conda installed this torch, so upgrade it the same way "
                "(`conda update pytorch` on the channel it came from). `pip install "
                "--upgrade` would overlay the conda files with a PyPI wheel and can "
                "change the backend with them."
            )
        return ""
    if not _TORCH_BACKEND_INDEX.fullmatch(local):
        vendor = _TORCH_BACKEND_FAMILY.match(local)
        if vendor is None:
            return ""
        # A tag that names a family but is not an index name is a vendor build, which is
        # what this repo's own AMD extras install. PyPI has no such version, and neither
        # does download.pytorch.org, so the only advice that keeps the accelerator is to
        # go back to the source the wheel came from.
        return (
            f"    That takes the default build from PyPI. This torch is a {local} build, "
            f"which no public index carries, so upgrade it from wherever it came from "
            f"(the ROCm builds unsloth's AMD extras install come from repo.radeon.com). "
            f"An unqualified pip upgrade replaces it with the default build."
        )
    family = local.rstrip("0123456789.").lower()
    family = {"cu": "CUDA", "rocm": "ROCm", "xpu": "XPU", "cpu": "CPU"}.get(family, local)
    return (
        f"    That takes the default build from PyPI. This torch is a {local} build, and each "
        f"https://download.pytorch.org/whl/ index carries only the releases built for it, so "
        f"pick the {family} index for the release you are moving to at "
        f"https://pytorch.org/get-started/locally/ and pass it as --index-url."
    )


def _torchvision_repair_advice(required = None, torch_version_raw = None):
    """The one sentence telling the user how to repair a broken torchvision."""
    if _has_no_matching_public_wheel(torch_version_raw):
        return (
            f"Reinstall the torchvision built for torch=={torch_version_raw}, "
            f"from wherever that torch came from."
        )
    return f"Reinstall it with `{_torchvision_repair_command(required, torch_version_raw)}`."


def _torchvision_repair_command(required = None, torch_version_raw = None):
    """The pip command to repair a broken torchvision binary in place.

    Pinned and `--no-deps`, both deliberately. Every torchvision wheel requires
    an exact `torch==X.Y.Z`, so an unpinned `pip install --upgrade
    --force-reinstall torchvision` resolves the newest torchvision and then
    replaces the user's torch to satisfy it -- on a Colab/Kaggle image that is
    the pinned torch every other wheel was built against, vLLM included. The
    pin also fixes the case where the version gate passed on its lower bound
    (torch 2.4 accepts torchvision >= 0.19, so an installed 0.20 reaches here):
    the companion release is what gets reinstalled, not the newest one.
    """
    if required is None:
        spec = "torchvision"
    elif len(required) >= 3:
        # Exact, because the pair is: torchvision 0.22.0 needs torch 2.7.0 and 0.22.1 needs 2.7.1, so
        # a 0.22.* wildcard resolves 0.22.1 while --no-deps keeps the mismatched torch.
        spec = f"torchvision=={required[0]}.{required[1]}.{required[2]}"
    else:
        spec = f"torchvision=={required[0]}.{required[1]}.*"
    index = _torch_wheel_index(torch_version_raw)
    return f'pip install --force-reinstall --no-deps --no-cache-dir{index} "{spec}"'


def _probe_torchvision_binary(
    torch_version_raw,
    torchvision_version_raw,
    required = None,
):
    """Import torchvision, so a broken binary is named here and not six frames
    deep in transformers.

    The table above compares metadata, which cannot see an ABI break: ops
    built against a different torch die in `_meta_registrations`, at
    `register_fake("torchvision::nms")`. Found by running Gemma4_(E2B)_GRPO,
    whose T4 branch installs vllm==0.9.2 beside Colab's torch and mismatches
    both; `disable_broken_vllm` already covers the vLLM half.

    Costs no extra import: transformers imports torchvision from `image_utils`
    the moment anything touches `processing_utils`.
    """
    try:
        import torchvision  # noqa: F401
        import torchvision.ops  # noqa: F401  where the compiled nms lives
    except Exception as error:
        shadow = _shadowing_torchvision_path()
        if shadow is not None and _TORCHVISION_ATTRIBUTE_RE.search(str(error)):
            # Named: the metadata says torchvision is installed and it is, so a reinstall
            # changes nothing. The file is the whole fix.
            raise ImportError(
                f"Unsloth: {shadow} is being imported as `torchvision`, ahead of the "
                f"installed package ({type(error).__name__}: {error}). Rename or move that "
                f"file; reinstalling torchvision will not change which one wins."
            ) from error
        # Anything else is left for whoever actually needs torchvision.
        if not _is_broken_torchvision_error(error):
            return
        raise ImportError(
            f"Unsloth: torchvision=={torchvision_version_raw} claims to match "
            f"torch=={torch_version_raw}, but its compiled operators do not "
            f"load ({type(error).__name__}: {error}). "
            f"{_torchvision_repair_advice(required, torch_version_raw)} "
            f"Set UNSLOTH_SKIP_TORCHVISION_CHECK=1 to skip this check."
        ) from error


def torchvision_compatibility_check():
    # Allow skipping via environment variable for custom environments
    if os.environ.get("UNSLOTH_SKIP_TORCHVISION_CHECK", "0").lower() in ("1", "true"):
        return

    if importlib.util.find_spec("torch") is None:
        raise ImportError("Unsloth: torch not found. Please install torch first.")
    if importlib.util.find_spec("torchvision") is None:
        return

    try:
        torch_version_raw = importlib_version("torch")
        torchvision_version_raw = importlib_version("torchvision")
    except Exception:
        return

    try:
        torch_v = Version(torch_version_raw)
        tv_v = Version(torchvision_version_raw)
    except Exception:
        return

    # Ground truth, takes precedence over the formula; see pytorch.org/get-started/previous-versions/
    TORCH_TORCHVISION_COMPAT = {
        (2, 9): (0, 24),
        (2, 8): (0, 23),
        (2, 7): (0, 22),
        (2, 6): (0, 21),
        (2, 5): (0, 20),
        (2, 4): (0, 19),
    }

    torch_release = torch_v.release
    if len(torch_release) < 2:
        return
    torch_major, torch_minor = torch_release[0], torch_release[1]

    # Known table first, then the formula for forward compatibility
    required = TORCH_TORCHVISION_COMPAT.get((torch_major, torch_minor))

    if required is None:
        required = _infer_required_torchvision(torch_major, torch_minor)

    if required is None:
        return

    # Carry torch's own patch into the companion: they move together (2.7.0/0.22.0, 2.7.1/0.22.1),
    # so the repair names one wheel instead of a minor-wide range --no-deps cannot satisfy.
    if len(torch_release) >= 3:
        required = (required[0], required[1], torch_release[2])

    required_tv_str = f"{required[0]}.{required[1]}.0"

    if tv_v >= Version(required_tv_str):
        logger.info(
            f"Unsloth: torch=={torch_version_raw} and "
            f"torchvision=={torchvision_version_raw} are compatible."
        )
        _probe_torchvision_binary(torch_version_raw, torchvision_version_raw, required)
        return

    message = (
        f"Unsloth: torch=={torch_version_raw} requires "
        f"torchvision>={required_tv_str}, "
        f"but found torchvision=={torchvision_version_raw}. "
        f'Try updating torchvision via `pip install --upgrade "torchvision>={required_tv_str}"`. '
        f"Please refer to https://pytorch.org/get-started/previous-versions/ "
        f"for more information."
    )

    is_custom = _is_custom_torch_build(torch_version_raw) or _is_custom_torch_build(
        torchvision_version_raw
    )

    # Nightly/dev/alpha/beta/rc builds mismatch expectedly, so those and source builds only warn;
    # a stable mismatch fails fast to prevent runtime operator errors.
    _pre_tags = (".dev", "a0", "b0", "rc", "alpha", "beta", "nightly")
    is_prerelease = any(t in torch_version_raw for t in _pre_tags) or any(
        t in torchvision_version_raw for t in _pre_tags
    )

    # Only downgrade to warning for custom/source or prerelease builds. Stable mismatches should fail fast to
    # prevent runtime operator errors.
    if is_custom or is_prerelease:
        reason = "custom/source build" if is_custom else "pre-release build"
        logger.warning(
            f"{message}\n"
            f"Detected a {reason}. "
            f"Continuing with a warning. "
            f"Set UNSLOTH_SKIP_TORCHVISION_CHECK=1 to silence this."
        )
        return

    raise ImportError(message)


def _unsatisfied_transformers_requirements():
    """Base (no-extras) requirements the environment does not satisfy, as
    [(name, specifier, installed_version), ...]; installed_version is None when the
    package is absent. Read from the installed distribution's own metadata, so it is
    whatever that transformers asks for - git main, 4.57.x or 5.x - rather than a
    table that would rot. Never raises; returns [] on anything unexpected.
    """
    try:
        from importlib.metadata import requires as _dist_requires
        from packaging.requirements import Requirement
    except Exception:
        return []

    try:
        raw_requirements = _dist_requires("transformers")
    except Exception:
        # transformers not installed, or its dist-info is missing / unreadable.
        return []
    if not raw_requirements:
        return []

    unsatisfied = []
    for raw_requirement in raw_requirements:
        try:
            requirement = Requirement(raw_requirement)
        except Exception:
            continue  # Unparseable requirement line - ignore it, never guess.

        # extra = "" drops optional-extra requirements and inapplicable python_version / sys_platform
        # gates: packages the user is right not to have.
        if requirement.marker is not None:
            try:
                if not requirement.marker.evaluate({"extra": ""}):
                    continue
            except Exception:
                continue  # Undecidable marker - assume it does not apply.

        try:
            installed = importlib_version(requirement.name)
        except PackageNotFoundError:
            # Absent, which --no-deps causes as readily as a stale version: transformers checks its base
            # requirements at root import and raises PackageNotFoundError with the same misleading hint.
            unsatisfied.append((requirement.name, str(requirement.specifier), None))
            continue
        except Exception:
            continue  # Metadata unreadable - we cannot judge it, so stay quiet.

        if not requirement.specifier:
            continue  # Installed, and no floor it could fall below.

        try:
            # Parse explicitly: SpecifierSet.contains() reports a non-PEP440 version as "not contained"
            # rather than raising, which would be a false positive.
            installed_version = TrueVersion(installed)
        except Exception:
            continue  # Not a PEP 440 version - we cannot judge it, so stay quiet.

        try:
            # prereleases = True so a legitimate 1.0.0rc1 does not read as a violation.
            if requirement.specifier.contains(installed_version, prereleases = True):
                continue
        except Exception:
            continue  # Bad specifier - stay quiet.

        unsatisfied.append((requirement.name, str(requirement.specifier), installed))

    return unsatisfied


def check_transformers_dependency_versions():
    """Warn when transformers' own declared dependency floors are unmet.

    A notebook needing a bleeding-edge model installs transformers from git with
    `--no-deps` on purpose, so pip cannot re-resolve torch - and so pip never
    enforces what that transformers requires either. The install "succeeds" and the
    failure lands at import, advising `pip install transformers -U`. That remedy is
    wrong here: the user is deliberately on main, so upgrading undoes the install
    they wanted, or loops. The dependency is what needs upgrading. This runs first
    and says so, naming it. Warns rather than raises; see the _gpu_init.py call.
    """
    if os.environ.get("UNSLOTH_SKIP_TRANSFORMERS_DEPENDENCY_CHECK", "0").lower() in (
        "1",
        "true",
    ):
        return
    try:
        # find_spec RAISES ValueError rather than returning None for a transformers in sys.modules with
        # __spec__ None or unset (a stub, or one mid-teardown). Not worth failing the import over.
        if importlib.util.find_spec("transformers") is None:
            return
    except Exception:
        return

    try:
        unsatisfied = _unsatisfied_transformers_requirements()
    except Exception:
        return
    if not unsatisfied:
        return

    try:
        transformers_version = importlib_version("transformers")
    except Exception:
        transformers_version = "unknown"

    lines = [
        f"Unsloth: transformers=={transformers_version} declares dependencies that "
        f"your environment does not satisfy:"
    ]
    upgrades = []
    for name, specifier, installed in unsatisfied:
        found = f"found {name}=={installed}" if installed is not None else "it is not installed"
        lines.append(f"    {name}{specifier} is required, but {found}")
        upgrades.append(f'"{name}{specifier}"')
    lines.append("")
    verb = "Upgrade" if all(i is not None for _, _, i in unsatisfied) else "Install or upgrade"
    lines.append(f"{verb} the dependencies, not transformers:")
    lines.append(f"    pip install --upgrade {' '.join(upgrades)}")
    lines.append("")
    lines.append(
        "transformers may suggest `pip install transformers -U` instead. Ignore that "
        "if you installed transformers from git main on purpose (for example with "
        "`pip install --no-deps git+https://github.com/huggingface/transformers.git` "
        "for a new model) - upgrading transformers would only undo the install you "
        "wanted. Set UNSLOTH_SKIP_TRANSFORMERS_DEPENDENCY_CHECK=1 to silence this."
    )

    logger.warning("\n".join(lines))


# torch releases that first provide a given attribute, used only to name the
# upgrade in the message below. Measured against the wheels rather than guessed:
# torch 2.6.0 has neither MX dtype and 2.8.0 has both, and both landed in 2.7.0
# (pytorch/pytorch#146427 and #146578). An attribute that is not in here still
# gets the diagnosis, just without a release to name, which is the point: the
# table is a convenience and never the mechanism.
_TORCH_ATTRIBUTE_FLOORS = {
    "float8_e8m0fnu": "2.7.0",
    "float4_e2m1fn_x2": "2.7.0",
}

# Packages unsloth imports and cannot add a missing torch attribute to. The same
# AttributeError raised from user code is a typo, so it is left exactly as torch
# wrote it.
_TORCH_ATTRIBUTE_DEPENDENTS = frozenset(
    (
        "accelerate",
        "bitsandbytes",
        "cut_cross_entropy",
        "diffusers",
        "peft",
        "torchao",
        "torchaudio",
        "torchvision",
        "transformers",
        "trl",
        "unsloth_zoo",
        "vllm",
        "xformers",
    )
)


class UnslothTorchTooOldError(AttributeError):
    """The installed torch is older than a library unsloth imports needs.

    An AttributeError subclass on purpose: it replaces one torch itself raised,
    so `hasattr(torch, name)` and `getattr(torch, name, default)` keep behaving
    exactly as they did. Only an unguarded access changes, and only by carrying
    the diagnosis instead of four words.
    """


@functools.lru_cache(maxsize = None)
def _installed_version(package):
    """The installed version of `package`, or "unknown", asked at most once.

    Cached because this sits on a path the interpreter takes for EVERY attribute
    access on torch that normal lookup missed, and a missing attribute is not
    always fatal: `hasattr(torch, name)` and `getattr(torch, name, default)` are
    how these same libraries feature-probe, and both come through here. Measured
    on this workspace, `importlib.metadata.version` costs about 850 microseconds
    per call because it walks sys.path, so an uncached lookup turned a failed
    probe from roughly 0.8 microseconds into roughly 850, a thousandfold, for a
    string that cannot change inside one process. With the cache the wrapper is
    back to the cost of building the message.
    """
    try:
        return importlib_version(package)
    except Exception:
        return "unknown"


def _torch_too_old_message(attribute, package, exception):
    # torch.__version__ rather than the metadata version, because it carries the
    # build (2.6.0+cu124) and that is what the user reads in every other message.
    torch_version = getattr(sys.modules.get("torch"), "__version__", None)
    if not torch_version:
        torch_version = _installed_version("torch")
    package_version = _installed_version(package)

    floor = _TORCH_ATTRIBUTE_FLOORS.get(attribute)
    requirement = f'"torch>={floor}"' if floor else '"torch"'

    lines = [
        str(exception),
        "",
        f"Unsloth: {package}=={package_version} uses torch.{attribute}, and the "
        f"installed torch=={torch_version} does not have it, so this environment pairs "
        f"a {package} with a torch that cannot serve it.",
    ]
    if floor is not None:
        # A known floor is evidence of the direction: the attribute was ADDED in that
        # release, so the torch here is the older half and upgrading it is the remedy.
        lines += [
            f"torch.{attribute} first appears in torch {floor}, so this torch is the "
            f"older half.",
            "",
            f"Upgrade torch, which leaves {package} where it is:",
            f"    pip install --upgrade {requirement}{_torch_companion_specs()}",
        ]
    else:
        # No floor is no evidence of direction. The attribute may be newer than this
        # torch, and it may equally have been REMOVED by it, which an older dependency
        # reaching for a retired API hits; prescribing an upgrade there is the opposite
        # remedy. So state both directions and prescribe neither.
        lines += [
            "",
            f"torch.{attribute} is not in this release's table of added attributes, so "
            f"it is either newer than this torch or removed by it. Check which release "
            f"of torch has torch.{attribute} at "
            f"https://pytorch.org/docs/stable/torch.html and move whichever half is out "
            f"of step: upgrade torch when the attribute is newer,",
            f"    pip install --upgrade {requirement}{_torch_companion_specs()}",
            f"or install a {package} that matches this torch when the attribute is gone.",
        ]
    accelerator = _torch_accelerator_note(torch_version)
    if accelerator:
        lines.append(accelerator)
    lines += [
        "",
        f"Installing a {package} that matches this torch works too. Pip allowed the "
        f"pair because {package} declares a lower torch floor than its own modules "
        f"actually touch, so there was nothing for the resolver to reject.",
    ]
    return "\n".join(lines)


def patch_torch_missing_attribute_error():
    """Name the torch upgrade when a dependency reaches for a dtype torch lacks.

    The reported failure was a Studio training run whose transformers (5.10) was
    newer than its torch: `import unsloth` died inside unsloth_zoo's temporary
    patches, importing transformers.processing_utils, on a module-scope
    `_UE8M0_SF_DTYPE = torch.float8_e8m0fnu` in
    transformers/integrations/finegrained_fp8.py, as a bare

        AttributeError: module 'torch' has no attribute 'float8_e8m0fnu'

    with no mention of a version and nothing for the user to act on. Both
    pyprojects still permit the pair, and pip cannot be made to reject it:
    transformers declares `torch>=2.2` through `torch>=2.5` depending on the
    release, well below what its own modules touch, so there is no floor for the
    resolver to enforce. A static transformers-to-torch table here would be no
    better, because it would have to be wrong somewhere: torch 2.6.0 with
    transformers 4.57.6 is a supported combination that works.

    So probe the behaviour, at the only place that knows the answer. torch
    answers a missing attribute through a module-level __getattr__, which the
    interpreter calls only after normal lookup has already failed, so wrapping
    that one function costs nothing on every successful attribute access and
    turns the one failing access into a diagnosis. Idempotent, and the original
    stays reachable as `__unsloth_original__`.
    """
    try:
        import torch
    except Exception:
        # A missing or broken torch has its own, clearer error further down
        # _gpu_init.py; there is nothing to wrap here.
        return False

    original = torch.__dict__.get("__getattr__")
    if original is None:
        # No module-level __getattr__ to wrap, so torch raises its AttributeError
        # from the interpreter itself and there is nothing to improve.
        return False
    if getattr(original, "__unsloth_patched__", False):
        return True

    @functools.wraps(original)
    def __getattr__(name):
        try:
            return original(name)
        except AttributeError as exception:
            try:
                # The frame that made the access, and only that one. Walking
                # further up would blame transformers for a typo in a callback
                # transformers happened to call.
                requester = sys._getframe(1).f_globals.get("__name__") or ""
            except Exception:
                # sys._getframe is CPython-only and raises an audit event, so a
                # hardened interpreter can refuse it. A bare `raise` here would
                # rethrow THAT exception instead of torch's AttributeError, and
                # `hasattr(torch, name)` only swallows AttributeError, so an
                # ordinary feature probe would start propagating a RuntimeError.
                # Unattributable is not fatal: leave it to torch's own error.
                requester = ""
            package = requester.split(".", 1)[0]
            if package not in _TORCH_ATTRIBUTE_DEPENDENTS:
                raise
            raise UnslothTorchTooOldError(
                _torch_too_old_message(name, package, exception)
            ) from exception

    __getattr__.__unsloth_patched__ = True
    __getattr__.__unsloth_original__ = original
    torch.__getattr__ = __getattr__
    return True


# The '#' argument formats in triton's CUDA driver shim need this macro, and
# CPython stopped requiring it in 3.13.
_PY_SSIZE_T_CLEAN = "PY_SSIZE_T_CLEAN"

# `#define` / `#undef PY_SSIZE_T_CLEAN` as their own directives, anchored to the start of
# a line so a mention inside a comment or a string literal is neither.
_PY_SSIZE_T_CLEAN_DIRECTIVE = re.compile(
    r"^[ \t]*#[ \t]*(define|undef)[ \t]+" + _PY_SSIZE_T_CLEAN + r"\b",
    re.M,
)
_PYTHON_H_INCLUDE = re.compile(r"^[ \t]*#[ \t]*include[ \t]*[<\"][^>\"]*Python\.h[>\"]", re.M)


_C_COMMENT_OR_STRING = re.compile(
    r"/\*.*?\*/|//[^\n]*|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'",
    re.S,
)


def _blank_c_spans(source: str, keep_strings: bool = False) -> str:
    """`source` with comment spans blanked, offsets and lines intact, and string literals
    blanked too unless `keep_strings`.

    The compiler removes comments before the preprocessor ever sees a directive, so a
    `#define PY_SSIZE_T_CLEAN` that only appears inside a `/* ... */` is not a definition
    at all, and treating it as one silences the warning on a shim that still dies at the
    first kernel launch. Blanked rather than deleted so every span keeps its length: the
    scan below compares directive positions against the `#include <Python.h>` position,
    and deleting text would move one relative to the other. Newlines are kept for the same
    reason, since both patterns are line-anchored.

    `keep_strings` is for the caller that has to read inside a literal: the PyArg_Parse
    scan decides on the format string itself, so blanking literals would make every shim
    look clean. String spans are still matched either way, since that is what stops a
    `//` or a `/*` inside a literal from blanking the rest of the file.
    """

    def blank(match):
        text = match.group(0)
        if keep_strings and not text.startswith(("/*", "//")):
            return text
        return "".join("\n" if character == "\n" else " " for character in text)

    return _C_COMMENT_OR_STRING.sub(blank, source)


_PYARG_CALL = re.compile(r"PyArg_Parse\w*\(")
_HASH_FORMAT = re.compile(r"\"[^\"]*#")


def _parses_a_hash_format(source: str) -> bool:
    """Does a call the compiler still sees parse a '#' format?

    Two masked copies of the same offsets, because the two halves of the question need
    opposite things. The CALL has to be real code, so it is located in the copy with both
    comments and string literals blanked: a documentation string quoting an example
    `PyArg_ParseTuple(args, "ss#ii", ...)` is not a call, and a commented-out one is not
    either. The FORMAT then has to be readable, so the argument list is read from the copy
    that blanks comments only. Blanking preserves length, so one offset serves both.
    """
    code_only = _blank_c_spans(source)
    with_literals = _blank_c_spans(source, keep_strings = True)
    for call in _PYARG_CALL.finditer(code_only):
        end = with_literals.find(";", call.end())
        if end == -1:
            end = len(with_literals)
        if _HASH_FORMAT.search(with_literals, call.end(), end):
            return True
    return False


def _defines_py_ssize_t_clean_before_python_h(source: str) -> bool:
    """Is the macro in effect where CPython needs it: defined, and still defined, at the
    point Python.h is included.

    The requirement is positional, not textual. CPython documents
    `#define PY_SSIZE_T_CLEAN` immediately before `#include <Python.h>`; a definition
    that lands after the include, one that is undefined again before it, and a bare
    mention in a comment or a string literal all leave the '#' formats unsafe exactly as
    if the macro were absent. The `token in source` test this replaces called all four of
    those fixed, so the warning stayed silent on a shim that still dies at the first
    kernel launch.

    The LAST directive before the include is the one that decides, which is what makes
    the `#define` / `#undef` pair read correctly rather than just the presence of either.
    """
    source = _blank_c_spans(source)
    include = _PYTHON_H_INCLUDE.search(source)
    if include is None:
        return False
    state = False
    for directive in _PY_SSIZE_T_CLEAN_DIRECTIVE.finditer(source):
        if directive.start() >= include.start():
            break
        state = directive.group(1) == "define"
    return state


# Every distribution this repository knows of that provides the `triton` import name. Only a
# fallback for interpreters without `packages_distributions`; the mapping is asked for first
# wherever it exists, so a provider added upstream is still found on 3.10 and later.
# Providers published only on download.pytorch.org, so a reinstall of one needs torch's own
# accelerator index rather than PyPI. `pytorch-triton*` is matched by prefix; these are the
# renamed spellings that prefix no longer covers.
_TORCH_INDEX_TRITON_PROVIDERS = ("triton-xpu", "triton-rocm")

_TRITON_PROVIDERS = (
    "triton",
    "triton-windows",
    "pytorch-triton",
    "pytorch-triton-rocm",
    "pytorch-triton-xpu",
    # The renamed spellings. Both accelerator providers were renamed on
    # download.pytorch.org: `pytorch-triton-rocm` stops at 3.5.1 and `triton-rocm` carries
    # 3.7 onwards, the same split `pytorch-triton-xpu` and `triton-xpu` have. Missing here,
    # a current ROCm or XPU shim reads as `triton==unknown` on the probing fallback, and
    # the repair command then installs generic CUDA `triton` over the installed provider.
    "triton-xpu",
    "triton-rocm",
)


def _canonical_distribution(name):
    """PEP 503 normalised, so `pytorch_triton_xpu` and `pytorch-triton-xpu` are one name."""
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def _torch_required_triton_distributions():
    """The Triton distributions the INSTALLED torch declares it needs, normalised.

    torch pins its Triton by name as well as version (`triton~=3.8.0`,
    `pytorch-triton-xpu==...`), so when a rename leaves the old dist-info in place beside
    the new one this is the only record that says which of the two belongs to this torch.
    Empty when torch declares none, or when its metadata cannot be read at all.
    """
    try:
        from importlib.metadata import requires as _requires
        declared = _requires("torch") or ()
    except Exception:
        return frozenset()
    found = set()
    for requirement in declared:
        name = re.split(r"[\s;\[<>=!~(]", str(requirement).strip(), maxsplit = 1)[0]
        if "triton" in name.lower():
            found.add(_canonical_distribution(name))
    return frozenset(found)


def _distribution_owning_path(names, path):
    """The distribution among `names` whose installed file list contains `path`, or None.

    Providers can coexist: `studio/install_python_stack.py::_ensure_xpu_triton` documents
    generic `triton` beside `pytorch-triton-xpu` on the same paths, and
    `packages_distributions()` then reports both without saying which one last wrote the
    file. Taking the first entry can name the CUDA provider for an XPU install and hand the
    user a command that replaces their working Triton, which is the substitution this whole
    helper exists to avoid. The offending `driver.c` is a concrete path, so ask which
    distribution actually ships it.
    """
    if not path:
        return None
    try:
        from importlib.metadata import distribution as _distribution
        target = Path(path).resolve()
    except Exception:
        return None
    claimants = []
    for name in names:
        try:
            files = _distribution(name).files or ()
        except Exception:
            continue
        for entry in files:
            try:
                if Path(entry.locate()).resolve() == target:
                    claimants.append(name)
                    break
            except Exception:
                continue
    if len(claimants) == 1:
        return claimants[0]
    if not claimants:
        return None
    # Coexisting providers can both RECORD the same file, and then the first entry is
    # nothing but ordering. Three tiebreaks, most authoritative first.
    #
    # 1. The distribution torch itself requires. A renamed provider leaves the old
    #    dist-info behind (`pytorch-triton-xpu` beside `triton-xpu` after an upgrade), so
    #    both claim the file and both name the backend; only torch says which one belongs
    #    to the torch that is installed.
    required = _torch_required_triton_distributions()
    named = [name for name in claimants if _canonical_distribution(name) in required]
    if len(named) == 1:
        return named[0]
    # 2. Failing that, torch's backend family, because the harmful answer is always the one
    #    that installs a CUDA build over an accelerator one.
    local = _torch_local_tag(getattr(sys.modules.get("torch"), "__version__", None)).lower()
    for family in ("xpu", "rocm"):
        if family in local:
            matches = [name for name in claimants if family in name.lower()]
            return matches[0] if len(matches) == 1 else None
    # 3. And with no accelerator to preserve, the generic provider.
    generic = [name for name in claimants if name.lower() == "triton"]
    return generic[0] if len(generic) == 1 else None


def _triton_distribution(offending_path = None):
    """Which distribution owns the imported ``triton`` package, and its version.

    The `triton` PyPI distribution is only one of the providers this repository supports.
    Windows installs `triton-windows`, ROCm installs `pytorch-triton-rocm`, Intel XPU
    installs `pytorch-triton-xpu`, and all of them land on the same `triton` import name.
    Asking `importlib.metadata.version("triton")` on any of those raises
    PackageNotFoundError, which is how the version used to be reported as "unknown", and
    telling that user to `pip install --force-reinstall triton` either finds no wheel for
    their platform or overwrites their platform build with a CUDA one.

    `packages_distributions()` maps top-level import name to the distributions providing
    it, which is the only mapping that answers this without guessing from the platform.
    Falls back to the import name so the message still reads sensibly if the metadata is
    missing entirely (an unpacked or vendored tree).
    """
    names = []
    try:
        from importlib.metadata import packages_distributions
        names = list(packages_distributions().get("triton") or [])
    except Exception:
        names = []
    if not names:
        # `packages_distributions` is Python 3.10+, and pyproject still admits 3.9, where
        # the import above raises and leaves this empty -- reporting `triton==unknown` and
        # recommending the CUDA `triton` over the platform build, which is the exact
        # failure this helper exists to avoid. The backport answers the same question when
        # it is installed; when it is not, ask each provider directly, which needs no
        # mapping at all: a provider that is not installed has no version to report.
        try:
            from importlib_metadata import packages_distributions as _backport
            names = list(_backport().get("triton") or [])
        except Exception:
            names = []
    if not names:
        names = [name for name in _TRITON_PROVIDERS if _installed_version(name) != "unknown"]
    # The distribution that actually ships the offending file wins over ordering, when the
    # file is known and the metadata answers.
    owner = _distribution_owning_path(names, offending_path)
    if owner is not None:
        names = [owner] + [name for name in names if name != owner]
    # Prefer a provider that can actually report a version; a stale empty dist-info
    # otherwise wins over the real one purely on ordering.
    for name in names:
        try:
            return name, importlib_version(name)
        except Exception:
            continue
    try:
        return "triton", importlib_version("triton")
    except Exception:
        return (names[0] if names else "triton"), "unknown"


def _triton_reinstall_command(distribution, triton_version):
    """The pip command that repairs a broken Triton shim without changing which Triton is installed.

    Pinned to the installed version, because torch pins Triton exactly (torch 2.6.0 requires
    `triton==3.2.0`) and pip's `--force-reinstall` is documented only as "Reinstall all packages
    even if they are already up-to-date" -- it still resolves the newest release, so the bare
    command trades a broken-but-matching Triton for a working one torch cannot load. The
    pytorch-triton-* providers are published only on download.pytorch.org, so those also need
    torch's own accelerator index. An "unknown" version (vendored or stripped metadata) has no
    pin to give, and an invented one would resolve to nothing at all, so that case stays bare.
    """
    index = ""
    # `triton-xpu` as well as `pytorch-triton-*`: torch 2.10 renamed the XPU provider and
    # pyproject pins the new name straight from download.pytorch.org (studio/setup.ps1
    # records the rename too). Matching only the old spelling asked PyPI for `triton-xpu`,
    # where that pinned wheel is not published at all.
    if distribution.startswith("pytorch-triton") or distribution in _TORCH_INDEX_TRITON_PROVIDERS:
        torch_version_raw = getattr(sys.modules.get("torch"), "__version__", None)
        index = _torch_wheel_index(torch_version_raw)
    if not triton_version or triton_version == "unknown":
        spec = distribution
    else:
        spec = f'"{distribution}=={triton_version}"'
    return f"pip install --force-reinstall --no-cache-dir{index} {spec}"


def _triton_repair_advice(distribution, triton_version):
    """How to repair this Triton, as one line, or two when no index can serve it.

    A pip command only helps when some index publishes the pinned version. The AMD extras
    install `triton==3.6.0+rocm7.2.0.gitba5c1517` from a repo.radeon.com URL under the
    plain `triton` name, and neither PyPI nor download.pytorch.org has that version at
    all, so a pinned command there resolves to nothing. Name the source instead, the same
    way the torchvision advice does for a torch with no public wheel.
    """
    command = _triton_reinstall_command(distribution, triton_version)
    if _is_conda_managed((distribution,), triton_version):
        # conda-forge publishes `triton`, and a pip --force-reinstall over a conda-managed
        # one overlays conda's own files with a wheel, leaving two package managers owning
        # the same paths. Same reasoning, and same remedy shape, as the conda branch of the
        # torch upgrade advice.
        return (
            f"conda installed this Triton, so repair it the same way "
            f"(`conda install --force-reinstall {distribution}={triton_version}` on the "
            f"channel it came from). A pip reinstall would overlay the conda files with a "
            f"wheel and can change the backend with them."
        )
    local = (triton_version or "").split("+", 1)[1] if "+" in (triton_version or "") else ""
    if local and "--index-url" not in command:
        return (
            f"Reinstall the Triton your torch pins from wherever this one came from: the "
            f"+{local} tag marks a vendor build that no public index publishes (unsloth's "
            f"AMD extras take theirs from repo.radeon.com), so a pinned pip command "
            f"resolves to nothing:\n    {command}"
        )
    return f"Reinstall the Triton your torch pins:\n    {command}"


def _triton_driver_shims_missing_py_ssize_t_clean():
    """Installed triton backend driver shims that cannot parse their own
    arguments on this interpreter, as [(backend, path), ...].

    Static, because the failure is a property of the C source triton compiles at
    run time: `loadBinary` parses with the format "ss#ii" into a Py_ssize_t, and
    CPython below 3.13 raises SystemError for a '#' format unless the extension
    was compiled with PY_SSIZE_T_CLEAN defined before Python.h.
    """
    if sys.version_info >= (3, 13):
        return []
    try:
        spec = importlib.util.find_spec("triton")
    except Exception:
        return []
    if spec is None or not spec.submodule_search_locations:
        return []

    offenders = []
    for location in spec.submodule_search_locations:
        backends = Path(location) / "backends"
        if not backends.is_dir():
            continue
        for driver in sorted(backends.glob("*/driver.c")):
            try:
                source = driver.read_text(encoding = "utf-8", errors = "replace")
            except Exception:
                continue
            if "Python.h" not in source:
                continue
            if _defines_py_ssize_t_clean_before_python_h(source):
                continue
            if not _parses_a_hash_format(source):
                continue
            offenders.append((driver.parent.name, str(driver)))
    return offenders


def check_triton_py_ssize_t_clean():
    """Warn when triton's driver shim will fail at the first kernel launch.

    unsloth #2760: training died at the first Triton kernel with

        SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats

    raised from `driver.active.utils.load_binary`, on Python 3.12. Measured
    against the wheels, every triton on PyPI from 3.0.0 to 3.8.0 defines the
    macro in `backends/<backend>/driver.c`, so raising the `triton>=3.0.0` floor
    in pyproject.toml would not have prevented this and cannot: torch pins triton
    exactly (torch 2.6.0 requires `triton==3.2.0`), so any floor above what the
    oldest supported torch carries makes that torch unresolvable rather than
    safe. What is actually broken in these reports is a rebuilt or repackaged
    triton whose shim lost the macro, which no dependency specifier can describe,
    so name it here instead. Warns rather than raises: nothing has failed yet,
    and a process that launches no Triton kernel never will.
    """
    if os.environ.get("UNSLOTH_SKIP_TRITON_SHIM_CHECK", "0").lower() in ("1", "true"):
        return
    try:
        offenders = _triton_driver_shims_missing_py_ssize_t_clean()
    except Exception:
        return
    if not offenders:
        return

    distribution, triton_version = _triton_distribution(offenders[0][1] if offenders else None)
    python_version = ".".join(str(part) for part in sys.version_info[:3])
    reinstall = _triton_repair_advice(distribution, triton_version)

    logger.warning(
        f"Unsloth: {distribution}=={triton_version} ships a "
        f"{', '.join(backend for backend, _ in offenders)} driver shim that includes "
        f"Python.h without defining {_PY_SSIZE_T_CLEAN}, and Python {python_version} "
        f"requires that macro for the '#' argument formats the shim uses. The first "
        f"Triton kernel launch in this process will fail with\n"
        f"    SystemError: {_PY_SSIZE_T_CLEAN} macro must be defined for '#' formats\n"
        f"Every published Triton build defines it, so this is a rebuilt or repackaged one. "
        f"{reinstall}\n"
        f"Python 3.13 and later do not need the macro, so moving to a newer Python "
        f"also clears it. Affected file(s): "
        f"{', '.join(path for _, path in offenders)}. Set "
        f"UNSLOTH_SKIP_TRITON_SHIM_CHECK=1 to silence this."
    )


def _transformers_rescopes_submodule_prefix_renamings():
    """Does this transformers scope a submodule's conversion mapping to where the submodule lives?

    `get_model_conversion_mapping` recurses into every `PreTrainedModel` submodule and
    merges each one's registered conversions into the parent's mapping. A submodule's
    mapping is written against ITS OWN key space, so a renaming anchored at the start of
    the key is meaningless once the submodule is nested, and upstream's correction
    (transformers PR #45567) was to re-scope those by the submodule's dotted path: it
    added a `PrefixChange` transform and a `model_prefix` argument to
    `extract_weight_conversions_for_model`.

    Asked of that API, never of a version number. A transformers without
    `conversion_mapping` at all (4.x) has no recursion to correct, so it answers True as
    well -- True means "this build is fine".

    Upstream has spelled the scoping three ways since, so all three count: the
    `model_prefix` argument of 5.6 to 5.9, `PrefixChange.with_submodel_prefix` over the
    same range, and the `scope_prefix` field every transform carries from 5.10 on. Any one
    of them present means the recursion knows where a submodule's mapping belongs; none of
    them present is the defect. Anything unrecognisable answers True, because a warning is
    worth nothing if it fires on builds nobody can show are broken.
    """
    try:
        from transformers import conversion_mapping
    except Exception:
        return True
    extract = getattr(conversion_mapping, "extract_weight_conversions_for_model", None)
    if extract is None:
        # No per-submodule extraction, so no recursion to correct. This is every 5.x before
        # 5.4.0, which reads the top model's mapping and stops, and it is also the answer
        # for any future build whose machinery we cannot recognise.
        return True
    try:
        from transformers import core_model_loading
    except Exception:
        # Imported separately and allowed to fail: `core_model_loading` pulls in torch, and
        # the two spellings it carries are only ever extra evidence. The signature check
        # below is the one that has to stand on its own.
        core_model_loading = None
    if core_model_loading is not None:
        transform = getattr(core_model_loading, "WeightTransform", None)
        if transform is not None and hasattr(transform, "scope_prefix"):
            return True
        prefix_change = getattr(core_model_loading, "PrefixChange", None)
        if prefix_change is not None and hasattr(prefix_change, "with_submodel_prefix"):
            return True
    try:
        return "model_prefix" in inspect.signature(extract).parameters
    except Exception:
        return True


def _transformers_drops_prequantized_vlm_quant_state():
    """True when the installed transformers loses bnb-4bit quant_state on composite models.

    transformers 5.4.0 made `get_model_conversion_mapping` recurse into
    `PreTrainedModel` submodules (transformers PR #44300). That pulled the
    `qwen3_5_text` `WeightRenaming` -- written for the standalone text model, and
    spelled `^model.language_model.` -> `^model.(?!language_model.)` -- into the
    mapping of the composite `Qwen3_5ForConditionalGeneration`. Renamings run
    before the bitsandbytes converter, so every checkpoint key lost its
    `language_model.` segment: `...weight` was rescued and loaded as a raw packed
    uint8 `nn.Parameter`, while `weight.absmax`, `weight.quant_map`,
    `weight.nested_absmax`, `weight.nested_quant_map` and
    `weight.quant_state.bitsandbytes__nf4` matched nothing, became unexpected keys
    and were discarded. `Bnb4bitDeserialize` therefore never ran and every
    quantized Linear came back with `quant_state = None`.

    transformers PR #45567 replaced that entry with a scoped `PrefixChange` and
    shipped in 5.6.0, so among RELEASES the defect window is exactly 5.4.0 and 5.5.0
    to 5.5.4. That window is not what this asks. It asks
    `_transformers_rescopes_submodule_prefix_renamings`, because a version number is
    the wrong instrument for a source install: upstream main reported `5.3.0.dev0` at
    the commit that introduced the defect and `5.6.0.dev0` at the commit that fixed it,
    so a version interval is wrong at both ends for anyone on git main.
    Measured on unsloth/qwen3.8-27b-unsloth-bnb-4bit: 352 of 352 quantized Linear
    modules lose quant_state on 5.4.0 and 5.5.4, and 0 of 352 on 5.2.0, 5.3.0 and
    every release from 5.6.2 to 5.17.0. Flat text-only checkpoints such as
    unsloth/Qwen3-0.6B-unsloth-bnb-4bit carry no `model.language_model.` prefix
    and are unaffected, which is why the window went unnoticed.

    Three text sub-model types carry a prefix-stripping entry in 5.4.0 and 5.5.x,
    so the affected families are those whose text config is one of
    `qwen3_5_text`, `qwen3_5_moe_text` or `gemma3n_text`.
    """
    return not _transformers_rescopes_submodule_prefix_renamings()


def check_transformers_prequantized_vlm_quant_state():
    """Warn when transformers will silently drop a pre-quantized VLM's quant_state.

    unsloth #9867, #10010, #10017, #10276: loading a pre-quantized bnb-4bit
    multimodal checkpoint raised

        RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x5120 and 1x15728640)

    which reads like a corrupt checkpoint and sent reporters off regenerating
    perfectly good ones. It is not: the loader threw the quantization metadata
    away. See `_transformers_drops_prequantized_vlm_quant_state` for the
    mechanism. Warns rather than raises, and says so before the misleading shape
    error appears: a run that only touches text-only or unquantized checkpoints
    is unaffected, and this must not break it.
    """
    if os.environ.get("UNSLOTH_SKIP_TRANSFORMERS_QUANT_STATE_CHECK", "0").lower() in (
        "1",
        "true",
    ):
        return
    if not _transformers_drops_prequantized_vlm_quant_state():
        return
    try:
        transformers_version = importlib_version("transformers")
    except Exception:
        transformers_version = "unknown"

    # The runtime repair covers exactly these releases, and it is installed in the same
    # import. Warning anyway would tell users to downgrade or upgrade away from a version
    # that now works, which is worse than saying nothing: the advice contradicts the fix.
    # Checked on the live attribute, so a repair that declined to install still warns.
    try:
        from transformers import conversion_mapping
        if getattr(
            getattr(conversion_mapping, "get_model_conversion_mapping", None),
            _COMPOSITE_PREFIX_RENAMING_FLAG,
            False,
        ):
            return
    except Exception:
        pass

    logger.warning(
        f"Unsloth: transformers=={transformers_version} drops the bitsandbytes "
        f"quant_state of pre-quantized multimodal checkpoints while loading them, so "
        f"every quantized layer comes back unquantized and the first forward fails "
        f"with\n"
        f"    RuntimeError: mat1 and mat2 shapes cannot be multiplied (... and 1x...)\n"
        f"The checkpoint is fine and must not be regenerated. This build scopes no "
        f"submodule conversion mapping, which is the defect transformers PR #44300 "
        f"introduced and PR #45567 fixed; among releases that is 5.4.0 and 5.5.0 to "
        f"5.5.4. Move to a transformers that carries the fix, "
        f'`pip install --no-deps "transformers>=5.6.0"` while unsloth still caps at '
        f"5.5.0, or fall back to 5.3.0 or 4.57.6. Text-only checkpoints are "
        f"unaffected. Set UNSLOTH_SKIP_TRANSFORMERS_QUANT_STATE_CHECK=1 to silence "
        f"this."
    )


# Fix TRL OpenEnv 0.26 NameError: name 'SamplingParams' is not defined
def fix_openenv_no_vllm():
    spec = importlib.util.find_spec("trl")
    if spec is None:
        return
    trl_location = spec.origin
    if trl_location is None:
        trl_location = spec.submodule_search_locations[0]
    else:
        trl_location = os.path.split(trl_location)[0]
    openenv = Path(trl_location) / "experimental" / "openenv" / "utils.py"
    if not openenv.exists():
        return

    try:
        with open(openenv, "r+", encoding = "utf-8") as f:
            text = f.read()
            bad = (
                "if is_vllm_available():\n"
                "    from vllm import SamplingParams\n"
                "    from vllm.sampling_params import GuidedDecodingParams\n"
            )
            replace_with = bad + (
                "else:\n"
                "    from typing import Any\n"
                "    SamplingParams = Any\n"
                "    GuidedDecodingParams = Any\n"
                "\n"
            )
            if bad + "\n" + "\n" in text and replace_with not in text:
                text = text.replace(bad + "\n" + "\n", replace_with)
                f.seek(0)
                f.write(text)
                f.truncate()
                logger.info("Unsloth: Patching TRL OpenEnv to fix SamplingParams not defined")
    except Exception as e:
        logger.info(f"Unsloth: Failed patching TRL OpenEnv with error = {str(e)}")


# Fix Exeuctorch needing get_mapped_key
def fix_executorch():
    spec = importlib.util.find_spec("executorch")
    if spec is None:
        return
    executorch_location = spec.origin
    if executorch_location is None:
        executorch_location = spec.submodule_search_locations[0]
    else:
        executorch_location = os.path.split(executorch_location)[0]
    executorch = Path(executorch_location) / "examples" / "models" / "__init__.py"
    if not executorch.exists():
        return

    try:
        what = r"""
        import sys
        import types
        import re
        from typing import Any, Optional
        def get_mapped_key(key: str, mapping_dict: dict[str, str]) -> str:
            try:
                # Checks if there is a layer # in the key
                if any(k.isdigit() for k in key.split(".")):
                    # Replace layer number with "{}" to create key for lookup
                    abstract_key = re.sub(r"(\.\d+)", ".{}", key)
                    layer_num = re.search(r"\d+", key).group(0)
                    new_key = mapping_dict[abstract_key]
                    new_key = new_key.format(layer_num)
                else:
                    new_key = mapping_dict[key]
            except KeyError as e:
                raise Exception(
                    f'Error converting the state dict. Found unexpected key: "{key}". '
                    "Please make sure you're loading a checkpoint with the right format. "
                ) from e

            return new_key

        torchtune = types.ModuleType("torchtune")
        torchtune.__path__ = []
        models = types.ModuleType("torchtune.models")
        models.__path__ = []
        convert_weights = types.ModuleType("torchtune.models.convert_weights")
        convert_weights.get_mapped_key = get_mapped_key
        torchtune.models = models
        models.convert_weights = convert_weights
        sys.modules["torchtune"] = torchtune
        sys.modules["torchtune.models"] = models
        sys.modules["torchtune.models.convert_weights"] = convert_weights
        """
        what = textwrap.dedent(what)

        with open(executorch, "r+", encoding = "utf-8") as f:
            text = f.read()
            bad = "from enum import Enum\n"
            if bad in text and what not in text:
                text = text.replace(bad + "\n", bad + "\n" + what)
                f.seek(0)
                f.write(text)
                f.truncate()
                logger.info("Unsloth: Patching Executorch to fix get_mapped_key")
    except Exception as e:
        logger.info(f"Unsloth: Failed Executorch with error = {str(e)}")


def fix_diffusers_warnings():
    # Silence Flax classes are deprecated and will be removed in Diffusers v1.0.0.
    os.environ["DIFFUSERS_VERBOSITY"] = "error"


def fix_huggingface_hub():
    # huggingface_hub.is_offline_mode got removed, so add it back
    import huggingface_hub
    if not hasattr(huggingface_hub, "is_offline_mode"):
        huggingface_hub.is_offline_mode = lambda: huggingface_hub.constants.HF_HUB_OFFLINE


def fix_triton_compiled_kernel_missing_attrs():
    """
    Triton 3.6.0+ removed direct `num_ctas` and `cluster_dims` attributes from
    CompiledKernel, but torch 2.9.x Inductor still expects them in
    torch/_inductor/runtime/triton_heuristics.py make_launcher() (line ~1757).

    The scope dict eagerly evaluates:
        binary.metadata.num_ctas, *binary.metadata.cluster_dims
    when hasattr(binary, "metadata") is True, but metadata lacks cluster_dims.
    This crashes before reaching the new launch path that doesn't need cta_args.

    Upstream fix: pytorch/pytorch@97bd4db added hasattr guards.
    We monkey-patch CompiledKernel.__init__ to inject the missing attributes
    so the older hasattr(binary, "num_ctas") branch succeeds instead.
    """
    try:
        import torch
    except (ImportError, ModuleNotFoundError):
        return

    try:
        import triton
        import triton.compiler.compiler as triton_compiler
    except (ImportError, ModuleNotFoundError):
        return

    # Only needed when CompiledKernel lacks num_ctas as a direct attr but has metadata
    # (triton >= 3.6.0 with torch < 2.10).
    _ck_cls = triton_compiler.CompiledKernel
    if hasattr(_ck_cls, "num_ctas"):
        return

    _orig_init = _ck_cls.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        if not hasattr(self, "num_ctas"):
            self.num_ctas = getattr(self.metadata, "num_ctas", 1)
        if not hasattr(self, "cluster_dims") and not hasattr(self, "clusterDims"):
            self.cluster_dims = (1, 1, 1)

    _ck_cls.__init__ = _patched_init
    logger.info(
        "Unsloth: Patched triton CompiledKernel with num_ctas/cluster_dims "
        "for torch.compile compatibility."
    )


def fix_dynamo_config_thread_visibility():
    """torch 2.12 made torch._dynamo/_inductor config overrides thread-local
    (ContextVars), so `config.recompile_limit = 1024` set on the main thread is
    invisible to the autograd worker threads that run backward. Gradient
    checkpointing recompiles fullgraph gpt-oss kernels there against the default
    limit of 8, raising FailOnRecompileLimitHit at step 0. Mirror direct config
    assignments into the process-global entry default (torch <= 2.11 semantics).
    config.patch(...) and config.load_config(...) also assign via __setattr__ but
    are thread-local by design, so skip mirroring while inside one (tracked per
    thread). No-op below torch 2.12 and on any torch without this internal layout.
    """
    try:
        import torch

        if Version(torch.__version__) < Version("2.12.0"):
            return
        import torch._dynamo.config as _dynamo_config
        from torch.utils._config_module import ConfigModule
        from contextvars import ContextVar
    except Exception:
        return

    try:
        probe = getattr(_dynamo_config, "_config", {}).get("recompile_limit", None)
        if probe is None or not isinstance(getattr(probe, "user_override", None), ContextVar):
            # Overrides are not context-local on this torch; nothing to fix.
            return
        original_setattr = ConfigModule.__setattr__
        if getattr(original_setattr, "__unsloth_patched__", False):
            return
    except Exception:
        return

    mirrored_modules = ("torch._dynamo.config", "torch._inductor.config")

    # config.patch() and config.load_config() also assign via __setattr__, but their writes are
    # thread-local by design, so a per-thread depth counter keeps them out of the global default.
    import threading

    _scoped_depth = threading.local()

    def _in_scoped_write():
        return getattr(_scoped_depth, "n", 0) > 0

    def _bump(delta):
        _scoped_depth.n = getattr(_scoped_depth, "n", 0) + delta

    original_patch = ConfigModule.patch
    if not getattr(original_patch, "__unsloth_patched__", False):

        @functools.wraps(original_patch)
        def _patched_patch(self, *args, **kwargs):
            ctx = original_patch(self, *args, **kwargs)
            try:
                cls = type(ctx)
                if not getattr(cls, "__unsloth_patch_wrapped__", False):
                    _enter0, _exit0 = cls.__enter__, cls.__exit__

                    def _enter(s, _e = _enter0):
                        _bump(1)
                        try:
                            return _e(s)
                        finally:
                            _bump(-1)

                    def _exit(
                        s,
                        *a,
                        _x = _exit0,
                    ):
                        _bump(1)
                        try:
                            return _x(s, *a)
                        finally:
                            _bump(-1)

                    cls.__enter__, cls.__exit__ = _enter, _exit
                    cls.__unsloth_patch_wrapped__ = True
            except Exception:
                pass
            return ctx

        _patched_patch.__unsloth_patched__ = True
        ConfigModule.patch = _patched_patch

    # load_config restores a saved config by calling setattr per key (thread-local).
    original_load_config = getattr(ConfigModule, "load_config", None)
    if callable(original_load_config) and not getattr(
        original_load_config, "__unsloth_patched__", False
    ):

        @functools.wraps(original_load_config)
        def _patched_load_config(self, *args, **kwargs):
            _bump(1)
            try:
                return original_load_config(self, *args, **kwargs)
            finally:
                _bump(-1)

        _patched_load_config.__unsloth_patched__ = True
        ConfigModule.load_config = _patched_load_config

    @functools.wraps(original_setattr)
    def _patched_setattr(self, name, value):
        original_setattr(self, name, value)
        if _in_scoped_write():
            return  # transient patch / load_config write: keep it thread-local
        # Aliases (cache_size_limit -> recompile_limit) re-enter with the real name.
        if self.__dict__.get("__name__", None) in mirrored_modules:
            try:
                entry = self.__dict__["_config"].get(name, None)
                if entry is not None and entry.alias is None:
                    entry.default = value
            except Exception:
                pass

    _patched_setattr.__unsloth_patched__ = True
    ConfigModule.__setattr__ = _patched_setattr

    # No replay of existing overrides: unsloth installs this before setting any dynamo/inductor
    # config, so the wrapper mirrors every later assignment, and replaying would bake a
    # still-active config.patch override into the global default.
    logger.info(
        "Unsloth: Patched torch config modules so dynamo/inductor settings "
        "(e.g. recompile_limit) apply across threads on torch >= 2.12."
    )


def patch_trunc_normal_precision_issue():
    """
    Patch torch.nn.init.trunc_normal_ for low precision tensors to run init in fp32.

    torch.nn.init.trunc_normal_ can saturate at truncation bounds in fp16/bf16 on
    some versions/backends. This was observed in TorchTitan investigations where
    low-precision truncation produced boundary-heavy initialization behavior:
    https://github.com/pytorch/torchtitan/pull/2342

    To avoid that failure mode, initialize into a temporary fp32 tensor, then copy
    back to the original dtype.
    """
    try:
        import torch
    except (ImportError, ModuleNotFoundError):
        return

    if getattr(torch.nn.init, "_unsloth_trunc_normal_patched", False):
        return

    original_trunc_normal = torch.nn.init.trunc_normal_
    if getattr(original_trunc_normal, "__unsloth_trunc_normal_patched__", False):
        torch.nn.init._unsloth_trunc_normal_patched = True
        return

    low_precision_dtypes = {torch.float16, torch.bfloat16}

    def _call_original(target, mean, std, a, b, generator):
        if generator is None:
            return original_trunc_normal(target, mean = mean, std = std, a = a, b = b)
        try:
            return original_trunc_normal(target, mean = mean, std = std, a = a, b = b, generator = generator)
        except TypeError as exc:
            # Older torch versions may not accept a generator keyword argument.
            msg = str(exc).lower()
            if "unexpected keyword argument" in msg and "generator" in msg:
                return original_trunc_normal(target, mean = mean, std = std, a = a, b = b)
            raise

    try:
        from torch.distributed._tensor import DTensor
    except Exception:
        DTensor = None

    @torch.no_grad()
    def _patched_trunc_normal_(
        tensor,
        mean: float = 0.0,
        std: float = 1.0,
        a: float = -2.0,
        b: float = 2.0,
        generator = None,
    ):
        if DTensor is not None and isinstance(tensor, DTensor):
            local_tensor = getattr(tensor, "_local_tensor", None)
            if local_tensor is None:
                return _call_original(tensor, mean, std, a, b, generator)
            if local_tensor.dtype in low_precision_dtypes:
                local_fp32 = local_tensor.float()
                _call_original(local_fp32, mean, std, a, b, generator)
                local_tensor.copy_(local_fp32.to(dtype = local_tensor.dtype))
                return tensor
            return _call_original(tensor, mean, std, a, b, generator)

        if tensor.dtype in low_precision_dtypes:
            tensor_fp32 = tensor.float()
            _call_original(tensor_fp32, mean, std, a, b, generator)
            tensor.copy_(tensor_fp32.to(dtype = tensor.dtype))
            return tensor

        return _call_original(tensor, mean, std, a, b, generator)

    _patched_trunc_normal_.__unsloth_trunc_normal_patched__ = True
    _patched_trunc_normal_._unsloth_original = original_trunc_normal
    torch.nn.init._unsloth_trunc_normal_original = original_trunc_normal
    torch.nn.init.trunc_normal_ = _patched_trunc_normal_
    torch.nn.init._unsloth_trunc_normal_patched = True
    logger.info("Unsloth: Patched torch.nn.init.trunc_normal_ for fp16/bf16 stability.")


def check_vllm_torch_sm100_compatibility():
    """
    Check for incompatible vLLM + torch < 2.9.0 + SM100 (Blackwell) combination.

    vLLM's distributed module (device_communicators) crashes with std::bad_alloc
    when imported on SM100 GPUs (B200/B100) with torch < 2.9.0. This is due to
    C++ code in vLLM's NCCL/distributed layer being incompatible with older
    torch versions on the newer Blackwell architecture.

    This check runs early (before vLLM import) to provide a helpful error message
    instead of a cryptic std::bad_alloc crash.
    """
    if importlib.util.find_spec("vllm") is None:
        return

    try:
        torch_version = Version(importlib_version("torch"))
        if torch_version >= Version("2.9.0"):
            return  # torch >= 2.9.0 is compatible
    except Exception:
        return  # Can't determine torch version, skip check

    try:
        import torch

        if not torch.cuda.is_available():
            return

        has_sm100 = False
        sm100_gpu_name = None
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            if major == 10:
                has_sm100 = True
                sm100_gpu_name = torch.cuda.get_device_name(i)
                break

        if not has_sm100:
            return
    except Exception:
        return

    try:
        vllm_version = importlib_version("vllm")
    except Exception:
        vllm_version = "unknown"

    # Incompatible combination: raise a helpful error
    raise RuntimeError(
        f"Unsloth: Incompatible configuration detected.\n\n"
        f"  GPU: {sm100_gpu_name} (SM100 / Blackwell architecture)\n"
        f"  torch version: {torch_version}\n"
        f"  vLLM version: {vllm_version}\n\n"
        f"vLLM's distributed module crashes with std::bad_alloc on SM100 GPUs "
        f"(B200/B100/Blackwell) when using torch < 2.9.0.\n\n"
        f"To fix this, please upgrade torch:\n"
        f"  pip install --upgrade torch>=2.9.0\n\n"
        f"Alternatively, if you don't need vLLM:\n"
        f"  pip uninstall vllm"
    )


def fix_vllm_pdl_blackwell():
    """
    Fix vLLM PDL (Programmatic Dependent Launch) bug on Blackwell GPUs (SM100).

    The issue: vLLM's LoRA Triton kernels use tl.extra.cuda.gdc_wait() for PDL
    optimization on SM90+ GPUs. This fails on SM100 (B200/B100) during CUDA graph
    capture because Triton's pipeliner can't handle gdc_wait in complex kernels.

    See: https://github.com/vllm-project/vllm/issues/30872
    """
    if importlib.util.find_spec("vllm") is None:
        return

    try:
        import torch

        if not torch.cuda.is_available():
            return

        has_sm100 = False
        sm100_gpu_name = None
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            if major == 10:
                has_sm100 = True
                sm100_gpu_name = torch.cuda.get_device_name(i)
                break

        if not has_sm100:
            return
    except Exception:
        return

    def _spec_exists(name):
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, OSError, ModuleNotFoundError, ValueError):
            return False

    has_utils = _spec_exists("vllm.lora.ops.triton_ops.utils")
    has_expand_op = _spec_exists("vllm.lora.ops.triton_ops.lora_expand_op")
    has_shrink_op = _spec_exists("vllm.lora.ops.triton_ops.lora_shrink_op")

    if not has_utils and not has_expand_op and not has_shrink_op:
        # Old vLLM version without PDL support: nothing to patch.
        return

    # vLLM version already includes the fix?
    VLLM_PDL_FIX_VERSION = "0.15.0"
    try:
        vllm_version = Version(importlib_version("vllm"))
        if vllm_version >= Version(VLLM_PDL_FIX_VERSION):
            logger.info(
                f"Unsloth: SM100 ({sm100_gpu_name}) detected but vLLM {vllm_version} "
                f"should include PDL fix - skipping workaround"
            )
            return
    except Exception as e:
        logger.debug(f"Unsloth: vLLM version check failed ({e}), applying PDL workaround.")

    os.environ["TRITON_DISABLE_PDL"] = "1"

    def fake_supports_pdl(*args, **kwargs):
        return False

    patched = []
    patched_names = set()

    def _record_patch(name):
        if name not in patched_names:
            patched.append(name)
            patched_names.add(name)

    # Patch the source module (utils.py) where supports_pdl is defined, and clear its @lru_cache to avoid stale results.
    try:
        utils_module = importlib.import_module("vllm.lora.ops.triton_ops.utils")
        if hasattr(utils_module, "supports_pdl"):
            original_fn = utils_module.supports_pdl
            if hasattr(original_fn, "cache_clear"):
                original_fn.cache_clear()
            utils_module.supports_pdl = fake_supports_pdl
            _record_patch("utils")
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass

    # Also patch consumer modules that imported supports_pdl before this ran.
    consumer_modules = {
        "lora_expand_op": "vllm.lora.ops.triton_ops.lora_expand_op",
        "lora_shrink_op": "vllm.lora.ops.triton_ops.lora_shrink_op",
        "fused_moe_lora_op": "vllm.lora.ops.triton_ops.fused_moe_lora_op",
    }
    for name, path in consumer_modules.items():
        try:
            module = importlib.import_module(path)
            if hasattr(module, "supports_pdl"):
                module.supports_pdl = fake_supports_pdl
                _record_patch(name)
        except (ImportError, ModuleNotFoundError, AttributeError):
            pass

    # Patch any additional already-loaded triton ops consumers that expose supports_pdl.
    for module_name, module in tuple(sys.modules.items()):
        if not module_name.startswith("vllm.lora.ops.triton_ops."):
            continue
        if module is None or not hasattr(module, "supports_pdl"):
            continue
        module.supports_pdl = fake_supports_pdl
        _record_patch(module_name.rsplit(".", 1)[-1])

    if patched:
        logger.info(
            f"Unsloth: Applied PDL fix for SM100 ({sm100_gpu_name}) - patched: {', '.join(patched)}"
        )
    else:
        # Just set the env var: vLLM might be an older version without supports_pdl.
        logger.info(f"Unsloth: Set TRITON_DISABLE_PDL=1 for SM100 ({sm100_gpu_name})")


def patch_openspiel_env_async():
    """Apply nest_asyncio for OpenEnv EnvClient async compatibility.

    OpenEnv's EnvClient uses async methods (reset/step). In Jupyter notebooks
    these work via top-level await, but converted scripts need
    asyncio.get_event_loop().run_until_complete() wrappers. Applying nest_asyncio
    ensures nested event loop calls work in all contexts without replacing the
    original async methods (which would break scripts that already have their own
    sync wrappers).
    """
    try:
        import inspect
        from openenv.core.env_client import EnvClient

        if not inspect.iscoroutinefunction(EnvClient.reset):
            return  # Already sync, nothing to do

        try:
            import nest_asyncio
            nest_asyncio.apply()
            logger.info("Unsloth: Applied nest_asyncio for OpenEnv EnvClient async compatibility")
        except ImportError:
            logger.info(
                "Unsloth: nest_asyncio not installed, OpenEnv async methods may need manual wrapping"
            )
    except (ImportError, AttributeError):
        pass  # openenv not installed


def patch_torchcodec_audio_decoder():
    """Call unsloth_zoo's AudioDecoder patch."""
    try:
        from unsloth_zoo.dataset_utils import patch_torchcodec_audio_decoder as _patch
        _patch()
    except (ImportError, AttributeError, RuntimeError):
        pass


# torch.minor -> compatible torchcodec.minor strings (see notebook_validator.py).
# torchcodec ships no `Requires-Dist: torch`, so pip cannot catch a mismatch and this table
# is the only check. Lockstep releases only; 0.12+ is the ABI-stable rule below.
_TORCH_TORCHCODEC_MINORS: dict[str, set[str]] = {
    "2.11": {"0.11"},
    "2.10": {"0.10"},
    "2.9": {"0.8", "0.9"},
    "2.8": {"0.6", "0.7"},
    "2.7": {"0.3", "0.4", "0.5"},
    "2.6": {"0.2"},
    "2.5": {"0.1"},
}


# torch.minor -> the pyproject extra that pins the matching torchcodec line.
_TORCH_TORCHCODEC_EXTRAS: dict[str, str] = {
    "2.11": "audio-torch211",
    "2.10": "audio-torch210",
}

# torchcodec 0.12+ is ABI-stable against torch >=2.11 (its build sets TORCH_TARGET_VERSION
# to 2.11), so that half of the matrix is open-ended rather than a finite set of minors.
# Mirrors notebook_validator.TORCHCODEC_ABI_STABLE_{TORCH,CODEC}.
_TORCHCODEC_ABI_STABLE_TORCH = (2, 11)
_TORCHCODEC_ABI_STABLE_CODEC = (0, 12)


def _torchcodec_exclusive_upper(pin: str) -> str:
    """Next torchcodec minor as an exclusive pip upper bound (0.10 -> <0.11.0)."""
    major, minor = pin.split(".", 1)
    return f"<{major}.{int(minor) + 1}.0"


def _shell_env_ref(name: str) -> str:
    """How to reference an environment variable in the shell this host pastes into.

    PowerShell is Studio's supported Windows shell and does not expand `$NAME`; it needs
    `$env:NAME`, so the POSIX spelling silently produced an empty `--index-url`.
    """
    if sys.platform.startswith("win"):
        return f"$env:{name}"
    return f'"${name}"'


def _torch_index_url_for_remedy(local_tag: str) -> str:
    """The index a torchcodec remedy should name for a `+local_tag` torch.

    An explicitly configured index wins, exactly as it does in
    install_python_stack._torchcodec_index_url: on an authenticated, corporate or
    air-gapped host, telling the user to install from download.pytorch.org either fails
    outright or bypasses the artifact source the install was configured with, and the
    mirror is where the matching build actually is.
    """
    if os.environ.get("UNSLOTH_TORCH_INDEX_URL", "").strip():
        # The variable, not its value: an authenticated mirror carries credentials in its
        # userinfo or query token, and this warning lands in terminals and CI logs. The
        # shell expands it, so the command still works and nothing is written down.
        return _shell_env_ref("UNSLOTH_TORCH_INDEX_URL")
    leaf = os.environ.get("UNSLOTH_TORCH_INDEX_FAMILY", "").strip().strip("/") or local_tag
    if os.environ.get("UNSLOTH_PYTORCH_MIRROR", "").strip():
        # UNSLOTH_PYTORCH_MIRROR replaces the base every install_python_stack index is built
        # from, so naming the public site fails on an air-gapped host. Expanded, not disclosed,
        # as above.
        return f"{_shell_env_ref('UNSLOTH_PYTORCH_MIRROR')}/{leaf}"
    return f"https://download.pytorch.org/whl/{leaf}"


def _torchcodec_version_mismatch_hint() -> str | None:
    """Return a user-facing hint when installed torchcodec mismatches torch."""
    try:
        import importlib.metadata as importlib_metadata
        import torch
        from packaging.version import Version

        torchcodec_version = importlib_metadata.version("torchcodec")
    except Exception:
        return None

    def _release(version: str) -> tuple:
        parts = Version(version.split("+", 1)[0]).release
        return tuple(parts[:2]) + (0,) * (2 - len(parts[:2]))

    def _at_least(version: str, floor: tuple) -> bool:
        # The FULL Version, not its release tuple: PEP 440 sorts `2.11.0rc1` below `2.11`,
        # while `.release` drops the rc and read it as being at the ABI floor. That approved a
        # pairing outside the ABI-stable contract and silenced this hint on it.
        return Version(version.split("+", 1)[0]) >= Version(".".join(str(p) for p in floor))

    try:
        torch_release = _release(torch.__version__)
        codec_release = _release(torchcodec_version)
        torch_at_abi = _at_least(torch.__version__, _TORCHCODEC_ABI_STABLE_TORCH)
        codec_at_abi = _at_least(torchcodec_version, _TORCHCODEC_ABI_STABLE_CODEC)
    except Exception:
        # Non-PEP440 version strings must never break `import unsloth`.
        return None
    if torch_at_abi and codec_at_abi:
        return None  # ABI-stable pairing, not locked to one torch minor
    torch_minor = ".".join(str(p) for p in torch_release)
    codec_minor = ".".join(str(p) for p in codec_release)

    def _index_flag(recommended: "tuple[int, ...]") -> str:
        """`--index-url ...` when torch carries an accelerator tag, else "".

        torchcodec ships one wheel per accelerator, so a remedy that installs the right
        version from the default index still lands a codec that cannot dlopen on a cu126 or
        cu128 venv. Mirrors install_python_stack._torchcodec_index_url, including the lines
        it will not pin: torchcodec 0.1 and 0.2 exist on PyPI only.
        """
        if recommended < (0, 3):
            return ""
        local = str(getattr(torch, "__version__", "")).partition("+")[2].strip().lower()
        if local == "cpu" or re.fullmatch(r"cu\d+", local or ""):
            return f"--index-url {_torch_index_url_for_remedy(local)} "
        return ""

    allowed = _TORCH_TORCHCODEC_MINORS.get(torch_minor)
    if allowed is None:
        # No lockstep row: below the table stays silent; at or past the ABI floor this is a
        # pre-0.12 codec, since 0.12+ already returned above.
        if not torch_at_abi:
            return None
        abi_pin = ".".join(str(p) for p in _TORCHCODEC_ABI_STABLE_CODEC)
        install_hint = (
            f"`pip install {_index_flag(_TORCHCODEC_ABI_STABLE_CODEC)}'torchcodec>={abi_pin}.0'`"
        )
    elif codec_minor in allowed:
        return None
    else:
        pin = sorted(allowed)[-1]
        upper = _torchcodec_exclusive_upper(pin)
        install_hint = f"`pip install {_index_flag(tuple(int(x) for x in pin.split('.')))}'torchcodec>={pin},{upper}'`"
        extra = _TORCH_TORCHCODEC_EXTRAS.get(torch_minor)
        # Only when no index pin is needed. An extra cannot carry one -- the marker picks the
        # version, not the index -- and --index-url on the whole command would resolve unsloth
        # from the torch index too, handing back the same unloadable wheel.
        if extra is not None and not _index_flag(tuple(int(x) for x in pin.split("."))):
            install_hint += f" or `pip install 'unsloth[{extra}]'`"
    return (
        f"torchcodec {torchcodec_version} is incompatible with torch {torch.__version__}; "
        f"install a matching build with {install_hint}."
    )


def _installed_torchcodec_version() -> "str | None":
    """The installed codec's version WITHOUT importing it, or None when nothing is installed.

    This is only ever asked once the codec has FAILED to load, and an unloadable wheel usually
    raises while `torchcodec/__init__` imports its decoders. Python drops a module whose
    initialisation raised, so importing it again just repeats the exception and the accelerator
    mismatch went undiagnosed in exactly the case this exists to name. The metadata is written
    by the installer and needs no native library.
    """
    try:
        from importlib.metadata import version
        return str(version("torchcodec"))
    except Exception:
        pass
    try:
        import torchcodec  # a layout the metadata cannot describe, e.g. a source checkout
        return str(getattr(torchcodec, "__version__", "") or "")
    except Exception:
        return None


def _torchcodec_provenance_hint() -> "str | None":
    """A remedy for a codec whose ACCELERATOR build does not match torch's, or None.

    torchcodec ships one wheel per accelerator, so a cu128 venv holding PyPI's default 0.11
    has the right VERSION and still cannot dlopen -- docker/Dockerfile pins cu128 by hand
    for exactly this. The version hint above returns None there, since the minors agree, so
    without this the codec is disabled with nothing said about how to repair it.

    Only called once the codec has actually failed to load, so a working pairing this cannot
    explain never produces a warning.
    """
    try:
        import torch
    except Exception:
        return None
    codec_version = _installed_torchcodec_version()
    if codec_version is None:
        return None
    torch_local = str(getattr(torch, "__version__", "")).partition("+")[2].strip().lower()
    codec_local = codec_version.partition("+")[2].strip().lower()
    if torch_local == codec_local:
        return None  # same provenance, so the failure is something this cannot name
    if not (torch_local == "cpu" or re.fullmatch(r"cu\d+", torch_local or "")):
        return None  # rocm and xpu publish no torchcodec under this name
    index = _torch_index_url_for_remedy(torch_local)
    codec_from = codec_local or "the default index"
    # The window the table allows, not a bare name: the accelerator indexes carry the whole
    # codec line, so `pip install torchcodec` on torch 2.9 trades a wrong-accelerator build
    # for a wrong-VERSION one and audio stays disabled. Past the table, the ABI floor pins.
    try:
        from packaging.version import Version
        parts = Version(str(torch.__version__).split("+", 1)[0]).release
        torch_release = tuple(parts[:2]) + (0,) * (2 - len(parts[:2]))
    except Exception:
        torch_release = ()
    allowed = _TORCH_TORCHCODEC_MINORS.get(".".join(str(p) for p in torch_release))
    if allowed:
        pin = sorted(allowed)[-1]
        want = f"'torchcodec>={pin},{_torchcodec_exclusive_upper(pin)}'"
    elif torch_release and torch_release >= _TORCHCODEC_ABI_STABLE_TORCH:
        abi = ".".join(str(p) for p in _TORCHCODEC_ABI_STABLE_CODEC)
        want = f"'torchcodec>={abi}.0'"
    else:
        want = "torchcodec"
    # "may be", not "is": this only establishes different indexes. torchcodec is published
    # per accelerator on every line, 0.12+ included, so a mismatch stays possible -- but the
    # load can equally have failed on a missing libavutil, which no reinstall fixes. Name both.
    return (
        f"torchcodec {codec_version or '?'} came from {codec_from} while "
        f"torch {getattr(torch, '__version__', '?')} is a {torch_local} build, so the codec "
        f"may be built for a different accelerator; audio is disabled. Try "
        f"`pip install --force-reinstall --no-deps --index-url {index} {want}`. "
        f"If that does not help, the failure is likely FFmpeg rather than the wheel: "
        f"torchcodec needs libavutil / libavcodec from FFmpeg 4 through 8."
    )


def _ffmpeg_on_loader_path():
    """Every library torchcodec links (per the shipped libtorchcodec_core*.so NEEDED entries) resolvable by the dynamic loader. Distros package them separately, so a host missing only libswscale cannot load the codec; calling that present sends the user at a torch ABI bug."""
    import ctypes.util
    import glob
    import os

    dirs = [d for d in os.environ.get("PATH", "").split(os.pathsep) if d]
    # A prefix on LD_LIBRARY_PATH may ship only versioned files (libavcodec.so.61), which the loader resolves but find_library never sees: it reads the ld cache and linker names.
    libdirs = [
        d
        for v in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")
        for d in os.environ.get(v, "").split(os.pathsep)
        if d
    ]
    for name in ("avutil", "avcodec", "avformat", "avdevice", "avfilter", "swscale", "swresample"):
        if ctypes.util.find_library(name):
            continue
        # find_library does not glob, so walk PATH for Windows names like avutil-59.dll. Windows only: WSL puts the Windows PATH on the Linux one, and those DLLs cannot load here.
        if os.name == "nt" and any(glob.glob(os.path.join(d, name + "-*.dll")) for d in dirs):
            continue
        if os.name != "nt" and any(
            glob.glob(os.path.join(d, "lib" + name + ".so*"))
            or glob.glob(os.path.join(d, "lib" + name + ".*dylib"))
            for d in libdirs
        ):
            continue
        return False
    return True


def _torchcodec_load_failure(exc):
    """Why an installed torchcodec did not import: "ffmpeg" (its FFmpeg libraries are not on the loader path), "native" (they are, so the cause is elsewhere) or "broken" (a failure that does not involve libtorchcodec at all)."""
    import traceback

    # One libtorchcodec message covers a missing FFmpeg, a torch mismatch and other runtime deps, so the text cannot pick between them. Ask the system: FFmpeg missing from the loader path is the one cause establishable here.
    if "libtorchcodec" not in "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    ):
        return "broken"
    return "native" if _ffmpeg_on_loader_path() else "ffmpeg"


_TORCHCODEC_FALLBACK_NOTES = {
    "ffmpeg": "cannot load its FFmpeg libraries; install an FFmpeg full-shared build only if you need torchcodec itself",
    "native": "cannot load its native libraries although FFmpeg is on the loader path; likely an FFmpeg major it does not support (it takes 4 to 8), a missing CUDA NPP runtime (nvidia-npp), or a build that does not match this torch",
    "broken": "fails to import for a reason other than its FFmpeg libraries; reinstall torchcodec against this torch build",
}


def disable_torchcodec_if_broken():
    """Make broken torchcodec behave as if uninstalled (#5446).

    transformers and datasets both detect torchcodec via find_spec, which
    returns True even when the native libs cannot dlopen. We flip their
    flags and seat a sys.modules sentinel so downstream imports fall through
    their existing except ImportError handlers cleanly.
    """
    mismatch_hint = _torchcodec_version_mismatch_hint()
    if mismatch_hint is not None:
        try:
            import warnings
            warnings.warn(mismatch_hint, stacklevel = 2)
        except Exception:
            # Warning filters promoted to errors (PYTHONWARNINGS=error, pytest -W error) must not abort the
            # disable fallback below.
            pass
    try:
        import importlib.util
        if importlib.util.find_spec("torchcodec") is None:
            return  # absent or already disabled

        # RuntimeError on dlopen failure, OSError on chained libavutil.so misses, and a damaged or
        # version-skewed wheel can raise anything else; the package is present, so every shape is "broken".
        from torchcodec.decoders import AudioDecoder
    except Exception as load_error:
        if mismatch_hint is None:
            # Versions agree, so the load failed for another reason. A mismatched accelerator
            # build is the one this can still name, and the one pinning the index repairs.
            try:
                provenance_hint = _torchcodec_provenance_hint()
                if provenance_hint is not None:
                    import warnings
                    warnings.warn(provenance_hint, stacklevel = 2)
            except Exception:
                pass  # a diagnostic must never abort the disable fallback below
        # transformers: flip the flag (<5) and/or rebind the lru_cache'd func (>=5).
        try:
            import transformers.utils.import_utils as tf_import_utils

            try:
                tf_import_utils._torchcodec_available = False
            except AttributeError:
                pass

            is_avail = getattr(tf_import_utils, "is_torchcodec_available", None)
            if is_avail is not None:
                try:
                    is_avail.cache_clear()
                except AttributeError:
                    pass
                tf_import_utils.is_torchcodec_available = lambda: False
        except ImportError:
            pass

        # datasets >= 4.0: own flag gating audio/video/features/formatters.
        try:
            import datasets.config as datasets_config
            if hasattr(datasets_config, "TORCHCODEC_AVAILABLE"):
                datasets_config.TORCHCODEC_AVAILABLE = False
        except ImportError:
            pass

        # Drop half-loaded entries and seat the absence sentinel: after this, `import torchcodec` raises
        # ModuleNotFoundError and find_spec returns None.
        for _stale in [
            n
            for n in list(sys.modules)
            if n == "torchcodec"
            or n.startswith("torchcodec.")
            or n == "datasets.features._torchcodec"
        ]:
            sys.modules.pop(_stale, None)
        sys.modules["torchcodec"] = None
        decodes = patch_datasets_audio_decoding_without_torchcodec()
        try:
            import warnings

            note = _TORCHCODEC_FALLBACK_NOTES[_torchcodec_load_failure(load_error)]
            tail = (
                "audio datasets decode through soundfile and PyAV instead (wav/flac/mp3/ogg, m4a/aac/webm)"
                if decodes
                else "audio datasets will not decode until soundfile and PyAV are installed (pip install soundfile av)"
            )
            warnings.warn(f"Unsloth: torchcodec is installed but {note}; {tail}.", stacklevel = 2)
        except Exception:
            pass  # a report must never abort the disable fallback above


def _audio_decode_with_av(source, stream_index = None):
    """Mono float32 at the native rate through PyAV's bundled FFmpeg: every container torchcodec would have read (m4a, aac, webm, wma, amr) without a system FFmpeg. Kept identical to studio/backend/utils/datasets/audio_decode.py; a test holds the two together."""
    import av
    import numpy as np

    chunks = []
    rate = 0
    resampler = None
    with av.open(source, mode = "r", metadata_errors = "ignore") as container:
        if not container.streams.audio:
            raise ValueError("audio container has no audio stream")
        # datasets.Audio(stream_index=...) is the container's absolute stream index, as torchcodec reads it; None is the best audio stream.
        try:
            if stream_index is None:
                # av_find_best_stream, which torchcodec uses: the default-disposition track wins over the first. PyAV < 13 has no wrapper, so take the first audio track there.
                best = getattr(container.streams, "best", None)
                stream = best("audio") if best is not None else container.streams.audio[0]
            else:
                stream = container.streams[stream_index]
        except IndexError:
            raise ValueError(
                f"stream {stream_index} is not in the container, which has {len(container.streams)} streams"
            ) from None
        if stream.type != "audio":
            raise ValueError(f"stream {stream_index} is not an audio stream")
        for frame in container.decode(stream):
            if resampler is None:
                rate = int(frame.sample_rate or 0)
                if rate <= 0:
                    raise ValueError("decoded audio has an invalid sample rate")
                resampler = av.AudioResampler(format = "flt", layout = "mono", rate = rate)
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        if resampler is not None:
            for out in resampler.resample(None):
                chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError("audio container decoded to no samples")
    return np.concatenate(chunks).astype(np.float32, copy = False), rate


def _audio_read_mono(source, stream_index = None):
    """soundfile first (wav, flac, mp3, ogg), PyAV for the rest. `source` is a path, a bytes buffer or an open file."""
    import numpy as np
    import soundfile as sf

    if stream_index not in (None, 0):
        # libsndfile only knows single-stream files, so an explicit other stream is PyAV's alone.
        return _audio_decode_with_av(source, stream_index)
    try:
        array, rate = sf.read(source, dtype = "float32", always_2d = False)
    except Exception as sf_error:  # noqa: BLE001  libsndfile raises its own hierarchy
        try:
            import av  # noqa: F401
        except ImportError:
            raise sf_error
        if hasattr(source, "seek"):
            source.seek(0)
        try:
            return _audio_decode_with_av(source, stream_index)
        except Exception as av_error:  # noqa: BLE001
            raise RuntimeError(
                f"audio could not be decoded by soundfile ({sf_error}) or PyAV ({av_error})"
            ) from av_error
    if array.ndim > 1:
        # soundfile returns (frames, channels); torchcodec returns (channels, frames).
        array = np.mean(array, axis = -1)
    return array, rate


def _audio_resample(array, rate, target):
    """librosa when installed, else swresample through PyAV; both are already on the audio extras."""
    try:
        import librosa
    except Exception:  # noqa: BLE001  an old librosa beside numpy 2 raises AttributeError at import, not ImportError
        librosa = None
    if librosa is not None:
        return librosa.resample(array, orig_sr = rate, target_sr = target)
    import av
    import numpy as np

    frame = av.AudioFrame.from_ndarray(
        np.ascontiguousarray(array, dtype = np.float32)[np.newaxis, :], format = "flt", layout = "mono"
    )
    frame.sample_rate = rate
    resampler = av.AudioResampler(format = "flt", layout = "mono", rate = target)
    chunks = [out.to_ndarray().reshape(-1) for out in resampler.resample(frame)]
    chunks += [out.to_ndarray().reshape(-1) for out in resampler.resample(None)]
    return np.concatenate(chunks)


def patch_datasets_audio_decoding_without_torchcodec():
    """Keep `datasets` Audio columns decodable when torchcodec is unusable (#8642).

    `datasets` >= 4 decodes audio only through torchcodec, which needs an FFmpeg full-shared
    install to dlopen its native libraries; with `TORCHCODEC_AVAILABLE` cleared above every
    audio row raises "please install torchcodec" for a package that is installed. This seats
    a decoder on `datasets.features.audio.Audio` that reads through soundfile, then PyAV's
    bundled FFmpeg, and resamples to the cast rate, returning the pre-4.0 dict contract
    `{"path", "array", "sampling_rate"}`. Studio installs the same decoder from its own copy
    (utils/datasets/audio_decode.py) because its API process never imports unsloth.
    No-op on `datasets` < 4, on a working torchcodec, and without soundfile. Idempotent.
    """
    try:
        from datasets import config
        from datasets.features.audio import Audio
    except ImportError:
        return False
    if not hasattr(config, "TORCHCODEC_AVAILABLE") or config.TORCHCODEC_AVAILABLE:
        return False
    if getattr(Audio, "_unsloth_audio_fallback", False):
        return True
    try:
        import soundfile  # noqa: F401
    except Exception:  # noqa: BLE001  libsndfile absent raises OSError, not ImportError
        return False
    original_encode = Audio.encode_example

    def _token_for_url(path, token_per_repo_id):
        if not token_per_repo_id:
            return None
        try:
            from datasets.utils.py_utils import string_to_dict

            source_url = path.split("::")[-1]
            pattern = (
                config.HUB_DATASETS_URL
                if source_url.startswith(config.HF_ENDPOINT)
                else config.HUB_DATASETS_HFFS_URL
            )
            fields = string_to_dict(source_url, pattern)
        except Exception:  # noqa: BLE001
            fields = None
        if fields is None:
            values = list(token_per_repo_id.values())
            return values[0] if len(values) == 1 else None
        return token_per_repo_id.get(fields["repo_id"])

    def decode_example(
        self,
        value,
        token_per_repo_id = None,
    ):
        import io

        from datasets.download.download_config import DownloadConfig
        from datasets.utils.file_utils import is_local_path, xopen

        if not self.decode:
            raise RuntimeError(
                "Decoding is disabled for this feature. Please use Audio(decode=True) instead."
            )
        path, raw = value["path"], value["bytes"]
        if path is None and raw is None:
            raise ValueError(
                f"An audio sample should have one of 'path' or 'bytes' but both are None in {value}."
            )
        if raw is not None:
            source = io.BytesIO(raw)
        elif is_local_path(path):
            source = path
        else:
            source = xopen(
                path,
                "rb",
                download_config = DownloadConfig(token = _token_for_url(path, token_per_repo_id)),
            )
        array, sampling_rate = _audio_read_mono(source, getattr(self, "stream_index", None))
        target = self.sampling_rate
        if target and sampling_rate != target:
            array = _audio_resample(array, sampling_rate, target)
            sampling_rate = target
        return {"path": path, "array": array, "sampling_rate": sampling_rate}

    def encode_example(self, value):
        import io
        from pathlib import Path

        import soundfile as sf

        if isinstance(value, str):
            return {"bytes": None, "path": value}
        if isinstance(value, Path):
            return {"bytes": None, "path": str(value.absolute())}
        if isinstance(value, (bytes, bytearray)):
            return {"bytes": bytes(value), "path": None}
        if isinstance(value, dict) and value.get("array") is not None:
            import numpy as np

            array = np.asarray(value["array"])
            if array.dtype == object:
                array = np.asarray(
                    array.tolist(), dtype = "float32"
                )  # a nested list back from Arrow arrives as an object array
            if array.ndim == 2 and array.shape[0] < array.shape[1]:
                array = (
                    array.T
                )  # torchcodec hands out (channels, samples); libsndfile writes (frames, channels)
            buf = io.BytesIO()
            sf.write(buf, array, value["sampling_rate"], format = "WAV")
            return {"bytes": buf.getvalue(), "path": value.get("path")}
        if isinstance(value, dict) and ("bytes" in value or "path" in value):
            return {"bytes": value.get("bytes"), "path": value.get("path")}
        return original_encode(self, value)

    Audio.decode_example = decode_example
    Audio.encode_example = encode_example
    Audio._unsloth_audio_fallback = True
    return True


def disable_torchaudio_if_cuda_mismatched():
    """Make a CUDA-mismatched torchaudio behave as if uninstalled.

    `torchaudio._extension.utils._check_cuda_version` compares the CUDA
    version torchaudio was BUILT against with torch's, and raises on any
    difference:

        RuntimeError: Detected that PyTorch and TorchAudio were compiled with
        different CUDA versions.

    That check runs at extension init, so it takes down the whole import --
    including for callers that only ever wanted CPU-side audio I/O, and
    including callers that never asked for torchaudio at all and merely
    imported something that does. Measured on a Kaggle 2xT4 session running
    `Kaggle-Muse_Glimmer_(30B)-GRPO`, a text model: it died at cell 4 on this,
    having never reached anything to do with audio.

    Same shape as `disable_torchcodec_if_broken` and for the same reason: the
    package is present, `find_spec` says so, and the failure is at native
    init rather than at resolution, so downstream `except ImportError`
    handlers never get their chance. Seating the sentinel gives them one.

    What this deliberately does NOT do is patch out `_check_cuda_version`.
    The check is right -- torchaudio's CUDA ops really are unusable against a
    different runtime -- and silencing it in place would leave those ops
    reachable and wrong. Making the package absent is the honest version of
    the same repair, and it is loud: the warning names both versions and the
    wheel that would fix it.
    """
    try:
        import importlib.util
        if importlib.util.find_spec("torchaudio") is None:
            return
        import torchaudio  # noqa: F401
    except ImportError:
        return
    except (RuntimeError, OSError) as exc:
        if "different CUDA versions" not in str(exc) and "torchaudio" not in str(exc).lower():
            # Some other failure: swallowing it would hide a real one behind a message about CUDA versions.
            raise
        try:
            import warnings
            warnings.warn(
                f"Unsloth: torchaudio cannot initialise against this torch and has been "
                f"disabled for this process, so anything that needs it will report it as "
                f"missing rather than crash at import. Install the matching wheel to "
                f"restore it. Original error: {exc}",
                stacklevel = 2,
            )
        except Exception:
            # Warning filters promoted to errors must not abort the repair.
            pass

        try:
            import transformers.utils.import_utils as tf_import_utils
            try:
                tf_import_utils._torchaudio_available = False
            except AttributeError:
                pass
            # `speech` is transformers' composite backend and is nothing but torchaudio (is_speech_available
            # returns is_torchaudio_available()). On 4.x both read one module global; on 5.x each is
            # separately lru_cached, so an answer computed before this repair stays True and
            # requires_backends(..., "speech") waves callers into a torchaudio that is now a None sentinel.
            for _name in ("is_torchaudio_available", "is_speech_available"):
                is_avail = getattr(tf_import_utils, _name, None)
                if is_avail is None:
                    continue
                try:
                    is_avail.cache_clear()
                except AttributeError:
                    pass
                setattr(tf_import_utils, _name, lambda: False)
        except ImportError:
            pass

        try:
            import datasets.config as datasets_config
            if hasattr(datasets_config, "TORCHAUDIO_AVAILABLE"):
                datasets_config.TORCHAUDIO_AVAILABLE = False
        except ImportError:
            pass

        for _stale in [
            n for n in list(sys.modules) if n == "torchaudio" or n.startswith("torchaudio.")
        ]:
            sys.modules.pop(_stale, None)
        sys.modules["torchaudio"] = None


def disable_broken_wandb():
    """Disable wandb if it's installed but cannot actually import.

    wandb can fail to import when there's a protobuf version mismatch
    (e.g., wandb < 0.19.11 with protobuf >= 6.0). This causes cascading
    import failures through trl -> transformers/accelerate -> wandb that
    crash unsloth's import chain.

    There are two separate is_wandb_available() functions used by trl:
      - transformers.integrations.integration_utils.is_wandb_available
        (used by most trl trainers)
      - accelerate.utils.imports.is_wandb_available
        (used by trl/trainer/callbacks.py)

    Both must be patched to fully prevent broken wandb imports.
    """
    if importlib.util.find_spec("wandb") is None:
        return  # wandb not installed, nothing to do

    try:
        import wandb
    except Exception:
        # wandb is installed but broken: patch every checker to skip it.
        logger.info(
            "Unsloth: wandb is installed but broken (likely a protobuf version mismatch). "
            "Disabling wandb to prevent import errors. To fix, run: pip install --upgrade wandb"
        )
        _wandb_false = lambda: False
        try:
            import transformers.integrations.integration_utils as tf_integration
            tf_integration.is_wandb_available = _wandb_false
        except (ImportError, AttributeError):
            pass
        # Patch accelerate.utils.imports and the accelerate.utils re-export, since
        # `from accelerate.utils import is_wandb_available` reads the latter.
        try:
            import accelerate.utils.imports as acc_imports
            acc_imports.is_wandb_available = _wandb_false
        except (ImportError, AttributeError):
            pass
        try:
            import accelerate.utils as acc_utils
            acc_utils.is_wandb_available = _wandb_false
        except (ImportError, AttributeError):
            pass
        os.environ["WANDB_DISABLED"] = "true"


# peft 0.19.x's transformers_weight_conversion.py imports transformers.conversion_mapping and
# transformers.core_model_loading at module top; neither exists on transformers <5, so the
# import raises ModuleNotFoundError, swallowed by the bare except below. Stub the two
# submodules only when broken; peft calls them only behind `if is_transformers_ge_v5:`.

# Stamped on stub modules so a second call is a strict no-op and third parties can introspect __unsloth_stub__.
# ---------------------------------------------------------------------------

_UNSLOTH_STUB_SENTINEL = "__unsloth_stub__"
_PEFT_TENSOR_PARALLEL_FALLBACK_SYMBOLS = (
    "ALL_PARALLEL_STYLES",
    "ColwiseParallel",
    "EmbeddingParallel",
    "RowwiseParallel",
)


def _extract_peft_tensor_parallel_imported_symbols():
    """Return names PEFT imports from ``transformers.integrations.tensor_parallel``.

    Parsed from ``peft.utils.save_and_load._maybe_shard_state_dict_for_tp`` to
    avoid a stale hard-coded symbol list.
    """
    try:
        import peft.utils.save_and_load as _save_and_load
    except Exception:
        return ()
    try:
        sharding_fn = _save_and_load._maybe_shard_state_dict_for_tp
    except AttributeError:
        return ()

    try:
        source = inspect.getsource(sharding_fn)
    except Exception as exc:
        logger.debug("Failed to inspect PEFT tensor-parallel imports: %r", exc)
        return _PEFT_TENSOR_PARALLEL_FALLBACK_SYMBOLS

    import_pattern = re.compile(
        r"from\s+transformers\.integrations\.tensor_parallel\s+import\s*\((.*?)\)",
        re.S,
    )
    import_pattern_single = re.compile(
        r"from\s+transformers\.integrations\.tensor_parallel\s+import\s+([A-Za-z_][A-Za-z0-9_\s,]*)",
        re.S,
    )
    matches = import_pattern.findall(source)
    if not matches:
        matches = import_pattern_single.findall(source)

    symbols = []
    seen = set()
    for match in matches:
        pieces = re.split(r"[,\n]", match)
        for piece in pieces:
            candidate = piece.strip()
            if not candidate:
                continue
            if candidate.endswith(")"):
                candidate = candidate[:-1].strip()
            if not candidate.isidentifier():
                continue
            if candidate in seen:
                continue
            symbols.append(candidate)
            seen.add(candidate)
    return tuple(symbols) or _PEFT_TENSOR_PARALLEL_FALLBACK_SYMBOLS


def _raise_on_peft_tensor_parallel_symbol_use(symbol_name):
    raise NotImplementedError(
        f"Unsloth: cannot use unsupported "
        f"`transformers.integrations.tensor_parallel.{symbol_name}` on this "
        f"transformers installation. Please upgrade transformers before "
        f"using PEFT tensor-parallel adapter sharding features."
    )


def fix_peft_transformers_tensor_parallel_import_compat():
    """Add placeholders to ``transformers.integrations.tensor_parallel`` for symbols
    PEFT expects but this transformers build omits, keeping existing objects.

    Returns ``True`` when patched, ``False`` when no patch is needed, ``None``
    when transformers / PEFT context is absent.
    """
    try:
        tensor_parallel_spec = importlib.util.find_spec("transformers.integrations.tensor_parallel")
    except ModuleNotFoundError:
        return None
    if tensor_parallel_spec is None:
        return None

    required_symbols = _extract_peft_tensor_parallel_imported_symbols()
    if not required_symbols:
        return None

    try:
        tp_mod = importlib.import_module("transformers.integrations.tensor_parallel")
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "transformers",
            "transformers.integrations",
            "transformers.integrations.tensor_parallel",
        }:
            raise
        return None
    missing = [symbol for symbol in required_symbols if not hasattr(tp_mod, symbol)]
    if not missing:
        return False

    def _install_symbol_placeholder(symbol_name):
        if symbol_name == "ALL_PARALLEL_STYLES":

            class _UnslothTensorParallelStyles(dict):
                def __getitem__(self, key):
                    _raise_on_peft_tensor_parallel_symbol_use(symbol_name)

                def get(self, *args, **kwargs):
                    _raise_on_peft_tensor_parallel_symbol_use(symbol_name)

                def __contains__(self, key):
                    _raise_on_peft_tensor_parallel_symbol_use(symbol_name)

                def __iter__(self):
                    _raise_on_peft_tensor_parallel_symbol_use(symbol_name)

                def __len__(self):
                    _raise_on_peft_tensor_parallel_symbol_use(symbol_name)

            value = _UnslothTensorParallelStyles()
        else:

            class _UnslothTensorParallelPlaceholder:
                def __init__(self, *args, **kwargs):
                    _raise_on_peft_tensor_parallel_symbol_use(symbol_name)

            value = _UnslothTensorParallelPlaceholder
            value.__name__ = f"UnslothTensorParallelPlaceholder{symbol_name}"

        setattr(value, _UNSLOTH_STUB_SENTINEL, True)
        setattr(tp_mod, symbol_name, value)

    for symbol in missing:
        _install_symbol_placeholder(symbol)

    return True


def _peft_stub_module_importable(name):
    """True iff ``import {name}`` would succeed without side effects."""
    if name in sys.modules and sys.modules[name] is not None:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _make_peft_stub_module(fullname):
    import types as _types

    mod = _types.ModuleType(fullname)
    mod.__file__ = f"<unsloth stub: {fullname}>"
    mod.__package__ = fullname.rpartition(".")[0]
    setattr(mod, _UNSLOTH_STUB_SENTINEL, True)
    return mod


def _build_transformers_conversion_mapping_stub():
    """Build (not install) peft's 3 symbols, so the same objects can also
    backfill a REAL module missing only some of them."""
    mod = _make_peft_stub_module("transformers.conversion_mapping")

    # peft does .copy() plus keyed assignment at module top; a real dict suffices.
    mod._MODEL_TO_CONVERSION_PATTERN = {}

    def get_checkpoint_conversion_mapping(model_type, *args, **kwargs):
        # None is peft's "no conversion registered"; both callsites early-return on it.
        return None

    def get_model_conversion_mapping(model, *args, **kwargs):
        return None

    mod.get_checkpoint_conversion_mapping = get_checkpoint_conversion_mapping
    mod.get_model_conversion_mapping = get_model_conversion_mapping
    return mod


def _install_transformers_conversion_mapping_stub():
    """Stub the 3 symbols peft 0.19.x imports from this module at top level."""
    name = "transformers.conversion_mapping"
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, _UNSLOTH_STUB_SENTINEL, False):
        return existing

    mod = _build_transformers_conversion_mapping_stub()
    sys.modules[name] = mod
    # Attach to parent so attribute-style access matches a real submodule.
    parent = sys.modules.get("transformers")
    if parent is not None and not hasattr(parent, "conversion_mapping"):
        try:
            parent.conversion_mapping = mod
        except Exception:
            # Frozen parent: the sys.modules entry is enough for `from ... import`.
            pass
    return mod


def _build_transformers_core_model_loading_stub():
    """Build (not install) peft's 8 symbols, so the same objects can also
    backfill a REAL module missing only some of them.

    ``Concatenate`` and ``ConversionOps`` MUST be real classes (peft subclasses
    them at module top); the rest only appear in runtime calls gated behind
    ``is_transformers_ge_v5``."""
    mod = _make_peft_stub_module("transformers.core_model_loading")

    class ConversionOps:
        def convert(self, *args, **kwargs):  # pragma: no cover - inert stub
            raise NotImplementedError(
                "unsloth stub: transformers.core_model_loading.ConversionOps "
                "is a no-op on transformers <5. Upgrade transformers to v5+ "
                "to use peft.utils.transformers_weight_conversion at runtime."
            )

        @property
        def reverse_op(self):  # pragma: no cover - inert stub
            raise NotImplementedError

    class Concatenate(ConversionOps):
        def __init__(
            self,
            dim = 0,
            *args,
            **kwargs,
        ):
            self.dim = dim

    class MergeModulelist(ConversionOps):
        def __init__(self, *args, **kwargs):
            pass

    class Transpose(ConversionOps):
        def __init__(
            self,
            dim0 = 0,
            dim1 = 1,
            *args,
            **kwargs,
        ):
            self.dim0 = dim0
            self.dim1 = dim1

    class WeightConverter:
        def __init__(self, *args, **kwargs):
            # Accept any signature; upstream class evolves.
            self.args = args
            self.kwargs = kwargs

    class WeightRenaming:
        def __init__(
            self,
            source_patterns = None,
            target_patterns = None,
            *args,
            **kwargs,
        ):
            self.source_patterns = source_patterns
            self.target_patterns = target_patterns

    def dot_natural_key(key):
        return key

    def rename_source_key(original_key, renamings, converters):
        return original_key, None

    mod.ConversionOps = ConversionOps
    mod.Concatenate = Concatenate
    mod.MergeModulelist = MergeModulelist
    mod.Transpose = Transpose
    mod.WeightConverter = WeightConverter
    mod.WeightRenaming = WeightRenaming
    mod.dot_natural_key = dot_natural_key
    mod.rename_source_key = rename_source_key
    return mod


def _install_transformers_core_model_loading_stub():
    """Install the core_model_loading stub, unless a real module is present."""
    name = "transformers.core_model_loading"
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, _UNSLOTH_STUB_SENTINEL, False):
        return existing

    mod = _build_transformers_core_model_loading_stub()
    sys.modules[name] = mod
    parent = sys.modules.get("transformers")
    if parent is not None and not hasattr(parent, "core_model_loading"):
        try:
            parent.core_model_loading = mod
        except Exception:
            pass
    return mod


# Names peft's transformers_weight_conversion imports at module top; a real module missing ANY
# of them breaks that import as hard as an absent one.
_PEFT_REQUIRED_SYMBOLS = {
    "transformers.conversion_mapping": (
        "_MODEL_TO_CONVERSION_PATTERN",
        "get_checkpoint_conversion_mapping",
        "get_model_conversion_mapping",
    ),
    "transformers.core_model_loading": (
        "Concatenate",
        "ConversionOps",
        "MergeModulelist",
        "Transpose",
        "WeightConverter",
        "WeightRenaming",
        "dot_natural_key",
        "rename_source_key",
    ),
}
_PEFT_STUB_BUILDERS = {
    "transformers.conversion_mapping": _build_transformers_conversion_mapping_stub,
    "transformers.core_model_loading": _build_transformers_core_model_loading_stub,
}


def _backfill_missing_peft_symbols(name):
    """Add to a REAL transformers submodule only the peft symbols it lacks.

    transformers 5.0.0.dev0 ships ``conversion_mapping`` without
    ``_MODEL_TO_CONVERSION_PATTERN``, so peft's top-level import raises
    ImportError even though the module imports fine; stubbing it wholesale would
    replace working transformers code. The donors are inert, which is right
    where the symbol never existed but wrong for a transformers 5 that HAS
    conversions and merely renamed one, hence the warning.

    Strictly additive and idempotent. Returns the names added."""
    try:
        mod = importlib.import_module(name)
    except Exception:
        return ()
    if getattr(mod, _UNSLOTH_STUB_SENTINEL, False):
        return ()  # our own stub already provides the full set
    missing = [s for s in _PEFT_REQUIRED_SYMBOLS[name] if not hasattr(mod, s)]
    if not missing:
        return ()
    donor = _PEFT_STUB_BUILDERS[name]()
    added = []
    for symbol in missing:
        try:
            setattr(mod, symbol, getattr(donor, symbol))
            added.append(symbol)
        except Exception:
            pass  # frozen or slotted module object
    if added:
        _warn_peft_symbols_backfilled(name, added)
    return tuple(added)


# An empty pattern is peft's own starting point; the rest stand in for real upstream
# behaviour, so those are worth warning about.
_PEFT_INERT_BACKFILL_IS_FINE = frozenset(("_MODEL_TO_CONVERSION_PATTERN",))


def _warn_peft_symbols_backfilled(name, added):
    """Say when an inert stand-in went into a module that is otherwise real."""
    substantive = [s for s in added if s not in _PEFT_INERT_BACKFILL_IS_FINE]
    if not substantive:
        return
    warnings.warn(
        f"Unsloth: {name} is missing {', '.join(substantive)}, so peft could "
        f"not be imported. Added inert stand-ins to get the import through; "
        f"weight conversions that rely on them will be skipped. Upgrading "
        f"transformers is the real fix.",
        RuntimeWarning,
        stacklevel = 2,
    )


def fix_peft_transformers_weight_conversion_import():
    """Make ``from peft.utils import transformers_weight_conversion`` import
    cleanly on (peft 0.19.x, transformers 4.x) by stubbing the two missing
    transformers-v5 submodules. See header block above for details.

    Must run BEFORE ``patch_peft_weight_converter_compatibility`` -- that
    function's bare ``except (ImportError, AttributeError): return`` would
    otherwise silently no-op.

    No-op if peft / transformers missing, or if the peft module already
    imports cleanly. Idempotent and strictly additive (never overwrites a
    real ``transformers.conversion_mapping`` / ``core_model_loading``).

    Returns True if patched, False if no action needed, None if peft absent."""
    if importlib.util.find_spec("peft") is None:
        return None

    # Already importable: either we patched, or transformers is v5+.
    try:
        importlib.import_module("peft.utils.transformers_weight_conversion")
        return False
    except ModuleNotFoundError as exc:
        # Only act on our specific drift class.
        missing = getattr(exc, "name", "") or ""
        if missing not in (
            "transformers.conversion_mapping",
            "transformers.core_model_loading",
        ):
            return False
    except ImportError as exc:
        # Older Python ImportError has no .name; string-match instead.
        msg = str(exc)
        if (
            "transformers.conversion_mapping" not in msg
            and "transformers.core_model_loading" not in msg
        ):
            return False

    # Need transformers loaded to attach stubs to its package.
    transformers_root = sys.modules.get("transformers")
    if transformers_root is None:
        try:
            transformers_root = importlib.import_module("transformers")
        except Exception:
            return False

    # Stub only the genuinely missing submodules; never clobber real ones.
    patched_any = False
    if not _peft_stub_module_importable("transformers.conversion_mapping"):
        _install_transformers_conversion_mapping_stub()
        patched_any = True

    if not _peft_stub_module_importable("transformers.core_model_loading"):
        _install_transformers_core_model_loading_stub()
        patched_any = True

    # Present but incomplete: transformers 5.x kept both modules and dropped names peft still
    # imports at module top. The stubs above only fire when a module is ABSENT, so backfill the
    # missing names onto the real module, which is strictly additive.
    patched_any = _backfill_missing_conversion_symbols() or patched_any

    # An importable submodule can still lack individual symbols; backfill just those rather than
    # replacing a real module wholesale.
    backfilled = {}
    for _submodule in _PEFT_REQUIRED_SYMBOLS:
        added = _backfill_missing_peft_symbols(_submodule)
        if added:
            backfilled[_submodule] = added
            patched_any = True
    if backfilled:
        logger.info(
            "Unsloth: backfilled peft symbols missing from transformers: "
            + "; ".join(f"{m}: {', '.join(s)}" for m, s in backfilled.items())
        )

    if not patched_any:
        # Real submodules present and complete; the failure was for another reason.
        return False

    # Force a fresh import now that stubs are in place, dropping any cached None entry first so importlib retries.
    pkg = "peft.utils.transformers_weight_conversion"
    if pkg in sys.modules and sys.modules[pkg] is None:
        del sys.modules[pkg]
    try:
        importlib.import_module(pkg)
    except Exception:
        # Other upstream drift; stubs stay installed so a later retry succeeds.
        return True

    logger.info(
        "Unsloth: stubbed transformers.conversion_mapping / "
        "transformers.core_model_loading so peft.utils."
        "transformers_weight_conversion imports cleanly on "
        "transformers <5."
    )
    return True


# What peft.utils.transformers_weight_conversion imports at module top, kept beside the stubs
# so the two lists cannot drift apart.
_PEFT_CONVERSION_SYMBOLS = {
    "transformers.conversion_mapping": (
        "_MODEL_TO_CONVERSION_PATTERN",
        "get_checkpoint_conversion_mapping",
        "get_model_conversion_mapping",
    ),
    "transformers.core_model_loading": (
        "Concatenate",
        "ConversionOps",
        "MergeModulelist",
        "Transpose",
        "WeightConverter",
        "WeightRenaming",
        "dot_natural_key",
        "rename_source_key",
    ),
}

# Of those, the ones peft calls rather than merely imports. The stubs are deliberately inert: on
# transformers <5 the whole module is ours and peft's converter never runs. An inert body on a REAL
# transformers is a different matter, since peft would call it and get a wrong answer, so those get a
# placeholder that says what is wrong instead.
_PEFT_CONVERSION_RUNTIME_SYMBOLS = frozenset(
    (
        "transformers.core_model_loading.dot_natural_key",
        "transformers.core_model_loading.rename_source_key",
        "transformers.core_model_loading.WeightRenaming",
        "transformers.core_model_loading.WeightConverter",
        # build_peft_weight_mapping buckets entries with isinstance(op, Concatenate) /
        # isinstance(op, MergeModulelist) and builds Transpose outright, so an inert stub silences the
        # isinstance arms and the conversion is skipped. ConversionOps stays import-only: peft
        # subclasses it and never asks about instances.
        "transformers.core_model_loading.Concatenate",
        "transformers.core_model_loading.MergeModulelist",
        "transformers.core_model_loading.Transpose",
        "transformers.conversion_mapping.get_checkpoint_conversion_mapping",
        "transformers.conversion_mapping.get_model_conversion_mapping",
    )
)


def _unsupported_conversion_symbol(qualified, donor_value = None):
    """A stand-in that satisfies the import and refuses to answer wrongly.

    Shaped like whatever it replaces: peft runs `isinstance(entry, X)` on the
    class-valued names, so those stay classes -- never instantiable, which
    makes the isinstance answer False, and False is right when the class does
    not exist.
    """
    short = qualified.rsplit(".", 1)[-1]
    message = (
        f"Unsloth: this transformers does not provide {qualified}, which "
        "peft.utils.transformers_weight_conversion calls to convert LoRA "
        "weights. Unsloth supplied a placeholder so the import succeeds; "
        "answering for it would silently mis-convert the adapter. Pin a "
        "transformers that still exports it, or a peft that does not need it."
    )
    if isinstance(donor_value, type):
        # isinstance has to raise, not answer False: peft buckets conversion entries by type, and a
        # placeholder that quietly matches nothing drops the operations instead of reporting that it
        # cannot do the job. Subclassing still works, since creating a class constructs nothing.
        class _RefusingMeta(type):
            def __instancecheck__(cls, instance):
                raise RuntimeError(message)

        def _refuse_init(self, *args, **kwargs):
            raise RuntimeError(message)

        return _RefusingMeta(
            short,
            (object,),
            {
                "__init__": _refuse_init,
                "__doc__": message,
            },
        )

    def _refuse(*args, **kwargs):
        raise RuntimeError(message)

    _refuse.__name__ = _refuse.__qualname__ = short
    _refuse.__doc__ = message
    return _refuse


# The model types peft acts on, snapshotted from conversion_mapping._MODEL_TO_CONVERSION_PATTERN
# because the case handled here is that map being gone. A list and not a rule: deepseek_v3,
# dots1, longcat_flash, minimax, mellum, qwen3_next, solar_open and flex_olmo are all fused MoE
# and none of them say so.
_PEFT_MOE_CONVERSION_PATTERNS = {
    # The two base patterns map to themselves, and omitting them was not harmless: mixtral says
    # nothing about MoE, so the substring hint answered the default and the drift test failed on
    # transformers 5.5.0. qwen3_5_moe is here for 5.3.0, where it is a separate key.
    "mixtral": "mixtral",
    "qwen2_moe": "qwen2_moe",
    "qwen3_5_moe": "qwen2_moe",
    "minimax": "mixtral",
    "minimax_m2": "mixtral",
    "afmoe": "qwen2_moe",
    "cohere2_moe": "qwen2_moe",
    "deepseek_v2": "qwen2_moe",
    "deepseek_v3": "qwen2_moe",
    "deepseek_v32": "qwen2_moe",
    "dots1": "qwen2_moe",
    "ernie4_5_moe": "qwen2_moe",
    "exaone_moe": "qwen2_moe",
    "flex_olmo": "qwen2_moe",
    "glm4_moe": "qwen2_moe",
    "glm4_moe_lite": "qwen2_moe",
    "glm4v_moe": "qwen2_moe",
    "glm_moe_dsa": "qwen2_moe",
    "hunyuan_v1_moe": "qwen2_moe",
    "longcat_flash": "qwen2_moe",
    "mellum": "qwen2_moe",
    "olmoe": "qwen2_moe",
    "qwen3_moe": "qwen2_moe",
    "qwen3_next": "qwen2_moe",
    "qwen3_omni_moe": "qwen2_moe",
    "qwen3_omni_moe_thinker": "qwen2_moe",
    "solar_open": "qwen2_moe",
}

# MoE-named types whose conversion family is NOT one of the two fused ones, so peft's
# _convert_peft_config_moe finds no mapping entry and returns without a rewrite. The substring
# hint below raised for all three on the name alone. Checked before the hint, never instead of
# the snapshot above.
_PEFT_MOE_NAMED_NOT_FUSED = frozenset(
    (
        "granitemoehybrid",
        "granitemoeshared",
        "qwen3_5_moe_text",
    )
)

# The other half of the carve-out: MoE-named types not in the conversion map AT ALL, where
# peft's .get() answers None and skips the rewrite, so a refusal breaks an ordinary adapter
# load (qwen3_vl_moe, lfm2_moe). A name absent from only one of 5.3.0 / 5.5.0 stays fused:
# refusing a fused type costs a message, answering None is a silent mis-conversion.
_PEFT_MOE_NAMED_NOT_CONVERTED = frozenset(
    (
        "ernie4_5_vl_moe",
        "glm4v_moe_text",
        "glm4v_moe_vision",
        "granitemoe",
        "jetmoe",
        "lfm2_moe",
        "phimoe",
        "qwen3_vl_moe",
        "qwen3_vl_moe_text",
    )
)

# How many pairs a candidate map must match before we believe it is the conversion map under a
# new name. Three: a coincidence does not pass, a few renamed types still do.
_CONVERSION_MAP_MATCHES = 3


def _recover_conversion_pattern_map(real):
    """Find the model-type map under whatever name this transformers uses.

    peft copies this dict and looks model families up in it, so an empty one
    is not a harmless placeholder: `_convert_peft_config_moe` misses the
    lookup and leaves legacy LoRA targets unconverted, with no error. The most
    likely reason for the name to disappear is a rename, so go by shape --
    a non-empty module-level `dict[str, str]` -- rather than by name.

    Shape alone is not enough to install one, though. A module that renames the
    map is just as likely to carry some other `dict[str, str]` (an alias table,
    a doc map), and the largest of those is not the conversion map. So a
    candidate also has to agree with the known model-type -> pattern pairs
    above, and the best agreement wins rather than the biggest dict. Nothing
    convincing means nothing recovered: the caller then installs the map that
    raises on a MoE lookup, which is the safe answer, not the wrong one.
    """
    best = None
    best_matches = 0
    for attribute in vars(real).values():
        if not isinstance(attribute, dict) or not attribute:
            continue
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in attribute.items()):
            continue
        matches = sum(1 for k, v in _PEFT_MOE_CONVERSION_PATTERNS.items() if attribute.get(k) == v)
        if matches < _CONVERSION_MAP_MATCHES:
            continue
        if matches > best_matches or (matches == best_matches and len(attribute) > len(best)):
            best, best_matches = attribute, matches
    return best


_MISSING = object()


class _UnavailableConversionPatternMap(dict):
    """A conversion map that answers only for entries someone put in it, and raises otherwise.

    Stands in when transformers has REMOVED the model-type map rather than renamed it, so
    there is nothing to recover. Importing peft's converter still works, which is the point
    of the backfill, but a lookup we cannot answer honestly raises where it happens instead
    of returning None and letting the adapter load mis-converted.

    ``copy`` returns another one of these because peft copies the map at import
    (`_MODEL_TO_CONVERSION_PATTERN = _MODEL_TO_CONVERSION_PATTERN.copy()`) and a plain dict
    copy would be silent again. Writes still work, so peft's own `["mixtral"] = "mixtral"`
    lands and answers normally.
    """

    _MESSAGE = (
        "Unsloth: this transformers exports no model-type conversion map, so peft cannot "
        "convert legacy LoRA targets for fused MoE checkpoints. Re-save the adapter with a "
        "transformers that still ships transformers.conversion_mapping, or load it with a "
        "peft that does not need the conversion."
    )

    # Only fused-MoE lookups are unsafe to answer with a silent None: peft reaches
    # _convert_peft_config_moe for any type with a conversion mapping, and there None is what the
    # real map gives too, so raising for all would break ordinary adapter loads. The substring
    # hints sit on top of the snapshot, for a fused type added later that follows the convention.
    _MOE_HINTS = ("moe", "mixtral")

    def _is_moe(self, key):
        name = str(key).lower()
        if name in _PEFT_MOE_CONVERSION_PATTERNS:
            return True
        if name in _PEFT_MOE_NAMED_NOT_FUSED or name in _PEFT_MOE_NAMED_NOT_CONVERTED:
            return False
        return any(hint in name for hint in self._MOE_HINTS)

    def _answer(self, key, default):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        if self._is_moe(key):
            raise RuntimeError(self._MESSAGE)
        return default

    def copy(self):
        new = type(self)()
        dict.update(new, self)
        return new

    def get(
        self,
        key,
        default = None,
    ):
        return self._answer(key, default)

    def __getitem__(self, key):
        answer = self._answer(key, _MISSING)
        if answer is _MISSING:
            raise KeyError(key)
        return answer


def _backfill_conversion_symbols_once(builders, added):
    """One pass over the modules. Returns True if any was left unimportable.

    Split out so the caller can run it again: a module that could not be
    imported this time may import fine once a module later in the pass has been
    backfilled.
    """
    skipped = False
    for name, symbols in _PEFT_CONVERSION_SYMBOLS.items():
        real = sys.modules.get(name)
        if real is None:
            try:
                real = importlib.import_module(name)
            except Exception:
                skipped = True
                continue
        if getattr(real, _UNSLOTH_STUB_SENTINEL, False):
            continue  # ours already, and complete
        missing = [s for s in symbols if not hasattr(real, s)]
        if not missing:
            continue
        # Build the stub off to the side rather than installing it, so the real module keeps its
        # identity and everything else it exports.
        saved = sys.modules.pop(name, None)
        try:
            donor = builders[name]()
        finally:
            if saved is not None:
                sys.modules[name] = saved
            else:
                sys.modules.pop(name, None)
        for symbol in missing:
            qualified = f"{name}.{symbol}"
            if symbol == "_MODEL_TO_CONVERSION_PATTERN":
                # peft copies this and looks families up in it, so the stub's empty dict silently drops every
                # alias. Recover the real one.
                recovered = _recover_conversion_pattern_map(real)
                if recovered is None:
                    logger.warning(
                        "Unsloth: this transformers exports no model-type "
                        "conversion map, so peft cannot convert legacy LoRA "
                        "targets for fused MoE checkpoints. Adapters for other "
                        "architectures are unaffected."
                    )
                # An empty dict is the one shape that fails SILENTLY: peft does
                # _MODEL_TO_CONVERSION_PATTERN.copy() at import then .get(model_type, None), and a None makes
                # _convert_peft_config_moe return early, so every affected adapter loads with legacy targets
                # unconverted and no message anywhere.
                setattr(
                    real,
                    symbol,
                    dict(recovered) if recovered else _UnavailableConversionPatternMap(),
                )
                added.append(qualified)
                continue
            if qualified in _PEFT_CONVERSION_RUNTIME_SYMBOLS:
                # peft calls this one. The stub bodies exist to make the import work on transformers <5, where
                # peft's converter never runs; on a real transformers it would run and answer wrongly.
                setattr(
                    real,
                    symbol,
                    _unsupported_conversion_symbol(qualified, getattr(donor, symbol, None)),
                )
                added.append(qualified)
                continue
            if hasattr(donor, symbol):
                setattr(real, symbol, getattr(donor, symbol))
                added.append(qualified)
    return skipped


def _backfill_missing_conversion_symbols():
    """Add only the names a real module is missing, taken from our own stub.

    Never replaces a module and never overwrites a name transformers defines,
    so this is a no-op on every release that still exports them.
    """
    builders = {
        "transformers.conversion_mapping": _install_transformers_conversion_mapping_stub,
        "transformers.core_model_loading": _install_transformers_core_model_loading_stub,
    }
    added = []
    # One pass is not enough when the drifts coincide: conversion_mapping imports names from
    # core_model_loading at module top, so while those are missing its import raises and the pass
    # skips it, and _gpu_init calls this guard once. Repeat while a pass adds a symbol and leaves
    # a module unimportable; each pass adds one, so the bound is the module count.
    for _attempt in range(len(_PEFT_CONVERSION_SYMBOLS) + 1):
        before = len(added)
        skipped = _backfill_conversion_symbols_once(builders, added)
        if not skipped or len(added) == before:
            break
    if added:
        logger.info(
            "Unsloth: backfilled %s so peft.utils."
            "transformers_weight_conversion imports on this transformers.",
            ", ".join(added),
        )
    return bool(added)


def patch_peft_weight_converter_compatibility():
    """Allow PEFT converter rebuilds on legacy converter constructors."""
    try:
        from peft.utils import transformers_weight_conversion as twc
    except (ImportError, AttributeError):
        return

    _patch_peft_moe_target_conversion(twc)

    if getattr(twc, "_unsloth_weight_converter_compat_patch", False):
        return

    import threading

    original_build = twc.build_peft_weight_mapping
    patch_lock = threading.RLock()

    def _patch_weight_converter_ctors(weight_conversions, patched):
        seen_classes = set()

        for conversion in weight_conversions:
            conversion_cls = conversion.__class__
            if conversion_cls in seen_classes:
                continue
            seen_classes.add(conversion_cls)

            original_init = conversion_cls.__init__
            params = inspect.signature(original_init).parameters
            supports_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            supports_distributed = "distributed_operation" in params
            supports_quantization = "quantization_operation" in params
            if supports_kwargs or (supports_distributed and supports_quantization):
                continue

            def _compat_init(
                self,
                *args,
                __original_init = original_init,
                __supports_distributed = supports_distributed,
                __supports_quantization = supports_quantization,
                **kwargs,
            ):
                unsupported = {}
                if not __supports_distributed and "distributed_operation" in kwargs:
                    unsupported["distributed_operation"] = kwargs.pop("distributed_operation")
                if not __supports_quantization and "quantization_operation" in kwargs:
                    unsupported["quantization_operation"] = kwargs.pop("quantization_operation")
                result = __original_init(self, *args, **kwargs)
                for name, value in unsupported.items():
                    if hasattr(self, name):
                        setattr(self, name, value)
                return result

            conversion_cls.__init__ = _compat_init
            patched.append((conversion_cls, original_init))

    @functools.wraps(original_build)
    def _build_peft_weight_mapping_compat(
        weight_conversions,
        adapter_name,
        peft_config = None,
    ):
        if not weight_conversions:
            return original_build(weight_conversions, adapter_name, peft_config)

        patched_classes = []
        with patch_lock:
            try:
                _patch_weight_converter_ctors(weight_conversions, patched_classes)
                return original_build(weight_conversions, adapter_name, peft_config)
            finally:
                for conversion_cls, original_init in patched_classes:
                    conversion_cls.__init__ = original_init

    twc.build_peft_weight_mapping = _build_peft_weight_mapping_compat
    twc._unsloth_weight_converter_compat_patch = True


def _patch_peft_moe_target_conversion(twc):
    """Keep PEFT 0.19 MoE conversion from rewriting explicit Unsloth targets."""
    if getattr(twc, "_unsloth_moe_target_conversion_patch", False):
        return

    original_convert_moe = getattr(twc, "_convert_peft_config_moe", None)
    if original_convert_moe is None:
        return

    @functools.wraps(original_convert_moe)
    def _convert_peft_config_moe_unsloth(peft_config, model_type: str) -> None:
        if getattr(peft_config, "target_parameters", None):
            return

        target_modules = getattr(peft_config, "target_modules", None)
        if isinstance(target_modules, str):
            if "." in target_modules:
                return
            return original_convert_moe(peft_config, model_type)

        if not target_modules:
            return original_convert_moe(peft_config, model_type)

        explicit_targets = {
            target for target in target_modules if isinstance(target, str) and "." in target
        }
        if not explicit_targets:
            return original_convert_moe(peft_config, model_type)

        bare_targets = set(target_modules) - explicit_targets
        if not bare_targets:
            return

        peft_config.target_modules = bare_targets
        original_convert_moe(peft_config, model_type)
        peft_config.target_modules = set(peft_config.target_modules or ()) | explicit_targets

    twc._convert_peft_config_moe = _convert_peft_config_moe_unsloth
    twc._unsloth_moe_target_conversion_patch = True


CAUSAL_CONV1D_BROKEN = False
_CAUSAL_CONV1D_PREFIX = "causal_conv1d"
_CAUSAL_CONV1D_BLOCKER_SENTINEL = "_unsloth_causal_conv1d_blocker"
VLLM_BROKEN = False
_VLLM_PREFIX = "vllm"
_VLLM_BLOCKER_SENTINEL = "_unsloth_vllm_blocker"
_ROCM_ENV_HINT_KEYS = (
    "ROCM_PATH",
    "ROCM_HOME",
    "HIP_PATH",
    "HSA_PATH",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
)
_ROCM_PATH_HINTS = (
    Path("/opt/rocm"),
    Path("/dev/kfd"),
    Path("/sys/module/amdgpu"),
)
_AMDGPU_ASIC_ID_TABLE_PATH_ENV = "AMDGPU_ASIC_ID_TABLE_PATH"
_AMDGPU_ASIC_ID_CANDIDATE_PATHS = (
    Path("/usr/share/libdrm/amdgpu.ids"),
    Path("/usr/local/share/libdrm/amdgpu.ids"),
    Path("/opt/rocm/share/libdrm/amdgpu.ids"),
    Path("/opt/amdgpu/share/libdrm/amdgpu.ids"),
)


def _log_rocm_detection(message):
    if UNSLOTH_ENABLE_LOGGING:
        logger.info(message)


@functools.lru_cache(1)
def _is_rocm_torch_build() -> bool:
    # Most official ROCm wheels carry a +rocmX.Y local version, but some custom or source builds do
    # not, so fall back to runtime hints.
    try:
        torch_version_raw = str(importlib_version("torch")).lower()
        if "rocm" in torch_version_raw:
            _log_rocm_detection("Unsloth: ROCm detection matched torch version tag (+rocm).")
            return True
    except Exception:
        pass

    # Environment hints commonly present on ROCm runtimes.
    for key in _ROCM_ENV_HINT_KEYS:
        value = os.environ.get(key, "")
        if isinstance(value, str) and value.strip():
            _log_rocm_detection(f"Unsloth: ROCm detection matched environment key `{key}`.")
            return True

    # Filesystem / driver hints for ROCm stacks.
    for path in _ROCM_PATH_HINTS:
        try:
            if path.exists():
                _log_rocm_detection(f"Unsloth: ROCm detection matched filesystem hint `{path}`.")
                return True
        except Exception:
            continue

    _log_rocm_detection("Unsloth: ROCm detection did not match any known hints.")
    return False


def _iter_amdgpu_asic_id_table_candidates():
    try:
        torch_spec = importlib.util.find_spec("torch")
    except Exception:
        torch_spec = None

    roots = []
    if torch_spec is not None:
        if torch_spec.origin:
            roots.append(Path(torch_spec.origin).resolve().parent)
        if torch_spec.submodule_search_locations:
            for location in torch_spec.submodule_search_locations:
                roots.append(Path(location).resolve())

    seen = set()
    for root in roots:
        for candidate in (
            root / "share" / "libdrm" / "amdgpu.ids",
            root.parent / "share" / "libdrm" / "amdgpu.ids",
            root.parent.parent / "share" / "libdrm" / "amdgpu.ids",
        ):
            candidate_str = str(candidate)
            if candidate_str in seen:
                continue
            seen.add(candidate_str)
            yield candidate

    for candidate in _AMDGPU_ASIC_ID_CANDIDATE_PATHS:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        yield candidate


def configure_amdgpu_asic_id_table_path():
    # Honor an existing valid user-provided path.
    configured = os.environ.get(_AMDGPU_ASIC_ID_TABLE_PATH_ENV, "").strip()
    if configured:
        configured_path = Path(configured)
        try:
            if configured_path.is_file():
                return str(configured_path)
        except Exception:
            pass

    # Only attempt this on ROCm-like environments.
    if not _is_rocm_torch_build():
        return None

    for candidate in _iter_amdgpu_asic_id_table_candidates():
        try:
            if candidate.is_file():
                os.environ[_AMDGPU_ASIC_ID_TABLE_PATH_ENV] = str(candidate)
                if UNSLOTH_ENABLE_LOGGING:
                    logger.info(f"Unsloth: Set {_AMDGPU_ASIC_ID_TABLE_PATH_ENV}={candidate}")
                return str(candidate)
        except Exception:
            continue

    return None


# bitsandbytes Windows ROCm fix: cextension.py calls get_rocm_gpu_arch() (bnb >= 0.47) and
# get_rocm_warpsize() (0.49.x) at import, shelling out to rocminfo / hipInfo.exe via PATH. Neither
# is on PATH on Windows (AMD torch wheels put hipInfo.exe in venv Scripts), so ROCM_GPU_ARCH
# becomes "unknown" and warp size defaults to 64, wrong on RDNA (wave 32) and breaking 4-bit
# blocksizes and ALLOW_PREQUANTIZED_MODELS. Upstream fix unmerged (bitsandbytes#1969), so a
# MetaPathFinder swaps both helpers right after bitsandbytes.cuda_specs executes. Must run before
# `import unsloth_zoo`.

# ---------------------------------------------------------------------------

_BNB_CUDA_SPECS_MODULE = "bitsandbytes.cuda_specs"
_BNB_ROCM_FIX_FINDER_SENTINEL = "_unsloth_bnb_rocm_fix_finder"
_BNB_ROCM_FIX_FUNCTION_FLAG = "__unsloth_bnb_rocm_fix__"


def _torch_rocm_device_props():
    """Device-0 props on a ROCm torch build with a visible GPU, else None.
    Never raises; bnb's own import initializes the device context anyway."""
    try:
        import torch

        if not getattr(getattr(torch, "version", None), "hip", None):
            return None
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0)
    except Exception:
        return None


def _iter_hipinfo_paths():
    """Yield existing hipInfo.exe paths: PATH, interpreter scripts dir (venv
    and conda layouts), then HIP SDK / AMD installer locations."""
    import shutil
    import sysconfig

    candidates = []
    try:
        resolved = shutil.which("hipinfo.exe")
        if resolved:
            candidates.append(resolved)
    except Exception:
        pass
    try:
        scripts_dir = sysconfig.get_path("scripts")
        if scripts_dir:
            candidates.append(os.path.join(scripts_dir, "hipInfo.exe"))
    except Exception:
        pass
    executable_dir = os.path.dirname(sys.executable or "")
    if executable_dir:
        candidates.append(os.path.join(executable_dir, "hipInfo.exe"))
        candidates.append(os.path.join(executable_dir, "Scripts", "hipInfo.exe"))
    for env_key in ("HIP_PATH", "ROCM_PATH"):
        root = os.environ.get(env_key, "").strip()
        if root:
            candidates.append(os.path.join(root, "bin", "hipInfo.exe"))
    rocm_root = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "AMD", "ROCm")
    try:
        if os.path.isdir(rocm_root):
            for version_dir in sorted(os.listdir(rocm_root), reverse = True):
                candidates.append(os.path.join(rocm_root, version_dir, "bin", "hipInfo.exe"))
    except Exception:
        pass

    seen = set()
    for candidate in candidates:
        try:
            key = os.path.normcase(os.path.normpath(candidate))
            if key in seen:
                continue
            seen.add(key)
            if os.path.isfile(candidate):
                yield candidate
        except Exception:
            continue


def _run_hipinfo(hipinfo_path):
    """Run hipInfo.exe and return its stdout, or "" on any failure."""
    import subprocess
    try:
        result = subprocess.run(
            [hipinfo_path],
            capture_output = True,
            text = True,
            timeout = 15,
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.stdout or ""
    except Exception as e:
        _log_rocm_detection(f"Unsloth: `{hipinfo_path}` failed: {e}")
        return ""


def _unsloth_get_rocm_gpu_arch():
    """Replaces bnb's get_rocm_gpu_arch: torch device props first (no
    subprocess), then hipInfo.exe by absolute path, then a quiet "unknown"."""
    try:
        import torch
        if not getattr(getattr(torch, "version", None), "hip", None):
            return "unknown"
    except Exception:
        return "unknown"
    props = _torch_rocm_device_props()
    if props is not None:
        try:
            # gcnArchName may carry feature flags, e.g. "gfx90a:sramecc+:xnack-".
            arch = str(props.gcnArchName).split(":")[0].strip()
            if arch.startswith("gfx"):
                return arch
        except Exception:
            pass
    for hipinfo_path in _iter_hipinfo_paths():
        match = re.search(r"gcnArchName:\s+gfx([a-zA-Z\d]+)", _run_hipinfo(hipinfo_path))
        if match:
            return "gfx" + match.group(1)
    _log_rocm_detection(
        "Unsloth: Could not detect the ROCm GPU architecture - bitsandbytes will see `unknown`."
    )
    return "unknown"


def _unsloth_get_rocm_warpsize():
    """Replaces bnb 0.49.x get_rocm_warpsize: upstream defaults to 64 when
    rocminfo is missing, wrong on RDNA (wave 32)."""
    try:
        import torch
        if not getattr(getattr(torch, "version", None), "hip", None):
            return 32  # upstream behavior: NVIDIA warp size is always 32
    except Exception:
        return 64  # upstream behavior: default to 64 on failure
    props = _torch_rocm_device_props()
    if props is not None:
        # torch 2.11 ROCm exposes warp_size; some builds used warpSize.
        for attribute_name in ("warp_size", "warpSize"):
            warp_size = getattr(props, attribute_name, None)
            if isinstance(warp_size, int) and warp_size in (32, 64):
                return warp_size
    for hipinfo_path in _iter_hipinfo_paths():
        match = re.search(r"^\s*warpSize:\s+(\d+)", _run_hipinfo(hipinfo_path), re.MULTILINE)
        if match and int(match.group(1)) in (32, 64):
            return int(match.group(1))
    _log_rocm_detection(
        "Unsloth: Could not detect the ROCm warp size - defaulting to 64 "
        "(bitsandbytes' own default)."
    )
    return 64


setattr(_unsloth_get_rocm_gpu_arch, _BNB_ROCM_FIX_FUNCTION_FLAG, True)
setattr(_unsloth_get_rocm_warpsize, _BNB_ROCM_FIX_FUNCTION_FLAG, True)


def _bnb_rocm_helper_is_broken(function):
    """True only for upstream's subprocess-only detectors; co_names works
    where getsource fails. Versions consulting torch props are untouched."""
    if function is None or not callable(function):
        return False
    if getattr(function, _BNB_ROCM_FIX_FUNCTION_FLAG, False):
        return False  # Already ours.
    try:
        function = inspect.unwrap(function)
    except Exception:
        pass
    code = getattr(function, "__code__", None)
    co_names = getattr(code, "co_names", ()) if code is not None else ()
    if not co_names:
        return False  # C function or opaque wrapper -- do not touch.
    if "get_device_properties" in co_names or "gcnArchName" in co_names:
        return False  # Fixed upstream -- no-op.
    return "subprocess" in co_names


def _patch_bnb_cuda_specs_module(module):
    """Swap broken ROCm detection helpers on an executed cuda_specs module.
    Returns True when the module ends up patched (now or previously)."""
    patched = False
    for attribute_name, replacement in (
        ("get_rocm_gpu_arch", _unsloth_get_rocm_gpu_arch),
        ("get_rocm_warpsize", _unsloth_get_rocm_warpsize),
    ):
        original = getattr(module, attribute_name, None)
        if getattr(original, _BNB_ROCM_FIX_FUNCTION_FLAG, False):
            patched = True  # Already ours.
            continue
        if not _bnb_rocm_helper_is_broken(original):
            continue
        setattr(module, attribute_name, replacement)
        patched = True
        logger.info(
            f"Unsloth: Patched bitsandbytes.cuda_specs.{attribute_name} - "
            f"avoids PATH-dependent subprocess GPU detection on Windows ROCm."
        )
    return patched


class _BnbCudaSpecsPatchLoader(importlib.abc.Loader):
    __slots__ = ("_loader",)

    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        create_module = getattr(self._loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        # Patch after the module body ran, before cextension calls it. The finder stays on
        # sys.meta_path so importlib.reload(bitsandbytes.cuda_specs) re-patches.
        try:
            _patch_bnb_cuda_specs_module(module)
        except Exception as e:
            _log_rocm_detection(f"Unsloth: bitsandbytes ROCm detection patch failed: {e}")

    def __getattr__(self, name):
        # Delegate get_source / get_filename etc. so introspection works.
        return getattr(self._loader, name)


class _BnbCudaSpecsPatchFinder(importlib.abc.MetaPathFinder):
    __slots__ = (_BNB_ROCM_FIX_FINDER_SENTINEL,)

    def __init__(self):
        setattr(self, _BNB_ROCM_FIX_FINDER_SENTINEL, True)

    def find_spec(
        self,
        fullname,
        path = None,
        target = None,
    ):
        if fullname != _BNB_CUDA_SPECS_MODULE:
            return None
        # Delegate to the remaining finders (editable installs, frozen apps) and wrap the loader that
        # would actually be used.
        spec = None
        for finder in sys.meta_path:
            if finder is self or getattr(finder, _BNB_ROCM_FIX_FINDER_SENTINEL, False):
                continue
            finder_find_spec = getattr(finder, "find_spec", None)
            if finder_find_spec is None:
                continue
            try:
                spec = finder_find_spec(fullname, path, target)
            except Exception:
                spec = None
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return None
        if not hasattr(spec.loader, "exec_module"):
            return None  # Legacy loader -- let the stock machinery handle it.
        spec.loader = _BnbCudaSpecsPatchLoader(spec.loader)
        return spec


def _repair_imported_bitsandbytes_rocm_constants():
    """bnb imported before unsloth: noise already fired, but fix detectors
    and cached constants, incl. by-value ROCM_WARP_SIZE_64 copies."""
    cuda_specs = sys.modules.get(_BNB_CUDA_SPECS_MODULE)
    if cuda_specs is None:
        return
    if not _patch_bnb_cuda_specs_module(cuda_specs):
        return

    try:
        arch = cuda_specs.get_rocm_gpu_arch()
    except Exception:
        arch = "unknown"
    warp_size_64 = None
    get_rocm_warpsize = getattr(cuda_specs, "get_rocm_warpsize", None)
    if callable(get_rocm_warpsize):
        try:
            warp_size_64 = get_rocm_warpsize() == 64
        except Exception:
            warp_size_64 = None

    for module_name, module in list(sys.modules.items()):
        if module is None or module is cuda_specs:
            continue
        if module_name != "bitsandbytes" and not module_name.startswith("bitsandbytes."):
            continue
        try:
            if arch != "unknown" and getattr(module, "ROCM_GPU_ARCH", None) == "unknown":
                module.ROCM_GPU_ARCH = arch
            if warp_size_64 is not None and isinstance(
                getattr(module, "ROCM_WARP_SIZE_64", None), bool
            ):
                module.ROCM_WARP_SIZE_64 = warp_size_64
        except Exception:
            continue
    logger.info("Unsloth: Repaired bitsandbytes ROCm arch / warp-size constants in place.")


def fix_bitsandbytes_rocm_arch_detection():
    """Fix bnb's import-time ROCm arch / warp-size detection on Windows
    (see header above). No-op on non-Windows, non-ROCm, missing or
    upstream-fixed bnb. Idempotent. Opt out: UNSLOTH_DISABLE_BNB_ROCM_FIX=1."""
    if os.environ.get("UNSLOTH_DISABLE_BNB_ROCM_FIX", "0") == "1":
        return
    if sys.platform != "win32":
        return
    if not _is_rocm_torch_build():
        return

    # Already imported: prevention impossible, repair in place instead.
    if _BNB_CUDA_SPECS_MODULE in sys.modules:
        try:
            _repair_imported_bitsandbytes_rocm_constants()
        except Exception:
            pass
        return

    try:
        if importlib.util.find_spec("bitsandbytes") is None:
            return
    except Exception:
        return

    for finder in sys.meta_path:
        if getattr(finder, _BNB_ROCM_FIX_FINDER_SENTINEL, False):
            return  # Already installed -- idempotent.
    sys.meta_path.insert(0, _BnbCudaSpecsPatchFinder())
    _log_rocm_detection("Unsloth: Installed the bitsandbytes ROCm arch detection patch hook.")


def _is_causal_conv1d_name(module_name: str) -> bool:
    return module_name == _CAUSAL_CONV1D_PREFIX or module_name.startswith(
        _CAUSAL_CONV1D_PREFIX + "."
    )


def _is_vllm_name(module_name: str) -> bool:
    return module_name == _VLLM_PREFIX or module_name.startswith(_VLLM_PREFIX + ".")


def _resolve_module_name(module_name, package):
    if not isinstance(module_name, str):
        return module_name
    if module_name.startswith("."):
        try:
            return importlib.util.resolve_name(module_name, package)
        except Exception:
            return module_name
    return module_name


def _is_broken_causal_conv1d_error(error) -> bool:
    checked = set()
    current = error
    while current is not None and id(current) not in checked:
        checked.add(id(current))
        message = str(current).lower()
        if (
            ("causal_conv1d_cuda" in message and "undefined symbol" in message)
            or ("_zn3c103hip28c10_hip_check_implementation" in message)
            or ("causal_conv1d" in message and "undefined symbol" in message)
        ):
            return True
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return False


def _is_broken_vllm_error(error) -> bool:
    checked = set()
    current = error
    while current is not None and id(current) not in checked:
        checked.add(id(current))
        message = str(current).lower()
        if (
            ("vllm/_c" in message or "vllm._c" in message)
            and (
                "undefined symbol" in message
                or "cannot open shared object file" in message
                or ".so:" in message
            )
        ) or ("vllm" in message and "undefined symbol" in message):
            return True
        # A forced extension load raises the bare loader error with no "vllm._C" wrapper, so match any
        # .so failure; callers feed only vLLM imports.
        if "cannot open shared object file" in message:
            return True
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return False


_VLLM_RELEASES_URL = "https://github.com/vllm-project/vllm/releases"
_VLLM_INSTALL_DOCS_URL = "https://docs.vllm.ai/en/latest/getting_started/installation/gpu/"

# A plain release version, e.g. "0.23.0". Anything else (rc / dev / post) has no matching
# GitHub release asset, so never name a wheel for it.
_VLLM_RELEASE_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")

# Normalises platform.machine() onto the arch spelling used in wheel names.
_VLLM_WHEEL_ARCHES = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "x64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
}


def _both_arches(manylinux_tag):
    return {"x86_64": manylinux_tag, "aarch64": manylinux_tag}


# vLLM ships one CUDA build per release unsuffixed and the other under a +cuXXX tag, and which
# major is default flipped at 0.20.0 while the manylinux tag moved and differs between the two
# wheels of one release. The name cannot be derived: vLLM's documented "+cu${CUDA_VERSION}"
# pattern 404s (vllm-project/vllm#37847) and no release published a +cu128 wheel.
# Each entry is (min_version, max_version, {cuda_major: (local_tag, {arch: manylinux})}),
# transcribed from the real release assets; an absent major or arch was never published for
# that range. Extend on a new release; until then newer versions fall back to the release page.
_VLLM_WHEEL_ASSETS = (
    (
        "0.11.0",
        "0.11.0",
        {12: ("", {"x86_64": "manylinux1", "aarch64": "manylinux2014"})},
    ),
    (
        "0.11.1",
        "0.11.2",
        {
            12: ("", {"x86_64": "manylinux1", "aarch64": "manylinux2014"}),
            13: ("cu130", {"x86_64": "manylinux1"}),
        },
    ),
    (
        "0.12.0",
        "0.12.0",
        {
            12: ("", _both_arches("manylinux_2_31")),
            13: ("cu130", {"x86_64": "manylinux_2_31"}),
        },
    ),
    (
        "0.13.0",
        "0.19.1",
        {
            12: ("", _both_arches("manylinux_2_31")),
            13: ("cu130", _both_arches("manylinux_2_35")),
        },
    ),
    (
        "0.20.0",
        "0.20.2",
        {
            12: ("cu129", _both_arches("manylinux_2_31")),
            13: ("", _both_arches("manylinux_2_35")),
        },
    ),
    (
        "0.21.0",
        "0.21.0",
        {
            12: ("cu129", _both_arches("manylinux_2_34")),
            13: ("", _both_arches("manylinux_2_24")),
        },
    ),
    (
        "0.22.0",
        "0.26.0",
        {
            12: ("cu129", _both_arches("manylinux_2_28")),
            13: ("", _both_arches("manylinux_2_28")),
        },
    ),
)

# From this release on, the default wheel is the CUDA 13 build and the CUDA 12 one carries
# +cu129, so the right variant can be named even when the manylinux tag is unknown.
_VLLM_CUDA13_DEFAULT_SINCE = "0.20.0"


def _get_vllm_wheel_url(vllm_version, cuda_major, cpu_arch):
    """URL of the published vLLM wheel for this release/CUDA/arch, else None."""
    if not _VLLM_RELEASE_VERSION_RE.match(vllm_version or ""):
        return None
    arch = _VLLM_WHEEL_ARCHES.get(str(cpu_arch).lower())
    if arch is None:
        return None
    try:
        wanted = TrueVersion(vllm_version)
    except Exception:
        return None
    for low, high, by_cuda in _VLLM_WHEEL_ASSETS:
        if not TrueVersion(low) <= wanted <= TrueVersion(high):
            continue
        local_tag, manylinux_by_arch = by_cuda.get(cuda_major, (None, {}))
        manylinux = manylinux_by_arch.get(arch)
        if manylinux is None:
            return None
        local = f"+{local_tag}" if local_tag else ""
        return (
            f"{_VLLM_RELEASES_URL}/download/v{vllm_version}/"
            f"vllm-{vllm_version}{local}-cp38-abi3-{manylinux}_{arch}.whl"
        )
    return None


def _get_vllm_wheel_variant_hint(vllm_version, cuda_major):
    """Which wheel of a release to pick, when we cannot name the exact file."""
    if not _VLLM_RELEASE_VERSION_RE.match(vllm_version or ""):
        return None
    try:
        if TrueVersion(vllm_version) < TrueVersion(_VLLM_CUDA13_DEFAULT_SINCE):
            return None
    except Exception:
        return None
    if cuda_major >= 13:
        return "the default wheel (the one with no `+cuXXX` suffix)"
    if cuda_major == 12:
        return "the `+cu129` wheel"
    return None


def _get_vllm_cuda_mismatch_message(error):
    """If the error is a CUDA version mismatch, return a helpful install message."""
    checked = set()
    current = error
    wanted_cuda = None
    while current is not None and id(current) not in checked:
        checked.add(id(current))
        message = str(current)
        # Extract the CUDA version vllm was built for, e.g. "libcudart.so.12"
        match = re.search(r"libcudart\.so\.(\d+)", message)
        if match:
            wanted_cuda = int(match.group(1))
            break
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    if wanted_cuda is None:
        return None

    # Detect what CUDA version is actually available on the system
    system_cuda_display = None  # Human-readable, e.g. "13.0"
    system_cuda_major = None
    # A random name is never reused, so a failed attempt would be litter rather than something the next run
    # overwrites.
    # torchao also hangs a handler off aten._grouped_mm at import time, and that operator only exists from torch
    # 2.8. Same skew, different lookup: supplying the torch.nn.functional names above does not help it.
    # Second guard for the same hazard, in case the alias is the file rather than the directory: never chain to
    # this very hook.
    # Ask the import system rather than probing for a filename, so the package (`sitecustomize/__init__.py`) and
    # .pyc forms chain too.
    # Debug mode calls find_device(...).type on gather/broadcast inputs
    try:
        import torch
        cuda_version = torch.version.cuda
        if cuda_version:
            system_cuda_display = cuda_version
            system_cuda_major = int(str(cuda_version).split(".")[0])
    except Exception:
        pass

    if system_cuda_major is None or system_cuda_major == wanted_cuda:
        return None  # Not a mismatch or can't determine

    try:
        vllm_version = importlib_version("vllm").split("+")[0]
    except Exception:
        vllm_version = None

    system = ""
    cpu_arch = "x86_64"
    try:
        import platform
        system = platform.system()
        cpu_arch = platform.machine()
    except Exception:
        pass

    header = (
        f"Unsloth: vLLM was built for CUDA {wanted_cuda} but this system has "
        f"CUDA {system_cuda_display}. "
    )

    # vLLM only publishes CUDA wheels for Linux (Windows users go through WSL).
    if system and system != "Linux":
        return (
            f"{header}Please reinstall a vLLM build for CUDA {system_cuda_major}; "
            f"vLLM publishes CUDA wheels for Linux only, so see\n\n  "
            f"{_VLLM_INSTALL_DOCS_URL}"
        )

    wheel_url = _get_vllm_wheel_url(vllm_version, system_cuda_major, cpu_arch)
    if wheel_url is not None:
        return (
            f"{header}Please reinstall vLLM with the correct CUDA version:\n\n  "
            f"uv pip install {wheel_url}"
        )

    # Unknown / unmapped release: never invent a filename, point at the real assets.
    hint = _get_vllm_wheel_variant_hint(vllm_version, system_cuda_major)
    hint = f"Download {hint} for your platform" if hint else "Pick the wheel for your platform"
    release_page = (
        f"{_VLLM_RELEASES_URL}/tag/v{vllm_version}"
        if _VLLM_RELEASE_VERSION_RE.match(vllm_version or "")
        else _VLLM_RELEASES_URL
    )
    return (
        f"{header}Please reinstall a vLLM build for CUDA {system_cuda_major}. "
        f"{hint} from\n\n  {release_page}"
    )


class _CausalConv1dImportBlockerLoader(importlib.abc.Loader):
    __slots__ = ("module_name",)

    def __init__(self, module_name):
        self.module_name = module_name

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise ModuleNotFoundError(f"No module named '{self.module_name}'")


class _CausalConv1dImportBlockerFinder(importlib.abc.MetaPathFinder):
    __slots__ = (_CAUSAL_CONV1D_BLOCKER_SENTINEL,)

    def __init__(self):
        setattr(self, _CAUSAL_CONV1D_BLOCKER_SENTINEL, True)

    def find_spec(
        self,
        fullname,
        path = None,
        target = None,
    ):
        if not CAUSAL_CONV1D_BROKEN or not _is_causal_conv1d_name(fullname):
            return None
        return importlib.machinery.ModuleSpec(
            name = fullname,
            loader = _CausalConv1dImportBlockerLoader(fullname),
            is_package = fullname == _CAUSAL_CONV1D_PREFIX,
        )


class _VllmImportBlockerLoader(importlib.abc.Loader):
    __slots__ = ("module_name",)

    def __init__(self, module_name):
        self.module_name = module_name

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise ModuleNotFoundError(f"No module named '{self.module_name}'")


class _VllmImportBlockerFinder(importlib.abc.MetaPathFinder):
    __slots__ = (_VLLM_BLOCKER_SENTINEL,)

    def __init__(self):
        setattr(self, _VLLM_BLOCKER_SENTINEL, True)

    def find_spec(
        self,
        fullname,
        path = None,
        target = None,
    ):
        if not VLLM_BROKEN or not _is_vllm_name(fullname):
            return None
        return importlib.machinery.ModuleSpec(
            name = fullname,
            loader = _VllmImportBlockerLoader(fullname),
            is_package = fullname == _VLLM_PREFIX,
        )


def _patch_find_spec_for_causal_conv1d():
    current_find_spec = importlib.util.find_spec
    if getattr(current_find_spec, "_unsloth_causal_conv1d_find_spec_patch", False):
        return

    def _blocked_find_spec(name, package = None):
        resolved_name = _resolve_module_name(name, package)
        if CAUSAL_CONV1D_BROKEN and isinstance(resolved_name, str):
            if _is_causal_conv1d_name(resolved_name):
                return None
        return current_find_spec(name, package)

    _blocked_find_spec._unsloth_causal_conv1d_find_spec_patch = True
    _blocked_find_spec._unsloth_original_find_spec = current_find_spec
    importlib.util.find_spec = _blocked_find_spec


def _patch_find_spec_for_vllm():
    current_find_spec = importlib.util.find_spec
    if getattr(current_find_spec, "_unsloth_vllm_find_spec_patch", False):
        return

    def _blocked_find_spec(name, package = None):
        resolved_name = _resolve_module_name(name, package)
        if VLLM_BROKEN and isinstance(resolved_name, str):
            if _is_vllm_name(resolved_name):
                return None
        return current_find_spec(name, package)

    _blocked_find_spec._unsloth_vllm_find_spec_patch = True
    _blocked_find_spec._unsloth_original_find_spec = current_find_spec
    importlib.util.find_spec = _blocked_find_spec


def _install_causal_conv1d_blocker():
    _patch_find_spec_for_causal_conv1d()
    for finder in sys.meta_path:
        if getattr(finder, _CAUSAL_CONV1D_BLOCKER_SENTINEL, False):
            return
    sys.meta_path.insert(0, _CausalConv1dImportBlockerFinder())


def _install_vllm_blocker():
    _patch_find_spec_for_vllm()
    for finder in sys.meta_path:
        if getattr(finder, _VLLM_BLOCKER_SENTINEL, False):
            return
    sys.meta_path.insert(0, _VllmImportBlockerFinder())


def _clear_causal_conv1d_modules():
    for module_name in list(sys.modules):
        if _is_causal_conv1d_name(module_name):
            sys.modules.pop(module_name, None)


def _clear_vllm_modules():
    for module_name in list(sys.modules):
        if _is_vllm_name(module_name):
            sys.modules.pop(module_name, None)


# vLLM's compiled extensions: a CUDA-major ABI break hits all of them, so probing the eagerly
# loaded _C and its siblings reliably trips it.
_VLLM_COMPILED_EXTENSIONS = (
    "vllm._C",
    "vllm._C_stable_libtorch",
    "vllm._moe_C",
    "vllm._rocm_C",
)


def disable_broken_vllm(error = None):
    """Disable vLLM dynamically when its shared library is ABI-broken."""
    global VLLM_BROKEN
    if VLLM_BROKEN:
        _install_vllm_blocker()
        return True

    failure = error
    if failure is None:
        try:
            if importlib.util.find_spec("vllm") is None:
                return False
        except Exception:
            return False

        try:
            import vllm  # noqa: F401

            # Lazy vLLM lets a bare `import vllm` succeed with an ABI-broken extension; force-load each to
            # surface the .so failure here. A missing one raises ModuleNotFoundError (skipped below).
            for _ext in _VLLM_COMPILED_EXTENSIONS:
                try:
                    importlib.import_module(_ext)
                except ModuleNotFoundError:
                    pass
            return False
        except Exception as import_error:
            failure = import_error

    if not _is_broken_vllm_error(failure):
        return False

    VLLM_BROKEN = True
    _clear_vllm_modules()
    _install_vllm_blocker()
    cuda_msg = _get_vllm_cuda_mismatch_message(failure)
    if cuda_msg:
        logger.warning(cuda_msg)
    else:
        logger.warning(
            "Unsloth: Detected broken vLLM binary extension; "
            "disabling vLLM imports and continuing import.\n"
            "Please reinstall via `uv pip install unsloth vllm torchvision torchaudio "
            "--torch-backend=auto`."
        )
    return True


def _disable_transformers_causal_conv1d():
    try:
        import transformers.utils.import_utils as tf_import_utils
    except Exception:
        return

    if hasattr(tf_import_utils, "is_causal_conv1d_available"):
        tf_import_utils.is_causal_conv1d_available = lambda: False

    for attr_name in (
        "_causal_conv1d_available",
        "_is_causal_conv1d_available",
    ):
        if hasattr(tf_import_utils, attr_name):
            setattr(tf_import_utils, attr_name, False)


def disable_broken_causal_conv1d():
    """Disable causal_conv1d dynamically when its shared library is ABI-broken.

    This mirrors Unsloth's FlashAttention fallback behavior: if importing causal_conv1d
    fails with a known binary symbol error, we disable it at startup so model imports do
    not hard-fail.
    """
    global CAUSAL_CONV1D_BROKEN
    if CAUSAL_CONV1D_BROKEN:
        _install_causal_conv1d_blocker()
        _disable_transformers_causal_conv1d()
        return

    try:
        if importlib.util.find_spec("causal_conv1d") is None:
            return
    except Exception:
        return

    try:
        import causal_conv1d  # noqa: F401
        return
    except Exception as error:
        if not _is_broken_causal_conv1d_error(error):
            return

    CAUSAL_CONV1D_BROKEN = True
    _clear_causal_conv1d_modules()
    _install_causal_conv1d_blocker()
    _disable_transformers_causal_conv1d()
    print(
        "Unsloth: Detected broken causal_conv1d binary; "
        "disabling causal_conv1d fast path and continuing import."
    )


_BNB_ROCM_DLL_RE = re.compile(r"libbitsandbytes_rocm(\d+)\.dll", re.IGNORECASE)


def _is_hip_torch_build():
    """True only when torch itself is a HIP/ROCm build. Env hints (HIP_PATH
    etc.) do not count: CUDA bitsandbytes raises at import when the ROCm
    override is set. Wheel tag first (no torch import); torch.version.hip
    fallback for source builds."""
    try:
        if "rocm" in str(importlib_version("torch")).lower():
            return True
    except Exception:
        pass
    try:
        import torch
        return bool(getattr(torch.version, "hip", None))
    except Exception:
        return False


def _detect_installed_bnb_rocm_version():
    """Highest installed ``libbitsandbytes_rocm<NN>.dll`` suffix ("72", "713")
    or ``None``. Listing order is unordered, so take the numeric max."""
    try:
        spec = importlib.util.find_spec("bitsandbytes")
    except Exception:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None

    suffixes = []
    for pkg_dir in spec.submodule_search_locations:
        try:
            entries = os.listdir(pkg_dir)
        except Exception:
            continue
        for entry in entries:
            match = _BNB_ROCM_DLL_RE.fullmatch(entry)
            if match is not None:
                suffixes.append(match.group(1))
    if not suffixes:
        return None
    return max(suffixes, key = lambda value: int(value))


def maybe_set_windows_rocm_bnb_version():
    """Pin ``BNB_ROCM_VERSION`` from the installed wheel on Windows + ROCm torch.

    AMD's Windows wheel ships one ``libbitsandbytes_rocm<NN>.dll`` whose
    suffix can disagree with ``torch.version.hip`` (HIP 7.13 vs rocm72.dll),
    breaking the native 4-bit/8-bit paths. Pin the installed suffix before
    bitsandbytes is first imported.

    No-op unless ALL of: Windows, a real HIP torch build (env hints like
    HIP_PATH do not count), a ROCm DLL installed, and no explicit user value.
    Linux is untouched. Values seeded by Unsloth's venv sitecustomize.py
    (marked ``UNSLOTH_BNB_ROCM_VERSION_SOURCE=sitecustomize``) are
    redetectable defaults, not overrides; ``UNSLOTH_SKIP_BNB_ROCM_VERSION=1``
    opts out and drops a seeded default. Returns the value set, else None.
    """
    if sys.platform != "win32":
        return None
    if os.environ.get("UNSLOTH_SKIP_BNB_ROCM_VERSION") == "1":
        # Real opt-out: drop our seeded default (marker present); explicit user values carry no marker and are kept.
        if os.environ.get("UNSLOTH_BNB_ROCM_VERSION_SOURCE") == "sitecustomize":
            os.environ.pop("BNB_ROCM_VERSION", None)
            os.environ.pop("UNSLOTH_BNB_ROCM_VERSION_SOURCE", None)
        return None
    if "BNB_ROCM_VERSION" in os.environ and (
        os.environ.get("UNSLOTH_BNB_ROCM_VERSION_SOURCE") != "sitecustomize"
    ):
        return None
    if not _is_hip_torch_build():
        return None
    version = _detect_installed_bnb_rocm_version()
    if version is None:
        return None
    os.environ["BNB_ROCM_VERSION"] = version
    os.environ["UNSLOTH_BNB_ROCM_VERSION_SOURCE"] = "detected"
    if UNSLOTH_ENABLE_LOGGING:
        logger.info(
            f"Unsloth: set BNB_ROCM_VERSION={version} "
            "(detected from the installed bitsandbytes ROCm wheel on Windows)."
        )
    return version


def patch_accelerate_recursively_apply():
    """
    Make Accelerate's recursive utilities tolerate Unsloth's EmptyLogits
    sentinel. recursively_apply returns the sentinel unchanged instead of
    raising TypeError, and find_device skips it while still finding real
    tensors, falling back to PartialState().device only for sentinel-only
    payloads. Both wrappers are idempotent and are propagated to every
    already imported accelerate namespace.
    """
    try:
        import accelerate.utils.operations as acc_ops
    except Exception:
        return

    original_recursively_apply = getattr(acc_ops, "recursively_apply", None)
    if original_recursively_apply is not None and not getattr(
        original_recursively_apply, "__unsloth_patched__", False
    ):

        @functools.wraps(original_recursively_apply)
        def _patched_recursively_apply(func, data, *args, **kwargs):
            if type(data).__name__ == "EmptyLogits":
                cls = type(data)
                if cls.__eq__ is object.__eq__:
                    # Debug mode compares gathered metadata across ranks with ==.
                    cls.__eq__ = lambda self, other: type(other).__name__ == "EmptyLogits"
                return data
            return original_recursively_apply(func, data, *args, **kwargs)

        _patched_recursively_apply.__unsloth_patched__ = True

        for mod_name, mod in tuple(sys.modules.items()):
            if mod_name.startswith("accelerate") and mod is not None:
                if getattr(mod, "recursively_apply", None) is original_recursively_apply:
                    try:
                        setattr(mod, "recursively_apply", _patched_recursively_apply)
                    except Exception:
                        pass

    original_find_device = getattr(acc_ops, "find_device", None)
    if original_find_device is not None and not getattr(
        original_find_device, "__unsloth_patched__", False
    ):
        from collections.abc import Mapping

        @functools.wraps(original_find_device)
        def _patched_find_device(data):
            import torch

            found_sentinel = False

            def _search(obj):
                nonlocal found_sentinel
                if type(obj).__name__ == "EmptyLogits":
                    found_sentinel = True
                elif isinstance(obj, Mapping):
                    for value in obj.values():
                        device = _search(value)
                        if device is not None:
                            return device
                elif isinstance(obj, (tuple, list)):
                    for value in obj:
                        device = _search(value)
                        if device is not None:
                            return device
                elif isinstance(obj, torch.Tensor):
                    return obj.device
                return None

            device = _search(data)
            if device is None and found_sentinel:
                # Debug mode calls find_device(...).type on gather/broadcast inputs.
                try:
                    from accelerate.state import PartialState
                    return PartialState().device
                except Exception:
                    pass
            return device

        _patched_find_device.__unsloth_patched__ = True

        for mod_name, mod in tuple(sys.modules.items()):
            if mod_name.startswith("accelerate") and mod is not None:
                if getattr(mod, "find_device", None) is original_find_device:
                    try:
                        setattr(mod, "find_device", _patched_find_device)
                    except Exception:
                        pass


# The one ImportError worth answering False to. Matching "torchao" anywhere is too wide: a
# missing submodule ("No module named 'torchao.quantization'") or libtorchao_ops_cuda.so also
# says it and both mean genuinely broken, so match the version complaint itself. peft's wording
# is first; the rest are how other libraries phrase it, so a reword does not silently turn this
# back into a raise.
_TORCHAO_STALE_VERSION_ERROR = re.compile(
    r"incompatible version of torchao"
    r"|torchao.{0,120}?only versions?\s+(?:above|below|>=|<=)"
    r"|(?:requires|needs|expected)\s+torchao\s*[><=!]",
    re.IGNORECASE | re.DOTALL,
)


def fix_peft_stale_torchao_import_error():
    """Stop an old torchao from aborting LoRA creation that never uses it.

    ``peft.import_utils.is_torchao_available`` returns False when torchao is
    absent but raises when it is installed and older than peft's minimum, and
    ``dispatch_torchao`` calls it for every LoRA layer, so one stale optional
    dependency ends ``get_peft_model``. "Installed but unusable" is closer to
    "not installed" than to "fatal", so answer False and warn once.

    Returns True when patched, False when no patch is needed, None when peft
    is absent.
    """
    try:
        import peft.import_utils as peft_import_utils
    except Exception:
        return None

    original = getattr(peft_import_utils, "is_torchao_available", None)
    if original is None:
        return None
    if getattr(original, "__unsloth_patched__", False):
        return False

    warned = [False]

    @functools.wraps(original)
    def is_torchao_available(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except ImportError as exc:
            # Only the version complaint; any other torchao import failure is a real problem and must still surface.
            message = str(exc)
            if _TORCHAO_STALE_VERSION_ERROR.search(message) is None:
                raise
            if not warned[0]:
                warned[0] = True
                logger.warning(
                    f"Unsloth: Ignoring an unusable torchao so LoRA can still "
                    f"be built ({message}). Run "
                    f"`pip install --upgrade torchao` if you need torchao "
                    f"quantization."
                )
            return False

    is_torchao_available.__unsloth_patched__ = True

    patched = False
    try:
        peft_import_utils.is_torchao_available = is_torchao_available
        patched = True
    except Exception:
        return False

    # `from peft.import_utils import is_torchao_available` binds the original into each importing
    # module, so patching import_utils alone leaves the real caller, peft.tuners.lora.torchao,
    # still raising.
    for mod_name, mod in tuple(sys.modules.items()):
        if not mod_name.startswith("peft") or mod is None:
            continue
        if getattr(mod, "is_torchao_available", None) is original:
            try:
                setattr(mod, "is_torchao_available", is_torchao_available)
            except Exception:
                pass
    return patched


# Matches both spellings a torchao removal produces: the class name, and the module that used to
# define it. Only these two, so a torchao that is BROKEN rather than newer still raises.
_PEFT_TORCHAO_MISSING_TENSOR_SUBCLASS = re.compile(
    r"linear_?activation_?quantized_?tensor|affine_?quantized_?tensor",
    re.IGNORECASE | re.DOTALL,
)


# The two imports `dispatch_torchao` performs, in upstream's order.
_PEFT_TORCHAO_TENSOR_SUBCLASSES = (
    ("torchao.dtypes", "AffineQuantizedTensor"),
    ("torchao.quantization", "LinearActivationQuantizedTensor"),
)


def _peft_torchao_tensor_subclasses():
    """``(classes, missing)`` for the two subclasses `dispatch_torchao` checks: a tuple usable as
    the second argument of ``isinstance``, plus the names that are gone. Only a failure naming one
    of those two counts as gone; anything else is a broken install and is re-raised."""
    classes = []
    missing = []
    for module_name, class_name in _PEFT_TORCHAO_TENSOR_SUBCLASSES:
        try:
            classes.append(getattr(importlib.import_module(module_name), class_name))
        except (ImportError, AttributeError) as exc:
            if _PEFT_TORCHAO_MISSING_TENSOR_SUBCLASS.search(str(exc)) is None:
                raise
            missing.append(class_name)
    return tuple(classes), tuple(missing)


def _guard_peft_torchao_dispatcher(original):
    """Wrap one `dispatch_torchao` so a removed tensor subclass costs only that class.

    The wrapper calls through, so a torchao shipping both classes behaves exactly as upstream.
    Only on a failure naming one of the two does it redo upstream's work against the classes that
    remain, so an AffineQuantizedTensor weight still gets a TorchaoLoraLinear.
    """
    warned = [False]
    # The defining module's own globals, so the degraded path uses the very objects upstream would
    # have, including an is_torchao_available already patched by the sibling fix.
    namespace = getattr(original, "__globals__", None)
    if not isinstance(namespace, dict):
        namespace = {}

    def _upstream(name, module_name):
        value = namespace.get(name, None)
        if value is not None:
            return value
        return getattr(importlib.import_module(module_name), name, None)

    def _redo_dispatch(classes, args, kwargs):
        """Upstream's body, with `classes` standing in for the two-class isinstance tuple.

        Arguments are read by POSITION because peft 0.19 renamed the third parameter from
        `lora_config` to `config`, and forwarded as `target, adapter_name, **kwargs`, which is
        exactly how peft <= 0.18 builds TorchaoLoraLinear.
        """
        try:
            signature = inspect.signature(original)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
        except (TypeError, ValueError):
            return None
        positional = [
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        layer_kwargs = {}
        for name, parameter in signature.parameters.items():
            if parameter.kind is parameter.VAR_KEYWORD:
                layer_kwargs = dict(arguments.get(name, None) or {})
        if len(positional) < 2:
            return None
        target = arguments.get(positional[0], None)
        adapter_name = arguments.get(positional[1], None)
        if target is None or adapter_name is None:
            return None

        base_tuner_layer = _upstream("BaseTunerLayer", "peft.tuners.tuners_utils")
        if base_tuner_layer is not None and isinstance(target, base_tuner_layer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target
        if not hasattr(target_base_layer, "weight"):
            return None
        is_torchao_available = _upstream("is_torchao_available", "peft.import_utils")
        if is_torchao_available is not None and not is_torchao_available():
            return None
        if not classes or not isinstance(target_base_layer.weight, classes):
            return None  # upstream's answer for an unrecognised weight: the loop moves on
        torchao_lora_linear = _upstream("TorchaoLoraLinear", "peft.tuners.lora.torchao")
        if torchao_lora_linear is None:
            return None
        return torchao_lora_linear(target, adapter_name, **layer_kwargs)

    @functools.wraps(original)
    def dispatch_torchao(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except ImportError as exc:
            message = str(exc)
            if _PEFT_TORCHAO_MISSING_TENSOR_SUBCLASS.search(message) is None:
                raise
            classes, missing = _peft_torchao_tensor_subclasses()
            if not missing:
                # The message named one of the two classes but both are there, so this came from
                # somewhere else in the dispatcher and is a real failure. Redoing the dispatch
                # would swallow it and re-run whatever construction already happened.
                raise
        if not warned[0]:
            warned[0] = True
            kept = ", ".join(cls.__name__ for cls in classes) or "none of them"
            logger.warning(
                f"Unsloth: This torchao no longer ships "
                f"{' and '.join(missing) or 'a tensor subclass'}, which peft's "
                f"torchao LoRA dispatcher imports to run one isinstance check "
                f"({message}), so that dispatcher raised for every LoRA layer. "
                f"Unsloth now runs the same check against the subclasses this "
                f"torchao does ship ({kept}), so weights of those types still get "
                f"peft's torchao LoRA layer and everything else gets an ordinary "
                f"one. Only weights of the removed subclass cannot be matched; "
                f"install `torchao<0.18` if you need those."
            )
        return _redo_dispatch(classes, args, kwargs)

    dispatch_torchao.__unsloth_patched__ = True
    return dispatch_torchao


def fix_peft_torchao_missing_tensor_subclass():
    """Stop a removed torchao class from aborting LoRA creation that never uses torchao.

    peft's dispatch table runs every dispatcher in order and keeps the first non-None, so one that
    RAISES ends `get_peft_model` for models it does not even apply to. `dispatch_torchao` imports
    AffineQuantizedTensor and LinearActivationQuantizedTensor at call time purely for one
    isinstance check, and torchao 0.18 deleted the second outright. QLoRA never sees it, because
    dispatch_bnb_4bit matches first; plain LoRA falls through and dies.

    AffineQuantizedTensor is still there, so declining outright would quietly give real
    torchao-quantized weights an ordinary LoRA layer. The wrapper redoes the isinstance check
    against whichever classes this torchao ships instead, and costs nothing when both are present.

    True when patched, False when no patch is needed, None when peft is absent.
    """
    try:
        import peft  # noqa: F401
    except Exception:
        return None
    try:
        # Normally already imported, and costs nothing extra: the torchao imports that break live
        # inside the functions, not the module body.
        import peft.tuners.lora.torchao  # noqa: F401
    except Exception:
        pass

    # `from .torchao import dispatch_torchao` copies the function into peft.tuners.lora.model,
    # where the dispatch list is built, so patching the definer alone leaves the real caller
    # raising. One wrapper per distinct original, shared, so a missing class warns once.
    wrapped = []
    patched = False
    for mod_name, mod in tuple(sys.modules.items()):
        if not mod_name.startswith("peft") or mod is None:
            continue
        original = getattr(mod, "dispatch_torchao", None)
        if original is None or not callable(original):
            continue
        if getattr(original, "__unsloth_patched__", False):
            continue
        replacement = None
        for seen, wrapper in wrapped:
            if seen is original:
                replacement = wrapper
                break
        if replacement is None:
            try:
                replacement = _guard_peft_torchao_dispatcher(original)
            except Exception:
                continue
            wrapped.append((original, replacement))
        try:
            setattr(mod, "dispatch_torchao", replacement)
            patched = True
        except Exception:
            pass
    return patched


# Every name torchao 0.18.0 imports from torch.nn.functional. scaled_grouped_mm is on the path
# of a plain `import torchao`; scaled_dot_product_attention is listed for completeness, and the
# loop below skips symbols torch provides.
_TORCHAO_TORCH_SYMBOLS = (
    "ScalingType",
    "SwizzleType",
    "scaled_grouped_mm",
    "scaled_dot_product_attention",
)


def _make_torch_symbol_placeholder(name, detail):
    """A stand-in that imports cleanly and refuses to be used.

    Pretending to be a real enum would be worse than the crash it replaces: it
    could hand a float8 path a meaningless value. torchao 0.17 left these names
    undefined on older torch anyway, so anything wanting them already raised.
    """
    message = (
        f"Unsloth: `torch.nn.functional.{name}` does not exist in this torch. "
        f"{detail} Unsloth supplied a placeholder so that importing torchao "
        f"(and therefore unsloth) still works, but this symbol cannot be used. "
        f"Install `torchao<0.18` to use float8/MX features on this torch, or "
        f"upgrade torch."
    )

    class _Meta(type):
        def __getattr__(cls, item):
            raise RuntimeError(message)

        def __call__(cls, *args, **kwargs):
            raise RuntimeError(message)

        def __repr__(cls):
            return f"<unsloth placeholder for torch.nn.functional.{name}>"

    placeholder = _Meta(name, (), {"__doc__": message})
    # So we can recognise our own object later and never double-patch.
    type.__setattr__(placeholder, "__unsloth_placeholder__", True)
    return placeholder


# The same skew one layer down: torchao 0.18 does @implements([aten._grouped_mm.default]) at
# module scope in float8/float8_tensor.py, and that op only arrived in torch 2.8, so older torch
# raises AttributeError. This goes through torch.ops, so the schema itself has to exist.
_ATEN_GROUPED_MM_SCHEMA = (
    "_grouped_mm(Tensor self, Tensor mat2, Tensor? offs=None, "
    "Tensor? bias=None, ScalarType? out_dtype=None) -> Tensor"
)

# Module level on purpose: a torch.library.Library deregisters everything it defined once
# collected, so a local would undo itself.
_aten_grouped_mm_library = None


def _torch_op_is_missing(namespace, name):
    """Is `torch.ops.<namespace>.<name>` absent on this torch?

    Only a plain AttributeError counts: anything else means we could not tell,
    and when unsure we must not register into the aten namespace.
    """
    try:
        import torch
        ns = getattr(torch.ops, namespace)
    except Exception:
        return False
    try:
        getattr(ns, name)
    except AttributeError:
        return True
    except Exception:
        return False
    return False


def _ensure_aten_grouped_mm(detail):
    """Define an unusable `aten::_grouped_mm` so torchao's decorator resolves.

    No real implementation on purpose: torchao only wants somewhere to hang a
    float8 handler this torch never dispatches to, so the schema is the whole
    requirement, and a silently wrong grouped matmul would be worse than the
    crash it replaces. Calling it raises and says why.
    """
    global _aten_grouped_mm_library
    if _aten_grouped_mm_library is not None:
        return False
    if not _torch_op_is_missing("aten", "_grouped_mm"):
        return False

    message = (
        f"Unsloth: `torch.ops.aten._grouped_mm` does not exist in this torch. "
        f"{detail} Unsloth registered a placeholder schema so that importing "
        f"torchao (and therefore unsloth) still works, but the operator cannot "
        f"be used. Upgrade to torch >= 2.8 for grouped matmul, or install "
        f"`torchao<0.18`."
    )

    def _refuse(
        self,
        mat2,
        offs = None,
        bias = None,
        out_dtype = None,
    ):
        raise RuntimeError(message)

    try:
        import torch

        # FRAGMENT adds to a namespace someone else owns; DEF would try to claim "aten" outright and be rejected.
        library = torch.library.Library("aten", "FRAGMENT")
        library.define(_ATEN_GROUPED_MM_SCHEMA)
        library.impl("_grouped_mm", _refuse, "CompositeExplicitAutograd")
    except Exception:
        # A torch that will not let us register keeps its own error.
        return False

    _aten_grouped_mm_library = library
    return True


def fix_torchao_torch_symbol_skew():
    """Let `import unsloth` survive a torchao built for a newer torch.

    torchao 0.17 guarded the import behind `torch_version_at_least("2.10.0")`;
    0.18.0 left it unguarded at module level, so torch below 2.10 raises
    "cannot import name 'ScalingType' from 'torch.nn.functional'" while
    importing transformers, naming neither torchao nor torch.

    Narrow by design: only fires when torchao is installed, its version has
    the bug, and torch really lacks the symbol. Nothing is masked otherwise.
    """
    if importlib.util.find_spec("torchao") is None:
        return False
    try:
        torchao_version = importlib_version("torchao")
    except Exception:
        return False
    try:
        # 0.17 and earlier guard their own import.
        if Version(torchao_version) < Version("0.18.0"):
            return False
    except Exception:
        return False

    try:
        import torch
        import torch.nn.functional as F
        torch_version = str(torch.__version__)
    except Exception:
        return False

    detail = (
        f"torchao {torchao_version} imports it unconditionally, but "
        f"torch {torch_version} does not provide it (it arrived in "
        f"torch 2.10)."
    )

    patched = []
    for name in _TORCHAO_TORCH_SYMBOLS:
        if hasattr(F, name):
            continue  # real torch symbol, or already placed by us
        try:
            setattr(F, name, _make_torch_symbol_placeholder(name, detail))
            patched.append(name)
        except Exception:
            pass
    if patched:
        logger.info(
            "Unsloth: torchao %s needs torch.nn.functional.%s, which torch %s "
            "does not have. Adding an unusable placeholder so the import "
            "succeeds; install torchao<0.18 to use float8/MX features.",
            torchao_version,
            "/".join(patched),
            torch_version,
        )

    # The aten-op half of the same skew, independent of the loop above: a torch can have every
    # functional symbol and still lack the operator, or the reverse.
    op_detail = (
        f"torchao {torchao_version} registers a handler for it at "
        f"import time, but torch {torch_version} does not provide it "
        f"(it arrived in torch 2.8)."
    )
    if _ensure_aten_grouped_mm(op_detail):
        patched.append("aten::_grouped_mm")
        logger.info(
            "Unsloth: torchao %s registers a handler for torch.ops.aten."
            "_grouped_mm, which torch %s does not have. Registering an "
            "unusable placeholder schema so the import succeeds.",
            torchao_version,
            torch_version,
        )
    return bool(patched)


# vLLM inspects architectures in a separate process that imports torchao itself, never sees a
# parent monkey-patch, and fails the same way. sitecustomize is the one hook reaching a process
# we do not launch: site imports it at startup off the inherited PYTHONPATH. A .pth would need
# a real site directory, which a library has no business writing into.

_SUBPROCESS_FIX_DIRNAME = "unsloth_subprocess_import_fix"


def _subprocess_fix_directory():
    """A private directory for the generated sitecustomize.

    Everything on PYTHONPATH runs in every subprocess and /tmp is shared on
    Linux, so a fixed name there would let whoever created it first run code as
    everyone else. Scope it per user and refuse a path this user does not own.

    Ownership alone is not enough: ``exist_ok = True`` does not apply ``mode``
    to an existing directory, so one left group- or world-writable stays that
    way and anyone who can write into it can replace the ``sitecustomize.py``.
    Take the write bits away, and refuse the directory if they will not go.
    """
    import stat
    import tempfile

    name = _SUBPROCESS_FIX_DIRNAME
    try:
        name += "-%d" % os.getuid()
    except AttributeError:
        name += "-" + (os.environ.get("USERNAME") or "user")
    directory = os.path.join(tempfile.gettempdir(), name)
    os.makedirs(directory, mode = 0o700, exist_ok = True)
    if hasattr(os, "getuid"):
        info = os.lstat(directory)
        if stat.S_ISLNK(info.st_mode) or info.st_uid != os.getuid():
            raise RuntimeError(
                "refusing a subprocess fix directory owned by another user: " + directory
            )
        # chmod then re-read: some network and FUSE mounts ignore mode bits and report success without
        # changing anything.
        if stat.S_IMODE(info.st_mode) & 0o022:
            try:
                os.chmod(directory, 0o700)
            except Exception:
                pass
            info = os.lstat(directory)
            if stat.S_IMODE(info.st_mode) & 0o022:
                raise RuntimeError(
                    "refusing a group- or world-writable subprocess fix "
                    "directory (mode %04o): %s" % (stat.S_IMODE(info.st_mode), directory)
                )
    return directory


def _subprocess_sitecustomize_source():
    """The sitecustomize we hand to child processes.

    It chains rather than shadows: `sitecustomize` is a single global name that
    other things legitimately install, so replacing it would disable them in
    every subprocess. Ours runs the next one on sys.path first, then the fix.
    """
    return '''"""Written by unsloth. Makes `import torchao` survive a torch that
predates the symbols torchao 0.18 imports unconditionally, in processes
unsloth did not launch (notably vLLM's model-architecture inspector)."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _same_path(a, b):
    """Two spellings of one file or directory, symlinks included."""
    try:
        if os.path.samefile(a, b):
            return True
    except Exception:
        pass
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def _chain_to_the_real_sitecustomize():
    """Do not shadow somebody else's sitecustomize."""
    import importlib.util
    from importlib.machinery import PathFinder
    for entry in sys.path:
        # Ask the import system rather than probing for a filename, so the
        # package (`sitecustomize/__init__.py`) and .pyc forms chain too.
        try:
            # Canonical comparison, not a string one: a symlink alias of our
            # own directory also on sys.path would otherwise look like a
            # different location, and the two spellings would chain to each
            # other until the stack ran out.
            if not entry or _same_path(entry, _HERE):
                continue
            spec = PathFinder.find_spec("sitecustomize", [entry])
        except Exception:
            continue
        if spec is None or spec.loader is None:
            continue
        # Second guard for the same hazard, in case the alias is the file
        # rather than the directory: never chain to this very hook.
        try:
            origin = getattr(spec, "origin", None)
            if origin and (_same_path(origin, __file__)
                           or _same_path(os.path.dirname(origin), _HERE)):
                continue
        except Exception:
            continue
        try:
            mod = importlib.util.module_from_spec(spec)
            # A package needs its own name in sys.modules for `from . import`
            # to resolve, and it needs it AFTER initialization too: a callback
            # it registers (atexit, a hook) does its relative imports later,
            # and would find our non-package module under `sitecustomize` and
            # fail. So leave the real one there on success -- which is also
            # what `import sitecustomize` returns when we are not installed --
            # and only put the previous entry back if it failed to load.
            previous = sys.modules.get("sitecustomize")
            sys.modules["sitecustomize"] = mod
            try:
                spec.loader.exec_module(mod)
            except BaseException:
                if previous is None:
                    sys.modules.pop("sitecustomize", None)
                else:
                    sys.modules["sitecustomize"] = previous
                raise
            break
        except Exception:
            # A broken sitecustomize elsewhere must not stop us, and must not
            # take down every subprocess either.
            break


def _apply():
    import importlib.metadata as md
    try:
        version = md.version("torchao")
    except Exception:
        return
    try:
        major, minor = (int("".join(c for c in p if c.isdigit()) or 0)
                        for p in str(version).split(".")[:2])
        if (major, minor) < (0, 18):
            return
    except Exception:
        return
    try:
        import torch
        import torch.nn.functional as F
    except Exception:
        return
    detail = ("torchao %s imports it unconditionally, but torch %s does not "
              "provide it (it arrived in torch 2.10)." % (version, torch.__version__))
    for name in ("ScalingType", "SwizzleType", "scaled_grouped_mm",
                 "scaled_dot_product_attention"):
        if hasattr(F, name):
            continue
        message = ("Unsloth: `torch.nn.functional.%s` does not exist in this "
                   "torch. %s Unsloth supplied a placeholder so that importing "
                   "torchao still works, but this symbol cannot be used. "
                   "Install `torchao<0.18`, or upgrade torch." % (name, detail))

        def _make(msg):
            class _Meta(type):
                def __getattr__(cls, item):
                    raise RuntimeError(msg)

                def __call__(cls, *args, **kwargs):
                    raise RuntimeError(msg)
            return _Meta

        try:
            placeholder = _make(message)(name, (), {"__doc__": message})
            type.__setattr__(placeholder, "__unsloth_placeholder__", True)
            setattr(F, name, placeholder)
        except Exception:
            pass

    # torchao also hangs a handler off aten._grouped_mm at import time, and
    # that operator only exists from torch 2.8. Same skew, different lookup:
    # supplying the torch.nn.functional names above does not help it.
    try:
        torch.ops.aten._grouped_mm
    except AttributeError:
        op_message = (
            "Unsloth: `torch.ops.aten._grouped_mm` does not exist in this "
            "torch. torchao %s registers a handler for it at import time, but "
            "torch %s does not provide it (it arrived in torch 2.8). Unsloth "
            "registered a placeholder schema so the import succeeds; the "
            "operator itself cannot be used."
            % (version, torch.__version__))

        def _refuse(self, mat2, offs=None, bias=None, out_dtype=None):
            raise RuntimeError(op_message)

        try:
            # Held on the module so the Library is not collected -- that
            # would deregister the schema again.
            global _ATEN_LIBRARY
            _ATEN_LIBRARY = torch.library.Library("aten", "FRAGMENT")
            _ATEN_LIBRARY.define(
                "_grouped_mm(Tensor self, Tensor mat2, Tensor? offs=None, "
                "Tensor? bias=None, ScalarType? out_dtype=None) -> Tensor")
            _ATEN_LIBRARY.impl("_grouped_mm", _refuse,
                               "CompositeExplicitAutograd")
        except Exception:
            pass
    except Exception:
        pass


_ATEN_LIBRARY = None


class _TorchaoImportHook:
    """Runs the fix when a child actually imports torchao, and not before.

    This module is on PYTHONPATH, so it starts every Python descendant, most
    of which never touch torchao. Calling `_apply()` here would import torch
    in all of them, adding seconds of startup and torch's memory to unrelated
    utilities and workers. A meta_path finder costs a string compare instead,
    and still runs before torchao's module body, which is the only ordering
    the fix needs.
    """

    def find_spec(self, fullname, path = None, target = None):
        if fullname != "torchao":
            return None
        try:
            sys.meta_path.remove(self)  # once, and before _apply imports torch
        except ValueError:
            pass
        try:
            _apply()
        except Exception:
            pass
        return None  # let the normal finders import torchao


# Chaining gives our name away to the real sitecustomize, so keep our own
# module object alive here: the finder below outlives this file's execution.
_SELF = sys.modules.get(__name__)
_chain_to_the_real_sitecustomize()
try:
    sys.meta_path.insert(0, _TorchaoImportHook())
except Exception:
    # Never let this abort interpreter startup: it would break every
    # subprocess, which is far worse than the import error it fixes.
    pass
'''


def _write_hook_atomically(target, source):
    """Put `source` at `target` without ever writing through a symlink.

    A predictable temporary name (`sitecustomize.py.<pid>.tmp`) can be
    pre-created as a symlink while the directory is still group- or
    world-writable: tightening it afterwards does not revoke what is already
    inside. The write would land on a file the other user owns, and os.replace
    renames the link itself into place, so `sitecustomize.py` stays theirs to
    rewrite. A random name opened O_EXCL|O_NOFOLLOW cannot be pre-empted and
    refuses a symlink instead of following it.
    """
    import binascii

    directory = os.path.dirname(target)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_BINARY", 0)
    payload = source.encode("utf-8")
    for _ in range(16):
        tmp = os.path.join(
            directory,
            ".sitecustomize.py.%s.tmp" % binascii.hexlify(os.urandom(8)).decode(),
        )
        try:
            handle = os.open(tmp, flags, 0o600)  # nobody else may rewrite it
        except FileExistsError:
            continue
        try:
            try:
                os.fchmod(handle, 0o600)  # a loose umask must not widen it
            except Exception:
                pass
            with os.fdopen(handle, "wb") as stream:
                stream.write(payload)
            os.replace(tmp, target)  # atomic
        except BaseException:
            # A random name is never reused, so a failed attempt is litter rather than something the next
            # run overwrites.
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise
        return
    raise RuntimeError("could not create a private temporary file in " + directory)


def _existing_hook_is_trustworthy(target):
    """Can the file already at `target` only have been written by us?

    Tightening the directory does not revoke access to what is already inside
    it: a file planted while it was group- or world-writable stays foreign
    owned, or is a symlink into somewhere still writable. This file runs in
    every Python descendant, so anything but a private regular file of ours is
    replaced outright, even when its contents match, which is exactly what an
    attacker would arrange to skip the write below.
    """
    import stat

    try:
        info = os.lstat(target)
    except FileNotFoundError:
        return True  # nothing there yet; the write creates it
    except Exception:
        return False
    if not stat.S_ISREG(info.st_mode):  # symlink, directory, fifo, device
        return False
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        return False
    # Writable by anyone else means its contents prove nothing.
    return not stat.S_IMODE(info.st_mode) & 0o022


def _torch_really_has(F, name):
    """Does torch itself provide `name`, or is it a placeholder we installed?

    `_gpu_init` runs `fix_torchao_torch_symbol_skew()` just before the function
    below, so a plain `hasattr` would read as healthy in exactly the
    environments the child fix exists for.
    """
    symbol = getattr(F, name, None)
    if symbol is None:
        return False
    return not getattr(symbol, "__unsloth_placeholder__", False)


def propagate_torchao_fix_to_subprocesses():
    """Make the torchao fix apply to child processes too.

    A no-op unless the fix is needed: returns early when torchao is absent,
    old enough to guard its own import, or when torch already has the symbols.
    Nothing is written and PYTHONPATH is untouched on a healthy pair.

    The generated file inlines the symbol fix rather than importing unsloth: a
    sitecustomize runs at the start of every subprocess on the machine, so an
    `import unsloth` there would pay the full import cost each time and could
    recurse through this function. Anything else found to be needed should be
    inlined here too. The other two torchao fixes need no child: both guard
    work in this process (LoRA construction, `import torchtune` via xcodec2),
    and the vLLM inspector this hook exists for imports neither.

    Returns the directory added to PYTHONPATH, or None.
    """
    if importlib.util.find_spec("torchao") is None:
        return None
    try:
        if Version(importlib_version("torchao")) < Version("0.18.0"):
            return None
    except Exception:
        return None
    try:
        import torch.nn.functional as F

        # Both halves of the skew must be absent for there to be nothing to do. Today a torch missing
        # the operator also misses the functional symbols (2.8 vs 2.10), so the second check never
        # fires alone. Our own patches do not count: the in-process fix ran first.
        if (
            all(_torch_really_has(F, n) for n in _TORCHAO_TORCH_SYMBOLS)
            and _aten_grouped_mm_library is None
            and not _torch_op_is_missing("aten", "_grouped_mm")
        ):
            return None  # this torch is new enough; nothing to do
    except Exception:
        return None

    try:
        directory = _subprocess_fix_directory()
        target = os.path.join(directory, "sitecustomize.py")
        source = _subprocess_sitecustomize_source()
        # Rewrite only when it differs, so concurrent runs do not fight and a reader never sees a
        # truncated file. Matching contents count as evidence only when the file is ours. A directory
        # in the way makes os.replace raise, which becomes "no subprocess fix" rather than a hook we
        # do not trust.
        if _existing_hook_is_trustworthy(target):
            try:
                existing = open(target, "r", encoding = "utf-8").read()
            except Exception:
                existing = None
        else:
            existing = None
        if existing != source:
            _write_hook_atomically(target, source)
    except Exception as exception:
        logger.warning(
            "Unsloth: could not stage the torchao subprocess fix (%s). vLLM "
            "may fail to inspect model architectures.",
            exception,
        )
        return None

    # os.pathsep, not ":": Windows uses ";".
    current = os.environ.get("PYTHONPATH", "")
    # An empty component is an import location, not padding: PYTHONPATH="$PYTHONPATH:/opt/lib"
    # leaves one when PYTHONPATH was unset, and CPython reads it as the cwd. A SET-BUT-EMPTY
    # PYTHONPATH is the opposite: CPython ignores it, so it must not become a lone "" that ADDS
    # the cwd.
    parts = current.split(os.pathsep) if current else []
    if directory not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([directory] + parts)
        logger.info(
            "Unsloth: torchao %s needs torch symbols this torch lacks. Added "
            "a sitecustomize to PYTHONPATH so subprocesses (vLLM's model "
            "inspector) can import torchao too.",
            importlib_version("torchao"),
        )
    return directory


# torchao 0.18.0 moved torchao/dtypes/nf4tensor.py under quantization/quantize_/workflows/nf4/,
# but torchtune (and xcodec2 through it) still imports the old path. Same shape as the vLLM
# tokenizer stub: a meta path finder APPENDED after the real ones, so an older torchao wins and
# the alias resolves lazily.
_TORCHAO_NF4_OLD = "torchao.dtypes.nf4tensor"
_TORCHAO_NF4_NEW = "torchao.quantization.quantize_.workflows.nf4.nf4_tensor"
_TORCHAO_NF4_SENTINEL = "__unsloth_torchao_nf4_alias__"


class _TorchaoNF4AliasLoader(importlib.abc.Loader):
    __slots__ = ("module_name", "real_spec")

    def __init__(self, module_name):
        self.module_name = module_name
        self.real_spec = None

    def create_module(self, spec):
        # Return the RELOCATED module itself, not a stub with a hand-copied surface: torchtune then
        # sees whatever torchao ships, and this cannot rot as symbols are added.
        module = importlib.import_module(_TORCHAO_NF4_NEW)
        # module_from_spec is about to overwrite this shared object's __spec__ with the old-name one
        # (_bootstrap.py assigns it unconditionally), leaving find_spec reporting the old name and
        # making reload run the no-op exec_module below instead of the file.
        self.real_spec = getattr(module, "__spec__", None)
        return module

    def exec_module(self, module):
        # Already imported, so nothing to execute. Put back the __spec__ module_from_spec just clobbered.
        if self.real_spec is not None:
            try:
                module.__spec__ = self.real_spec
            except Exception:
                pass
        return None


class _TorchaoNF4AliasFinder(importlib.abc.MetaPathFinder):
    __slots__ = (_TORCHAO_NF4_SENTINEL,)

    def __init__(self):
        setattr(self, _TORCHAO_NF4_SENTINEL, True)

    def find_spec(
        self,
        fullname,
        path = None,
        target = None,
    ):
        if fullname != _TORCHAO_NF4_OLD:
            return None
        try:
            if importlib.util.find_spec(_TORCHAO_NF4_NEW) is None:
                return None  # neither layout: let the real ImportError happen
        except Exception:
            return None
        return importlib.machinery.ModuleSpec(
            name = fullname,
            loader = _TorchaoNF4AliasLoader(fullname),
            is_package = False,
        )


def fix_torchao_nf4tensor_move():
    if importlib.util.find_spec("torchao") is None:
        return
    for finder in sys.meta_path:
        if getattr(finder, _TORCHAO_NF4_SENTINEL, False):
            return
    # Appended, not inserted at 0, so a real module on older torchao wins.
    sys.meta_path.append(_TorchaoNF4AliasFinder())


_TORCHAO_INTMM_MODULES = (
    "torchao.kernel.intmm",  # every release up to and including 0.18.0
    "torchao.quantization.quantize_.workflows.int8.kernels",  # main after pytorch/ao#4718
)
_TORCHAO_INTMM_MODULE = _TORCHAO_INTMM_MODULES[0]  # kept for callers that named the old home
_TORCHAO_INTMM_SENTINEL = "__unsloth_torchao_intmm_patch__"
_TORCHAO_INT_MM_ENV = "UNSLOTH_TORCHAO_INT_MM_FIX"

# The copy below hard-codes torchao's cuBLAS guards and fp32 fallback: ALL must be in the installed source.
_TORCHAO_SAFE_INT_MM_MARKERS = (
    "input.__repr__()",
    "dynamo_is_compiling()",
    "out_dtype(torch.ops.aten.mm.default",
    "j_is_nonzero_multiple_of_8",
    "k_is_nonzero_multiple_of_8",
    "mat2.is_contiguous()",
)


def _is_fake_tensor(x):
    """Covers exactly the set the substring repr probe did: ``is_fake`` unwraps FunctionalTensor too."""
    try:
        from torch._subclasses.fake_tensor import is_fake
        return bool(is_fake(x))
    except Exception:
        pass
    try:
        from torch._subclasses.fake_tensor import FakeTensor
        return isinstance(x, FakeTensor)
    except Exception:
        return False


def _make_safe_int_mm(mod, original):
    """A FULL copy of torchao 0.17.0's body, not a wrapper: its very first statement IS the probe."""
    import torch

    out_dtype = mod.out_dtype
    dynamo_is_compiling = mod.dynamo_is_compiling

    @functools.wraps(original)
    def safe_int_mm(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
        if dynamo_is_compiling() or _is_fake_tensor(input):
            if input.device.type == "cpu":
                # Matmul in int32 is slow on CPU and not supported well by Inductor cpp backend
                return out_dtype(
                    torch.ops.aten.mm.default, torch.int32, input.float(), mat2.float()
                )
            return out_dtype(torch.ops.aten.mm.default, torch.int32, input, mat2)

        assert (
            mat2.device == input.device
        ), f"need both tensors to be on the same device but got {mat2.device} and {input.device}"
        device_cpu = "cpu" in [mat2.device.type, input.device.type]
        j_is_nonzero_multiple_of_8 = (input.shape[1] % 8 == 0) and (input.shape[1] > 0)
        k_is_nonzero_multiple_of_8 = (mat2.shape[1] % 8 == 0) and (mat2.shape[1] > 0)
        bad_dimensions_for_cublas = not (j_is_nonzero_multiple_of_8 and k_is_nonzero_multiple_of_8)

        if device_cpu or bad_dimensions_for_cublas:
            return torch.matmul(input.cpu().to(torch.int32), mat2.cpu().to(torch.int32)).to(
                input.device.type
            )

        if not mat2.is_contiguous():  # silently gives incorrect result without this
            mat2 = mat2.contiguous()
        if (not input.is_contiguous()) and (
            input.shape[0] % 8 != 0
        ):  # gives cryptic error without this
            input = input.contiguous()
        try:
            return out_dtype(torch.ops.aten.mm.default, torch.int32, input, mat2)
        except Exception:
            # H100 float8: "addmm_cuda" not implemented for 'Float8_e4m3fn'
            return torch.matmul(input.to(torch.float32), mat2.to(torch.float32)).to(torch.int32)

    safe_int_mm.__unsloth_patched__ = True
    safe_int_mm.__unsloth_original__ = original
    return safe_int_mm


def _patch_torchao_intmm_module(mod):
    """Rebind ``safe_int_mm`` on either torchao home; True when this call installed it."""
    original = getattr(mod, "safe_int_mm", None)
    if original is None or not callable(original):
        return False
    if getattr(original, "__unsloth_patched__", False):
        return False
    # Off the module, never imported here, so this fails closed rather than borrowing our own operators.
    if not hasattr(mod, "out_dtype") or not hasattr(mod, "dynamo_is_compiling"):
        return False
    try:
        source = inspect.getsource(original)
    except Exception:
        return False
    missing = [marker for marker in _TORCHAO_SAFE_INT_MM_MARKERS if marker not in source]
    if missing:
        if "input.__repr__()" not in missing:
            logger.warning(
                "Unsloth: torchao's safe_int_mm still probes input.__repr__() but its body "
                "changed (%s missing), so the capture-safe replacement was not installed. "
                "Eager int8 keeps syncing the device on every linear.",
                ", ".join(missing),
            )
        return False
    patched = _make_safe_int_mm(mod, original)
    try:
        mod.safe_int_mm = patched
    except Exception:
        return False
    # `torchao.kernel` and `torchao.quantization` re-export the function OBJECT, so they need a sweep.
    for name, other in tuple(sys.modules.items()):
        if other is None or not (name == "torchao" or name.startswith("torchao.")):
            continue
        if getattr(other, "safe_int_mm", None) is original:
            try:
                setattr(other, "safe_int_mm", patched)
            except Exception:
                pass
    return True


class _TorchaoIntmmLoader(importlib.abc.Loader):
    """The real loader, plus the patch once the module body finishes. Failures here leave torchao unpatched."""

    __slots__ = ("_loader",)

    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        create = getattr(self._loader, "create_module", None)
        if create is None:
            return None
        return create(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        try:
            _patch_torchao_intmm_module(module)
        except Exception:
            pass

    def __getattr__(self, attribute):
        return getattr(self._loader, attribute)


class _TorchaoIntmmPatchFinder(importlib.abc.MetaPathFinder):
    """Inserted at the FRONT of sys.meta_path: the module really exists, so PathFinder would answer first."""

    __slots__ = (_TORCHAO_INTMM_SENTINEL, "_finding")

    def __init__(self):
        setattr(self, _TORCHAO_INTMM_SENTINEL, True)
        self._finding = False  # find_spec below walks sys.meta_path again

    def find_spec(
        self,
        fullname,
        path = None,
        target = None,
    ):
        if fullname not in _TORCHAO_INTMM_MODULES or self._finding:
            return None
        self._finding = True
        try:
            spec = importlib.util.find_spec(fullname)
        except Exception:
            return None
        finally:
            self._finding = False
        if spec is None or spec.loader is None:
            return None
        if not hasattr(spec.loader, "exec_module"):
            return None  # a loader from before PEP 451; leave the import entirely alone
        try:
            spec.loader = _TorchaoIntmmLoader(spec.loader)
        except Exception:
            return None
        return spec


def fix_torchao_safe_int_mm_repr_probe():
    """Stop torchao's int8 GEMM from syncing the device to ask whether it is being traced.

    ``safe_int_mm`` branches on ``"FakeTensor" in input.__repr__()``; a real CUDA tensor's repr calls
    ``.item()``, so eager int8 syncs per linear and ``torch.cuda.graph`` capture fails outright. The
    gate is structural, not version-based, and the meta path finder covers a torchao imported after
    this call, which the int8 prequant path needs since it never calls ``quantize_``.
    ``UNSLOTH_TORCHAO_INT_MM_FIX=0`` keeps upstream's behaviour. True when patched or the finder was
    installed, False when there is nothing to do, None when torchao is absent or the fix is off."""
    if os.environ.get(_TORCHAO_INT_MM_ENV, "1").strip() == "0":
        return None
    try:
        if importlib.util.find_spec("torchao") is None:
            return None
    except Exception:
        return None
    patched_now = False
    for name in _TORCHAO_INTMM_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        try:
            patched_now = _patch_torchao_intmm_module(module) or patched_now
        except Exception:
            pass
    if all(name in sys.modules for name in _TORCHAO_INTMM_MODULES):
        return patched_now  # nothing left for a finder to catch
    for finder in sys.meta_path:
        if getattr(finder, _TORCHAO_INTMM_SENTINEL, False):
            return patched_now
    sys.meta_path.insert(0, _TorchaoIntmmPatchFinder())
    return True


# `datasets` fingerprints through dill, and dill._dill._is_builtin_module pickles a module by
# reference only if its __file__ starts with a sys prefix, ends with an extension suffix, or
# contains the literal `site-packages`. An install matching none (pip install --target, a
# PYTHONPATH overlay, a vendored tree) is pickled BY VALUE, and Dataset.from_dict then walks
# datasets/utils/_dill.py:_save_arrowTable -> create_arrowTable -> its globals -> the pyarrow
# MODULE, dying on pyarrow's Cython MonthDayNano, whose __module__ is `builtins`. Reproduced with
# the DIRECTORY NAME as the only variable.
_DILL_FIX_SENTINEL = "_unsloth_dill_by_reference_fix"
_DILL_FIX_ENV = "UNSLOTH_DISABLE_DILL_FIX"

# Never widened for these: dill's by-value contract is about the module the user works IN, and
# `python -m pkg` gives __main__ a real __spec__ that would satisfy the rule below.
_DILL_NEVER_BY_REFERENCE = frozenset(("__main__", "__mp_main__"))


def _dill_path_pickles_by_value(origin):
    """dill's rule, applied to a path, WITHOUT importing dill.

    Only a gate: a false positive here costs one `import dill`, because the
    real `_is_builtin_module` is consulted again before anything is patched.
    """
    if not origin:
        return False
    try:
        real = os.path.realpath(origin)
    except Exception:
        return False
    # The LITERAL path only, as dill does: it reads 'site-packages' in module.__file__ and resolves
    # only for the prefix comparisons. Searching `real` too would answer "not affected" for a
    # symlink into a site-packages-named directory dill still pickles by value.
    if "site-packages" in origin:
        return False
    for name in ("base_prefix", "base_exec_prefix", "exec_prefix", "prefix", "real_prefix"):
        prefix = getattr(sys, name, None)
        if not prefix:
            continue
        try:
            if origin.startswith(prefix) or real.startswith(os.path.realpath(prefix)):
                return False
        except Exception:
            continue
    return True


def _dill_environment_is_affected():
    """True only when a package on the `datasets` fingerprint path lives
    somewhere dill would pickle by value. An ordinary install answers False and
    the patch below is never installed at all."""
    for name in ("datasets", "pyarrow"):
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            continue
        if spec is None:
            continue
        if _dill_path_pickles_by_value(getattr(spec, "origin", None)):
            return True
    return False


def _dill_install_root(origin):
    """The directory a package was installed INTO, from one module's origin.

    `/opt/layer/python/pyarrow/__init__.py` and `/opt/layer/python/dill.py`
    both give `/opt/layer/python`, which is the `--target` directory or the
    PYTHONPATH entry -- the site-packages equivalent for this install.
    """
    if not origin:
        return None
    try:
        real = os.path.realpath(origin)
    except Exception:
        return None
    parent = os.path.dirname(real)
    # Any initializer, not just the source: a bytecode-only deployment gives pyarrow/__init__.pyc,
    # and matching __init__.py exactly would leave the root at .../pyarrow, where no sibling
    # metadata is found.
    if os.path.splitext(os.path.basename(real))[0] == "__init__":
        parent = os.path.dirname(parent)
    return parent or None


def _dill_distribution_paths(root):
    """The FILES some installed distribution put into `root`, as real paths.

    Only files a distribution RECORDED, never a directory: a directory cannot
    say which of its contents were installed, so claiming one would put a
    co-located `google/myconfig.py` on the dependency side of a `google`
    distribution. A top-level MODULE name is unambiguous (one file); a package
    name is not, and is skipped.

    Being under the directory is NOT the question. `pip install --target .` and
    a Lambda bundle put dependencies into the application's own directory, so
    treating the whole tree as dependency-owned would move `myproj.config` to
    by-reference too. Its mutable state would stop participating in a
    `recurse=True` fingerprint and `datasets` would serve a stale cached result
    -- silently, which is worse than the crash this patch exists to prevent.

    Recorded PATHS rather than top-level names, because a name cannot answer
    the question under a shared namespace, and because paths keep the modules a
    leading underscore would discard (`_soundfile`, `_multiprocess`) without
    guessing which underscore is metadata.
    """
    files = set()
    try:
        entries = os.listdir(root)
        root_real = os.path.realpath(root)
    except Exception:
        return files
    prefix = root_real.rstrip(os.sep) + os.sep

    def add(base, rel):
        rel = rel.strip().replace("\\", "/")
        if not rel:
            return
        try:
            full = os.path.realpath(os.path.join(base, *rel.split("/")))
        except Exception:
            return
        # Inside the root, and not the metadata itself. installed-files.txt entries are relative to the
        # egg-info dir and start with `..`, so containment is checked after resolving.
        if not full.startswith(prefix):
            return
        head = full[len(prefix) :].split(os.sep, 1)[0]
        if head == "__pycache__" or head.endswith((".dist-info", ".egg-info", ".data")):
            return
        files.add(full)

    for entry in entries:
        if not entry.endswith((".dist-info", ".egg-info")):
            continue
        meta = os.path.join(root, entry)
        listed = False
        # RECORD is the one file a wheel install always leaves behind; installed-files.txt is its egg-info equivalent.
        for record, base in (("RECORD", root), ("installed-files.txt", meta)):
            path = os.path.join(meta, record)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, encoding = "utf-8", errors = "replace") as f:
                    for line in f:
                        add(base, line.split(",", 1)[0])
                listed = True
            except Exception:
                pass
            break
        if listed:
            continue
        # No file list anywhere. A top_level.txt name is honoured only when it resolves to ONE file:
        # `dill` -> dill.py is unambiguous, `google` is a directory the metadata cannot account for.
        # Declining costs the original loud PicklingError, not a silently pinned fingerprint.
        top_level = os.path.join(meta, "top_level.txt")
        try:
            with open(top_level, encoding = "utf-8") as f:
                for line in f:
                    name = line.strip()
                    if not name or name.startswith("#"):
                        continue
                    name = name.replace("/", ".").split(".")[0]
                    # One file on disk, or it claims nothing.
                    if os.path.isfile(os.path.join(root, name + ".py")):
                        add(root, name + ".py")
        except Exception:
            continue
    return files


def _dill_module_is_importable_by_name(module, files = ()):
    """Whether pickling `module` BY REFERENCE is valid AND in scope.

    Valid exactly when an unpickler gets the same object back from
    `import <name>`, which an ordinary site-packages install already does for
    every one of these modules.

    In scope only when the module's own FILE is one an installed distribution
    recorded. `files` holds absolute resolved paths, so several off-prefix
    roots collect into one set without vouching for each other: two Lambda
    layers may each carry a `config.py`, and the paths differ. Only misplaced
    LIBRARIES move back to their ordinary-install behaviour.

    Narrow in three further ways: the module must be the live `sys.modules`
    entry under its own name, must be file-backed, and `__main__` /
    `__mp_main__` are excluded outright.
    """
    name = getattr(module, "__name__", None)
    if not name or name in _DILL_NEVER_BY_REFERENCE:
        return False
    if sys.modules.get(name) is not module:
        return False
    spec = getattr(module, "__spec__", None)
    if spec is None or getattr(spec, "name", None) != name:
        return False
    origin = getattr(spec, "origin", None)
    if not origin or not files:
        return False
    try:
        real = os.path.realpath(origin)
    except Exception:
        return False
    if real in files:
        return True
    # A deployment that compiles to adjacent bytecode and drops the sources leaves RECORD naming
    # pkg/__init__.py while the live spec points at pkg/__init__.pyc. Same installed file.
    stem, ext = os.path.splitext(real)
    return ext in (".pyc", ".pyo") and stem + ".py" in files


def fix_dill_module_by_value_pickling():
    """Let dill pickle importable modules by reference on an off-prefix install.

    No-op unless the environment is one dill would otherwise choke on, so a
    normal install keeps dill's behaviour byte for byte, fingerprints included.
    """
    if os.environ.get(_DILL_FIX_ENV, "0") in ("1", "True", "true"):
        return False
    if not _dill_environment_is_affected():
        return False
    try:
        import dill._dill as _dill_module
    except Exception:
        return False

    original = getattr(_dill_module, "_is_builtin_module", None)
    if original is None or getattr(original, _DILL_FIX_SENTINEL, False):
        return False

    # Ask dill itself before touching anything: the gate above is a copy of dill's rule and a copy can go stale.
    probe = None
    roots = []
    for name in ("datasets", "pyarrow"):
        candidate = sys.modules.get(name)
        if candidate is None:
            try:
                spec = importlib.util.find_spec(name)
            except Exception:
                spec = None
            if spec is None or not getattr(spec, "origin", None):
                continue
            candidate = importlib.util.module_from_spec(spec)
        try:
            if not original(candidate):
                probe = probe or candidate
                root = _dill_install_root(getattr(candidate, "__file__", None))
                if root and root not in roots:
                    roots.append(root)
        except Exception:
            continue
    if probe is None or not roots:
        return False

    # Every off-prefix path entry carrying installed metadata, not just the one datasets or pyarrow
    # lives in: layers can be spread, and a transform reaching a dependency in a second layer hits
    # the very failure this prevents. A root with no metadata contributes nothing. Read once, at
    # patch time: re-scanning per module would put directory listings in every fingerprint.
    for entry in list(sys.path):
        if not entry or not os.path.isdir(entry):
            continue
        if not _dill_path_pickles_by_value(os.path.join(entry, "__unsloth_probe__.py")):
            continue
        real_entry = os.path.realpath(entry)
        if real_entry not in roots:
            roots.append(real_entry)

    owned_files = set()
    for root in roots:
        owned_files |= _dill_distribution_paths(root)
    if not owned_files:
        return False

    @functools.wraps(original)
    def _is_builtin_module(
        module,
        _original = original,
        _files = frozenset(owned_files),
    ):
        try:
            if _original(module):
                return True
        except Exception:
            return False
        try:
            return _dill_module_is_importable_by_name(module, _files)
        except Exception:
            return False

    setattr(_is_builtin_module, _DILL_FIX_SENTINEL, True)
    _dill_module._is_builtin_module = _is_builtin_module
    # dill.session binds the name at import time, so patching the defining module alone leaves that
    # copy on the original.
    session = sys.modules.get("dill.session")
    if session is not None and getattr(session, "_is_builtin_module", None) is original:
        session._is_builtin_module = _is_builtin_module
    if UNSLOTH_ENABLE_LOGGING:
        logger.info(
            "Unsloth: patched dill to pickle importable modules by reference; "
            f"{getattr(probe, '__name__', '?')} is installed outside a "
            "site-packages tree."
        )
    return True


# Windows refuses sentencepiece's compiled extension on some machines. Smart App Control and
# App Control for Business judge by reputation, one file at a time, so a freshly published
# unsigned .pyd can be refused on a machine where everything else loads. The user sees a Bad
# Image dialog naming the file, and transformers keeps saying the package is available,
# because it decides that from find_spec and installed metadata and loads nothing.
#
# There is no way to ask whether this machine will refuse the file that does not involve
# handing the file to the loader, which is the thing being avoided: a probe IS the dialog. So
# on Windows the extension is simply never imported.
DISABLE_SENTENCEPIECE_VARIABLE = "UNSLOTH_DISABLE_SENTENCEPIECE"
_SENTENCEPIECE_TRUTHY = frozenset({"1", "true", "yes", "on"})
_SENTENCEPIECE_FALSY = frozenset({"0", "false", "no", "off"})


def sentencepiece_should_be_disabled():
    """Whether to make sentencepiece absent in this process.

    Windows by default; every other platform only when asked. WSL reports ``linux`` and is
    treated as the Linux box it is, since App Control does not enforce over ELF binaries in
    the guest.

    An unrecognised value falls back to the platform default rather than raising. This runs at
    the very top of the process, where a typo in an environment variable must not be fatal.
    """
    value = (os.environ.get(DISABLE_SENTENCEPIECE_VARIABLE) or "").strip().lower()
    if value in _SENTENCEPIECE_TRUTHY:
        return True
    if value in _SENTENCEPIECE_FALSY:
        return False
    return sys.platform == "win32"


def disable_sentencepiece_on_windows():
    """Make ``import sentencepiece`` fail the way an uninstalled package does.

    A ``None`` entry in ``sys.modules`` is CPython's documented sentinel for this: the import
    raises ``ModuleNotFoundError`` (an ``ImportError``, so every ``try/except ImportError`` in
    transformers already handles it) and nothing on disk is opened, so the compiled extension
    is never handed to the Windows loader and there is no dialog to see.

    Deliberately NOT a monkey patch of ``is_sentencepiece_available``. transformers derives
    that from ``find_spec``, which now finds the sentinel and answers False on its own, so the
    process is in the ordinary "sentencepiece was never installed" configuration rather than a
    state where the flag and the package disagree. That distinction is load bearing: on
    transformers 4.52 through 4.57 a flag that lies sends ``tokenizer_class_from_name`` into a
    fallback that imports the slow tokenizer module and reaches its unguarded
    ``import sentencepiece as spm``, which is the loader error this is meant to prevent.

    Must run before transformers is imported, since transformers reads availability during
    its own import. Once transformers is in sys.modules this declines rather than installing a
    sentinel it has already contradicted. Returns True only when this call is what made it
    absent.
    """
    if not sentencepiece_should_be_disabled():
        return False
    if "sentencepiece" in sys.modules:
        # Already imported by something earlier, or already disabled by an earlier call.
        # Replacing a live module here would break whoever is holding it.
        return sys.modules["sentencepiece"] is None
    if "transformers" in sys.modules:
        # Too late, and installing it anyway would be worse than doing nothing. transformers
        # has already read availability from find_spec and cached "installed", so the sentinel
        # would only produce the disagreement described above, and a tokenizer that loads
        # today would start raising ModuleNotFoundError. Whoever imported transformers first
        # keeps the ordinary behaviour.
        return False
    sys.modules["sentencepiece"] = None
    if UNSLOTH_ENABLE_LOGGING:
        logger.info(
            "Unsloth: sentencepiece is not imported on Windows, so a code integrity policy "
            "cannot refuse its extension. Models needing it will say so. Set "
            f"{DISABLE_SENTENCEPIECE_VARIABLE}=0 to import it again."
        )
    return True
