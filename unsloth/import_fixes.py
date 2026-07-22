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

import os
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
from importlib.metadata import version as importlib_version
from packaging.version import Version as TrueVersion
import re
import logging
import textwrap
import warnings
import sys
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
            new_version += ".1"  # Add .1 for dev / alpha / beta / rc
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

    # Apply to stderr for FBGEMM and CUTLASS errors
    sys.stderr = HidePrintMessage(sys.stderr)
    # https://github.com/pytorch/FBGEMM/blob/d99cd96490ec4aabac2ee95b1e76ea4dcfcfa628/fbgemm_gpu/experimental/gemm/triton_gemm/utils.py#L43-L52
    sys.stderr.add_filter("TMA benchmarks will be running")
    # CUTLASS/FBGEMM MMA instruction error on SM90 vs SM100 (Blackwell) GPUs
    # https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp
    sys.stderr.add_filter("Arch conditional MMA instruction used without targeting")
    # CUTLASS arch conditional errors for various architectures
    sys.stderr.add_filter("CUTE_INVALID_CONTROL_PATH")
    # CUTLASS TMA-related errors when not targeting correct architecture
    sys.stderr.add_filter("Trying to use tma without CUTE_ARCH_TMA")
    # torchao logs a cosmetic "Skipping import of cpp extensions" WARNING on torch < 2.11. The
    # bnb-4bit / Unsloth paths don't use torchao's cpp kernels, so drop only that record rather
    # than raising the whole torchao logger to ERROR.
    logging.getLogger("torchao").addFilter(
        HideLoggingMessage("Skipping import of cpp extensions due to incompatible torch version")
    )
    # torch >= 2.11 path: torchao dlopens each prebuilt _C*.so and logs "Failed to load
    # .../_C*.so" when one can't (ABI tag mismatch in the wheel, e.g. a cp310 .so under a
    # cp312 runtime on Colab, or an arch-specific kernel the GPU lacks). It falls back to
    # non-cpp paths and Unsloth doesn't use these kernels, so drop the cosmetic record.
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

    # SWIG builtin type warnings (from bitsandbytes/triton SWIG bindings)
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


# Fix up AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
# MUST do this at the start primarily due to tensorflow causing issues
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


# Fix Xformers performance issues since 0.0.25
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


# ValueError: 'aimv2' is already used by a Transformers config, pick another name.
def fix_vllm_aimv2_issue():
    spec = importlib.util.find_spec("vllm")
    if spec is None:
        return
    vllm_version = importlib_version("vllm")
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


# vLLM >= 0.22 (PR #35024) deleted `vllm.transformers_utils.tokenizer`, but an
# older unsloth_zoo still imports it unguarded and crashes (issue #6385). Supply
# a stub via a meta path finder appended AFTER the real finders, so it only
# activates when vLLM no longer ships the module.
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
    # GuidedDecodingParmas is renamed to StructuredOutputsParams in vLLM
    # https://github.com/vllm-project/vllm/pull/22772/files
    # trl still wants to use GuidedDecodingParams. This is a temporary patch till trl updates
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
    # transformers >= 4.48's `_is_package_available(name)` returns a tuple
    # (bool, version_or_None). TRL caches that tuple in module-level
    # `_*_available` flags and the matching `is_*_available()` accessors
    # return the tuple directly. A non-empty tuple is always truthy, so
    # `if is_X_available():` fires even when X is absent, triggering an
    # unconditional `import X` that fails. The surfaced case is
    # `vllm_ascend` (blocks `from trl import GRPOConfig, GRPOTrainer`
    # outside Huawei Ascend hosts); `llm_blender`, `deepspeed`, `joblib`
    # share the same shape. Coerce every tuple-cached flag in
    # trl.import_utils to bool; the existing accessors that just return
    # the cached value then naturally yield a bool.
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
    # Ignore Environment variable `HF_TOKEN` is set
    try:
        from huggingface_hub._login import logger as huggingface_hub_logger
        huggingface_hub_logger.addFilter(HideLoggingMessage("`HF_TOKEN`"))
        del huggingface_hub_logger
    except:
        pass


def patch_ipykernel_hf_xet():
    # HF-XET == 1.1.10 and ipykernel == 7.0.0 / 7.0.1 causes issues
    # See https://github.com/huggingface/xet-core/issues/526
    # 2025-10-13T20:37:33.028737Z ERROR  Python exception updating progress:, error: PyErr { type: <class 'LookupError'>, value: LookupError(<ContextVar name='shell_parent' at 0x7535b4cebd80>), traceback: Some(<traceback object at 0x753408489f40>) }, caller: "src/progress_update.rs:313"
    # at /home/runner/work/xet-core/xet-core/error_printer/src/lib.rs:28
    if importlib.util.find_spec("hf_xet") is None:
        return
    if importlib.util.find_spec("ipykernel") is None:
        return
    if importlib.util.find_spec("huggingface_hub") is None:
        return

    ipykernel_version = Version(importlib_version("ipykernel"))
    if (
        (Version(importlib_version("hf_xet")) == Version("1.1.10"))
        and (
            (ipykernel_version == Version("7.0.0"))
            or (
                ipykernel_version == Version("7.0.1")
            )  # 7.0.1 seems to also break with LookupError: <ContextVar name='shell_parent' at 0x7a9775143ec0>
        )
    ):
        print(
            "#### Unsloth: `hf_xet==1.1.10` and `ipykernel==7.0.0` or `ipykernel==7.0.1` breaks progress bars. Using ASCII progress bars.\n"
            "#### Unsloth: To re-enable progress bars, please upgrade to `ipykernel>=7.1.0` or wait for a fix to\n"
            "https://github.com/huggingface/xet-core/issues/526"
        )
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()


def patch_trackio():
    # Set some environment variables to customize the Trackio dashboard for experiment tracking
    # See https://github.com/unslothai/notebooks/pull/110
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


def check_fbgemm_gpu_version():
    if importlib.util.find_spec("fbgemm_gpu") is None:
        return
    try:
        fbgemm_gpu_version = importlib_version("fbgemm_gpu_genai")
    except:
        return
    # We noticed some SegFault or bad alloc errors on lower versions of fbgemm_gpu.
    # Instead of raising an error, disable FBGEMM and fall back to Triton kernels.
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

    # Only patch the new variant that iterates over self.modules().
    # Ref: https://github.com/huggingface/transformers/pull/41993/files#diff-6b72b98c4c2dcfc6cc606843917733f5d858374fbc22a735ff483bbc0c1e63eaL1979-R1996
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
                # Vision models may not implement get_input_embeddings (e.g. GLM
                # V4.6 skips only `self.visual`); skip them
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

    # Install one process-wide torch.load shim that stays inert unless the calling
    # thread is inside _load_rng_state, so we gate only at the real rng load with
    # no global-swap race and no effect on other torch.load callers.
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
    # Use fullmatch so the entire local identifier must match, not just a prefix.
    # cu/rocm require a trailing digit (e.g. cu124, rocm6.3). cpu/xpu are exact.
    # Case-insensitive since some builds may use uppercase.
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

    # Known compatibility table (ground truth, takes precedence over formula).
    # See https://pytorch.org/get-started/previous-versions/
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

    required_tv_str = f"{required[0]}.{required[1]}.0"

    if tv_v >= Version(required_tv_str):
        logger.info(
            f"Unsloth: torch=={torch_version_raw} and "
            f"torchvision=={torchvision_version_raw} are compatible."
        )
        return

    # Version mismatch detected
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

    # Detect nightly/dev/alpha/beta/rc builds from the raw version string.
    # These often have version mismatches that are expected.
    _pre_tags = (".dev", "a0", "b0", "rc", "alpha", "beta", "nightly")
    is_prerelease = any(t in torch_version_raw for t in _pre_tags) or any(
        t in torchvision_version_raw for t in _pre_tags
    )

    # Only downgrade to warning for custom/source or prerelease builds.
    # Stable mismatches should fail fast to prevent runtime operator errors.
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

    # Only needed when the CompiledKernel class lacks num_ctas as a direct attr
    # but has metadata (triton >= 3.6.0 with torch < 2.10)
    _ck_cls = triton_compiler.CompiledKernel
    if hasattr(_ck_cls, "num_ctas"):
        return  # Old triton with direct attrs -- no patch needed

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

    # config.patch(...) and config.load_config(...) also assign via __setattr__, but
    # their writes are thread-local by design; a per-thread depth counter marks them
    # so they are not mirrored into the process-global default.
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
                cls = type(ctx)  # patch() builds a fresh ConfigPatch class each call
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

    # No replay of existing overrides: unsloth installs this before it sets any
    # dynamo/inductor config, so the wrapper mirrors every later assignment. Replaying
    # would also bake a still-active config.patch override into the global default.
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
    # vLLM installed? (without importing it)
    if importlib.util.find_spec("vllm") is None:
        return

    try:
        torch_version = Version(importlib_version("torch"))
        if torch_version >= Version("2.9.0"):
            return  # torch >= 2.9.0 is compatible
    except Exception:
        return  # Can't determine torch version, skip check

    # Any SM100 (Blackwell) GPU?
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

    # Any SM100 (Blackwell) GPU? Fix applies globally via env var + monkey-patch.
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

    # PDL-related modules present?
    has_utils = _spec_exists("vllm.lora.ops.triton_ops.utils")
    has_expand_op = _spec_exists("vllm.lora.ops.triton_ops.lora_expand_op")
    has_shrink_op = _spec_exists("vllm.lora.ops.triton_ops.lora_shrink_op")

    if not has_utils and not has_expand_op and not has_shrink_op:
        # Old vLLM version without PDL support - nothing to patch
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

    # Apply the PDL fix
    os.environ["TRITON_DISABLE_PDL"] = "1"

    def fake_supports_pdl(*args, **kwargs):
        return False

    patched = []
    patched_names = set()

    def _record_patch(name):
        if name not in patched_names:
            patched.append(name)
            patched_names.add(name)

    # Patch the source module (utils.py) where supports_pdl is defined. It uses
    # @lru_cache, so clear the cache to avoid stale results.
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
        # Just set the env var - vLLM might be an older version without supports_pdl
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


def disable_torchcodec_if_broken():
    """Make broken torchcodec behave as if uninstalled (#5446).

    transformers and datasets both detect torchcodec via find_spec, which
    returns True even when the native libs cannot dlopen. We flip their
    flags and seat a sys.modules sentinel so downstream imports fall through
    their existing except ImportError handlers cleanly.
    """
    try:
        import importlib.util
        if importlib.util.find_spec("torchcodec") is None:
            return  # absent or already disabled

        # RuntimeError on dlopen failure; OSError covers chained libavutil.so misses.
        from torchcodec.decoders import AudioDecoder
    except (ImportError, RuntimeError, OSError):
        # transformers: flip flag (<5) and/or rebind lru_cache'd func (>=5).
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

        # Drop half-loaded entries and seat the absence sentinel. After this,
        # import torchcodec raises ModuleNotFoundError and find_spec returns None.
        for _stale in [
            n
            for n in list(sys.modules)
            if n == "torchcodec"
            or n.startswith("torchcodec.")
            or n == "datasets.features._torchcodec"
        ]:
            sys.modules.pop(_stale, None)
        sys.modules["torchcodec"] = None


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
        # wandb is installed but broken - patch all checkers to skip it
        logger.info(
            "Unsloth: wandb is installed but broken (likely a protobuf version mismatch). "
            "Disabling wandb to prevent import errors. To fix, run: pip install --upgrade wandb"
        )
        _wandb_false = lambda: False
        # Patch transformers' is_wandb_available (used by most trl trainers)
        try:
            import transformers.integrations.integration_utils as tf_integration
            tf_integration.is_wandb_available = _wandb_false
        except (ImportError, AttributeError):
            pass
        # Patch accelerate's is_wandb_available. Patch both the source module and
        # the re-export namespace, since `from accelerate.utils import
        # is_wandb_available` reads accelerate.utils, not accelerate.utils.imports.
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
        # Set env var as additional fallback
        os.environ["WANDB_DISABLED"] = "true"


# ---------------------------------------------------------------------------
# peft 0.19.x + transformers 4.x drift
# ---------------------------------------------------------------------------
# peft 0.19.x's ``peft/utils/transformers_weight_conversion.py`` unconditionally
# imports ``transformers.conversion_mapping`` and ``transformers.core_model_loading``
# at module top. Neither submodule exists on transformers <5, so the import
# explodes with ModuleNotFoundError -- silently swallowed by the bare except
# in ``patch_peft_weight_converter_compatibility`` below. Fix: when (and only
# when) the import is broken, stub the two missing submodules with the symbols
# peft pulls at module top. The stubs are inert at runtime because peft itself
# only calls into them behind ``if is_transformers_ge_v5:`` gates.
# ---------------------------------------------------------------------------

# Stamped on stub modules so a second call is a strict no-op and so third
# parties can introspect ``__unsloth_stub__`` to detect our patch.
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


def _install_transformers_conversion_mapping_stub():
    """Stub the 3 symbols peft 0.19.x imports from this module at top level."""
    name = "transformers.conversion_mapping"
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, _UNSLOTH_STUB_SENTINEL, False):
        return existing

    mod = _make_peft_stub_module(name)

    # peft does ``.copy()`` + keyed assignment at module top; real dict suffices.
    mod._MODEL_TO_CONVERSION_PATTERN = {}

    def get_checkpoint_conversion_mapping(model_type, *args, **kwargs):
        # ``None`` = peft's "no conversion registered"; both callsites
        # early-return on it.
        return None

    def get_model_conversion_mapping(model, *args, **kwargs):
        return None

    mod.get_checkpoint_conversion_mapping = get_checkpoint_conversion_mapping
    mod.get_model_conversion_mapping = get_model_conversion_mapping

    sys.modules[name] = mod
    # Attach to parent so attribute-style access matches a real submodule.
    parent = sys.modules.get("transformers")
    if parent is not None and not hasattr(parent, "conversion_mapping"):
        try:
            parent.conversion_mapping = mod
        except Exception:
            # Frozen parent: sys.modules entry is enough for ``from ... import``.
            pass
    return mod


def _install_transformers_core_model_loading_stub():
    """Stub the 8 symbols peft 0.19.x imports from this module at top level.

    ``Concatenate`` and ``ConversionOps`` MUST be real classes (peft
    subclasses them at module top); the rest only appear in runtime
    ``isinstance`` / construction calls gated behind ``is_transformers_ge_v5``."""
    name = "transformers.core_model_loading"
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, _UNSLOTH_STUB_SENTINEL, False):
        return existing

    mod = _make_peft_stub_module(name)

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

    sys.modules[name] = mod
    parent = sys.modules.get("transformers")
    if parent is not None and not hasattr(parent, "core_model_loading"):
        try:
            parent.core_model_loading = mod
        except Exception:
            pass
    return mod


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

    # Already importable? Either we patched, or transformers is v5+.
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
        # Older Python ImportError has no `.name`; string-match instead.
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

    if not patched_any:
        # Real submodules present; failure was for some other reason.
        return False

    # Force a fresh import now that stubs are in place. Drop any cached
    # ``None`` entry first so importlib retries.
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
    # Most official ROCm wheels include a local version suffix like +rocmX.Y.
    # Some custom/source builds do not, so we fall back to runtime hints.
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
    # Try torch-adjacent ids table paths first without importing torch.
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


# ---------------------------------------------------------------------------
# bitsandbytes Windows ROCm fix: cextension.py runs get_rocm_gpu_arch()
# (bnb >= 0.47) and get_rocm_warpsize() (0.49.x) at import, shelling out to
# rocminfo / hipinfo.exe via PATH. Neither is on PATH on Windows (AMD torch
# wheels put hipInfo.exe in venv Scripts), so every import logs ERROR +
# WARNING, ROCM_GPU_ARCH becomes "unknown", and warp size defaults to 64:
# wrong on RDNA (wave 32), breaking 4-bit blocksizes and
# ALLOW_PREQUANTIZED_MODELS. Upstream fix unmerged (bitsandbytes#1969), so a
# MetaPathFinder swaps both helpers for torch-device-props-first versions
# right after bitsandbytes.cuda_specs executes, before cextension reads
# them. Must run before `import unsloth_zoo` (imports bnb on ROCm).
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
            # gcnArchName may carry feature flags, e.g. "gfx90a:sramecc+:xnack-"
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
        # Patch after the module body ran, before cextension calls it. The
        # finder stays on sys.meta_path (same lifecycle as the blockers
        # above) so importlib.reload(bitsandbytes.cuda_specs) re-patches.
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
        # Delegate to remaining finders (editable installs, frozen apps)
        # and wrap the loader that would actually be used.
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
        # Forced extension load raises the bare loader error (no "vllm._C"
        # wrapper); match any .so failure as callers feed only vLLM imports.
        if "cannot open shared object file" in message:
            return True
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return False


def _get_vllm_cuda_mismatch_message(error):
    """If the error is a CUDA version mismatch, return a helpful install message."""
    import re as _re

    checked = set()
    current = error
    wanted_cuda = None
    while current is not None and id(current) not in checked:
        checked.add(id(current))
        message = str(current)
        # Extract the CUDA version vllm was built for, e.g. "libcudart.so.12"
        match = _re.search(r"libcudart\.so\.(\d+)", message)
        if match:
            wanted_cuda = match.group(1)
            break
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    if wanted_cuda is None:
        return None

    # Detect what CUDA version is actually available on the system
    system_cuda_display = None  # Human-readable, e.g. "13.0"
    system_cuda_tag = None  # For wheel URL, e.g. "130"
    try:
        import torch
        cuda_version = torch.version.cuda  # e.g. "13.0" or "12.8"
        if cuda_version:
            system_cuda_display = cuda_version
            system_cuda_tag = cuda_version.replace(".", "")[:3]  # "130" or "128"
    except Exception:
        pass

    if system_cuda_tag is None or system_cuda_tag.startswith(wanted_cuda):
        return None  # Not a mismatch or can't determine

    try:
        vllm_version = importlib_version("vllm").split("+")[0]
    except Exception:
        vllm_version = "VLLM_VERSION"

    cpu_arch = "x86_64"
    try:
        import platform
        cpu_arch = platform.machine()
    except Exception:
        pass

    return (
        f"Unsloth: vLLM was built for CUDA {wanted_cuda} but this system has "
        f"CUDA {system_cuda_display}. Please reinstall vLLM with the correct CUDA version:\n"
        f"\n"
        f"  uv pip install https://github.com/vllm-project/vllm/releases/download/"
        f"v{vllm_version}/vllm-{vllm_version}+cu{system_cuda_tag}-cp38-abi3-"
        f"manylinux_2_35_{cpu_arch}.whl"
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


# vLLM's compiled extensions. A CUDA-major ABI break hits all of them, so
# probing the eagerly-loaded _C and its siblings reliably trips it.
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

            # Lazy vLLM lets a bare `import vllm` succeed even when an extension
            # is ABI-broken; force-load each to surface the .so failure here.
            # A missing one raises ModuleNotFoundError (skipped below).
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
        # Real opt-out: drop our seeded default (marker present); explicit
        # user values carry no marker and are kept.
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
                    # Debug mode compares gathered metadata across ranks with ==
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
                # Debug mode calls find_device(...).type on gather/broadcast inputs
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
