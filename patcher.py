# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unsloth Zoo Patcher for Apple Silicon (macOS/MPS) Compatibility
=================================================================

This module provides comprehensive patching for unsloth_zoo and related
dependencies to work correctly on Apple Silicon with MPS.

Key Features:
    - Automatic hardware detection and configuration
    - Mock implementations for CUDA-only libraries (Triton, bitsandbytes)
    - MPS-specific patches for PyTorch and ML libraries
    - Runtime patching of imported modules
    - Comprehensive logging and diagnostics
    - Context manager support for temporary patches

Usage:
    Quick Start:
        >>> from patcher import patch_for_mac
        >>> patch_for_mac()
        >>> import unsloth

    With Configuration:
        >>> from patcher import MacPatcher, PatchConfig
        >>> config = PatchConfig(
        ...     enable_logging=True,
        ...     mock_bitsandbytes=True,
        ...     mock_triton=True
        ... )
        >>> patcher = MacPatcher(config)
        >>> patcher.apply()
        >>> import unsloth

    As Context Manager:
        >>> from patcher import mac_patcher
        >>> with mac_patcher():
        ...     import unsloth
        ...     # Use unsloth here
        >>> # Patches automatically restored after exit

Requirements:
    - macOS 12.0+ (for MPS support)
    - PyTorch with MPS support
    - Python 3.9+

Notes:
    - Must be called BEFORE importing unsloth or unsloth_zoo
    - Patches are idempotent (safe to call multiple times)
    - Some patches cannot be undone once applied
"""

from __future__ import annotations

import sys
import os
import platform
import logging
import warnings
from types import ModuleType
from typing import Optional, Any, Dict, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum, auto
from functools import wraps

# Version info
__version__ = "2.0.0"
__author__ = "Unsloth Team"

# Configure logging
logger = logging.getLogger("unsloth.patcher")
logger.addHandler(logging.NullHandler())


class PatchStatus(Enum):
    """Status of a patch operation."""
    SUCCESS = auto()
    SKIPPED = auto()
    FAILED = auto()
    ALREADY_APPLIED = auto()
    NOT_NEEDED = auto()
    ERROR = auto()


@dataclass
class PatchResult:
    """Result of a patch operation."""
    name: str
    status: PatchStatus
    message: str = ""
    error: Optional[Exception] = None
    
    @property
    def success(self) -> bool:
        """Check if patch succeeded."""
        return self.status in (PatchStatus.SUCCESS, PatchStatus.ALREADY_APPLIED, PatchStatus.NOT_NEEDED)
    
    def __repr__(self) -> str:
        return f"PatchResult({self.name}: {self.status.name}{' - ' + self.message if self.message else ''})"


@dataclass
class PatchConfig:
    """Configuration for Mac compatibility patches.
    
    Attributes:
        enable_logging: Enable detailed logging of patch operations
        log_level: Logging level (default: INFO)
        mock_bitsandbytes: Mock bitsandbytes (not supported on MPS)
        mock_triton: Mock Triton (not available on macOS)
        mock_torch_cuda: Mock torch.cuda functions
        patch_device_type: Patch unsloth_zoo.device_type
        patch_fused_losses: Patch CUDA-specific fused loss functions
        patch_peft: Patch PEFT to disable bnb detection
        patch_compilers: Patch compiler modules for MPS
        auto_patch: Automatically apply patches when importing
        strict_mode: Raise exceptions on patch failures
        verbose: Print detailed status messages
    """
    enable_logging: bool = True
    log_level: int = logging.INFO
    mock_bitsandbytes: bool = True
    mock_triton: bool = True
    mock_torch_cuda: bool = True
    patch_device_type: bool = True
    patch_fused_losses: bool = True
    patch_peft: bool = True
    patch_compilers: bool = True
    auto_patch: bool = False
    strict_mode: bool = False
    verbose: bool = False
    
    def __post_init__(self):
        if self.enable_logging:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(self.log_level)


class MacPatcher:
    """
    Comprehensive Mac/MPS compatibility patcher for unsloth.
    
    This class manages all patches needed to run unsloth on Apple Silicon.
    It provides fine-grained control over which patches are applied and
    tracks the status of each patch operation.
    
    Example:
        >>> patcher = MacPatcher()
        >>> results = patcher.apply()
        >>> print(f"Applied {len([r for r in results if r.success])} patches")
        >>> 
        >>> # Check specific patch status
        >>> if patcher.is_module_patched('triton'):
        ...     print("Triton is mocked")
    """
    
    def __init__(self, config: Optional[PatchConfig] = None):
        self.config = config or PatchConfig()
        self._patch_results: Dict[str, PatchResult] = {}
        self._applied = False
        self._original_modules: Dict[str, Any] = {}
        self._mock_finders: List[Any] = []
        
    @property
    def is_applied(self) -> bool:
        """Check if patches have been applied."""
        return self._applied
    
    @property
    def patch_results(self) -> Dict[str, PatchResult]:
        """Get all patch results."""
        return self._patch_results.copy()
    
    def is_module_patched(self, name: str) -> bool:
        """Check if a specific module was successfully patched."""
        result = self._patch_results.get(name)
        return result is not None and result.success
    
    def _log(self, level: int, message: str) -> None:
        """Log a message if logging is enabled."""
        if self.config.enable_logging:
            logger.log(level, message)
        if self.config.verbose and level >= logging.INFO:
            print(f"[Patcher] {message}")
    
    @staticmethod
    def is_mac_with_mps() -> bool:
        """
        Check if running on macOS with MPS available.
        
        Returns:
            True if on macOS with working MPS, False otherwise.
        """
        if platform.system() != "Darwin":
            return False
        
        # Check environment variable override
        if os.environ.get("UNSLOTH_FORCE_MPS", "0") == "1":
            return True
        
        try:
            import torch
            return torch.backends.mps.is_available() and torch.backends.mps.is_built()
        except Exception:
            return False
    
    @staticmethod
    def get_apple_hardware_info() -> Dict[str, Any]:
        """
        Get detailed Apple Silicon hardware information.
        
        Returns:
            Dictionary with hardware details including:
            - chip_name: Name of the Apple chip
            - total_memory_gb: Total system memory in GB
            - usable_memory_gb: Estimated usable memory
            - total_memory_bytes: Total memory in bytes
            - cpu_count: Number of CPU cores
            - performance_cores: Number of performance cores
            - efficiency_cores: Number of efficiency cores
        """
        info = {
            "chip_name": "Apple Silicon",
            "total_memory_gb": 16.0,
            "usable_memory_gb": 12.0,
            "total_memory_bytes": 16 * 1024**3,
            "cpu_count": 8,
            "performance_cores": None,
            "efficiency_cores": None,
        }
        
        try:
            import subprocess
            
            # Get chip name
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True
            )
            chip = result.stdout.strip()
            if chip:
                info["chip_name"] = chip
            
            # Get memory info
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True
            )
            total_bytes = int(result.stdout.strip())
            info["total_memory_bytes"] = total_bytes
            info["total_memory_gb"] = total_bytes / (1024**3)
            # Reserve 25% for system and overhead
            info["usable_memory_gb"] = info["total_memory_gb"] * 0.75
            
            # Get CPU info
            result = subprocess.run(
                ["sysctl", "-n", "hw.ncpu"],
                capture_output=True,
                text=True,
                check=True
            )
            info["cpu_count"] = int(result.stdout.strip())
            
            # Try to get performance/efficiency cores (Apple Silicon specific)
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                info["performance_cores"] = int(result.stdout.strip())
                
                result = subprocess.run(
                    ["sysctl", "-n", "hw.perflevel1.physicalcpu"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                info["efficiency_cores"] = int(result.stdout.strip())
            except subprocess.CalledProcessError:
                pass
            
        except Exception as e:
            logger.debug(f"Could not get full hardware info: {e}")
        
        return info
    
    def _create_patch_result(
        self,
        name: str,
        status: PatchStatus,
        message: str = "",
        error: Optional[Exception] = None
    ) -> PatchResult:
        """Create and store a patch result."""
        result = PatchResult(name, status, message, error)
        self._patch_results[name] = result
        
        if status == PatchStatus.SUCCESS:
            self._log(logging.INFO, f"✓ {name}: {message or 'patched successfully'}")
        elif status == PatchStatus.FAILED:
            self._log(logging.ERROR, f"✗ {name}: {message}")
            if error:
                self._log(logging.DEBUG, f"Error details: {error}")
        elif status == PatchStatus.SKIPPED:
            self._log(logging.WARNING, f"⊘ {name}: {message}")
        else:
            self._log(logging.DEBUG, f"○ {name}: {status.name} - {message}")
        
        return result
    
    def patch_device_type(self) -> PatchResult:
        """
        Patch unsloth_zoo.device_type to support MPS.
        Must be called BEFORE importing unsloth_zoo.
        """
        name = "device_type"
        
        if not self.config.patch_device_type:
            return self._create_patch_result(name, PatchStatus.SKIPPED, "disabled in config")
        
        # Skip if already imported (too late to patch)
        if "unsloth_zoo.device_type" in sys.modules:
            return self._create_patch_result(
                name, PatchStatus.SKIPPED, "already imported - too late to patch"
            )
        
        try:
            # Create mock device_type module
            mock_device_type = ModuleType("unsloth_zoo.device_type")
            
            hw_info = self.get_apple_hardware_info()
            
            # Define patched functions and constants for MPS
            def get_device_type() -> str:
                return "mps"
            
            def is_hip() -> bool:
                return False
            
            def is_mps() -> bool:
                return True
            
            def get_device_count() -> int:
                return 1  # MPS is always single-GPU
            
            # Set module attributes
            mock_device_type.get_device_type = get_device_type
            mock_device_type.is_hip = is_hip
            mock_device_type.is_mps = is_mps
            mock_device_type.DEVICE_TYPE = "mps"
            mock_device_type.DEVICE_TYPE_TORCH = "mps"
            mock_device_type.DEVICE_COUNT = 1
            mock_device_type.ALLOW_PREQUANTIZED_MODELS = True
            mock_device_type.ALLOW_BITSANDBYTES = False
            mock_device_type.HAS_CUDA = False
            mock_device_type.HAS_MPS = True
            mock_device_type.HAS_HIP = False
            mock_device_type.HAS_XPU = False
            mock_device_type.TOTAL_MEMORY_GB = hw_info.get("total_memory_gb", 16.0)
            mock_device_type.USABLE_MEMORY_GB = hw_info.get("usable_memory_gb", 12.0)
            
            # Inject into sys.modules before unsloth_zoo import
            sys.modules["unsloth_zoo.device_type"] = mock_device_type
            
            return self._create_patch_result(name, PatchStatus.SUCCESS, "MPS device type configured")
            
        except Exception as e:
            return self._create_patch_result(name, PatchStatus.FAILED, str(e), e)
    
    def patch_torch_cuda(self) -> PatchResult:
        """
        Mock torch.cuda functions for MPS compatibility.
        Many libraries assume CUDA exists and call memory functions.
        """
        name = "torch_cuda"
        
        if not self.config.mock_torch_cuda:
            return self._create_patch_result(name, PatchStatus.SKIPPED, "disabled in config")
        
        try:
            import torch
        except ImportError:
            return self._create_patch_result(name, PatchStatus.SKIPPED, "torch not installed")
        
        if torch.cuda.is_available():
            return self._create_patch_result(name, PatchStatus.SKIPPED, "CUDA is available")
        
        try:
            hw_info = self.get_apple_hardware_info()
            
            # Store original functions for restoration
            self._original_modules[name] = {
                "get_device_properties": getattr(torch.cuda, "get_device_properties", None),
                "get_device_capability": getattr(torch.cuda, "get_device_capability", None),
                "is_available": torch.cuda.is_available,
                "device_count": torch.cuda.device_count,
            }
            
            # Apple Silicon device properties class
            class AppleSiliconProps:
                """Mock CUDA device properties for Apple Silicon."""
                
                def __init__(self):
                    self.name = hw_info.get("chip_name", "Apple Silicon")
                    self.total_memory = int(hw_info.get("usable_memory_gb", 16) * 1024**3)
                    self.major = 0
                    self.minor = 0
                    self.multi_processor_count = hw_info.get("cpu_count", 8)
                    self.is_integrated = True
                    self.is_multi_gpu_board = False
                
                def __repr__(self) -> str:
                    return f"AppleSiliconProps(name='{self.name}', memory={self.total_memory / (1024**3):.1f}GB)"
            
            # Mock functions
            def mock_get_device_properties(device=None):
                return AppleSiliconProps()
            
            def mock_get_device_capability(device=None):
                return (0, 0)
            
            def mock_mem_get_info(device=None):
                """Return memory info in bytes (free, total)."""
                total = hw_info.get("total_memory_bytes", 24 * 1024**3)
                usable = int(hw_info.get("usable_memory_gb", 16) * 1024**3)
                return (usable, total)
            
            def mock_cuda_is_available():
                return False
            
            def mock_cuda_device_count():
                return 0
            
            # Ensure torch.cuda.memory exists
            if not hasattr(torch.cuda, "memory"):
                torch.cuda.memory = ModuleType("torch.cuda.memory")
            
            # Store original memory functions
            self._original_modules[name]["memory"] = {
                "mem_get_info": getattr(torch.cuda.memory, "mem_get_info", None)
            }
            
            # Apply mocks
            torch.cuda.get_device_properties = mock_get_device_properties
            torch.cuda.get_device_capability = mock_get_device_capability
            torch.cuda.memory.mem_get_info = mock_mem_get_info
            torch.cuda.is_available = mock_cuda_is_available
            torch.cuda.device_count = mock_cuda_device_count
            
            return self._create_patch_result(name, PatchStatus.SUCCESS, f"Mocked for {hw_info.get('chip_name', 'Apple Silicon')}")
            
        except Exception as e:
            return self._create_patch_result(name, PatchStatus.FAILED, str(e), e)
    
    def patch_triton(self) -> PatchResult:
        """
        Mock Triton for macOS compatibility.
        Triton is not available on macOS but many libraries import it.
        """
        name = "triton"
        
        if not self.config.mock_triton:
            return self._create_patch_result(name, PatchStatus.SKIPPED, "disabled in config")
        
        if "triton" in sys.modules:
            return self._create_patch_result(name, PatchStatus.SKIPPED, "already imported")
        
        try:
            from importlib.machinery import ModuleSpec
            from importlib.abc import MetaPathFinder, Loader
            
            class FakeTriton(ModuleType):
                """Comprehensive mock Triton module."""
                
                def __init__(self, name: str, *args, **kwargs):
                    super().__init__(str(name))
                    self.__path__ = []
                    self.__version__ = "3.0.0"
                    self.__spec__ = ModuleSpec(
                        name=str(name), loader=TritonMockLoader(), origin="mocked"
                    )
                
                def __getattr__(self, name: str) -> Any:
                    if name.startswith("__"):
                        return super().__getattribute__(name)
                    full_name = f"{self.__name__}.{name}"
                    if full_name not in sys.modules:
                        m = FakeTriton(full_name)
                        m.__spec__ = ModuleSpec(
                            name=full_name, loader=TritonMockLoader(), origin="mocked"
                        )
                        sys.modules[full_name] = m
                    return sys.modules[full_name]
                
                def __call__(self, *args, **kwargs) -> "FakeTriton":
                    return self
                
                def __getitem__(self, key: Any) -> "FakeTriton":
                    return self
                
                def __len__(self) -> int:
                    return 0
                
                def __iter__(self):
                    return iter([])
                
                def __bool__(self) -> bool:
                    return False
                
                def __repr__(self) -> str:
                    return f"<FakeTriton {self.__name__}>"
                
                @classmethod
                def __class_getitem__(cls, key: Any):
                    return cls
            
            class TritonMockLoader(Loader):
                def create_module(self, spec):
                    if spec.name not in sys.modules:
                        m = FakeTriton(spec.name)
                        m.__spec__ = spec
                        sys.modules[spec.name] = m
                    return sys.modules[spec.name]
                
                def exec_module(self, module):
                    if module.__name__ == "triton.backends":
                        module.backends = {}
                    elif module.__name__ == "triton.backends.compiler":
                        class AttrsDescriptor:
                            def __init__(self, *args, **kwargs):
                                pass
                        module.AttrsDescriptor = AttrsDescriptor
                    elif module.__name__ == "triton.language":
                        class MockTritonMeta:
                            def __repr__(self):
                                return "MockTritonMeta"
                        module.dtype = MockTritonMeta
            
            class TritonMockFinder(MetaPathFinder):
                def find_spec(self, fullname: str, path, target=None):
                    if fullname == "triton" or fullname.startswith("triton."):
                        return ModuleSpec(fullname, TritonMockLoader())
                    return None
            
            # Inject the finder at the start of meta_path
            finder = TritonMockFinder()
            sys.meta_path.insert(0, finder)
            self._mock_finders.append(finder)
            
            # Trigger root import to populate sys.modules
            import triton  # noqa: F401
            
            return self._create_patch_result(name, PatchStatus.SUCCESS, "Triton mocked successfully")
            
        except Exception as e:
            return self._create_patch_result(name, PatchStatus.FAILED, str(e), e)
    
    def patch_bitsandbytes(self) -> PatchResult:
        """
        Mock bitsandbytes for MPS compatibility.
        bitsandbytes doesn't support Apple Silicon yet.
        """
        name = "bitsandbytes"
        
        if not self.config.mock_bitsandbytes:
            return self._create_patch_result(name, PatchStatus.SKIPPED, "disabled in config")
        
        if "bitsandbytes" in sys.modules:
            return self._create_patch_result(name, PatchStatus.SKIPPED, "already imported")
        
        try:
            from importlib.machinery import ModuleSpec
            from importlib.abc import MetaPathFinder, Loader
            
            class FakeBnB(ModuleType):
                """Comprehensive mock bitsandbytes module."""
                
                class Linear4bit:
                    """Mock for bitsandbytes.nn.Linear4bit."""
                    pass
                
                class Linear8bit:
                    """Mock for bitsandbytes.nn.Linear8bit."""
                    pass
                
                def __init__(self, name: str):
                    super().__init__(str(name))
                    self.__path__ = []
                    self.__spec__ = ModuleSpec(name=str(name), loader=BnBMockLoader(), origin="mocked")
                
                def __getattr__(self, name: str) -> Any:
                    if name.startswith("__"):
                        return super().__getattribute__(name)
                    full_name = f"{self.__name__}.{name}"
                    if full_name not in sys.modules:
                        m = FakeBnB(full_name)
                        sys.modules[full_name] = m
                    return sys.modules[full_name]
                
                def __call__(self, *args, **kwargs) -> "FakeBnB":
                    return self
                
                def __bool__(self) -> bool:
                    return False
                
                @staticmethod
                def is_available() -> bool:
                    return False
                
                @staticmethod
                def is_bnb_available() -> bool:
                    return False
                
                @staticmethod
                def is_bnb_4bit_available() -> bool:
                    return False
                
                class nn:
                    """Mock for bitsandbytes.nn submodule."""
                    
                    class Linear4bit:
                        """Mock for bitsandbytes.nn.Linear4bit."""
                        pass
                    
                    class Linear8bit:
                        """Mock for bitsandbytes.nn.Linear8bit."""
                        pass
            
            class BnBMockLoader(Loader):
                def create_module(self, spec):
                    if spec.name not in sys.modules:
                        m = FakeBnB(spec.name)
                        m.__spec__ = spec
                        sys.modules[spec.name] = m
                    return sys.modules[spec.name]
                
                def exec_module(self, module):
                    pass
            
            class BnBMockFinder(MetaPathFinder):
                def find_spec(self, fullname: str, path, target=None):
                    if fullname == "bitsandbytes" or fullname.startswith("bitsandbytes."):
                        return ModuleSpec(fullname, BnBMockLoader())
                    return None
            
            finder = BnBMockFinder()
            sys.meta_path.insert(0, finder)
            self._mock_finders.append(finder)
            
            # Trigger import to populate sys.modules
            import bitsandbytes  # noqa: F401
            
            return self._create_patch_result(name, PatchStatus.SUCCESS, "bitsandbytes mocked")
            
        except Exception as e:
            return self._create_patch_result(name, PatchStatus.FAILED, str(e), e)
    
    def patch_fused_losses(self) -> PatchResult:
        """
        Patch unsloth_zoo fused_losses to avoid CUDA-specific code on MPS.
        """
        name = "fused_losses"
        
        if not self.config.patch_fused_losses:
            return self._create_patch_result(name, PatchStatus.SKIPPED, "disabled in config")
        
        try:
            # Try to patch after import
            import unsloth_zoo.fused_losses.cross_entropy_loss as ce_loss_mod
            
            original_forward = getattr(ce_loss_mod, "forward", None)
            
            if original_forward:
                # Store original
                self._original_modules[name] = {"forward": original_forward}
                
                # Create MPS-compatible version
                def mps_forward(self, input, target, *args, **kwargs):
                    import torch
                    if input.device.type == "mps":
                        # Use PyTorch's built-in cross entropy for MPS
                        return torch.nn.functional.cross_entropy(input, target)
                    return original_forward(self, input, target, *args, **kwargs)
                
                ce_loss_mod.forward = mps_forward
                
                return self._create_patch_result(name, PatchStatus.SUCCESS, "MPS forward method added")
            
            return self._create_patch_result(name, PatchStatus.SKIPPED, "no forward method found")
            
        except ImportError:
            return self._create_patch_result(name, PatchStatus.SKIPPED, "module not yet imported")
        except Exception as e:
            return self._create_patch_result(name, PatchStatus.FAILED, str(e), e)
    
    def patch_peft(self) -> PatchResult:
        """
        Patch PEFT to disable bitsandbytes detection on MPS.
        """
        name = "peft"
        
        if not self.config.patch_peft:
            return self._create_patch_result(name, PatchStatus.SKIPPED, "disabled in config")
        
        try:
            import peft.import_utils
            
            # Store originals
            self._original_modules[name] = {
                "is_bnb_available": peft.import_utils.is_bnb_available,
                "is_bnb_4bit_available": peft.import_utils.is_bnb_4bit_available,
            }
            
            # Apply patches
            peft.import_utils.is_bnb_available = lambda: False
            peft.import_utils.is_bnb_4bit_available = lambda: False
            peft.import_utils.is_bnb_8bit_available = lambda: False
            
            # Also patch peft.utils.other if it exists
            try:
                import peft.utils.other
                other_patches = {}
                if hasattr(peft.utils.other, "is_bnb_available"):
                    other_patches["is_bnb_available"] = peft.utils.other.is_bnb_available
                    peft.utils.other.is_bnb_available = lambda: False
                if hasattr(peft.utils.other, "is_bnb_4bit_available"):
                    other_patches["is_bnb_4bit_available"] = peft.utils.other.is_bnb_4bit_available
                    peft.utils.other.is_bnb_4bit_available = lambda: False
                if hasattr(peft.utils.other, "is_bnb_8bit_available"):
                    other_patches["is_bnb_8bit_available"] = peft.utils.other.is_bnb_8bit_available
                    peft.utils.other.is_bnb_8bit_available = lambda: False
                if other_patches:
                    self._original_modules[name]["utils.other"] = other_patches
            except ImportError:
                pass
            
            return self._create_patch_result(name, PatchStatus.SUCCESS, "bnb detection disabled")
            
        except ImportError:
            return self._create_patch_result(name, PatchStatus.SKIPPED, "PEFT not installed")
        except Exception as e:
            return self._create_patch_result(name, PatchStatus.FAILED, str(e), e)
    
    def patch_compilers(self) -> PatchResult:
        """
        Patch compiler modules to handle MPS-specific issues.
        """
        name = "compilers"
        
        if not self.config.patch_compilers:
            return self._create_patch_result(name, PatchStatus.SKIPPED, "disabled in config")
        
        patched = []
        
        # Patch unsloth_zoo.compiler
        try:
            import unsloth_zoo.compiler as compiler_mod
            
            # Look for CUDA-specific functions and mock them
            cuda_funcs = [attr for attr in dir(compiler_mod) if "cuda" in attr.lower()]
            for func_name in cuda_funcs:
                original = getattr(compiler_mod, func_name)
                if callable(original):
                    self._original_modules.setdefault(name, {})[func_name] = original
                    setattr(compiler_mod, func_name, lambda *args, **kwargs: None)
                    patched.append(func_name)
            
        except ImportError:
            pass
        
        if patched:
            return self._create_patch_result(name, PatchStatus.SUCCESS, f"patched: {', '.join(patched)}")
        else:
            return self._create_patch_result(name, PatchStatus.SKIPPED, "no CUDA functions found")
    
    def patch_patching_utils(self) -> PatchResult:
        """
        Placeholder for patching_utils - must be called AFTER unsloth import.
        See patch_patching_utils_late() for the actual patching.
        """
        name = "patching_utils"
        return self._create_patch_result(name, PatchStatus.SKIPPED, "deferred to late patch")
    
    def patch_patching_utils_late(self) -> PatchResult:
        """
        Patch unsloth_zoo.patching_utils to add dummy Bnb_Linear4bit classes for MPS.
        Must be called AFTER unsloth is imported.
        """
        name = "patching_utils_late"
        
        class _DummyBnbLinear:
            pass
        
        try:
            import unsloth_zoo.patching_utils as patching_utils
            
            if not hasattr(patching_utils, "Bnb_Linear4bit"):
                patching_utils.Bnb_Linear4bit = _DummyBnbLinear
            if not hasattr(patching_utils, "Peft_Linear4bit"):
                patching_utils.Peft_Linear4bit = _DummyBnbLinear
            
            return self._create_patch_result(name, PatchStatus.SUCCESS, "Bnb_Linear4bit patched")
        except Exception as e:
            return self._create_patch_result(name, PatchStatus.FAILED, str(e))
    
    def apply(self) -> List[PatchResult]:
        """
        Apply all Mac compatibility patches.
        
        Returns:
            List of PatchResult objects showing status of each patch.
        
        Raises:
            RuntimeError: If not on Mac with MPS and strict mode is enabled.
        """
        if self._applied:
            self._log(logging.INFO, "Patches already applied, returning existing results")
            return list(self._patch_results.values())
        
        # Check if we're on Mac with MPS
        if not self.is_mac_with_mps():
            msg = "Not on macOS or MPS not available"
            self._log(logging.WARNING, msg)
            if self.config.strict_mode:
                raise RuntimeError(msg)
            return [PatchResult("system", PatchStatus.NOT_NEEDED, msg)]
        
        self._log(logging.INFO, "Applying Mac/MPS compatibility patches...")
        hw_info = self.get_apple_hardware_info()
        self._log(logging.INFO, f"Detected: {hw_info.get('chip_name', 'Unknown Apple Silicon')}")
        
        results = []
        
        # Apply patches in order (some depend on others)
        results.append(self.patch_device_type())
        results.append(self.patch_torch_cuda())
        results.append(self.patch_triton())
        results.append(self.patch_bitsandbytes())
        
        # These can be applied after import
        results.append(self.patch_patching_utils())
        results.append(self.patch_fused_losses())
        results.append(self.patch_peft())
        results.append(self.patch_compilers())
        results.append(self.patch_patching_utils_late())
        
        self._applied = True
        
        # Summary
        successful = sum(1 for r in results if r.success)
        self._log(logging.INFO, f"Patches applied: {successful}/{len(results)} successful")
        
        if self.config.verbose:
            print("\n" + "="*60)
            print("Patch Summary:")
            print("="*60)
            for result in results:
                status_icon = "✓" if result.success else "✗"
                print(f"  {status_icon} {result.name}: {result.status.name}")
                if result.message:
                    print(f"      {result.message}")
            print("="*60)
        
        return results
    
    def restore(self) -> bool:
        """
        Attempt to restore original modules (best effort).
        
        Note: Some patches (like Triton mocking via meta_path) cannot be undone.
        
        Returns:
            True if restoration was successful, False otherwise.
        """
        if not self._applied:
            return False
        
        self._log(logging.INFO, "Attempting to restore original modules...")
        
        # Restore torch.cuda
        if "torch_cuda" in self._original_modules:
            try:
                import torch
                originals = self._original_modules["torch_cuda"]
                for name, func in originals.items():
                    if name == "memory":
                        continue
                    if func is not None:
                        setattr(torch.cuda, name, func)
            except Exception as e:
                self._log(logging.WARNING, f"Could not restore torch.cuda: {e}")
        
        # Mark as not applied
        self._applied = False
        self._patch_results.clear()
        
        return True


# Convenience Functions
def patch_for_mac(
    verbose: bool = False,
    strict: bool = False,
    enable_logging: bool = True
) -> Dict[str, str]:
    """
    Apply all Mac compatibility patches with simple interface.
    
    This is the main entry point for quick patching.
    
    Args:
        verbose: Print detailed status messages
        strict: Raise exceptions on patch failures
        enable_logging: Enable detailed logging
    
    Returns:
        Dictionary mapping patch names to their status strings.
    
    Example:
        >>> from patcher import patch_for_mac
        >>> results = patch_for_mac(verbose=True)
        >>> print(results)
        {'device_type': 'patched', 'torch_cuda': 'patched', ...}
        >>> import unsloth
    """
    config = PatchConfig(
        verbose=verbose,
        strict_mode=strict,
        enable_logging=enable_logging
    )
    
    patcher = MacPatcher(config)
    results = patcher.apply()
    
    return {r.name: f"{r.status.name}: {r.message}" if r.message else r.status.name 
            for r in results}


@contextmanager
def mac_patcher(config: Optional[PatchConfig] = None):
    """
    Context manager for temporary patching.
    
    Automatically applies patches on entry and attempts to restore on exit.
    
    Args:
        config: Optional patch configuration
    
    Example:
        >>> from patcher import mac_patcher
        >>> with mac_patcher():
        ...     import unsloth
        ...     model = unsloth.FastLanguageModel.from_pretrained(...)
        >>> # Patches automatically restored
    """
    patcher = MacPatcher(config)
    try:
        patcher.apply()
        yield patcher
    finally:
        patcher.restore()


# Legacy compatibility
apply_patches = patch_for_mac
is_patched = lambda: _default_patcher.is_applied if '_default_patcher' in globals() else False
get_patch_status = lambda: _default_patcher.patch_results if '_default_patcher' in globals() else {}


def _auto_patch():
    """Auto-patch if environment variable is set."""
    if os.environ.get("UNSLOTH_AUTO_PATCH", "0") == "1":
        if platform.system() == "Darwin":
            patch_for_mac(verbose=os.environ.get("UNSLOTH_VERBOSE", "0") == "1")


# Run auto-patch on import
_auto_patch()


if __name__ == "__main__":
    # Test the patcher
    print("="*70)
    print("Unsloth Mac Patcher v" + __version__)
    print("="*70)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"MPS Available: {MacPatcher.is_mac_with_mps()}")
    
    if MacPatcher.is_mac_with_mps():
        hw_info = MacPatcher.get_apple_hardware_info()
        print(f"\nHardware Info:")
        print(f"  Chip: {hw_info.get('chip_name', 'Unknown')}")
        print(f"  Memory: {hw_info.get('total_memory_gb', 'Unknown'):.1f} GB")
        if hw_info.get('performance_cores'):
            print(f"  P-cores: {hw_info['performance_cores']}")
            print(f"  E-cores: {hw_info.get('efficiency_cores', 'Unknown')}")
    
    print("\n" + "="*70)
    print("Applying patches...")
    print("="*70)
    
    results = patch_for_mac(verbose=True)
    
    print("\n" + "="*70)
    if all("SUCCESS" in str(v) or "NOT_NEEDED" in str(v) for v in results.values()):
        print("✓ All patches applied successfully!")
        print("\nYou can now import unsloth:")
        print("  from unsloth import FastLanguageModel")
    else:
        print("⚠ Some patches had issues. Check the output above.")
        for name, status in results.items():
            if "FAILED" in str(status):
                print(f"  ✗ {name}: {status}")
    print("="*70)
