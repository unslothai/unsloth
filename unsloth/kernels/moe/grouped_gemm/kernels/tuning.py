"""
Manual tuning utils
"""

from collections import OrderedDict
from dataclasses import asdict, dataclass, fields
from itertools import product
from typing import Optional, T, OrderedDict

import pandas as pd
import torch
import triton
from triton.runtime.errors import OutOfResources

from grouped_gemm.kernels.autotuning import (
    BOOLS,
    DEFAULT_K_BLOCK_SIZES,
    DEFAULT_M_BLOCK_SIZES,
    DEFAULT_N_BLOCK_SIZES,
    DEFAULT_NUM_STAGES,
    DEFAULT_NUM_WARPS,
)


@dataclass
class DeviceProperties:
    """
    Stores device-specific properties for tuning purposes.
    
    Args:
        NUM_SM (`int`): Number of streaming multiprocessors.
        NUM_REGS (`int`): Number of registers.
        SIZE_SMEM (`int`): Size of shared memory.
        WARP_SIZE (`int`): Size of a warp.
    """
    NUM_SM: int
    NUM_REGS: int
    SIZE_SMEM: int
    WARP_SIZE: int


_DEVICE_PROPERTIES: Optional[DeviceProperties] = None


def get_device_properties() -> DeviceProperties:
    """
    Retrieves and returns the current device's properties.
    
    Returns:
        `DeviceProperties`: An object containing device-specific properties.
    """
    global _DEVICE_PROPERTIES
    if _DEVICE_PROPERTIES is None:
        properties = triton.runtime.driver.active.utils.get_device_properties(
            torch.cuda.current_device()
        )
        NUM_SM = properties["multiprocessor_count"]
        NUM_REGS = properties["max_num_regs"]
        SIZE_SMEM = properties["max_shared_mem"]
        WARP_SIZE = properties["warpSize"]
        _DEVICE_PROPERTIES = DeviceProperties(NUM_SM, NUM_REGS, SIZE_SMEM, WARP_SIZE)
    return _DEVICE_PROPERTIES


@dataclass
class KernelConfig:
    """
    Base configuration class for kernel tuning parameters.
    
    Args:
        BLOCK_SIZE_M (`int`): Block size in M dimension.
        BLOCK_SIZE_N (`int`): Block size in N dimension.
        BLOCK_SIZE_K (`int`): Block size in K dimension.
        num_warps (`int`): Number of warps.
        num_stages (`int`): Number of stages.
        flatten (`bool`): Whether to flatten the kernel.
        permute_x (`bool`): Whether to permute X dimension.
        permute_y (`bool`): Whether to permute Y dimension.
        fuse_mul_post (`bool`): Whether to fuse post multiplication.
        use_tma_store (`bool`): Whether to use TMA store.
    """
    BLOCK_SIZE_M: int   = 32

    BLOCK_SIZE_N: int   = 32

    BLOCK_SIZE_K: int   = 32

    num_warps: int      = 4

    num_stages: int     = 2

    flatten: bool       = True

    permute_x: bool     = False

    permute_y: bool     = False

    fuse_mul_post: bool = False

    use_tma_store: bool = False


    def to_string(self, include_tuning_params: bool = False, include_tma: bool = False) -> str:
        """
        Converts the kernel configuration to a string representation.
        
        Args:
            include_tuning_params (`bool`): Whether to include tuning parameters in the string.
            include_tma (`bool`): Whether to include TMA parameters in the string.
        
        Returns:
            `str`: String representation of the kernel configuration.
        """
        s = []
        if self.permute_x:
            s.append("permute_x")
        if self.permute_y:
            s.append("permute_y")
        if include_tuning_params:
            s.append(
                f"BLOCK_SIZE_M={self.BLOCK_SIZE_M},BLOCK_SIZE_N={self.BLOCK_SIZE_N},BLOCK_SIZE_K={self.BLOCK_SIZE_K},num_warps={self.num_warps},num_stages={self.num_stages},flatten={self.flatten}"
            )
        if include_tma:
            for f in fields(self):
                if f.name.startswith("use_tma_"):
                    if getattr(self, f.name):
                        s.append(f.name)
        return ",".join(s)


@dataclass
class KernelConfigForward(KernelConfig):
    """
    Configuration class for forward kernel operations.
    """
    use_tma_load_w: bool = False

    use_tma_load_x: bool = False



@dataclass
class KernelConfigBackward_dW(KernelConfig):
    """
    Configuration class for backward kernel operations with respect to W.
    """
    use_tma_load_dy: bool = False

    use_tma_load_x: bool  = False



@dataclass
class KernelConfigBackward_dX(KernelConfig):
    """
    Configuration class for backward kernel operations with respect to X.
    """
    use_tma_load_dy: bool = False

    use_tma_load_w: bool  = False



@dataclass
class KernelResult:
    """
    Stores results from kernel execution.
    
    Args:
        torch_time (`float`): Execution time for PyTorch.
        triton_time (`float`): Execution time for Triton.
        speedup (`float`): Speedup factor.
        kernel_config (`KernelConfig`): Configuration used for the kernel.
    """
    torch_time: float
    triton_time: float
    speedup: float
    kernel_config: KernelConfig

    def to_dict(self) -> OrderedDict:
        """
        Converts the kernel result to an ordered dictionary.
        
        Returns:
            `OrderedDict`: Dictionary representation of the kernel result.
        """
        return OrderedDict(
            **asdict(self.kernel_config),
            torch_time=self.torch_time,
            triton_time=self.triton_time,
            speedup=self.speedup,
        )

    @staticmethod
    def to_dataframe(
        results: list["KernelResult"], sort_by: str = "speedup", ascending: bool = False
    ) -> pd.DataFrame:
        """
        Converts a list of kernel results to a pandas DataFrame.
        
        Args:
            results (`list[KernelResult]`): List of kernel results.
            sort_by (`str`): Column to sort by.
            ascending (`bool`): Whether to sort in ascending order.
        
        Returns:
            `pd.DataFrame`: DataFrame containing the results.
        """
        df = pd.DataFrame([result.to_dict() for result in results])
        df = df.sort_values(by=sort_by, ascending=ascending)
        return df

    @staticmethod
    def to_csv(
        results: list["KernelResult"],
        sort_by: str    = "speedup",
        ascending: bool = False,
        filename: str   = "results.csv",
    ) -> None:
        """
        Saves a list of kernel results to a CSV file.
        
        Args:
            results (`list[KernelResult]`): List of kernel results.
            sort_by (`str`): Column to sort by.
            ascending (`bool`): Whether to sort in ascending order.
            filename (`str`): Name of the output file.
        """
        df = KernelResult.to_dataframe(results, sort_by, ascending)
        df.to_csv(filename, index=False)

    @staticmethod
    def print_table(
        results: list["KernelResult"],
        sort_by: str     = "speedup",
        ascending: bool  = False,
        num_results: int = 10,
    ) -> None:
        """
        Prints a table of the top kernel results.
        
        Args:
            results (`list[KernelResult]`): List of kernel results.
            sort_by (`str`): Column to sort by.
            ascending (`bool`): Whether to sort in ascending order.
            num_results (`int`): Number of results to display.
        """
        df = KernelResult.to_dataframe(results, sort_by, ascending)
        print(df.head(num_results).to_string(index=False))


def get_kernel_configs(
    BLOCK_M: list[int]        = DEFAULT_M_BLOCK_SIZES,
    BLOCK_N: list[int]        = DEFAULT_N_BLOCK_SIZES,
    BLOCK_K: list[int]        = DEFAULT_K_BLOCK_SIZES,
    num_warps: list[int]      = DEFAULT_NUM_WARPS,
    num_stages: list[int]     = DEFAULT_NUM_STAGES,
    use_tma_loads: list[bool] = BOOLS,
    fuse_permute: list[bool]  = BOOLS,
) -> tuple[list[KernelConfigForward], list[KernelConfigBackward_dW], list[KernelConfigBackward_dX]]:
    """
    Generates kernel configurations for tuning.
    
    Args:
        BLOCK_M (`list[int]`): List of block sizes for M dimension.
        BLOCK_N (`list[int]`): List of block sizes for N dimension.
        BLOCK_K (`list[int]`): List of block sizes for K dimension.
        num_warps (`list[int]`): List of warp counts.
        num_stages (`list[int]`): List of stage counts.
        use_tma_loads (`list[bool]`): List of TMA load flags.
        fuse_permute (`list[bool]`): List of permutation flags.
    
    Returns:
        `tuple[list[KernelConfigForward], list[KernelConfigBackward_dW], list[KernelConfigBackward_dX]]`: Generated kernel configurations.
    """
    kernel_configs_fwd = []
    kernel_configs_backward_dW = []
    kernel_configs_backward_dX = []
    for block_m, block_n, block_k, w, s, use_tma_load, permute in product(
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, use_tma_loads, fuse_permute
    ):
        kernel_configs_fwd.append(
            KernelConfigForward(
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=block_k,
                num_warps=w,
                num_stages=s,
                use_tma_load_x=use_tma_load,
                use_tma_load_w=use_tma_load,
                use_tma_store=False,
                permute_x=permute,
                permute_y=permute,
            )
        )
        kernel_configs_backward_dW.append(
            KernelConfigBackward_dW(
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=block_k,
                num_warps=w,
                num_stages=s,
                use_tma_load_dy=use_tma_load,
                use_tma_load_x=use_tma_load,
                use_tma_store=False,
                permute_x=permute,
                permute_y=permute,
            )
        )
        kernel_configs_backward_dX.append(
            KernelConfigBackward_dX(
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=block_k,
                num_warps=w,
                num_stages=s,
                use_tma_load_dy=use_tma_load,
                use_tma_load_w=use_tma_load,
                use_tma_store=False,
                permute_x=permute,
                permute_y=permute,
            )
        )

    kernel_configs_fwd = prune_kernel_configs_fwd(kernel_configs_fwd)
    kernel_configs_backward_dW = prune_kernel_configs_backward_dW(
        kernel_configs_backward_dW
    )
    kernel_configs_backward_dX = prune_kernel_configs_backward_dX(
        kernel_configs_backward_dX
    )
    return kernel_configs_fwd, kernel_configs_backward_dW, kernel_configs_backward_dX


def prune_kernel_configs_fwd(configs: list[KernelConfigForward]) -> list[KernelConfigForward]:
    """
    Prunes forward kernel configurations based on constraints.
    
    Args:
        configs (`list[KernelConfigForward]`): List of forward kernel configurations.
    
    Returns:
        `list[KernelConfigForward]`: Pruned list of configurations.
    """
    pruned_configs = []
    for config in configs:
        if config.use_tma_load_x and config.permute_x:
            continue
        if config.permute_x and config.permute_y:
            continue
        if config.use_tma_store and config.permute_y:
            continue
        pruned_configs.append(config)
    return pruned_configs


def prune_kernel_configs_backward_dX(configs: list[KernelConfigBackward_dX]) -> list[KernelConfigBackward_dX]:
    """
    Prunes backward kernel configurations with respect to X based on constraints.
    
    Args:
        configs (`list[KernelConfigBackward_dX]`): List of backward kernel configurations for X.
    
    Returns:
        `list[KernelConfigBackward_dX]`: Pruned list of configurations.
    """
    pruned_configs = []
    for config in configs:
        if config.use_tma_load_dy and config.permute_y:
            continue
        if config.permute_x and config.permute_y:
            continue
        if config.use_tma_store and config.permute_x:
            continue
        pruned_configs.append(config)
    return pruned_configs


def prune_kernel_configs_backward_dW(configs: list[KernelConfigBackward_dW]) -> list[KernelConfigBackward_dW]:
    """
    Prunes backward kernel configurations with respect to W based on constraints.
    
    Args:
        configs (`list[KernelConfigBackward_dW]`): List of backward kernel configurations for W.
    
    Returns:
        `list[KernelConfigBackward_dW]`: Pruned list of configurations.
    """
    pruned_configs = []
    for config in configs:
        if config.use_tma_load_dy and config.permute_y:
            continue
        if config.use_tma_load_x and config.permute_x:
            continue
        if config.permute_x and config.permute_y:
            continue
        pruned_configs.append(config)
    return pruned_configs


class TritonTuningContext:
    """
    Context manager for handling Triton kernel tuning.
    
    Args:
        kernel_config (`KernelConfig`): Configuration for the kernel.
    """
    def __init__(self, kernel_config: KernelConfig):
        self.kernel_config = kernel_config
        self.success = True

    def __enter__(self) -> TritonTuningContext:
        """
        Enters the Triton tuning context.
        
        Returns:
            `TritonTuningContext`: The context object.
        """
        # Setup code can be added here if needed
        return self

    def __exit__(self, exc_type: Optional[type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> bool:
        """
        Exits the Triton tuning context and handles exceptions.
        
        Args:
            exc_type (`type[BaseException]`, optional): Type of exception.
            exc_value (`BaseException`, optional): Exception instance.
            traceback (`TracebackType`, optional): Traceback information.
        
        Returns:
            `bool`: Whether to suppress the exception.
        """
        if exc_type is OutOfResources:
            name = exc_value.name
            required = exc_value.required
            limit = exc_value.limit
            print(
                f"Kernel config {self.kernel_config} failed: {name}, required: {required}, limit: {limit}"
            )
            self.success = False
        elif exc_type is not None:
            print(
                f"Error running Triton grouped GEMM for kernel config: {self.kernel_config}: {exc_value}"
            )
            self.success = False
        # Return False to propagate exceptions, True to suppress them
        return True