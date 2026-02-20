"""
Standardized MLX/MPS/Metal Benchmark
Tests all 5 variants consistently for each operation:
1. MPS (PyTorch)
2. Metal fused kernels
3. MLX.fast (if available)
4. MLX composed
5. MX.compile
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import patcher first to set up mocks
import importlib.util
import os
patcher_path = Path(__file__).parent.parent / "patcher.py"
spec = importlib.util.spec_from_file_location("patcher", patcher_path)
patcher = importlib.util.module_from_spec(spec)
sys.modules["patcher"] = patcher  # Register in sys.modules before exec
spec.loader.exec_module(patcher)

import mlx.core as mx
import mlx.nn as nn
import torch
import time
import numpy as np
from typing import Optional, Tuple, Dict, List
import importlib.util
import gc

# Import Unsloth MLX modules directly (avoid unsloth_zoo)
kernels_dir = Path(__file__).parent.parent / "unsloth" / "kernels" / "metal"

def load_module_directly(module_name: str, file_path: Path):
    """Load a Python module directly from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load pure MLX modules directly
swiglu_module = load_module_directly("swiglu_mlx", kernels_dir / "swiglu_mlx.py")
rms_module = load_module_directly("rms_layernorm_mlx", kernels_dir / "rms_layernorm_mlx.py")
geglu_module = load_module_directly("geglu_mlx", kernels_dir / "geglu_mlx.py")

swiglu_forward = swiglu_module.swiglu_forward
swiglu_backward = swiglu_module.swiglu_backward
mlx_rms_layernorm_forward = rms_module.mlx_rms_layernorm_forward
mlx_rms_layernorm_backward = rms_module.mlx_rms_layernorm_backward
mlx_geglu_exact_forward = geglu_module.mlx_geglu_exact_forward
mlx_geglu_exact_backward = geglu_module.mlx_geglu_exact_backward

class BenchmarkResult:
    """Stores benchmark results for a single variant."""
    def __init__(self, name: str, latency_ms: float, error: Optional[str] = None, 
                 vram_active_mb: Optional[float] = None, vram_peak_mb: Optional[float] = None, notes: str = ""):
        self.name = name
        self.latency_ms = latency_ms
        self.error = error
        self.vram_active_mb = vram_active_mb
        self.vram_peak_mb = vram_peak_mb
        self.notes = notes


def get_mem_stats() -> dict:
    """Get current memory stats in MB - matches comprehensive_apple_benchmark.py."""
    res = {"mps_active": 0.0, "mps_reserved": 0.0, "mlx_active": 0.0, "mlx_peak": 0.0}
    try:
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            res["mps_active"] = torch.mps.current_allocated_memory() / 1e6
            try:
                driver_mem = torch.mps.driver_allocated_memory() / 1e6
                if driver_mem > 0:
                    res["mps_reserved"] = driver_mem
                else:
                    res["mps_reserved"] = torch.mps.peak_allocated_memory() / 1e6
            except (AttributeError, Exception) as e:
                print(f"  [DEBUG] driver_allocated_memory error: {e}")
                res["mps_reserved"] = torch.mps.peak_allocated_memory() / 1e6
    except Exception:
        pass
    
    try:
        res["mlx_active"] = mx.get_active_memory() / 1e6
        res["mlx_peak"] = mx.get_peak_memory() / 1e6
    except Exception:
        pass
    
    return res


def estimate_tensor_memory(shape: tuple, dtype) -> float:
    """Estimate memory usage for a tensor in MB."""
    elements = 1
    for dim in shape:
        elements *= dim
    
    bytes_per_element = {
        torch.float16: 2,
        torch.float32: 4,
        torch.float64: 8,
        'float16': 2,
        'float32': 4,
    }.get(dtype, 4)
    
    return elements * bytes_per_element / (1024 * 1024)

class StandardizedBenchmark:
    """Standardized benchmark testing all 5 variants consistently."""
    
    def __init__(self, warmup_iters: int = 10, benchmark_iters: int = 50):
        self.warmup_iters = warmup_iters
        self.benchmark_iters = benchmark_iters
        self.device = torch.device("mps")
        
    def _benchmark_mps(self, fn, *args, **kwargs) -> Tuple[float, float, float]:
        """Benchmark PyTorch MPS implementation. Returns (latency_ms, active_mb, peak_mb)."""
        mx.synchronize()
        torch.mps.synchronize()
        
        # Warmup
        for _ in range(self.warmup_iters):
            _ = fn(*args, **kwargs)
            torch.mps.synchronize()
        
        # Reset peak stats after warmup
        torch.mps.reset_peak_memory_stats()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(self.benchmark_iters):
            _ = fn(*args, **kwargs)
            torch.mps.synchronize()
        end = time.perf_counter()
        
        torch.mps.synchronize()
        mem = get_mem_stats()
        
        latency_ms = (end - start) / self.benchmark_iters * 1000
        return latency_ms, mem["mps_active"], mem["mps_reserved"]
    
    def _benchmark_metal(self, fn, *args, **kwargs) -> Tuple[float, float, float]:
        """Benchmark Metal fused kernel. Returns (latency_ms, active_mb, peak_mb)."""
        mx.synchronize()
        
        for _ in range(self.warmup_iters):
            result = fn(*args, **kwargs)
            mx.eval(result)
            mx.synchronize()
        
        start = time.perf_counter()
        for _ in range(self.benchmark_iters):
            result = fn(*args, **kwargs)
            mx.eval(result)
            mx.synchronize()
        end = time.perf_counter()
        
        mx.synchronize()
        mem = get_mem_stats()
        
        latency_ms = (end - start) / self.benchmark_iters * 1000
        return latency_ms, mem["mlx_active"], mem["mlx_peak"]
    
    def _benchmark_mlx_fast(self, fn, *args, **kwargs) -> Tuple[float, float, float]:
        """Benchmark MLX.fast operations. Returns (latency_ms, active_mb, peak_mb)."""
        mx.synchronize()
        
        for _ in range(self.warmup_iters):
            result = fn(*args, **kwargs)
            mx.eval(result)
            mx.synchronize()
        
        start = time.perf_counter()
        for _ in range(self.benchmark_iters):
            result = fn(*args, **kwargs)
            mx.eval(result)
            mx.synchronize()
        end = time.perf_counter()
        
        mx.synchronize()
        mem = get_mem_stats()
        
        latency_ms = (end - start) / self.benchmark_iters * 1000
        return latency_ms, mem["mlx_active"], mem["mlx_peak"]
    
    def _benchmark_mlx_composed(self, fn, *args, **kwargs) -> Tuple[float, float, float]:
        """Benchmark MLX composed operations (no compile). Returns (latency_ms, active_mb, peak_mb)."""
        mx.synchronize()
        
        for _ in range(self.warmup_iters):
            result = fn(*args, **kwargs)
            mx.eval(result)
            mx.synchronize()
        
        start = time.perf_counter()
        for _ in range(self.benchmark_iters):
            result = fn(*args, **kwargs)
            mx.eval(result)
            mx.synchronize()
        end = time.perf_counter()
        
        mx.synchronize()
        mem = get_mem_stats()
        
        latency_ms = (end - start) / self.benchmark_iters * 1000
        return latency_ms, mem["mlx_active"], mem["mlx_peak"]
    
    def _benchmark_mlx_compiled(self, compiled_fn, *args, **kwargs) -> Tuple[float, float, float]:
        """Benchmark MLX compiled operations. Returns (latency_ms, active_mb, peak_mb)."""
        mx.synchronize()
        
        for _ in range(self.warmup_iters):
            result = compiled_fn(*args, **kwargs)
            mx.eval(result)
            mx.synchronize()
        
        start = time.perf_counter()
        for _ in range(self.benchmark_iters):
            result = compiled_fn(*args, **kwargs)
            mx.eval(result)
            mx.synchronize()
        end = time.perf_counter()
        
        mx.synchronize()
        mem = get_mem_stats()
        
        latency_ms = (end - start) / self.benchmark_iters * 1000
        return latency_ms, mem["mlx_active"], mem["mlx_peak"]
    
    def benchmark_swiglu(self, batch_size: int, seq_len: int, hidden_dim: int) -> Dict[str, BenchmarkResult]:
        """Benchmark SwiGLU activation - all 5 variants."""
        print(f"\n{'='*80}")
        print(f" SwiGLU Benchmark: B={batch_size}, S={seq_len}, H={hidden_dim}")
        print(f"{'='*80}")
        
        results = {}
        
        # Create test data
        torch_gate = torch.randn(batch_size, seq_len, hidden_dim, device=self.device, dtype=torch.float16)
        torch_up = torch.randn(batch_size, seq_len, hidden_dim, device=self.device, dtype=torch.float16)
        
        mlx_gate = mx.array(torch_gate.cpu().numpy())
        mlx_up = mx.array(torch_up.cpu().numpy())
        
        # 1. MPS (PyTorch)
        def mps_swiglu(g, u):
            return torch.nn.functional.silu(g) * u
        
        try:
            lat, active, peak = self._benchmark_mps(mps_swiglu, torch_gate, torch_up)
            results['MPS'] = BenchmarkResult('MPS', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" MPS (PyTorch)           | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" MPS (PyTorch)           | FAILED: {e}")
        
        # 2. Metal Fused
        try:
            lat, active, peak = self._benchmark_metal(swiglu_forward, mlx_gate, mlx_up)
            results['Metal Fused'] = BenchmarkResult('Metal Fused', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" Metal Fused             | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" Metal Fused             | FAILED: {e}")
        
        # 3. MLX.fast (use nn.silu if available)
        try:
            def mlx_fast_swiglu(g, u):
                return nn.silu(g) * u  # Optimized SiLU * up
            lat, active, peak = self._benchmark_mlx_fast(mlx_fast_swiglu, mlx_gate, mlx_up)
            results['MLX.fast'] = BenchmarkResult('MLX.fast', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" MLX.fast                | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" MLX.fast                | FAILED: {e}")
        
        # 4. MLX Composed
        try:
            def mlx_composed_swiglu(g, u):
                return mx.multiply(mx.multiply(g, mx.sigmoid(g)), u)
            lat, active, peak = self._benchmark_mlx_composed(mlx_composed_swiglu, mlx_gate, mlx_up)
            results['MLX Composed'] = BenchmarkResult('MLX Composed', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" MLX Composed            | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" MLX Composed            | FAILED: {e}")
        
        # 5. MX.compile
        try:
            def mlx_composed_swiglu(g, u):
                return mx.multiply(mx.multiply(g, mx.sigmoid(g)), u)
            compiled_swiglu = mx.compile(mlx_composed_swiglu)
            lat, active, peak = self._benchmark_mlx_compiled(compiled_swiglu, mlx_gate, mlx_up)
            results['MX.compile'] = BenchmarkResult('MX.compile', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" MX.compile              | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" MX.compile              | FAILED: {e}")
        
        # Print speedups relative to MPS
        if 'MPS' in results:
            baseline = results['MPS'].latency_ms
            print(f"\n Speedups vs MPS:")
            for name, result in results.items():
                if name != 'MPS':
                    speedup = baseline / result.latency_ms
                    print(f"   {name:20s} | {speedup:5.2f}x")
        
        return results
    
    def benchmark_rmsnorm(self, batch_size: int, seq_len: int, dim: int) -> Dict[str, BenchmarkResult]:
        """Benchmark RMS Normalization - all 5 variants."""
        print(f"\n{'='*80}")
        print(f" RMSNorm Benchmark: B={batch_size}, S={seq_len}, D={dim}")
        print(f"{'='*80}")
        
        results = {}
        
        # Create test data
        torch_x = torch.randn(batch_size, seq_len, dim, device=self.device, dtype=torch.float16)
        torch_weight = torch.randn(dim, device=self.device, dtype=torch.float16)
        
        mlx_x = mx.array(torch_x.cpu().numpy())
        mlx_weight = mx.array(torch_weight.cpu().numpy())
        eps = 1e-6
        
        # 1. MPS (PyTorch)
        def mps_rmsnorm(x, weight):
            variance = x.pow(2).mean(-1, keepdim=True)
            x_norm = x * torch.rsqrt(variance + eps)
            return weight * x_norm
        
        try:
            lat, active, peak = self._benchmark_mps(mps_rmsnorm, torch_x, torch_weight)
            results['MPS'] = BenchmarkResult('MPS', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" MPS (PyTorch)           | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" MPS (PyTorch)           | FAILED: {e}")
        
        # 2. Metal Fused
        try:
            lat, active, peak = self._benchmark_metal(mlx_rms_layernorm_forward, mlx_x, mlx_weight, eps)
            results['Metal Fused'] = BenchmarkResult('Metal Fused', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" Metal Fused             | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" Metal Fused             | FAILED: {e}")
        
        # 3. MLX.fast (mx.fast.rms_norm)
        try:
            def mlx_fast_rmsnorm(x, weight):
                return mx.fast.rms_norm(x, weight, eps)
            lat, active, peak = self._benchmark_mlx_fast(mlx_fast_rmsnorm, mlx_x, mlx_weight)
            results['MLX.fast'] = BenchmarkResult('MLX.fast', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" MLX.fast                | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" MLX.fast                | FAILED: {e}")
        
        # 4. MLX Composed
        try:
            def mlx_composed_rmsnorm(x, weight):
                var = mx.mean(mx.square(x), axis=-1, keepdims=True)
                return x * weight / mx.sqrt(var + eps)
            lat, active, peak = self._benchmark_mlx_composed(mlx_composed_rmsnorm, mlx_x, mlx_weight)
            results['MLX Composed'] = BenchmarkResult('MLX Composed', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" MLX Composed            | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" MLX Composed            | FAILED: {e}")
        
        # 5. MX.compile
        try:
            def mlx_composed_rmsnorm(x, weight):
                var = mx.mean(mx.square(x), axis=-1, keepdims=True)
                return x * weight / mx.sqrt(var + eps)
            compiled_rmsnorm = mx.compile(mlx_composed_rmsnorm)
            lat, active, peak = self._benchmark_mlx_compiled(compiled_rmsnorm, mlx_x, mlx_weight)
            results['MX.compile'] = BenchmarkResult('MX.compile', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" MX.compile              | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" MX.compile              | FAILED: {e}")
        
        # Print speedups relative to MPS
        if 'MPS' in results:
            baseline = results['MPS'].latency_ms
            print(f"\n Speedups vs MPS:")
            for name, result in results.items():
                if name != 'MPS':
                    speedup = baseline / result.latency_ms
                    print(f"   {name:20s} | {speedup:5.2f}x")
        
        return results
    
    def benchmark_mlp(self, batch_size: int, seq_len: int, dim: int, hidden_dim: int) -> Dict[str, BenchmarkResult]:
        """Benchmark MLP (gate_proj + up_proj + down_proj) - all 5 variants."""
        print(f"\n{'='*80}")
        print(f" MLP Benchmark: B={batch_size}, S={seq_len}, D={dim}, H={hidden_dim}")
        print(f"{'='*80}")
        
        results = {}
        
        # Create test data
        torch_x = torch.randn(batch_size, seq_len, dim, device=self.device, dtype=torch.float16)
        torch_gate = torch.randn(dim, hidden_dim, device=self.device, dtype=torch.float16)
        torch_up = torch.randn(dim, hidden_dim, device=self.device, dtype=torch.float16)
        torch_down = torch.randn(hidden_dim, dim, device=self.device, dtype=torch.float16)
        
        mlx_x = mx.array(torch_x.cpu().numpy())
        mlx_gate = mx.array(torch_gate.cpu().numpy())
        mlx_up = mx.array(torch_up.cpu().numpy())
        mlx_down = mx.array(torch_down.cpu().numpy())
        
        # 1. MPS (PyTorch)
        def mps_mlp(x, gate, up, down):
            g = torch.matmul(x, gate)
            u = torch.matmul(x, up)
            act = torch.nn.functional.silu(g) * u
            return torch.matmul(act, down)
        
        try:
            lat, active, peak = self._benchmark_mps(mps_mlp, torch_x, torch_gate, torch_up, torch_down)
            results['MPS'] = BenchmarkResult('MPS', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" MPS (PyTorch)           | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" MPS (PyTorch)           | FAILED: {e}")
        
        # 2. Metal Fused (using our swiglu_forward)
        try:
            def metal_mlp(x, gate, up, down):
                g = x @ gate
                u = x @ up
                act = swiglu_forward(g, u)
                return act @ down
            lat, active, peak = self._benchmark_metal(metal_mlp, mlx_x, mlx_gate, mlx_up, mlx_down)
            results['Metal Fused'] = BenchmarkResult('Metal Fused', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" Metal Fused             | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" Metal Fused             | FAILED: {e}")
        
        # 3. MLX.fast (use nn.silu)
        try:
            def mlx_fast_mlp(x, gate, up, down):
                g = x @ gate
                u = x @ up
                act = nn.silu(g) * u  # Optimized SiLU
                return act @ down
            lat, active, peak = self._benchmark_mlx_fast(mlx_fast_mlp, mlx_x, mlx_gate, mlx_up, mlx_down)
            results['MLX.fast'] = BenchmarkResult('MLX.fast', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" MLX.fast                | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" MLX.fast                | FAILED: {e}")
        
        # 4. MLX Composed
        try:
            def mlx_composed_mlp(x, gate, up, down):
                g = x @ gate
                u = x @ up
                act = mx.multiply(mx.multiply(g, mx.sigmoid(g)), u)
                return act @ down
            lat, active, peak = self._benchmark_mlx_composed(mlx_composed_mlp, mlx_x, mlx_gate, mlx_up, mlx_down)
            results['MLX Composed'] = BenchmarkResult('MLX Composed', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" MLX Composed            | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" MLX Composed            | FAILED: {e}")
        
        # 5. MX.compile
        try:
            def mlx_composed_mlp(x, gate, up, down):
                g = x @ gate
                u = x @ up
                act = mx.multiply(mx.multiply(g, mx.sigmoid(g)), u)
                return act @ down
            compiled_mlp = mx.compile(mlx_composed_mlp)
            lat, active, peak = self._benchmark_mlx_compiled(compiled_mlp, mlx_x, mlx_gate, mlx_up, mlx_down)
            results['MX.compile'] = BenchmarkResult('MX.compile', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" MX.compile              | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" MX.compile              | FAILED: {e}")
        
        # Print speedups relative to MPS
        if 'MPS' in results:
            baseline = results['MPS'].latency_ms
            print(f"\n Speedups vs MPS:")
            for name, result in results.items():
                if name != 'MPS':
                    speedup = baseline / result.latency_ms
                    print(f"   {name:20s} | {speedup:5.2f}x")
        
        return results
    
    def benchmark_geglu(self, batch_size: int, seq_len: int, dim: int, hidden_dim: int) -> Dict[str, BenchmarkResult]:
        """Benchmark GEGLU activation - all 5 variants."""
        print(f"\n{'='*80}")
        print(f" GEGLU Benchmark: B={batch_size}, S={seq_len}, D={dim}, H={hidden_dim}")
        print(f"{'='*80}")
        
        results = {}
        
        # Create test data
        torch_x = torch.randn(batch_size, seq_len, dim, device=self.device, dtype=torch.float16)
        torch_gate = torch.randn(dim, hidden_dim, device=self.device, dtype=torch.float16)
        torch_up = torch.randn(dim, hidden_dim, device=self.device, dtype=torch.float16)
        
        mlx_x = mx.array(torch_x.cpu().numpy())
        mlx_gate = mx.array(torch_gate.cpu().numpy())
        mlx_up = mx.array(torch_up.cpu().numpy())
        
        # 1. MPS (PyTorch)
        def mps_geglu(x, gate, up):
            g = torch.matmul(x, gate)
            u = torch.matmul(x, up)
            # GeGLU: GELU(gate) * up
            return torch.nn.functional.gelu(g) * u
        
        try:
            lat, active, peak = self._benchmark_mps(mps_geglu, torch_x, torch_gate, torch_up)
            results['MPS'] = BenchmarkResult('MPS', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" MPS (PyTorch)           | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" MPS (PyTorch)           | FAILED: {e}")
        
        # 2. Metal Fused
        try:
            def metal_geglu(x, gate, up):
                g = x @ gate
                u = x @ up
                return mlx_geglu_exact_forward(g, u)
            lat, active, peak = self._benchmark_metal(metal_geglu, mlx_x, mlx_gate, mlx_up)
            results['Metal Fused'] = BenchmarkResult('Metal Fused', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" Metal Fused             | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" Metal Fused             | FAILED: {e}")
        
        # 3. MLX.fast (use nn.gelu)
        try:
            def mlx_fast_geglu(x, gate, up):
                g = x @ gate
                u = x @ up
                return nn.gelu(g) * u  # Optimized GELU
            lat, active, peak = self._benchmark_mlx_fast(mlx_fast_geglu, mlx_x, mlx_gate, mlx_up)
            results['MLX.fast'] = BenchmarkResult('MLX.fast', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" MLX.fast                | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" MLX.fast                | FAILED: {e}")
        
        # 4. MLX Composed
        try:
            def mlx_composed_geglu(x, gate, up):
                g = x @ gate
                u = x @ up
                return nn.gelu(g) * u
            lat, active, peak = self._benchmark_mlx_composed(mlx_composed_geglu, mlx_x, mlx_gate, mlx_up)
            results['MLX Composed'] = BenchmarkResult('MLX Composed', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" MLX Composed            | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" MLX Composed            | FAILED: {e}")
        
        # 5. MX.compile
        try:
            def mlx_composed_geglu(x, gate, up):
                g = x @ gate
                u = x @ up
                return nn.gelu(g) * u
            compiled_geglu = mx.compile(mlx_composed_geglu)
            lat, active, peak = self._benchmark_mlx_compiled(compiled_geglu, mlx_x, mlx_gate, mlx_up)
            results['MX.compile'] = BenchmarkResult('MX.compile', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" MX.compile              | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" MX.compile              | FAILED: {e}")
        
        # Print speedups relative to MPS
        if 'MPS' in results:
            baseline = results['MPS'].latency_ms
            print(f"\n Speedups vs MPS:")
            for name, result in results.items():
                if name != 'MPS':
                    speedup = baseline / result.latency_ms
                    print(f"   {name:20s} | {speedup:5.2f}x")
        
        return results
    
    def benchmark_rope(self, batch_size: int, seq_len: int, num_heads: int, head_dim: int) -> Dict[str, BenchmarkResult]:
        """Benchmark RoPE (Rotary Position Embedding) - all 5 variants."""
        print(f"\n{'='*80}")
        print(f" RoPE Benchmark: B={batch_size}, S={seq_len}, H={num_heads}, D={head_dim}")
        print(f"{'='*80}")
        
        results = {}
        
        # Create test data
        torch_x = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device, dtype=torch.float16)
        
        mlx_x = mx.array(torch_x.cpu().numpy())
        
        # RoPE parameters
        inv_freq = 1.0 / (10000 ** (mx.arange(0, head_dim, 2) / head_dim))
        
        # 1. MPS (PyTorch)
        def mps_rope(x):
            # Simplified RoPE implementation
            seq_len = x.shape[2]
            t = torch.arange(seq_len, device=x.device, dtype=x.dtype)
            freqs = torch.outer(t, torch.tensor(inv_freq.tolist(), device=x.device, dtype=x.dtype))
            emb = torch.cat([freqs, freqs], dim=-1)
            cos_emb = torch.cos(emb)
            sin_emb = torch.sin(emb)
            
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            rotated = torch.cat([-x2, x1], dim=-1)
            return x * cos_emb + rotated * sin_emb
        
        try:
            lat, active, peak = self._benchmark_mps(mps_rope, torch_x)
            results['MPS'] = BenchmarkResult('MPS', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" MPS (PyTorch)           | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" MPS (PyTorch)           | FAILED: {e}")
        
        # 2. Metal Fused - use mx.fast.rope if available
        try:
            # mx.fast.rope signature: rope(a, dims, *, traditional, base, scale, offset, freqs)
            # dims is the number of dimensions to rotate (head_dim)
            
            if hasattr(mx.fast, 'rope'):
                def metal_rope(x_mlx):
                    return mx.fast.rope(x_mlx, head_dim, traditional=True, base=10000.0, scale=1.0, offset=0)
                lat, active, peak = self._benchmark_metal(metal_rope, mlx_x)
                results['Metal Fused'] = BenchmarkResult('Metal Fused', lat, vram_active_mb=active, vram_peak_mb=peak)
                print(f" Metal Fused             | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
            else:
                print(f" Metal Fused             | SKIPPED: mx.fast.rope not available")
        except Exception as e:
            print(f" Metal Fused             | FAILED: {e}")
        
        # 3. MLX.fast (if available)
        try:
            # MLX doesn't have built-in RoPE, use composed
            def mlx_fast_rope(x):
                seq_len = x.shape[2]
                t = mx.arange(seq_len)
                freqs = mx.outer(t, inv_freq)
                emb = mx.concatenate([freqs, freqs], axis=-1)
                cos_emb = mx.cos(emb)
                sin_emb = mx.sin(emb)
                
                x1 = x[..., ::2]
                x2 = x[..., 1::2]
                rotated = mx.concatenate([-x2, x1], axis=-1)
                return x * cos_emb + rotated * sin_emb
            lat, active, peak = self._benchmark_mlx_fast(mlx_fast_rope, mlx_x)
            results['MLX.fast'] = BenchmarkResult('MLX.fast', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" MLX.fast                | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" MLX.fast                | FAILED: {e}")
        
        # 4. MLX Composed
        try:
            def mlx_composed_rope(x):
                seq_len = x.shape[2]
                t = mx.arange(seq_len)
                freqs = mx.outer(t, inv_freq)
                emb = mx.concatenate([freqs, freqs], axis=-1)
                cos_emb = mx.cos(emb)
                sin_emb = mx.sin(emb)
                
                x1 = x[..., ::2]
                x2 = x[..., 1::2]
                rotated = mx.concatenate([-x2, x1], axis=-1)
                return x * cos_emb + rotated * sin_emb
            lat, active, peak = self._benchmark_mlx_composed(mlx_composed_rope, mlx_x)
            results['MLX Composed'] = BenchmarkResult('MLX Composed', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" MLX Composed            | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" MLX Composed            | FAILED: {e}")
        
        # 5. MX.compile
        try:
            def mlx_composed_rope(x):
                seq_len = x.shape[2]
                t = mx.arange(seq_len)
                freqs = mx.outer(t, inv_freq)
                emb = mx.concatenate([freqs, freqs], axis=-1)
                cos_emb = mx.cos(emb)
                sin_emb = mx.sin(emb)
                
                x1 = x[..., ::2]
                x2 = x[..., 1::2]
                rotated = mx.concatenate([-x2, x1], axis=-1)
                return x * cos_emb + rotated * sin_emb
            compiled_rope = mx.compile(mlx_composed_rope)
            lat, active, peak = self._benchmark_mlx_compiled(compiled_rope, mlx_x)
            results['MX.compile'] = BenchmarkResult('MX.compile', lat, vram_active_mb=active, vram_peak_mb=peak)
            print(f" MX.compile              | {lat:8.3f} ms | {active:5.1f}/{peak:5.1f} MB")
        except Exception as e:
            print(f" MX.compile              | FAILED: {e}")
        
        # Print speedups relative to MPS
        if 'MPS' in results:
            baseline = results['MPS'].latency_ms
            print(f"\n Speedups vs MPS:")
            for name, result in results.items():
                if name != 'MPS':
                    speedup = baseline / result.latency_ms
                    print(f"   {name:20s} | {speedup:5.2f}x")
        
        return results


def main():
    """Run standardized benchmarks."""
    print("="*80)
    print(" Standardized MLX/MPS/Metal Benchmark")
    print(" Testing all 5 variants for each operation")
    print("="*80)
    
    benchmark = StandardizedBenchmark(warmup_iters=10, benchmark_iters=50)
    
    # Test configurations
    configs = [
        # (batch_size, seq_len, dim, hidden_dim)
        (1, 1, 4096, 14336),     # Single token
        (8, 512, 4096, 14336),   # Batch training
    ]
    
    all_results = {}
    
    for batch_size, seq_len, dim, hidden_dim in configs:
        print(f"\n\n{'#'*80}")
        print(f"# Configuration: B={batch_size}, S={seq_len}, D={dim}, H={hidden_dim}")
        print(f"{'#'*80}")
        
        # SwiGLU
        results = benchmark.benchmark_swiglu(batch_size, seq_len, hidden_dim)
        all_results[f'swiglu_b{batch_size}'] = results
        
        # RMSNorm
        results = benchmark.benchmark_rmsnorm(batch_size, seq_len, dim)
        all_results[f'rmsnorm_b{batch_size}'] = results
        
        # MLP
        results = benchmark.benchmark_mlp(batch_size, seq_len, dim, hidden_dim)
        all_results[f'mlp_b{batch_size}'] = results
        
        # GEGLU
        results = benchmark.benchmark_geglu(batch_size, seq_len, dim, hidden_dim)
        all_results[f'geglu_b{batch_size}'] = results
        
        # RoPE (for attention)
        results = benchmark.benchmark_rope(batch_size, seq_len, 32, 128)
        all_results[f'rope_b{batch_size}'] = results
    
    # Summary
    print("\n\n" + "="*80)
    print(" SUMMARY")
    print("="*80)
    
    for test_name, results in all_results.items():
        print(f"\n{test_name}:")
        if 'MPS' in results:
            baseline = results['MPS'].latency_ms
            for variant, result in results.items():
                speedup = baseline / result.latency_ms if result.latency_ms > 0 else 0
                if result.vram_active_mb is not None and result.vram_peak_mb is not None:
                    mem_str = f"{result.vram_active_mb:5.1f}/{result.vram_peak_mb:5.1f} MB"
                else:
                    mem_str = "     N/A"
                print(f"  {variant:20s}: {result.latency_ms:7.3f} ms | {mem_str} ({speedup:5.2f}x vs MPS)")


if __name__ == "__main__":
    main()
