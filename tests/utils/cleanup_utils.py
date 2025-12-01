import gc
import logging
import os
import shutil
import torch
import sys
import warnings


def clear_memory(variables_to_clear = None, verbose = False, clear_all_caches = True):
    """
    Comprehensive memory clearing for persistent memory leaks.

    Args:
        variables_to_clear: List of variable names to clear
        verbose: Print memory status
        clear_all_caches: Clear all types of caches (recommended for memory leaks)
    """

    # Save current logging levels
    saved_log_levels = {}
    for name, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            saved_log_levels[name] = logger.level
    root_level = logging.getLogger().level

    if variables_to_clear is None:
        variables_to_clear = [
            "inputs",
            "model",
            "base_model",
            "processor",
            "tokenizer",
            "base_processor",
            "base_tokenizer",
            "trainer",
            "peft_model",
            "bnb_config",
        ]

    # 1. Clear LRU caches FIRST (very important for memory leaks)
    if clear_all_caches:
        clear_all_lru_caches(verbose)

    # 2. Delete specified variables
    g = globals()
    deleted_vars = []
    for var in variables_to_clear:
        if var in g:
            del g[var]
            deleted_vars.append(var)

    if verbose and deleted_vars:
        print(f"Deleted variables: {deleted_vars}")

    # 3. Multiple garbage collection passes (important for circular references)
    for i in range(3):
        collected = gc.collect()
        if verbose and collected > 0:
            print(f"GC pass {i+1}: collected {collected} objects")

    # 4. CUDA cleanup
    if torch.cuda.is_available():
        # Get memory before cleanup
        if verbose:
            mem_before = torch.cuda.memory_allocated() / 1024**3

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Additional CUDA cleanup for persistent leaks
        if clear_all_caches:
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

            # Clear JIT cache
            if hasattr(torch.jit, "_state") and hasattr(
                torch.jit._state, "_clear_class_state"
            ):
                torch.jit._state._clear_class_state()

            # Force another CUDA cache clear
            torch.cuda.empty_cache()

        # Final garbage collection
        gc.collect()

        if verbose:
            mem_after = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            print(
                f"GPU memory - Before: {mem_before:.2f} GB, After: {mem_after:.2f} GB"
            )
            print(f"GPU reserved memory: {mem_reserved:.2f} GB")
            if mem_before > 0:
                print(f"Memory freed: {mem_before - mem_after:.2f} GB")

    # restore original logging levels
    logging.getLogger().setLevel(root_level)
    for name, level in saved_log_levels.items():
        if name in logging.Logger.manager.loggerDict:
            logger = logging.getLogger(name)
            logger.setLevel(level)


def clear_all_lru_caches(verbose = True):
    """Clear all LRU caches in loaded modules."""
    cleared_caches = []

    # Modules to skip to avoid warnings
    skip_modules = {
        "torch.distributed",
        "torchaudio",
        "torch._C",
        "torch.distributed.reduce_op",
        "torchaudio.backend",
    }

    # Create a static list of modules to avoid RuntimeError
    modules = list(sys.modules.items())

    # Method 1: Clear caches in all loaded modules
    for module_name, module in modules:
        if module is None:
            continue

        # Skip problematic modules
        if any(module_name.startswith(skip) for skip in skip_modules):
            continue

        try:
            # Look for functions with lru_cache
            for attr_name in dir(module):
                try:
                    # Suppress warnings when checking attributes
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        warnings.simplefilter("ignore", UserWarning)
                        warnings.simplefilter("ignore", DeprecationWarning)

                    attr = getattr(module, attr_name)
                    if hasattr(attr, "cache_clear"):
                        attr.cache_clear()
                        cleared_caches.append(f"{module_name}.{attr_name}")
                except Exception:
                    continue  # Skip problematic attributes
        except Exception:
            continue  # Skip problematic modules

    # Method 2: Clear specific known caches
    known_caches = [
        "transformers.utils.hub.cached_file",
        "transformers.tokenization_utils_base.get_tokenizer",
        "torch._dynamo.utils.counters",
    ]

    for cache_path in known_caches:
        try:
            parts = cache_path.split(".")
            module = sys.modules.get(parts[0])
            if module:
                obj = module
                for part in parts[1:]:
                    obj = getattr(obj, part, None)
                    if obj is None:
                        break
                if obj and hasattr(obj, "cache_clear"):
                    obj.cache_clear()
                    cleared_caches.append(cache_path)
        except Exception:
            continue  # Skip problematic caches

    if verbose and cleared_caches:
        print(f"Cleared {len(cleared_caches)} LRU caches")


def clear_specific_lru_cache(func):
    """Clear cache for a specific function."""
    if hasattr(func, "cache_clear"):
        func.cache_clear()
        return True
    return False


# Additional utility for monitoring cache sizes
def monitor_cache_sizes():
    """Monitor LRU cache sizes across modules."""
    cache_info = []

    for module_name, module in sys.modules.items():
        if module is None:
            continue
        try:
            for attr_name in dir(module):
                try:
                    attr = getattr(module, attr_name)
                    if hasattr(attr, "cache_info"):
                        info = attr.cache_info()
                        cache_info.append(
                            {
                                "function": f"{module_name}.{attr_name}",
                                "size": info.currsize,
                                "hits": info.hits,
                                "misses": info.misses,
                            }
                        )
                except:
                    pass
        except:
            pass

    return sorted(cache_info, key = lambda x: x["size"], reverse = True)


def safe_remove_directory(path):
    try:
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)
            return True
        else:
            print(f"Path {path} is not a valid directory")
            return False
    except Exception as e:
        print(f"Failed to remove directory {path}: {e}")
        return False
