import gc
import logging
import os
import shutil
import torch
import sys
import warnings


def clear_memory(
    variables_to_clear = None,
    verbose = False,
    clear_all_caches = True,
):
    """Comprehensive memory clearing for persistent memory leaks."""

    # Save logging levels to restore later.
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

    # Clear LRU caches first (important for memory leaks).
    if clear_all_caches:
        clear_all_lru_caches(verbose)

    # Delete specified variables.
    g = globals()
    deleted_vars = []
    for var in variables_to_clear:
        if var in g:
            del g[var]
            deleted_vars.append(var)

    if verbose and deleted_vars:
        print(f"Deleted variables: {deleted_vars}")

    # Multiple GC passes for circular references.
    for i in range(3):
        collected = gc.collect()
        if verbose and collected > 0:
            print(f"GC pass {i+1}: collected {collected} objects")

    # CUDA cleanup
    if torch.cuda.is_available():
        if verbose:
            mem_before = torch.cuda.memory_allocated() / 1024**3

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if clear_all_caches:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

            # Clear JIT cache.
            if hasattr(torch.jit, "_state") and hasattr(
                torch.jit._state, "_clear_class_state"
            ):
                torch.jit._state._clear_class_state()

            torch.cuda.empty_cache()

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

    # Restore original logging levels.
    logging.getLogger().setLevel(root_level)
    for name, level in saved_log_levels.items():
        if name in logging.Logger.manager.loggerDict:
            logger = logging.getLogger(name)
            logger.setLevel(level)


def clear_all_lru_caches(verbose = True):
    """Clear all LRU caches in loaded modules."""
    cleared_caches = []

    # Skip these to avoid warnings.
    skip_modules = {
        "torch.distributed",
        "torchaudio",
        "torch._C",
        "torch.distributed.reduce_op",
        "torchaudio.backend",
    }

    # Static list to avoid RuntimeError during iteration.
    modules = list(sys.modules.items())

    # Clear caches in all loaded modules.
    for module_name, module in modules:
        if module is None:
            continue

        if any(module_name.startswith(skip) for skip in skip_modules):
            continue

        try:
            for attr_name in dir(module):
                try:
                    # Suppress warnings when checking attributes.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        warnings.simplefilter("ignore", UserWarning)
                        warnings.simplefilter("ignore", DeprecationWarning)

                    attr = getattr(module, attr_name)
                    if hasattr(attr, "cache_clear"):
                        attr.cache_clear()
                        cleared_caches.append(f"{module_name}.{attr_name}")
                except Exception:
                    continue
        except Exception:
            continue

    # Clear specific known caches.
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
            continue

    if verbose and cleared_caches:
        print(f"Cleared {len(cleared_caches)} LRU caches")


def clear_specific_lru_cache(func):
    """Clear cache for a specific function."""
    if hasattr(func, "cache_clear"):
        func.cache_clear()
        return True
    return False


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
