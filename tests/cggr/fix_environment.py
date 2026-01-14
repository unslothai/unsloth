
import sys
import types
import importlib

def fix_triton_inductor():
    try:
        # Try to import the actual compiler module
        import triton.backends.compiler as compiler
        if not hasattr(compiler, 'AttrsDescriptor'):
            # Monkeypatch missing AttrsDescriptor
            compiler.AttrsDescriptor = type('AttrsDescriptor', (), {})
            print("Patched triton.backends.compiler.AttrsDescriptor")
        if not hasattr(compiler, 'BaseBackend'):
            # This shouldn't happen if it was the REAL module, but lets be safe
            pass
    except ImportError:
        # If it doesn't exist, create a dummy one that has enough for inductor
        m = types.ModuleType('triton.backends.compiler')
        m.AttrsDescriptor = type('AttrsDescriptor', (), {})
        # We need more things for the real triton import to work later
        # So we should probably not do this unless we have to.
        sys.modules['triton.backends.compiler'] = m
        print("Created dummy triton.backends.compiler")

if __name__ == "__main__":
    fix_triton_inductor()
    from unsloth import FastLanguageModel
    print("🚀 Unsloth import successful!")
