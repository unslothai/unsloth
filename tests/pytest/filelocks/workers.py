from __future__ import annotations


def load_from_pretrained(model_name: str, load_in_4bit: bool = False) -> None:
    """
    Load a small model via Unsloth. Intentionally minimal: we don't do generation;
    the goal is to hit the file-locking code path and then exit.
    """
    from unsloth import FastLanguageModel

    # Keep it tiny to avoid memory pressure across many processes
    # (Unsloth accepts these kwargs; extras are ignored by HF for tiny-random models).
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=True,
        max_seq_length=32,
        dtype=None,
        # device_map: let HF decide; tiny CPU loads are fine; 4bit needs CUDA, the test will gate it.
    )

    # Touch tokenizer to ensure both artifacts load OK.
    _ = tokenizer("hi", return_tensors="pt")
    # Clean up quickly
    del model, tokenizer
    print("load_from_pretrained: done")


def import_unsloth() -> None:
    import unsloth  # noqa: F401
    print("import_unsloth: done")
    # Print version so parent can see something on stdout if it wants.
    try:
        import inspect
        print(getattr(unsloth, "__version__", "unknown"))
    except Exception:
        pass
