from enum import StrEnum


class QuantizationMethod(StrEnum):
    BNB = "bitsandbytes"
    GPTQ = "gptq"


def unwrap(module):
    return module.base_layer if hasattr(module, "base_layer") else module


def has_bias(module):
    module = unwrap(module)
    if (
        hasattr(module, "bias")
        and (module.bias is not None)
        and module.bias.count_nonzero() > 0
    ):
        return True
    return False


def find_layer(model, layer_type):
    layers = []
    for name, m in model.named_modules():
        if isinstance(m, layer_type):
            # print(f"{name}: {m}")
            layers.append((name, m))

    return layers
