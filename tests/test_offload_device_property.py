"""Tests _pin_device_to_decoder in vision.py: while the embedding is offloaded to RAM,
`model.device` must report the decoder device, because `inputs.to(model.device)` is the
documented idiom and CPU ids make generation build position_ids on the wrong device.
No GPU needed: a meta parameter stands in for the accelerator."""

import ast, os
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISION = os.path.join(HERE, "unsloth", "models", "vision.py")


def _load_pinner():
    src = open(VISION, encoding = "utf-8").read()
    mod = ast.parse(src)
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_pin_device_to_decoder":
            ns = {"torch": torch}
            exec(ast.get_source_segment(src, node), ns)
            return ns["_pin_device_to_decoder"]
    raise AssertionError("_pin_device_to_decoder not found in vision.py")


pin = _load_pinner()
DECODER = "meta" if not torch.cuda.is_available() else "cuda"


class _Base(nn.Module):
    """Stands in for transformers' ModuleUtilsMixin.device: first parameter wins."""

    @property
    def device(self):
        return next(self.parameters()).device


def _model(cls, decoder_device = DECODER):
    model = cls()
    model.embed_tokens = nn.Embedding(32, 8)
    model.layer = nn.Linear(8, 8, bias = False)
    model.layer.to(decoder_device)
    return model


def test_reports_decoder_device_while_offloaded():
    class _M(_Base):
        pass

    model = _model(_M)
    assert model.device.type == "cpu", "embedding must be first, or the repro is not reproduced"
    assert pin(model) is True
    assert model.device.type == DECODER, model.device


def test_untouched_model_of_same_class_is_unaffected():
    # The property lives on the class, so a second instance that never offloaded must
    # keep the stock answer.
    class _M(_Base):
        pass

    pin(_model(_M))
    assert _model(_M).device.type == "cpu"


def test_whole_model_on_cpu_falls_back():
    # model.to("cpu") after an offload: there is no accelerator parameter left to report.
    class _M(_Base):
        pass

    model = _model(_M, decoder_device = "cpu")
    pin(model)
    assert model.device.type == "cpu", model.device


def test_follows_a_later_move():
    class _M(_Base):
        pass

    model = _model(_M, decoder_device = "cpu")
    pin(model)
    model.layer.to(DECODER)
    assert model.device.type == DECODER, model.device


def test_flag_adds_no_parameters_or_state():
    class _M(_Base):
        pass

    model = _model(_M)
    before = list(model.state_dict())
    pin(model)
    assert list(model.state_dict()) == before
    assert [n for n, _ in model.named_parameters()] == ["embed_tokens.weight", "layer.weight"]


def test_repeated_pin_is_idempotent():
    class _M(_Base):
        pass

    model = _model(_M)
    pin(model)
    pin(model)
    assert model.device.type == DECODER, model.device


def test_class_without_device_property_is_left_alone():
    class _M(nn.Module):
        pass

    assert pin(_model(_M)) is False


if __name__ == "__main__":
    test_reports_decoder_device_while_offloaded()
    print("[PASS] offloaded model reports the decoder device")
    test_untouched_model_of_same_class_is_unaffected()
    print("[PASS] non-offloaded instance of the same class unaffected")
    test_whole_model_on_cpu_falls_back()
    print("[PASS] all-cpu model falls back to the stock answer")
    test_follows_a_later_move()
    print("[PASS] a later .to() move is followed")
    test_flag_adds_no_parameters_or_state()
    print("[PASS] no extra parameters or state_dict keys")
    test_repeated_pin_is_idempotent()
    print("[PASS] repeated pin is idempotent")
    test_class_without_device_property_is_left_alone()
    print("[PASS] a class with no device property is left alone")
    print("OK: model.device reports the decoder device while the embedding is offloaded")
