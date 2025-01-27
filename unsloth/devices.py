import sys

import torch

if sys.platform == "darwin":
    from unsloth import mac_specific


def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    else:
        return mac_specific.has_mps


def get_cuda_device_string():
    return "cuda"


def get_optimal_device_name():
    if torch.cuda.is_available():
        return get_cuda_device_string()

    if has_mps():
        return "mps"

    return "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())



def torch_gc():

    if torch.cuda.is_available():
        with torch.cuda.device(get_cuda_device_string()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    if has_mps():
        mac_specific.torch_mps_gc()






