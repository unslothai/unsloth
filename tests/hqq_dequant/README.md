## HQQ Dequant Kernel

Standalone asymmetric dequant kernels for `hqq`.

Supports `hqq` [`BaseQuantConfig`](https://github.com/mobiusml/hqq/blob/aad68687e042ed628b5a655969406d501a203949/hqq/core/quantize.py#L872-L935) settings currently:
- `nbits` = `{4, 8}`
  - Quantization bits, `{1, 2, 3} bits` not yet supported 
- `axis` = `{0, 1}`
  - Axis along which weights are quantized
  - Anecdotal evidence of better accuracy with `axis=0`
  - Not all built-in `hqq` dequant implementations are available for both axis -- this kernel supports both.
- `group_size`
  - Grouping size of weights during quantization 
  - The kernel should work for any (power of 2) group sizes, but tested only for common sizes (`64`, `128`)
  - mmon sizes
- manual and `autotune` kernels (should ease interoperability with `torch.compile`)
- `quant_zero`
  - Additional quantization of the zeropoints
  - Currently only supports scalar scale / zero quantization of the zeros, which is the default setting of [`hqq.BaseQuantizeConfig`](https://github.com/mobiusml/hqq/blob/aad68687e042ed628b5a655969406d501a203949/hqq/core/quantize.py#L920-L924)
- `quant_scale`
  - Additional quantization of the scales
  - Not supported currently, as the default setting for [`hqq.BaseQuantizeConfig`](https://github.com/mobiusml/hqq/blob/aad68687e042ed628b5a655969406d501a203949/hqq/core/quantize.py#L876) is `quant_scale=False` (scales are not additionally quantized)
  
## Accuracy
See `test_hqq_dequant.py` for comprehensive tests across `dtypes`, `group_sizes`, `axis`, and other relevant params.

Run with
```
pytest -sv test_hqq_dequant.py`
```

## Performance
Please take with grain of salt, as I only benched against `HQQBackend.PYTORCH` on my laptop (RTX 3050):

```
python benchmark_hqq_dequant.py
```

| shape       | axis | group_size | nbits | dtype          | quant_scale | quant_zero | block_size | hqq(HQQBackend.PYTORCH) | triton  | speedup |
|-------------|------|------------|-------|----------------|-------------|------------|------------|-------------------------|---------|---------|
| (4096, 4096) | 1    | 64         | 4     | torch.bfloat16 | False       | False      | 32         | 15.3904                 | 2.3977  | 6.42x   |
| (4096, 4096) | 1    | 64         | 4     | torch.bfloat16 | False       | False      | 64         | 15.3313                 | 2.3957  | 6.40x   |
| (4096, 4096) | 1    | 64         | 4     | torch.bfloat16 | False       | False      | 128        | 15.3985                 | 2.3967  | 6.42x   |
| (4096, 4096) | 1    | 64         | 4     | torch.bfloat16 | False       | False      | 256        | 15.4044                 | 2.3986  | 6.42x   |
| (4096, 4096) | 1    | 64         | 4     | torch.bfloat16 | False       | False      | 512        | 15.4192                 | 2.4153  | 6.38x   |
| (4096, 4096) | 1    | 64         | 4     | torch.bfloat16 | False       | False      | 1024       | 15.4055                 | 25.1655 | 0.61x   |
| (4096, 4096) | 1    | 64         | 4     | torch.bfloat16 | False       | False      | autotune   | 15.3446                 | 2.3976  | 6.40x   |
| (4096, 4096) | 1    | 64         | 4     | torch.bfloat16 | False       | True       | 32         | 15.5533                 | 2.3839  | 6.52x   |
| (4096, 4096) | 1    | 64         | 4     | torch.bfloat16 | False       | True       | 64         | 15.6986                 | 2.3869  | 6.58x   |
| (4096, 4096) | 1    | 64         | 4     | torch.bfloat16 | False       | True       | 128        | 15.5906                 | 2.3807  | 6.55x   |
| (4096, 4096) | 1    | 64         | 4     | torch.bfloat16 | False       | True       | 256        | 15.6426                 | 2.3936  | 6.54x   |
| (4096, 4096) | 1    | 64         | 4     | torch.bfloat16 | False       | True       | 512        | 15.5842                 | 2.4072  | 6.47x   |
| (4096, 4096) | 1    | 64         | 4     | torch.bfloat16 | False       | True       | 1024       | 15.6129                 | 38.3974 | 0.41x   |
| (4096, 4096) | 1    | 64         | 4     | torch.bfloat16 | False       | True       | autotune   | 15.5552                 | 2.3805  | 6.53x   |
| (4096, 4096) | 1    | 128        | 4     | torch.bfloat16 | False       | False      | 32         | 15.3647                 | 2.3708  | 6.48x   |
| (4096, 4096) | 1    | 128        | 4     | torch.bfloat16 | False       | False      | 64         | 15.4205                 | 2.3707  | 6.50x   |
| (4096, 4096) | 1    | 128        | 4     | torch.bfloat16 | False       | False      | 128        | 15.3875                 | 2.3736  | 6.48x   |
| (4096, 4096) | 1    | 128        | 4     | torch.bfloat16 | False       | False      | 256        | 15.4178                 | 2.3885  | 6.45x   |
| (4096, 4096) | 1    | 128        | 4     | torch.bfloat16 | False       | False      | 512        | 15.3764                 | 5.5952  | 2.75x   |
| (4096, 4096) | 1    | 128        | 4     | torch.bfloat16 | False       | False      | 1024       | 15.3659                 | 28.3112 | 0.54x   |
| (4096, 4096) | 1    | 128        | 4     | torch.bfloat16 | False       | False      | autotune   | 15.3566                 | 2.3720  | 6.47x   |
| (4096, 4096) | 1    | 128        | 4     | torch.bfloat16 | False       | True       | 32         | 15.4933                 | 2.3652  | 6.55x   |
| (4096, 4096) | 1    | 128        | 4     | torch.bfloat16 | False       | True       | 64         | 15.6100                 | 2.3629  | 6.61x   |
| (4096, 4096) | 1    | 128        | 4     | torch.bfloat16 | False       | True       | 128        | 15.5169                 | 2.3707  | 6.55x   |
| (4096, 4096) | 1    | 128        | 4     | torch.bfloat16 | False       | True       | 256        | 15.5769                 | 2.3819  | 6.54x   |
| (4096, 4096) | 1    | 128        | 4     | torch.bfloat16 | False       | True       | 512        | 15.5484                 | 46.7231 | 0.33x   |
| (4096, 4096) | 1    | 128        | 4     | torch.bfloat16 | False       | True       | 1024       | 15.4976                 | 39.2632 | 0.39x   |
| (4096, 4096) | 1    | 128        | 4     | torch.bfloat16 | False       | True       | autotune   | 15.5105                 | 2.3612  | 6.57x   |
| (4096, 4096) | 0    | 64         | 4     | torch.bfloat16 | False       | False      | 32         | 17.7245                 | 2.3934  | 7.41x   |
| (4096, 4096) | 0    | 64         | 4     | torch.bfloat16 | False       | False      | 64         | 17.7356                 | 2.3985  | 7.39x   |
| (4096, 4096) | 0    | 64         | 4     | torch.bfloat16 | False       | False      | 128        | 17.7039                 | 2.3962  | 7.39x   |
| (4096, 4096) | 0    | 64         | 4     | torch.bfloat16 | False       | False      | 256        | 17.7170                 | 2.4007  | 7.38x   |
| (4096, 4096) | 0    | 64         | 4     | torch.bfloat16 | False       | False      | 512        | 17.7893                 | 2.4305  | 7.32x   |
| (4096, 4096) | 0    | 64         | 4     | torch.bfloat16 | False       | False      | 1024       | 17.7887                 | 3.4368  | 5.18x   |
| (4096, 4096) | 0    | 64         | 4     | torch.bfloat16 | False       | False      | autotune   | 17.8211                 | 2.3958  | 7.44x   |
| (4096, 4096) | 0    | 64         | 4     | torch.bfloat16 | False       | True       | 32         | 17.9001                 | 2.3820  | 7.51x   |
| (4096, 4096) | 0    | 64         | 4     | torch.bfloat16 | False       | True       | 64         | 18.0115                 | 2.3831  | 7.56x   |
| (4096, 4096) | 0    | 64         | 4     | torch.bfloat16 | False       | True       | 128        | 17.9640                 | 2.3884  | 7.52x   |
| (4096, 4096) | 0    | 64         | 4     | torch.bfloat16 | False       | True       | 256        | 17.9970                 | 2.3892  | 7.53x   |
| (4096, 4096) | 0    | 64         | 4     | torch.bfloat16 | False       | True       | 512        | 17.9618                 | 2.4060  | 7.47x   |
| (4096, 4096) | 0    | 64         | 4     | torch.bfloat16 | False       | True       | 1024       | 18.0256                 | 41.0300 | 0.44x   |
| (4096, 4096) | 0    | 64         | 4     | torch.bfloat16 | False       | True       | autotune   | 18.0029                 | 2.3838  | 7.55x   |
| (4096, 4096) | 0    | 128        | 4     | torch.bfloat16 | False       | False      | 32         | 15.3639                 | 2.3799  | 6.46x   |
| (4096, 4096) | 0    | 128        | 4     | torch.bfloat16 | False       | False      | 64         | 15.4093                 | 2.3827  | 6.47x   |
| (4096, 4096) | 0    | 128        | 4     | torch.bfloat16 | False       | False      | 128        | 15.3549                 | 2.3800  | 6.45x   |
| (4096, 4096) | 0    | 128        | 4     | torch.bfloat16 | False       | False      | 256        | 15.4489                 | 2.3996  | 6.44x   |
| (4096, 4096) | 0    | 128        | 4     | torch.bfloat16 | False       | False      | 512        | 15.3766                 | 3.7026  | 4.15x   |
| (4096, 4096) | 0    | 128        | 4     | torch.bfloat16 | False       | False      | 1024       | 15.4355                 | 26.2775 | 0.59x   |
| (4096, 4096) | 0    | 128        | 4     | torch.bfloat16 | False       | False      | autotune   | 15.3563                 | 2.3682  | 6.48x   |
| (4096, 4096) | 0    | 128        | 4     | torch.bfloat16 | False       | True       | 32         | 15.6545                 | 2.3809  | 6.58x   |
| (4096, 4096) | 0    | 128        | 4     | torch.bfloat16 | False       | True       | 64         | 15.5018                 | 2.3688  | 6.54x   |
| (4096, 4096) | 0    | 128        | 4     | torch.bfloat16 | False       | True       | 128        | 15.5865                 | 2.3731  | 6.57x   |
| (4096, 4096) | 0    | 128        | 4     | torch.bfloat16 | False       | True       | 256        | 15.5484                 | 2.3861  | 6.52x   |
| (4096, 4096) | 0    | 128        | 4     | torch.bfloat16 | False       | True       | 512        | 15.6000                 | 44.5326 | 0.35x   |
| (4096, 4096) | 0    | 128        | 4     | torch.bfloat16 | False       | True       | 1024       | 15.5037                 | 41.6425 | 0.37x   |
| (4096, 4096) | 0    | 128        | 4     | torch.bfloat16 | False       | True       | autotune   | 15.5015                 | 2.3781  | 6.52x   |


## Notes
The kernel requires `triton >= 3.0.0` which is not compatible with stable `xformers`:
- This required fixing the `triton` import `unsloth.__init__.py` per this [PR](https://github.com/unslothai/unsloth/pull/227)
- Initially tried to add the kernels under `unsloth.kernels` but `import xformers` from `unsloth.models.__init__.py` errors out due to `xformers` `triton` kernels incompatible with `triton >= 3.0.0`.
- Note that `xformers` is technically not required with the release of `torch 2.3` since `xformers.attn_bias.LowerTriangularMask` is available as `torch.nn.attention.bias`.