# Unsloth guard, not upstream source. The vendor directory is appended to sys.path for truststore, which would make
# this directory importable as `diffusers_minimax_h3_image`. Its modules must only be loaded by file path through
# core.inference.diffusion_minimax_h3_image._load_vendored(), so a package import fails loudly instead.
raise ImportError(
    "diffusers_minimax_h3_image is vendored source loaded by "
    "core.inference.diffusion_minimax_h3_image._load_vendored(), not an importable package"
)
