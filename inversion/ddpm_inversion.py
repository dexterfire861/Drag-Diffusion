# -------------------------------------------------------------------
# NOVEL CONTRIBUTION (part 1 of 2):
#   DDPM forward-process inversion — stores per-timestep noise maps.
#   Inspired by InstructUDrag (2024); closed-form marginal avoids
#   any UNet pass during inversion.
# -------------------------------------------------------------------

import torch
from diffusers import DDPMScheduler


def ddpm_invert(
    x0_latent: torch.Tensor,
    scheduler: DDPMScheduler,
    num_inference_steps: int = 50,
    seed: int = 42,
    device: torch.device = None,
) -> dict:
    """
    Stores one noise sample ε_t for each denoising timestep.

    x0_latent: [1, 4, 64, 64] float32, already scaled by 0.18215
    Returns: {timestep_int -> ε_t Tensor[1, 4, 64, 64]}
    """
    if device is None:
        device = x0_latent.device

    scheduler.set_timesteps(num_inference_steps)

    # MPS requires the generator to live on the same device as the tensor
    if device.type == "mps":
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    noise_maps = {}
    for t in scheduler.timesteps:
        t_int = t.item()
        eps_t = torch.randn(
            x0_latent.shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        noise_maps[t_int] = eps_t

    return noise_maps


def reconstruct_xt(
    x0_latent: torch.Tensor,
    t_int: int,
    eps_t: torch.Tensor,
    scheduler: DDPMScheduler,
    device: torch.device,
) -> torch.Tensor:
    """
    x_t = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * ε_t
    Useful for validation: reconstruct and check noise extraction round-trips.
    """
    abar_t = scheduler.alphas_cumprod[t_int].to(device=device, dtype=torch.float32)
    return abar_t.sqrt() * x0_latent + (1 - abar_t).sqrt() * eps_t
