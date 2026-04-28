import torch
from PIL import Image
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from utils.image_utils import get_dtype, pil_to_tensor, tensor_to_pil, encode_image, decode_latent, create_composite
from utils.mask_utils import prepare_latent_mask, gaussian_blur_mask
from inversion.ddpm_inversion import ddpm_invert
from noise_shift.noise_shift import shift_all_noise_maps


MODEL_ID = "stabilityai/stable-diffusion-2-1"


class ObjectRelocationPipeline:
    """
    Texture-preserving object relocation via DDPM noise prior shift.

    Pipeline:
      1. Pixel-space copy-paste → composite image (object at target, source filled)
      2. DDPM inversion of original → noise maps
      3. (ours) Shift noise maps source→target (InstructUDrag Eq. 4)
      4. SDEdit: add noise to composite at t_start, denoise with shifted noise maps

    Ablation: use_noise_shift=False uses fresh randn instead of shifted maps — same
    composite starting point, different noise prior. Lower perceptual distance = better
    texture preservation.

    Novel contribution: DDPM inversion + noise shift (InstructUDrag 2024, Eq. 4).
    Base model: SD 2.1 (Rombach et al., 2022).
    """

    def __init__(self, model_id: str = MODEL_ID, device: torch.device = None,
                 local_files_only: bool = True):
        if device is None:
            from utils.image_utils import get_device
            device = get_device()
        self.device = device
        dtype = get_dtype(device)

        print(f"Loading SD 2.1 on {device} ({dtype})...")
        kw = dict(local_files_only=local_files_only)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", **kw).to(device=device, dtype=dtype)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", **kw)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", **kw).to(device=device, dtype=dtype)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", **kw).to(device=device, dtype=dtype)
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler", **kw)
        self.prediction_type = self.scheduler.config.prediction_type
        # Cap at 512 on MPS to avoid OOM; Colab (CUDA) uses the native 768
        native_size = self.unet.config.sample_size * 8
        self.image_size = 512 if device.type == "mps" else native_size
        self.latent_size = self.image_size // 8
        print(f"All components loaded. Prediction type: {self.prediction_type}, image size: {self.image_size}")

    @torch.no_grad()
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return self.text_encoder(tokens)[0].float()

    def _ddpm_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev_int: int,
        eps_pred: torch.Tensor,
        stored_noise: torch.Tensor,
    ) -> torch.Tensor:
        """DDPM reverse step, injecting stored (optionally shifted) noise."""
        t_int = t.item()
        abar_t = self.scheduler.alphas_cumprod[t_int].to(device=self.device, dtype=torch.float32)

        if t_prev_int >= 0:
            abar_prev = self.scheduler.alphas_cumprod[t_prev_int].to(device=self.device, dtype=torch.float32)
        else:
            abar_prev = torch.tensor(1.0, device=self.device)

        # SD 2.1 v-prediction → convert to noise prediction
        if self.prediction_type == "v_prediction":
            eps_pred = abar_t.sqrt() * eps_pred + (1 - abar_t).sqrt() * x_t

        pred_x0 = (x_t - (1 - abar_t).sqrt() * eps_pred) / abar_t.sqrt()
        pred_x0 = pred_x0.clamp(-4.0, 4.0)

        coeff1 = abar_prev.sqrt() * (1 - abar_t / abar_prev) / (1 - abar_t)
        coeff2 = (abar_t / abar_prev).sqrt() * (1 - abar_prev) / (1 - abar_t)
        mu = coeff1 * pred_x0 + coeff2 * x_t

        if t_prev_int < 0 or t_prev_int == 0:
            return mu

        beta_t = 1 - abar_t / abar_prev
        sigma_t = (beta_t * (1 - abar_prev) / (1 - abar_t)).sqrt()
        return mu + sigma_t * stored_noise

    def __call__(
        self,
        image: Image.Image,
        prompt: str,
        source_mask: Image.Image,
        target_mask: Image.Image,
        use_noise_shift: bool = True,
        seed: int = 42,
        num_inference_steps: int = 50,
        sdedit_strength: float = 0.7,
        guidance_scale: float = 7.5,
        feather_sigma: float = 2.0,
    ) -> Image.Image:
        """
        Move the object defined by source_mask to target_mask.

        Pipeline:
          1. Pixel copy-paste → composite (establishes WHERE the object goes)
          2. SDEdit from composite (denoises seams)
          3. RePaint background lock (preserves background exactly, blends seams)
          4. DDPM noise shift (our contribution — preserves object texture during blend)

        sdedit_strength: fraction of timesteps to denoise. 0.7 = good seam blending.
            Lower = more faithful to composite; higher = more model creativity.
        use_noise_shift=True  → ours; False → baseline for ablation.
        """
        device = self.device
        sz = self.image_size

        # 1. Pixel-space copy-paste
        composite = create_composite(
            image.resize((sz, sz)),
            source_mask.resize((sz, sz)),
            target_mask.resize((sz, sz)),
        )

        # 2. Encode original + composite
        vae_dtype = next(self.vae.parameters()).dtype
        x0_orig = encode_image(self.vae, pil_to_tensor(image.resize((sz, sz)), device).to(vae_dtype))
        x0_composite = encode_image(self.vae, pil_to_tensor(composite, device).to(vae_dtype))

        # 3. Encode prompts for CFG
        encoder_hs = self._encode_prompt(prompt)
        uncond_hs = self._encode_prompt("")

        # 4. Prepare latent masks
        M_src = prepare_latent_mask(source_mask.resize((sz, sz)), device, self.latent_size)
        M_tgt = prepare_latent_mask(target_mask.resize((sz, sz)), device, self.latent_size)

        # RePaint background mask
        M_src_soft = gaussian_blur_mask(M_src, sigma=2.0)
        M_tgt_soft = gaussian_blur_mask(M_tgt, sigma=2.0)
        bg_mask = (1.0 - M_src_soft.clamp(0, 1)) * (1.0 - M_tgt_soft.clamp(0, 1))  # [1,1,H,W]

        # Composite lock — locks the target region to the copy-pasted composite,
        # preserving the pasted object pixels throughout the main denoising pass.
        # Tighter sigma than bg_mask so the seam boundary stays partially free.
        M_tgt_lock = gaussian_blur_mask(M_tgt, sigma=2.0)

        # 5. DDPM inversion of original — keep unshifted copy for background blend
        noise_maps_orig = ddpm_invert(x0_orig, self.scheduler, num_inference_steps, seed, device)

        # 6. Optionally shift noise maps (our contribution — InstructUDrag Eq. 4)
        if use_noise_shift:
            noise_maps = shift_all_noise_maps(noise_maps_orig, M_src, M_tgt, device, feather_sigma)
        else:
            noise_maps = noise_maps_orig  # baseline: same noise, no shift

        # 7. SDEdit start: add noise to composite at the chosen timestep
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        start_idx = int((1.0 - sdedit_strength) * len(timesteps))
        start_idx = max(0, min(start_idx, len(timesteps) - 1))
        t_start_int = timesteps[start_idx].item()

        abar_start = self.scheduler.alphas_cumprod[t_start_int].to(device=device, dtype=torch.float32)
        start_noise = noise_maps.get(t_start_int, torch.randn_like(x0_composite))
        x_t = abar_start.sqrt() * x0_composite + (1 - abar_start).sqrt() * start_noise

        # 8. Denoising loop with RePaint background locking
        unet_dtype = next(self.unet.parameters()).dtype
        active_timesteps = timesteps[start_idx:]
        lock_cutoff = len(active_timesteps) // 2

        for i, t in enumerate(active_timesteps):
            t_global_idx = start_idx + i
            t_prev_int = timesteps[t_global_idx + 1].item() if t_global_idx + 1 < len(timesteps) else -1

            with torch.no_grad():
                x_t_input = x_t.to(unet_dtype)
                t_batch = t.unsqueeze(0).to(device)
                latent_batch = torch.cat([x_t_input, x_t_input], dim=0)
                cond_batch = torch.cat([uncond_hs, encoder_hs], dim=0).to(unet_dtype)
                noise_pred = self.unet(latent_batch, t_batch.repeat(2), cond_batch).sample.float()
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                eps_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            stored = noise_maps.get(t_prev_int, torch.randn_like(x_t))
            x_t = self._ddpm_step(x_t, t, t_prev_int, eps_pred, stored)

            # RePaint: lock background + target region after each step
            if t_prev_int >= 0:
                abar_prev = self.scheduler.alphas_cumprod[t_prev_int].to(device=device, dtype=torch.float32)
                orig_noise = noise_maps_orig[t_prev_int]
                x_t_orig = abar_prev.sqrt() * x0_orig + (1 - abar_prev).sqrt() * orig_noise
                x_t = x_t * (1.0 - bg_mask) + x_t_orig * bg_mask

                # Lock target region to composite only during early (high-noise) steps
                if i < lock_cutoff:
                    comp_noise = noise_maps[t_prev_int]
                    x_t_comp = abar_prev.sqrt() * x0_composite + (1 - abar_prev).sqrt() * comp_noise
                    x_t = x_t * (1.0 - M_tgt_lock) + x_t_comp * M_tgt_lock

        # 9. Second pass: inpaint source region at SDEdit strength 0.5
        # Locks everything outside the source to the first-pass result; only the
        # vacated source hole is free to regenerate with background context.
        x0_result = x_t.clone()
        src_lock_soft = (1.0 - gaussian_blur_mask(M_src, sigma=2.0)).clamp(0, 1)

        inp_gen_device = device
        inp_gen = torch.Generator(device=inp_gen_device).manual_seed(seed + 1)
        inp_noise_maps = {}
        for t in timesteps:
            inp_noise_maps[t.item()] = torch.randn(
                x0_result.shape, generator=inp_gen, device=device, dtype=torch.float32
            )

        inp_strength = 0.5
        start_idx2 = int((1.0 - inp_strength) * len(timesteps))
        start_idx2 = max(0, min(start_idx2, len(timesteps) - 1))
        t_start2_int = timesteps[start_idx2].item()
        abar_s2 = self.scheduler.alphas_cumprod[t_start2_int].to(device=device, dtype=torch.float32)
        x_src = abar_s2.sqrt() * x0_result + (1 - abar_s2).sqrt() * inp_noise_maps[t_start2_int]

        for i2, t2 in enumerate(timesteps[start_idx2:]):
            t2_global = start_idx2 + i2
            t2_prev_int = timesteps[t2_global + 1].item() if t2_global + 1 < len(timesteps) else -1

            with torch.no_grad():
                lat2 = torch.cat([x_src.to(unet_dtype)] * 2, dim=0)
                cond2 = torch.cat([uncond_hs, encoder_hs], dim=0).to(unet_dtype)
                np2 = self.unet(lat2, t2.unsqueeze(0).to(device).repeat(2), cond2).sample.float()
                np2_u, np2_c = np2.chunk(2)
                eps2 = np2_u + guidance_scale * (np2_c - np2_u)

            x_src = self._ddpm_step(x_src, t2, t2_prev_int, eps2,
                                    inp_noise_maps.get(t2_prev_int, torch.randn_like(x_src)))

            if t2_prev_int >= 0:
                abar_p2 = self.scheduler.alphas_cumprod[t2_prev_int].to(device=device, dtype=torch.float32)
                x0_result_noisy = abar_p2.sqrt() * x0_result + (1 - abar_p2).sqrt() * inp_noise_maps[t2_prev_int]
                x_src = x_src * (1.0 - src_lock_soft) + x0_result_noisy * src_lock_soft

        x_t = x_src

        # 10. Decode
        decoded = decode_latent(self.vae, x_t.to(vae_dtype))
        return tensor_to_pil(decoded), composite
