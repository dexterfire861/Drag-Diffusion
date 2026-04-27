# test_setup.py
import torch
from diffusers import DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load SD 2.1 components
model_id = "stabilityai/stable-diffusion-2-1"

vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)

print("All components loaded successfully")