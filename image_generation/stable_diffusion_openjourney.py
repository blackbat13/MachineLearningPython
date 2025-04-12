# pip install transformers diffusers torch

from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "prompthero/openjourney-v4",
    custom_pipeline="lpw_stable_diffusion",  # For longer prompts
    use_safetensors=True
).to("cuda" if torch.cuda.is_available() else "cpu")

image = pipe(
    prompt="a cyberpunk cat wearing VR goggles",
    num_inference_steps=2,  # Turbo models work with just 2-4 steps
    guidance_scale=0.0  # Disable classifier-free guidance
).images[0]

image.save("sdxl_output.png")