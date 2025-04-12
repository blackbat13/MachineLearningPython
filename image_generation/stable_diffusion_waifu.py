# pip install transformers diffusers torch

from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "hakurei/waifu-diffusion-v1-4", 
    safety_checker=None  # Disable content filtering
).to("cuda" if torch.cuda.is_available() else "cpu")

image = pipe("1girl in mecha armor, intricate details").images[0]

image.save("waifu_output.png")