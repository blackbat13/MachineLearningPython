# pip install transformers diffusers torch

from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")

image = pipe(
    prompt="a cyberpunk cat wearing VR goggles",
    num_inference_steps=2,  # Turbo models work with just 2-4 steps
    guidance_scale=0.0  # Disable classifier-free guidance
).images[0]

image.save("output.png")

# from diffusers import StableDiffusionPipeline

# pipe = StableDiffusionPipeline.from_pretrained(
#     "hakurei/waifu-diffusion-v1-4", 
#     safety_checker=None  # Disable content filtering
# ).to("cuda")

# image = pipe("1girl in mecha armor, intricate details").images[0]

# from diffusers import DiffusionPipeline

# pipe = DiffusionPipeline.from_pretrained(
#     "prompthero/openjourney-v4",
#     custom_pipeline="lpw_stable_diffusion",  # For longer prompts
#     use_safetensors=True
# )