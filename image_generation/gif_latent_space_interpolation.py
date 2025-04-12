import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
pipe = pipe.to("cuda")

# Generate two seed images
prompt_start = "A spaceship on launchpad"
prompt_end = "A spaceship in deep space"
image1 = pipe(prompt_start).images[0]
image2 = pipe(prompt_end).images[0]

# Interpolate between latent vectors
latents1 = pipe(prompt_start, output_hidden_states=True).hidden_states[-1]
latents2 = pipe(prompt_end, output_hidden_states=True).hidden_states[-1]

frames = []
for alpha in torch.linspace(0, 1, 15):
    blended_latents = latents1 * (1 - alpha) + latents2 * alpha
    image = pipe.decode_latents(blended_latents)
    frames.append(Image.fromarray(image))