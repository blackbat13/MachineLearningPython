from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

# Initialize pipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda")

# Generate frames with evolving prompts
frames = []
for i in range(10):
    # Modify prompt dynamically
    prompt = f"A flower blooming, frame {i}/10, vibrant colors"
    
    # Generate image with consistent seed
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5, generator=torch.Generator().manual_seed(42)).images[0]
    
    frames.append(image)

# Save as GIF
frames[0].save(
    "animation.gif",
    save_all=True,
    append_images=frames[1:],
    duration=100,  # Milliseconds per frame
    loop=0,        # Infinite loop
    optimize=True
)