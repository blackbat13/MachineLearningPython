# pip install transformers diffusers torch

from diffusers import AutoPipelineForText2Image
import torch
import gradio as gr

# Load the model
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")

# Define the generation function
def generate_image(prompt, steps, guidance):
    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance
    ).images[0]
    return image

# Create the Gradio interface
interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here"),
        gr.Slider(2, 10, value=2, step=1, label="Inference Steps"),
        gr.Slider(0.0, 20.0, value=0.0, step=0.1, label="Guidance Scale")
    ],
    outputs=gr.Image(label="Generated Image"),
    title="Stable Diffusion XL Image Generator",
    description="Generate images using the Stable Diffusion XL Turbo model."
)

# Launch the interface
interface.launch()