# pip install transformers diffusers torch gradio accelerate

from diffusers import AutoPipelineForText2Image
import torch
import gradio as gr

# Load models
pipes = dict()

pipes["sdxl-turbo"] = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")

pipes["openjourney-v4"] = AutoPipelineForText2Image.from_pretrained(
    "prompthero/openjourney-v4",
    custom_pipeline="lpw_stable_diffusion",  # For longer prompts
    use_safetensors=True,
    torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")

pipes["waifu-diffusion"] = AutoPipelineForText2Image.from_pretrained(
    "hakurei/waifu-diffusion", 
    safety_checker=None,  # Disable content filtering
    torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")

# Define the generation function
def generate_images(prompt, steps, guidance, pipe_name):
    pipe = pipes[pipe_name]
    images = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance
    ).images
    return images[0]

# Create the Gradio interface
interface = gr.Interface(
    fn=generate_images,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here"),
        gr.Slider(2, 20, value=2, step=1, label="Inference Steps"),
        gr.Slider(0.0, 20.0, value=0.0, step=0.1, label="Guidance Scale"),
        gr.Dropdown(["sdxl-turbo", "openjourney-v4", "waifu-diffusion"], label="Model", multiselect=False),
    ],
    outputs=gr.Image(label="Generated Image"),
    title="Stable Diffusion Image Generator",
    description="Generate images using Stable Diffusion."
)

# Launch the interface
interface.launch(share=True)