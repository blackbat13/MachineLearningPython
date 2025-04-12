# AnimateDiff Example
from diffusers import MotionAdapter, AnimateDiffPipeline

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
pipe = AnimateDiffPipeline.from_pretrained(
    "emilianJR/epiCRealism", 
    motion_adapter=adapter
).to("cuda")

# Generate video frames
output = pipe(
    prompt="A robot dancing, 4k, cinematic",
    num_frames=16,
    num_inference_steps=25
)
frames = output.frames[0]

# Convert to GIF
frames[0].save(
    "robot_dance.gif",
    save_all=True,
    append_images=frames[1:],
    duration=100,
    loop=0
)