import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Load the model and scheduler
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# Define the prompt and image size
prompt = "a simple oil painting of marbeling and flowing colors"
height = 784
width = 784

# Generate the image
with torch.no_grad():
    image = pipe(prompt, height=height, width=width).images[0]

# Save the image
image.save("large_image-2.png")
