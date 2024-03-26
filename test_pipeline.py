from utils.sd_utils import diffusion_model_configs, generate_image
from diffusers import StableDiffusionXLPipeline

selected_model = "segmind/SSD-1B"

pipe = StableDiffusionXLPipeline.from_pretrained(**diffusion_model_configs[selected_model])
pipe.to("cuda")

prompt = "cinematic film still, 4k, realistic, cinematic photo of a panda wearing a blue spacesuit, sitting in a bar, long shot, low light, looking straight at the camera, upper body shot, shallow depth of field, vignette, intricate design, highly detailed, bokeh, cinemascope"
neg_prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off"

image = generate_image(pipe=pipe, prompt=prompt, neg_prompt=neg_prompt)
image.show()