import torch


diffusion_model_configs = {
    "segmind/SSD-1B": {
        'pretrained_model_name_or_path': 'C:\\Coding\\stable-diffusion\\Segmind\\checkpoints\\SSD-1B', # path to checkpoints
        'torch_dtype': torch.float16, 
        'use_safetensors': True, 
        'variant': "fp16"},
}


def generate_image(prompt, pipe, neg_prompt="ugly, blurry, poor quality"):
    image = pipe(prompt=prompt, negative_prompt=neg_prompt, height=1024, width=1024).images[0]
    return image