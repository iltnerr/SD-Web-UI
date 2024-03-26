import torch
import threading
import time
import subprocess
import os

from pathlib import Path
from flask import Flask, render_template, request
from utils.sd_utils import diffusion_model_configs, generate_image
from utils.creds import user, server, port
from diffusers import StableDiffusionXLPipeline
from waitress import serve


lock = threading.Lock()

selected_model = "segmind/SSD-1B"
copied_imgs_f = "tmp/copied_imgs.txt"
out_dir = "static/output"

# RPI
target = f"{user}@{server}:/home/{user}/Desktop/digiframe/playlists/stable_diffusion"

pipe = StableDiffusionXLPipeline.from_pretrained(**diffusion_model_configs[selected_model])
pipe.to("cuda")

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():

    default_prompt = "cinematic film still, 4k, realistic, cinematic photo of a panda wearing a blue spacesuit, sitting in a bar, long shot, low light, looking straight at the camera, upper body shot, shallow depth of field, vignette, intricate design, highly detailed, bokeh, cinemascope"
    default_neg_prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off"
    disp_img_path = "static/display_image.jpg"
    
    # set fallback prompts
    prompt = request.form.get('prompt')
    neg_prompt = request.form.get('neg_prompt')
    prompt = prompt if prompt is not None else default_prompt
    neg_prompt = neg_prompt if neg_prompt is not None else default_neg_prompt
    
    if request.method == 'POST':
        button_clicked = request.form['button']

        if button_clicked == 'generate':
            return generate_func(disp_img_path, prompt, neg_prompt, default_prompt)
        
        elif button_clicked == 'copy':
            source = request.form['imagename']
            return copy_func(source, target, disp_img_path, default_prompt, default_neg_prompt)
        
        else:
            return default_render(disp_img_path, prompt, neg_prompt)
    
    else:
        return default_render(disp_img_path, prompt, neg_prompt)

@app.route('/gallery', methods=['POST', 'GET'])
def gallery():
    image_files = [f"{out_dir}/{f}" for f in os.listdir(out_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    image_files = list(reversed(image_files))
    last_elems = min(200, len(image_files))
    return render_template('gallery.html', image_files=image_files[:last_elems])
    
def generate_func(disp_img_path, prompt, neg_prompt, default_prompt):
    
    if prompt is not None and prompt != default_prompt:
        if len(prompt) > 1:
            if lock.acquire(blocking=True): # For multiple requests: First come, first serve (running thread is not interrupted).
               
                print(f"\n\n\nPos. Prompt:\n<<<{prompt}>>>\n\nNeg. Prompt:\n<<<{neg_prompt}>>>\n")
                image = generate_image(pipe=pipe, prompt=prompt, neg_prompt=neg_prompt)
                
                timestr = time.strftime("%Y%m%d-%H%M%S")
                disp_img_path = f"{out_dir}/{timestr}.jpg"
                image.save(disp_img_path)
                print(f"Image saved under {disp_img_path}.")

                lock.release()

    copy_msg = ''
    display_btn = '' if prompt != default_prompt else ' style="display: none;"' # after generating an image, copy button should be visible
    return render_template('index.html', display_image=disp_img_path, prompt=prompt, neg_prompt=neg_prompt, display_btn=display_btn, copy_msg=copy_msg)

def copy_func(source, target, disp_img_path, prompt, neg_prompt):

    with open(copied_imgs_f) as f:
        copied_imgs = [line.rstrip('\n') for line in f]
    
    if source not in copied_imgs:
        print(f"Copy file {source} to {target}")    
        subprocess.run(["scp", source, target])
    
        with open(copied_imgs_f, "w") as f:
            f.write(f"{source}\n")
    
    copy_msg = 'Image added to gallery'
    display_btn = ' style="display: none;"' # copy button hidden
    return render_template('index.html', display_image=disp_img_path, prompt=prompt, neg_prompt=neg_prompt, display_btn=display_btn, copy_msg=copy_msg)

def default_render(disp_img_path, prompt, neg_prompt):
    
    copy_msg = ''
    display_btn = ' style="display: none;"'
    return render_template('index.html', display_image=disp_img_path, prompt=prompt, neg_prompt=neg_prompt, display_btn=display_btn, copy_msg=copy_msg)

def initialize():

    # clean up list of copied images
    if os.path.isfile(copied_imgs_f):
        os.remove(copied_imgs_f)
    
    Path(copied_imgs_f).touch() # create emtpy txt file

    # clean up old images
    img_list = list(reversed(os.listdir(out_dir)))
    
    if len(img_list) > 200:
        t_now = time.time()
        
        for f in img_list[200:]:
            f = os.path.join(out_dir, f)
            if os.stat(f).st_mtime < t_now - 30*86400:
                if os.path.isfile(f):
                    os.remove(os.path.join(f))
                    print(f"Removed file {f}")
                
    print("Finished Initialization\n")
                
                
if __name__ == '__main__':
    initialize()
    serve(app, host="0.0.0.0", port=port)
    #app.run(debug=True, host="0.0.0.0", port=port)