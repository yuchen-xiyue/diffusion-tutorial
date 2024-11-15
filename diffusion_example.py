import os
import argparse
from datetime import datetime

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

# Configuration
MODEL_NAME = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda" # "cpu"
OUTPUT_FOLDER = "outputs"
USE_WANDB = True
INFERENCE_STEPS = 50
TIMESTEPS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

if USE_WANDB: 
    import wandb
    WANDB_PROJECT_NAME = "Diffusion Example"

if not os.path.exists(OUTPUT_FOLDER): 
    os.mkdir(OUTPUT_FOLDER)

# Get timestemp for saving folder name. 
get_timestemp = lambda: datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


# Diffusion process happen in latent space.
# This function could get intermediate image result from latent variants. 
def latents_to_pil_images(pipe: StableDiffusionPipeline, latents: torch.Tensor): 
    """
    Diffusion process happens in latent space.
    This function could get intermediate image result from latent variants. 
    """

    latents = 1 / pipe.vae.config.scaling_factor * latents # unscale variants
    # decode latent variants to pytorch images: [-1, 1] -> [0, 255]
    images_pt  = ((pipe.vae.decode(latents, return_dict=False)[0].cpu()/2 + .5).clamp(0, 1) * 255).to(torch.uint8)
    # convert pytorch images to numpy image formats
    images_np  = images_pt.permute(0, 2, 3, 1).numpy()
    # convert numpy images to Image objects
    images_pil = [Image.fromarray(img) for img in images_np]

    return images_pil


def generate(
        prompts:str, 
        num_images: int=1, 
        height: int=512, 
        width : int=512, 
        num_inference_steps : int=INFERENCE_STEPS, 
        timesteps_to_capture: list[int]=TIMESTEPS
        ): 
    
    output = []

    """Diffuse with pipeline and callback function. """
    def callback_dynamic_cfg(pipline, step, timestep, callback_kwargs): 
        if step in timesteps_to_capture: 
            with torch.no_grad(): 
                output.append(latents_to_pil_images(pipe=pipline, latents=callback_kwargs['latents']))
        return callback_kwargs
    

    pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
    imgs = pipe(
            prompts, 
            height=height, 
            width=width, 
            num_images_per_prompt=num_images, 
            num_inference_steps=num_inference_steps, 
            callback_on_step_end=callback_dynamic_cfg
            )[0]
    output.append(imgs)

    """Alternatively, diffuse with scheduler instead of callback function. """
    # # Extract text embeddings
    # text_inputs = pipe.tokenizer([prompts] * num, return_tensors="pt").to(DEVICE)
    # text_embeddings = pipe.text_encoder(text_inputs.input_ids)[0]

    # # Initiate latent variable
    # latents = torch.randn(
    #                 (num, pipe.unet.config.in_channels, height // 8, width // 8),
    #                 generator=torch.manual_seed(42),
    #                 ).to(DEVICE)
            
    # # Diffuse with DDIM scheduler
    # pipe.scheduler.set_timesteps(num_inference_steps)
    # with torch.no_grad():
    #     for i, t in tqdm(enumerate(pipe.scheduler.timesteps), desc="Denoise process"):
    #         if i in timesteps_to_capture: 
    #             output.append(latent_to_pil_images(pipe=pipe, latents=latents))
    #         noise_pred = pipe.unet(latents, t, encoder_hidden_states=text_embeddings).sample
    #         latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # output.append(latent_to_pil_images(pipe=pipe, latents=latents))
    return output


def save_imgs(output: list[list[Image.Image]]): 
    """Locally save the results. """
    output_folder = os.path.join(OUTPUT_FOLDER, get_timestemp())
    os.mkdir(output_folder)
    output = zip(*output)
    for i, img_list in enumerate(output): 
        result_folder = os.path.join(output_folder, f"result-{i}")
        os.mkdir(result_folder)
        for j, img in enumerate(img_list): 
            if j != len(img_list) - 1: 
                img.save(os.path.join(result_folder, f"step-{TIMESTEPS[j]}.png"))
            else: 
                img.save(os.path.join(result_folder, f"result(step-{INFERENCE_STEPS}).png"))
    return

def log_imgs_with_wandb(output: list[list[Image.Image]]): 
    output = zip(*output)
    for i, img_list in enumerate(output): 
        wandb.log({
            f"Result-{i}": [
                wandb.Image(img, caption=f"step-{TIMESTEPS[j]}" 
                    if j != len(img_list) - 1 
                    else f"result(step-{INFERENCE_STEPS})")
                for j, img in enumerate(img_list)
                ]
            })
    

class InferenceManager: 
    def __init__(self, configs): 
        self.configs = configs
        self.use_wandb = configs['use_wandb']
        self.output = None

    def __enter__(self):
        if self.use_wandb: 
            wandb.init(
                project=WANDB_PROJECT_NAME, 
                config=self.configs
                )
        return self
    
    def __exit__(self, *args): 
        save_imgs(self.output)
        if self.use_wandb: 
            log_imgs_with_wandb(self.output)
            wandb.finish()


def main():
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("prompts", type=str, help="Type in your prompts for generation. ")
    parser.add_argument("-n", "--num_images", type=int, default=1, help="How many images to generate, default 1")
    parser.add_argument("-w", "--width",  type=int, default=512, help="Output image width, 512 by default. ")
    parser.add_argument("-g", "--height", type=int, default=512, help="Output image height, 512 by default. ")
    args = parser.parse_args()

    # merge configurations
    configs = {
        **{
            'model': MODEL_NAME, 
            'inference_step': INFERENCE_STEPS, 
            'use_wandb': USE_WANDB
            }, **vars(args)
    }
    
    with InferenceManager(configs) as im: 
        im.output = generate(
            prompts=configs['prompts'], 
            num_images=configs['num_images'], 
            height=configs['height'], 
            width=configs['width']
            )
        
    
if __name__ == "__main__":
    main()