# Tutorial for CM22012 Advanced Programming
## Deploying Diffusion Model on GPU Cluster

### Dependencies
```shell
pip install notebook pillow ipywidgets 'transformers[torch]'
pip install wandb
pip install --upgrade diffusers transformers scipy
```

### Usage
```shell
CUDA_VISIBLE_DEVICES=<i-th-device> python diffusion_example.py <prompt-text> -n <num-imgs> -g <img_height> -w <img_width> -s <seed>
```
Simply, 
```shell
python diffusion_example.py <prompt-text>
```
Will generate one single image for input prompt. 
```
Simply, 
```shell
python diffusion_example.py <prompt-text>
```
Example: 
```
diffusion_example.py "a pile of books" -n 2 -s 42
```
Output: 
![Alt text](books.png?raw=true "Result")

### Useful links
- Hugging Face AI community: https://huggingface.co/
- WandB: https://wandb.ai/
- Stable Diffusion Paper: https://arxiv.org/pdf/2112.10752