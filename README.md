# SIMuLDiTex: a Single Image Multiscale and Lightweight Diffusion Model for Texture Synthesis
[Pierrick Chatillon](https://scholar.google.com/citations?user=8MgK55oAAAAJ&hl=en) | [Julien Rabin](https://sites.google.com/site/rabinjulien/) | [David Tschumperlé](https://tschumperle.users.greyc.fr/)


[Arxiv]() [Paper]() [HAL](https://hal.science/hal-04994907)

### Official pytorch implementation of the paper: "SIMuLDiTex: A Single Image Multiscale and Lightweight Diffusion Model for Texture Synthesis"

Examples of large scale synthesis, after training on the image displayed on the left:
![](images/carpet_fig_10_80_time29.png)
![](images/wall_fig_10_80_time24.png)
![](images/rust_fig_10_80_time22.png)


Interpolation between 4 textures, synthesis of size 2048, downscaled by a factor 2, and compressed:
![](images/interpolation.gif)








### Installation

These commands will create a conda environment called simulditex with the required dependencies, then place you in it :
```
conda env create -f requirements.yml
conda activate simulditex
```


### Pretrained models

This repo contains pretrained models under ./runs/ . Please refer to experiments.ipynb for further inference details. 


###  Training

Training parameters are described in the train.py parser.

```
python train.py --name <name_of_the_experiment> 
```

### Multi-GPU Training

As inherited from [this repo](https://github.com/lucidrains/denoising-diffusion-pytorch), the code is compatible with <a href="https://huggingface.co/docs/accelerate/accelerator">🤗 Accelerator</a>. You can easily do multi-gpu training in two steps using their `accelerate` CLI.

At the project root directory, run

```
accelerate config
```

Then, the multi-gpu training can be launched with

```
accelerate launch train.py --name <name_of_the_experiment> 
```

### Inference

All experiments with hyperparameters are replicable in the notebook experiments.ipynb.
The notebook saves the results in ./images/results/



### Acknowledgments
This was built upon the very useful [PyTorch diffusion implementaion](https://github.com/lucidrains/denoising-diffusion-pytorch), and this amazing signal resizing repo [ResizeRight](https://github.com/assafshocher/ResizeRight).

### Citation
If you use this code for your research, please cite our paper:

```

```

### License
This work is under the MIT license.

### Disclaimer
The code is provided "as is" with ABSOLUTELY NO WARRANTY expressed or implied.