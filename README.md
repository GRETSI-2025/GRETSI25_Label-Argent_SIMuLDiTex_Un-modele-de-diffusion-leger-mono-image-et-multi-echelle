# SIMuLDiTex



# A geometrically aware auto-encoder for multi-texture synthesis
[Pierrick Chatillon](https://scholar.google.com/citations?user=8MgK55oAAAAJ&hl=en) | [Julien Rabin](https://sites.google.com/site/rabinjulien/) | [David Tschumperlé](https://tschumperle.users.greyc.fr/)


[Arxiv](TODO) [Paper](TODO)

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



###  Train

Training parameters are described in the train.py parser.

```
python train.py --name <name_of_the_experiment> 
```

### Inference

All experiments with hyperparameters are replicable in the notebook experiments.ipynb

### Results 



### Acknowledgments
This was built upon the very useful [PyTorch diffusion implementaion](https://github.com/lucidrains/denoising-diffusion-pytorch), and this amazing signal resizing repo [ResizeRight](https://github.com/assafshocher/ResizeRight)

### Citation
If you use this code for your research, please cite our paper:

```

```

### License
This work is under the CC-BY-NC-4.0 license.

### Disclaimer
The code is provided "as is" with ABSOLUTELY NO WARRANTY expressed or implied.