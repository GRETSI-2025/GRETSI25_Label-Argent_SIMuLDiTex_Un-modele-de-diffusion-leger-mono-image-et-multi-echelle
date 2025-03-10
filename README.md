# SIMuLDiTex



# A geometrically aware auto-encoder for multi-texture synthesis
[Pierrick Chatillon](https://scholar.google.com/citations?user=8MgK55oAAAAJ&hl=en) | [Julien Rabin](https://sites.google.com/site/rabinjulien/) | [David Tschumperlé](https://tschumperle.users.greyc.fr/)


[Arxiv](TODO) [Paper](TODO)

### Official pytorch implementation of the paper: "SIMuLDiTex: A Single Image Multiscale and Lightweight Diffusion Model for Texture Synthesis"


![Interpolation between 4 textures, synthesis of size 2048, downscaled by a factor 2, and compressed](images/interpolation.gif)








### Installation

These commands will create a conda environment called TextureAE with the required dependencies, then place you in it :
```
conda env create -f requirements.yml
conda activate TextureAE
```

### Pretrained models



###  Train




```
python code/train.py --name <name_of_the_experiment> \
--dataset_folder <path_to_dataset> #<path_to_dataset> should be an absolute path
```



Please refer to code/config.py for described additional arguments.
All the models, arguments and tensorboard logs for an experiments are stored under the same folder ./runs/name_of_the_experiment/

### Inference



```
python inference.py --name <name_of_the_experiment> 
```


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