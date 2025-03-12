from SIMuLDiTex.SIMuLDiTex import Unet, GaussianDiffusion, Trainer
import argparse,json,os,re
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def get_latest_model_index(directory):
    files = os.listdir(directory)
    model_files = [f for f in files if re.match(r'model-(\d+)\.pt', f)]
    indices = []
    for model in model_files:
        match = re.match(r'model-(\d+)\.pt', model)
        if match:
            indices.append(int(match.group(1)))
    if indices:
        return max(indices)
    else:
        return None




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='Path to the saved args.json file')
    parser.add_argument('--dataset_path', default='./images/data/wall/', type=str, help='dataset path')
    parser.add_argument('--name', type=str, default='test', help='name of the results folder')
    

    parser.add_argument('--T', default= 200, type=int, help='Total diffusion timesteps')
    
    parser.add_argument('--lr', default= 1e-4, type=float, help='learning rate')
    parser.add_argument('--bs', default= 32, type=int, help='batch size')
    parser.add_argument('--attn', default= False, type=str2bool, help='use attention at UNET bottleneck')
    parser.add_argument('--fourier_unit', default= True, type=str2bool, help='use FourierUnit at UNET bottleneck')
    parser.add_argument('--dim', default=16, type=int, help='base dimension of UNET')
    parser.add_argument('--img_size', default=128,  type=int, help='size of crops during training')

    parser.add_argument('--octaves', default=3, type=int, help='number of halving of the resolution')
    parser.add_argument('--scales', default=2, type=int, help='number of resolutions treated per octave')

    
    parser.add_argument('--training_steps', default=20000, type=int, help='steps every job')
    parser.add_argument('--save_every', default=5000, type=int, help='save_every')

    args = parser.parse_args()
    os.makedirs('./runs/%s'%args.name,exist_ok=True)



    with open('./runs/%s/args.json'%args.name, 'w+') as f:
        json.dump(vars(args), f, indent=4)
    print(args.name) 
    model = Unet(
        dim =args.dim,
        dim_mults = (1, 2, 4, 4),
        mid_attn=args.attn,
        mid_fourier=args.fourier_unit
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = args.img_size,
        timesteps = args.T,           # number of steps
        sampling_timesteps = args.T    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )


    ckpt=get_latest_model_index('./runs/%s'%args.name)
    trainer = Trainer(
        diffusion,
        folder=args.dataset_path,
        results_folder='./runs/%s'%args.name,
        train_batch_size = args.bs,
        train_lr = args.lr,
        train_num_steps =  args.training_steps,         # total training steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = False,         # whether to calculate fid during training
        save_and_sample_every=args.save_every,
        octaves=args.octaves,
        scales=args.scales
    )
    if ckpt is not None:
        trainer.load(ckpt)
    if ckpt is None or ckpt<(10*args.training_steps//args.save_every): # if code launched 10 times
        trainer.train()
