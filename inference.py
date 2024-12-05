### IMPORT LIBRARIES ###
import torch
from model import DiT_models
import os
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from open_clip import create_model_from_pretrained
from block.CT_encoder import CT_Encoder
from load_data import NpyDataset, transform_test, get_sampler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torchvision.utils import save_image


## DEFINE VARIABLES ###
seed = 0
image_size = 224
latent_size = image_size // 8
ckpt = "/storage/student13/DiffMa-Diffusion-Mamba/results/brain/004-DiT-L-2/checkpoints/0100000.pt"
model = "DiT-L/2"
load_ckpt_type = "ema"
sample_num_step = 250
vae = "ema"
ct_ckpt = "./pretrain_ct_vision_embedder/brain_patch_size_2.pt"
input_val_folder = '/storage/student13/DiffMa-Diffusion-Mamba/datasets/brain/B_test'
output_val_folder = '/storage/student13/DiffMa-Diffusion-Mamba/datasets/brain/A_test'
sample_global_batch_size = 8
sample_num_workers = 1
save_dir = "./result_samples/"

### MAIN EXECUTION ###

def find_model(model_name, load_ckpt_type):
    """
    Finds a pre-trained model. Alternatively, loads a model from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if load_ckpt_type in checkpoint: 
        checkpoint = checkpoint[load_ckpt_type]
    return checkpoint

def main(seed, model, latent_size, ckpt, vae, image_size, ct_ckpt, input_val_folder, output_val_folder, sample_num_step, sample_global_batch_size, sample_num_workers, save_dir, load_ckpt_type):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)

    if torch.cuda.is_available():
        device = 'cuda' 
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    
    model_name = DiT_models[model](input_size=latent_size).to(device)
    state_dict = find_model(ckpt, load_ckpt_type)
    model_name.load_state_dict(state_dict)
    model_name.eval()

    diffusion = create_diffusion(str(sample_num_step))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae}").to(device)
    clip_model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    image_encoder = clip_model.visual.to(device)
    image_encoder.eval()

    ct_encoder = CT_Encoder(img_size=image_size // 8, 
                            patch_size=int(model[-1]), 
                            in_channels=4, 
                            embed_dim=512, 
                            contain_mask_token=True,
                            ).to(device)
    ct_ckpt_path = ct_ckpt or f"./pretrain_ct_encoder/patch_size_2.pt"
    ct_state_dict = find_model(ct_ckpt_path, load_ckpt_type)
    ct_encoder.load_state_dict(ct_state_dict)
    ct_encoder.eval()

    val_dataset = NpyDataset(input_val_folder, output_val_folder, transform=transform_test)
    sampler=get_sampler(val_dataset)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=int(sample_global_batch_size // dist.get_world_size()), 
        shuffle=False, 
        sampler=sampler, 
        num_workers=sample_num_workers, 
        drop_last=False,
        )
    print((f"Dataset contains {len(val_dataset)}."))
    item = 0
    for x_ct, z_mri in val_loader:
        item+=1
        # if item<15:
        #     continue
        n = x_ct.shape[0]
        z = torch.randn(n, 4, latent_size, latent_size, device=device)  #Random noise

        x_ct = x_ct.to(device)
        x_ct = torch.cat([x_ct] * 3, dim=1)
        x_ct_ = x_ct
        # save_image(x_ct, "sample_ct.png", nrow=4, normalize=True, value_range=(-1, 1))

        z_mri = z_mri.to(device)
        z_mri = torch.cat([z_mri] * 3, dim=1)


        with torch.no_grad():
            if not torch.all((z_mri >= -1) & (z_mri <= 1)):
                z_mri = ((z_mri - z_mri.min()) * 1.0 / (z_mri.max() - z_mri.min())) * 2.0 - 1.0  #4.21改
            x_ = vae.encode(x_ct).latent_dist.sample().mul_(0.18215)
            x_ct = image_encoder(x_ct)
            ct_weight, x_ct_2 = ct_encoder(x_)

        model_kwargs = dict(y=x_ct, y2=x_ct_2, w=ct_weight)

        # Sample images:
        samples = diffusion.p_sample_loop(model_name.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device)
        samples = vae.decode(samples / 0.18215).sample
        
        os.makedirs('./' + save_dir, exist_ok=True)
        save_image(samples, save_dir + '/' + str(item) + '_sample_gen.png', nrow=4, normalize=True, value_range=(-1, 1))
        save_image(z_mri, save_dir + '/' + str(item) + '_sample_mri.png', nrow=4, normalize=True, value_range=(-1, 1))
        save_image(x_ct_, save_dir + '/' + str(item) + '_sample_ct.png', nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, required=True)
    # args = parser.parse_args()
    # cli_config = OmegaConf.create({k: v for k, v in args.__dict__.items() if v is not None and k != 'config'})
    # args = OmegaConf.merge(OmegaConf.load(args.config), cli_config)
    main(seed, model, latent_size, ckpt, vae, image_size, ct_ckpt, input_val_folder, output_val_folder, sample_num_step, sample_global_batch_size, sample_num_workers, save_dir, load_ckpt_type)