#################################################################################
#                                  IMPORT LIBRARIES                             #
#################################################################################
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
import os
from glob import glob
from loguru import logger
import wandb
from model import DiT_models
from copy import deepcopy
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from block.CT_encoder import CT_Encoder
from open_clip import create_model_from_pretrained
from load_data import NpyDataset, transform_train, get_sampler
from torch.utils.data import DataLoader
from time import time
import argparse
from omegaconf import OmegaConf


#################################################################################
#                                  VARIABLES                                    #
#################################################################################
global_batch_size = 2
global_seed = 4
results_dir = "./results"
model = "DiT-L/2"
dt_rank = 16
d_state = 16
epochs = 50
image_size = 224
init_from_pretrained_ckpt = True
pretrain_ckpt_path = "./result_ckpt/DiT-L/2-5000"
ct_ckpt = "./pretrain_ct_vision_embedder/brain_patch_size_2.pt"
autocast_bool = False # Automatic mixed precision
lr = 1e-4
lr_pretrained = 1e-4
input_folder = './datasets/brain/A_train'
output_folder = './datasets/brain/B_train'
num_workers = 8
init_train_steps = 800_000
accumulation_steps = 1
log_every = 10
ckpt_every = 10000

#################################################################################
#                                  TRAINING HELPER                              #
#################################################################################
@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def create_logger(logging_dir):
    """Create a logger that writes to a log file and sldout"""
    if dist.get_rank() == 0:
        logger.add(f"{logging_dir}/log"+f"_{dist.get_rank()}.txt", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
    return logger

def find_model_ema(model_name):
    assert os.path.isfile(model_name), f"Could not find the path at {model_name}"
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    return checkpoint

def find_model(model_name):
    assert os.path.isfile(model_name), f"Cannot find the path at {model_name}"
    checkpoint = torch.load(model_name, map_location= lambda storage, loc: storage)
    checkpoint = checkpoint["model"]
    return checkpoint

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

#################################################################################
#                                  Training Loop                                #
#################################################################################



def main(model, global_batch_size, global_seed, results_dir, epochs, image_size, init_from_pretrained_ckpt, pretrain_ckpt_path, ct_ckpt, autocast, lr, lr_pretrained, input_folder, output_folder, num_workers, init_train_steps, accumulation_steps, log_every, ckpt_every):
    """
    Train the diffusion transformer model
    """
    assert torch.cuda.is_available(), "Training requires GPU"
    scaler = GradScaler()
    assert global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size" 
    rank = dist.get_rank() # Retrieves the rank (ID) of the current process
    device = rank % torch.cuda.device_count() # Maps the process rank to a specific GPU device.
    seed = global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder
    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{results_dir}/*"))
        model_string_name = model.replace("/", "-")  
        experiment_dir = f"{results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        if wandb:
            wandb.init(project=model.replace('/','_'))
            # wandb.init(project=args.model.replace('/','_'), id='ylhfep72', resume='must')   # load the previous run
            wandb.config = {"learning_rate": 0.0001, 
                            "epochs": epochs, 
                            "batch_size": global_batch_size,
                            "save-path": experiment_dir,
                            "autocast": autocast,
                            }
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model
    assert image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE Encoder)"
    latent_size = image_size // 8
    model = DiT_models[model](input_size = latent_size)

    if init_from_pretrained_ckpt:
        # Load model
        model_state_dict_ = find_model(pretrain_ckpt_path)
        model.load_state_dict(model_state_dict_)
        # Load ema
        ema = deepcopy(model).to(device)
        ema_state_dict_ = find_model_ema(pretrain_ckpt_path)
        ema.load_state_dict(ema_state_dict_)
        # Log
        logger.info(f"Loaded pretrained model from {pretrain_ckpt_path}")
    else:
        ema = deepcopy(model).to(device)

    requires_grad(ema, False) # Freeze ema model

    model = DDP(model.to(device), device_ids=[rank])

    diffusion = create_diffusion(timestep_respacing="") # Default: 1000 steps, linear noise schedule at .diffusion/__init__.py
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae}").to(device)

    # Load CT encoder
    ct_encoder = CT_Encoder(
        img_size=image_size // 8,
        patch_size=int(model[-1]), 
        in_channels=4,
        embed_dim = 512, # Correspond to 512 which is the dimension for BiomedCLIP
        contain_mask_token=True,
        ).to(device)
    
    ct_ckpt_path = ct_ckpt
    ct_state_dict = find_model_ema(ct_ckpt_path)
    ct_encoder.load_state_dict(ct_state_dict)
    ct_encoder.eval()

    if rank == 0:
        logger.info(f"DiffMa Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Use half-precision training? {autocast_bool}")
    
    # Load BioMedClip
    clip_model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    image_encoder = clip_model.visual.to(device)

    # Learning rate
    if init_from_pretrained_ckpt:
        learning_rate = lr_pretrained
    else:
        learning_rate = lr
    
    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)

    train_dataset = NpyDataset(image_folder=input_folder, output_img_folder=output_folder, transform=transform_train)
    sampler = get_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size = int(global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler = sampler,
        num_workers = num_workers,
        drop_last=True,
    )

    if rank == 0:
        logger.info(f"Dataset contains {len(train_dataset)}")
    
    # Prepare model for training
    update_ema(ema, model.module, 0)
    model.train() # Enable embedding dropout
    ema.eval() # EMA model always in eval mode
    image_encoder.eval() # Image encoder always in eval mode

    # Variables for monitoring/logging
    if init_from_pretrained_ckpt:
        train_steps = init_train_steps
    else:
        train_steps = 0

    log_steps = 0
    running_loss = 0
    start_time = time()

    if rank == 0:
        logger.info(f"Start training for {epochs} epochs")
    for epoch in epochs:
        sampler.set_epoch(epoch) # to shuffle the dataset
        if rank == 0:
            logger.info(f"Beginning epoch {epoch}")
        item = 0
        for x_in, z_out in train_loader:
            item += 1
            
            x_in = torch.cat([x_in] * 3, dim = 1)
            z_out = torch.cat([z_out] * 3, dim = 1)

            x_in = x_in.to(device)
            z_out = z_out.to(device)

            with torch.no_grad():
                if not torch.all((z_out >= -1) & (z_out <= 1)):
                    z_out = ((z_out - z_out.min()) * 1.0 / (z_out.max() - z_out.min())) * 2.0 - 1.0
                # .latent_dist: Represents the distribution of the latent
                # .sample(): Draws a random sample from the latent
                # .mul_(0.18215): Multiplies the sample by 0.18215 (refered in the Diffusion pipeline)
                z_out = vae.encode(z_out).latent_dist.sample().mul_(0.18215)
                x_in_vae = vae.encode(x_in).latent_dist.sample().mul_(0.18215)
                weight, x_in_2 = ct_encoder(x_in_vae)
                x_in = image_encoder(x_in)

            t = torch.randint(0, diffusion.num_timesteps, (1,), device=device)
        
            model_kwargs = dict(y=x_in, y2=x_in_2, w=weight)

            with autocast(enabled=autocast_bool):
                loss_dict = diffusion.training_losses(model, z_out, t, **model_kwargs)
                loss = loss_dict["loss"].mean()

            if rank == 0 and wandb:
                wandb.log({"loss": loss.item()})
            
            if torch.isnan(loss).any():
                logger.info(f"NaN detected... Ignore losses...")
                continue

            with autocast(enabled=autocast_bool):
                scaler.scale(loss).backward()

            if train_steps % accumulation_steps == 0:
                scaler.step(opt)
                scaler.update()
                update_ema(ema, model.module)
                opt.zero_grad()

            # Log loss value
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % log_every:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_every / (end_time - start_time)

                # Reduce loss history over all processes
                percent_epoch = (global_batch_size // dist.get_world_size()) * item / len(train_loader) * 100
                avg_loss = torch.tensor(running_loss / log_steps, device = device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM) # aggregate tensors from all processes
                avg_loss = avg_loss.item() / dist.get_world_size() # average loss over all processes in distributed training

                if rank == 0:
                    logger.info(f"({percent_epoch:.1f}%) | (step={train_steps:07d}) Train loss: {avg_loss:.4f}, Train steps/sec: {steps_per_sec:.2f}")

                # Reset monitoring variables
                running_loss = 0
                log_steps = 0
                start_time = time()
            
            # Save diffusion checkpoint
            if train_steps % ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        # "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    if rank == 0:
                        logger.info(f"Saved checkpoint at  {checkpoint_path}")
                    dist.barrier() # Ensure all processes have saved the checkpoint before continuing\
    model.eval() # Disable embedding dropout


    logger.info('Training complete')
    if rank == 0 and wandb:
        wandb.finish()
    dist.destroy_process_group() # Clean up

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="Enable WandB.")
    parser.add_argument("--autocast", action="store_true", help="Whether to use half-precision training.")
    args = parser.parse_args()
    cli_config = OmegaConf.create({k: v for k, v in args.__dict__.items() if v is not None and k != 'config'})
    args = OmegaConf.merge(OmegaConf.load(args.config), cli_config)

    main(model, global_batch_size, global_seed, results_dir, epochs, image_size, init_from_pretrained_ckpt, pretrain_ckpt_path, ct_ckpt, autocast_bool, lr, lr_pretrained, input_folder, output_folder, num_workers, init_train_steps, accumulation_steps, log_every, ckpt_every)















            
                









    







