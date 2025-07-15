#!/usr/bin/env python3

"""
Trains a Karras et al. (2022) diffusion model (EDM) on 1D well log data.
This version is cleaned up to focus solely on training and periodic saving,
with all evaluation logic removed to prevent errors and streamline the process.
"""
import argparse
from copy import deepcopy
import math
import json
from pathlib import Path
import os
import sys

import accelerate
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import optim
from torch.utils import data
from tqdm.auto import tqdm
import numpy as np

import k_diffusion as K

# --- WellLogDataset SINIFI DOĞRUDAN BURAYA EKLENDİ ---
class WellLogDataset(data.Dataset):
    """Loads 1-D well log samples stored as `.npy` files."""
    def __init__(self, root: str, length: int) -> None:
        self.root = Path(root)
        self.paths = sorted(self.root.glob('*.npy'))
        if not self.paths:
            raise RuntimeError(f'no .npy files found in {self.root}')
        self.length = length

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        arr = np.load(self.paths[index]).astype(np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        tensor = torch.from_numpy(arr)
        if tensor.shape[-1] > self.length:
            start = torch.randint(0, tensor.shape[-1] - self.length + 1, ()).item()
            tensor = tensor[..., start:start + self.length]
        elif tensor.shape[-1] < self.length:
            pad = self.length - tensor.shape[-1]
            tensor = F.pad(tensor, (0, pad))
        tensor = tensor.unsqueeze(-1)
        aug_cond = torch.zeros(9, dtype=torch.float32)
        return tensor, tensor.clone(), aug_cond
# -------------------------------------------------------------------------

def ensure_distributed():
    """Initialize default process group for distributed training if not already initialized."""
    if not dist.is_initialized():
        dist.init_process_group(world_size=1, rank=0, store=dist.HashStore())

def main():
    p = argparse.ArgumentParser(description="Trains a Karras et al. (2022) diffusion model.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', type=str, required=True, help='path to the JSON configuration file')
    p.add_argument('--name', type=str, default='model', help='run name for checkpoint filenames')
    p.add_argument('--batch-size', type=int, default=8, help='batch size per process')
    p.add_argument('--num-workers', type=int, default=2, help='number of DataLoader worker processes')
    p.add_argument('--grad-accum-steps', type=int, default=1, help='gradient accumulation steps')
    p.add_argument('--mixed-precision', type=str, choices=['no','fp16','bf16'], help='mixed precision mode')
    p.add_argument('--resume', type=str, help='path to checkpoint to resume training from')
    p.add_argument('--seed', type=int, help='random seed for reproducibility')
    p.add_argument('--lr', type=float, help='override learning rate from config')
    p.add_argument('--end-step', type=int, help='stop training after this many steps')
    p.add_argument('--demo-every', type=int, default=5000, help='save a demo sample every N steps')
    p.add_argument('--save-every', type=int, default=10000, help='save a model checkpoint every N steps')
    p.add_argument('--sample-n', type=int, default=16, help='number of samples to generate for demo')
    args = p.parse_args()

    # Setup
    mp.set_start_method('spawn', force=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps,
                                         mixed_precision=args.mixed_precision)
    ensure_distributed()
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)

    # YENİ EKLENEN BİLGİLENDİRME KISMI
    if accelerator.is_main_process:
        print("\n--- Checkpoint Bilgisi ---")
        print(f"Eğitim checkpoint'leri her {args.save_every} adımda bir kaydedilecektir.")
        print(f"Dosya adları '{args.name}_[adım_sayısı].pth' formatında olacaktır.")
        print(f"Kaydedileceği dizin: {os.getcwd()}")
        print("--------------------------\n")

    if args.seed is not None:
        seeds = torch.randint(-2**63, 2**63 - 1, (accelerator.num_processes,),
                              generator=torch.Generator().manual_seed(args.seed))
        torch.manual_seed(int(seeds[accelerator.process_index]))

    # Configs
    config = K.config.load_config(args.config)
    model_config, dataset_config, opt_config, sched_config, ema_sched_config = \
        config['model'], config['dataset'], config['optimizer'], config['lr_sched'], config['ema_sched']

    # Model
    inner_model = K.config.make_model(config)
    inner_model_ema = deepcopy(inner_model)
    if accelerator.is_main_process:
        print(f'Model parameters: {K.utils.n_params(inner_model):,}')

    # Dataset & DataLoader
    if dataset_config['type'] == 'custom':
        ds_conf = dataset_config.get('config', {})
        train_set = WellLogDataset(root=ds_conf['root'], length=ds_conf['length'])
    else:
        raise ValueError(f"This script only supports 'custom' dataset type; got '{dataset_config['type']}'")
    if accelerator.is_main_process:
        print(f'Dataset loaded: {len(train_set):,} samples')
    train_dl = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    # Optimizer & Schedulers
    opt = optim.AdamW(inner_model.parameters(),
                      lr=(args.lr or opt_config['lr']),
                      betas=tuple(opt_config['betas']),
                      eps=opt_config['eps'],
                      weight_decay=opt_config['weight_decay'])
    sched = K.utils.ConstantLRWithWarmup(opt, warmup=sched_config['warmup'])
    ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'], max_value=ema_sched_config['max_value'])

    # Prepare with Accelerate
    model, model_ema, opt, train_dl = accelerator.prepare(
        inner_model, inner_model_ema, opt, train_dl)
    denoiser = K.config.make_denoiser_wrapper(config)(model)
    denoiser_ema = K.config.make_denoiser_wrapper(config)(model_ema)

    @torch.no_grad()
    @K.utils.eval_mode(denoiser_ema)
    def generate_demo(step):
        if accelerator.is_main_process:
            tqdm.write(f'Generating demo samples at step {step}...')
            x = torch.randn([args.sample_n, model_config['input_channels'], model_config['input_size'][0], 1], device=device) * model_config['sigma_max']
            sigmas = K.sampling.get_sigmas_karras(50, model_config['sigma_min'], model_config['sigma_max'], rho=7.0, device=device)
            x_0 = K.sampling.sample_dpmpp_2m_sde(denoiser_ema, x, sigmas, disable=True)
            demo_file = f'{args.name}_demo_{step:08d}.pth'
            torch.save(x_0, demo_file)
            print(f"[step {step}] Saved demo samples to {demo_file}")

    def save_checkpoint(step, epoch):
        if accelerator.is_main_process:
            ckpt_file = f'{args.name}_ckpt_{step:08d}.pth'
            print(f"[step {step}] Saving checkpoint to {ckpt_file}")
            accelerator.save({
                'model': accelerator.unwrap_model(model).state_dict(),
                'model_ema': accelerator.unwrap_model(model_ema).state_dict(),
                'opt': opt.state_dict(),
                'step': step,
                'epoch': epoch
            }, ckpt_file)

    # Training Loop
    step = 0
    epoch = 0
    print("Setup complete. Starting training loop...")
    try:
        while True:
            for batch in tqdm(train_dl, disable=not accelerator.is_main_process):
                with accelerator.accumulate(model):
                    reals, _, aug_cond = batch
                    noise = torch.randn_like(reals)
                    sigma = K.config.make_sample_density(model_config)([reals.size(0)], device=device)
                    loss = denoiser.loss(reals, noise, sigma, aug_cond=aug_cond).mean()
                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        opt.step()
                        sched.step()
                        opt.zero_grad()
                        ema_decay = ema_sched.get_value()
                        K.utils.ema_update(model, model_ema, ema_decay)
                        ema_sched.step()

                if accelerator.is_main_process:
                    if step % 100 == 0:
                        print(f"Epoch {epoch}, Step {step}: loss = {loss.item():.4f}")
                    if args.demo_every and step > 0 and step % args.demo_every == 0:
                        generate_demo(step)
                    if args.save_every and step > 0 and step % args.save_every == 0:
                        save_checkpoint(step, epoch)
                
                step += 1
                if args.end_step is not None and step >= args.end_step:
                    break
            
            if args.end_step is not None and step >= args.end_step:
                print("Reached end step.")
                break
            epoch += 1

    except KeyboardInterrupt:
        if accelerator.is_main_process:
            print("Training interrupted by user.")
    finally:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print("Saving final checkpoint...")
            save_checkpoint(step, epoch)
            print("Training loop has ended.")

if __name__ == '__main__':
    main()
