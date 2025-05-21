#!/usr/bin/env python3
"""Training script for grayscale images with a fixed generator.

This script is a simplified variant of ``train.py`` that bundles a
``GrayscaleDataset`` implementation and a small constant LR scheduler.
It targets small grayscale datasets and uses gradient accumulation to
keep memory usage down.
"""

import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import accelerate
import numpy as np
import torch
from PIL import Image
from torch import optim
from torch.utils import data
from torchvision import utils
from tqdm.auto import tqdm

import k_diffusion as K
import safetensors.torch as safetorch


class GrayscaleDataset(data.Dataset):
    """Loads grayscale PNG images from a folder tree."""

    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.image_paths = []
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith(".png"):
                        self.image_paths.append(os.path.join(folder_path, filename))
        print(f"Found {len(self.image_paths)} images for the dataset")

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.image_paths)

    def __getitem__(self, idx: int):  # type: ignore[override]
        img_path = self.image_paths[idx]
        with Image.open(img_path) as img:
            if img.mode != "L":
                img = img.convert("L")
            img_tensor = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
            img_tensor = img_tensor.unsqueeze(0)
        aug_cond = torch.zeros(9)
        return img_tensor, 0, aug_cond


class SimpleConstantLR:
    """Minimal constant LR scheduler replacement."""

    def __init__(self, optimizer: optim.Optimizer, warmup: float = 0.0) -> None:
        self.optimizer = optimizer
        self.warmup = warmup
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1

    def state_dict(self) -> dict:
        return {"step_count": self.step_count}

    def load_state_dict(self, state_dict: dict) -> None:
        self.step_count = state_dict.get("step_count", 0)

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--batch-size", type=int, default=4, help="the batch size")
    p.add_argument("--config", type=str, required=True, help="the configuration file")
    p.add_argument("--demo-every", type=int, default=250, help="demo grid interval")
    p.add_argument("--end-step", type=int, default=None, help="the step to end at")
    p.add_argument("--mixed-precision", type=str, help="mixed precision type")
    p.add_argument("--name", type=str, default="model", help="the name of the run")
    p.add_argument("--num-workers", type=int, default=2, help="data loader workers")
    p.add_argument("--save-every", type=int, default=1000, help="checkpoint interval")
    p.add_argument("--checkpoint", action="store_true", help="use gradient checkpointing")
    args = p.parse_args()

    gradient_accumulation_steps = 2
    accelerator = accelerate.Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    device = accelerator.device
    print(f"Using device: {device}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Batch size: {args.batch_size} (effective: {args.batch_size * gradient_accumulation_steps})")

    config = K.config.load_config(args.config)
    model_config = config["model"]
    opt_config = config["optimizer"]
    sched_config = config["lr_sched"]
    ema_sched_config = config["ema_sched"]

    assert model_config["input_channels"] == 1, "Model must be for grayscale"
    image_size = model_config["input_size"]

    model = K.config.make_model(config)
    model_ema = deepcopy(model)

    if args.checkpoint:
        print("Enabling gradient checkpointing...")
        with K.models.checkpointing(True):
            pass

    print(f"Parameters: {K.utils.n_params(model):,}")

    opt = optim.AdamW(
        model.parameters(),
        lr=opt_config["lr"],
        betas=tuple(opt_config["betas"]),
        eps=opt_config["eps"],
        weight_decay=opt_config["weight_decay"],
    )

    sched = SimpleConstantLR(opt, warmup=sched_config["warmup"])
    ema_sched = K.utils.EMAWarmup(power=ema_sched_config["power"], max_value=ema_sched_config["max_value"])

    dataset = GrayscaleDataset(config["dataset"]["location"])
    train_dl = data.DataLoader(
        dataset,
        args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )

    model, model_ema, opt, train_dl = accelerator.prepare(model, model_ema, opt, train_dl)

    denoiser = K.config.make_denoiser_wrapper(config)(model)
    denoiser_ema = K.config.make_denoiser_wrapper(config)(model_ema)

    sigma_min = model_config["sigma_min"]
    sigma_max = model_config["sigma_max"]
    sample_density = K.config.make_sample_density(model_config)

    demo_seed = 42
    step = 0
    ema_stats = {}
    losses = []
    elapsed = 0.0
    state_path = Path(f"{args.name}_state.json")

    @torch.no_grad()
    @K.utils.eval_mode(denoiser_ema)
    def demo() -> None:
        print("Generating demo samples...")
        filename = f"{args.name}_demo_{step:08}.png"
        n = 4
        torch.manual_seed(demo_seed + step)
        x = torch.randn([n, model_config["input_channels"], image_size[0], image_size[1]]).to(device) * sigma_max
        sigmas = K.sampling.get_sigmas_karras(50, sigma_min, sigma_max, rho=7.0, device=device)
        x0 = K.sampling.sample_dpmpp_2m_sde(denoiser_ema, x, sigmas, eta=0.0, solver_type="heun")
        grid = utils.make_grid(x0, nrow=2)
        K.utils.to_pil_image(grid).save(filename)
        print(f"Saved {filename}")

    def save() -> None:
        filename = f"{args.name}_{step:08}.pth"
        print(f"Saving to {filename}...")
        obj = {
            "config": config,
            "model": accelerator.unwrap_model(model).state_dict(),
            "model_ema": accelerator.unwrap_model(model_ema).state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "ema_sched": ema_sched.state_dict(),
            "step": step,
            "ema_stats": ema_stats,
            "demo_seed": demo_seed,
        }
        accelerator.save(obj, filename)
        try:
            safetorch.save_file(
                accelerator.unwrap_model(model_ema).state_dict(),
                f"{args.name}_{step:08}.safetensors",
                metadata={"config": json.dumps(config)},
            )
        except Exception as e:  # pragma: no cover - saving is best effort
            print(f"Error saving safetensors: {e}")
        if accelerator.is_main_process:
            state_obj = {"latest_checkpoint": filename}
            json.dump(state_obj, open(state_path, "w"))

    try:
        torch.cuda.empty_cache()
        while True:
            progress = tqdm(train_dl)
            for batch in progress:
                with accelerator.accumulate(model):
                    if device.type == "cuda":
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                    else:
                        start = time.time()

                    reals, _, aug_cond = batch
                    if step == 0:
                        print(f"Input tensor shape: {reals.shape}, dtype: {reals.dtype}")
                    noise = torch.randn_like(reals)
                    sigma = sample_density([reals.shape[0]], device=device)
                    with K.models.checkpointing(args.checkpoint):
                        loss_val = denoiser.loss(reals, noise, sigma, aug_cond=aug_cond)
                    loss = accelerator.gather(loss_val).mean().item()
                    losses.append(loss)
                    accelerator.backward(loss_val.mean())
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                        sched.step()
                        opt.zero_grad()
                        ema_decay = ema_sched.get_value()
                        K.utils.ema_update_dict(ema_stats, {"loss": loss}, ema_decay)
                        K.utils.ema_update(model, model_ema, ema_decay)
                        ema_sched.step()
                if device.type == "cuda":
                    end.record()
                    torch.cuda.synchronize()
                    elapsed += start.elapsed_time(end) / 1000.0
                else:
                    elapsed += time.time() - start
                if accelerator.sync_gradients:
                    if len(losses) >= 10:
                        avg_loss = sum(losses) / len(losses)
                        losses = []
                        progress.set_description(f"Step: {step}, Loss: {avg_loss:.4f}")
                    if step % args.demo_every == 0:
                        torch.cuda.empty_cache()
                        demo()
                    if step % args.save_every == 0 and step > 0:
                        torch.cuda.empty_cache()
                        save()
                    if args.end_step is not None and step >= args.end_step:
                        print(f"Reached end step {args.end_step}. Saving final checkpoint.")
                        save()
                        return
                    step += 1
                    if step % 20 == 0:
                        torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print("Training interrupted.")
        if step > 0:
            save()
    except Exception as e:  # pragma: no cover - best effort
        print(f"Error during training: {e}")
        if step > 0:
            try:
                save()
            except Exception:
                print("Could not save checkpoint after error")


if __name__ == "__main__":
    main()
