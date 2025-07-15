import argparse
import os
import torch
import numpy as np
from tqdm import trange
from pathlib import Path

import k_diffusion as K

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic 1D well logs from a trained diffusion model.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint (.pth) file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the model config JSON file.')
    parser.add_argument('-n', '--n-samples', type=int, default=16, help='Number of synthetic logs to generate.')
    parser.add_argument('--steps', type=int, default=50, help='Number of denoising steps.')
    parser.add_argument('--output-dir', type=str, default='synthetic_logs', help='Directory to save the generated logs.')
    parser.add_argument('--seed', type=int, help='Random seed for generation.')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Config
    config = K.config.load_config(args.config)
    model_config = config['model']
    height, width = model_config['input_size']
    channels = model_config['input_channels']
    print(f"Model expected input size: {height}x{width}, channels: {channels}")

    # Load Model
    print(f"Loading model checkpoint from '{args.checkpoint}'...")
    inner_model = K.config.make_model(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Use EMA weights for better quality, fall back to main model if EMA not present
    state_key = 'model_ema' if 'model_ema' in ckpt else 'model'
    inner_model.load_state_dict(ckpt[state_key])
    inner_model.eval().requires_grad_(False)
    print(f"Loaded model weights from '{state_key}' in checkpoint.")

    model = K.Denoiser(inner_model, sigma_data=model_config['sigma_data'])
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Seed set to {args.seed}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Sampling
    print(f"Generating {args.n_samples} synthetic logs with {args.steps} steps...")
    with torch.no_grad():
        x = torch.randn(args.n_samples, channels, height, width, device=device) * model_config['sigma_max']
        sigmas = K.sampling.get_sigmas_karras(args.steps, model_config['sigma_min'], model_config['sigma_max'], rho=7.0, device=device)
        x_0 = K.sampling.sample_dpmpp_2m_sde(model, x, sigmas, disable=False)

        # Save generated logs
        for i in trange(args.n_samples, desc="Saving logs"):
            log_array = x_0[i].cpu().numpy().squeeze()
            csv_path = os.path.join(args.output_dir, f"synthetic_log_{i+1:03d}.csv")
            np.savetxt(csv_path, log_array, delimiter=",", header="Vs_mean", comments="")
            
    print(f"Saved {args.n_samples} synthetic logs to directory: '{args.output_dir}'")

if __name__ == '__main__':
    main()
