"""
Decodes synthetic discrete tokens back to raw signals and visualizes them.
"""
import argparse
import os
import yaml
from decoder import VQVAESignalDecoder
import numpy as np
import matplotlib
import pathlib
import faulthandler
faulthandler.enable()

# Use 'Agg' backend so matplotlib doesn't try to open a window on the headless server
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_synthetic_signals(signals, labels, save_path, max_plots=9):
    """
    Creates a grid plot of the generated multi-channel signals.
    signals shape: (Samples, TimeSteps, Channels)
    """
    num_samples = min(len(signals), max_plots)
    
    # Determine grid size (e.g., 3x3 for 9 samples)
    cols = 3
    rows = int(np.ceil(num_samples / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows), squeeze=False)
    axes = axes.flatten()
    
    # Define a color-blind friendly blue palette
    blue_palette = [
        '#054984', '#335067', '#0072b2', '#56b4e9', 
        '#009e73', '#004d40', '#1a237e', '#3f51b5'
    ]
    
    for i in range(num_samples):
        if i >= num_samples: break
        ax = axes[i]
        signal = signals[i]  # Shape: (TimeSteps, Channels)
        label = labels[i]
        
        # Plot each channel
        num_channels = signal.shape[1]
        for c in range(num_channels):
            color = blue_palette[c % len(blue_palette)]
            ax.plot(signal[:, c], label=f'Ch {c+1}', color=color, alpha=0.8, linewidth=1.2)
            
        ax.set_title(f"Synthetic Gesture Class: {label}", fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize='small')

    for j in range(num_samples, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {save_path}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to Transformer config (replicate_small.yaml)')
    parser.add_argument("--vqvae_config", type=str, default=None, help="Path to VQ-VAE config")
    parser.add_argument("--vqvae_ckpt", type=str, default=None, help="Path to VQ-VAE weights")
    parser.add_argument("--num_plots", type=int, default=9, help="Number of generated samples to plot")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    with open(args.config, "r") as file:
        transformer_config = yaml.safe_load(file)
    
    exp_name = transformer_config['exp_name']
    model_files_base_directory = os.path.join(pathlib.Path(__file__).resolve().parent.__str__(), "models")
    save_dir = os.path.join(model_files_base_directory, exp_name)

    # Resolve VQ-VAE paths
    vq_config_path = args.vqvae_config
    vq_ckpt_path = args.vqvae_ckpt

    if vq_config_path is None or vq_ckpt_path is None:
        # If not provided, try to derive from VQ_CONFIG env or default to tuned2
        vq_config_path = vq_config_path or "./VQVAE/tuned_config2.yaml"
        with open(vq_config_path, 'r') as file:
            vq_cfg = yaml.safe_load(file)
        vq_name = vq_cfg.get('name', 'tuned2')
        vq_ckpt_path = vq_ckpt_path or f"./VQVAE/models/{vq_name}/final_model.pth"

    with open(vq_config_path, 'r') as file:
        vqvae_config_full = yaml.safe_load(file)
        vqvae_config = vqvae_config_full.get('vqvae', vqvae_config_full)
        
    # 2. Initialize the decoder
    decoder = VQVAESignalDecoder(
        vqvae_model_path=vq_ckpt_path, 
        vqvae_config=vqvae_config
    )
    
    # 3. Decode the generated tokens back to continuous signals
    print(f"\n--- Starting Decoding Process for {exp_name} ---")
    
    # Search Priority:
    # 1. 'seen_synthetic_df' (latest generation naming)
    # 2. 'unseen_synthetic' (older generation naming)
    # 3. 'synthetic_df' (fallback)
    
    sample_file = None
    
    # Try seen_synthetic_df patterns
    potential_seen = sorted([f for f in os.listdir(save_dir) if f.startswith("seen_synthetic_df_") and f.endswith(".csv")], reverse=True)
    if potential_seen:
        # Prefer the 25_50 ratio for visualization if available, otherwise just the first one
        favored = "seen_synthetic_df_25_50.csv"
        sample_file = os.path.join(save_dir, favored if favored in potential_seen else potential_seen[0])
        
    if not sample_file or not os.path.exists(sample_file):
        # Try unseen_synthetic patterns
        potential_unseen = sorted([f for f in os.listdir(save_dir) if f.startswith("unseen_synthetic_") and f.endswith(".csv")], reverse=True)
        if potential_unseen:
            sample_file = os.path.join(save_dir, potential_unseen[0])
            
    if not sample_file or not os.path.exists(sample_file):
        # Try old generic synthetic_df patterns
        potential_old = sorted([f for f in os.listdir(save_dir) if f.startswith("synthetic_df_") and f.endswith(".csv")], reverse=True)
        if potential_old:
            sample_file = os.path.join(save_dir, potential_old[0])

    if sample_file and os.path.exists(sample_file):
        print(f"Processing tokens from: {sample_file}")
        raw_signals, labels = decoder.decode_dataset(
            csv_path=sample_file,
            save_dir=save_dir
        )
        
        # 4. Visualize and save the plots
        print("\n--- Starting Visualization ---")
        plot_filename = os.path.join(save_dir, "synthetic_signals_grid.png")
        plot_synthetic_signals(raw_signals, labels, save_path=plot_filename, max_plots=args.num_plots)
    else:
        print(f"No synthetic data found to visualize in {save_dir}")
