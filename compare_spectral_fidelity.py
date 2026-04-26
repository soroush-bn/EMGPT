import os
import yaml
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import welch
from pathlib import Path

def calculate_features(signal):
    """Calculate RMS and MAV for a signal [Time, Channels]"""
    rms = np.sqrt(np.mean(signal**2, axis=0))
    mav = np.mean(np.abs(signal), axis=0)
    return rms, mav

def get_psd(signal, fs=2000):
    """Calculate PSD for a signal [Time, Channels]"""
    # Average PSD across all channels
    f, pxx = welch(signal, fs=fs, axis=0, nperseg=256)
    return f, np.mean(pxx, axis=1)

def run_fidelity_check(config_path, vq_config_path):
    with open(config_path, "r") as f:
        tr_config = yaml.safe_load(f)
    with open(vq_config_path, 'r') as f:
        vq_config_full = yaml.safe_load(f)
        vq_config = vq_config_full.get('vqvae', vq_config_full)

    exp_name = tr_config['exp_name']
    model_dir = os.path.join("models", exp_name)
    save_dir = os.path.join(model_dir, "fidelity_reports")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Setup Data Paths
    # Real data (raw baseline from participant files or preprocessed unseen)
    # To compare correctly, we use the reconstructed SEEN data (ratio 70_5 contains 70% real)
    # But for a true baseline, let's use the reconstructed unseen if available
    baseline_path = os.path.join(model_dir, "unseen_reconstructed_final.csv")
    if not os.path.exists(baseline_path):
        # Fallback to any reconstructed file or preprocessed data
        baseline_path = f"./VQVAE/models/{vq_config['name']}/unseen_data_preprocessed.csv"

    ratios = ["70_5", "60_15", "50_25", "25_50"]
    recon_files = {r: os.path.join(model_dir, f"synthetic_{r}_reconstructed.csv") for r in ratios}
    
    print(f"Loading baseline: {baseline_path}")
    df_base = pd.read_csv(baseline_path)
    feature_cols = [c for c in df_base.columns if c != 'gt']
    
    results = []

    # --- Analysis 1: Statistical Distribution (RMS/MAV) ---
    print("Analyzing Statistical Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    all_rms = {"Baseline": []}
    for r in ratios: all_rms[r] = []

    # Sample RMS across the dataset
    # We'll take chunks of 300 samples (window size)
    ws = 300
    for i in range(0, len(df_base)-ws, ws):
        chunk = df_base[feature_cols].iloc[i:i+ws].values
        rms, _ = calculate_features(chunk)
        all_rms["Baseline"].append(np.mean(rms))

    for r, path in recon_files.items():
        if os.path.exists(path):
            df_r = pd.read_csv(path)
            for i in range(0, len(df_r)-ws, ws):
                chunk = df_r[feature_cols].iloc[i:i+ws].values
                rms, _ = calculate_features(chunk)
                all_rms[r].append(np.mean(rms))

    # Plot Boxplots
    axes[0].boxplot([all_rms[k] for k in all_rms.keys() if all_rms[k]], labels=[k for k in all_rms.keys() if all_rms[k]])
    axes[0].set_title("RMS Amplitude Distribution")
    axes[0].set_ylabel("RMS Value")
    axes[0].grid(alpha=0.2)

    # --- Analysis 2: Spectral Fidelity (PSD) ---
    print("Analyzing Spectral Fidelity...")
    f_base, p_base = get_psd(df_base[feature_cols].values)
    axes[1].semilogy(f_base, p_base, label="Baseline (Real)", color='black', lw=2)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(ratios)))
    for i, (r, path) in enumerate(recon_files.items()):
        if os.path.exists(path):
            df_r = pd.read_csv(path)
            f_r, p_r = get_psd(df_r[feature_cols].values)
            axes[1].semilogy(f_r, p_r, label=f"Ratio {r}", color=colors[i], alpha=0.8)

    axes[1].set_title("Power Spectral Density (PSD) Comparison")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power/Frequency (dB/Hz)")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "global_fidelity_metrics.png"), dpi=200)
    plt.close()

    # --- Analysis 3: Targeted Frequency Comparison for P6 & P7 ---
    # (Similar to vis_original_vs_reconstructed but in freq domain)
    participants = [
        (6, vq_config['participants_list_ids'][5]),
        (7, vq_config['participants_list_ids'][6])
    ]

    for p_num, p_id in participants:
        raw_path = os.path.join(vq_config["raw_data_path"], p_id, vq_config["df_raw_name"])
        if not os.path.exists(raw_path): continue
        
        df_raw = pd.read_csv(raw_path)
        df_gest = df_raw[df_raw['label'].notna() & (df_raw['label'] != 'rest')].copy()
        if df_gest.empty: continue
        first_g = df_gest['label'].iloc[0]
        
        indices = df_raw.index[df_raw['label'] == first_g].to_numpy()
        breaks = np.where(np.diff(indices) > 1)[0]
        rep_starts = [indices[0]] + [indices[i+1] for i in breaks]
        
        # Setup Figure
        n_rows = 1 + len(ratios)
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 4 * n_rows))
        plt.subplots_adjust(hspace=0.4)
        
        sensor_col = feature_cols[0] # Plot first channel PSD

        # Row 0: Raw Spectral (First 3 Reps)
        for r in range(3):
            if r < len(rep_starts):
                seg = df_raw.iloc[rep_starts[r]:rep_starts[r]+4000][sensor_col].values * 1e6
                f, pxx = welch(seg, fs=2000, nperseg=256)
                axes[0, r].semilogy(f, pxx, color='black')
                axes[0, r].set_title(f"Original Rep {r+1} PSD", fontweight='bold')
            axes[0, r].grid(alpha=0.2)

        # Rows 1+: Reconstruction Spectral
        for row_idx, r_name in enumerate(ratios):
            r_path = recon_files[r_name]
            if not os.path.exists(r_path): continue
            df_r = pd.read_csv(r_path)
            
            target_id = df_r['gt'].iloc[0]
            indices_r = df_r.index[df_r['gt'] == target_id].to_numpy()
            breaks_r = np.where(np.diff(indices_r) > 1)[0]
            rep_starts_r = [indices_r[0]] + [indices_r[i+1] for i in breaks_r]
            
            for r in range(3):
                if r < len(rep_starts_r):
                    seg_r = df_r.iloc[rep_starts_r[r]:rep_starts_r[r]+4000][sensor_col].values
                    f, pxx = welch(seg_r, fs=2000, nperseg=256)
                    axes[row_idx+1, r].semilogy(f, pxx, color='red')
                    if r == 0:
                        axes[row_idx+1, r].set_ylabel(f"Ratio {r_name} PSD", fontweight='bold')
                axes[row_idx+1, r].grid(alpha=0.2)

        fig.suptitle(f"Spectral Comparison P{p_num} | Gesture: {first_g}", fontsize=16)
        plt.savefig(os.path.join(save_dir, f"spectral_fidelity_P{p_num}.png"), bbox_inches='tight')
        plt.close(fig)

    print(f"Fidelity reports saved in {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--vqvae_config', type=str, required=True)
    args = parser.parse_args()
    run_fidelity_check(args.config, args.vqvae_config)
