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

    # --- 1. Global Fidelity Plots (Individual Files) ---
    print("Generating Global Spectral Fidelity plots...")
    
    # Calculate baseline PSD once
    f_base, p_base = get_psd(df_base[feature_cols].values)
    
    # A. Baseline Only Plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(f_base, p_base, label="Baseline (Real)", color='#335067', lw=2)
    plt.title("Global Fidelity: Baseline Power Spectral Density (PSD)", fontsize=14, fontweight='bold')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    plt.legend()
    plt.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "global_fidelity_baseline.png"), dpi=200)
    plt.close()

    # B. Individual Ratio Plots (Comparison against Baseline)
    for i, (r_name, path) in enumerate(recon_files.items()):
        if os.path.exists(path):
            df_r = pd.read_csv(path)
            f_r, p_r = get_psd(df_r[feature_cols].values)
            
            plt.figure(figsize=(10, 6))
            plt.semilogy(f_base, p_base, label="Baseline (Real)", color='#335067', lw=1.5, alpha=0.5)
            plt.semilogy(f_r, p_r, label=f"Synthetic Ratio {r_name}", color='#054984', lw=2)
            
            plt.title(f"Global Fidelity: Ratio {r_name} vs Baseline PSD", fontsize=14, fontweight='bold')
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power/Frequency (dB/Hz)")
            plt.legend()
            plt.grid(alpha=0.15)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"global_fidelity_ratio_{r_name}.png"), dpi=200)
            plt.close()

    # --- 2. Targeted Frequency Comparison for P6 & P7 (Individual Files) ---
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
        
        sensor_col = feature_cols[0]

        # A. Baseline for Participant
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for r in range(3):
            if r < len(rep_starts):
                seg = df_raw.iloc[rep_starts[r]:rep_starts[r]+4000][sensor_col].values * 1e6
                f, pxx = welch(seg, fs=2000, nperseg=256)
                axes[r].semilogy(f, pxx, color='#335067', lw=1.5)
                axes[r].set_title(f"Original Rep {r+1} PSD", fontweight='bold')
                axes[r].set_xlabel("Frequency (Hz)")
                axes[r].set_ylabel("dB/Hz")
            axes[r].grid(alpha=0.15)
        fig.suptitle(f"P{p_num} Baseline Spectral Analysis | Gesture: {first_g}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f"spectral_P{p_num}_baseline.png"))
        plt.close()

        # B. Ratios for Participant
        for r_name in ratios:
            r_path = recon_files[r_name]
            if not os.path.exists(r_path): continue
            df_r = pd.read_csv(r_path)
            
            target_id = df_r['gt'].iloc[0]
            indices_r = df_r.index[df_r['gt'] == target_id].to_numpy()
            breaks_r = np.where(np.diff(indices_r) > 1)[0]
            rep_starts_r = [indices_r[0]] + [indices_r[i+1] for i in breaks_r]
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for r in range(3):
                if r < len(rep_starts_r):
                    seg_r = df_r.iloc[rep_starts_r[r]:rep_starts_r[r]+4000][sensor_col].values
                    f, pxx = welch(seg_r, fs=2000, nperseg=256)
                    axes[r].semilogy(f, pxx, color='#054984', lw=2)
                    axes[r].set_title(f"Synthetic Ratio {r_name} Rep {r+1} PSD", fontweight='bold')
                    axes[r].set_xlabel("Frequency (Hz)")
                    axes[r].set_ylabel("dB/Hz")
                axes[r].grid(alpha=0.15)
            fig.suptitle(f"P{p_num} Ratio {r_name} Spectral Analysis | Gesture: {first_g}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(save_dir, f"spectral_P{p_num}_ratio_{r_name}.png"))
            plt.close()

    print(f"Fidelity reports saved in {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--vqvae_config', type=str, required=True)
    args = parser.parse_args()
    run_fidelity_check(args.config, args.vqvae_config)
