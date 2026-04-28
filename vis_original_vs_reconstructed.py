import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import yaml
from viz_style import COLORS, apply_ax_style

def visualize_gesture_reconstruction(original_path, reconstructed_path, save_dir="./gesture_plots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Loading data from {original_path} and {reconstructed_path}...")
    if not os.path.exists(original_path) or not os.path.exists(reconstructed_path):
        print("Error: One or more data files not found.")
        return

    df_orig = pd.read_csv(original_path)
    df_recon = pd.read_csv(reconstructed_path)
    sensor_to_plot = [c for c in df_orig.columns if c != 'gt'][0] 

    df_orig['change'] = df_orig['gt'].diff().ne(0).astype(int)
    df_orig['block_id'] = df_orig['change'].cumsum()
    group = df_orig.groupby('block_id')
    block_indices = group.apply(lambda x: (x.index[0], x.index[-1], x['gt'].iloc[0])).values
    
    gesture_map = {}
    for start, end, g_id in block_indices:
        if g_id not in gesture_map: gesture_map[g_id] = []
        gesture_map[g_id].append((start, end))

    for g_id in sorted(gesture_map.keys()):
        all_reps = gesture_map[g_id]
        num_participants = 11
        for p_idx in range(num_participants):
            reps_per_p = max(1, len(all_reps) // num_participants)
            participant_reps = all_reps[p_idx * reps_per_p : (p_idx + 1) * reps_per_p]
            if len(participant_reps) == 0: continue

            n_rows = len(participant_reps)
            fig, axes = plt.subplots(n_rows, 2, figsize=(15, 3 * n_rows), squeeze=False)
            fig.suptitle(f"Participant {p_idx} | Gesture {g_id}", fontsize=16, fontweight='bold', color=COLORS['text_primary'])

            for r_idx, (start, end) in enumerate(participant_reps):
                if start >= len(df_recon): continue
                actual_end = min(end, len(df_recon))
                orig_segment = df_orig.iloc[start:actual_end][sensor_to_plot].values
                recon_segment = df_recon.iloc[start:actual_end][sensor_to_plot].values

                axes[r_idx, 0].plot(orig_segment, color=COLORS['secondary'], lw=1.5, alpha=0.7)
                apply_ax_style(axes[r_idx, 0], title="Original Data" if r_idx==0 else None, ylabel=f"Seg {r_idx + 1}")
                
                if len(recon_segment) > 0:
                    axes[r_idx, 1].plot(recon_segment, color=COLORS['primary'], lw=1.5, alpha=0.8)
                    apply_ax_style(axes[r_idx, 1], title="Reconstructed Data" if r_idx==0 else None, color_title=True)
                else:
                    axes[r_idx, 1].text(0.5, 0.5, "No Reconstructed Data", ha='center')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_name = f"participant{p_idx}_gt{int(g_id)}.png"
            plt.savefig(os.path.join(save_dir, save_name), dpi=100)
            plt.close()

def visualize_ratio_comparison(vq_config, tr_config, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    participants = [(6, vq_config['participants_list_ids'][5]), (7, vq_config['participants_list_ids'][6])]
    exp_name = tr_config['exp_name']
    base_model_dir = os.path.join("models", exp_name)
    
    recon_files = [f for f in os.listdir(base_model_dir) if f.startswith("synthetic_") and f.endswith("_reconstructed.csv")]
    ratios = sorted([f.replace("synthetic_", "").replace("_reconstructed.csv", "") for f in recon_files])
    if not ratios: return

    for p_num, p_id in participants:
        raw_path = os.path.join(vq_config["raw_data_path"], p_id, vq_config["df_raw_name"])
        if not os.path.exists(raw_path): continue
        df_raw = pd.read_csv(raw_path)
        df_gestures = df_raw[df_raw['label'].notna() & (df_raw['label'] != 'rest')].copy()
        if df_gestures.empty: continue
        first_gesture_name = df_gestures['label'].iloc[0]
        indices = df_raw.index[df_raw['label'] == first_gesture_name].to_numpy()
        breaks = np.where(np.diff(indices) > 1)[0]
        rep_starts = [indices[0]] + [indices[i+1] for i in breaks]
        
        df_sample_recon = pd.read_csv(os.path.join(base_model_dir, f"synthetic_{ratios[0]}_reconstructed.csv"))
        sensor_to_plot = [c for c in df_sample_recon.columns if c != 'gt'][0]
        
        n_rows = 1 + len(ratios)
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 4 * n_rows))
        
        # Row 0: Raw
        for r in range(3):
            if r < len(rep_starts):
                seg = df_raw.iloc[rep_starts[r] : rep_starts[r] + 4000][sensor_to_plot].values * 1e6
                axes[0, r].plot(seg, color=COLORS['secondary'], alpha=0.7, lw=1.5)
                apply_ax_style(axes[0, r], title=f"Original Rep {r+1}")
        
        # Rows 1+: Reconstructions
        for row_idx, ratio in enumerate(ratios):
            df_recon = pd.read_csv(os.path.join(base_model_dir, f"synthetic_{ratio}_reconstructed.csv"))
            target_id = df_recon['gt'].iloc[0] 
            indices_recon = df_recon.index[df_recon['gt'] == target_id].to_numpy()
            breaks_recon = np.where(np.diff(indices_recon) > 1)[0]
            rep_starts_recon = [indices_recon[0]] + [indices_recon[i+1] for i in breaks_recon]
            
            for r in range(3):
                if r < len(rep_starts_recon):
                    seg_recon = df_recon.iloc[rep_starts_recon[r] : rep_starts_recon[r] + 4000][sensor_to_plot].values
                    axes[row_idx+1, r].plot(seg_recon, color=COLORS['primary'], alpha=0.8, lw=1.5)
                    apply_ax_style(axes[row_idx+1, r], ylabel=f"Ratio {ratio}" if r==0 else None, color_title=True)

        fig.suptitle(f"Participant {p_num} | Gesture: {first_gesture_name}\nPrompt Ratio Comparison", fontsize=16, fontweight='bold', color=COLORS['text_primary'])
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f"ratio_comparison_P{p_num}.png"))
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--vqvae_config', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f: tr_config = yaml.safe_load(f)
    with open(args.vqvae_config, 'r') as f:
        vq_config_full = yaml.safe_load(f)
        vq_config = vq_config_full.get('vqvae', vq_config_full)

    base_model_dir = os.path.join("models", tr_config['exp_name'])
    visualize_gesture_reconstruction(f"./VQVAE/models/{vq_config['name']}/unseen_data_preprocessed.csv", 
                                     f"{base_model_dir}/unseen_reconstructed_final.csv", 
                                     save_dir=args.save_dir if args.save_dir else f"{base_model_dir}/unseen_gesture_plots")
    visualize_ratio_comparison(vq_config, tr_config, os.path.join(base_model_dir, "ratio_comparisons"))
