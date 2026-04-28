import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import yaml

def visualize_gesture_reconstruction(original_path, reconstructed_path, save_dir="./gesture_plots"):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Loading data from {original_path} and {reconstructed_path}...")
    if not os.path.exists(original_path) or not os.path.exists(reconstructed_path):
        print("Error: One or more data files not found.")
        return

    df_orig = pd.read_csv(original_path)
    df_recon = pd.read_csv(reconstructed_path)

    # Ensure we are only looking at sensor columns (drop gt for plotting)
    feature_cols = [c for c in df_orig.columns if c != 'gt']
    # We will plot the first sensor for clarity, but you can change this index
    sensor_to_plot = feature_cols[0] 

    print("Identifying gesture segments...")
    # Find indices where the gesture ID changes
    df_orig['change'] = df_orig['gt'].diff().ne(0).astype(int)
    df_orig['block_id'] = df_orig['change'].cumsum()

    group = df_orig.groupby('block_id')
    block_indices = group.apply(lambda x: (x.index[0], x.index[-1], x['gt'].iloc[0])).values
    
    gesture_map = {}
    for start, end, g_id in block_indices:
        if g_id not in gesture_map:
            gesture_map[g_id] = []
        gesture_map[g_id].append((start, end))

    print(f"Detected {len(gesture_map)} unique gestures.")

    for g_id in sorted(gesture_map.keys()):
        all_reps = gesture_map[g_id]
        print(f"Processing Gesture {g_id}: Found {len(all_reps)} total segments.")

        # Determine how many participants we can plot (assuming segments are grouped by participant)
        # For unseen data, we might have fewer reps per person.
        
        # We'll group reps by participant more safely
        # Assuming each participant has some number of reps
        num_participants = 11
        for p_idx in range(num_participants):
            # In a clean dataset, reps are usually sequential for participants
            # But let's just take a slice and plot what we have
            # If your dataset has a different number of reps, this logic helps visualize whatever is there
            
            # Estimate which reps belong to this participant
            # If we don't know for sure, we at least won't crash
            reps_per_p = max(1, len(all_reps) // num_participants)
            participant_reps = all_reps[p_idx * reps_per_p : (p_idx + 1) * reps_per_p]

            if len(participant_reps) == 0:
                continue

            # Create a dynamic grid based on available reps
            n_rows = len(participant_reps)
            fig, axes = plt.subplots(n_rows, 2, figsize=(15, 3 * n_rows), squeeze=False)
            fig.suptitle(f"Participant {p_idx} | Gesture {g_id}\n(Left: Original, Right: Reconstructed)", fontsize=16)

            for r_idx, (start, end) in enumerate(participant_reps):
                # Ensure we don't go out of bounds of the reconstructed data
                if start >= len(df_recon):
                    print(f"  Warning: Segment {start}:{end} is beyond reconstructed data length ({len(df_recon)}). Skipping.")
                    continue
                
                # Adjust end if it exceeds recon length
                actual_end = min(end, len(df_recon))
                
                orig_segment = df_orig.iloc[start:actual_end][sensor_to_plot].values
                recon_segment = df_recon.iloc[start:actual_end][sensor_to_plot].values

                # Use color-blind friendly blue shades
                axes[r_idx, 0].plot(orig_segment, color='#335067', lw=1.5, alpha=0.7)
                axes[r_idx, 0].set_ylabel(f"Seg {r_idx + 1}", fontweight='bold')
                if r_idx == 0:
                    axes[r_idx, 0].set_title("Original Data", fontsize=14, fontweight='bold', color='#335067')
                
                if len(recon_segment) > 0:
                    axes[r_idx, 1].plot(recon_segment, color='#054984', lw=1.5, alpha=0.8)
                else:
                    axes[r_idx, 1].text(0.5, 0.5, "No Reconstructed Data", ha='center')
                    
                if r_idx == 0:
                    axes[r_idx, 1].set_title("Reconstructed Data", fontsize=14, fontweight='bold', color='#054984')
                
                axes[r_idx, 0].grid(alpha=0.15)
                axes[r_idx, 1].grid(alpha=0.15)
                for ax in axes[r_idx]:
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            save_name = f"participant{p_idx}_gt{int(g_id)}.png"
            plt.savefig(os.path.join(save_dir, save_name), dpi=100)
            plt.close()

    print(f"Done! All plots saved in '{save_dir}'")

def visualize_ratio_comparison(vq_config, tr_config, save_dir):
    """
    Plots original raw signal (top row, 3 reps) vs 
    reconstructions from different ratios (bottom rows).
    Target: Participants 6 and 7, first occurring gesture.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    participants = [
        (6, vq_config['participants_list_ids'][5]),
        (7, vq_config['participants_list_ids'][6])
    ]
    
    exp_name = tr_config['exp_name']
    model_files_base_directory = os.path.join(pathlib.Path(__file__).resolve().parent.__str__(), "models")
    base_model_dir = os.path.join(model_files_base_directory, exp_name)
    
    # 1. Identify all available reconstruction files
    recon_files = [f for f in os.listdir(base_model_dir) if f.startswith("synthetic_") and f.endswith("_reconstructed.csv")]
    # Extract ratios and sort them for consistent plotting
    ratios = sorted([f.replace("synthetic_", "").replace("_reconstructed.csv", "") for f in recon_files])
    
    if not ratios:
        print("No reconstructed synthetic datasets found.")
        return

    for p_num, p_id in participants:
        raw_path = os.path.join(vq_config["raw_data_path"], p_id, vq_config["df_raw_name"])
        if not os.path.exists(raw_path):
            print(f"Raw data for P{p_num} not found at {raw_path}")
            continue
            
        print(f"Generating Ratio Comparison Plot for Participant {p_num}...")
        df_raw = pd.read_csv(raw_path)
        
        # Filter out 'rest' and NaNs to find the first real gesture
        df_gestures = df_raw[df_raw['label'].notna() & (df_raw['label'] != 'rest')].copy()
        if df_gestures.empty:
            continue
            
        first_gesture_name = df_gestures['label'].iloc[0]
        print(f"  Target Gesture: {first_gesture_name}")
        
        # Get indices for this gesture
        indices = df_raw.index[df_raw['label'] == first_gesture_name].to_numpy()
        # Find start of each repetition (where indices jump)
        breaks = np.where(np.diff(indices) > 1)[0]
        rep_starts = [indices[0]] + [indices[i+1] for i in breaks]
        
        # We want the first 3 repetitions from the SEEN data (75% part)
        # Each rep is 4000 samples. 
        # But wait, we need to align these with the reconstruction.
        # The reconstruction is based on the ENCODED dataset, which follows the dataset split.
        
        # Load one reconstruction to get column names and check length
        df_sample_recon = pd.read_csv(os.path.join(base_model_dir, f"synthetic_{ratios[0]}_reconstructed.csv"))
        feature_cols = [c for c in df_sample_recon.columns if c != 'gt']
        sensor_to_plot = feature_cols[0]
        
        # Setup Figure: Top row (3 reps of raw) + One row per ratio
        n_rows = 1 + len(ratios)
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 4 * n_rows))
        plt.subplots_adjust(hspace=0.4)
        
        # --- ROW 0: RAW SIGNAL (First 3 Reps) ---
        for r in range(3):
            if r < len(rep_starts):
                start = rep_starts[r]
                # Convert units to match preprocessed scale (approx)
                seg = df_raw.iloc[start : start + 4000][sensor_to_plot].values * 1e6
                axes[0, r].plot(seg, color='#335067', alpha=0.7, lw=1.5)
                axes[0, r].set_title(f"Original Rep {r+1}", fontweight='bold', color='#335067')
            axes[0, r].grid(alpha=0.15)
            axes[0, r].spines['top'].set_visible(False)
            axes[0, r].spines['right'].set_visible(False)
        
        # --- ROWS 1+: RECONSTRUCTIONS ---
        for row_idx, ratio in enumerate(ratios):
            recon_path = os.path.join(base_model_dir, f"synthetic_{ratio}_reconstructed.csv")
            df_recon = pd.read_csv(recon_path)
            
            # Find segments of this gesture in the reconstructed data
            df_recon['change'] = df_recon['gt'].diff().ne(0).astype(int)
            df_recon['block_id'] = df_recon['change'].cumsum()
            
            # Find the blocks for our target gesture
            target_blocks = df_recon[df_recon['gt'] != -1].groupby('block_id') 
            
            # Find all segments of the first occurring gesture ID in df_recon
            target_id = df_recon['gt'].iloc[0] 
            indices_recon = df_recon.index[df_recon['gt'] == target_id].to_numpy()
            breaks_recon = np.where(np.diff(indices_recon) > 1)[0]
            rep_starts_recon = [indices_recon[0]] + [indices_recon[i+1] for i in breaks_recon]
            
            for r in range(3):
                if r < len(rep_starts_recon):
                    start_r = rep_starts_recon[r]
                    seg_recon = df_recon.iloc[start_r : start_r + 4000][sensor_to_plot].values
                    axes[row_idx+1, r].plot(seg_recon, color='#054984', alpha=0.8, lw=1.5)
                    if r == 0:
                        axes[row_idx+1, r].set_ylabel(f"Ratio {ratio}", fontweight='bold', fontsize=12, color='#054984')
                axes[row_idx+1, r].grid(alpha=0.15)
                axes[row_idx+1, r].spines['top'].set_visible(False)
                axes[row_idx+1, r].spines['right'].set_visible(False)

        fig.suptitle(f"Participant {p_num} ({p_id}) | Gesture: {first_gesture_name}\nTop: Raw | Bottom: Reconstructions by Prompt Ratio", fontsize=16)
        save_path = os.path.join(save_dir, f"ratio_comparison_P{p_num}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved comparison to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to Transformer config (replicate_small.yaml)")
    parser.add_argument('--vqvae_config', type=str, required=True, help="Path to VQ-VAE config (tuned_config2.yaml)")
    parser.add_argument('--save_dir', type=str, default=None, help="Custom save directory for plots")
    args = parser.parse_args()

    # Load configs
    with open(args.config, 'r') as f:
        tr_config = yaml.safe_load(f)
    with open(args.vqvae_config, 'r') as f:
        vq_config_full = yaml.safe_load(f)
        vq_config = vq_config_full.get('vqvae', vq_config_full)

    exp_name = tr_config['exp_name']
    vq_name = vq_config['name']
    model_files_base_directory = os.path.join(pathlib.Path(__file__).resolve().parent.__str__(), "models")
    base_model_dir = os.path.join(model_files_base_directory, exp_name)
    
    # 1. Run Original vs Reconstructed (Unseen/Final)
    ORIG_PATH = f"./VQVAE/models/{vq_name}/unseen_data_preprocessed.csv"
    RECON_PATH = f"{base_model_dir}/unseen_reconstructed_final.csv"
    SAVE_DIR = args.save_dir if args.save_dir else f"{base_model_dir}/unseen_gesture_plots"
    visualize_gesture_reconstruction(ORIG_PATH, RECON_PATH, save_dir=SAVE_DIR)

    # 2. Run Ratio Comparison for P6 and P7
    RATIO_SAVE_DIR = os.path.join(base_model_dir, "ratio_comparisons")
    visualize_ratio_comparison(vq_config, tr_config, RATIO_SAVE_DIR)
