import os
import yaml 
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
import numpy as np
import argparse

from dataset import EMGDataset
from evaluation import evaluate_model
from model import SDformerVQVAE
from train import train_vqvae
from visualizer import Visualizer

parser = argparse.ArgumentParser(description='Run VQVAE pipeline')
parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file')
args = parser.parse_args()

with open(args.config, "r") as file:
    full_config = yaml.safe_load(file)
    # If the config is nested (merged), extract the vqvae part
    if 'vqvae' in full_config:
        config = full_config['vqvae']
    else:
        config = full_config

save_dir = f"./models/{config['name']}/"
os.makedirs(save_dir, exist_ok=True)

with open(os.path.join(save_dir, "config.yaml"), "w") as file:
    yaml.dump(config, file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Pipeline initialized on {device}. Saving results to: {save_dir}")
# --- 2. Load Dataset ---
print("\n--- Loading Datasets ---")
train_dataset = EMGDataset(config, window_size=config['window_size'], stride=config['stride'], split='train')
unseen_dataset = EMGDataset(config, window_size=config['window_size'], stride=config['stride'], split='unseen')

train_dataset.save_df(os.path.join(save_dir, "train_data_preprocessed.csv")) 
unseen_dataset.save_df(os.path.join(save_dir, "unseen_data_preprocessed.csv")) 

print(f"Train dataset: {len(train_dataset)} samples")
print(f"Unseen dataset: {len(unseen_dataset)} samples")
print("$" * 50)

# --- 3. Train/Validation Split ---
# Train on 75% (train split), Test on 25% (unseen split)
val_dataset = unseen_dataset

print(f"Data Split: {len(train_dataset)} Training | {len(val_dataset)} Validation (Unseen)")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)


# Sanity Check
first_batch = next(iter(train_loader))
assert first_batch.shape == (config['batch_size'], 8, config['window_size']), \
    f"Unexpected batch shape: {first_batch.shape}"

# --- 4. Initialize Model ---
model = SDformerVQVAE(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['learning_rate']))

# --- 5. Train Model ---
print("\n--- Starting Training ---")
model = train_vqvae(model, train_loader, device, optimizer, config)

# --- 6. Save Final Model ---
model_path = os.path.join(save_dir, "final_model.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# --- 7. Evaluate Model ---
print("\n--- Starting Evaluation ---")
evaluate_model(model, val_loader, device, config)

# --- 8. Visualization Suite ---
print("\n--- Generating Diagnostic Plots ---")
viz = Visualizer(model, device, config)

print("1/4. Visualizing Codebook...")
viz.visualize_codebook()

print("2/4. Checking Data Distribution...")
viz.plot_data_distribution(val_loader, num_samples=2000)

print("3/4. Plotting Random Sample Reconstructions...")
viz.plot_single_reconstruction(val_loader, sample_index=0)
viz.plot_single_reconstruction(val_loader, sample_index=10)

print("4/4. Tracing Full Gesture Pipeline...")
# Re-load full dataset for visualization and encoding
full_dataset = EMGDataset(config, window_size=config['window_size'], stride=config['stride'], split='all')

gestures = [11, 8, 17] # Power Grip, OK, Rest
for label_id in gestures:
    for rep in [0, 4]: # Participant 1 & 2
        try:
            viz.plot_gesture_pipeline(full_dataset.df, label_id=label_id, repetition_index=rep)
        except: pass

# --- 9. Generate Encoded Dataset (Codebooks Only) ---
print("\n--- Generating Encoded Dataset (Tokens) ---")
encoded_save_path = os.path.join(save_dir, "encoded_df.csv")
train_encoded_save_path = os.path.join(save_dir, "train_encoded_df.csv")
unseen_encoded_save_path = os.path.join(save_dir, "unseen_encoded_df.csv")

def encode_and_save(dataset, save_path):
    loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False)
    model.eval()
    all_codes = []
    all_labels = []

    print(f"Mapping labels to windows for {os.path.basename(save_path)}...")
    gt_values = dataset.df['gt'].values
    total_windows = len(dataset)
    window_centers = [i * dataset.stride + dataset.window_size // 2 for i in range(total_windows)]

    print(f"Processing {total_windows} windows...")

    with torch.no_grad():
        batch_start_idx = 0
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            current_batch_size = batch.size(0)
            
            # Pass through model
            _,_, _, indices = model(batch)
            
            # --- FIX: Reshape flattened indices ---
            if indices.dim() == 1:
                indices = indices.view(current_batch_size, -1)
                
            batch_codes = indices.cpu().numpy()
            all_codes.append(batch_codes)
            
            batch_indices = range(batch_start_idx, batch_start_idx + current_batch_size)
            center_indices = [min(window_centers[idx], len(gt_values)-1) for idx in batch_indices]
            batch_labels = gt_values[center_indices]
            all_labels.append(batch_labels)
            
            batch_start_idx += current_batch_size
            
            if i % 100 == 0:
                print(f"  Encoded {batch_start_idx} / {total_windows} samples...")

    if not all_codes:
        print(f"Warning: No data to save for {save_path}")
        return

    final_codes = np.concatenate(all_codes, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)

    token_cols = [f"col_{i}" for i in range(final_codes.shape[1])]
    df_encoded = pd.DataFrame(final_codes, columns=token_cols)
    df_encoded.insert(0, "gt", final_labels)
    df_encoded.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

# Encode separately
encode_and_save(train_dataset, train_encoded_save_path)
encode_and_save(unseen_dataset, unseen_encoded_save_path)

# --- 7. NEW: Final Evaluation and Specific Visualizations ---
print("\n--- Running Final Evaluation & Targeted Visualizations ---")
eval_results = evaluate_model(model, val_loader, device, config)

# Save metrics to a YAML file for Phase 1 results
with open(os.path.join(save_dir, "evaluation_metrics.yaml"), "w") as f:
    yaml.dump(eval_results, f)

# Targeted Visualizations for P6 and P7
# participants_list_ids : ["033106b27b","bc4dd952fe","31afab1e30","97c6aaac2d","7037a93026","98aa5fac2d","ecfa481b42","e49db6578f","27f6898a3f","3f858df9cf","9780ed81f4"]
# Participant 6 is index 5 ("98aa5fac2d"), Participant 7 is index 6 ("ecfa481b42")
target_participants = [
    (6, config['participants_list_ids'][5]), 
    (7, config['participants_list_ids'][6])
]

label_mapping = {
    "Thumb Extension":0,"index Extension":1,"Middle Extension":2,"Ring Extension":3,
    "Pinky Extension":4,"Thumbs Up":5,"Right Angle":6,"Peace":7,"OK":8,"Horn":9,"Hang Loose":10,
    "Power Grip":11,"Hand Open":12,"Wrist Extension":13,"Wrist Flexion":14,"Ulnar deviation":15,"Radial Deviation":16    
}

for p_num, p_id in target_participants:
    print(f"Generating triple-comparison plots for Participant {p_num} ({p_id})...")
    raw_path = os.path.join(config["raw_data_path"], p_id, config["df_raw_name"])
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        # Convert units if needed (mirroring dataset.py)
        if 'emg' in config['sensor_type']:
            emg_cols = [c for c in raw_df.columns if 'emg' in c.lower()]
            raw_df[emg_cols] = raw_df[emg_cols] * 1e6
        
        for label_name in label_mapping.keys():
            viz.plot_unseen_comparison(raw_df, p_num, label_mapping[label_name], label_name, train_dataset)

# Also keep the full one for backward compatibility if needed, or just combine them
print("Combining for full encoded_df.csv...")
df_train = pd.read_csv(train_encoded_save_path)
df_unseen = pd.read_csv(unseen_encoded_save_path)
df_full = pd.concat([df_train, df_unseen], ignore_index=True)
df_full.to_csv(encoded_save_path, index=False)

print(f"\nPipeline Completed Successfully! All results in: {save_dir}")