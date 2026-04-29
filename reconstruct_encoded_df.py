import os
import pathlib
import yaml
import torch
import pandas as pd
import numpy as np
import argparse

from VQVAE.model import SDformerVQVAE

def reconstruct_pipeline(original_data_path, encoded_data_path, vqvae, device, vqvae_config, window_size=300, stride=150):
    """
    Reconstructs the original signals from the encoded tokens using the VQ-VAE decoder.
    Now includes unnormalization to match original data scale.
    """
    print(f"Loading original data from {original_data_path} for column info...")
    df_orig = pd.read_csv(original_data_path)
    feature_cols = [c for c in df_orig.columns if c != 'gt']
    
    # --- Fit Scaler (MUST match training logic) ---
    from VQVAE.dataset import EMGDataset
    print("Fitting scaler from training data for unnormalization...")
    train_ds = EMGDataset(vqvae_config, split='train')
    scaler = train_ds.scaler

    print(f"Loading encoded tokens from {encoded_data_path}...")
    df_encoded = pd.read_csv(encoded_data_path)
    
    token_cols = [c for c in df_encoded.columns if c != 'gt']
    indices = torch.tensor(df_encoded[token_cols].values, dtype=torch.long).to(device)
    
    # --- DECODING PHASE ---
    print("Decoding tokens back to continuous signal...")
    all_reconstructed_chunks = []
    
    with torch.no_grad():
        batch_size = 128
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            z_q = vqvae.quantizer.embedding[batch_indices.reshape(-1)]
            z_q = z_q.reshape(batch_indices.shape[0], batch_indices.shape[1], -1).permute(0, 2, 1)
            
            reconstructed_batch = vqvae.decoder(z_q) # [B, C, W]
            
            margin = (window_size - stride) // 2
            chunk = reconstructed_batch[:, :, margin : margin + stride]
            all_reconstructed_chunks.append(chunk.permute(0, 2, 1).cpu().numpy())
        
    recon_np = np.concatenate(all_reconstructed_chunks, axis=0)
    final_signal_norm = recon_np.reshape(-1, len(feature_cols))
    
    # --- UNNORMALIZATION ---
    print("Applying inverse scaling...")
    final_signal = scaler.inverse_transform(final_signal_norm)
    
    df_reconstructed = pd.DataFrame(final_signal, columns=feature_cols)
    
    common_len = min(len(df_orig), len(df_reconstructed))
    df_reconstructed_final = df_reconstructed.iloc[:common_len].copy()
    df_reconstructed_final['gt'] = df_orig['gt'].iloc[:common_len].values

    print(f"Sync complete. Final Shape: {df_reconstructed_final.shape}")
    return df_reconstructed_final, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to Transformer config (replicate_small.yaml)")
    parser.add_argument('--vqvae_config', type=str, required=True, help="Path to VQ-VAE config (tuned_config2.yaml)")
    args = parser.parse_args()

    # Load configs
    with open(args.config, 'r') as f:
        tr_config = yaml.safe_load(f)
    with open(args.vqvae_config, 'r') as f:
        vqvae_config_full = yaml.safe_load(f)
        vqvae_config = vqvae_config_full.get('vqvae', vqvae_config_full)


    exp_name = tr_config['exp_name']
    vq_name = vqvae_config['name']

    # --- Path Configuration ---
    model_files_base_directory = os.path.join(pathlib.Path(__file__).resolve().parent.__str__(), "models")
    vqvae_models_dir = os.path.join(pathlib.Path(__file__).resolve().parent.__str__(), "VQVAE", "models")
    CONFIG_PATH = args.vqvae_config
    # Point to the UNSEEN preprocessed data from the VQ-VAE phase
    ORIGINAL_DATA_PATH = os.path.join(vqvae_models_dir, vq_name, "unseen_data_preprocessed.csv")
    
    # Priority for encoded data:
    # 1. New 'unseen_synthetic' files
    # 2. Generic 'unseen_synthetic_encoded_samples.csv'
    # 3. Fallback to old naming 'synthetic_df_...'
    
    base_model_dir = os.path.join(model_files_base_directory, exp_name)
    
    # 1. Find all synthetic datasets matching the pattern
    synthetic_files = [f for f in os.listdir(base_model_dir) if f.startswith("seen_synthetic_df_") and f.endswith(".csv")]
    
    if not synthetic_files:
        print(f"No synthetic datasets found in {base_model_dir}")
        return

    MODEL_WEIGHTS_PATH = os.path.join(vqvae_models_dir, vq_name, "final_model.pth") 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and Load Model
    model = SDformerVQVAE(vqvae_config).to(device)
    
    if os.path.exists(MODEL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        print(f"Weights loaded from {MODEL_WEIGHTS_PATH}. Using device: {device}")
    else:
        print(f"Error: Weights not found at {MODEL_WEIGHTS_PATH}")
        return

    for file_name in synthetic_files:
        # Extract ratio from file_name (e.g., seen_synthetic_df_70_5.csv -> 70_5)
        ratio = file_name.replace("seen_synthetic_df_", "").replace(".csv", "")
        
        ENCODED_DATA_PATH = os.path.join(base_model_dir, file_name)
        SAVE_OUTPUT_PATH = os.path.join(base_model_dir, f"synthetic_{ratio}_reconstructed.csv")
        
        print(f"\n--- Processing {file_name} ---")
        try:
            recon_df, _ = reconstruct_pipeline(
                ORIGINAL_DATA_PATH, 
                ENCODED_DATA_PATH, 
                model, 
                device,
                window_size=vqvae_config.get('window_size', 300),
                stride=vqvae_config.get('stride', 30) # Updated to 30 per your 10% stride description
            )
            
            # Save results
            os.makedirs(os.path.dirname(SAVE_OUTPUT_PATH), exist_ok=True)
            recon_df.to_csv(SAVE_OUTPUT_PATH, index=False)
            print(f"Successfully saved reconstructed data to: {SAVE_OUTPUT_PATH}")
            
        except Exception as e:
            print(f"Reconstruction failed for {file_name}: {e}")

if __name__ == "__main__":
    main()
