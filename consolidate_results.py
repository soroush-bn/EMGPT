import os
import shutil
import yaml
import argparse
from pathlib import Path

def consolidate(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    exp_name = config.get('exp_name', 'experiment')
    vqvae_name = config.get('vqvae', {}).get('name', 'unseen')
    
    results_dir = f"{exp_name}_results"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Consolidating results into: {results_dir}")
    
    # --- Phase 1: VQ-VAE ---
    vqvae_src = os.path.join("VQVAE", "models", vqvae_name)
    if os.path.exists(vqvae_src):
        phase1_dir = os.path.join(results_dir, "phase1_vqvae")
        os.makedirs(phase1_dir, exist_ok=True)
        
        # Copy VQ-VAE figures
        fig_src = os.path.join(vqvae_src, "figs")
        if os.path.exists(fig_src):
            shutil.copytree(fig_src, os.path.join(phase1_dir, "figures"), dirs_exist_ok=True)
            
        # Copy VQ-VAE model and metrics
        for f in ["final_model.pth", "config.yaml", "evaluation_metrics.yaml"]:
            src_f = os.path.join(vqvae_src, f)
            if os.path.exists(src_f):
                shutil.copy2(src_f, phase1_dir)
        print("  [Phase 1] VQ-VAE consolidated.")

    # --- Phase 2: Transformer & Attention ---
    trans_src = os.path.join("models", exp_name)
    if os.path.exists(trans_src):
        phase2_dir = os.path.join(results_dir, "phase2_transformer")
        os.makedirs(phase2_dir, exist_ok=True)
        
        # Find latest iteration
        iters = [d for d in os.listdir(trans_src) if d.startswith("iter_")]
        if iters:
            latest_iter = sorted(iters, key=lambda x: int(x.split('_')[1]))[-1]
            shutil.copytree(os.path.join(trans_src, latest_iter), os.path.join(phase2_dir, "latest_checkpoint"), dirs_exist_ok=True)
            
            info_src = os.path.join(trans_src, latest_iter, "info.yml")
            if os.path.exists(info_src):
                shutil.copy2(info_src, os.path.join(phase2_dir, "evaluation_summary.yml"))
        
        # Attention Heatmaps
        attn_src = os.path.join(trans_src, "final_visualization", "attention")
        if os.path.exists(attn_src):
            shutil.copytree(attn_src, os.path.join(phase2_dir, "attention_heatmaps"), dirs_exist_ok=True)
            
        print("  [Phase 2] Transformer and Attention consolidated.")

        # --- Phase 3: Generation & Fidelity ---
        phase3_dir = os.path.join(results_dir, "phase3_generation")
        os.makedirs(phase3_dir, exist_ok=True)
        
        # Copy Synthetic CSVs
        for f in os.listdir(trans_src):
            if f.endswith(".csv") and ("synthetic" in f or "generated" in f or "reconstructed" in f):
                shutil.copy2(os.path.join(trans_src, f), phase3_dir)
        
        # Reconstruction Comparisons
        recon_comp_src = os.path.join(trans_src, "reconstruction_comparisons")
        if os.path.exists(recon_comp_src):
            shutil.copytree(recon_comp_src, os.path.join(phase3_dir, "reconstruction_comparisons"), dirs_exist_ok=True)
        elif os.path.exists(os.path.join(trans_src, "final_visualization", "reconstruction_comparisons")):
            # Fallback if final_viz.lsf moved it
            shutil.copytree(os.path.join(trans_src, "final_visualization", "reconstruction_comparisons"), 
                            os.path.join(phase3_dir, "reconstruction_comparisons"), dirs_exist_ok=True)

        # Ratio Comparisons (Raw vs Recon)
        ratio_src = os.path.join(trans_src, "ratio_comparisons")
        if os.path.exists(ratio_src):
            shutil.copytree(ratio_src, os.path.join(phase3_dir, "ratio_comparisons"), dirs_exist_ok=True)
            
        # Fidelity Reports
        fidelity_src = os.path.join(trans_src, "fidelity_reports")
        if os.path.exists(fidelity_src):
            shutil.copytree(fidelity_src, os.path.join(phase3_dir, "fidelity_reports"), dirs_exist_ok=True)
            
        # Other diagnostic plots (TSNE, Atomic, etc.) - Excluding already copied Attention/Recon
        final_viz_src = os.path.join(trans_src, "final_visualization")
        if os.path.exists(final_viz_src):
            diag_dir = os.path.join(phase3_dir, "diagnostic_plots")
            os.makedirs(diag_dir, exist_ok=True)
            for item in os.listdir(final_viz_src):
                item_path = os.path.join(final_viz_src, item)
                if os.path.isfile(item_path):
                    shutil.copy2(item_path, diag_dir)
                elif os.path.isdir(item_path) and item not in ["attention", "reconstruction_comparisons"]:
                    shutil.copytree(item_path, os.path.join(diag_dir, item), dirs_exist_ok=True)
                    
        print("  [Phase 3] Generation and Fidelity consolidated.")

        # --- Phase 4: Classification ---
        phase4_dir = os.path.join(results_dir, "phase4_classification")
        os.makedirs(phase4_dir, exist_ok=True)
        class_res = os.path.join(trans_src, "classification_experiments_results.csv")
        if os.path.exists(class_res):
            shutil.copy2(class_res, phase4_dir)
        print("  [Phase 4] Classification results consolidated.")

    # Main config for reference
    shutil.copy2(config_path, results_dir)
    print(f"Consolidation complete: {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    consolidate(args.config)
