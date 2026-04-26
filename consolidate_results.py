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
    
    # Phase 1: VQ-VAE
    vqvae_src = os.path.join("VQVAE", "models", vqvae_name)
    if os.path.exists(vqvae_src):
        phase1_dir = os.path.join(results_dir, "phase1_vqvae")
        os.makedirs(phase1_dir, exist_ok=True)
        # Copy figures
        fig_src = os.path.join(vqvae_src, "figs")
        if os.path.exists(fig_src):
            shutil.copytree(fig_src, os.path.join(phase1_dir, "figures"), dirs_exist_ok=True)
        # Copy model and config
        for f in ["final_model.pth", "config.yaml", "evaluation_metrics.yaml"]:
            if os.path.exists(os.path.join(vqvae_src, f)):
                shutil.copy2(os.path.join(vqvae_src, f), phase1_dir)
        
        # Explicitly copy unseen comparison figures
        if os.path.exists(fig_src):
            for f in os.listdir(fig_src):
                if f.startswith("unseen_comparison_P"):
                    shutil.copy2(os.path.join(fig_src, f), phase1_dir)
        print("  [Phase 1] VQ-VAE results and targeted figures copied.")

    # Phase 2 & 3 & 4: Transformer, Generation, Classification
    trans_src = os.path.join("models", exp_name)
    if os.path.exists(trans_src):
        # Phase 2: Transformer (Weights & Info)
        phase2_dir = os.path.join(results_dir, "phase2_transformer")
        os.makedirs(phase2_dir, exist_ok=True)
        # Find latest iter folder
        iters = [d for d in os.listdir(trans_src) if d.startswith("iter_")]
        if iters:
            latest_iter = sorted(iters, key=lambda x: int(x.split('_')[1]))[-1]
            shutil.copytree(os.path.join(trans_src, latest_iter), os.path.join(phase2_dir, "latest_checkpoint"), dirs_exist_ok=True)
        print("  [Phase 2] Transformer latest checkpoint copied.")

        # Phase 3: Generation (Synthetic Data & Visualization)
        phase3_dir = os.path.join(results_dir, "phase3_generation")
        os.makedirs(phase3_dir, exist_ok=True)
        # Copy synthetic CSVs
        for f in os.listdir(trans_src):
            if f.endswith(".csv") and ("synthetic" in f or "generated" in f):
                shutil.copy2(os.path.join(trans_src, f), phase3_dir)
        # Copy figures from final_visualization if they exist
        viz_src = os.path.join(trans_src, "final_visualization")
        if os.path.exists(viz_src):
            shutil.copytree(viz_src, os.path.join(phase3_dir, "visualizations"), dirs_exist_ok=True)
        print("  [Phase 3] Generation results and visualizations copied.")

        # Phase 4: Classification
        phase4_dir = os.path.join(results_dir, "phase4_classification")
        os.makedirs(phase4_dir, exist_ok=True)
        class_res = os.path.join(trans_src, "classification_experiments_results.csv")
        if os.path.exists(class_res):
            shutil.copy2(class_res, phase4_dir)
        print("  [Phase 4] Classification results copied.")

    # Copy the main config for reference
    shutil.copy2(config_path, results_dir)
    print(f"Consolidation complete: {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    consolidate(args.config)
