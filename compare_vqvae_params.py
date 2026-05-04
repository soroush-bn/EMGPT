import os
import yaml
import torch
import copy
import argparse
import time
import subprocess
from torch.utils.data import DataLoader
from VQVAE.dataset import EMGDataset
from VQVAE.model import SDformerVQVAE
from VQVAE.train import train_vqvae
from VQVAE.evaluation import evaluate_model

def run_training_local(config, device):
    save_dir = f"./VQVAE/models/{config['name']}/"
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if already trained
    model_path = os.path.join(save_dir, "final_model.pth")
    if os.path.exists(model_path):
        print(f"Model {config['name']} already exists. Skipping.")
        metrics_path = os.path.join(save_dir, "evaluation_metrics.yaml")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = yaml.safe_load(f)
            return metrics.get('recon_loss', 1e9)
        return 1e9

    print(f"\n--- Training Model: {config['name']} ---")
    # Load Datasets
    train_dataset = EMGDataset(config, window_size=config['window_size'], stride=config['stride'], split='train')
    unseen_dataset = EMGDataset(config, window_size=config['window_size'], stride=config['stride'], split='unseen')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(unseen_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)

    # Initialize Model
    model = SDformerVQVAE(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['learning_rate']))
    
    # Train
    model = train_vqvae(model, train_loader, device, optimizer, config)
    
    # Save
    torch.save(model.state_dict(), model_path)
    
    # Evaluate
    eval_results = evaluate_model(model, val_loader, device, config)
    with open(os.path.join(save_dir, "evaluation_metrics.yaml"), "w") as f:
        yaml.dump(eval_results, f)
        
    return eval_results['recon_loss']

def submit_lsf_job(param, value, config_path):
    job_name = f"tune_{param}_{value}"
    # Construct bsub command. Adjust resources (e.g., -gpu, -n, -W) as needed for your HPC
    cmd = [
        "bsub",
        "-J", job_name,
        "-o", f"logs/{job_name}.out",
        "-e", f"logs/{job_name}.err",
        "-gpu", "num=1:mode=exclusive_process",
        "-q", "gpu", # Or your specific queue
        "-W", "02:00",
        "python", "compare_vqvae_params.py",
        "--config", config_path,
        "--param", param,
        "--value", str(value),
        "--mode", "single"
    ]
    print(f"Submitting job: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        # Extract job ID if needed, but we'll poll by job name
        print(f"Successfully submitted {job_name}")
        return job_name
    else:
        print(f"Error submitting {job_name}: {result.stderr}")
        return None

def wait_for_jobs(job_names):
    print(f"Waiting for jobs to complete: {job_names}")
    while True:
        finished = True
        result = subprocess.run(["bjobs"], capture_output=True, text=True)
        running_jobs = result.stdout
        
        for name in job_names:
            if name in running_jobs:
                finished = False
                break
        
        if finished:
            print("All jobs in this group have finished.")
            break
        
        time.sleep(30) # Poll every 30 seconds

def get_metrics(name):
    metrics_path = f"./VQVAE/models/{name}/evaluation_metrics.yaml"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = yaml.safe_load(f)
        return metrics.get('recon_loss', 1e9)
    return 1e9

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--param', type=str)
    parser.add_argument('--value', type=str)
    parser.add_argument('--mode', type=str, choices=['orchestrator', 'single'], default='orchestrator')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)
        base_config = full_config['vqvae'] if 'vqvae' in full_config else full_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == 'single':
        param = args.param
        val = int(args.value)
        config = copy.deepcopy(base_config)
        config[param] = val
        config['name'] = f"tune_{param}_{val}"
        run_training_local(config, device)
    else:
        # Orchestrator Mode
        os.makedirs("logs", exist_ok=True)
        tuning_params = {
            'codebook_size': [256, 512, 1024, 2048],
            'code_dim': [128, 256, 512, 1024],
            'stride': [100, 200, 300, 500]
        }

        best_params = copy.deepcopy(base_config)
        
        for param, values in tuning_params.items():
            print(f"\n{'='*20} TUNING {param.upper()} {'='*20}")
            job_names = []
            for val in values:
                # Update current config with best found so far
                current_config = copy.deepcopy(best_params)
                current_config[param] = val
                name = f"tune_{param}_{val}"
                
                # We need to save this temporary config or pass via CLI
                # Since we already have compare_vqvae_params.py --param/--value, we use that.
                # However, the --config passed to bsub should ideally be the latest best_params.
                # Let's save a temp best_config.yaml
                with open("temp_best_config.yaml", "w") as f:
                    yaml.dump({'vqvae': best_params}, f)
                
                job_name = submit_lsf_job(param, val, "temp_best_config.yaml")
                if job_name:
                    job_names.append(job_name)
            
            wait_for_jobs(job_names)
            
            # Analyze results
            results = {}
            for val in values:
                name = f"tune_{param}_{val}"
                loss = get_metrics(name)
                results[val] = loss
                print(f"Result for {param}={val}: Recon Loss = {loss:.6f}")
            
            best_val = min(results, key=results.get)
            best_params[param] = best_val
            print(f"BEST {param} so far: {best_val}")

        print("\n--- Final Best Parameters ---")
        for p in tuning_params.keys():
            print(f"{p}: {best_params[p]}")
        
        with open("final_best_vqvae_params.yaml", "w") as f:
            yaml.dump({'vqvae': best_params}, f)

if __name__ == "__main__":
    main()

