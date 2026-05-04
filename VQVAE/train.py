import torch
import torch.nn as nn
from tqdm import tqdm
import os

def train_vqvae(model, dataloader, device, optimizer, config):
    model.train()
    criterion_recon = nn.MSELoss()
    
    checkpoint_dir = f"./models/{config['name']}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    master_process = config.get('master_process', True)

    if config.get('wandb_log', False) and master_process:
        try:
            import wandb

            wandb_api_key = os.environ.get('WANDB_API_KEY', None)
            if wandb_api_key:
                print("Logging into W&B using WANDB_API_KEY environment variable...")
                wandb.login(key=wandb_api_key)
            else:
                print("Warning: WANDB_API_KEY not found. Attempting to use cached credentials...")
                try:
                    wandb.login()
                except Exception as e:
                    print(f"W&B login failed: {e}")
                    print("Please set WANDB_API_KEY environment variable or run 'wandb login' manually")
                    config['wandb_log'] = False

            if config.get('wandb_log', False):
                wandb.init(project=config.get('wandb_project', 'vqvae'), name=config['name'], config=config)
        except Exception as e:
            print(f"W&B import/init error: {e}")
            config['wandb_log'] = False

    epoch_pbar = tqdm(range(config['number_of_epochs']), desc="Training Progress")

    for epoch in epoch_pbar:
        total_loss = 0
        total_recon = 0
        total_embed = 0
        total_commitment = 0
        total_codebook = 0
        
        all_active_codes = []
        all_perplexities = []

        for batch_idx, x in enumerate(dataloader):
            x = x.to(device)

            optimizer.zero_grad()

            x_recon, commitment_loss, codebook_loss, indices = model(x)

            loss_recon = criterion_recon(x_recon, x)
            loss_embed =  commitment_loss+ codebook_loss
            loss = loss_recon + loss_embed

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_embed += loss_embed.item()
            total_commitment += commitment_loss.item()
            total_codebook += codebook_loss.item()
            
            # --- Metrics for Codebook Usage ---
            with torch.no_grad():
                # Flatten indices to [Batch * SeqLen]
                flat_indices = indices.flatten()
                # Use histc to count occurrences of each code
                counts = torch.histc(flat_indices.float(), bins=config['codebook_size'], min=0, max=config['codebook_size']-1)
                
                # Active codes in this batch
                current_active = (counts > 0).sum().item()
                
                # Perplexity calculation
                probs = counts / (counts.sum() + 1e-10)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                current_perplexity = torch.exp(entropy).item()
                
                all_active_codes.append(current_active)
                all_perplexities.append(current_perplexity)

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_embed = total_embed / len(dataloader)
        avg_commitment = total_commitment / len(dataloader)
        avg_codebook = total_codebook / len(dataloader)
        
        avg_perplexity = sum(all_perplexities) / len(all_perplexities)
        avg_active = sum(all_active_codes) / len(all_active_codes)

        tqdm.write(f"Epoch [{epoch+1}/{config['number_of_epochs']}] | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | Perplexity: {avg_perplexity:.2f} | Active: {avg_active:.1f}")
        
        if config.get('wandb_log', False) and master_process:
            try:
                import wandb
                import matplotlib.pyplot as plt

                log_dict = {
                    "epoch": epoch + 1,
                    "epoch/total_loss": avg_loss,
                    "epoch/recon_loss": avg_recon,
                    "epoch/embed_loss": avg_embed,
                    "epoch/commitment_loss": avg_commitment,
                    "epoch/codebook_loss": avg_codebook,
                    "epoch/perplexity": avg_perplexity,
                    "epoch/active_codes": avg_active,
                    "epoch/mse": avg_recon,
                }

                # Log reconstruction plot every 10 epochs
                if (epoch + 1) % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        # Take the first sample from the last batch
                        orig = x[0].cpu().numpy()
                        recon = x_recon[0].cpu().numpy()
                        
                        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
                        for ch in range(min(4, orig.shape[0])):
                            axes[ch].plot(orig[ch], label='Original', color='black', alpha=0.6)
                            axes[ch].plot(recon[ch], label='Reconstructed', color='green', linestyle='--')
                            axes[ch].legend(loc='upper right')
                        plt.tight_layout()
                        log_dict["plots/reconstruction"] = wandb.Image(fig)
                        plt.close(fig)
                    model.train()

                wandb.log(log_dict)
            except Exception as e:
                print(f"W&B logging error: {e}")
        if (epoch + 1) % 5 == 0 or (epoch + 1) == config['number_of_epochs']:
            save_path = os.path.join(checkpoint_dir, f"vqvae_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            tqdm.write(f"--> Checkpoint saved: {save_path}")
            if config.get('wandb_log', False) and master_process:
                try:
                    import wandb

                    wandb.save(save_path)
                except Exception as e:
                    print(f"W&B save error: {e}")

    if config.get('wandb_log', False) and master_process:
        try:
            import wandb

            wandb.finish()
        except Exception as e:
            print(f"W&B finish error: {e}")

    return model