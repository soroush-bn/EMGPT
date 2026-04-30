import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Add root to path to import viz_style
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from viz_style import COLORS, BLUE_PALETTE, apply_ax_style

class Visualizer:
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.SAMPLING_RATE = 2000
        self.REP_DURATION_SEC = 2.0
        self.SAMPLES_PER_REP = int(self.SAMPLING_RATE * self.REP_DURATION_SEC)
        self.save_dir = f"./models/{config['name']}/figs/"
        os.makedirs(self.save_dir, exist_ok=True)

    def visualize_codebook(self, perplexity=30):
        self.model.eval()
        embeddings = self.model.quantizer.embedding.detach().cpu().numpy()
        usage_counts = self.model.quantizer.ema_cluster_size.detach().cpu().numpy()
        top_indices = np.argsort(usage_counts)[::-1]
        
        try:
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
            embeddings_2d = tsne.fit_transform(embeddings)
            fig = plt.figure(figsize=(10, 8))
            sc = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=np.log1p(usage_counts), cmap='viridis', alpha=0.7, s=30)
            plt.colorbar(sc, label='Log Usage Count')
            plt.title("t-SNE of Codebook Vectors", fontweight='bold', color=COLORS['text_primary'])
            plt.savefig(os.path.join(self.save_dir, "codebook_tsne.png"), bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"[Visualizer] t-SNE failed: {e}")

        fig2, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        with torch.no_grad():
            for i, code_idx in enumerate(top_indices[:3]):
                code_vec = self.model.quantizer.embedding[code_idx]
                fake_latent = code_vec.view(1, -1, 1).repeat(1, 1, 20).to(self.device)
                sig = self.model.decoder(fake_latent)[0].cpu().numpy()
                for ch in range(8): 
                    color = BLUE_PALETTE[ch % len(BLUE_PALETTE)]
                    axes[i].plot(sig[ch], color=color, alpha=0.8)
                apply_ax_style(axes[i], title=f"Code #{code_idx} (Usage: {int(usage_counts[code_idx])})")
        plt.savefig(os.path.join(self.save_dir, "atomic_patterns.png"), bbox_inches='tight')
        plt.close(fig2)

    def plot_data_distribution(self, dataloader, num_samples=2000):
        self.model.eval()
        all_data = []
        count = 0
        with torch.no_grad():
            for x in dataloader:
                all_data.append(x.cpu().numpy())
                count += x.shape[0]
                if count >= num_samples:
                    break
        
        data = np.concatenate(all_data, axis=0)[:num_samples] # [N, C, T]
        data_flat = data.transpose(1, 0, 2).reshape(8, -1) # [8, N*T]
        
        fig, axes = plt.subplots(4, 2, figsize=(15, 12))
        axes = axes.flatten()
        for i in range(8):
            axes[i].hist(data_flat[i], bins=50, color=BLUE_PALETTE[i % len(BLUE_PALETTE)], alpha=0.7)
            apply_ax_style(axes[i], title=f"Channel {i+1} Distribution")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "data_distribution.png"), bbox_inches='tight')
        plt.close(fig)

    def plot_single_reconstruction(self, dataloader, sample_index=0):
        self.model.eval()
        batch_size = dataloader.batch_size
        batch_idx = sample_index // batch_size
        intra_batch_idx = sample_index % batch_size
        
        for i, x in enumerate(dataloader):
            if i == batch_idx:
                x = x.to(self.device)
                with torch.no_grad():
                    x_recon, _, _, _ = self.model(x)
                
                orig = x[intra_batch_idx].cpu().numpy()
                recon = x_recon[intra_batch_idx].cpu().numpy()
                
                fig, axes = plt.subplots(8, 1, figsize=(12, 16), sharex=True)
                t = np.arange(orig.shape[1])
                for ch in range(8):
                    axes[ch].plot(t, orig[ch], color=COLORS['secondary'], alpha=0.7, label='Original')
                    axes[ch].plot(t, recon[ch], color=COLORS['primary'], alpha=0.8, label='Reconstruction', linestyle='--')
                    apply_ax_style(axes[ch], ylabel=f'Ch {ch+1}')
                    if ch == 0:
                        axes[ch].legend(loc='upper right')
                
                axes[7].set_xlabel("Time Steps")
                plt.suptitle(f"Sample {sample_index} Reconstruction", fontsize=16, fontweight='bold')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(os.path.join(self.save_dir, f"recon_sample_{sample_index}.png"), bbox_inches='tight')
                plt.close(fig)
                break

    def plot_gesture_pipeline(self, df, label_id, dataset_obj, repetition_index=0):
        self.model.eval()
        indices = df.index[df['gt'] == label_id].to_numpy()
        if len(indices) == 0: return

        breaks = np.where(np.diff(indices) > 1)[0]
        block_starts = [indices[0]] + [indices[i+1] for i in breaks]
        valid_starts = []
        for s in block_starts:
            for offset in [0, 1, 2, 3]:
                if self._is_valid_rep(df, s + offset*self.SAMPLES_PER_REP, label_id):
                    valid_starts.append(s + offset*self.SAMPLES_PER_REP)

        if repetition_index >= len(valid_starts): return
        final_start = valid_starts[repetition_index]

        emg_cols = [c for c in df.columns if 'emg' in c.lower()]
        raw_vals = df[emg_cols].values
        raw_seg = raw_vals[final_start : final_start + self.SAMPLES_PER_REP]
        # Apply the same logic as training/unseen: Filter -> Scale
        full_filtered = dataset_obj._apply_filters(raw_vals)
        proc_seg_norm = dataset_obj.scaler.transform(full_filtered[final_start : final_start + self.SAMPLES_PER_REP])

        # To keep everything in same scale for comparison:
        # We unnormalize the 'Preprocessed' view as well
        proc_seg_unnorm = dataset_obj.scaler.inverse_transform(proc_seg_norm)

        inp = torch.tensor(proc_seg_norm, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            recon, _, _, _ = self.model(inp)
            recon_norm = recon[0].cpu().numpy() # [C, T]
            recon_unnorm = dataset_obj.scaler.inverse_transform(recon_norm.T).T

        fig, axes = plt.subplots(8, 3, figsize=(18, 12), sharex='col')
        t = np.linspace(0, self.REP_DURATION_SEC, self.SAMPLES_PER_REP)

        titles = [f"Raw (Label {label_id})", "Preprocessed", "Reconstruction"]
        for i, tit in enumerate(titles):
            color_tit = (i == 2)
            axes[0, i].set_title(tit, fontweight='bold', color=COLORS['text_primary'] if not color_tit else COLORS['text_secondary'])

        for ch in range(8):
            axes[ch, 0].plot(t, raw_seg[:, ch], color=COLORS['secondary'], alpha=0.7)
            axes[ch, 1].plot(t, proc_seg_unnorm[:, ch], color=COLORS['primary'], alpha=0.8)
            axes[ch, 2].plot(t, recon_unnorm[ch], color=COLORS['primary'], alpha=0.9, linestyle='--')

            apply_ax_style(axes[ch, 0], ylabel=f'Ch {ch+1}')
            apply_ax_style(axes[ch, 1])
            apply_ax_style(axes[ch, 2])

        axes[7, 1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"pipeline_trace_label_{label_id}_rep_{repetition_index}.png"), bbox_inches='tight')
        plt.close(fig)

    def plot_unseen_comparison(self, raw_df, participant_id, label_id, label_name, dataset_obj):
        self.model.eval()
        indices = raw_df.index[raw_df['label'] == label_name].to_numpy()
        if len(indices) < self.SAMPLES_PER_REP: return
        final_start = indices[-self.SAMPLES_PER_REP]
        emg_cols = [c for c in raw_df.columns if 'emg' in c.lower()]
        raw_vals = raw_df[emg_cols].values
        raw_seg = raw_vals[final_start : final_start + self.SAMPLES_PER_REP]
        
        full_filtered = dataset_obj._apply_filters(raw_vals)
        proc_seg_norm = dataset_obj.scaler.transform(full_filtered[final_start : final_start + self.SAMPLES_PER_REP])
        proc_seg_unnorm = dataset_obj.scaler.inverse_transform(proc_seg_norm)

        inp = torch.tensor(proc_seg_norm, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            recon, _, _, _ = self.model(inp)
            recon_norm = recon[0].cpu().numpy()
            recon_unnorm = dataset_obj.scaler.inverse_transform(recon_norm.T).T

        fig, axes = plt.subplots(8, 3, figsize=(18, 12), sharex='col')
        t = np.linspace(0, self.REP_DURATION_SEC, self.SAMPLES_PER_REP)
        for i, tit in enumerate([f"Raw (Label {label_id})", "Preprocessed", "Reconstruction"]):
            color_tit = (i == 2)
            axes[0, i].set_title(tit, fontweight='bold', color=COLORS['text_primary'] if not color_tit else COLORS['text_secondary'])

        for ch in range(8):
            axes[ch, 0].plot(t, raw_seg[:, ch], color=COLORS['secondary'], alpha=0.7)
            axes[ch, 1].plot(t, proc_seg_unnorm[:, ch], color=COLORS['primary'], alpha=0.8)
            axes[ch, 2].plot(t, recon_unnorm[ch], color=COLORS['primary'], alpha=0.9, linestyle='--')
            apply_ax_style(axes[ch, 0], ylabel=f'Ch {ch+1}')
            apply_ax_style(axes[ch, 1])
            apply_ax_style(axes[ch, 2])

        plt.suptitle(f"Participant {participant_id} - {label_name} (Unseen Repetition)", fontsize=16, fontweight='bold', color=COLORS['text_primary'])
        axes[7, 1].set_xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.save_dir, f"unseen_comparison_P{participant_id}_{label_name.replace(' ', '_')}.png"), bbox_inches='tight')
        plt.close(fig)

    def _is_valid_rep(self, df, start_idx, label_id):
        if start_idx + self.SAMPLES_PER_REP >= len(df): return False
        return (df['gt'].iloc[start_idx] == label_id) and \
               (df['gt'].iloc[start_idx + self.SAMPLES_PER_REP // 2] == label_id) and \
               (df['gt'].iloc[start_idx + self.SAMPLES_PER_REP - 1] == label_id)
