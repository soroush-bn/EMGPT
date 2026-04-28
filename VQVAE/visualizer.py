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
        full_filtered = dataset_obj._apply_filters(raw_vals)
        proc_seg_full = dataset_obj.scaler.transform(full_filtered[final_start : final_start + self.SAMPLES_PER_REP])
        
        inp = torch.tensor(proc_seg_full, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            recon, _, _, _ = self.model(inp)
            recon_seg = recon[0].cpu().numpy()

        fig, axes = plt.subplots(8, 3, figsize=(18, 12), sharex='col')
        t = np.linspace(0, self.REP_DURATION_SEC, self.SAMPLES_PER_REP)
        
        titles = [f"Raw (Label {label_id})", "Preprocessed", "Reconstruction"]
        for i, tit in enumerate(titles):
            color_tit = (i == 2)
            axes[0, i].set_title(tit, fontweight='bold', color=COLORS['text_primary'] if not color_tit else COLORS['text_secondary'])

        for ch in range(8):
            axes[ch, 0].plot(t, raw_seg[:, ch], color=COLORS['secondary'], alpha=0.7)
            axes[ch, 1].plot(t, proc_seg_full[:, ch], color=COLORS['primary'], alpha=0.8)
            axes[ch, 2].plot(t, recon_seg[ch], color=COLORS['primary'], alpha=0.9, linestyle='--')
            
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
        proc_seg_full = dataset_obj.scaler.transform(full_filtered[final_start : final_start + self.SAMPLES_PER_REP])
        inp = torch.tensor(proc_seg_full, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            recon, _, _, _ = self.model(inp)
            recon_seg = recon[0].cpu().numpy()

        fig, axes = plt.subplots(8, 3, figsize=(18, 12), sharex='col')
        t = np.linspace(0, self.REP_DURATION_SEC, self.SAMPLES_PER_REP)
        for i, tit in enumerate([f"Raw (Label {label_id})", "Preprocessed", "Reconstruction"]):
            color_tit = (i == 2)
            axes[0, i].set_title(tit, fontweight='bold', color=COLORS['text_primary'] if not color_tit else COLORS['text_secondary'])

        for ch in range(8):
            axes[ch, 0].plot(t, raw_seg[:, ch], color=COLORS['secondary'], alpha=0.7)
            axes[ch, 1].plot(t, proc_seg_full[:, ch], color=COLORS['primary'], alpha=0.8)
            axes[ch, 2].plot(t, recon_seg[ch], color=COLORS['primary'], alpha=0.9, linestyle='--')
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
