import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from encoded_model import ConditionedGPT, GPTConfig
from encoded_dataset import EncodedEMGDataset
import argparse
import pathlib
from viz_style import COLORS, BLUE_PALETTE, apply_ax_style

def visualize_attention(config_path, model_path, layer_to_viz=0, sample_idx=0, save_dir="./attention_plots"):
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
    
    config = full_config
    vqvae_config = full_config.get('vqvae', full_config)

    if not os.path.exists(save_dir): os.makedirs(save_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    gptconf = GPTConfig(
        n_layer=config.get('n_layer', 8),
        n_head=config.get('n_head', 8),
        n_embd=config.get('n_embd', 512),
        block_size=config.get('block_size', 75),
        vocab_size=config.get('vocab_size', 512),
        num_classes=config.get('num_classes', 17),
        dropout=0, bias=config.get('bias', True)
    )
    model = ConditionedGPT(gptconf)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix): state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device); model.eval()

    attention_weights = []
    def hook_fn(module, input, output):
        x = input[0]; B, T, C = x.size()
        q, k, v = module.c_attn(x).split(module.n_embd, dim=2)
        k = k.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
        q = q.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(k.size(-1))))
        mask = torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = torch.nn.functional.softmax(att, dim=-1)
        attention_weights.append(att.detach().cpu().numpy())

    model.transformer.h[layer_to_viz].attn.register_forward_hook(hook_fn)

    val_data_path = config.get('val_data_path')
    dataset = EncodedEMGDataset(csv_files=[val_data_path])
    block_size = config.get('block_size', 75)
    full_seq = dataset.tokens[sample_idx][:block_size]
    X = torch.tensor(full_seq, dtype=torch.long).unsqueeze(0).to(device)
    label = torch.tensor([dataset.labels[sample_idx]]).to(device)

    with torch.no_grad(): _ = model(X, labels=label)

    from decoder import VQVAESignalDecoder
    from VQVAE.dataset import EMGDataset
    vq_ckpt = f"VQVAE/models/{vqvae_config['name']}/final_model.pth"
    try:
        train_ds = EMGDataset(vqvae_config, split='train')
        decoder = VQVAESignalDecoder(vqvae_model_path=vq_ckpt, vqvae_config=vqvae_config, scaler=train_ds.scaler)
        recon_input = decoder.decode_window(X)[0]
        fig_inp, ax_inp = plt.subplots(figsize=(10, 4))
        for c in range(recon_input.shape[1]):
            color = BLUE_PALETTE[c % len(BLUE_PALETTE)]
            ax_inp.plot(recon_input[:, c], color=color, alpha=0.7, lw=1.2)
        apply_ax_style(ax_inp, title=f"Reconstructed Input Signal (Sample {sample_idx} | Label {label.item()})")
        plt.savefig(os.path.join(save_dir, f"input_signal_sample_{sample_idx}.png"))
        plt.close(fig_inp)
    except Exception as e: print(f"Could not reconstruct input signal: {e}")

    weights = attention_weights[0][0]
    n_heads = weights.shape[0]; heads_to_plot = min(n_heads, 16) 
    cols = 4; rows = int(np.ceil(heads_to_plot / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    fig.suptitle(f"Layer {layer_to_viz} Attention Heads | Gesture {label.item()}", fontsize=20, fontweight='bold', color=COLORS['text_primary'])
    axes = axes.flatten()
    for i in range(heads_to_plot):
        sns.heatmap(weights[i], ax=axes[i], cmap='Blues', cbar=False)
        apply_ax_style(axes[i], title=f"Head {i}", xlabel="Key Position", ylabel="Query Position", color_title=True)
    for j in range(heads_to_plot, len(axes)): fig.delaxes(axes[j])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, f"attention_layer_{layer_to_viz}_sample_{sample_idx}.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default=".")
    args = parser.parse_args()
    visualize_attention(args.config, args.ckpt, args.layer, args.sample, args.save_dir)
