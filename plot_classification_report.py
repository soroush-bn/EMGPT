import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from viz_style import COLORS, apply_ax_style

def plot_classification_results(csv_path, save_dir):
    """
    Reads the classification results CSV and plots a grouped bar chart.
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    x_labels = []
    for _, row in df.iterrows():
        label = row['Experiment'].replace("Exp ", "")
        if str(row['Ratio']) != "nan" and row['Ratio'] != "N/A":
            label += f"\n({row['Ratio']})"
        x_labels.append(label)

    x = np.arange(len(x_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Between-Subj: secondary color, Within-Subj: primary color
    rects1 = ax.bar(x - width/2, df['Between-Subj Acc'] * 100, width, label='Between-Subject', color=COLORS['secondary'], alpha=0.85)
    rects2 = ax.bar(x + width/2, df['Within-Subj Acc'] * 100, width, label='Within-Subject', color=COLORS['primary'], alpha=0.85)

    apply_ax_style(ax, title='Classification Performance across Data Ratios', ylabel='Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=9)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold', color=COLORS['text_primary'])

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "classification_results_bar_chart.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Classification bar chart saved to: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    args = parser.parse_args()
    plot_classification_results(args.csv, args.save_dir)
