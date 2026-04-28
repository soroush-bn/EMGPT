import matplotlib.pyplot as plt

# --- Color Palette (Color-blind friendly Blue Shades) ---
COLORS = {
    'primary': '#054984',      # Deep Blue (Preprocessed / Synthetic / Reconstruction)
    'secondary': '#335067',    # Muted Blue-Grey (Raw / Baseline / Real)
    'accent': '#0072b2',       # Bright Blue
    'highlight': '#56b4e9',    # Sky Blue
    'grid': '#000000',         # Black for grid lines
    'text_primary': '#335067', # Muted Blue-Grey for titles
    'text_secondary': '#054984'# Deep Blue for secondary labels
}

# --- Shared Blue Palette for multi-channel plots ---
BLUE_PALETTE = [
    '#054984', '#335067', '#0072b2', '#56b4e9', 
    '#009e73', '#004d40', '#1a237e', '#3f51b5'
]

# --- Global Font Styles ---
FONT_STYLES = {
    'title_size': 16,
    'subtitle_size': 14,
    'label_size': 12,
    'tick_size': 10,
    'legend_size': 'small',
    'font_weight': 'bold'
}

def apply_ax_style(ax, title=None, xlabel=None, ylabel=None, color_title=False):
    """
    Standardizes the look of a single axis.
    """
    if title:
        color = COLORS['text_primary'] if not color_title else COLORS['text_secondary']
        ax.set_title(title, fontsize=FONT_STYLES['subtitle_size'], fontweight=FONT_STYLES['font_weight'], color=color)
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_STYLES['label_size'])
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_STYLES['label_size'], fontweight=FONT_STYLES['font_weight'])

    ax.grid(True, alpha=0.15, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=FONT_STYLES['tick_size'])

def setup_figure(title=None, figsize=(15, 7)):
    """
    Creates a standardized figure and applies a global title.
    """
    fig = plt.figure(figsize=figsize)
    if title:
        plt.suptitle(title, fontsize=FONT_STYLES['title_size'], fontweight=FONT_STYLES['font_weight'], color=COLORS['text_primary'])
    return fig
