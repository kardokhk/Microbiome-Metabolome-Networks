"""Common plotting utilities and style settings."""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Output directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / 'output' / 'figures'
TABLES_DIR = PROJECT_ROOT / 'output' / 'tables'

# Ensure directories exist
FIGURES_DIR.mkdir(exist_ok=True, parents=True)
TABLES_DIR.mkdir(exist_ok=True, parents=True)


def setup_publication_style():
    """Set matplotlib style for publication-quality figures."""

    # Reset to defaults first
    mpl.rcParams.update(mpl.rcParamsDefault)

    # Font settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['legend.title_fontsize'] = 9

    # Figure settings
    plt.rcParams['figure.dpi'] = 100  # Display DPI
    plt.rcParams['figure.figsize'] = (7, 5)
    plt.rcParams['savefig.dpi'] = 300  # Save DPI
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1

    # Axes settings
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    # Grid settings
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linewidth'] = 0.5

    # Line settings
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 6

    # Legend settings
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.loc'] = 'best'

    # Tick settings
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'

    # Set seaborn style
    sns.set_style("ticks", {
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
    })


def save_figure(fig, filename, formats=['pdf', 'png'], dpi=300, subdir=None):
    """Save figure in multiple formats.

    Args:
        fig: Matplotlib figure object
        filename: Base filename (without extension)
        formats: List of formats to save
        dpi: DPI for raster formats
        subdir: Optional subdirectory within figures/ (e.g., '01_study_overview')
    """
    # Determine output directory
    if subdir:
        output_dir = FIGURES_DIR / subdir
        output_dir.mkdir(exist_ok=True, parents=True)
    else:
        output_dir = FIGURES_DIR
    
    for fmt in formats:
        output_path = output_dir / f"{filename}.{fmt}"
        fig.savefig(output_path, format=fmt, dpi=dpi, bbox_inches='tight')
        # Print relative path for clarity
        rel_path = output_path.relative_to(FIGURES_DIR)
        print(f"  Saved: {rel_path}")


def add_significance_stars(ax, x1, x2, y, pvalue, height=0.02):
    """Add significance stars to plot.

    Args:
        ax: Matplotlib axes
        x1, x2: x positions for bracket
        y: y position for bracket
        pvalue: p-value
        height: Height of bracket
    """
    # Determine significance level
    if pvalue < 0.001:
        stars = '***'
    elif pvalue < 0.01:
        stars = '**'
    elif pvalue < 0.05:
        stars = '*'
    else:
        stars = 'ns'

    # Draw bracket
    ax.plot([x1, x1, x2, x2], [y, y+height, y+height, y], 'k-', linewidth=1)

    # Add stars
    ax.text((x1+x2)/2, y+height, stars, ha='center', va='bottom', fontsize=10)


def format_pvalue(pvalue):
    """Format p-value for display.

    Args:
        pvalue: p-value

    Returns:
        Formatted string
    """
    if pvalue < 0.001:
        return "p < 0.001"
    elif pvalue < 0.01:
        return f"p = {pvalue:.3f}"
    else:
        return f"p = {pvalue:.2f}"


def add_panel_label(ax, label, x=-0.1, y=1.05, fontsize=14, fontweight='bold'):
    """Add panel label (A, B, C, etc.) to subplot.

    Args:
        ax: Matplotlib axes
        label: Panel label (e.g., 'A', 'B')
        x, y: Position in axes coordinates
        fontsize: Font size
        fontweight: Font weight
    """
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight=fontweight,
            va='top', ha='right')


def truncate_labels(labels, max_length=30):
    """Truncate long labels.

    Args:
        labels: List of labels
        max_length: Maximum length

    Returns:
        List of truncated labels
    """
    return [label if len(label) <= max_length else label[:max_length-3] + '...'
            for label in labels]


def wrap_labels(labels, max_width=20):
    """Wrap long labels to multiple lines.

    Args:
        labels: List of labels
        max_width: Maximum width per line

    Returns:
        List of wrapped labels
    """
    import textwrap
    return ['\n'.join(textwrap.wrap(label, max_width)) for label in labels]


def add_value_labels(ax, spacing=5, format_str='{:.0f}'):
    """Add value labels to bar plot.

    Args:
        ax: Matplotlib axes
        spacing: Spacing from bar top
        format_str: Format string for values
    """
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        label = format_str.format(y_value)

        ax.annotate(label,
                   (x_value, y_value),
                   xytext=(0, spacing),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=8)


def create_legend_outside(ax, **kwargs):
    """Create legend outside plot area.

    Args:
        ax: Matplotlib axes
        **kwargs: Additional arguments for legend
    """
    default_kwargs = {
        'bbox_to_anchor': (1.05, 1),
        'loc': 'upper left',
        'frameon': False
    }
    default_kwargs.update(kwargs)
    return ax.legend(**default_kwargs)


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """Annotate heatmap with values.

    Args:
        im: AxesImage object
        data: Data array (if different from im data)
        valfmt: Format string
        textcolors: Tuple of (low, high) colors
        threshold: Threshold for color switching
        **textkw: Additional text kwargs
    """
    import numpy as np

    if data is None:
        data = im.get_array()

    if threshold is None:
        threshold = data.max() / 2.

    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(data[i, j] > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def despine(ax, top=True, right=True, left=False, bottom=False):
    """Remove spines from plot.

    Args:
        ax: Matplotlib axes
        top, right, left, bottom: Which spines to remove
    """
    if top:
        ax.spines['top'].set_visible(False)
    if right:
        ax.spines['right'].set_visible(False)
    if left:
        ax.spines['left'].set_visible(False)
    if bottom:
        ax.spines['bottom'].set_visible(False)


def set_axis_scientific(ax, axis='y', scilimits=(-3, 3)):
    """Set axis to scientific notation.

    Args:
        ax: Matplotlib axes
        axis: 'x' or 'y'
        scilimits: Limits for scientific notation
    """
    if axis == 'y':
        ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits)
    elif axis == 'x':
        ax.ticklabel_format(axis='x', style='scientific', scilimits=scilimits)


def add_grid(ax, axis='y', alpha=0.3, linestyle='--'):
    """Add grid to plot.

    Args:
        ax: Matplotlib axes
        axis: 'x', 'y', or 'both'
        alpha: Grid transparency
        linestyle: Line style
    """
    ax.grid(True, axis=axis, alpha=alpha, linestyle=linestyle, linewidth=0.5)


# Initialize style on import
setup_publication_style()
