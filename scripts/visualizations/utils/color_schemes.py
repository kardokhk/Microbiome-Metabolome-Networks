"""Consistent color schemes for all visualizations."""

import matplotlib.pyplot as plt
import seaborn as sns

# Disease category colors
DISEASE_COLORS = {
    'IBD': '#3498db',           # Blue
    'CRC': '#e74c3c',           # Red
    'Gastric': '#e67e22',       # Orange
    'IBS': '#2ecc71',           # Green
    'ESRD': '#9b59b6',          # Purple
}

# Dataset-specific colors (maps to disease categories)
DATASET_COLORS = {
    'YACHIDA_CRC_2019': '#e74c3c',                      # CRC - Red
    'ERAWIJANTARI_GASTRIC_CANCER_2020': '#e67e22',     # Gastric - Orange
    'FRANZOSA_IBD_2019': '#3498db',                     # IBD - Blue
    'iHMP_IBDMDB_2019': '#2980b9',                      # IBD - Darker blue
    'MARS_IBS_2020': '#2ecc71',                         # IBS - Green
    'WANG_ESRD_2020': '#9b59b6',                        # ESRD - Purple
}

# Control vs Disease group colors
GROUP_COLORS = {
    'control': '#27ae60',       # Green
    'disease': '#e74c3c',       # Red
    'Control': '#27ae60',
    'Disease': '#e74c3c',
}

# Differential status colors
STATUS_COLORS = {
    'depleted_in_disease': '#3498db',      # Blue
    'enriched_in_disease': '#e74c3c',      # Red
    'control_specific': '#27ae60',         # Dark green
    'disease_specific': '#c0392b',         # Dark red
    'no_change': '#95a5a6',                # Gray
    'stable': '#95a5a6',
}

# Correlation sign colors
CORRELATION_COLORS = {
    'positive': '#3498db',      # Blue
    'negative': '#e74c3c',      # Red
}

# Phylum colors (major gut phyla)
PHYLUM_COLORS = {
    'firmicutes': '#e74c3c',
    'firmicutes a': '#e74c3c',
    'firmicutes c': '#c0392b',
    'bacteroidota': '#3498db',
    'bacteroidetes': '#3498db',
    'actinobacteriota': '#2ecc71',
    'actinobacteria': '#2ecc71',
    'proteobacteria': '#f39c12',
    'verrucomicrobiota': '#9b59b6',
    'verrucomicrobia': '#9b59b6',
    'other': '#95a5a6',
}

# Metabolite class colors
METABOLITE_CLASS_COLORS = {
    'SCFA': '#e74c3c',
    'Bile acid': '#f39c12',
    'Amino acid': '#3498db',
    'Lipid': '#9b59b6',
    'Carbohydrate': '#2ecc71',
    'Nucleotide': '#1abc9c',
    'Vitamin': '#e67e22',
    'Other': '#95a5a6',
    'Unknown': '#bdc3c7',
}


def get_dataset_color(dataset_name):
    """Get color for a dataset."""
    return DATASET_COLORS.get(dataset_name, '#95a5a6')


def get_disease_category(dataset_name):
    """Map dataset to disease category."""
    mapping = {
        'YACHIDA_CRC_2019': 'CRC',
        'ERAWIJANTARI_GASTRIC_CANCER_2020': 'Gastric',
        'FRANZOSA_IBD_2019': 'IBD',
        'iHMP_IBDMDB_2019': 'IBD',
        'MARS_IBS_2020': 'IBS',
        'WANG_ESRD_2020': 'ESRD',
    }
    return mapping.get(dataset_name, 'Unknown')


def get_phylum_color(phylum_name):
    """Get color for a phylum (case-insensitive)."""
    if not phylum_name or str(phylum_name).lower() == 'nan':
        return PHYLUM_COLORS['other']

    phylum_lower = str(phylum_name).lower()
    for key, color in PHYLUM_COLORS.items():
        if key in phylum_lower:
            return color
    return PHYLUM_COLORS['other']


def create_discrete_colormap(n_colors, palette='Set2'):
    """Create discrete colormap with n colors."""
    if n_colors <= 8:
        return sns.color_palette(palette, n_colors)
    else:
        return sns.color_palette('husl', n_colors)


# Colormaps for continuous data
CONTINUOUS_CMAPS = {
    'correlation': 'RdBu_r',           # Red-Blue diverging
    'log2fc': 'RdBu_r',                # Red-Blue diverging
    'pvalue': 'YlOrRd',                # Yellow-Orange-Red
    'enrichment': 'YlGnBu',            # Yellow-Green-Blue
    'density': 'viridis',              # Viridis
}


def get_continuous_cmap(data_type='default'):
    """Get continuous colormap for data type."""
    return CONTINUOUS_CMAPS.get(data_type, 'viridis')
