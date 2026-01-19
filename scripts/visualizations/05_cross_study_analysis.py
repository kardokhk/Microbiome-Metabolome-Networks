#!/usr/bin/env python3
"""
Generate cross-study integration visualizations.

This script creates the KEY FINDINGS figures showing conserved patterns
across all datasets, including the 35 species consistently depleted in disease.

Outputs:
- Figure 5A: Conserved keystones comparison (control vs disease)
- Figure 5B: Depleted species heatmap (THE KEY FINDING!)
- Figure 5C: Cross-dataset pattern summary
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.visualizations.utils import (
    DISEASE_COLORS, DATASET_COLORS, GROUP_COLORS,
    get_disease_category, get_dataset_color, get_phylum_color,
    setup_publication_style, save_figure,
    add_panel_label, FIGURES_DIR, TABLES_DIR,
    DATASETS, DISEASE_NAMES, RESULTS_DIR,
    load_cross_study_data, load_differential_data
)


def extract_species_name(full_name):
    """Extract genus and species from full taxonomic name."""
    if pd.isna(full_name):
        return 'Unknown'

    parts = str(full_name).split(';')

    # Try to get genus and species
    genus = None
    species = None

    for part in parts:
        part = part.strip()
        if part.startswith('g  ') or part.startswith('g__'):
            genus = part.replace('g  ', '').replace('g__', '').strip()
        elif part.startswith('s  ') or part.startswith('s__'):
            species = part.replace('s  ', '').replace('s__', '').strip()

    if genus and species:
        # Remove genus name from species if present
        if species.startswith(genus):
            species = species[len(genus):].strip()
        return f"{genus.capitalize()} {species}"
    elif genus:
        return genus.capitalize()
    elif species:
        return species
    else:
        return 'Unknown'


def extract_phylum(full_name):
    """Extract phylum from full taxonomic name."""
    if pd.isna(full_name):
        return 'Unknown'

    parts = str(full_name).split(';')

    for part in parts:
        part = part.strip()
        if part.startswith('p  ') or part.startswith('p__'):
            return part.replace('p  ', '').replace('p__', '').strip().capitalize()

    return 'Unknown'


def create_conserved_keystones_comparison():
    """Create Figure 5A: Conserved keystones (control vs disease)."""
    print("Creating Figure 5A: Conserved keystones comparison...")

    # Load cross-study data
    data = load_cross_study_data()

    if data is None or 'conserved_control' not in data or 'conserved_disease' not in data:
        print("  Cross-study data not available")
        return

    # Count species by number of datasets
    control_df = data['conserved_control']
    disease_df = data['conserved_disease']

    # Get distribution by n_datasets
    control_counts = control_df['n_datasets'].value_counts().sort_index()
    disease_counts = disease_df['n_datasets'].value_counts().sort_index()

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Bar chart comparison
    x = np.arange(3, 7)  # Datasets 3-6
    width = 0.35

    control_values = [control_counts.get(i, 0) for i in x]
    disease_values = [disease_counts.get(i, 0) for i in x]

    bars1 = ax1.bar(x - width/2, control_values, width, label='Control',
                    color=GROUP_COLORS['control'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, disease_values, width, label='Disease',
                    color=GROUP_COLORS['disease'], alpha=0.8)

    ax1.set_xlabel('Number of Datasets', fontsize=11)
    ax1.set_ylabel('Number of Conserved Keystone Species', fontsize=11)
    ax1.set_title('Cross-Study Keystone Conservation', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.legend(frameon=True)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)

    # Panel B: Summary pie charts
    total_control = len(control_df)
    total_disease = len(disease_df)

    categories = ['≥3 datasets', '≥4 datasets', '≥5 datasets', 'All 6 datasets']
    control_cumsum = [
        len(control_df[control_df['n_datasets'] >= 3]),
        len(control_df[control_df['n_datasets'] >= 4]),
        len(control_df[control_df['n_datasets'] >= 5]),
        len(control_df[control_df['n_datasets'] == 6]),
    ]
    disease_cumsum = [
        len(disease_df[disease_df['n_datasets'] >= 3]),
        len(disease_df[disease_df['n_datasets'] >= 4]),
        len(disease_df[disease_df['n_datasets'] >= 5]),
        len(disease_df[disease_df['n_datasets'] == 6]),
    ]

    # Create table
    summary_data = pd.DataFrame({
        'Threshold': categories,
        'Control': control_cumsum,
        'Disease': disease_cumsum,
        'Difference': np.array(control_cumsum) - np.array(disease_cumsum)
    })

    # Display as table in ax2
    ax2.axis('tight')
    ax2.axis('off')

    table = ax2.table(cellText=summary_data.values,
                     colLabels=summary_data.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.25])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header
    for i in range(len(summary_data.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows
    for i in range(len(summary_data)):
        for j in range(len(summary_data.columns)):
            if j == 0:
                table[(i+1, j)].set_facecolor('#ecf0f1')

    ax2.set_title('Conserved Keystones Summary', fontsize=12, fontweight='bold', pad=20)

    # Add panel labels
    add_panel_label(ax1, 'A')
    add_panel_label(ax2, 'B')

    plt.tight_layout()

    # Save
    save_figure(fig, '05A_conserved_keystones_comparison', subdir='05_cross_study_integration')
    plt.close()

    print(f"  Control keystones (≥3 datasets): {control_cumsum[0]}")
    print(f"  Disease keystones (≥3 datasets): {disease_cumsum[0]}")
    print(f"  Difference: {control_cumsum[0] - disease_cumsum[0]} species")


def create_depleted_species_heatmap():
    """Create Figure 5B: Heatmap of species depleted across datasets (KEY FINDING!)."""
    print("Creating Figure 5B: Depleted species heatmap (KEY FINDING)...")

    # Load cross-study depleted species
    data = load_cross_study_data()

    if data is None or 'depleted' not in data:
        print("  Depleted species data not available")
        return

    depleted_df = data['depleted']
    print(f"  Found {len(depleted_df)} species depleted across ≥3 datasets")

    # Load log2FC values for each species across datasets
    log2fc_matrix = []
    species_list = depleted_df['species'].values

    for dataset in DATASETS:
        diff_data = load_differential_data(dataset)

        if diff_data is None or 'keystones' not in diff_data:
            continue

        diff_keystones = diff_data['keystones']

        # Get log2FC for each depleted species
        dataset_log2fc = []
        for species in species_list:
            match = diff_keystones[diff_keystones['species'] == species]

            if len(match) > 0:
                log2fc = match.iloc[0]['log2_fc']
                dataset_log2fc.append(log2fc)
            else:
                dataset_log2fc.append(0)  # Not found in this dataset

        log2fc_matrix.append(dataset_log2fc)

    # Convert to array
    log2fc_matrix = np.array(log2fc_matrix).T  # Species x Datasets

    # Create nice species labels
    species_labels = [extract_species_name(s) for s in species_list]

    # Limit to top 30 for visibility
    n_show = min(30, len(species_list))
    log2fc_show = log2fc_matrix[:n_show, :]
    species_show = species_labels[:n_show]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 12))

    # Create heatmap
    im = ax.imshow(log2fc_show, cmap='RdBu_r', aspect='auto', vmin=-5, vmax=5)

    # Set ticks
    ax.set_xticks(np.arange(len(DATASETS)))
    ax.set_yticks(np.arange(n_show))

    # Set labels
    dataset_labels = [DISEASE_NAMES[d] for d in DATASETS]
    ax.set_xticklabels(dataset_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(species_show, fontsize=8, style='italic')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('log2(Disease/Control)', rotation=270, labelpad=15, fontsize=10)

    # Add title
    ax.set_title(f'Top {n_show} Species Consistently Depleted in Disease\n(Depleted in ≥3 Datasets)',
                fontsize=12, fontweight='bold', pad=10)

    # Add grid
    ax.set_xticks(np.arange(len(DATASETS)+1)-.5, minor=True)
    ax.set_yticks(np.arange(n_show+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

    # Add significance markers (asterisks for strong depletion)
    for i in range(n_show):
        for j in range(len(DATASETS)):
            value = log2fc_show[i, j]
            if value < -2:  # Strong depletion
                ax.text(j, i, '***', ha='center', va='center',
                       fontsize=8, fontweight='bold', color='black')
            elif value < -1:  # Moderate depletion
                ax.text(j, i, '*', ha='center', va='center',
                       fontsize=10, fontweight='bold', color='black')

    plt.tight_layout()

    # Save
    save_figure(fig, '05B_depleted_species_heatmap', subdir='05_cross_study_integration')
    plt.close()


def create_cross_dataset_summary():
    """Create Figure 5C: Cross-dataset pattern summary."""
    print("Creating Figure 5C: Cross-dataset pattern summary...")

    # Load data from all differential analyses
    disruption_data = []

    for dataset in DATASETS:
        diff_data = load_differential_data(dataset)

        if diff_data is None or 'summary' not in diff_data:
            continue

        summary = diff_data['summary']

        try:
            depleted = summary.loc['Depleted in disease (log2FC < -1)', 'difference']
            enriched = summary.loc['Enriched in disease (log2FC > 1)', 'difference']
            control_specific = summary.loc['Control-specific keystones', 'control']
            disease_specific = summary.loc['Disease-specific keystones', 'disease']

            disruption_data.append({
                'Dataset': dataset,
                'Disease': DISEASE_NAMES[dataset],
                'Category': get_disease_category(dataset),
                'Depleted': abs(depleted),
                'Enriched': enriched,
                'Control-specific': control_specific,
                'Disease-specific': disease_specific
            })
        except Exception as e:
            print(f"  Warning: Could not extract data for {dataset}: {e}")

    if not disruption_data:
        print("  No disruption data available")
        return

    df = pd.DataFrame(disruption_data)

    # Create figure with 2 panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Stacked bar chart of changes
    datasets = df['Disease'].values
    x_pos = np.arange(len(datasets))
    width = 0.6

    # Stack depleted (negative) and enriched (positive)
    ax1.barh(x_pos, -df['Depleted'], width, label='Depleted',
            color='#3498db', alpha=0.8)
    ax1.barh(x_pos, df['Enriched'], width, label='Enriched',
            color='#e74c3c', alpha=0.8)

    ax1.set_yticks(x_pos)
    ax1.set_yticklabels(datasets, fontsize=10)
    ax1.set_xlabel('Number of Species', fontsize=11)
    ax1.set_title('Keystone Species Changes\n(Disease vs Control)', fontsize=12, fontweight='bold')
    ax1.legend(frameon=True, loc='lower right')
    ax1.axvline(x=0, color='black', linewidth=1, linestyle='-')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # Panel B: Group-specific species
    x_pos2 = np.arange(len(datasets))
    width2 = 0.35

    bars1 = ax2.barh(x_pos2 - width2/2, df['Control-specific'], width2,
                    label='Control-specific', color=GROUP_COLORS['control'], alpha=0.8)
    bars2 = ax2.barh(x_pos2 + width2/2, df['Disease-specific'], width2,
                    label='Disease-specific', color=GROUP_COLORS['disease'], alpha=0.8)

    ax2.set_yticks(x_pos2)
    ax2.set_yticklabels(datasets, fontsize=10)
    ax2.set_xlabel('Number of Species', fontsize=11)
    ax2.set_title('Group-Specific Keystone Species', fontsize=12, fontweight='bold')
    ax2.legend(frameon=True)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            width_val = bar.get_width()
            if width_val > 0:
                ax2.text(width_val, bar.get_y() + bar.get_height()/2.,
                        f'{int(width_val)}', ha='left', va='center',
                        fontsize=8, fontweight='bold')

    # Add panel labels
    add_panel_label(ax1, 'A')
    add_panel_label(ax2, 'B')

    plt.tight_layout()

    # Save
    save_figure(fig, '05C_cross_dataset_summary', subdir='05_cross_study_integration')
    plt.close()

    # Print summary
    print(f"  Average depleted species: {df['Depleted'].mean():.0f}")
    print(f"  Average enriched species: {df['Enriched'].mean():.0f}")


def create_summary_table():
    """Create summary table of cross-study findings."""
    print("Creating summary table...")

    data = load_cross_study_data()

    if data is None:
        return

    summary = {
        'Metric': [
            'Total conserved keystones (control, ≥3 datasets)',
            'Total conserved keystones (disease, ≥3 datasets)',
            'Reduction in disease (%)',
            'Species consistently depleted (≥3 datasets)',
            'Top depleted species (present in most datasets)'
        ],
        'Value': []
    }

    # Add values
    if 'conserved_control' in data:
        n_control = len(data['conserved_control'][data['conserved_control']['n_datasets'] >= 3])
        summary['Value'].append(n_control)
    else:
        summary['Value'].append('N/A')

    if 'conserved_disease' in data:
        n_disease = len(data['conserved_disease'][data['conserved_disease']['n_datasets'] >= 3])
        summary['Value'].append(n_disease)

        # Calculate reduction
        if isinstance(summary['Value'][0], int):
            reduction = 100 * (1 - n_disease / summary['Value'][0])
            summary['Value'].append(f"{reduction:.1f}%")
        else:
            summary['Value'].append('N/A')
    else:
        summary['Value'].append('N/A')
        summary['Value'].append('N/A')

    if 'depleted' in data:
        n_depleted = len(data['depleted'])
        summary['Value'].append(n_depleted)

        # Get top species
        top_species = data['depleted'].nlargest(1, 'n_datasets_depleted')
        if len(top_species) > 0:
            top_name = extract_species_name(top_species.iloc[0]['species'])
            n_datasets = top_species.iloc[0]['n_datasets_depleted']
            summary['Value'].append(f"{top_name} ({n_datasets} datasets)")
        else:
            summary['Value'].append('N/A')
    else:
        summary['Value'].append('N/A')
        summary['Value'].append('N/A')

    summary_df = pd.DataFrame(summary)

    # Save
    csv_file = TABLES_DIR / 'table_cross_study_summary.csv'
    summary_df.to_csv(csv_file, index=False)
    print(f"  Saved: {csv_file.name}")

    return summary_df


def main():
    """Main function to generate all cross-study visualizations."""

    print("="*80)
    print("CROSS-STUDY INTEGRATION VISUALIZATIONS")
    print("="*80)
    print()

    # Setup style
    setup_publication_style()

    # Generate figures
    create_conserved_keystones_comparison()
    print()

    create_depleted_species_heatmap()
    print()

    create_cross_dataset_summary()
    print()

    # Generate table
    summary_df = create_summary_table()
    print()

    # Print summary
    print("="*80)
    print("KEY FINDINGS")
    print("="*80)
    if summary_df is not None:
        for idx, row in summary_df.iterrows():
            print(f"{row['Metric']}: {row['Value']}")

    print()
    print("Outputs saved to:")
    print(f"  Figures: {FIGURES_DIR}/")
    print(f"  Tables: {TABLES_DIR}/")
    print()
    print("✓ Cross-study integration visualizations complete!")


if __name__ == '__main__':
    main()
