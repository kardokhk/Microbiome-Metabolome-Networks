#!/usr/bin/env python3
"""
Generate differential analysis visualizations (control vs disease).

Outputs:
- Figure 4A: Volcano plots (per dataset)
- Figure 4B: Keystone score scatter plots
- Figure 4C: Network disruption summary (all datasets)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.visualizations.utils import (
    DISEASE_COLORS, DATASET_COLORS, GROUP_COLORS, STATUS_COLORS,
    get_disease_category, get_dataset_color, get_phylum_color,
    setup_publication_style, save_figure,
    add_panel_label, FIGURES_DIR, TABLES_DIR,
    DATASETS, DISEASE_NAMES, RESULTS_DIR,
    load_differential_data
)


def create_volcano_plot(dataset):
    """Create volcano plot for one dataset."""

    # Load differential data
    diff_data = load_differential_data(dataset)

    if diff_data is None or 'keystones' not in diff_data:
        print(f"  No differential data for {dataset}")
        return None

    df = diff_data['keystones']

    # Calculate -log10(p-value) if available, otherwise use arbitrary significance
    if 'pvalue' in df.columns:
        df['-log10_pvalue'] = -np.log10(df['pvalue'].clip(lower=1e-100))
    else:
        # Use absolute delta as proxy for significance
        df['-log10_pvalue'] = np.abs(df['delta_keystone_score']) * 10

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot points by status
    for status, color in STATUS_COLORS.items():
        if status not in df['status'].values:
            continue

        mask = df['status'] == status
        subset = df[mask]

        label = status.replace('_', ' ').title()

        ax.scatter(subset['log2_fc'], subset['-log10_pvalue'],
                  c=color, label=label, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

    # Add threshold lines
    ax.axvline(x=-1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Labels
    ax.set_xlabel('log2(Disease/Control)', fontsize=11)
    ax.set_ylabel('-log10(Significance)', fontsize=11)
    ax.set_title(f'{DISEASE_NAMES[dataset]}: Differential Keystone Species',
                fontsize=12, fontweight='bold')

    # Legend
    ax.legend(frameon=True, loc='upper right', fontsize=8)

    # Grid
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()

    return fig


def create_scatter_plot(dataset):
    """Create keystone score scatter plot for one dataset."""

    # Load differential data
    diff_data = load_differential_data(dataset)

    if diff_data is None or 'keystones' not in diff_data:
        return None

    df = diff_data['keystones']

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot points by status
    for status, color in STATUS_COLORS.items():
        if status not in df['status'].values:
            continue

        mask = df['status'] == status
        subset = df[mask]

        label = status.replace('_', ' ').title()

        ax.scatter(subset['keystone_score_control'], subset['keystone_score_disease'],
                  c=color, label=label, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

    # Add diagonal line (y=x)
    max_val = max(df['keystone_score_control'].max(), df['keystone_score_disease'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5, label='No change')

    # Labels
    ax.set_xlabel('Keystone Score (Control)', fontsize=11)
    ax.set_ylabel('Keystone Score (Disease)', fontsize=11)
    ax.set_title(f'{DISEASE_NAMES[dataset]}: Keystone Score Changes',
                fontsize=12, fontweight='bold')

    # Legend
    ax.legend(frameon=True, loc='upper left', fontsize=8)

    # Grid
    ax.grid(alpha=0.3, linestyle='--')

    # Equal aspect
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    return fig


def create_network_disruption_summary():
    """Create Figure 4C: Network disruption summary across all datasets."""
    print("Creating Figure 4C: Network disruption summary...")

    # Collect data from all datasets
    disruption_data = []

    for dataset in DATASETS:
        diff_data = load_differential_data(dataset)

        if diff_data is None or 'keystones' not in diff_data:
            continue

        df = diff_data['keystones']

        # Count by status
        status_counts = df['status'].value_counts()

        disruption_data.append({
            'Dataset': dataset,
            'Disease': DISEASE_NAMES[dataset],
            'Category': get_disease_category(dataset),
            'Depleted': status_counts.get('depleted_in_disease', 0),
            'Enriched': status_counts.get('enriched_in_disease', 0),
            'Control-specific': status_counts.get('control_specific', 0),
            'Disease-specific': status_counts.get('disease_specific', 0),
            'Total analyzed': len(df)
        })

    if not disruption_data:
        print("  No disruption data available")
        return

    df = pd.DataFrame(disruption_data)

    # Calculate percentages
    df['% Depleted'] = 100 * df['Depleted'] / df['Total analyzed']
    df['% Enriched'] = 100 * df['Enriched'] / df['Total analyzed']

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Absolute counts (stacked horizontal bar)
    datasets = df['Disease'].values
    y_pos = np.arange(len(datasets))

    # Create colors based on disease category
    colors = [get_dataset_color(d) for d in df['Dataset'].values]

    # Plot depleted and enriched as diverging bars
    ax1.barh(y_pos, -df['Depleted'], label='Depleted', color='#3498db', alpha=0.8)
    ax1.barh(y_pos, df['Enriched'], label='Enriched', color='#e74c3c', alpha=0.8)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(datasets, fontsize=10)
    ax1.set_xlabel('Number of Species', fontsize=11)
    ax1.set_title('Keystone Species Changes by Dataset', fontsize=12, fontweight='bold')
    ax1.legend(frameon=True, loc='lower right')
    ax1.axvline(x=0, color='black', linewidth=1.5)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (dep, enr) in enumerate(zip(df['Depleted'], df['Enriched'])):
        if dep > 0:
            ax1.text(-dep/2, i, str(int(dep)), ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')
        if enr > 0:
            ax1.text(enr/2, i, str(int(enr)), ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')

    # Panel B: Percentages (grouped bar chart)
    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax2.barh(x - width/2, df['% Depleted'], width,
                    label='% Depleted', color='#3498db', alpha=0.8)
    bars2 = ax2.barh(x + width/2, df['% Enriched'], width,
                    label='% Enriched', color='#e74c3c', alpha=0.8)

    ax2.set_yticks(x)
    ax2.set_yticklabels(datasets, fontsize=10)
    ax2.set_xlabel('Percentage of Analyzed Species (%)', fontsize=11)
    ax2.set_title('Proportion of Changes by Dataset', fontsize=12, fontweight='bold')
    ax2.legend(frameon=True)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            width_val = bar.get_width()
            if width_val > 0:
                ax2.text(width_val, bar.get_y() + bar.get_height()/2.,
                        f'{width_val:.1f}%', ha='left', va='center', fontsize=8)

    # Add panel labels
    add_panel_label(ax1, 'A')
    add_panel_label(ax2, 'B')

    plt.tight_layout()

    # Save
    save_figure(fig, '04C_network_disruption_summary', subdir='04_differential_analysis')
    plt.close()

    # Print summary
    print(f"  Average depleted species: {df['Depleted'].mean():.0f} ({df['% Depleted'].mean():.1f}%)")
    print(f"  Average enriched species: {df['Enriched'].mean():.0f} ({df['% Enriched'].mean():.1f}%)")


def create_differential_summary_table():
    """Create summary table of differential analysis results."""
    print("Creating differential analysis summary table...")

    summary_data = []

    for dataset in DATASETS:
        diff_data = load_differential_data(dataset)

        if diff_data is None:
            continue

        row = {
            'Dataset': dataset,
            'Disease': DISEASE_NAMES[dataset],
        }

        # Get counts from keystones file
        if 'keystones' in diff_data:
            df = diff_data['keystones']
            status_counts = df['status'].value_counts()

            row['Depleted'] = status_counts.get('depleted_in_disease', 0)
            row['Enriched'] = status_counts.get('enriched_in_disease', 0)
            row['Control-specific'] = status_counts.get('control_specific', 0)
            row['Disease-specific'] = status_counts.get('disease_specific', 0)
            row['Total'] = len(df)

        summary_data.append(row)

    if not summary_data:
        print("  No data available")
        return

    summary_df = pd.DataFrame(summary_data)

    # Add totals row
    totals = {
        'Dataset': 'TOTAL',
        'Disease': 'All diseases',
        'Depleted': summary_df['Depleted'].sum(),
        'Enriched': summary_df['Enriched'].sum(),
        'Control-specific': summary_df['Control-specific'].sum(),
        'Disease-specific': summary_df['Disease-specific'].sum(),
        'Total': summary_df['Total'].sum()
    }

    summary_df = pd.concat([summary_df, pd.DataFrame([totals])], ignore_index=True)

    # Save
    csv_file = TABLES_DIR / 'table_differential_summary.csv'
    summary_df.to_csv(csv_file, index=False)
    print(f"  Saved: {csv_file.name}")

    return summary_df


def main():
    """Main function to generate differential analysis visualizations."""

    print("="*80)
    print("DIFFERENTIAL ANALYSIS VISUALIZATIONS")
    print("="*80)
    print()

    # Setup style
    setup_publication_style()

    # Generate per-dataset plots (volcano and scatter)
    print("Creating per-dataset volcano and scatter plots...")
    for dataset in DATASETS:
        print(f"  Processing {DISEASE_NAMES[dataset]}...")

        # Volcano plot
        fig_volcano = create_volcano_plot(dataset)
        if fig_volcano is not None:
            save_figure(fig_volcano, f'04A_volcano_{dataset}', subdir='04_differential_analysis')
            plt.close()

        # Scatter plot
        fig_scatter = create_scatter_plot(dataset)
        if fig_scatter is not None:
            save_figure(fig_scatter, f'04B_scatter_{dataset}', subdir='04_differential_analysis')
            plt.close()

    print()

    # Generate summary figure
    create_network_disruption_summary()
    print()

    # Generate summary table
    summary_df = create_differential_summary_table()
    print()

    # Print summary
    if summary_df is not None:
        print("="*80)
        print("SUMMARY")
        print("="*80)
        totals_row = summary_df.iloc[-1]
        print(f"Total species analyzed: {totals_row['Total']}")
        print(f"Total depleted in disease: {totals_row['Depleted']}")
        print(f"Total enriched in disease: {totals_row['Enriched']}")
        print(f"Total control-specific: {totals_row['Control-specific']}")
        print(f"Total disease-specific: {totals_row['Disease-specific']}")

    print()
    print("Outputs saved to:")
    print(f"  Figures: {FIGURES_DIR}/")
    print(f"  Tables: {TABLES_DIR}/")
    print()
    print("✓ Differential analysis visualizations complete!")


if __name__ == '__main__':
    main()
