#!/usr/bin/env python3
"""
Generate study overview and dataset characteristics visualizations.

Outputs:
- Figure 1A: Dataset composition (control vs disease samples)
- Figure 1B: Disease category summary
- Table 1: Dataset characteristics
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
    DISEASE_COLORS, DATASET_COLORS, GROUP_COLORS,
    get_disease_category, get_dataset_color,
    setup_publication_style, save_figure,
    add_panel_label, FIGURES_DIR, TABLES_DIR,
    DATASETS, DISEASE_NAMES, RESULTS_DIR
)


def load_sample_counts():
    """Load sample counts from differential summary files."""
    counts = []

    for dataset in DATASETS:
        summary_file = RESULTS_DIR / dataset / 'differential' / 'summary.csv'

        if summary_file.exists():
            summary = pd.read_csv(summary_file, index_col=0)

            # Extract sample counts
            try:
                n_control = int(summary.loc['Number of samples', 'control'])
                n_disease = int(summary.loc['Number of samples', 'disease'])

                counts.append({
                    'Dataset': dataset,
                    'Disease': DISEASE_NAMES[dataset],
                    'Disease Category': get_disease_category(dataset),
                    'Control': n_control,
                    'Disease_Samples': n_disease,
                    'Total': n_control + n_disease
                })
            except Exception as e:
                print(f"Error loading counts for {dataset}: {e}")

    return pd.DataFrame(counts)


def create_dataset_composition_plot():
    """Create Figure 1A: Dataset composition bar chart."""
    print("Creating Figure 1A: Dataset composition...")

    # Load data
    df = load_sample_counts()

    if df.empty:
        print("  No data available")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for stacked bar chart
    datasets = df['Dataset'].values
    control = df['Control'].values
    disease = df['Disease_Samples'].values

    # Create positions
    y_pos = np.arange(len(datasets))

    # Create bars
    bars1 = ax.barh(y_pos, control, label='Control', color=GROUP_COLORS['control'], alpha=0.8)
    bars2 = ax.barh(y_pos, disease, left=control, label='Disease', color=GROUP_COLORS['disease'], alpha=0.8)

    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Disease'].values, fontsize=10)
    ax.set_xlabel('Number of Samples', fontsize=11)
    ax.set_title('Sample Distribution Across Datasets', fontsize=12, fontweight='bold')

    # Add value labels
    for i, (c, d) in enumerate(zip(control, disease)):
        # Control label
        ax.text(c/2, i, str(c), ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        # Disease label
        ax.text(c + d/2, i, str(d), ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        # Total label
        ax.text(c + d + 20, i, f'n={c+d}', ha='left', va='center', fontsize=9)

    # Legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)

    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save
    save_figure(fig, '01A_dataset_composition', subdir='01_study_overview')
    plt.close()


def create_disease_category_summary():
    """Create Figure 1B: Disease category summary."""
    print("Creating Figure 1B: Disease category summary...")

    # Load data
    df = load_sample_counts()

    if df.empty:
        print("  No data available")
        return

    # Group by disease category
    category_summary = df.groupby('Disease Category').agg({
        'Control': 'sum',
        'Disease_Samples': 'sum',
        'Total': 'sum',
        'Dataset': 'count'
    }).rename(columns={'Dataset': 'N Datasets'})

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Samples by disease category
    categories = category_summary.index.values
    x_pos = np.arange(len(categories))
    width = 0.35

    bars1 = ax1.bar(x_pos - width/2, category_summary['Control'], width,
                    label='Control', color=GROUP_COLORS['control'], alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, category_summary['Disease_Samples'], width,
                    label='Disease', color=GROUP_COLORS['disease'], alpha=0.8)

    ax1.set_xlabel('Disease Category', fontsize=11)
    ax1.set_ylabel('Number of Samples', fontsize=11)
    ax1.set_title('Samples by Disease Category', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend(frameon=True)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)

    # Panel B: Number of datasets per category
    colors = [DISEASE_COLORS.get(cat, '#95a5a6') for cat in categories]

    ax2.bar(x_pos, category_summary['N Datasets'], color=colors, alpha=0.8)
    ax2.set_xlabel('Disease Category', fontsize=11)
    ax2.set_ylabel('Number of Datasets', fontsize=11)
    ax2.set_title('Datasets by Disease Category', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    # Add value labels
    for i, v in enumerate(category_summary['N Datasets']):
        ax2.text(i, v, str(int(v)), ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add panel labels
    add_panel_label(ax1, 'A')
    add_panel_label(ax2, 'B')

    plt.tight_layout()

    # Save
    save_figure(fig, '01B_disease_category_summary', subdir='01_study_overview')
    plt.close()


def create_dataset_characteristics_table():
    """Create Table 1: Dataset characteristics."""
    print("Creating Table 1: Dataset characteristics...")

    # Load data
    df = load_sample_counts()

    if df.empty:
        print("  No data available")
        return

    # Prepare table data
    table_data = []

    for idx, row in df.iterrows():
        table_data.append({
            'Dataset': row['Dataset'],
            'Disease': row['Disease'],
            'Category': row['Disease Category'],
            'Control (n)': row['Control'],
            'Disease (n)': row['Disease_Samples'],
            'Total (n)': row['Total'],
            '% Disease': f"{100 * row['Disease_Samples'] / row['Total']:.1f}%"
        })

    table_df = pd.DataFrame(table_data)

    # Add summary row
    summary_row = {
        'Dataset': 'TOTAL',
        'Disease': 'All diseases',
        'Category': '5 categories',
        'Control (n)': table_df['Control (n)'].sum(),
        'Disease (n)': table_df['Disease (n)'].sum(),
        'Total (n)': table_df['Total (n)'].sum(),
        '% Disease': f"{100 * table_df['Disease (n)'].sum() / table_df['Total (n)'].sum():.1f}%"
    }

    table_df = pd.concat([table_df, pd.DataFrame([summary_row])], ignore_index=True)

    # Save as CSV
    csv_file = TABLES_DIR / 'table1_dataset_characteristics.csv'
    table_df.to_csv(csv_file, index=False)
    print(f"  Saved: {csv_file.name}")

    # Create LaTeX table
    latex_file = TABLES_DIR / 'table1_dataset_characteristics.tex'

    with open(latex_file, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Dataset Characteristics and Sample Distribution}\n")
        f.write("\\label{tab:dataset_characteristics}\n")
        f.write("\\begin{tabular}{llrrrrc}\n")
        f.write("\\hline\n")
        f.write("Dataset & Disease & Category & Control & Disease & Total & \\% Disease \\\\\n")
        f.write("\\hline\n")

        for idx, row in table_df.iterrows():
            if idx == len(table_df) - 1:
                f.write("\\hline\n")
                f.write(f"\\textbf{{{row['Dataset']}}} & ")
            else:
                f.write(f"{row['Dataset']} & ")

            f.write(f"{row['Disease']} & ")
            f.write(f"{row['Category']} & ")
            f.write(f"{row['Control (n)']} & ")
            f.write(f"{row['Disease (n)']} & ")
            f.write(f"{row['Total (n)']} & ")
            f.write(f"{row['% Disease']} \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"  Saved: {latex_file.name}")

    return table_df


def main():
    """Main function to generate all overview visualizations."""

    print("="*80)
    print("STUDY OVERVIEW VISUALIZATIONS")
    print("="*80)
    print()

    # Setup style
    setup_publication_style()

    # Generate figures
    create_dataset_composition_plot()
    print()

    create_disease_category_summary()
    print()

    # Generate table
    table_df = create_dataset_characteristics_table()
    print()

    # Print summary
    if table_df is not None:
        print("="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total datasets: {len(DATASETS)}")
        print(f"Total samples: {table_df.iloc[-1]['Total (n)']}")
        print(f"Control samples: {table_df.iloc[-1]['Control (n)']}")
        print(f"Disease samples: {table_df.iloc[-1]['Disease (n)']}")
        print(f"Disease categories: 5 (IBD, CRC, Gastric, IBS, ESRD)")
        print()
        print("Outputs saved to:")
        print(f"  Figures: {FIGURES_DIR}/")
        print(f"  Tables: {TABLES_DIR}/")

    print()
    print("✓ Study overview visualizations complete!")


if __name__ == '__main__':
    main()
