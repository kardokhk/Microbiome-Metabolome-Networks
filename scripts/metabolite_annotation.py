"""Metabolite annotation and functional class analysis."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.metabolite_classifier import MetaboliteClassifier
from scripts.utils.config import DATASETS, RESULTS_DIR


class MetaboliteAnnotator:
    """Analyze metabolite functional classes in associations."""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.classifier = MetaboliteClassifier()
        self.datasets = DATASETS

        # Storage
        self.all_associations = defaultdict(lambda: {'control': None, 'disease': None})
        self.class_summaries = []

    def load_associations(self, dataset: str, group: str) -> pd.DataFrame:
        """Load and classify associations for a dataset/group.

        Args:
            dataset: Dataset name
            group: 'control' or 'disease'

        Returns:
            DataFrame with classified associations
        """
        assoc_file = self.results_dir / dataset / group / 'associations.csv'

        if not assoc_file.exists():
            print(f"  Warning: {assoc_file} not found, skipping...")
            return None

        # Load associations
        df = pd.read_csv(assoc_file)

        # Add metabolite classes
        df = self.classifier.classify_dataframe(df, metabolite_col='metabolite')

        # Add dataset and group info
        df['dataset'] = dataset
        df['group'] = group

        print(f"  Loaded {len(df):,} associations from {dataset} {group}")
        return df

    def analyze_dataset(self, dataset: str):
        """Analyze metabolite classes for control vs disease.

        Args:
            dataset: Dataset name
        """
        print(f"\nAnalyzing {dataset}...")

        # Load both groups
        control_assoc = self.load_associations(dataset, 'control')
        disease_assoc = self.load_associations(dataset, 'disease')

        # Store for cross-study analysis
        self.all_associations[dataset]['control'] = control_assoc
        self.all_associations[dataset]['disease'] = disease_assoc

        if control_assoc is None and disease_assoc is None:
            return

        # Analyze each group
        for group, df in [('control', control_assoc), ('disease', disease_assoc)]:
            if df is None:
                continue

            # Class distribution
            class_counts = df['metabolite_class'].value_counts()
            total = len(df)

            # Correlation strength by class
            class_stats = df.groupby('metabolite_class').agg({
                'rho': ['mean', 'std', 'count'],
                'qvalue': 'mean'
            }).round(3)
            class_stats.columns = ['mean_rho', 'std_rho', 'n_associations', 'mean_qvalue']
            class_stats['pct_total'] = (class_stats['n_associations'] / total * 100).round(1)
            class_stats = class_stats.sort_values('n_associations', ascending=False)

            # Store summary
            for class_name, row in class_stats.iterrows():
                self.class_summaries.append({
                    'dataset': dataset,
                    'group': group,
                    'metabolite_class': class_name,
                    'n_associations': row['n_associations'],
                    'pct_total': row['pct_total'],
                    'mean_rho': row['mean_rho'],
                    'std_rho': row['std_rho'],
                    'mean_qvalue': row['mean_qvalue']
                })

            print(f"  {group.capitalize()}: {total:,} associations")
            print(f"    Top classes: {', '.join(class_counts.head(5).index.tolist())}")

    def compare_control_vs_disease(self) -> pd.DataFrame:
        """Compare metabolite class usage between control and disease.

        Returns:
            DataFrame with class-level comparisons
        """
        comparisons = []

        for dataset in self.datasets:
            control_df = self.all_associations[dataset]['control']
            disease_df = self.all_associations[dataset]['disease']

            if control_df is None or disease_df is None:
                continue

            # Get class counts
            control_counts = control_df['metabolite_class'].value_counts().to_dict()
            disease_counts = disease_df['metabolite_class'].value_counts().to_dict()

            # All unique classes
            all_classes = set(control_counts.keys()) | set(disease_counts.keys())

            for class_name in all_classes:
                n_control = control_counts.get(class_name, 0)
                n_disease = disease_counts.get(class_name, 0)

                # Percent of total associations
                pct_control = n_control / len(control_df) * 100 if len(control_df) > 0 else 0
                pct_disease = n_disease / len(disease_df) * 100 if len(disease_df) > 0 else 0

                # Fold change
                if pct_control > 0 and pct_disease > 0:
                    log2_fc = np.log2(pct_disease / pct_control)
                elif pct_disease > 0:
                    log2_fc = 5  # Large positive
                elif pct_control > 0:
                    log2_fc = -5  # Large negative
                else:
                    log2_fc = 0

                # Mean correlation strength
                mean_rho_control = control_df[control_df['metabolite_class'] == class_name]['rho'].mean()
                mean_rho_disease = disease_df[disease_df['metabolite_class'] == class_name]['rho'].mean()

                comparisons.append({
                    'dataset': dataset,
                    'metabolite_class': class_name,
                    'n_control': n_control,
                    'n_disease': n_disease,
                    'pct_control': round(pct_control, 2),
                    'pct_disease': round(pct_disease, 2),
                    'delta_pct': round(pct_disease - pct_control, 2),
                    'log2_fc': round(log2_fc, 2),
                    'mean_rho_control': round(mean_rho_control, 3) if not np.isnan(mean_rho_control) else 0,
                    'mean_rho_disease': round(mean_rho_disease, 3) if not np.isnan(mean_rho_disease) else 0,
                })

        return pd.DataFrame(comparisons)

    def cross_study_summary(self) -> pd.DataFrame:
        """Summarize metabolite class usage across all studies.

        Returns:
            DataFrame with cross-study class statistics
        """
        summary_df = pd.DataFrame(self.class_summaries)

        if len(summary_df) == 0:
            return pd.DataFrame()

        # Aggregate across studies
        cross_study = summary_df.groupby(['metabolite_class', 'group']).agg({
            'n_associations': ['sum', 'mean', 'std'],
            'pct_total': 'mean',
            'mean_rho': 'mean',
            'dataset': 'count'  # How many datasets have this class
        }).round(3)

        cross_study.columns = ['total_associations', 'mean_associations_per_dataset',
                               'std_associations', 'mean_pct_total', 'mean_rho', 'n_datasets']
        cross_study = cross_study.reset_index()
        cross_study = cross_study.sort_values(['group', 'total_associations'], ascending=[True, False])

        return cross_study

    def plot_class_heatmap(self, output_file: Path):
        """Create heatmap of metabolite class usage across datasets.

        Args:
            output_file: Path to save figure
        """
        summary_df = pd.DataFrame(self.class_summaries)

        if len(summary_df) == 0:
            print("No data for heatmap")
            return

        # Pivot for heatmap (control only for clarity)
        control_df = summary_df[summary_df['group'] == 'control']
        pivot = control_df.pivot_table(
            index='metabolite_class',
            columns='dataset',
            values='pct_total',
            fill_value=0
        )

        # Filter to classes present in multiple datasets
        pivot = pivot[pivot.sum(axis=1) > 1].sort_values(by=pivot.columns[0], ascending=False)

        if len(pivot) == 0:
            print("No multi-dataset classes for heatmap")
            return

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': '% of associations'},
                    linewidths=0.5, ax=ax)
        ax.set_title('Metabolite Class Usage Across Datasets (Control Networks)', fontsize=14, weight='bold')
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Metabolite Class', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved heatmap to {output_file}")
        plt.close()

    def plot_control_vs_disease_comparison(self, comparison_df: pd.DataFrame, output_file: Path):
        """Plot control vs disease class comparison.

        Args:
            comparison_df: DataFrame from compare_control_vs_disease()
            output_file: Path to save figure
        """
        if len(comparison_df) == 0:
            print("No data for comparison plot")
            return

        # Average across datasets
        avg_comparison = comparison_df.groupby('metabolite_class').agg({
            'pct_control': 'mean',
            'pct_disease': 'mean',
            'log2_fc': 'mean'
        }).reset_index()

        # Filter to classes with at least 1% in either group
        avg_comparison = avg_comparison[
            (avg_comparison['pct_control'] > 1) | (avg_comparison['pct_disease'] > 1)
        ]
        avg_comparison = avg_comparison.sort_values('log2_fc', ascending=False)

        if len(avg_comparison) == 0:
            print("No significant classes for comparison")
            return

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: Control vs Disease percentages
        x = np.arange(len(avg_comparison))
        width = 0.35
        ax1.bar(x - width/2, avg_comparison['pct_control'], width, label='Control', color='steelblue', alpha=0.8)
        ax1.bar(x + width/2, avg_comparison['pct_disease'], width, label='Disease', color='coral', alpha=0.8)
        ax1.set_xlabel('Metabolite Class', fontsize=11)
        ax1.set_ylabel('% of Associations', fontsize=11)
        ax1.set_title('Metabolite Class Usage\nControl vs Disease', fontsize=12, weight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(avg_comparison['metabolite_class'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Panel 2: Log2 fold change
        colors = ['coral' if fc > 0 else 'steelblue' for fc in avg_comparison['log2_fc']]
        ax2.barh(avg_comparison['metabolite_class'], avg_comparison['log2_fc'], color=colors, alpha=0.8)
        ax2.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax2.set_xlabel('Log2(Disease / Control)', fontsize=11)
        ax2.set_ylabel('')
        ax2.set_title('Metabolite Class Enrichment\nin Disease Networks', fontsize=12, weight='bold')
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved comparison plot to {output_file}")
        plt.close()

    def run_analysis(self):
        """Run complete metabolite annotation analysis."""
        print("=" * 80)
        print("METABOLITE FUNCTIONAL CLASS ANALYSIS")
        print("=" * 80)

        # Analyze each dataset
        for dataset in self.datasets:
            self.analyze_dataset(dataset)

        # Generate summaries
        print("\n" + "=" * 80)
        print("GENERATING SUMMARIES")
        print("=" * 80)

        # Individual class summary
        summary_df = pd.DataFrame(self.class_summaries)
        out_file = self.results_dir / 'cross_study' / 'metabolite_class_associations.csv'
        summary_df.to_csv(out_file, index=False)
        print(f"\nSaved class-level associations: {out_file}")

        # Control vs disease comparison
        comparison_df = self.compare_control_vs_disease()
        out_file = self.results_dir / 'cross_study' / 'metabolite_class_comparison.csv'
        comparison_df.to_csv(out_file, index=False)
        print(f"Saved control vs disease comparison: {out_file}")

        # Cross-study summary
        cross_study_df = self.cross_study_summary()
        out_file = self.results_dir / 'cross_study' / 'metabolite_class_summary.csv'
        cross_study_df.to_csv(out_file, index=False)
        print(f"Saved cross-study summary: {out_file}")

        # Print key findings
        print("\n" + "=" * 80)
        print("KEY FINDINGS")
        print("=" * 80)

        if len(cross_study_df) > 0:
            print("\nTop metabolite classes in CONTROL networks:")
            top_control = cross_study_df[cross_study_df['group'] == 'control'].head(10)
            for _, row in top_control.iterrows():
                print(f"  {row['metabolite_class']:20s}: {row['total_associations']:8,.0f} associations "
                      f"({row['mean_pct_total']:5.1f}% avg), {row['n_datasets']:.0f} datasets")

            print("\nTop metabolite classes in DISEASE networks:")
            top_disease = cross_study_df[cross_study_df['group'] == 'disease'].head(10)
            for _, row in top_disease.iterrows():
                print(f"  {row['metabolite_class']:20s}: {row['total_associations']:8,.0f} associations "
                      f"({row['mean_pct_total']:5.1f}% avg), {row['n_datasets']:.0f} datasets")

        # Generate visualizations
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)

        self.plot_class_heatmap(self.results_dir / 'cross_study' / 'metabolite_class_heatmap.png')
        self.plot_control_vs_disease_comparison(
            comparison_df,
            self.results_dir / 'cross_study' / 'metabolite_class_comparison.png'
        )

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)


if __name__ == "__main__":
    annotator = MetaboliteAnnotator(RESULTS_DIR)
    annotator.run_analysis()
