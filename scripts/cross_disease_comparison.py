"""Cross-disease comparison - identify universal vs disease-specific dysbiosis markers."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.config import DATASETS, RESULTS_DIR
from scripts.utils.disease_groups import get_disease_groups


class CrossDiseaseComparator:
    """Compare microbiome-metabolome signatures across disease contexts."""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.datasets = DATASETS

        # Disease category mappings
        self.disease_categories = {
            'IBD': ['FRANZOSA_IBD_2019', 'iHMP_IBDMDB_2019'],
            'CRC': ['YACHIDA_CRC_2019'],
            'Gastric_Cancer': ['ERAWIJANTARI_GASTRIC_CANCER_2020'],
            'IBS': ['MARS_IBS_2020'],
            'Metabolic': ['WANG_ESRD_2020']
        }

        # Storage
        self.all_differential_keystones = {}
        self.all_network_comparisons = {}

    def load_differential_data(self):
        """Load differential keystone data from all datasets."""
        print(f"Loading differential analysis results from {len(self.datasets)} datasets...")

        for dataset in self.datasets:
            diff_file = self.results_dir / dataset / 'differential' / 'differential_keystones.csv'

            if not diff_file.exists():
                print(f"  Warning: {diff_file} not found")
                continue

            diff_df = pd.read_csv(diff_file)
            self.all_differential_keystones[dataset] = diff_df

            # Load network comparison if available
            net_comp_file = self.results_dir / dataset / 'differential' / 'network_comparison.csv'
            if net_comp_file.exists():
                net_comp = pd.read_csv(net_comp_file)
                self.all_network_comparisons[dataset] = net_comp

        print(f"  Loaded data from {len(self.all_differential_keystones)} datasets")

    def get_dataset_disease_category(self, dataset: str) -> str:
        """Get disease category for a dataset.

        Args:
            dataset: Dataset name

        Returns:
            Disease category name
        """
        for category, datasets in self.disease_categories.items():
            if dataset in datasets:
                return category
        return 'Unknown'

    def stratify_by_disease_category(self) -> pd.DataFrame:
        """Classify species as universal/common/specific based on disease categories.

        Returns:
            DataFrame with species classification
        """
        print(f"\n{'='*80}")
        print(f"STRATIFYING SPECIES BY DISEASE CATEGORY")
        print(f"{'='*80}")

        # Collect all depleted species by disease category
        depleted_by_category = defaultdict(lambda: defaultdict(list))

        for dataset, diff_df in self.all_differential_keystones.items():
            category = self.get_dataset_disease_category(dataset)

            # Get depleted species
            depleted = diff_df[diff_df['status'] == 'depleted_in_disease']

            for species in depleted['species']:
                depleted_by_category[species][category].append(dataset)

        # Build classification table
        stratification = []

        for species, categories in depleted_by_category.items():
            n_categories = len(categories)
            n_datasets = sum(len(datasets) for datasets in categories.values())

            # Classify
            if n_categories == len(self.disease_categories):
                classification = 'universal'
            elif n_categories >= 3:
                classification = 'common'
            elif n_categories == 1:
                classification = 'disease_specific'
            else:
                classification = 'moderate'

            stratification.append({
                'species': species,
                'n_disease_categories': n_categories,
                'disease_categories': ';'.join(sorted(categories.keys())),
                'n_datasets': n_datasets,
                'datasets': ';'.join([d for cat in categories.values() for d in cat]),
                'classification': classification
            })

        stratification_df = pd.DataFrame(stratification)
        stratification_df = stratification_df.sort_values(
            ['n_disease_categories', 'n_datasets'], ascending=[False, False]
        )

        # Save
        out_dir = self.results_dir / 'cross_disease'
        out_dir.mkdir(exist_ok=True, parents=True)
        out_file = out_dir / 'disease_stratification.csv'
        stratification_df.to_csv(out_file, index=False)

        print(f"\nSaved stratification: {out_file}")
        print(f"\nClassification summary:")
        class_counts = stratification_df['classification'].value_counts()
        for classification, count in class_counts.items():
            print(f"  {classification:20s}: {count} species")

        return stratification_df

    def identify_universal_markers(self, stratification_df: pd.DataFrame):
        """Identify species depleted across all disease categories.

        Args:
            stratification_df: Output from stratify_by_disease_category
        """
        print(f"\n{'='*80}")
        print(f"IDENTIFYING UNIVERSAL DYSBIOSIS MARKERS")
        print(f"{'='*80}")

        # Get universal markers (present in all categories)
        universal = stratification_df[
            stratification_df['classification'] == 'universal'
        ].copy()

        if len(universal) == 0:
            print("  No universal markers found (present in all disease categories)")
            return

        # Calculate mean log2FC across datasets
        universal_with_effect = []

        for _, row in universal.iterrows():
            species = row['species']
            datasets = row['datasets'].split(';')

            log2fc_values = []
            for dataset in datasets:
                if dataset in self.all_differential_keystones:
                    diff_df = self.all_differential_keystones[dataset]
                    species_row = diff_df[diff_df['species'] == species]
                    if len(species_row) > 0:
                        log2fc_values.append(species_row.iloc[0]['log2_fc'])

            if len(log2fc_values) > 0:
                universal_with_effect.append({
                    'species': species,
                    'n_disease_categories': row['n_disease_categories'],
                    'disease_categories': row['disease_categories'],
                    'n_datasets': row['n_datasets'],
                    'datasets': row['datasets'],
                    'mean_log2fc': np.mean(log2fc_values),
                    'std_log2fc': np.std(log2fc_values),
                    'min_log2fc': np.min(log2fc_values),
                    'max_log2fc': np.max(log2fc_values)
                })

        universal_df = pd.DataFrame(universal_with_effect)
        if len(universal_df) > 0:
            universal_df = universal_df.sort_values('mean_log2fc')
            universal_df = universal_df.round(3)

            # Save
            out_dir = self.results_dir / 'cross_disease'
            out_file = out_dir / 'universal_dysbiosis_markers.csv'
            universal_df.to_csv(out_file, index=False)

            print(f"\nSaved universal markers: {out_file}")
            print(f"Found {len(universal_df)} species depleted across all disease categories")
            print(f"\nTop 10 universal dysbiosis markers (most depleted):")
            for _, row in universal_df.head(10).iterrows():
                print(f"  {row['species'][:80]:80s} mean log2FC: {row['mean_log2fc']:+.2f}")

    def identify_disease_specific_markers(self, stratification_df: pd.DataFrame):
        """Identify species depleted in only one disease category.

        Args:
            stratification_df: Output from stratify_by_disease_category
        """
        print(f"\n{'='*80}")
        print(f"IDENTIFYING DISEASE-SPECIFIC MARKERS")
        print(f"{'='*80}")

        # Get disease-specific markers
        specific = stratification_df[
            stratification_df['classification'] == 'disease_specific'
        ].copy()

        if len(specific) == 0:
            print("  No disease-specific markers found")
            return

        # Add log2FC
        specific_with_effect = []

        for _, row in specific.iterrows():
            species = row['species']
            category = row['disease_categories']
            datasets = row['datasets'].split(';')

            log2fc_values = []
            for dataset in datasets:
                if dataset in self.all_differential_keystones:
                    diff_df = self.all_differential_keystones[dataset]
                    species_row = diff_df[diff_df['species'] == species]
                    if len(species_row) > 0:
                        log2fc_values.append(species_row.iloc[0]['log2_fc'])

            if len(log2fc_values) > 0:
                specific_with_effect.append({
                    'species': species,
                    'disease_category': category,
                    'n_datasets': row['n_datasets'],
                    'datasets': row['datasets'],
                    'mean_log2fc': np.mean(log2fc_values),
                    'std_log2fc': np.std(log2fc_values) if len(log2fc_values) > 1 else 0
                })

        specific_df = pd.DataFrame(specific_with_effect)
        if len(specific_df) > 0:
            specific_df = specific_df.sort_values(['disease_category', 'mean_log2fc'])
            specific_df = specific_df.round(3)

            # Save
            out_dir = self.results_dir / 'cross_disease'
            out_file = out_dir / 'disease_specific_markers.csv'
            specific_df.to_csv(out_file, index=False)

            print(f"\nSaved disease-specific markers: {out_file}")
            print(f"Found {len(specific_df)} disease-specific species")
            print(f"\nDisease-specific marker counts:")
            category_counts = specific_df['disease_category'].value_counts()
            for category, count in category_counts.items():
                print(f"  {category:20s}: {count} species")

    def meta_analysis_effect_sizes(self, stratification_df: pd.DataFrame):
        """Perform meta-analysis of effect sizes for depleted species.

        Args:
            stratification_df: Output from stratify_by_disease_category
        """
        print(f"\n{'='*80}")
        print(f"META-ANALYSIS OF EFFECT SIZES")
        print(f"{'='*80}")

        # For species present in ≥2 datasets, compute pooled effect size
        multi_study = stratification_df[stratification_df['n_datasets'] >= 2].copy()

        if len(multi_study) == 0:
            print("  No species found in ≥2 studies")
            return

        meta_results = []

        for _, row in multi_study.iterrows():
            species = row['species']
            datasets = row['datasets'].split(';')

            log2fc_values = []
            categories_list = []

            for dataset in datasets:
                if dataset in self.all_differential_keystones:
                    diff_df = self.all_differential_keystones[dataset]
                    species_row = diff_df[diff_df['species'] == species]
                    if len(species_row) > 0:
                        log2fc_values.append(species_row.iloc[0]['log2_fc'])
                        categories_list.append(self.get_dataset_disease_category(dataset))

            if len(log2fc_values) >= 2:
                # Simple fixed-effects meta-analysis (inverse variance weighting)
                # For now, just use mean (equal weights) since we don't have SEs
                pooled_log2fc = np.mean(log2fc_values)
                pooled_se = np.std(log2fc_values) / np.sqrt(len(log2fc_values))

                # Heterogeneity (I²): proportion of variance due to between-study differences
                variance = np.var(log2fc_values, ddof=1)
                within_study_variance = pooled_se ** 2
                between_study_variance = max(0, variance - within_study_variance)
                I2 = (between_study_variance / variance) * 100 if variance > 0 else 0

                meta_results.append({
                    'species': species,
                    'n_studies': len(log2fc_values),
                    'n_disease_categories': row['n_disease_categories'],
                    'disease_categories': ';'.join(sorted(set(categories_list))),
                    'pooled_log2fc': pooled_log2fc,
                    'pooled_se': pooled_se,
                    'heterogeneity_I2': I2,
                    'min_log2fc': np.min(log2fc_values),
                    'max_log2fc': np.max(log2fc_values)
                })

        if len(meta_results) == 0:
            print("  No species with sufficient data for meta-analysis")
            return

        meta_df = pd.DataFrame(meta_results)
        meta_df = meta_df.sort_values('pooled_log2fc')
        meta_df = meta_df.round(3)

        # Save
        out_dir = self.results_dir / 'cross_disease'
        out_file = out_dir / 'meta_analysis_effect_sizes.csv'
        meta_df.to_csv(out_file, index=False)

        print(f"\nSaved meta-analysis results: {out_file}")
        print(f"Analyzed {len(meta_df)} species (present in ≥2 studies)")
        print(f"\nTop 10 species by pooled effect size (most depleted):")
        for _, row in meta_df.head(10).iterrows():
            print(f"  {row['species'][:70]:70s} | "
                  f"log2FC: {row['pooled_log2fc']:+.2f} ± {row['pooled_se']:.2f} | "
                  f"I²: {row['heterogeneity_I2']:.0f}%")

    def compare_network_disruption(self):
        """Compare network-level changes across disease categories."""
        print(f"\n{'='*80}")
        print(f"NETWORK DISRUPTION BY DISEASE CATEGORY")
        print(f"{'='*80}")

        if len(self.all_network_comparisons) == 0:
            print("  No network comparison data available")
            return

        # Aggregate by disease category
        category_metrics = defaultdict(list)

        for dataset, net_comp in self.all_network_comparisons.items():
            category = self.get_dataset_disease_category(dataset)

            if len(net_comp) > 0:
                row = net_comp.iloc[0]
                category_metrics[category].append({
                    'dataset': dataset,
                    'modularity_control': row.get('modularity_control', np.nan),
                    'modularity_disease': row.get('modularity_disease', np.nan),
                    'density_control': row.get('density_control', np.nan),
                    'density_disease': row.get('density_disease', np.nan),
                    'n_nodes_control': row.get('n_nodes_control', np.nan),
                    'n_nodes_disease': row.get('n_nodes_disease', np.nan)
                })

        # Calculate mean changes per category
        disruption_summary = []

        for category, metrics_list in category_metrics.items():
            if len(metrics_list) == 0:
                continue

            # Extract metrics
            mod_control = [m['modularity_control'] for m in metrics_list if not np.isnan(m['modularity_control'])]
            mod_disease = [m['modularity_disease'] for m in metrics_list if not np.isnan(m['modularity_disease'])]
            dens_control = [m['density_control'] for m in metrics_list if not np.isnan(m['density_control'])]
            dens_disease = [m['density_disease'] for m in metrics_list if not np.isnan(m['density_disease'])]
            nodes_control = [m['n_nodes_control'] for m in metrics_list if not np.isnan(m['n_nodes_control'])]
            nodes_disease = [m['n_nodes_disease'] for m in metrics_list if not np.isnan(m['n_nodes_disease'])]

            disruption_summary.append({
                'disease_category': category,
                'n_datasets': len(metrics_list),
                'mean_modularity_control': np.mean(mod_control) if mod_control else np.nan,
                'mean_modularity_disease': np.mean(mod_disease) if mod_disease else np.nan,
                'modularity_change': np.mean(mod_disease) - np.mean(mod_control) if (mod_control and mod_disease) else np.nan,
                'mean_density_control': np.mean(dens_control) if dens_control else np.nan,
                'mean_density_disease': np.mean(dens_disease) if dens_disease else np.nan,
                'density_change': np.mean(dens_disease) - np.mean(dens_control) if (dens_control and dens_disease) else np.nan,
                'mean_nodes_control': np.mean(nodes_control) if nodes_control else np.nan,
                'mean_nodes_disease': np.mean(nodes_disease) if nodes_disease else np.nan,
                'node_loss': np.mean(nodes_control) - np.mean(nodes_disease) if (nodes_control and nodes_disease) else np.nan
            })

        if len(disruption_summary) == 0:
            print("  Insufficient data for network disruption analysis")
            return

        disruption_df = pd.DataFrame(disruption_summary)
        disruption_df = disruption_df.round(4)

        # Save
        out_dir = self.results_dir / 'cross_disease'
        out_file = out_dir / 'network_disruption_by_disease.csv'
        disruption_df.to_csv(out_file, index=False)

        print(f"\nSaved network disruption analysis: {out_file}")
        print(f"\nNetwork disruption patterns:")
        for _, row in disruption_df.iterrows():
            print(f"\n  {row['disease_category']}:")
            print(f"    Modularity change: {row['modularity_change']:+.4f}")
            print(f"    Density change: {row['density_change']:+.4f}")
            print(f"    Node loss: {row['node_loss']:.0f} nodes")

    def compute_disease_similarity(self):
        """Compute Jaccard similarity between disease signatures."""
        print(f"\n{'='*80}")
        print(f"DISEASE SIGNATURE SIMILARITY")
        print(f"{'='*80}")

        # Get depleted species per disease category
        depleted_by_category = defaultdict(set)

        for dataset, diff_df in self.all_differential_keystones.items():
            category = self.get_dataset_disease_category(dataset)
            depleted = diff_df[diff_df['status'] == 'depleted_in_disease']['species']
            depleted_by_category[category].update(depleted)

        categories = sorted(depleted_by_category.keys())

        # Compute pairwise Jaccard similarity
        similarity_matrix = []

        for cat1 in categories:
            row = {'disease_category': cat1}
            set1 = depleted_by_category[cat1]

            for cat2 in categories:
                set2 = depleted_by_category[cat2]

                # Jaccard index
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                jaccard = intersection / union if union > 0 else 0

                row[cat2] = round(jaccard, 3)

            similarity_matrix.append(row)

        similarity_df = pd.DataFrame(similarity_matrix)

        # Save
        out_dir = self.results_dir / 'cross_disease'
        out_file = out_dir / 'disease_similarity_matrix.csv'
        similarity_df.to_csv(out_file, index=False)

        print(f"\nSaved similarity matrix: {out_file}")
        print(f"\nJaccard similarity between disease signatures:")
        print(similarity_df.to_string(index=False))

    def run_analysis(self):
        """Run complete cross-disease comparison pipeline."""
        print(f"\n{'='*80}")
        print(f"CROSS-DISEASE COMPARISON ANALYSIS")
        print(f"{'='*80}")
        print(f"Datasets: {len(self.datasets)}")
        print(f"Disease categories: {len(self.disease_categories)}")
        for category, datasets in self.disease_categories.items():
            print(f"  {category:20s}: {', '.join(datasets)}")

        # Load data
        self.load_differential_data()

        # Disease stratification
        stratification_df = self.stratify_by_disease_category()

        # Identify universal markers
        self.identify_universal_markers(stratification_df)

        # Identify disease-specific markers
        self.identify_disease_specific_markers(stratification_df)

        # Meta-analysis
        self.meta_analysis_effect_sizes(stratification_df)

        # Network disruption patterns
        self.compare_network_disruption()

        # Disease similarity
        self.compute_disease_similarity()

        print(f"\n{'='*80}")
        print(f"CROSS-DISEASE COMPARISON COMPLETE!")
        print(f"{'='*80}")
        print(f"\nKey outputs in: {self.results_dir / 'cross_disease'}/")
        print(f"  - disease_stratification.csv")
        print(f"  - universal_dysbiosis_markers.csv")
        print(f"  - disease_specific_markers.csv")
        print(f"  - meta_analysis_effect_sizes.csv")
        print(f"  - network_disruption_by_disease.csv")
        print(f"  - disease_similarity_matrix.csv")


if __name__ == "__main__":
    comparator = CrossDiseaseComparator(RESULTS_DIR)
    comparator.run_analysis()
