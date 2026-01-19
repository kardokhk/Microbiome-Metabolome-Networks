"""Module enrichment analysis - test for over-representation of metabolite classes and taxa."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.config import DATASETS, RESULTS_DIR
from scripts.metabolite_classifier import MetaboliteClassifier


class ModuleEnrichmentAnalyzer:
    """Test for enrichment of metabolite classes and taxonomic groups in modules."""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.datasets = DATASETS
        self.classifier = MetaboliteClassifier()

        # Storage for cross-study analysis
        self.all_enrichments = {'control': [], 'disease': []}

    def extract_taxonomy(self, species_name: str, level: str = 'phylum') -> str:
        """Extract taxonomic information from species name.

        Args:
            species_name: Full taxonomic string (e.g., 'd  bacteria;p  firmicutes;...')
            level: Taxonomic level ('phylum', 'class', 'family')

        Returns:
            Taxon name at specified level, or 'Unknown'
        """
        if not species_name.startswith('d  bacteria'):
            return 'Unknown'

        try:
            parts = species_name.split(';')

            # Map levels to prefixes
            level_map = {
                'phylum': 'p  ',
                'class': 'c  ',
                'family': 'f  '
            }

            prefix = level_map.get(level, 'p  ')

            for part in parts:
                if part.strip().startswith(prefix):
                    return part.strip()[len(prefix):].strip()

            return 'Unknown'

        except Exception:
            return 'Unknown'

    def classify_nodes(self, modules_df: pd.DataFrame) -> pd.DataFrame:
        """Classify all nodes (metabolites and species) in modules.

        Args:
            modules_df: DataFrame with columns [node, module, node_type]

        Returns:
            DataFrame with added classification columns
        """
        modules_df = modules_df.copy()

        # Classify metabolites
        metabolite_mask = modules_df['node_type'] == 'metabolite'
        modules_df.loc[metabolite_mask, 'metabolite_class'] = modules_df.loc[
            metabolite_mask, 'node'
        ].apply(lambda x: self.classifier.classify_metabolite(x)[0])

        # Classify species by taxonomy
        species_mask = modules_df['node_type'] == 'species'
        modules_df.loc[species_mask, 'phylum'] = modules_df.loc[
            species_mask, 'node'
        ].apply(lambda x: self.extract_taxonomy(x, 'phylum'))

        modules_df.loc[species_mask, 'family'] = modules_df.loc[
            species_mask, 'node'
        ].apply(lambda x: self.extract_taxonomy(x, 'family'))

        return modules_df

    def test_metabolite_class_enrichment(self, modules_df: pd.DataFrame) -> pd.DataFrame:
        """Test for over-representation of metabolite classes in modules.

        Args:
            modules_df: DataFrame with module assignments and metabolite_class

        Returns:
            DataFrame with enrichment results
        """
        # Filter to metabolites only
        metabolites = modules_df[modules_df['node_type'] == 'metabolite'].copy()

        if len(metabolites) == 0:
            return pd.DataFrame()

        # Get module list (exclude Unknown class)
        metabolites = metabolites[metabolites['metabolite_class'] != 'Unknown']

        if len(metabolites) == 0:
            return pd.DataFrame()

        modules = metabolites['module'].unique()
        classes = metabolites['metabolite_class'].unique()

        total_metabolites = len(metabolites)

        enrichment_results = []

        for module_id in modules:
            module_metabolites = metabolites[metabolites['module'] == module_id]
            n_in_module = len(module_metabolites)

            for met_class in classes:
                # Contingency table:
                #              In_Module  Not_In_Module
                # In_Class         a           b
                # Not_Class        c           d

                a = len(module_metabolites[module_metabolites['metabolite_class'] == met_class])
                b = len(metabolites[metabolites['metabolite_class'] == met_class]) - a
                c = n_in_module - a
                d = total_metabolites - n_in_module - b

                # Skip if no metabolites of this class in module
                if a == 0:
                    continue

                # Fisher's exact test
                try:
                    odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')
                except Exception:
                    odds_ratio, p_value = np.nan, 1.0

                # Calculate expected count
                expected = (a + b) * (a + c) / total_metabolites

                enrichment_results.append({
                    'module': int(module_id),
                    'metabolite_class': met_class,
                    'n_in_module': int(a),
                    'n_total_in_class': int(a + b),
                    'module_size': int(n_in_module),
                    'expected': round(expected, 2),
                    'fold_enrichment': round(a / expected, 3) if expected > 0 else np.nan,
                    'odds_ratio': round(odds_ratio, 3) if not np.isnan(odds_ratio) else np.nan,
                    'p_value': p_value
                })

        if len(enrichment_results) == 0:
            return pd.DataFrame()

        enrichment_df = pd.DataFrame(enrichment_results)

        # FDR correction
        enrichment_df['q_value'] = multipletests(
            enrichment_df['p_value'], method='fdr_bh'
        )[1]

        # Mark as enriched if q < 0.05 and odds_ratio > 2
        enrichment_df['enriched'] = (
            (enrichment_df['q_value'] < 0.05) &
            (enrichment_df['odds_ratio'] > 2.0)
        )

        # Sort by q-value
        enrichment_df = enrichment_df.sort_values('q_value')

        return enrichment_df

    def test_taxonomic_enrichment(self, modules_df: pd.DataFrame,
                                   taxon_level: str = 'phylum') -> pd.DataFrame:
        """Test for over-representation of taxonomic groups in modules.

        Args:
            modules_df: DataFrame with module assignments and taxonomy
            taxon_level: 'phylum' or 'family'

        Returns:
            DataFrame with enrichment results
        """
        # Filter to species only
        species = modules_df[modules_df['node_type'] == 'species'].copy()

        if len(species) == 0 or taxon_level not in species.columns:
            return pd.DataFrame()

        # Remove Unknown
        species = species[species[taxon_level] != 'Unknown']

        if len(species) == 0:
            return pd.DataFrame()

        modules = species['module'].unique()
        taxa = species[taxon_level].unique()

        total_species = len(species)

        enrichment_results = []

        for module_id in modules:
            module_species = species[species['module'] == module_id]
            n_in_module = len(module_species)

            for taxon in taxa:
                # Contingency table
                a = len(module_species[module_species[taxon_level] == taxon])
                b = len(species[species[taxon_level] == taxon]) - a
                c = n_in_module - a
                d = total_species - n_in_module - b

                # Skip if no species of this taxon in module
                if a == 0:
                    continue

                # Fisher's exact test
                try:
                    odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')
                except Exception:
                    odds_ratio, p_value = np.nan, 1.0

                # Calculate expected count
                expected = (a + b) * (a + c) / total_species

                enrichment_results.append({
                    'module': int(module_id),
                    'taxon_level': taxon_level,
                    'taxon': taxon,
                    'n_in_module': int(a),
                    'n_total_in_taxon': int(a + b),
                    'module_size': int(n_in_module),
                    'expected': round(expected, 2),
                    'fold_enrichment': round(a / expected, 3) if expected > 0 else np.nan,
                    'odds_ratio': round(odds_ratio, 3) if not np.isnan(odds_ratio) else np.nan,
                    'p_value': p_value
                })

        if len(enrichment_results) == 0:
            return pd.DataFrame()

        enrichment_df = pd.DataFrame(enrichment_results)

        # FDR correction
        enrichment_df['q_value'] = multipletests(
            enrichment_df['p_value'], method='fdr_bh'
        )[1]

        # Mark as enriched if q < 0.05 and odds_ratio > 2
        enrichment_df['enriched'] = (
            (enrichment_df['q_value'] < 0.05) &
            (enrichment_df['odds_ratio'] > 2.0)
        )

        # Sort by q-value
        enrichment_df = enrichment_df.sort_values('q_value')

        return enrichment_df

    def analyze_dataset(self, dataset: str):
        """Perform enrichment analysis for one dataset (control and disease).

        Args:
            dataset: Dataset name
        """
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset}")
        print(f"{'='*80}")

        for group in ['control', 'disease']:
            print(f"\n  Analyzing {group.upper()} group...")

            # Load modules
            modules_file = self.results_dir / dataset / group / 'modules.csv'

            if not modules_file.exists():
                print(f"    Warning: {modules_file} not found, skipping")
                continue

            modules_df = pd.read_csv(modules_file)
            print(f"    Loaded {len(modules_df)} nodes in {modules_df['module'].nunique()} modules")

            # Classify nodes
            print(f"    Classifying nodes...")
            modules_df = self.classify_nodes(modules_df)

            # Create output directory
            out_dir = self.results_dir / dataset / group / 'enrichment'
            out_dir.mkdir(exist_ok=True, parents=True)

            # Test metabolite class enrichment
            print(f"    Testing metabolite class enrichment...")
            met_enrichment = self.test_metabolite_class_enrichment(modules_df)

            if len(met_enrichment) > 0:
                out_file = out_dir / 'metabolite_class_enrichment.csv'
                met_enrichment.to_csv(out_file, index=False)
                print(f"      Saved {len(met_enrichment)} tests ({met_enrichment['enriched'].sum()} significant)")

                # Store for cross-study
                met_enrichment['dataset'] = dataset
                met_enrichment['group'] = group
                self.all_enrichments[group].append(met_enrichment[met_enrichment['enriched']])

                # Print top enrichments
                top = met_enrichment[met_enrichment['enriched']].head(5)
                if len(top) > 0:
                    print(f"      Top enrichments:")
                    for _, row in top.iterrows():
                        print(f"        Module {row['module']}: {row['metabolite_class']} "
                              f"(OR={row['odds_ratio']:.1f}, q={row['q_value']:.2e})")
            else:
                print(f"      No metabolite class enrichments found")

            # Test taxonomic enrichment (phylum level)
            print(f"    Testing taxonomic enrichment (phylum)...")
            tax_enrichment = self.test_taxonomic_enrichment(modules_df, 'phylum')

            if len(tax_enrichment) > 0:
                out_file = out_dir / 'taxonomic_enrichment_phylum.csv'
                tax_enrichment.to_csv(out_file, index=False)
                print(f"      Saved {len(tax_enrichment)} tests ({tax_enrichment['enriched'].sum()} significant)")

                # Print top enrichments
                top = tax_enrichment[tax_enrichment['enriched']].head(5)
                if len(top) > 0:
                    print(f"      Top enrichments:")
                    for _, row in top.iterrows():
                        print(f"        Module {row['module']}: {row['taxon']} "
                              f"(OR={row['odds_ratio']:.1f}, q={row['q_value']:.2e})")
            else:
                print(f"      No taxonomic enrichments found")

            # Test taxonomic enrichment (family level)
            print(f"    Testing taxonomic enrichment (family)...")
            fam_enrichment = self.test_taxonomic_enrichment(modules_df, 'family')

            if len(fam_enrichment) > 0:
                out_file = out_dir / 'taxonomic_enrichment_family.csv'
                fam_enrichment.to_csv(out_file, index=False)
                print(f"      Saved {len(fam_enrichment)} tests ({fam_enrichment['enriched'].sum()} significant)")

    def cross_study_enrichment_analysis(self):
        """Identify metabolite classes consistently enriched across studies."""
        print(f"\n{'='*80}")
        print(f"CROSS-STUDY ENRICHMENT ANALYSIS")
        print(f"{'='*80}")

        for group in ['control', 'disease']:
            print(f"\n  Analyzing {group.upper()} enrichments...")

            if len(self.all_enrichments[group]) == 0:
                print(f"    No enrichments found")
                continue

            # Combine all significant enrichments
            combined = pd.concat(self.all_enrichments[group], ignore_index=True)

            # Count how many datasets each metabolite class is enriched in
            class_counts = combined.groupby('metabolite_class').agg({
                'dataset': lambda x: len(set(x)),  # n_datasets
                'module': 'count',  # n_module_enrichments
                'odds_ratio': 'mean',
                'fold_enrichment': 'mean'
            }).reset_index()

            class_counts.columns = [
                'metabolite_class', 'n_datasets', 'n_enrichments',
                'mean_odds_ratio', 'mean_fold_enrichment'
            ]

            class_counts = class_counts.sort_values('n_datasets', ascending=False)

            # Get dataset lists
            class_datasets = combined.groupby('metabolite_class')['dataset'].apply(
                lambda x: ';'.join(sorted(set(x)))
            ).reset_index()
            class_datasets.columns = ['metabolite_class', 'datasets']

            # Merge
            class_counts = class_counts.merge(class_datasets, on='metabolite_class')

            # Round
            class_counts['mean_odds_ratio'] = class_counts['mean_odds_ratio'].round(2)
            class_counts['mean_fold_enrichment'] = class_counts['mean_fold_enrichment'].round(2)

            # Save
            out_dir = self.results_dir / 'cross_study' / 'enrichment'
            out_dir.mkdir(exist_ok=True, parents=True)
            out_file = out_dir / f'conserved_enrichments_{group}.csv'
            class_counts.to_csv(out_file, index=False)

            print(f"    Saved conserved enrichments: {out_file}")
            print(f"\n    Top conserved enriched classes in {group}:")
            for _, row in class_counts.head(10).iterrows():
                print(f"      {row['metabolite_class']:20s}: {row['n_datasets']} datasets, "
                      f"{row['n_enrichments']} enrichments, OR={row['mean_odds_ratio']:.1f}")

    def compare_control_vs_disease_enrichments(self):
        """Compare which metabolite classes are enriched in control vs disease."""
        print(f"\n{'='*80}")
        print(f"CONTROL vs DISEASE ENRICHMENT COMPARISON")
        print(f"{'='*80}")

        if len(self.all_enrichments['control']) == 0 or len(self.all_enrichments['disease']) == 0:
            print("  Insufficient data for comparison")
            return

        # Combine enrichments
        control_combined = pd.concat(self.all_enrichments['control'], ignore_index=True)
        disease_combined = pd.concat(self.all_enrichments['disease'], ignore_index=True)

        # Get class counts per group
        control_classes = control_combined.groupby('metabolite_class').agg({
            'dataset': lambda x: len(set(x))
        }).reset_index()
        control_classes.columns = ['metabolite_class', 'n_datasets_control']

        disease_classes = disease_combined.groupby('metabolite_class').agg({
            'dataset': lambda x: len(set(x))
        }).reset_index()
        disease_classes.columns = ['metabolite_class', 'n_datasets_disease']

        # Merge
        comparison = control_classes.merge(
            disease_classes, on='metabolite_class', how='outer'
        ).fillna(0)

        comparison['n_datasets_control'] = comparison['n_datasets_control'].astype(int)
        comparison['n_datasets_disease'] = comparison['n_datasets_disease'].astype(int)

        # Calculate difference
        comparison['delta'] = comparison['n_datasets_disease'] - comparison['n_datasets_control']

        # Classify
        comparison['enrichment_pattern'] = 'unchanged'
        comparison.loc[comparison['delta'] >= 2, 'enrichment_pattern'] = 'enriched_in_disease'
        comparison.loc[comparison['delta'] <= -2, 'enrichment_pattern'] = 'enriched_in_control'

        # Sort by absolute delta
        comparison['abs_delta'] = comparison['delta'].abs()
        comparison = comparison.sort_values('abs_delta', ascending=False)
        comparison = comparison.drop('abs_delta', axis=1)

        # Save
        out_dir = self.results_dir / 'cross_study' / 'enrichment'
        out_dir.mkdir(exist_ok=True, parents=True)
        out_file = out_dir / 'control_vs_disease_enrichment_comparison.csv'
        comparison.to_csv(out_file, index=False)

        print(f"  Saved comparison: {out_file}")
        print(f"\n  Classes with differential enrichment patterns:")
        diff = comparison[comparison['enrichment_pattern'] != 'unchanged']
        if len(diff) > 0:
            for _, row in diff.iterrows():
                print(f"    {row['metabolite_class']:20s}: {row['enrichment_pattern']:25s} "
                      f"(Δ={row['delta']:+d})")
        else:
            print("    No significant differences found")

    def run_analysis(self):
        """Run complete enrichment analysis pipeline."""
        print(f"\n{'='*80}")
        print(f"MODULE ENRICHMENT ANALYSIS")
        print(f"{'='*80}")
        print(f"Datasets: {len(self.datasets)}")
        print(f"Output: {self.results_dir}")

        # Analyze each dataset
        for dataset in self.datasets:
            self.analyze_dataset(dataset)

        # Cross-study analyses
        self.cross_study_enrichment_analysis()
        self.compare_control_vs_disease_enrichments()

        print(f"\n{'='*80}")
        print(f"MODULE ENRICHMENT ANALYSIS COMPLETE!")
        print(f"{'='*80}")


if __name__ == "__main__":
    analyzer = ModuleEnrichmentAnalyzer(RESULTS_DIR)
    analyzer.run_analysis()
