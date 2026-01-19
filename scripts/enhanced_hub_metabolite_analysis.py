"""Enhanced hub metabolite analysis - cross-study integration and functional classification."""

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
from scripts.metabolite_classifier import MetaboliteClassifier


class EnhancedHubMetaboliteAnalyzer:
    """Enhanced cross-study hub metabolite analysis with functional classification."""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.datasets = DATASETS
        self.classifier = MetaboliteClassifier()

        # Disease categories
        self.disease_categories = {
            'IBD': ['FRANZOSA_IBD_2019', 'iHMP_IBDMDB_2019'],
            'CRC': ['YACHIDA_CRC_2019'],
            'Gastric_Cancer': ['ERAWIJANTARI_GASTRIC_CANCER_2020'],
            'IBS': ['MARS_IBS_2020'],
            'Metabolic': ['WANG_ESRD_2020']
        }

        # Storage
        self.all_hub_metabolites = {'control': {}, 'disease': {}}
        self.all_differential_hubs = {}

    def get_dataset_disease_category(self, dataset: str) -> str:
        """Get disease category for a dataset."""
        for category, datasets in self.disease_categories.items():
            if dataset in datasets:
                return category
        return 'Unknown'

    def load_hub_metabolite_data(self):
        """Load hub metabolite data from all datasets."""
        print(f"Loading hub metabolite data from {len(self.datasets)} datasets...")

        for dataset in self.datasets:
            for group in ['control', 'disease']:
                hub_file = self.results_dir / dataset / group / 'hub_metabolites.csv'

                if not hub_file.exists():
                    print(f"  Warning: {hub_file} not found")
                    continue

                hub_df = pd.read_csv(hub_file)

                # Classify metabolites
                hub_df = self.classifier.classify_dataframe(hub_df, 'metabolite')

                self.all_hub_metabolites[group][dataset] = hub_df

            # Load differential hubs if available
            diff_hub_file = self.results_dir / dataset / 'differential' / 'differential_hub_metabolites.csv'
            if diff_hub_file.exists():
                diff_hub_df = pd.read_csv(diff_hub_file)
                diff_hub_df = self.classifier.classify_dataframe(diff_hub_df, 'metabolite')
                self.all_differential_hubs[dataset] = diff_hub_df

        print(f"  Loaded hub data: {len(self.all_hub_metabolites['control'])} control, "
              f"{len(self.all_hub_metabolites['disease'])} disease")
        print(f"  Loaded differential hub data: {len(self.all_differential_hubs)} datasets")

    def cross_study_hub_convergence(self):
        """Improved cross-study hub convergence with lower threshold and functional grouping."""
        print(f"\n{'='*80}")
        print(f"CROSS-STUDY HUB CONVERGENCE (Enhanced)")
        print(f"{'='*80}")

        hub_status_data = []

        # Get all unique metabolites across all datasets
        all_metabolites = set()
        for group_hubs in self.all_hub_metabolites.values():
            for hub_df in group_hubs.values():
                all_metabolites.update(hub_df['metabolite'].unique())

        print(f"Analyzing {len(all_metabolites)} unique metabolites...")

        for metabolite in all_metabolites:
            # Count in how many datasets this metabolite is a hub (threshold ≥ 0.005)
            control_datasets = []
            disease_datasets = []

            control_scores = []
            disease_scores = []

            for dataset in self.datasets:
                # Control
                if dataset in self.all_hub_metabolites['control']:
                    control_df = self.all_hub_metabolites['control'][dataset]
                    met_row = control_df[control_df['metabolite'] == metabolite]
                    if len(met_row) > 0 and met_row.iloc[0]['hub_score'] >= 0.005:
                        control_datasets.append(dataset)
                        control_scores.append(met_row.iloc[0]['hub_score'])

                # Disease
                if dataset in self.all_hub_metabolites['disease']:
                    disease_df = self.all_hub_metabolites['disease'][dataset]
                    met_row = disease_df[disease_df['metabolite'] == metabolite]
                    if len(met_row) > 0 and met_row.iloc[0]['hub_score'] >= 0.005:
                        disease_datasets.append(dataset)
                        disease_scores.append(met_row.iloc[0]['hub_score'])

            n_control = len(control_datasets)
            n_disease = len(disease_datasets)

            # Only keep metabolites that are hubs in ≥1 dataset
            if n_control == 0 and n_disease == 0:
                continue

            # Classify hub status
            if n_control >= 2 and n_disease == 0:
                hub_status = 'conserved_control_hub'
            elif n_disease >= 2 and n_control == 0:
                hub_status = 'conserved_disease_hub'
            elif n_control >= 2 and n_disease >= 2:
                hub_status = 'universal_hub'
            elif n_control >= 2 and n_disease < n_control:
                hub_status = 'lost_in_disease'
            elif n_disease >= 2 and n_control < n_disease:
                hub_status = 'gained_in_disease'
            else:
                hub_status = 'variable'

            # Get metabolite class
            met_class, _ = self.classifier.classify_metabolite(metabolite)

            hub_status_data.append({
                'metabolite': metabolite,
                'metabolite_class': met_class,
                'n_control_hubs': n_control,
                'n_disease_hubs': n_disease,
                'datasets_control': ';'.join(control_datasets) if control_datasets else '',
                'datasets_disease': ';'.join(disease_datasets) if disease_datasets else '',
                'mean_hub_score_control': round(np.mean(control_scores), 6) if control_scores else 0,
                'mean_hub_score_disease': round(np.mean(disease_scores), 6) if disease_scores else 0,
                'hub_status': hub_status
            })

        hub_status_df = pd.DataFrame(hub_status_data)
        hub_status_df = hub_status_df.sort_values(
            ['n_control_hubs', 'n_disease_hubs'], ascending=[False, False]
        )

        # Save
        out_dir = self.results_dir / 'cross_study' / 'hub_metabolites'
        out_dir.mkdir(exist_ok=True, parents=True)
        out_file = out_dir / 'conserved_hub_status.csv'
        hub_status_df.to_csv(out_file, index=False)

        print(f"\nSaved hub status: {out_file}")
        print(f"Total metabolites analyzed: {len(hub_status_df)}")
        print(f"\nHub status distribution:")
        status_counts = hub_status_df['hub_status'].value_counts()
        for status, count in status_counts.items():
            print(f"  {status:30s}: {count}")

        return hub_status_df

    def analyze_hub_status_by_class(self, hub_status_df: pd.DataFrame):
        """Analyze which metabolite classes lose hub status in disease.

        Args:
            hub_status_df: Output from cross_study_hub_convergence
        """
        print(f"\n{'='*80}")
        print(f"HUB STATUS BY METABOLITE CLASS")
        print(f"{'='*80}")

        # Filter to metabolites present in ≥2 datasets (either control or disease)
        multi_study = hub_status_df[
            (hub_status_df['n_control_hubs'] >= 2) | (hub_status_df['n_disease_hubs'] >= 2)
        ].copy()

        if len(multi_study) == 0:
            print("  No metabolites found in ≥2 studies")
            return

        # Group by metabolite class
        class_summary = []

        for met_class in multi_study['metabolite_class'].unique():
            class_df = multi_study[multi_study['metabolite_class'] == met_class]

            n_hubs_total = len(class_df)
            n_lost_in_disease = len(class_df[class_df['hub_status'] == 'lost_in_disease'])
            n_conserved_control = len(class_df[class_df['hub_status'] == 'conserved_control_hub'])
            n_universal = len(class_df[class_df['hub_status'] == 'universal_hub'])

            # Calculate mean change in hub score
            control_scores = class_df['mean_hub_score_control']
            disease_scores = class_df['mean_hub_score_disease']
            mean_score_change = (disease_scores - control_scores).mean()

            class_summary.append({
                'metabolite_class': met_class,
                'n_hubs_total': n_hubs_total,
                'n_lost_in_disease': n_lost_in_disease,
                'n_conserved_control': n_conserved_control,
                'n_universal': n_universal,
                'fraction_lost': round(n_lost_in_disease / n_hubs_total, 3) if n_hubs_total > 0 else 0,
                'mean_hub_score_change': round(mean_score_change, 6)
            })

        class_summary_df = pd.DataFrame(class_summary)
        class_summary_df = class_summary_df.sort_values('fraction_lost', ascending=False)

        # Save
        out_dir = self.results_dir / 'cross_study' / 'hub_metabolites'
        out_file = out_dir / 'hub_status_by_class.csv'
        class_summary_df.to_csv(out_file, index=False)

        print(f"\nSaved hub status by class: {out_file}")
        print(f"\nMetabolite classes most affected in disease:")
        top = class_summary_df[class_summary_df['n_hubs_total'] >= 3].head(10)
        for _, row in top.iterrows():
            print(f"  {row['metabolite_class']:20s}: {row['fraction_lost']:.1%} lost "
                  f"({row['n_lost_in_disease']}/{row['n_hubs_total']} hubs)")

    def disease_specific_hub_profiles(self):
        """Identify hub metabolites specific to each disease category."""
        print(f"\n{'='*80}")
        print(f"DISEASE-SPECIFIC HUB PROFILES")
        print(f"{'='*80}")

        disease_hubs = defaultdict(lambda: defaultdict(list))

        # Collect disease-specific hubs (hub in disease but not control)
        for dataset, diff_hub_df in self.all_differential_hubs.items():
            category = self.get_dataset_disease_category(dataset)

            # Get disease-specific hubs
            disease_specific = diff_hub_df[
                diff_hub_df['status'] == 'disease_specific'
            ]

            for _, row in disease_specific.iterrows():
                metabolite = row['metabolite']
                met_class = row['metabolite_class']
                hub_score = row['hub_score_disease']

                disease_hubs[category][metabolite].append({
                    'dataset': dataset,
                    'hub_score': hub_score,
                    'metabolite_class': met_class
                })

        # Build summary
        disease_hub_profiles = []

        for category, metabolites in list(disease_hubs.items()):
            for metabolite, records in list(metabolites.items()):
                n_datasets = len(records)

                # Check if this metabolite is specific to this category
                other_categories = [cat for cat in self.disease_categories.keys() if cat != category]
                is_specific = True

                for other_cat in other_categories:
                    if metabolite in disease_hubs[other_cat]:
                        is_specific = False
                        break

                if n_datasets >= 1:  # At least 1 dataset (can adjust threshold)
                    mean_hub_score = np.mean([r['hub_score'] for r in records])
                    met_class = records[0]['metabolite_class']

                    disease_hub_profiles.append({
                        'disease_category': category,
                        'metabolite': metabolite,
                        'metabolite_class': met_class,
                        'n_datasets': n_datasets,
                        'datasets': ';'.join([r['dataset'] for r in records]),
                        'mean_hub_score': round(mean_hub_score, 6),
                        'category_specific': is_specific
                    })

        if len(disease_hub_profiles) == 0:
            print("  No disease-specific hub profiles found")
            return

        disease_hub_df = pd.DataFrame(disease_hub_profiles)
        disease_hub_df = disease_hub_df.sort_values(
            ['disease_category', 'n_datasets', 'mean_hub_score'],
            ascending=[True, False, False]
        )

        # Save
        out_dir = self.results_dir / 'cross_study' / 'hub_metabolites'
        out_file = out_dir / 'disease_specific_hub_profiles.csv'
        disease_hub_df.to_csv(out_file, index=False)

        print(f"\nSaved disease-specific hub profiles: {out_file}")
        print(f"Total disease-specific hubs: {len(disease_hub_df)}")
        print(f"\nHubs per disease category:")
        for category in sorted(disease_hub_df['disease_category'].unique()):
            cat_df = disease_hub_df[disease_hub_df['disease_category'] == category]
            specific = cat_df[cat_df['category_specific']].shape[0]
            print(f"  {category:20s}: {len(cat_df)} hubs ({specific} category-specific)")

    def hub_connectivity_changes(self):
        """Analyze connectivity changes for hub metabolites."""
        print(f"\n{'='*80}")
        print(f"HUB METABOLITE CONNECTIVITY CHANGES")
        print(f"{'='*80}")

        connectivity_changes = []

        for dataset, diff_hub_df in self.all_differential_hubs.items():
            # Get metabolites with significant degree changes
            significant = diff_hub_df[
                (diff_hub_df['hub_score_control'] > 0) | (diff_hub_df['hub_score_disease'] > 0)
            ].copy()

            for _, row in significant.iterrows():
                degree_control = row.get('degree_control', 0)
                degree_disease = row.get('degree_disease', 0)

                connectivity_change = degree_disease - degree_control

                # Only keep if change is substantial (>5 connections difference)
                if abs(connectivity_change) >= 5:
                    connectivity_changes.append({
                        'dataset': dataset,
                        'metabolite': row['metabolite'],
                        'metabolite_class': row['metabolite_class'],
                        'degree_control': int(degree_control),
                        'degree_disease': int(degree_disease),
                        'connectivity_change': int(connectivity_change),
                        'hub_score_control': row['hub_score_control'],
                        'hub_score_disease': row['hub_score_disease'],
                        'status': row['status']
                    })

        if len(connectivity_changes) == 0:
            print("  No substantial connectivity changes found")
            return

        connectivity_df = pd.DataFrame(connectivity_changes)
        connectivity_df = connectivity_df.sort_values('connectivity_change')

        # Save
        out_dir = self.results_dir / 'cross_study' / 'hub_metabolites'
        out_file = out_dir / 'hub_connectivity_changes.csv'
        connectivity_df.to_csv(out_file, index=False)

        print(f"\nSaved connectivity changes: {out_file}")
        print(f"Total metabolites with substantial connectivity changes: {len(connectivity_df)}")

        # Top losers
        top_losers = connectivity_df.head(10)
        if len(top_losers) > 0:
            print(f"\nTop 10 metabolites with largest connectivity loss:")
            for _, row in top_losers.iterrows():
                print(f"  {row['metabolite'][:60]:60s} | {row['dataset']:25s} | "
                      f"Δdegree: {row['connectivity_change']:+4d}")

    def run_analysis(self):
        """Run complete enhanced hub metabolite analysis."""
        print(f"\n{'='*80}")
        print(f"ENHANCED HUB METABOLITE ANALYSIS")
        print(f"{'='*80}")
        print(f"Datasets: {len(self.datasets)}")

        # Load data
        self.load_hub_metabolite_data()

        # Cross-study hub convergence (enhanced)
        hub_status_df = self.cross_study_hub_convergence()

        # Hub status by metabolite class
        self.analyze_hub_status_by_class(hub_status_df)

        # Disease-specific hub profiles
        self.disease_specific_hub_profiles()

        # Connectivity changes
        self.hub_connectivity_changes()

        print(f"\n{'='*80}")
        print(f"ENHANCED HUB METABOLITE ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print(f"\nKey outputs in: {self.results_dir / 'cross_study' / 'hub_metabolites'}/")
        print(f"  - conserved_hub_status.csv")
        print(f"  - hub_status_by_class.csv")
        print(f"  - disease_specific_hub_profiles.csv")
        print(f"  - hub_connectivity_changes.csv")


if __name__ == "__main__":
    analyzer = EnhancedHubMetaboliteAnalyzer(RESULTS_DIR)
    analyzer.run_analysis()
