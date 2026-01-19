"""Hub metabolite analysis - identify keystone metabolites in networks."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.config import DATASETS, RESULTS_DIR


class HubMetaboliteAnalyzer:
    """Identify hub metabolites in species-metabolite networks."""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.datasets = DATASETS

        # Storage for cross-study analysis
        self.all_hubs = defaultdict(lambda: {'control': None, 'disease': None})
        self.conserved_hubs_control = []
        self.conserved_hubs_disease = []

    def load_network(self, dataset: str, group: str) -> nx.Graph:
        """Load network from edge list.

        Args:
            dataset: Dataset name
            group: 'control' or 'disease'

        Returns:
            NetworkX graph or None
        """
        edge_file = self.results_dir / dataset / group / 'network_edges.csv'

        if not edge_file.exists():
            print(f"  Warning: {edge_file} not found")
            return None

        # Load edges
        edges_df = pd.read_csv(edge_file)

        # Create graph
        G = nx.Graph()
        for _, row in edges_df.iterrows():
            G.add_edge(row['source'], row['target'], weight=row['abs_weight'])

        print(f"  Loaded network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def identify_metabolite_nodes(self, G: nx.Graph) -> set:
        """Identify which nodes are metabolites vs species.

        Args:
            G: NetworkX graph

        Returns:
            Set of metabolite node names
        """
        metabolite_nodes = set()

        for node in G.nodes():
            # Metabolites don't start with "d  bacteria" (species taxonomy)
            if not node.startswith('d  bacteria'):
                metabolite_nodes.add(node)

        return metabolite_nodes

    def compute_hub_scores(self, G: nx.Graph, metabolite_nodes: set) -> pd.DataFrame:
        """Compute centrality metrics for metabolite nodes.

        Args:
            G: NetworkX graph
            metabolite_nodes: Set of metabolite node names

        Returns:
            DataFrame with hub scores
        """
        if len(metabolite_nodes) == 0:
            return pd.DataFrame()

        hub_scores = []
        n_nodes = G.number_of_nodes()

        # For each metabolite (skip betweenness for speed)
        for metabolite in metabolite_nodes:
            if metabolite not in G:
                continue

            # Degree (number of species connected)
            degree = G.degree(metabolite)

            # Weighted degree (sum of edge weights)
            weighted_degree = sum(G[metabolite][neighbor]['weight']
                                 for neighbor in G.neighbors(metabolite))

            # Hub score (normalized by network size)
            hub_score = degree / n_nodes

            hub_scores.append({
                'metabolite': metabolite,
                'degree': degree,
                'weighted_degree': round(weighted_degree, 3),
                'hub_score': round(hub_score, 6)
            })

        # Convert to DataFrame and sort
        df = pd.DataFrame(hub_scores)
        if len(df) > 0:
            df = df.sort_values('hub_score', ascending=False)

        return df

    def analyze_dataset(self, dataset: str):
        """Analyze hub metabolites for control and disease.

        Args:
            dataset: Dataset name
        """
        print(f"\nAnalyzing {dataset}...")

        for group in ['control', 'disease']:
            print(f"  Processing {group}...")

            # Load network
            G = self.load_network(dataset, group)
            if G is None:
                continue

            # Identify metabolite nodes
            metabolite_nodes = self.identify_metabolite_nodes(G)
            print(f"  Found {len(metabolite_nodes)} metabolite nodes")

            if len(metabolite_nodes) == 0:
                continue

            # Compute hub scores
            hub_df = self.compute_hub_scores(G, metabolite_nodes)

            if len(hub_df) == 0:
                continue

            # Save results
            out_dir = self.results_dir / dataset / group
            out_dir.mkdir(exist_ok=True, parents=True)

            # Full hub scores
            out_file = out_dir / 'hub_metabolites.csv'
            hub_df.to_csv(out_file, index=False)
            print(f"  Saved {len(hub_df)} hub metabolites to {out_file}")

            # Top hubs only (hub_score >= threshold)
            threshold = 0.005  # Top ~0.5% of network
            top_hubs = hub_df[hub_df['hub_score'] >= threshold].copy()

            if len(top_hubs) > 0:
                out_file = out_dir / 'top_hub_metabolites.csv'
                top_hubs.to_csv(out_file, index=False)
                print(f"  Saved {len(top_hubs)} top hub metabolites (score >= {threshold})")

            # Store for cross-study analysis
            hub_df['dataset'] = dataset
            hub_df['group'] = group
            self.all_hubs[dataset][group] = hub_df

    def differential_hub_analysis(self, dataset: str):
        """Compare hub metabolites between control and disease.

        Args:
            dataset: Dataset name
        """
        control_hubs = self.all_hubs[dataset]['control']
        disease_hubs = self.all_hubs[dataset]['disease']

        if control_hubs is None or disease_hubs is None:
            return

        print(f"\n  Differential analysis for {dataset}...")

        # Merge control and disease
        control_hubs = control_hubs.set_index('metabolite')[['degree', 'hub_score']]
        control_hubs.columns = ['degree_control', 'hub_score_control']

        disease_hubs = disease_hubs.set_index('metabolite')[['degree', 'hub_score']]
        disease_hubs.columns = ['degree_disease', 'hub_score_disease']

        # Outer join to get all metabolites
        diff_df = control_hubs.join(disease_hubs, how='outer').fillna(0)

        # Calculate fold changes
        diff_df['log2_fc'] = np.where(
            (diff_df['hub_score_control'] > 0) & (diff_df['hub_score_disease'] > 0),
            np.log2(diff_df['hub_score_disease'] / diff_df['hub_score_control']),
            np.where(diff_df['hub_score_disease'] > 0, 5, -5)  # Large FC for specific
        )
        diff_df['delta_hub_score'] = diff_df['hub_score_disease'] - diff_df['hub_score_control']

        # Classify
        diff_df['status'] = 'unchanged'
        diff_df.loc[diff_df['log2_fc'] > 1, 'status'] = 'enriched_in_disease'
        diff_df.loc[diff_df['log2_fc'] < -1, 'status'] = 'depleted_in_disease'
        diff_df.loc[(diff_df['hub_score_control'] > 0) & (diff_df['hub_score_disease'] == 0), 'status'] = 'control_specific'
        diff_df.loc[(diff_df['hub_score_control'] == 0) & (diff_df['hub_score_disease'] > 0), 'status'] = 'disease_specific'

        # Sort by absolute fold change
        diff_df['abs_log2_fc'] = diff_df['log2_fc'].abs()
        diff_df = diff_df.sort_values('abs_log2_fc', ascending=False)
        diff_df = diff_df.drop('abs_log2_fc', axis=1)

        # Add flags
        diff_df['in_control'] = diff_df['hub_score_control'] > 0
        diff_df['in_disease'] = diff_df['hub_score_disease'] > 0

        # Round
        diff_df = diff_df.round(6)

        # Save
        out_dir = self.results_dir / dataset / 'differential'
        out_dir.mkdir(exist_ok=True, parents=True)
        out_file = out_dir / 'differential_hub_metabolites.csv'

        diff_df.reset_index().to_csv(out_file, index=False)
        print(f"  Saved differential hub metabolites: {out_file}")

        # Summary
        status_counts = diff_df['status'].value_counts()
        print(f"  Summary:")
        for status, count in status_counts.items():
            print(f"    {status}: {count}")

    def cross_study_analysis(self):
        """Identify conserved hub metabolites across studies."""
        print("\n" + "=" * 80)
        print("CROSS-STUDY HUB METABOLITE ANALYSIS")
        print("=" * 80)

        for group in ['control', 'disease']:
            print(f"\nAnalyzing conserved hubs in {group.upper()} networks...")

            # Collect all hub metabolites across datasets
            hub_counts = defaultdict(lambda: {'datasets': set(), 'total_degree': 0, 'mean_hub_score': []})

            for dataset in self.datasets:
                hub_df = self.all_hubs[dataset][group]

                if hub_df is None:
                    continue

                # Only consider top hubs (above threshold)
                top_hubs = hub_df[hub_df['hub_score'] >= 0.005]

                for _, row in top_hubs.iterrows():
                    metabolite = row['metabolite']
                    hub_counts[metabolite]['datasets'].add(dataset)
                    hub_counts[metabolite]['total_degree'] += row['degree']
                    hub_counts[metabolite]['mean_hub_score'].append(row['hub_score'])

            # Build conserved hub dataframe
            conserved = []
            for metabolite, data in hub_counts.items():
                n_datasets = len(data['datasets'])
                if n_datasets >= 2:  # Present in at least 2 datasets
                    conserved.append({
                        'metabolite': metabolite,
                        'n_datasets': n_datasets,
                        'datasets': ';'.join(sorted(data['datasets'])),
                        'total_degree': data['total_degree'],
                        'mean_hub_score': round(np.mean(data['mean_hub_score']), 6)
                    })

            conserved_df = pd.DataFrame(conserved)
            if len(conserved_df) > 0:
                conserved_df = conserved_df.sort_values('n_datasets', ascending=False)

                # Save
                out_dir = self.results_dir / 'cross_study' / group
                out_dir.mkdir(exist_ok=True, parents=True)
                out_file = out_dir / 'conserved_hub_metabolites.csv'
                conserved_df.to_csv(out_file, index=False)
                print(f"  Saved {len(conserved_df)} conserved hub metabolites: {out_file}")

                # Print top conserved hubs
                print(f"\n  Top conserved hub metabolites in {group}:")
                for _, row in conserved_df.head(20).iterrows():
                    print(f"    {row['metabolite']:60s} - {row['n_datasets']} datasets, "
                          f"mean hub score: {row['mean_hub_score']:.6f}")

                # Store
                if group == 'control':
                    self.conserved_hubs_control = conserved_df
                else:
                    self.conserved_hubs_disease = conserved_df
            else:
                print(f"  No conserved hub metabolites found in {group}")

    def run_analysis(self):
        """Run complete hub metabolite analysis."""
        print("=" * 80)
        print("HUB METABOLITE ANALYSIS")
        print("=" * 80)

        # Analyze each dataset
        for dataset in self.datasets:
            self.analyze_dataset(dataset)

        # Differential analysis for each dataset
        print("\n" + "=" * 80)
        print("DIFFERENTIAL HUB ANALYSIS (Control vs Disease)")
        print("=" * 80)

        for dataset in self.datasets:
            self.differential_hub_analysis(dataset)

        # Cross-study analysis
        self.cross_study_analysis()

        print("\n" + "=" * 80)
        print("HUB METABOLITE ANALYSIS COMPLETE!")
        print("=" * 80)


if __name__ == "__main__":
    analyzer = HubMetaboliteAnalyzer(RESULTS_DIR)
    analyzer.run_analysis()
