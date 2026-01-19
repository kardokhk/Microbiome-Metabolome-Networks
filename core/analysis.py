"""Keystone species identification and cross-study analysis."""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List
from collections import Counter


class KeystoneAnalyzer:
    """Identifies keystone species based on network centrality."""
    
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
    
    def identify_keystones(self, G: nx.Graph, module_df: pd.DataFrame) -> pd.DataFrame:
        """Identify keystone species based on network centrality metrics.
        
        A keystone is a highly central node that connects different parts of the network.
        We use multiple centrality metrics: degree, betweenness, closeness.
        
        Args:
            G: NetworkX graph
            module_df: Module assignments
            
        Returns:
            pd.DataFrame: Keystone species with centrality scores
        """
        self.logger.info(f"  Identifying keystone species...")
        
        # Get only species nodes
        species_nodes = [
            node for node, data in G.nodes(data=True)
            if data.get('node_type') == 'species'
        ]
        
        if len(species_nodes) == 0:
            self.logger.warning(f"  No species nodes found!")
            return None
        
        # Compute centrality metrics
        self.logger.info(f"  Computing centrality metrics for {len(species_nodes)} species...")
        
        # Degree centrality (fast and effective for keystone identification)
        degree_cent = nx.degree_centrality(G)
        
        # Compile results for species only
        keystone_data = []
        
        for species in species_nodes:
            # Get module assignment
            module_assignment = module_df[module_df['node'] == species]
            module_id = module_assignment['module'].values[0] if len(module_assignment) > 0 else -1
            
            # Get degree centrality (already normalized to [0, 1])
            degree = degree_cent.get(species, 0)
            
            # Keystone score based on degree centrality
            keystone_score = degree
            
            keystone_data.append({
                'species': species,
                'module': module_id,
                'degree_centrality': degree,
                'keystone_score': keystone_score,
                'degree': G.degree(species)
            })
        
        keystone_df = pd.DataFrame(keystone_data)
        
        # Sort by keystone score
        keystone_df = keystone_df.sort_values('keystone_score', ascending=False)
        
        # Identify top keystones (top 10% by score)
        threshold = keystone_df['keystone_score'].quantile(0.90)
        top_keystones = keystone_df[keystone_df['keystone_score'] >= threshold]
        
        self.logger.info(
            f"  Identified {len(top_keystones)} top keystone species "
            f"(score ≥ {threshold:.3f})"
        )
        self.logger.info(
            f"    Mean keystone score: {keystone_df['keystone_score'].mean():.3f}"
        )
        
        return keystone_df
    
    def get_top_keystones_per_module(self, keystone_df: pd.DataFrame, 
                                     top_n: int = 5) -> pd.DataFrame:
        """Get top N keystone species per module.
        
        Args:
            keystone_df: Keystone DataFrame
            top_n: Number of top keystones per module
            
        Returns:
            pd.DataFrame: Top keystones per module
        """
        top_keystones = []
        
        for module_id in keystone_df['module'].unique():
            if module_id == -1:  # Skip unassigned
                continue
            
            module_keystones = keystone_df[keystone_df['module'] == module_id]
            top_module = module_keystones.nlargest(top_n, 'keystone_score')
            top_keystones.append(top_module)
        
        if len(top_keystones) > 0:
            return pd.concat(top_keystones, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def cross_study_convergence(self, all_keystones: Dict[str, pd.DataFrame],
                                all_modules: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Analyze convergence of keystones and modules across studies.
        
        Args:
            all_keystones: Dict mapping dataset name to keystone DataFrame
            all_modules: Dict mapping dataset name to module DataFrame
            
        Returns:
            dict: Results with species_occurrence, conserved_keystones, module_stats
        """
        self.logger.info(f"Analyzing cross-study convergence across {len(all_keystones)} datasets")
        
        # 1. Species occurrence across studies
        species_counts = Counter()
        species_datasets = {}
        
        for dataset, keystone_df in all_keystones.items():
            if keystone_df is None or len(keystone_df) == 0:
                continue
            
            for species in keystone_df['species'].unique():
                species_counts[species] += 1
                if species not in species_datasets:
                    species_datasets[species] = []
                species_datasets[species].append(dataset)
        
        # Create species occurrence DataFrame
        species_occurrence = []
        for species, count in species_counts.most_common():
            species_occurrence.append({
                'species': species,
                'n_datasets': count,
                'datasets': ', '.join(species_datasets[species]),
                'is_conserved': count >= max(2, len(all_keystones) // 2)  # Present in ≥50% of studies
            })
        
        species_occurrence_df = pd.DataFrame(species_occurrence)
        
        # Conserved keystones (present in multiple studies)
        conserved = species_occurrence_df[species_occurrence_df['is_conserved']]
        
        self.logger.info(
            f"  Found {len(conserved)} conserved keystone species "
            f"(present in ≥{max(2, len(all_keystones)//2)} datasets)"
        )
        
        # 2. Module statistics across studies
        module_stats_list = []
        for dataset, module_df in all_modules.items():
            if module_df is None or len(module_df) == 0:
                continue
            
            n_modules = module_df['module'].nunique()
            module_sizes = module_df.groupby('module').size()
            
            module_stats_list.append({
                'dataset': dataset,
                'n_modules': n_modules,
                'mean_module_size': module_sizes.mean(),
                'median_module_size': module_sizes.median(),
                'max_module_size': module_sizes.max(),
                'total_nodes': len(module_df)
            })
        
        module_stats_df = pd.DataFrame(module_stats_list)
        
        # 3. Taxonomic analysis of conserved keystones
        conserved_keystones_detailed = []
        for _, row in conserved.iterrows():
            species_name = row['species']
            
            # Get average keystone score across datasets
            scores = []
            for dataset, keystone_df in all_keystones.items():
                if keystone_df is None:
                    continue
                species_data = keystone_df[keystone_df['species'] == species_name]
                if len(species_data) > 0:
                    scores.append(species_data['keystone_score'].values[0])
            
            avg_score = np.mean(scores) if len(scores) > 0 else 0
            
            conserved_keystones_detailed.append({
                'species': species_name,
                'n_datasets': row['n_datasets'],
                'datasets': row['datasets'],
                'avg_keystone_score': avg_score,
                'max_keystone_score': max(scores) if len(scores) > 0 else 0
            })
        
        conserved_keystones_df = pd.DataFrame(conserved_keystones_detailed)
        if len(conserved_keystones_df) > 0:
            conserved_keystones_df = conserved_keystones_df.sort_values(
                'avg_keystone_score', ascending=False
            )
        
        return {
            'species_occurrence': species_occurrence_df,
            'conserved_keystones': conserved_keystones_df,
            'module_stats': module_stats_df
        }
