"""Differential network analysis comparing control vs disease."""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional
from scipy import stats


class DifferentialAnalyzer:
    """Performs differential analysis between control and disease networks."""
    
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
    
    def compare_keystone_scores(self, 
                                control_keystones: pd.DataFrame,
                                disease_keystones: pd.DataFrame) -> pd.DataFrame:
        """Compare keystone scores between control and disease.
        
        Args:
            control_keystones: Keystone DataFrame from control samples
            disease_keystones: Keystone DataFrame from disease samples
            
        Returns:
            pd.DataFrame: Differential keystone analysis
        """
        # Merge on species
        merged = pd.merge(
            control_keystones[['species', 'keystone_score', 'degree']],
            disease_keystones[['species', 'keystone_score', 'degree']],
            on='species',
            how='outer',
            suffixes=('_control', '_disease')
        )
        
        # Fill NaN with 0 (species absent in one group)
        merged['keystone_score_control'] = merged['keystone_score_control'].fillna(0)
        merged['keystone_score_disease'] = merged['keystone_score_disease'].fillna(0)
        merged['degree_control'] = merged['degree_control'].fillna(0)
        merged['degree_disease'] = merged['degree_disease'].fillna(0)
        
        # Calculate fold change (log2)
        # Add pseudocount to avoid division by zero
        pseudocount = 1e-6
        merged['log2_fc'] = np.log2(
            (merged['keystone_score_disease'] + pseudocount) / 
            (merged['keystone_score_control'] + pseudocount)
        )
        
        # Absolute change
        merged['delta_keystone_score'] = (
            merged['keystone_score_disease'] - merged['keystone_score_control']
        )
        
        # Categorize
        merged['status'] = 'unchanged'
        merged.loc[merged['log2_fc'] > 1, 'status'] = 'enriched_in_disease'
        merged.loc[merged['log2_fc'] < -1, 'status'] = 'depleted_in_disease'
        
        # Presence/absence
        merged['in_control'] = merged['keystone_score_control'] > 0
        merged['in_disease'] = merged['keystone_score_disease'] > 0
        
        # Specific to one group
        merged.loc[
            merged['in_control'] & ~merged['in_disease'], 
            'status'
        ] = 'control_specific'
        merged.loc[
            ~merged['in_control'] & merged['in_disease'], 
            'status'
        ] = 'disease_specific'
        
        # Sort by absolute fold change
        merged = merged.sort_values('log2_fc', ascending=True)
        
        return merged
    
    def compare_network_structure(self,
                                  control_network: nx.Graph,
                                  disease_network: nx.Graph) -> Dict:
        """Compare structural properties of control vs disease networks.
        
        Args:
            control_network: Network from control samples
            disease_network: Network from disease samples
            
        Returns:
            dict: Network comparison statistics
        """
        stats_dict = {}
        
        # Basic properties
        stats_dict['n_nodes_control'] = control_network.number_of_nodes()
        stats_dict['n_nodes_disease'] = disease_network.number_of_nodes()
        stats_dict['n_edges_control'] = control_network.number_of_edges()
        stats_dict['n_edges_disease'] = disease_network.number_of_edges()
        
        # Density
        stats_dict['density_control'] = nx.density(control_network)
        stats_dict['density_disease'] = nx.density(disease_network)
        
        # Connected components
        stats_dict['n_components_control'] = nx.number_connected_components(control_network)
        stats_dict['n_components_disease'] = nx.number_connected_components(disease_network)
        
        # Average clustering
        stats_dict['avg_clustering_control'] = nx.average_clustering(control_network)
        stats_dict['avg_clustering_disease'] = nx.average_clustering(disease_network)
        
        # Degree distribution
        control_degrees = [d for n, d in control_network.degree()]
        disease_degrees = [d for n, d in disease_network.degree()]
        
        stats_dict['mean_degree_control'] = np.mean(control_degrees)
        stats_dict['mean_degree_disease'] = np.mean(disease_degrees)
        stats_dict['median_degree_control'] = np.median(control_degrees)
        stats_dict['median_degree_disease'] = np.median(disease_degrees)
        
        return stats_dict
    
    def identify_differential_edges(self,
                                    control_assoc: pd.DataFrame,
                                    disease_assoc: pd.DataFrame,
                                    min_abs_delta_rho: float = 0.2) -> pd.DataFrame:
        """Identify edges with significantly different correlations.
        
        Args:
            control_assoc: Association DataFrame from control samples
            disease_assoc: Association DataFrame from disease samples
            min_abs_delta_rho: Minimum absolute change in correlation
            
        Returns:
            pd.DataFrame: Differential edges
        """
        # Create edge identifiers
        control_assoc['edge_id'] = (
            control_assoc['species'] + '___' + control_assoc['metabolite']
        )
        disease_assoc['edge_id'] = (
            disease_assoc['species'] + '___' + disease_assoc['metabolite']
        )
        
        # Merge on edge ID
        merged = pd.merge(
            control_assoc[['edge_id', 'species', 'metabolite', 'rho', 'qvalue']],
            disease_assoc[['edge_id', 'rho', 'qvalue']],
            on='edge_id',
            how='outer',
            suffixes=('_control', '_disease')
        )
        
        # Fill NaN with 0 (edge absent in one group)
        merged['rho_control'] = merged['rho_control'].fillna(0)
        merged['rho_disease'] = merged['rho_disease'].fillna(0)
        
        # Calculate change
        merged['delta_rho'] = merged['rho_disease'] - merged['rho_control']
        
        # Filter by minimum change
        significant = merged[merged['delta_rho'].abs() >= min_abs_delta_rho].copy()
        
        # Categorize
        significant['status'] = 'unchanged'
        significant.loc[significant['delta_rho'] > 0, 'status'] = 'stronger_in_disease'
        significant.loc[significant['delta_rho'] < 0, 'status'] = 'weaker_in_disease'
        
        # Edge-specific
        significant.loc[
            (significant['rho_control'] == 0) & (significant['rho_disease'] != 0),
            'status'
        ] = 'disease_specific_edge'
        significant.loc[
            (significant['rho_control'] != 0) & (significant['rho_disease'] == 0),
            'status'
        ] = 'control_specific_edge'
        
        # Sort by absolute change
        significant = significant.sort_values('delta_rho', key=abs, ascending=False)
        
        return significant
    
    def summarize_differential_analysis(self,
                                       diff_keystones: pd.DataFrame,
                                       network_comparison: Dict,
                                       n_samples_control: int = None,
                                       n_samples_disease: int = None,
                                       diff_edges: pd.DataFrame = None) -> pd.DataFrame:
        """Create a summary table of differential analysis.
        
        Args:
            diff_keystones: Differential keystone DataFrame
            network_comparison: Network comparison statistics
            n_samples_control: Number of control samples
            n_samples_disease: Number of disease samples
            diff_edges: Optional differential edges DataFrame
            
        Returns:
            pd.DataFrame: Summary statistics
        """
        summary = []
        
        # Sample counts (if provided)
        if n_samples_control is not None and n_samples_disease is not None:
            summary.append({
                'metric': 'Number of samples',
                'control': n_samples_control,
                'disease': n_samples_disease,
                'difference': n_samples_disease - n_samples_control
            })
        
        # Keystone changes
        summary.append({
            'metric': 'Total species analyzed',
            'control': len(diff_keystones),
            'disease': len(diff_keystones),
            'difference': 0
        })
        
        summary.append({
            'metric': 'Control-specific keystones',
            'control': sum(diff_keystones['status'] == 'control_specific'),
            'disease': 0,
            'difference': -sum(diff_keystones['status'] == 'control_specific')
        })
        
        summary.append({
            'metric': 'Disease-specific keystones',
            'control': 0,
            'disease': sum(diff_keystones['status'] == 'disease_specific'),
            'difference': sum(diff_keystones['status'] == 'disease_specific')
        })
        
        summary.append({
            'metric': 'Depleted in disease (log2FC < -1)',
            'control': sum(diff_keystones['status'] == 'depleted_in_disease'),
            'disease': 0,
            'difference': -sum(diff_keystones['status'] == 'depleted_in_disease')
        })
        
        summary.append({
            'metric': 'Enriched in disease (log2FC > 1)',
            'control': 0,
            'disease': sum(diff_keystones['status'] == 'enriched_in_disease'),
            'difference': sum(diff_keystones['status'] == 'enriched_in_disease')
        })
        
        # Network structure
        summary.append({
            'metric': 'Network nodes',
            'control': network_comparison['n_nodes_control'],
            'disease': network_comparison['n_nodes_disease'],
            'difference': (
                network_comparison['n_nodes_disease'] - 
                network_comparison['n_nodes_control']
            )
        })
        
        summary.append({
            'metric': 'Network edges',
            'control': network_comparison['n_edges_control'],
            'disease': network_comparison['n_edges_disease'],
            'difference': (
                network_comparison['n_edges_disease'] - 
                network_comparison['n_edges_control']
            )
        })
        
        summary.append({
            'metric': 'Network density',
            'control': network_comparison['density_control'],
            'disease': network_comparison['density_disease'],
            'difference': (
                network_comparison['density_disease'] - 
                network_comparison['density_control']
            )
        })
        
        summary.append({
            'metric': 'Mean degree',
            'control': network_comparison['mean_degree_control'],
            'disease': network_comparison['mean_degree_disease'],
            'difference': (
                network_comparison['mean_degree_disease'] - 
                network_comparison['mean_degree_control']
            )
        })
        
        return pd.DataFrame(summary)
