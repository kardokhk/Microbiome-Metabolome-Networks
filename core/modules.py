"""Module detection using Leiden algorithm with igraph."""

import pandas as pd
import numpy as np
import networkx as nx
import igraph as ig
import leidenalg as la
from typing import Tuple, Dict
from collections import Counter


class ModuleDetector:
    """Detects functional modules in bipartite networks."""
    
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
    
    def networkx_to_igraph(self, G: nx.Graph) -> Tuple[ig.Graph, Dict]:
        """Convert NetworkX graph to igraph for Leiden algorithm.
        
        Args:
            G: NetworkX graph
            
        Returns:
            tuple: (igraph Graph, node mapping dict)
        """
        # Create node mapping
        nodes = list(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Create edge list with indices
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
        
        # Create igraph
        g = ig.Graph(n=len(nodes), edges=edges)
        g.vs['name'] = nodes
        
        # Add node attributes
        for node in G.nodes():
            idx = node_to_idx[node]
            for attr, value in G.nodes[node].items():
                if attr not in g.vs.attributes():
                    g.vs[attr] = [None] * len(nodes)
                g.vs[idx][attr] = value
        
        # Add edge weights (use absolute weight for module detection)
        weights = []
        for u, v in G.edges():
            edge_data = G[u][v]
            weights.append(edge_data.get('abs_weight', 1.0))
        
        g.es['weight'] = weights
        
        mapping = {
            'node_to_idx': node_to_idx,
            'idx_to_node': {v: k for k, v in node_to_idx.items()},
            'nodes': nodes
        }
        
        return g, mapping
    
    def detect_modules(self, G: nx.Graph, resolution: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Detect modules using Leiden algorithm.
        
        Args:
            G: NetworkX graph
            resolution: Resolution parameter for Leiden (higher = more modules)
            
        Returns:
            tuple: (module_assignments DataFrame, module_stats DataFrame)
        """
        self.logger.info(f"  Detecting modules with Leiden algorithm (resolution={resolution})")
        
        # Convert to igraph
        g, mapping = self.networkx_to_igraph(G)
        
        # Run Leiden algorithm
        partition = la.find_partition(
            g,
            la.ModularityVertexPartition,
            weights='weight',
            n_iterations=-1,
            seed=self.config['RANDOM_STATE']
        )
        
        # Extract module assignments
        modules = {}
        for node_idx, module_id in enumerate(partition.membership):
            node_name = g.vs[node_idx]['name']
            modules[node_name] = module_id
        
        n_modules = len(set(partition.membership))
        modularity = partition.quality()
        
        self.logger.info(f"  Found {n_modules} modules (modularity={modularity:.3f})")
        
        # Create module assignments DataFrame
        module_data = []
        for node, module_id in modules.items():
            node_type = G.nodes[node].get('node_type', 'unknown')
            module_data.append({
                'node': node,
                'module': module_id,
                'node_type': node_type
            })
        
        module_df = pd.DataFrame(module_data)
        
        # Compute module statistics
        module_stats = self._compute_module_stats(G, module_df)
        
        # Filter small modules (< min_size)
        min_size = self.config.get('MIN_MODULE_SIZE', 5)
        large_modules = module_stats[module_stats['n_nodes'] >= min_size]['module'].tolist()
        
        if len(large_modules) < n_modules:
            self.logger.info(f"  Filtering to {len(large_modules)} modules with ≥{min_size} nodes")
            module_df = module_df[module_df['module'].isin(large_modules)].copy()
            module_stats = module_stats[module_stats['module'].isin(large_modules)].copy()
        
        # Summary statistics
        self.logger.info(f"  Module summary:")
        self.logger.info(f"    Avg nodes per module: {module_stats['n_nodes'].mean():.1f}")
        self.logger.info(f"    Avg species per module: {module_stats['n_species'].mean():.1f}")
        self.logger.info(f"    Avg metabolites per module: {module_stats['n_metabolites'].mean():.1f}")
        
        return module_df, module_stats
    
    def _compute_module_stats(self, G: nx.Graph, module_df: pd.DataFrame) -> pd.DataFrame:
        """Compute statistics for each module.
        
        Args:
            G: NetworkX graph
            module_df: Module assignments DataFrame
            
        Returns:
            pd.DataFrame: Module statistics
        """
        stats = []
        
        for module_id in sorted(module_df['module'].unique()):
            module_nodes = module_df[module_df['module'] == module_id]
            
            n_species = (module_nodes['node_type'] == 'species').sum()
            n_metabolites = (module_nodes['node_type'] == 'metabolite').sum()
            
            # Get subgraph for this module
            module_node_list = module_nodes['node'].tolist()
            subgraph = G.subgraph(module_node_list)
            
            # Count positive and negative edges
            n_pos = sum(1 for _, _, d in subgraph.edges(data=True) if d.get('weight', 0) > 0)
            n_neg = sum(1 for _, _, d in subgraph.edges(data=True) if d.get('weight', 0) < 0)
            
            # Compute density
            density = nx.density(subgraph) if len(module_nodes) > 1 else 0
            
            stats.append({
                'module': module_id,
                'n_nodes': len(module_nodes),
                'n_species': n_species,
                'n_metabolites': n_metabolites,
                'n_edges': subgraph.number_of_edges(),
                'n_positive_edges': n_pos,
                'n_negative_edges': n_neg,
                'density': density
            })
        
        return pd.DataFrame(stats)
