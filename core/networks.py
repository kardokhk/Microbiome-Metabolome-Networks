"""Network construction from associations using sparse representations."""

import pandas as pd
import numpy as np
import networkx as nx
from scipy import sparse
from typing import Dict, Tuple, Optional


class NetworkBuilder:
    """Builds bipartite species-metabolite networks from associations."""
    
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
    
    def build_bipartite_network(self, associations: pd.DataFrame, 
                                min_abs_corr: float = None) -> Tuple[nx.Graph, dict]:
        """Build NetworkX bipartite graph from associations.
        
        Args:
            associations: DataFrame with species, metabolite, rho, qvalue columns
            min_abs_corr: Minimum absolute correlation (default from config)
            
        Returns:
            tuple: (NetworkX Graph, statistics dict)
        """
        if min_abs_corr is None:
            min_abs_corr = self.config['MIN_CORRELATION']
        
        # Filter by correlation strength
        assoc_filtered = associations[associations['rho'].abs() >= min_abs_corr].copy()
        
        self.logger.info(
            f"  Building network with {len(assoc_filtered):,} edges (|ρ| ≥ {min_abs_corr})"
        )
        
        if len(assoc_filtered) == 0:
            self.logger.warning(f"  No edges left after filtering!")
            return None, None
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes with bipartite attribute
        species_nodes = assoc_filtered['species'].unique()
        metabolite_nodes = assoc_filtered['metabolite'].unique()
        
        G.add_nodes_from(species_nodes, bipartite=0, node_type='species')
        G.add_nodes_from(metabolite_nodes, bipartite=1, node_type='metabolite')
        
        # Add edges with attributes
        edges = []
        for _, row in assoc_filtered.iterrows():
            edges.append((
                row['species'],
                row['metabolite'],
                {
                    'weight': row['rho'],
                    'abs_weight': abs(row['rho']),
                    'qvalue': row['qvalue']
                }
            ))
        
        G.add_edges_from(edges)
        
        # Compute statistics
        n_components = nx.number_connected_components(G)
        
        stats = {
            'n_nodes': G.number_of_nodes(),
            'n_species': len(species_nodes),
            'n_metabolites': len(metabolite_nodes),
            'n_edges': G.number_of_edges(),
            'n_positive_edges': sum(1 for _, _, d in G.edges(data=True) if d['weight'] > 0),
            'n_negative_edges': sum(1 for _, _, d in G.edges(data=True) if d['weight'] < 0),
            'density': nx.density(G),
            'n_connected_components': n_components
        }
        
        self.logger.info(
            f"  Network: {stats['n_nodes']} nodes "
            f"({stats['n_species']} species, {stats['n_metabolites']} metabolites), "
            f"{stats['n_edges']} edges"
        )
        self.logger.info(
            f"    Positive edges: {stats['n_positive_edges']} "
            f"({100*stats['n_positive_edges']/stats['n_edges']:.1f}%)"
        )
        self.logger.info(
            f"    Negative edges: {stats['n_negative_edges']} "
            f"({100*stats['n_negative_edges']/stats['n_edges']:.1f}%)"
        )
        self.logger.info(f"    Density: {stats['density']:.4f}")
        self.logger.info(f"    Connected components: {n_components}")
        
        # Get largest connected component
        if n_components > 1:
            largest_cc = max(nx.connected_components(G), key=len)
            G_main = G.subgraph(largest_cc).copy()
            stats['largest_component_nodes'] = G_main.number_of_nodes()
            stats['largest_component_edges'] = G_main.number_of_edges()
            
            self.logger.info(
                f"    Largest component: {stats['largest_component_nodes']} nodes, "
                f"{stats['largest_component_edges']} edges"
            )
            
            return G_main, stats
        else:
            stats['largest_component_nodes'] = stats['n_nodes']
            stats['largest_component_edges'] = stats['n_edges']
            return G, stats
    
    def networkx_to_sparse(self, G: nx.Graph) -> Tuple[sparse.csr_matrix, Dict]:
        """Convert NetworkX graph to sparse adjacency matrix.
        
        Args:
            G: NetworkX graph
            
        Returns:
            tuple: (sparse adjacency matrix, node mapping dict)
        """
        # Create node mapping
        nodes = list(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        n_nodes = len(nodes)
        
        # Build COO matrix (efficient for construction)
        row_indices = []
        col_indices = []
        data = []
        
        for u, v, edge_data in G.edges(data=True):
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            weight = edge_data.get('weight', 1.0)
            
            row_indices.append(u_idx)
            col_indices.append(v_idx)
            data.append(weight)
            
            # Symmetric (undirected)
            row_indices.append(v_idx)
            col_indices.append(u_idx)
            data.append(weight)
        
        adjacency = sparse.coo_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_nodes, n_nodes),
            dtype=np.float32
        )
        
        # Convert to CSR for efficient operations
        adjacency_csr = adjacency.tocsr()
        
        # Create mapping info
        mapping = {
            'node_to_idx': node_to_idx,
            'idx_to_node': {v: k for k, v in node_to_idx.items()},
            'nodes': nodes
        }
        
        # Calculate sparsity
        sparsity = 100 * (1 - adjacency_csr.nnz / (n_nodes * n_nodes))
        self.logger.debug(f"  Sparse matrix: {n_nodes}×{n_nodes}, sparsity={sparsity:.2f}%")
        
        return adjacency_csr, mapping
    
    def extract_edge_list(self, G: nx.Graph) -> pd.DataFrame:
        """Extract edge list as DataFrame for easy saving.
        
        Args:
            G: NetworkX graph
            
        Returns:
            pd.DataFrame: Edge list with attributes
        """
        edges = []
        for u, v, data in G.edges(data=True):
            edges.append({
                'source': u,
                'target': v,
                'weight': data['weight'],
                'abs_weight': data['abs_weight'],
                'qvalue': data['qvalue']
            })
        
        return pd.DataFrame(edges)
