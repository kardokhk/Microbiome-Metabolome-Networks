"""Memory-efficient data loading and management."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from scipy import sparse


class DatasetHandler:
    """Manages loading and memory-efficient storage of multi-omics data."""
    
    def __init__(self, dataset_name: str, data_dir: Path, logger):
        self.name = dataset_name
        self.data_dir = Path(data_dir) / dataset_name
        self.logger = logger
        
        # In-memory data (load once, keep in RAM)
        self.species_raw = None
        self.metabolites_raw = None
        self.metadata = None
        
        # Processed data (transformed in-place)
        self.species_processed = None
        self.metabolites_processed = None
        self.common_samples = None
        
        # Stratified sample lists
        self.control_samples = None
        self.disease_samples = None
        
        # Analysis results (sparse representations)
        self.associations = None
        self.network_adjacency = None  # Sparse matrix
        self.modules = None
        
    def load_raw_data(self) -> bool:
        """Load raw species, metabolite, and metadata files.
        
        Returns:
            bool: True if data loaded successfully
        """
        if not self.data_dir.exists():
            self.logger.error(f"Dataset directory not found: {self.data_dir}")
            return False
        
        try:
            # Load species data (prefer species.tsv, fallback to genera.tsv)
            species_file = self.data_dir / "species.tsv"
            if not species_file.exists():
                species_file = self.data_dir / "genera.tsv"
            
            if species_file.exists():
                self.species_raw = pd.read_csv(
                    species_file,
                    sep='\t',
                    index_col=0,
                    low_memory=False
                )
                self.logger.info(f"  Loaded microbiome: {self.species_raw.shape}")
            else:
                self.logger.warning(f"  No microbiome data found")
                return False
            
            # Load metabolite data
            mtb_file = self.data_dir / "mtb.tsv"
            if mtb_file.exists():
                self.metabolites_raw = pd.read_csv(
                    mtb_file,
                    sep='\t',
                    index_col=0,
                    low_memory=False
                )
                self.logger.info(f"  Loaded metabolites: {self.metabolites_raw.shape}")
            else:
                self.logger.warning(f"  No metabolite data found")
                return False
            
            # Load metadata
            metadata_file = self.data_dir / "metadata.tsv"
            if metadata_file.exists():
                self.metadata = pd.read_csv(
                    metadata_file,
                    sep='\t',
                    low_memory=False
                )
                # Set Sample as index
                if 'Sample' in self.metadata.columns:
                    self.metadata = self.metadata.set_index('Sample')
                self.logger.info(f"  Loaded metadata: {self.metadata.shape}")
            else:
                self.logger.warning(f"  No metadata found - disease stratification unavailable")
            
            # Find common samples
            self.common_samples = list(
                set(self.species_raw.index) & set(self.metabolites_raw.index)
            )
            
            if len(self.common_samples) == 0:
                self.logger.error(f"  No common samples between microbiome and metabolites")
                return False
            
            # Subset to common samples immediately (reduce memory)
            self.species_raw = self.species_raw.loc[self.common_samples]
            self.metabolites_raw = self.metabolites_raw.loc[self.common_samples]
            
            self.logger.info(f"  Common samples: {len(self.common_samples)}")
            return True
            
        except Exception as e:
            self.logger.error(f"  Error loading data: {e}")
            return False
    
    def get_data_type(self) -> str:
        """Determine if this is shotgun or 16S data based on presence of species.tsv.
        
        Returns:
            str: 'shotgun' or '16s'
        """
        species_file = self.data_dir / "species.tsv"
        return 'shotgun' if species_file.exists() else '16s'
    
    def estimate_memory_usage(self) -> dict:
        """Estimate current memory usage of loaded data.
        
        Returns:
            dict: Memory usage in MB for each component
        """
        usage = {}
        
        if self.species_raw is not None:
            usage['species_raw_mb'] = self.species_raw.memory_usage(deep=True).sum() / 1024**2
        
        if self.metabolites_raw is not None:
            usage['metabolites_raw_mb'] = self.metabolites_raw.memory_usage(deep=True).sum() / 1024**2
        
        if self.species_processed is not None:
            usage['species_processed_mb'] = self.species_processed.memory_usage(deep=True).sum() / 1024**2
        
        if self.metabolites_processed is not None:
            usage['metabolites_processed_mb'] = self.metabolites_processed.memory_usage(deep=True).sum() / 1024**2
        
        usage['total_mb'] = sum(usage.values())
        return usage
    
    def stratify_samples_by_disease(self) -> bool:
        """Stratify samples into control and disease groups based on metadata.
        
        Returns:
            bool: True if stratification successful
        """
        if self.metadata is None:
            self.logger.warning(f"  No metadata available for stratification")
            return False
        
        if 'Study.Group' not in self.metadata.columns:
            self.logger.warning(f"  'Study.Group' column not found in metadata")
            return False
        
        try:
            from scripts.utils.disease_groups import get_sample_groups
            
            # Get control and disease sample lists
            sample_groups = get_sample_groups(
                self.metadata.reset_index(),
                self.name,
                sample_column='Sample'
            )
            
            # Filter to common samples (those with both microbiome and metabolome data)
            self.control_samples = [
                s for s in sample_groups['control'] 
                if s in self.common_samples
            ]
            self.disease_samples = [
                s for s in sample_groups['disease'] 
                if s in self.common_samples
            ]
            
            self.logger.info(f"  Stratified samples: {len(self.control_samples)} control, {len(self.disease_samples)} disease")
            
            return len(self.control_samples) > 0 and len(self.disease_samples) > 0
            
        except Exception as e:
            self.logger.error(f"  Error stratifying samples: {e}")
            return False
    
    def get_stratified_data(self, group: str = 'control'):
        """Get microbiome and metabolome data for a specific group.
        
        Args:
            group: 'control' or 'disease'
            
        Returns:
            tuple: (species_df, metabolites_df) for the specified group
        """
        if group == 'control':
            samples = self.control_samples
        elif group == 'disease':
            samples = self.disease_samples
        else:
            raise ValueError(f"Group must be 'control' or 'disease', got '{group}'")
        
        if samples is None or len(samples) == 0:
            return None, None
        
        # Subset to group samples
        species_subset = self.species_processed.loc[samples] if self.species_processed is not None else None
        metabolites_subset = self.metabolites_processed.loc[samples] if self.metabolites_processed is not None else None
        
        return species_subset, metabolites_subset
    
    def clear_raw_data(self):
        """Free memory by clearing raw data after processing."""
        self.species_raw = None
        self.metabolites_raw = None
        self.logger.debug(f"  Cleared raw data from memory")
    
    def to_sparse_network(self, associations_df: pd.DataFrame, min_abs_corr: float = 0.3) -> sparse.coo_matrix:
        """Convert association dataframe to sparse adjacency matrix.
        
        Args:
            associations_df: DataFrame with species, metabolite, rho columns
            min_abs_corr: Minimum absolute correlation to include
            
        Returns:
            scipy.sparse.coo_matrix: Sparse adjacency matrix
        """
        # Filter by correlation strength
        sig_assoc = associations_df[associations_df['rho'].abs() >= min_abs_corr].copy()
        
        if len(sig_assoc) == 0:
            self.logger.warning(f"  No associations above |rho| >= {min_abs_corr}")
            return None
        
        # Create node indices
        all_species = sig_assoc['species'].unique()
        all_metabolites = sig_assoc['metabolite'].unique()
        
        # Create mapping: node name -> index
        node_to_idx = {}
        idx = 0
        for species in all_species:
            node_to_idx[species] = idx
            idx += 1
        for metab in all_metabolites:
            node_to_idx[metab] = idx
            idx += 1
        
        n_nodes = len(node_to_idx)
        
        # Build sparse matrix
        row_indices = []
        col_indices = []
        data = []
        
        for _, row in sig_assoc.iterrows():
            species_idx = node_to_idx[row['species']]
            metab_idx = node_to_idx[row['metabolite']]
            
            row_indices.append(species_idx)
            col_indices.append(metab_idx)
            data.append(row['rho'])
            
            # Symmetric (undirected graph)
            row_indices.append(metab_idx)
            col_indices.append(species_idx)
            data.append(row['rho'])
        
        adjacency = sparse.coo_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_nodes, n_nodes),
            dtype=np.float32
        )
        
        self.network_adjacency = adjacency
        
        # Store node mapping for later use
        self.node_mapping = {
            'node_to_idx': node_to_idx,
            'idx_to_node': {v: k for k, v in node_to_idx.items()},
            'species_nodes': list(all_species),
            'metabolite_nodes': list(all_metabolites)
        }
        
        self.logger.info(f"  Created sparse network: {n_nodes} nodes, {len(data)//2} edges")
        self.logger.info(f"  Sparsity: {100 * (1 - len(data) / (n_nodes * n_nodes)):.2f}%")
        
        return adjacency
