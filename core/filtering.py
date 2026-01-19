"""Intelligent filtering of microbiome and metabolite features."""

import pandas as pd
import numpy as np
from typing import Tuple


class DataFilter:
    """Applies abundance and prevalence filters to reduce dimensionality."""
    
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.dataset_name = None  # Will be set by pipeline
    
    def set_dataset(self, dataset_name: str):
        """Set current dataset name for dataset-specific filtering."""
        self.dataset_name = dataset_name
    
    def get_filter_params(self, param_name: str, data_type: str = 'shotgun'):
        """Get filtering parameter, applying dataset-specific overrides if configured.
        
        Args:
            param_name: Parameter name (e.g., 'METABOLITE_PREVALENCE')
            data_type: 'shotgun' or '16s'
            
        Returns:
            Parameter value (with dataset-specific override if applicable)
        """
        # Check for dataset-specific override
        if self.dataset_name and 'DATASET_SPECIFIC_FILTERS' in self.config:
            dataset_filters = self.config['DATASET_SPECIFIC_FILTERS'].get(self.dataset_name, {})
            if param_name in dataset_filters:
                return dataset_filters[param_name]
        
        # Return default value
        return self.config.get(param_name)
    
    def filter_species(self, species_df: pd.DataFrame, data_type: str = 'shotgun') -> pd.DataFrame:
        """Filter species/genera by prevalence and abundance.
        
        Args:
            species_df: DataFrame with samples as rows, species as columns
            data_type: 'shotgun' or '16s' - determines filtering stringency
            
        Returns:
            pd.DataFrame: Filtered species data
        """
        original_shape = species_df.shape
        
        # Get filtering parameters based on data type
        if data_type == 'shotgun':
            prevalence_threshold = self.config['SPECIES_PREVALENCE_SHOTGUN']
            min_abundance = self.config['SPECIES_MIN_ABUNDANCE']
        else:
            prevalence_threshold = self.config['SPECIES_PREVALENCE_16S']
            min_abundance = self.config['SPECIES_MIN_ABUNDANCE'] * 10  # Less stringent for 16S
        
        # 1. Prevalence filter: present in >= threshold proportion of samples
        prevalence = (species_df > 0).sum(axis=0) / len(species_df)
        keep_by_prevalence = prevalence >= prevalence_threshold
        
        # 2. Abundance filter: max abundance >= threshold in at least one sample
        max_abundance = species_df.max(axis=0)
        keep_by_abundance = max_abundance >= min_abundance
        
        # Combine filters
        keep_features = keep_by_prevalence & keep_by_abundance
        species_filtered = species_df.loc[:, keep_features]
        
        # 3. Remove constant columns (all same value)
        constant_cols = species_filtered.columns[species_filtered.std() == 0]
        if len(constant_cols) > 0:
            species_filtered = species_filtered.drop(columns=constant_cols)
            self.logger.debug(f"    Removed {len(constant_cols)} constant features")
        
        self.logger.info(
            f"    Species: {original_shape[1]} → {species_filtered.shape[1]} "
            f"(prevalence>={prevalence_threshold:.0%}, abundance>={min_abundance:.2e})"
        )
        
        return species_filtered
    
    def filter_metabolites(self, metabolites_df: pd.DataFrame, data_type: str = 'shotgun') -> pd.DataFrame:
        """Filter metabolites by missingness and prevalence.
        
        Args:
            metabolites_df: DataFrame with samples as rows, metabolites as columns
            data_type: 'shotgun' or '16s' - determines filtering stringency
            
        Returns:
            pd.DataFrame: Filtered metabolite data
        """
        original_shape = metabolites_df.shape
        
        # Convert to numeric, coercing errors to NaN
        metabolites_df = metabolites_df.apply(pd.to_numeric, errors='coerce')
        
        # Get filtering parameters (with dataset-specific overrides)
        max_missing = self.get_filter_params('METABOLITE_MAX_MISSING', data_type)
        prevalence_threshold = self.get_filter_params('METABOLITE_PREVALENCE', data_type)
        max_metabolites = self.get_filter_params('MAX_METABOLITES', data_type)
        
        # 1. Filter by missingness: drop if > max_missing proportion missing
        missing_proportion = metabolites_df.isnull().sum() / len(metabolites_df)
        keep_by_missing = missing_proportion <= max_missing
        
        # 2. Filter by prevalence: present (non-zero, non-null) in >= threshold samples
        present = (metabolites_df.notna()) & (metabolites_df != 0)
        prevalence = present.sum(axis=0) / len(metabolites_df)
        keep_by_prevalence = prevalence >= prevalence_threshold
        
        # Combine filters
        keep_features = keep_by_missing & keep_by_prevalence
        metabolites_filtered = metabolites_df.loc[:, keep_features]
        
        # 3. Remove constant columns
        constant_cols = metabolites_filtered.columns[metabolites_filtered.std() == 0]
        if len(constant_cols) > 0:
            metabolites_filtered = metabolites_filtered.drop(columns=constant_cols)
            self.logger.debug(f"    Removed {len(constant_cols)} constant metabolites")
        
        # 4. Apply max_metabolites filter if specified (keep top N by variance)
        if max_metabolites and metabolites_filtered.shape[1] > max_metabolites:
            # Calculate variance for each metabolite
            variances = metabolites_filtered.var(axis=0)
            # Keep top N most variable metabolites
            top_metabolites = variances.nlargest(max_metabolites).index
            n_before = metabolites_filtered.shape[1]
            metabolites_filtered = metabolites_filtered[top_metabolites]
            self.logger.info(
                f"    Applied variance filter: {n_before} → {max_metabolites} "
                f"(kept top {max_metabolites} most variable)"
            )
        
        self.logger.info(
            f"    Metabolites: {original_shape[1]} → {metabolites_filtered.shape[1]} "
            f"(missing<={max_missing:.0%}, prevalence>={prevalence_threshold:.0%})"
        )
        
        return metabolites_filtered
    
    def get_filter_stats(self, species_before: pd.DataFrame, species_after: pd.DataFrame,
                        metab_before: pd.DataFrame, metab_after: pd.DataFrame) -> dict:
        """Calculate filtering statistics for logging.
        
        Returns:
            dict: Statistics about filtering operation
        """
        stats = {
            'species_before': species_before.shape[1],
            'species_after': species_after.shape[1],
            'species_reduction_pct': 100 * (1 - species_after.shape[1] / species_before.shape[1]),
            'metabolites_before': metab_before.shape[1],
            'metabolites_after': metab_after.shape[1],
            'metabolites_reduction_pct': 100 * (1 - metab_after.shape[1] / metab_before.shape[1]),
            'total_features_before': species_before.shape[1] + metab_before.shape[1],
            'total_features_after': species_after.shape[1] + metab_after.shape[1],
            'potential_correlations_before': species_before.shape[1] * metab_before.shape[1],
            'potential_correlations_after': species_after.shape[1] * metab_after.shape[1],
        }
        
        if stats['potential_correlations_before'] > 0:
            stats['correlation_reduction_pct'] = 100 * (
                1 - stats['potential_correlations_after'] / stats['potential_correlations_before']
            )
        else:
            stats['correlation_reduction_pct'] = 0
        
        return stats
