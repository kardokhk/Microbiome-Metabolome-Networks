"""Data transformations: CLR for compositional data, z-score for metabolites."""

import pandas as pd
import numpy as np
from scipy.stats import gmean
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


class DataTransformer:
    """Handles normalization and transformation of multi-omics data."""
    
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
    
    def clr_transform(self, species_df: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
        """Center log-ratio transformation for compositional microbiome data.
        
        Args:
            species_df: DataFrame with samples as rows, species as columns
            pseudocount: Small value to add before log transform
            
        Returns:
            pd.DataFrame: CLR-transformed data
        """
        # Add pseudocount to avoid log(0)
        species_pseudo = species_df + pseudocount
        
        # Compute geometric mean per sample (row-wise)
        # Convert to float64 for scipy compatibility
        data_array = np.asarray(species_pseudo.values, dtype=np.float64)
        geom_means = gmean(data_array, axis=1)
        
        # CLR: log(x_i / geometric_mean)
        clr_data = np.log(species_pseudo.divide(geom_means, axis=0))
        
        clr_df = pd.DataFrame(
            clr_data,
            index=species_df.index,
            columns=species_df.columns
        )
        
        self.logger.info(
            f"    CLR transform: mean={clr_df.mean().mean():.3f}, "
            f"std={clr_df.std().mean():.3f}"
        )
        
        return clr_df
    
    def standardize_species_names(self, species_df: pd.DataFrame) -> pd.DataFrame:
        """Standardize species/genera names: lowercase, remove underscores.
        
        Args:
            species_df: DataFrame with species as columns
            
        Returns:
            pd.DataFrame: DataFrame with standardized column names
        """
        new_columns = {}
        for col in species_df.columns:
            # Lowercase, replace underscores with spaces, strip whitespace
            standardized = str(col).lower().replace('_', ' ').strip()
            new_columns[col] = standardized
        
        species_df = species_df.rename(columns=new_columns)
        self.logger.debug(f"    Standardized {len(new_columns)} species names")
        
        return species_df
    
    def impute_and_zscore_metabolites(self, metabolites_df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values with KNN and apply z-score normalization.
        
        Args:
            metabolites_df: DataFrame with samples as rows, metabolites as columns
            
        Returns:
            pd.DataFrame: Imputed and z-scored data
        """
        # Check for missing values
        missing_count = metabolites_df.isnull().sum().sum()
        
        if missing_count > 0:
            # KNN imputation
            n_neighbors = min(self.config['KNN_NEIGHBORS'], len(metabolites_df) - 1)
            imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
            
            metabolites_imputed = pd.DataFrame(
                imputer.fit_transform(metabolites_df),
                index=metabolites_df.index,
                columns=metabolites_df.columns
            )
            
            self.logger.info(f"    Imputed {missing_count} missing values (KNN, k={n_neighbors})")
        else:
            metabolites_imputed = metabolites_df
            self.logger.debug(f"    No missing values to impute")
        
        # Z-score standardization
        scaler = StandardScaler()
        metabolites_scaled = pd.DataFrame(
            scaler.fit_transform(metabolites_imputed),
            index=metabolites_imputed.index,
            columns=metabolites_imputed.columns
        )
        
        self.logger.info(
            f"    Z-score: mean={metabolites_scaled.mean().mean():.3f}, "
            f"std={metabolites_scaled.std().mean():.3f}"
        )
        
        return metabolites_scaled
    
    def transform_pipeline(self, species_df: pd.DataFrame, metabolites_df: pd.DataFrame,
                          data_type: str = 'shotgun') -> tuple:
        """Complete transformation pipeline for both data types.
        
        Args:
            species_df: Raw species data
            metabolites_df: Raw metabolite data
            data_type: 'shotgun' or '16s'
            
        Returns:
            tuple: (transformed_species, transformed_metabolites)
        """
        self.logger.info(f"  Transforming data...")
        
        # Standardize species names
        species_df = self.standardize_species_names(species_df)
        
        # CLR transform species
        species_clr = self.clr_transform(species_df)
        
        # Impute and z-score metabolites
        metabolites_zscore = self.impute_and_zscore_metabolites(metabolites_df)
        
        return species_clr, metabolites_zscore
