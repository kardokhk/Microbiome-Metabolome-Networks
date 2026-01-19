"""Optimized computation of species-metabolite associations using Spearman correlation."""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


class AssociationComputer:
    """Computes correlations between species and metabolites efficiently."""
    
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
    
    def compute_spearman_chunk(self, species_chunk: np.ndarray, metabolites: np.ndarray,
                              species_names: List[str], metabolite_names: List[str]) -> List[dict]:
        """Compute Spearman correlations for a chunk of species.
        
        This function is designed to be called in parallel processes.
        
        Args:
            species_chunk: Array of shape (n_samples, n_species_in_chunk)
            metabolites: Array of shape (n_samples, n_metabolites)
            species_names: List of species names in this chunk
            metabolite_names: List of all metabolite names
            
        Returns:
            List of dicts with species, metabolite, rho, pvalue
        """
        results = []
        
        n_species_chunk = species_chunk.shape[1]
        n_metabolites = metabolites.shape[1]
        
        for i in range(n_species_chunk):
            species_values = species_chunk[:, i]
            
            # Skip if constant
            if np.std(species_values) == 0:
                continue
            
            for j in range(n_metabolites):
                metab_values = metabolites[:, j]
                
                # Skip if constant
                if np.std(metab_values) == 0:
                    continue
                
                try:
                    rho, pval = spearmanr(species_values, metab_values)
                    
                    if not np.isnan(rho) and not np.isnan(pval):
                        results.append({
                            'species': species_names[i],
                            'metabolite': metabolite_names[j],
                            'rho': rho,
                            'pvalue': pval
                        })
                except:
                    continue
        
        return results
    
    def compute_associations_parallel(self, species_df: pd.DataFrame, 
                                     metabolites_df: pd.DataFrame,
                                     max_workers: int = 8) -> pd.DataFrame:
        """Compute all species-metabolite correlations using parallel processing.
        
        Args:
            species_df: CLR-transformed species (samples × species)
            metabolites_df: Z-scored metabolites (samples × metabolites)
            max_workers: Maximum number of parallel processes
            
        Returns:
            pd.DataFrame: Significant associations with FDR correction
        """
        n_species = species_df.shape[1]
        n_metabolites = metabolites_df.shape[1]
        total_tests = n_species * n_metabolites
        
        self.logger.info(
            f"  Computing {n_species} species × {n_metabolites} metabolites = "
            f"{total_tests:,} correlations"
        )
        
        # Convert to numpy for faster computation
        species_array = species_df.values
        metabolites_array = metabolites_df.values
        species_names = species_df.columns.tolist()
        metabolite_names = metabolites_df.columns.tolist()
        
        # Determine optimal chunk size
        # Aim for ~50-100 chunks to balance overhead vs parallelism
        target_chunks = max(50, min(100, max_workers * 10))
        chunk_size = max(1, n_species // target_chunks)
        
        # Create chunks
        chunks = []
        for i in range(0, n_species, chunk_size):
            end_idx = min(i + chunk_size, n_species)
            chunk_data = species_array[:, i:end_idx]
            chunk_names = species_names[i:end_idx]
            chunks.append((chunk_data, metabolites_array, chunk_names, metabolite_names))
        
        self.logger.info(f"  Processing {len(chunks)} chunks with {max_workers} workers")
        
        # Parallel computation
        all_results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    self.compute_spearman_chunk,
                    chunk_data, metab_array, chunk_names, metab_names
                ): i
                for i, (chunk_data, metab_array, chunk_names, metab_names) in enumerate(chunks)
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                chunk_results = future.result()
                all_results.extend(chunk_results)
                completed += 1
                
                if completed % 10 == 0 or completed == len(chunks):
                    self.logger.info(f"    Progress: {completed}/{len(chunks)} chunks completed")
        
        if len(all_results) == 0:
            self.logger.error(f"  No associations computed!")
            return None
        
        # Create DataFrame
        assoc_df = pd.DataFrame(all_results)
        self.logger.info(f"  Computed {len(assoc_df):,} raw associations")
        
        # FDR correction (Benjamini-Hochberg)
        self.logger.info(f"  Applying FDR correction...")
        _, qvals, _, _ = multipletests(assoc_df['pvalue'], method='fdr_bh')
        assoc_df['qvalue'] = qvals
        
        # Filter by FDR threshold
        fdr_threshold = self.config['FDR_THRESHOLD']
        assoc_sig = assoc_df[assoc_df['qvalue'] < fdr_threshold].copy()
        
        self.logger.info(
            f"  Significant associations (FDR < {fdr_threshold}): {len(assoc_sig):,} "
            f"({100 * len(assoc_sig) / len(assoc_df):.2f}%)"
        )
        
        # Summary statistics
        if len(assoc_sig) > 0:
            n_positive = (assoc_sig['rho'] > 0).sum()
            n_negative = (assoc_sig['rho'] < 0).sum()
            mean_abs_rho = assoc_sig['rho'].abs().mean()
            
            self.logger.info(
                f"    Positive: {n_positive:,} ({100*n_positive/len(assoc_sig):.1f}%), "
                f"Negative: {n_negative:,} ({100*n_negative/len(assoc_sig):.1f}%)"
            )
            self.logger.info(f"    Mean |ρ|: {mean_abs_rho:.3f}")
        
        return assoc_sig
    
    def get_association_summary(self, assoc_df: pd.DataFrame, dataset_name: str,
                               n_species: int, n_metabolites: int) -> dict:
        """Generate summary statistics for associations.
        
        Returns:
            dict: Summary statistics
        """
        if assoc_df is None or len(assoc_df) == 0:
            return {
                'dataset': dataset_name,
                'n_species': n_species,
                'n_metabolites': n_metabolites,
                'n_total_tests': n_species * n_metabolites,
                'n_significant': 0,
                'mean_abs_rho': 0,
                'n_positive': 0,
                'n_negative': 0
            }
        
        return {
            'dataset': dataset_name,
            'n_species': n_species,
            'n_metabolites': n_metabolites,
            'n_total_tests': n_species * n_metabolites,
            'n_significant': len(assoc_df),
            'mean_abs_rho': assoc_df['rho'].abs().mean(),
            'median_abs_rho': assoc_df['rho'].abs().median(),
            'n_positive': (assoc_df['rho'] > 0).sum(),
            'n_negative': (assoc_df['rho'] < 0).sum(),
            'min_qvalue': assoc_df['qvalue'].min(),
            'max_abs_rho': assoc_df['rho'].abs().max()
        }
