"""Data loading utilities."""

import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from scripts.utils.logger import setup_logger

logger = setup_logger("data_loader")

def load_dataset(dataset_name, data_dir, low_memory=True):
    """Load species, metabolites, and metadata for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Directory containing datasets
        low_memory: If True, use memory-efficient loading with dtypes optimization
    """
    dataset_dir = Path(data_dir) / dataset_name
    
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return None, None, None
    
    try:
        # Memory-efficient loading options
        read_opts = {
            'sep': '\t',
            'index_col': 0,
            'low_memory': low_memory
        }
        if low_memory:
            read_opts['dtype_backend'] = 'numpy_nullable'
        
        # Load species data (samples already as rows in the file)
        species_file = dataset_dir / "species.tsv"
        if species_file.exists():
            species = pd.read_csv(species_file, **read_opts)
            # Data is already in correct format: samples as rows, species as columns
            logger.info(f"Loaded species data: {species.shape}")
        else:
            logger.warning(f"No species data found for {dataset_name}")
            species = None
        
        # Load metabolite data (samples already as rows in the file)
        mtb_file = dataset_dir / "mtb.tsv"
        if mtb_file.exists():
            metabolites = pd.read_csv(mtb_file, **read_opts)
            # Data is already in correct format: samples as rows, metabolites as columns
            logger.info(f"Loaded metabolite data: {metabolites.shape}")
        else:
            logger.warning(f"No metabolite data found for {dataset_name}")
            metabolites = None
        
        # Load metadata
        metadata_file = dataset_dir / "metadata.tsv"
        if metadata_file.exists():
            metadata = pd.read_csv(metadata_file, sep='\t', index_col=1, low_memory=low_memory)  # Sample as index
            logger.info(f"Loaded metadata: {metadata.shape}")
        else:
            logger.warning(f"No metadata found for {dataset_name}")
            metadata = None
        
        return species, metabolites, metadata
    
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        return None, None, None

def get_common_samples(species, metabolites):
    """Get samples present in both species and metabolite data."""
    if species is None or metabolites is None:
        return []
    
    common = list(set(species.index) & set(metabolites.index))
    logger.info(f"Found {len(common)} common samples")
    return common
