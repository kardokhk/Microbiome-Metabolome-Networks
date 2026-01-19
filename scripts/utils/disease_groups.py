"""Disease group definitions for all datasets.

Each dataset has control and disease groups defined.
The pipeline will build separate networks for each group and compare them.
"""

# Disease group mappings for each dataset
# Format: {dataset_name: {'control': [...], 'disease': [...]}}

DISEASE_GROUP_DEFINITIONS = {
    'YACHIDA_CRC_2019': {
        'control': ['Healthy'],
        'disease': ['Stage_0', 'Stage_I_II', 'Stage_III_IV', 'HS', 'MP'],
        'disease_name': 'Colorectal Cancer'
    },
    
    'FRANZOSA_IBD_2019': {
        'control': ['Control'],
        'disease': ['CD', 'UC'],  # Crohn's Disease and Ulcerative Colitis
        'disease_name': 'Inflammatory Bowel Disease'
    },
    
    'MARS_IBS_2020': {
        'control': ['H'],  # Healthy
        'disease': ['C', 'D'],  # IBS-Constipation and IBS-Diarrhea
        'disease_name': 'Irritable Bowel Syndrome'
    },
    
    'iHMP_IBDMDB_2019': {
        'control': ['nonIBD'],
        'disease': ['CD', 'UC'],  # Crohn's Disease and Ulcerative Colitis
        'disease_name': 'Inflammatory Bowel Disease'
    },
    
    'WANG_ESRD_2020': {
        'control': ['Control'],
        'disease': ['ESRD'],  # End-Stage Renal Disease
        'disease_name': 'End-Stage Renal Disease'
    },
    
    'ERAWIJANTARI_GASTRIC_CANCER_2020': {
        'control': ['Healthy'],
        'disease': ['Gastrectomy'],  # Post-gastrectomy (gastric cancer)
        'disease_name': 'Gastric Cancer'
    }
}


def get_disease_groups(dataset_name: str) -> dict:
    """Get control and disease group definitions for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        dict: Dictionary with 'control', 'disease', and 'disease_name' keys
        
    Raises:
        ValueError: If dataset not found in definitions
    """
    if dataset_name not in DISEASE_GROUP_DEFINITIONS:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in disease group definitions. "
            f"Available datasets: {list(DISEASE_GROUP_DEFINITIONS.keys())}"
        )
    
    return DISEASE_GROUP_DEFINITIONS[dataset_name]


def is_control_sample(dataset_name: str, study_group: str) -> bool:
    """Check if a sample belongs to the control group.
    
    Args:
        dataset_name: Name of the dataset
        study_group: Study group value from metadata
        
    Returns:
        bool: True if sample is a control, False otherwise
    """
    groups = get_disease_groups(dataset_name)
    return study_group in groups['control']


def is_disease_sample(dataset_name: str, study_group: str) -> bool:
    """Check if a sample belongs to a disease group.
    
    Args:
        dataset_name: Name of the dataset
        study_group: Study group value from metadata
        
    Returns:
        bool: True if sample is a disease case, False otherwise
    """
    groups = get_disease_groups(dataset_name)
    return study_group in groups['disease']


def get_sample_groups(metadata_df, dataset_name: str, sample_column: str = 'Sample') -> dict:
    """Split metadata into control and disease sample lists.
    
    Args:
        metadata_df: Metadata DataFrame with Study.Group column
        dataset_name: Name of the dataset
        sample_column: Column name containing sample IDs
        
    Returns:
        dict: {'control': [sample_ids], 'disease': [sample_ids]}
    """
    if metadata_df is None or len(metadata_df) == 0:
        return {'control': [], 'disease': []}
    
    if 'Study.Group' not in metadata_df.columns:
        raise ValueError(f"'Study.Group' column not found in metadata for {dataset_name}")
    
    groups = get_disease_groups(dataset_name)
    
    # Get control samples
    control_mask = metadata_df['Study.Group'].isin(groups['control'])
    control_samples = metadata_df.loc[control_mask, sample_column].tolist()
    
    # Get disease samples
    disease_mask = metadata_df['Study.Group'].isin(groups['disease'])
    disease_samples = metadata_df.loc[disease_mask, sample_column].tolist()
    
    return {
        'control': control_samples,
        'disease': disease_samples
    }
