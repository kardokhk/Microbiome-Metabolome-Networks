"""Configuration for optimized analysis pipeline."""

from pathlib import Path

# Directories
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
RESULTS_DIR = OUTPUT_DIR / "results"
LOG_DIR = BASE_DIR / "logs"

# All 6 shotgun metagenomics datasets (have species-level data)
SHOTGUN_DATASETS = [
    "YACHIDA_CRC_2019",
    "ERAWIJANTARI_GASTRIC_CANCER_2020",
    "FRANZOSA_IBD_2019",
    "iHMP_IBDMDB_2019",
    "MARS_IBS_2020",
    "WANG_ESRD_2020"
]

# 16S rRNA datasets (genera-level only) - optional for future expansion
AMPLICON_DATASETS = [
    "SINHA_CRC_2016",
    "HE_INFANTS_MFGM_2019",
    "JACOBS_IBD_FAMILIES_2016",
    "POYET_BIO_ML_2019",
    "KANG_AUTISM_2017",
    "KOSTIC_INFANTS_DIABETES_2015",
    "WANDRO_PRETERMS_2018",
    "KIM_ADENOMAS_2020"
]

# Datasets to analyze (currently only shotgun)
DATASETS = SHOTGUN_DATASETS

# Filtering parameters - SHOTGUN datasets (more aggressive due to high dimensionality)
SPECIES_PREVALENCE_SHOTGUN = 0.10   # Present in ≥10% of samples (not 5%)
SPECIES_MIN_ABUNDANCE = 1e-5        # Minimum relative abundance threshold
METABOLITE_PREVALENCE = 0.15        # Present in ≥15% of samples
METABOLITE_MAX_MISSING = 0.50       # Drop if >50% missing (not 30%)

# Filtering parameters - 16S datasets (less aggressive, smaller feature space)
SPECIES_PREVALENCE_16S = 0.05       # Present in ≥5% of samples
METABOLITE_PREVALENCE_16S = 0.15    # Same as shotgun

# Dataset-specific overrides for extremely high-dimensional datasets
# FRANZOSA_IBD_2019 has 8,848 metabolites (4 complementary LC-MS methods)
# These overrides reduce to manageable size while preserving signal
DATASET_SPECIFIC_FILTERS = {
    'FRANZOSA_IBD_2019': {
        'METABOLITE_PREVALENCE': 0.15,      # Increase from 0.15 to 0.40 (present in ≥40% samples)
        'METABOLITE_MAX_MISSING': 0.50,     # Decrease from 0.50 to 0.30 (drop if >30% missing)
        'MAX_METABOLITES': None,            # Keep only top 2000 most variable metabolites
        'REASON': 'Untargeted LC-MS with 8,848 features causes 30x more correlations than other datasets'
    },
    'iHMP_IBDMDB_2019': {
        'METABOLITE_PREVALENCE': 0.15,      # Slightly stricter (10,000+ features)
        'METABOLITE_MAX_MISSING': 0.50,
        'MAX_METABOLITES': None,
        'REASON': 'Also uses untargeted LC-MS with very high dimensionality'
    }
}

# Analysis parameters
FDR_THRESHOLD = 0.1                 # q-value threshold for significant associations
MIN_CORRELATION = 0.3               # Minimum absolute correlation for network edges
MIN_MODULE_SIZE = 5                 # Minimum nodes per module (filter small modules)
KNN_NEIGHBORS = 5                   # For KNN imputation of missing metabolite values

# Parallel processing (max 8 threads as requested)
MAX_WORKERS = 20                     # Maximum parallel workers for CPU-bound tasks
N_JOBS = 20                          # For compatibility with old scripts

# Random seed for reproducibility
RANDOM_STATE = 42

# Configuration dictionary for core modules
CONFIG = {
    # Filtering
    'SPECIES_PREVALENCE_SHOTGUN': SPECIES_PREVALENCE_SHOTGUN,
    'SPECIES_PREVALENCE_16S': SPECIES_PREVALENCE_16S,
    'SPECIES_MIN_ABUNDANCE': SPECIES_MIN_ABUNDANCE,
    'METABOLITE_PREVALENCE': METABOLITE_PREVALENCE,
    'METABOLITE_MAX_MISSING': METABOLITE_MAX_MISSING,
    'DATASET_SPECIFIC_FILTERS': DATASET_SPECIFIC_FILTERS,
    
    # Associations
    'FDR_THRESHOLD': FDR_THRESHOLD,
    'MIN_CORRELATION': MIN_CORRELATION,
    
    # Modules
    'MIN_MODULE_SIZE': MIN_MODULE_SIZE,
    
    # Transformations
    'KNN_NEIGHBORS': KNN_NEIGHBORS,
    
    # Parallel
    'MAX_WORKERS': MAX_WORKERS,
    
    # Misc
    'RANDOM_STATE': RANDOM_STATE
}
