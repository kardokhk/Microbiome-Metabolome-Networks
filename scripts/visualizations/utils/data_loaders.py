"""Data loading utilities for visualization scripts."""

import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / 'output' / 'results'
DATA_DIR = PROJECT_ROOT / 'data'

# Dataset information
DATASETS = [
    'YACHIDA_CRC_2019',
    'ERAWIJANTARI_GASTRIC_CANCER_2020',
    'FRANZOSA_IBD_2019',
    'iHMP_IBDMDB_2019',
    'MARS_IBS_2020',
    'WANG_ESRD_2020'
]

DISEASE_NAMES = {
    'YACHIDA_CRC_2019': 'Colorectal Cancer',
    'ERAWIJANTARI_GASTRIC_CANCER_2020': 'Gastric Cancer',
    'FRANZOSA_IBD_2019': 'IBD',
    'iHMP_IBDMDB_2019': 'IBD (HMP2)',
    'MARS_IBS_2020': 'IBS',
    'WANG_ESRD_2020': 'End-Stage Renal Disease'
}


def load_dataset_info():
    """Load basic information about all datasets."""
    info = []

    for dataset in DATASETS:
        dataset_dir = RESULTS_DIR / dataset

        if not dataset_dir.exists():
            print(f"Warning: {dataset} directory not found")
            continue

        # Try to load summary if available
        summary_file = dataset_dir / 'differential' / 'summary.csv'

        if summary_file.exists():
            summary = pd.read_csv(summary_file, index_col=0)

            info.append({
                'dataset': dataset,
                'disease': DISEASE_NAMES.get(dataset, 'Unknown'),
                'n_control': summary.loc['Total species analyzed', 'control'] if 'control' in summary.columns else None,
                'n_disease': summary.loc['Total species analyzed', 'disease'] if 'disease' in summary.columns else None,
            })
        else:
            # Fallback: try to get info from metadata
            info.append({
                'dataset': dataset,
                'disease': DISEASE_NAMES.get(dataset, 'Unknown'),
                'n_control': None,
                'n_disease': None,
            })

    return pd.DataFrame(info)


def load_network_data(dataset, group='control'):
    """Load network data for a dataset and group.

    Args:
        dataset: Dataset name
        group: 'control' or 'disease'

    Returns:
        dict with network_edges, network_stats, modules, keystones
    """
    dataset_dir = RESULTS_DIR / dataset / group

    if not dataset_dir.exists():
        print(f"Warning: {dataset}/{group} directory not found")
        return None

    data = {}

    # Load network edges
    edges_file = dataset_dir / 'network_edges.csv'
    if edges_file.exists():
        data['edges'] = pd.read_csv(edges_file)

    # Load network stats
    stats_file = dataset_dir / 'network_stats.csv'
    if stats_file.exists():
        data['stats'] = pd.read_csv(stats_file, index_col=0)

    # Load modules
    modules_file = dataset_dir / 'modules.csv'
    if modules_file.exists():
        data['modules'] = pd.read_csv(modules_file)

    # Load module stats
    module_stats_file = dataset_dir / 'module_stats.csv'
    if module_stats_file.exists():
        data['module_stats'] = pd.read_csv(module_stats_file)

    # Load keystones
    keystones_file = dataset_dir / 'keystones.csv'
    if keystones_file.exists():
        data['keystones'] = pd.read_csv(keystones_file)

    # Load top keystones
    top_keystones_file = dataset_dir / 'top_keystones.csv'
    if top_keystones_file.exists():
        data['top_keystones'] = pd.read_csv(top_keystones_file)

    # Load hub metabolites
    hubs_file = dataset_dir / 'hub_metabolites.csv'
    if hubs_file.exists():
        data['hub_metabolites'] = pd.read_csv(hubs_file)

    return data


def load_keystone_data(dataset, group='control'):
    """Load keystone species data.

    Args:
        dataset: Dataset name
        group: 'control' or 'disease'

    Returns:
        DataFrame of keystones
    """
    keystones_file = RESULTS_DIR / dataset / group / 'keystones.csv'

    if keystones_file.exists():
        return pd.read_csv(keystones_file)
    return None


def load_differential_data(dataset):
    """Load differential analysis data.

    Args:
        dataset: Dataset name

    Returns:
        dict with differential_keystones, network_comparison, summary
    """
    diff_dir = RESULTS_DIR / dataset / 'differential'

    if not diff_dir.exists():
        print(f"Warning: {dataset}/differential directory not found")
        return None

    data = {}

    # Load differential keystones
    keystones_file = diff_dir / 'differential_keystones.csv'
    if keystones_file.exists():
        data['keystones'] = pd.read_csv(keystones_file)

    # Load hub metabolites
    hubs_file = diff_dir / 'differential_hub_metabolites.csv'
    if hubs_file.exists():
        data['hub_metabolites'] = pd.read_csv(hubs_file)

    # Load network comparison
    network_file = diff_dir / 'network_comparison.csv'
    if network_file.exists():
        data['network_comparison'] = pd.read_csv(network_file, index_col=0)

    # Load summary
    summary_file = diff_dir / 'summary.csv'
    if summary_file.exists():
        data['summary'] = pd.read_csv(summary_file, index_col=0)

    return data


def load_cross_study_data():
    """Load cross-study analysis data.

    Returns:
        dict with various cross-study results
    """
    cross_dir = RESULTS_DIR / 'cross_study'

    if not cross_dir.exists():
        print("Warning: cross_study directory not found")
        return None

    data = {}

    # Load conserved depleted species (KEY FINDING!)
    depleted_file = cross_dir / 'conserved_depleted_in_disease.csv'
    if depleted_file.exists():
        data['depleted'] = pd.read_csv(depleted_file)

    # Load conserved keystones (control)
    control_keystones = cross_dir / 'control' / 'conserved_keystones.csv'
    if control_keystones.exists():
        data['conserved_control'] = pd.read_csv(control_keystones)

    # Load conserved keystones (disease)
    disease_keystones = cross_dir / 'disease' / 'conserved_keystones.csv'
    if disease_keystones.exists():
        data['conserved_disease'] = pd.read_csv(disease_keystones)

    # Load conserved hub metabolites
    control_hubs = cross_dir / 'control' / 'conserved_hub_metabolites.csv'
    if control_hubs.exists():
        data['control_hubs'] = pd.read_csv(control_hubs)

    disease_hubs = cross_dir / 'disease' / 'conserved_hub_metabolites.csv'
    if disease_hubs.exists():
        data['disease_hubs'] = pd.read_csv(disease_hubs)

    # Load metabolite class data
    metab_summary = cross_dir / 'metabolite_class_summary.csv'
    if metab_summary.exists():
        data['metabolite_summary'] = pd.read_csv(metab_summary)

    return data


def load_cross_disease_data():
    """Load cross-disease comparison data.

    Returns:
        dict with cross-disease results
    """
    cross_dir = RESULTS_DIR / 'cross_disease'

    if not cross_dir.exists():
        print("Warning: cross_disease directory not found")
        return None

    data = {}

    # Load disease stratification
    strat_file = cross_dir / 'disease_stratification.csv'
    if strat_file.exists():
        data['stratification'] = pd.read_csv(strat_file)

    # Load disease-specific markers
    markers_file = cross_dir / 'disease_specific_markers.csv'
    if markers_file.exists():
        data['markers'] = pd.read_csv(markers_file)

    # Load meta-analysis
    meta_file = cross_dir / 'meta_analysis_effect_sizes.csv'
    if meta_file.exists():
        data['meta_analysis'] = pd.read_csv(meta_file)

    # Load network disruption
    disruption_file = cross_dir / 'network_disruption_by_disease.csv'
    if disruption_file.exists():
        data['disruption'] = pd.read_csv(disruption_file)

    # Load disease similarity
    similarity_file = cross_dir / 'disease_similarity_matrix.csv'
    if similarity_file.exists():
        data['similarity'] = pd.read_csv(similarity_file, index_col=0)

    return data


def load_enrichment_data(dataset, group='control'):
    """Load enrichment analysis data.

    Args:
        dataset: Dataset name
        group: 'control' or 'disease'

    Returns:
        dict with enrichment results
    """
    enrich_dir = RESULTS_DIR / dataset / group / 'enrichment'

    if not enrich_dir.exists():
        return None

    data = {}

    # Load taxonomic enrichment
    phylum_file = enrich_dir / 'taxonomic_enrichment_phylum.csv'
    if phylum_file.exists():
        data['phylum'] = pd.read_csv(phylum_file)

    family_file = enrich_dir / 'taxonomic_enrichment_family.csv'
    if family_file.exists():
        data['family'] = pd.read_csv(family_file)

    # Load metabolite class enrichment
    metab_file = enrich_dir / 'metabolite_class_enrichment.csv'
    if metab_file.exists():
        data['metabolite'] = pd.read_csv(metab_file)

    return data


def get_sample_counts():
    """Get sample counts for each dataset from metadata."""
    counts = {}

    for dataset in DATASETS:
        metadata_file = DATA_DIR / dataset / 'metadata.csv'

        if metadata_file.exists():
            try:
                metadata = pd.read_csv(metadata_file, index_col=0)
                counts[dataset] = {
                    'total': len(metadata),
                    'disease': DISEASE_NAMES.get(dataset, 'Unknown')
                }
            except Exception as e:
                print(f"Error loading metadata for {dataset}: {e}")
                counts[dataset] = {'total': None, 'disease': DISEASE_NAMES.get(dataset, 'Unknown')}
        else:
            counts[dataset] = {'total': None, 'disease': DISEASE_NAMES.get(dataset, 'Unknown')}

    return pd.DataFrame(counts).T
