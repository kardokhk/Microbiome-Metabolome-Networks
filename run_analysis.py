#!/usr/bin/env python3
"""
Complete microbiome-metabolome network analysis pipeline.

This script runs the entire analysis workflow:
1. Disease-stratified network analysis (control vs disease)
2. Hub metabolite identification  
3. Cross-study convergence analysis
4. Module enrichment analysis
5. Cross-disease comparison
6. Enhanced hub metabolite analysis
7. Metabolite functional annotation

Usage:
    python run_analysis.py

Outputs:
    - output/results/{dataset}/control/ - Control network results
    - output/results/{dataset}/disease/ - Disease network results
    - output/results/{dataset}/differential/ - Differential analysis
    - output/results/cross_study/ - Cross-study convergence
    - output/results/cross_disease/ - Cross-disease comparison
    - logs/ - Comprehensive log files
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.config import *
from scripts.utils.disease_groups import get_disease_groups
from core.logger import setup_logger
from core.data_handler import DatasetHandler
from core.filtering import DataFilter
from core.transformations import DataTransformer
from core.associations import AssociationComputer
from core.networks import NetworkBuilder
from core.modules import ModuleDetector
from core.analysis import KeystoneAnalyzer
from core.differential import DifferentialAnalyzer

# Import analysis modules
from scripts.hub_metabolite_analysis import HubMetaboliteAnalyzer
from scripts.module_enrichment_analysis import ModuleEnrichmentAnalyzer
from scripts.cross_disease_comparison import CrossDiseaseComparator
from scripts.enhanced_hub_metabolite_analysis import EnhancedHubMetaboliteAnalyzer
from scripts.metabolite_annotation import MetaboliteAnnotator


class ComprehensiveAnalysisPipeline:
    """Complete analysis pipeline with all downstream analyses."""
    
    def __init__(self, datasets, output_dir, log_dir, config):
        self.datasets = datasets
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.config = config
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize logger
        self.logger = setup_logger(self.log_dir)
        
        # Initialize core components
        self.filter = DataFilter(self.logger, config)
        self.transformer = DataTransformer(self.logger, config)
        self.associator = AssociationComputer(self.logger, config)
        self.networker = NetworkBuilder(self.logger, config)
        self.module_detector = ModuleDetector(self.logger, config)
        self.analyzer = KeystoneAnalyzer(self.logger, config)
        self.diff_analyzer = DifferentialAnalyzer(self.logger, config)
        
        # Storage for results (separated by control/disease)
        self.results = {}
        self.all_keystones_control = {}
        self.all_keystones_disease = {}
        self.all_differential_keystones = {}
    
    def process_group(self, handler: DatasetHandler, group: str, dataset_name: str):
        """Process a single group (control or disease).
        
        Args:
            handler: DatasetHandler with processed data
            group: 'control' or 'disease'
            dataset_name: Name of dataset
            
        Returns:
            dict: Results for this group
        """
        group_results = {
            'group': group,
            'success': False
        }
        
        self.logger.info(f"  Processing {group} group...")
        
        # Get stratified data
        species_data, metabolites_data = handler.get_stratified_data(group)
        
        if species_data is None or metabolites_data is None:
            self.logger.warning(f"  No data available for {group} group")
            return group_results
        
        group_results['n_samples'] = len(species_data)
        self.logger.info(f"    Samples: {group_results['n_samples']}")
        
        # Compute associations
        associations = self.associator.compute_associations_parallel(
            species_data,
            metabolites_data,
            max_workers=self.config['MAX_WORKERS']
        )
        
        if associations is None or len(associations) == 0:
            self.logger.warning(f"  No significant associations found for {group}")
            return group_results
        
        group_results['n_associations'] = len(associations)
        
        # Build network
        network, network_stats = self.networker.build_bipartite_network(
            associations,
            min_abs_corr=self.config['MIN_CORRELATION']
        )
        
        if network is None:
            self.logger.warning(f"  Failed to build network for {group}")
            return group_results
        
        group_results['network_stats'] = network_stats
        
        # Detect modules
        module_assignments, module_stats = self.module_detector.detect_modules(
            network, resolution=1.0
        )
        
        if module_assignments is None or len(module_assignments) == 0:
            self.logger.warning(f"  No modules detected for {group}")
            return group_results
        
        group_results['n_modules'] = module_stats['module'].nunique()
        
        # Identify keystones
        keystones = self.analyzer.identify_keystones(network, module_assignments)
        
        if keystones is None or len(keystones) == 0:
            self.logger.warning(f"  No keystones identified for {group}")
        else:
            group_results['n_keystones'] = len(keystones)
        
        # Store results
        group_results['associations'] = associations
        group_results['network'] = network
        group_results['modules'] = module_assignments
        group_results['module_stats'] = module_stats
        group_results['keystones'] = keystones
        group_results['success'] = True
        
        return group_results
    
    def process_dataset(self, dataset_name: str, dataset_num: int) -> dict:
        """Process dataset with control vs disease stratification.
        
        Args:
            dataset_name: Name of dataset
            dataset_num: Dataset number (for logging)
            
        Returns:
            dict: Processing results
        """
        start_time = time.time()
        
        self.logger.dataset_start(dataset_name, dataset_num, len(self.datasets))
        
        # Get disease group information
        try:
            disease_info = get_disease_groups(dataset_name)
            self.logger.info(f"  Disease: {disease_info['disease_name']}")
            self.logger.info(f"  Control groups: {disease_info['control']}")
            self.logger.info(f"  Disease groups: {disease_info['disease']}")
        except Exception as e:
            self.logger.error(f"  Error getting disease groups: {e}")
            return {'dataset': dataset_name, 'success': False, 'error': str(e)}
        
        result = {
            'dataset': dataset_name,
            'disease_name': disease_info['disease_name'],
            'success': False,
            'execution_time': 0
        }
        
        try:
            # 1. LOAD DATA
            self.logger.phase_start("Phase 1: Load Data & Stratify")
            handler = DatasetHandler(dataset_name, DATA_DIR, self.logger)
            
            if not handler.load_raw_data():
                result['error'] = "Failed to load data"
                result['execution_time'] = time.time() - start_time
                return result
            
            data_type = handler.get_data_type()
            self.logger.info(f"  Data type: {data_type}")
            
            # Stratify samples by disease status
            if not handler.stratify_samples_by_disease():
                result['error'] = "Failed to stratify samples"
                result['execution_time'] = time.time() - start_time
                return result
            
            result['n_control_samples'] = len(handler.control_samples)
            result['n_disease_samples'] = len(handler.disease_samples)
            
            # 2. FILTER DATA
            self.logger.phase_start("Phase 2: Filter Features")
            
            self.filter.set_dataset(dataset_name)
            
            if 'DATASET_SPECIFIC_FILTERS' in self.config:
                dataset_filters = self.config['DATASET_SPECIFIC_FILTERS'].get(dataset_name, {})
                if dataset_filters:
                    self.logger.info(f"  Applying dataset-specific filters: {dataset_filters.get('REASON', 'custom')}")
            
            species_filtered = self.filter.filter_species(
                handler.species_raw, data_type
            )
            
            metabolites_filtered = self.filter.filter_metabolites(
                handler.metabolites_raw, data_type
            )
            
            filter_stats = self.filter.get_filter_stats(
                handler.species_raw, species_filtered,
                handler.metabolites_raw, metabolites_filtered
            )
            self.logger.info(
                f"  Reduced correlation tests from {filter_stats['potential_correlations_before']:,} "
                f"to {filter_stats['potential_correlations_after']:,} "
                f"({filter_stats['correlation_reduction_pct']:.1f}% reduction)"
            )
            
            handler.clear_raw_data()
            
            # 3. TRANSFORM DATA
            self.logger.phase_start("Phase 3: Transform Data")
            
            species_clr, metabolites_zscore = self.transformer.transform_pipeline(
                species_filtered, metabolites_filtered, data_type
            )
            
            handler.species_processed = species_clr
            handler.metabolites_processed = metabolites_zscore
            
            result['n_species'] = species_clr.shape[1]
            result['n_metabolites'] = metabolites_zscore.shape[1]
            
            # 4-7. PROCESS CONTROL GROUP
            self.logger.phase_start("Phase 4-7: Process CONTROL Group")
            control_results = self.process_group(handler, 'control', dataset_name)
            result['control'] = control_results
            
            # 4-7. PROCESS DISEASE GROUP
            self.logger.phase_start("Phase 8-11: Process DISEASE Group")
            disease_results = self.process_group(handler, 'disease', dataset_name)
            result['disease'] = disease_results
            
            # 8. DIFFERENTIAL ANALYSIS
            if control_results['success'] and disease_results['success']:
                self.logger.phase_start("Phase 12: Differential Analysis")
                
                # Compare keystone scores
                diff_keystones = self.diff_analyzer.compare_keystone_scores(
                    control_results['keystones'],
                    disease_results['keystones']
                )
                
                # Compare network structure
                network_comparison = self.diff_analyzer.compare_network_structure(
                    control_results['network'],
                    disease_results['network']
                )
                
                # Summary
                diff_summary = self.diff_analyzer.summarize_differential_analysis(
                    diff_keystones,
                    network_comparison,
                    n_samples_control=control_results['n_samples'],
                    n_samples_disease=disease_results['n_samples']
                )
                
                result['differential'] = {
                    'keystones': diff_keystones,
                    'network_comparison': network_comparison,
                    'summary': diff_summary
                }
                
                # Log key findings
                depleted = diff_keystones[diff_keystones['status'] == 'depleted_in_disease']
                enriched = diff_keystones[diff_keystones['status'] == 'enriched_in_disease']
                
                self.logger.info(f"  Depleted in disease: {len(depleted)} species")
                self.logger.info(f"  Enriched in disease: {len(enriched)} species")
                
                # Store for cross-study analysis
                self.all_keystones_control[dataset_name] = control_results['keystones']
                self.all_keystones_disease[dataset_name] = disease_results['keystones']
                self.all_differential_keystones[dataset_name] = diff_keystones
                
            # 9. SAVE RESULTS
            self.logger.phase_start("Phase 13: Save Results")
            
            dataset_dir = self.output_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True, parents=True)
            
            # Save control results
            if control_results['success']:
                control_dir = dataset_dir / "control"
                control_dir.mkdir(exist_ok=True)
                self._save_group_results(control_results, control_dir)
            
            # Save disease results
            if disease_results['success']:
                disease_dir = dataset_dir / "disease"
                disease_dir.mkdir(exist_ok=True)
                self._save_group_results(disease_results, disease_dir)
            
            # Save differential results
            if 'differential' in result:
                diff_dir = dataset_dir / "differential"
                diff_dir.mkdir(exist_ok=True)
                
                result['differential']['keystones'].to_csv(
                    diff_dir / "differential_keystones.csv", index=False
                )
                result['differential']['summary'].to_csv(
                    diff_dir / "summary.csv", index=False
                )
                
                import pandas as pd
                pd.DataFrame([result['differential']['network_comparison']]).to_csv(
                    diff_dir / "network_comparison.csv", index=False
                )
                
                self.logger.info(f"  Saved differential analysis results")
            
            result['success'] = True
            result['execution_time'] = time.time() - start_time
            
            self.logger.info(f"✓ Dataset completed in {result['execution_time']:.1f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"✗ Error processing dataset: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            result['error'] = str(e)
            result['execution_time'] = time.time() - start_time
            return result
    
    def _save_group_results(self, group_results: dict, output_dir: Path):
        """Save results for a single group (control or disease).
        
        Args:
            group_results: Results dictionary for the group
            output_dir: Directory to save results
        """
        import networkx as nx
        
        # Associations
        group_results['associations'].to_csv(
            output_dir / "associations.csv", index=False
        )
        
        # Network edges
        edge_list = self.networker.extract_edge_list(group_results['network'])
        edge_list.to_csv(output_dir / "network_edges.csv", index=False)
        
        # Network GraphML
        nx.write_graphml(group_results['network'], output_dir / "network.graphml")
        
        # Modules
        group_results['modules'].to_csv(output_dir / "modules.csv", index=False)
        group_results['module_stats'].to_csv(output_dir / "module_stats.csv", index=False)
        
        # Keystones
        if group_results['keystones'] is not None:
            group_results['keystones'].to_csv(output_dir / "keystones.csv", index=False)
            
            top_keystones = self.analyzer.get_top_keystones_per_module(
                group_results['keystones'], top_n=5
            )
            if len(top_keystones) > 0:
                top_keystones.to_csv(output_dir / "top_keystones.csv", index=False)
        
        # Network stats
        import pandas as pd
        stats_df = pd.DataFrame([group_results['network_stats']])
        stats_df['group'] = group_results['group']
        stats_df.to_csv(output_dir / "network_stats.csv", index=False)
    
    def cross_study_analysis(self):
        """Perform cross-study convergence analysis."""
        if len(self.all_keystones_control) < 2:
            self.logger.warning("Need at least 2 datasets for cross-study analysis")
            return
        
        self.logger.section("CROSS-STUDY CONVERGENCE ANALYSIS", level=1)
        
        # Analyze control keystones
        self.logger.info("Analyzing CONTROL keystones across studies...")
        control_convergence = self.analyzer.cross_study_convergence(
            self.all_keystones_control,
            {}  # Modules not needed for this analysis
        )
        
        # Analyze disease keystones
        self.logger.info("Analyzing DISEASE keystones across studies...")
        disease_convergence = self.analyzer.cross_study_convergence(
            self.all_keystones_disease,
            {}
        )
        
        # Save results
        cross_study_dir = self.output_dir / "cross_study"
        cross_study_dir.mkdir(exist_ok=True, parents=True)
        
        # Control conserved keystones
        control_dir = cross_study_dir / "control"
        control_dir.mkdir(exist_ok=True)
        
        control_convergence['conserved_keystones'].to_csv(
            control_dir / "conserved_keystones.csv", index=False
        )
        self.logger.info(f"  Control: {len(control_convergence['conserved_keystones'])} conserved keystones")
        
        # Disease conserved keystones
        disease_dir = cross_study_dir / "disease"
        disease_dir.mkdir(exist_ok=True)
        
        disease_convergence['conserved_keystones'].to_csv(
            disease_dir / "conserved_keystones.csv", index=False
        )
        self.logger.info(f"  Disease: {len(disease_convergence['conserved_keystones'])} conserved keystones")
        
        # Find differential keystones (depleted in disease across studies)
        import pandas as pd
        all_differential = []
        for dataset_name, diff_df in self.all_differential_keystones.items():
            depleted = diff_df[diff_df['status'] == 'depleted_in_disease'].copy()
            depleted['dataset'] = dataset_name
            all_differential.append(depleted)
        
        if len(all_differential) > 0:
            differential_combined = pd.concat(all_differential, ignore_index=True)
            
            # Count occurrences
            depleted_counts = differential_combined['species'].value_counts()
            depleted_conserved = depleted_counts[depleted_counts >= 3]  # Present in ≥3 studies
            
            self.logger.info(f"  Found {len(depleted_conserved)} species consistently depleted in disease (≥3 datasets)")
            
            # Save
            depleted_df = pd.DataFrame({
                'species': depleted_conserved.index,
                'n_datasets_depleted': depleted_conserved.values
            })
            depleted_df.to_csv(cross_study_dir / "conserved_depleted_in_disease.csv", index=False)
            
            # Log top depleted
            self.logger.info("\nTop 10 species depleted in disease across studies:")
            for _, row in depleted_df.head(10).iterrows():
                self.logger.info(f"  {row['species']}: depleted in {row['n_datasets_depleted']} datasets")
    
    def run_main_pipeline(self):
        """Execute core pipeline."""
        pipeline_start = time.time()
        
        self.logger.section("DISEASE-STRATIFIED MICROBIOME-METABOLOME NETWORK ANALYSIS", level=1)
        self.logger.info(f"Datasets: {len(self.datasets)}")
        self.logger.info(f"Max parallel workers: {self.config['MAX_WORKERS']}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("")
        
        # Process each dataset
        for i, dataset_name in enumerate(self.datasets, 1):
            result = self.process_dataset(dataset_name, i)
            self.results[dataset_name] = result
        
        # Cross-study analysis
        self.cross_study_analysis()
        
        # Pipeline summary
        total_time = time.time() - pipeline_start
        self.logger.pipeline_summary(self.results, total_time)
        
        return total_time
    
    def run_downstream_analyses(self):
        """Run all downstream analyses."""
        self.logger.section("DOWNSTREAM ANALYSES", level=1)
        
        # 1. Hub Metabolite Analysis
        self.logger.section("Hub Metabolite Analysis", level=2)
        hub_start = time.time()
        try:
            hub_analyzer = HubMetaboliteAnalyzer(self.output_dir)
            hub_analyzer.run_analysis()
            hub_time = time.time() - hub_start
            self.logger.info(f"✓ Hub metabolite analysis completed in {hub_time:.1f}s")
        except Exception as e:
            self.logger.error(f"✗ Hub metabolite analysis failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
        
        # 2. Module Enrichment Analysis
        self.logger.section("Module Enrichment Analysis", level=2)
        enrich_start = time.time()
        try:
            enrichment_analyzer = ModuleEnrichmentAnalyzer(self.output_dir)
            enrichment_analyzer.run_analysis()
            enrich_time = time.time() - enrich_start
            self.logger.info(f"✓ Module enrichment analysis completed in {enrich_time:.1f}s")
        except Exception as e:
            self.logger.error(f"✗ Module enrichment analysis failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
        
        # 3. Cross-Disease Comparison
        self.logger.section("Cross-Disease Comparison", level=2)
        cross_disease_start = time.time()
        try:
            cross_disease_analyzer = CrossDiseaseComparator(self.output_dir)
            cross_disease_analyzer.run_analysis()
            cross_disease_time = time.time() - cross_disease_start
            self.logger.info(f"✓ Cross-disease comparison completed in {cross_disease_time:.1f}s")
        except Exception as e:
            self.logger.error(f"✗ Cross-disease comparison failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
        
        # 4. Enhanced Hub Metabolite Analysis
        self.logger.section("Enhanced Hub Metabolite Analysis", level=2)
        enhanced_hub_start = time.time()
        try:
            enhanced_hub_analyzer = EnhancedHubMetaboliteAnalyzer(self.output_dir)
            enhanced_hub_analyzer.run_analysis()
            enhanced_hub_time = time.time() - enhanced_hub_start
            self.logger.info(f"✓ Enhanced hub metabolite analysis completed in {enhanced_hub_time:.1f}s")
        except Exception as e:
            self.logger.error(f"✗ Enhanced hub metabolite analysis failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
        
        # 5. Metabolite Annotation
        self.logger.section("Metabolite Functional Annotation", level=2)
        annotation_start = time.time()
        try:
            metabolite_annotator = MetaboliteAnnotator(self.output_dir)
            metabolite_annotator.run_analysis()
            annotation_time = time.time() - annotation_start
            self.logger.info(f"✓ Metabolite annotation completed in {annotation_time:.1f}s")
        except Exception as e:
            self.logger.error(f"✗ Metabolite annotation failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def run(self):
        """Execute complete analysis pipeline."""
        total_start = time.time()
        
        # Run main pipeline
        pipeline_time = self.run_main_pipeline()
        
        # Run downstream analyses
        self.run_downstream_analyses()
        
        # Final summary
        total_time = time.time() - total_start
        
        self.logger.section("COMPLETE ANALYSIS FINISHED", level=1)
        self.logger.info(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        self.logger.info(f"Core pipeline: {pipeline_time:.1f}s")
        self.logger.info(f"Downstream analyses: {total_time - pipeline_time:.1f}s")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info(f"Logs saved to: {self.log_dir}")


def main():
    """Main entry point."""
    print("="*80)
    print("MICROBIOME-METABOLOME NETWORK ANALYSIS")
    print("Complete Analysis Pipeline")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    pipeline = ComprehensiveAnalysisPipeline(
        datasets=DATASETS,
        output_dir=RESULTS_DIR,
        log_dir=LOG_DIR,
        config=CONFIG
    )
    
    pipeline.run()
    
    print()
    print("="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print()
    print("Next step: Run visualizations with:")
    print("    python run_visualizations.py")
    print()


if __name__ == "__main__":
    main()
