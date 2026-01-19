#!/usr/bin/env python3
"""
Complete visualization suite for microbiome-metabolome network analysis.

This script generates all publication-quality figures and tables:
1. Study overview and dataset characteristics
2. Master Network Visualizations (Hubs, Keystones, Differential only)
3. Differential analysis visualizations (control vs disease)
4. Cross-study integration and convergence analysis

Usage:
    python run_visualizations.py

Outputs:
    - output/figures/ - All publication-ready figures
    - output/tables/ - Summary tables and statistics

Note: Run this AFTER completing the main analysis (run_analysis.py)
"""

import sys
from pathlib import Path
from datetime import datetime
import subprocess
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.config import RESULTS_DIR, OUTPUT_DIR


class VisualizationSuite:
    """Orchestrator for all visualization scripts."""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.results_dir = RESULTS_DIR
        self.figures_dir = OUTPUT_DIR / "figures"
        self.tables_dir = OUTPUT_DIR / "tables"
        
        # Create output directories
        self.figures_dir.mkdir(exist_ok=True, parents=True)
        self.tables_dir.mkdir(exist_ok=True, parents=True)
        
        # Define visualization scripts in execution order
        self.visualization_scripts = [
            {
                'path': 'scripts/visualizations/01_study_overview.py',
                'name': 'Study Overview',
                'description': 'Dataset characteristics and sample distributions'
            },
            # REMOVED: Basic hairball network visualizations
            {
                'path': 'scripts/visualizations/master_network_visualizations.py',
                'name': 'Master Network Visualizations',
                'description': 'Targeted network figures (Hubs, Keystones, Differential)'
            },
            {
                'path': 'scripts/visualizations/04_differential_analysis.py',
                'name': 'Differential Analysis',
                'description': 'Control vs disease comparison visualizations'
            },
            {
                'path': 'scripts/visualizations/05_cross_study_analysis.py',
                'name': 'Cross-Study Integration',
                'description': 'Cross-study convergence and key findings'
            },
        ]
        
        self.results = {}
    
    def validate_prerequisites(self):
        """Check that analysis results exist before running visualizations."""
        print("\nValidating analysis results...")
        
        # Check if results directory exists and has content
        if not self.results_dir.exists():
            print(f"\n✗ ERROR: Results directory not found: {self.results_dir}")
            print("Please run the main analysis first:")
            print("    python run_analysis.py")
            return False
        
        # Check for at least one dataset's results
        dataset_dirs = list(self.results_dir.glob("*/control"))
        if len(dataset_dirs) == 0:
            print(f"\n✗ ERROR: No dataset results found in {self.results_dir}")
            print("Please run the main analysis first:")
            print("    python run_analysis.py")
            return False
        
        print(f"✓ Found results for {len(dataset_dirs)} datasets")
        return True
    
    def run_script(self, script_info: dict) -> bool:
        """Run a visualization script.
        
        Args:
            script_info: Dictionary with script information
            
        Returns:
            True if successful, False otherwise
        """
        script_path = self.project_root / script_info['path']
        
        if not script_path.exists():
            print(f"\n⚠ Warning: {script_info['path']} not found, skipping...")
            return False
        
        print()
        print("="*80)
        print(f"RUNNING: {script_info['name']}")
        print("="*80)
        print(f"Description: {script_info['description']}")
        print()
        
        start_time = time.time()
        
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=self.project_root,
                capture_output=False,
                text=True
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                print()
                print(f"✓ {script_info['name']} completed successfully ({execution_time:.1f}s)")
                return True
            else:
                print()
                print(f"✗ {script_info['name']} failed with return code {result.returncode}")
                return False
        
        except Exception as e:
            execution_time = time.time() - start_time
            print()
            print(f"✗ Error running {script_info['name']}: {e}")
            return False
    
    def run_all_visualizations(self):
        """Run all visualization scripts in sequence."""
        print()
        print("="*80)
        print("MICROBIOME-METABOLOME NETWORK ANALYSIS")
        print("Comprehensive Visualization Suite")
        print("="*80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("This will generate all publication-quality figures and tables")
        print("from your completed analysis results.")
        
        # Validate prerequisites
        if not self.validate_prerequisites():
            return False
        
        # Run each visualization script
        total_start = time.time()
        
        for script_info in self.visualization_scripts:
            success = self.run_script(script_info)
            self.results[script_info['name']] = success
        
        # Print summary
        total_time = time.time() - total_start
        
        print()
        print("="*80)
        print("VISUALIZATION SUMMARY")
        print("="*80)
        print()
        
        successful = sum(1 for v in self.results.values() if v)
        total = len(self.results)
        
        print(f"Completed: {successful}/{total} visualization scripts")
        print()
        
        for script_name, success in self.results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {script_name}")
        
        print()
        print(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print()
        print("Output locations:")
        print(f"  Figures: {self.figures_dir}")
        print(f"  Tables:  {self.tables_dir}")
        print()
        
        if successful == total:
            print("="*80)
            print("ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
            print("="*80)
            return True
        else:
            print("="*80)
            print("SOME VISUALIZATIONS FAILED - CHECK LOGS ABOVE")
            print("="*80)
            return False


def main():
    """Main entry point."""
    suite = VisualizationSuite()
    success = suite.run_all_visualizations()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
