#!/usr/bin/env python3
"""
Master script to generate all visualizations.

Run this after the main pipeline has completed to create publication-quality
figures and tables.

Usage:
    python scripts/run_all_visualizations.py
"""

import sys
from pathlib import Path
from datetime import datetime
import subprocess

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_script(script_path, script_name):
    """Run a visualization script."""
    print()
    print("="*80)
    print(f"RUNNING: {script_name}")
    print("="*80)
    print()

    try:
        # Run the script using Python
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=False,
            text=True
        )

        if result.returncode == 0:
            print()
            print(f"✓ {script_name} completed successfully")
            return True
        else:
            print()
            print(f"✗ {script_name} failed with return code {result.returncode}")
            return False

    except Exception as e:
        print()
        print(f"✗ Error running {script_name}: {e}")
        return False


def main():
    """Run all visualization scripts in sequence."""

    start_time = datetime.now()

    print("="*80)
    print("MICROBIOME-METABOLOME NETWORK ANALYSIS")
    print("COMPREHENSIVE VISUALIZATION SUITE")
    print("="*80)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("This script will generate publication-quality figures and tables")
    print("from your completed analysis pipeline.")
    print()

    # Define scripts to run
    scripts = [
        ('visualizations/01_study_overview.py', 'Study Overview'),
        ('visualizations/network_visualizations.py', 'Network Visualizations'),
        ('visualizations/04_differential_analysis.py', 'Differential Analysis'),
        ('visualizations/05_cross_study_analysis.py', 'Cross-Study Integration (KEY FINDINGS)'),
    ]

    results = {}

    # Run each script
    for script_file, script_name in scripts:
        script_path = PROJECT_ROOT / 'scripts' / script_file

        if not script_path.exists():
            print(f"Warning: {script_file} not found, skipping...")
            results[script_name] = False
            continue

        success = run_script(script_path, script_name)
        results[script_name] = success

    # Print final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print()
    print("="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print()

    successful = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"Completed: {successful}/{total} scripts")
    print()

    for script_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: {script_name}")

    print()
    print(f"Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Outputs:")
    print(f"  Figures: {PROJECT_ROOT}/output/figures/")
    print(f"  Tables: {PROJECT_ROOT}/output/tables/")
    print()

    if successful == total:
        print("="*80)
        print("✓ ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
        print("="*80)
        return 0
    else:
        print("="*80)
        print(f"⚠ {total - successful} SCRIPT(S) FAILED")
        print("="*80)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
