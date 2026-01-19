#!/usr/bin/env python3
"""
Quick Start Guide for Master Network Visualizations

This script demonstrates how to generate network visualizations
for specific datasets or customize the visualization parameters.
"""

from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.visualizations.master_network_visualizations import (
    generate_all_visualizations,
    visualize_hub_metabolite_ego_network,
    visualize_keystone_species_ego_network,
    visualize_module_network,
    visualize_core_network,
    visualize_differential_disruption_network,
    load_network_and_metadata
)


def example_1_generate_all():
    """Example 1: Generate all visualizations for all datasets."""
    print("Example 1: Generate ALL visualizations")
    print("=" * 80)
    
    results_dir = project_root / "output" / "results"
    output_dir = project_root / "output" / "figures"
    
    generate_all_visualizations(results_dir, output_dir)


def example_2_specific_datasets():
    """Example 2: Generate visualizations for specific datasets only."""
    print("Example 2: Generate visualizations for specific datasets")
    print("=" * 80)
    
    results_dir = project_root / "output" / "results"
    output_dir = project_root / "output" / "figures"
    
    # Only process these datasets
    datasets = ['YACHIDA_CRC_2019', 'FRANZOSA_IBD_2019']
    
    generate_all_visualizations(results_dir, output_dir, datasets=datasets)


def example_3_single_visualization():
    """Example 3: Generate a single specific visualization."""
    print("Example 3: Generate a single hub metabolite network")
    print("=" * 80)
    
    dataset_dir = project_root / "output" / "results" / "YACHIDA_CRC_2019"
    output_dir = project_root / "output" / "figures"
    
    # Load network and metadata
    G, keystones, hubs, modules = load_network_and_metadata(dataset_dir, "control")
    
    # Generate hub metabolite ego network with custom parameters
    output_path = output_dir / "CUSTOM_hub_metabolite_network"
    visualize_hub_metabolite_ego_network(
        G, 
        hubs, 
        output_path, 
        top_n=15,  # Show top 15 hubs instead of 10
        max_neighbors=75  # Allow up to 75 neighbors instead of 50
    )
    
    print(f"Created custom visualization: {output_path}")


def example_4_custom_core_network():
    """Example 4: Generate core network with custom correlation threshold."""
    print("Example 4: Generate high-confidence core network")
    print("=" * 80)
    
    dataset_dir = project_root / "output" / "results" / "FRANZOSA_IBD_2019"
    output_dir = project_root / "output" / "figures"
    
    # Load network
    G, _, _, _ = load_network_and_metadata(dataset_dir, "disease")
    
    # Generate core network with very strict threshold
    output_path = output_dir / "CUSTOM_ultra_high_confidence_core"
    visualize_core_network(
        G, 
        output_path, 
        min_abs_corr=0.7,  # Only correlations with |ρ| ≥ 0.7
        max_nodes=200  # Limit to top 200 nodes
    )
    
    print(f"Created custom core network: {output_path}")


def example_5_custom_module():
    """Example 5: Visualize a specific module."""
    print("Example 5: Visualize a specific module")
    print("=" * 80)
    
    dataset_dir = project_root / "output" / "results" / "WANG_ESRD_2020"
    output_dir = project_root / "output" / "figures"
    
    # Load network and metadata
    G, _, _, modules = load_network_and_metadata(dataset_dir, "control")
    
    # Visualize module 0 (usually the largest)
    output_path = output_dir / "CUSTOM_module_0"
    visualize_module_network(G, modules, module_id=0, output_path=output_path)
    
    print(f"Created custom module visualization: {output_path}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MASTER NETWORK VISUALIZATIONS - QUICK START EXAMPLES")
    print("=" * 80 + "\n")
    
    # Uncomment the example you want to run:
    
    # Example 1: Generate all visualizations (takes ~10-30 minutes)
    # example_1_generate_all()
    
    # Example 2: Generate visualizations for specific datasets
    # example_2_specific_datasets()
    
    # Example 3: Generate a single custom visualization
    # example_3_single_visualization()
    
    # Example 4: Generate custom core network
    # example_4_custom_core_network()
    
    # Example 5: Visualize specific module
    # example_5_custom_module()
    
    print("\nTo run an example, uncomment it in the main section of this script.")
    print("\nFor full generation, run:")
    print("  python scripts/visualizations/master_network_visualizations.py")
