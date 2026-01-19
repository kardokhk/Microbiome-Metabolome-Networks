"""
Master script for generating publication-ready network visualizations.

This script generates focused, meaningful network visualizations for species-metabolite 
bipartite networks, avoiding the "hairball" problem of visualizing entire large networks.

Visualization strategies:
1. Hub metabolite ego networks - Top metabolites and their immediate neighbors
2. Keystone species ego networks - Top species and their immediate neighbors
3. Differential networks - Networks highlighting disrupted nodes in disease

All figures are saved individually (not multi-panel) for publication flexibility.
No titles are added (for figure legends in manuscripts).
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from adjustText import adjust_text
warnings.filterwarnings('ignore')

# Academic publication settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts for PDF
plt.rcParams['ps.fonttype'] = 42

# Color schemes
SPECIES_COLOR = '#4292C6'  # Blue
METABOLITE_COLOR = '#E34A33'  # Red/orange
KEYSTONE_COLOR = '#08519C'  # Dark blue for keystone species
HUB_COLOR = '#A50F15'  # Dark red for hub metabolites
DEPLETED_COLOR = '#807DBA'  # Purple for depleted in disease
ENRICHED_COLOR = '#41AB5D'  # Green for enriched in disease
POSITIVE_EDGE = '#41AB5D'  # Green
NEGATIVE_EDGE = '#807DBA'  # Purple


def load_network_and_metadata(dataset_dir: Path, group: str) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load network and associated metadata files."""
    group_dir = dataset_dir / group
    
    # Load network
    network_path = group_dir / "network.graphml"
    G = nx.read_graphml(network_path)
    
    # Load keystones
    keystones_df = pd.read_csv(group_dir / "keystones.csv")
    
    # Load hub metabolites
    hub_metabolites_df = pd.read_csv(group_dir / "hub_metabolites.csv")
    
    # Load modules
    modules_df = pd.read_csv(group_dir / "modules.csv")
    
    print(f"  Loaded {group} network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G, keystones_df, hub_metabolites_df, modules_df


def calculate_layout(G: nx.Graph, layout_type: str = 'spring', seed: int = 42) -> Dict:
    """Calculate network layout."""
    n_nodes = G.number_of_nodes()
    
    if layout_type == 'spring':
        # Adjust parameters based on network size
        if n_nodes > 500:
            iterations = 30
            k = 1.0 / np.sqrt(n_nodes)
        else:
            iterations = 50
            k = 0.5 / np.sqrt(n_nodes)
            
        pos = nx.spring_layout(G, k=k, iterations=iterations, seed=seed)
        
    elif layout_type == 'kamada_kawai':
        if n_nodes > 500:
            print("    Warning: Using spring layout (Kamada-Kawai is slow for large networks)")
            pos = nx.spring_layout(G, k=0.5/np.sqrt(n_nodes), iterations=30, seed=seed)
        else:
            pos = nx.kamada_kawai_layout(G)
            
    elif layout_type == 'bipartite':
        # Separate species and metabolites
        species = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'species']
        metabolites = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'metabolite']
        
        pos = {}
        # Species on left
        for i, node in enumerate(species):
            pos[node] = (0, i / max(len(species), 1))
        # Metabolites on right
        for i, node in enumerate(metabolites):
            pos[node] = (1, i / max(len(metabolites), 1))
            
    else:
        raise ValueError(f"Unknown layout: {layout_type}")
    
    return pos


def draw_network_base(G: nx.Graph, pos: Dict, ax: plt.Axes, 
                      node_colors: List, node_sizes: List,
                      edge_colors: List, edge_widths: List,
                      node_labels: Optional[Dict] = None,
                      draw_edges: bool = True) -> None:
    """Draw base network visualization."""
    # Draw edges first (background)
    if draw_edges and G.number_of_edges() > 0:
        # Ensure edge_colors are proper format (convert to list of strings if needed)
        edge_colors_clean = []
        for ec in edge_colors:
            if isinstance(ec, (tuple, list)):
                # Skip alpha if present, use RGB tuple
                if len(ec) == 4:
                    edge_colors_clean.append(ec[:3])
                else:
                    edge_colors_clean.append(ec)
            else:
                edge_colors_clean.append(ec)
        
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors_clean,
            width=edge_widths,
            alpha=0.4,
            ax=ax
        )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0.5,
        edgecolors='white',
        ax=ax
    )
    
    # Draw labels if provided with adjustText for automatic placement
    if node_labels:
        # Create text objects for each label
        texts = []
        for node, label in node_labels.items():
            if node in pos:
                x, y = pos[node]
                text = ax.text(
                    x, y, label,
                    fontsize=6,
                    fontweight='normal',
                    fontfamily='Arial',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7)
                )
                texts.append(text)
        
        # Automatically adjust text positions to avoid overlap
        if texts:
            adjust_text(
                texts,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                expand_points=(1.2, 1.2),
                expand_text=(1.1, 1.1),
                force_text=(0.5, 0.5),
                force_points=(0.2, 0.2),
                ax=ax
            )
    
    ax.axis('off')
    ax.set_aspect('equal')


def create_legend(ax: plt.Axes, legend_elements: List, loc: str = 'upper right') -> None:
    """Add legend to axis."""
    legend = ax.legend(
        handles=legend_elements,
        loc=loc,
        frameon=True,
        fancybox=False,
        framealpha=0.9,
        edgecolor='black',
        fontsize=9
    )
    legend.get_frame().set_linewidth(0.5)


def save_figure(fig: plt.Figure, output_path: Path, dpi: int = 300) -> None:
    """Save figure in both PNG and PDF formats."""
    png_path = output_path.with_suffix('.png')
    pdf_path = output_path.with_suffix('.pdf')
    
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    plt.close(fig)
    
    print(f"  Saved: {output_path.stem}")


# ============================================================================
# 1. HUB METABOLITE EGO NETWORKS
# ============================================================================

def visualize_hub_metabolite_ego_network(
    G: nx.Graph,
    hub_metabolites_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 10,
    max_neighbors: int = 50
) -> None:
    """Visualize ego networks of top hub metabolites."""
    # Get top hub metabolites
    top_hubs = hub_metabolites_df.nlargest(top_n, 'hub_score')['metabolite'].tolist()
    
    # Build ego network (hubs + their neighbors)
    nodes_to_include = set(top_hubs)
    for hub in top_hubs:
        if hub in G:
            neighbors = list(G.neighbors(hub))
            # Limit neighbors if too many
            if len(neighbors) > max_neighbors:
                # Select top neighbors by edge weight
                neighbor_weights = [(n, abs(float(G[hub][n].get('weight', 0)))) for n in neighbors]
                neighbor_weights.sort(key=lambda x: x[1], reverse=True)
                neighbors = [n for n, w in neighbor_weights[:max_neighbors]]
            nodes_to_include.update(neighbors)
    
    # Create subgraph
    G_sub = G.subgraph(nodes_to_include).copy()
    
    print(f"  Hub metabolite ego network: {G_sub.number_of_nodes()} nodes, {G_sub.number_of_edges()} edges")
    
    # Calculate layout
    pos = calculate_layout(G_sub, layout_type='spring')
    
    # Prepare node colors and sizes
    node_colors = []
    node_sizes = []
    
    for node in G_sub.nodes():
        if node in top_hubs:
            node_colors.append(HUB_COLOR)
            node_sizes.append(200)  # Large for hubs
        elif G_sub.nodes[node].get('node_type') == 'metabolite':
            node_colors.append(METABOLITE_COLOR)
            node_sizes.append(80)
        else:  # species
            node_colors.append(SPECIES_COLOR)
            node_sizes.append(30)
    
    # Prepare edge colors and widths
    edge_colors = []
    edge_widths = []
    
    for u, v, d in G_sub.edges(data=True):
        weight = float(d.get('weight', 0))
        abs_weight = abs(weight)
        
        if weight > 0:
            edge_colors.append(POSITIVE_EDGE)
        else:
            edge_colors.append(NEGATIVE_EDGE)
        
        # Scale width by correlation strength
        edge_widths.append(0.5 + abs_weight * 2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Create labels dict for hub metabolites only (clean metabolite names)
    node_labels = {}
    for hub in top_hubs:
        if hub in G_sub:
            # Clean up metabolite name for display
            label = hub.split('_')[0] if '_' in hub else hub
            node_labels[hub] = label
    
    draw_network_base(G_sub, pos, ax, node_colors, node_sizes, edge_colors, edge_widths, 
                     node_labels=node_labels)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=HUB_COLOR, label='Hub metabolites'),
        mpatches.Patch(color=METABOLITE_COLOR, label='Other metabolites'),
        mpatches.Patch(color=SPECIES_COLOR, label='Species'),
        plt.Line2D([0], [0], color=POSITIVE_EDGE, lw=2, label='Positive correlation'),
        plt.Line2D([0], [0], color=NEGATIVE_EDGE, lw=2, label='Negative correlation')
    ]
    create_legend(ax, legend_elements)
    
    save_figure(fig, output_path)


# ============================================================================
# 2. KEYSTONE SPECIES EGO NETWORKS
# ============================================================================

def visualize_keystone_species_ego_network(
    G: nx.Graph,
    keystones_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 10,
    max_neighbors: int = 50
) -> None:
    """Visualize ego networks of top keystone species."""
    # Get top keystones
    top_keystones = keystones_df.nlargest(top_n, 'keystone_score')['species'].tolist()
    
    # Build ego network
    nodes_to_include = set(top_keystones)
    for keystone in top_keystones:
        if keystone in G:
            neighbors = list(G.neighbors(keystone))
            # Limit neighbors if too many
            if len(neighbors) > max_neighbors:
                neighbor_weights = [(n, abs(float(G[keystone][n].get('weight', 0)))) for n in neighbors]
                neighbor_weights.sort(key=lambda x: x[1], reverse=True)
                neighbors = [n for n, w in neighbor_weights[:max_neighbors]]
            nodes_to_include.update(neighbors)
    
    # Create subgraph
    G_sub = G.subgraph(nodes_to_include).copy()
    
    print(f"  Keystone species ego network: {G_sub.number_of_nodes()} nodes, {G_sub.number_of_edges()} edges")
    
    # Calculate layout
    pos = calculate_layout(G_sub, layout_type='spring')
    
    # Prepare node colors and sizes
    node_colors = []
    node_sizes = []
    
    for node in G_sub.nodes():
        if node in top_keystones:
            node_colors.append(KEYSTONE_COLOR)
            node_sizes.append(200)  # Large for keystones
        elif G_sub.nodes[node].get('node_type') == 'species':
            node_colors.append(SPECIES_COLOR)
            node_sizes.append(30)
        else:  # metabolites
            node_colors.append(METABOLITE_COLOR)
            node_sizes.append(80)
    
    # Prepare edge colors and widths
    edge_colors = []
    edge_widths = []
    
    for u, v, d in G_sub.edges(data=True):
        weight = float(d.get('weight', 0))
        abs_weight = abs(weight)
        
        if weight > 0:
            edge_colors.append(POSITIVE_EDGE)
        else:
            edge_colors.append(NEGATIVE_EDGE)
        
        edge_widths.append(0.5 + abs_weight * 2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Create labels dict for keystone species only (extract genus/species name)
    node_labels = {}
    for keystone in top_keystones:
        if keystone in G_sub:
            # Extract species from full taxonomic path
            parts = keystone.split(';')
            if len(parts) >= 1:
                species_part = parts[-1]
                if species_part.startswith('s  '):
                    species_name = species_part[3:]
                elif species_part.startswith('s__'):
                    species_name = species_part[3:]
                else:
                    species_name = species_part.strip()
                
                if ' ' in species_name and len(species_name) > 20:
                    parts_name = species_name.split(' ', 1)
                    label = f"{parts_name[0]}\n{parts_name[1]}"
                else:
                    label = species_name
            else:
                label = keystone.split(';')[-1].strip()
            
            node_labels[keystone] = label
    
    draw_network_base(G_sub, pos, ax, node_colors, node_sizes, edge_colors, edge_widths,
                     node_labels=node_labels)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=KEYSTONE_COLOR, label='Keystone species'),
        mpatches.Patch(color=SPECIES_COLOR, label='Other species'),
        mpatches.Patch(color=METABOLITE_COLOR, label='Metabolites'),
        plt.Line2D([0], [0], color=POSITIVE_EDGE, lw=2, label='Positive correlation'),
        plt.Line2D([0], [0], color=NEGATIVE_EDGE, lw=2, label='Negative correlation')
    ]
    create_legend(ax, legend_elements)
    
    save_figure(fig, output_path)


# ============================================================================
# 3. MODULE NETWORKS (Placeholder for functions)
# ============================================================================

def visualize_module_network(G, modules_df, module_id, output_path):
    """Visualize a specific functional module."""
    module_nodes = modules_df[modules_df['module'] == module_id]['node'].tolist()
    if len(module_nodes) == 0:
        return
    
    G_sub = G.subgraph(module_nodes).copy()
    pos = calculate_layout(G_sub, layout_type='spring')
    degrees = dict(G_sub.degree())
    max_degree = max(degrees.values()) if degrees else 1
    
    node_colors = []
    node_sizes = []
    
    for node in G_sub.nodes():
        degree = degrees[node]
        normalized_degree = degree / max_degree
        if G_sub.nodes[node].get('node_type') == 'species':
            node_colors.append(SPECIES_COLOR)
            node_sizes.append(20 + normalized_degree * 80)
        else:
            node_colors.append(METABOLITE_COLOR)
            node_sizes.append(40 + normalized_degree * 160)
            
    edge_colors = [POSITIVE_EDGE if float(d.get('weight', 0)) > 0 else NEGATIVE_EDGE 
                  for u, v, d in G_sub.edges(data=True)]
    edge_widths = [0.3 + abs(float(d.get('weight', 0))) * 1.5 
                  for u, v, d in G_sub.edges(data=True)]
    
    fig, ax = plt.subplots(figsize=(12, 12))
    draw_network_base(G_sub, pos, ax, node_colors, node_sizes, edge_colors, edge_widths)
    save_figure(fig, output_path)


# ============================================================================
# 4. HIGH-CONFIDENCE CORE NETWORKS (Placeholder for functions)
# ============================================================================

def visualize_core_network(G, output_path, min_abs_corr=0.5, max_nodes=300):
    """Visualize high-confidence core network."""
    edges_to_keep = [(u, v) for u, v, d in G.edges(data=True) if abs(float(d.get('weight', 0))) >= min_abs_corr]
    G_sub = G.edge_subgraph(edges_to_keep).copy()
    
    if G_sub.number_of_nodes() > max_nodes:
        degrees = dict(G_sub.degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:max_nodes]
        G_sub = G_sub.subgraph(top_nodes).copy()
        
    if G_sub.number_of_nodes() == 0:
        return

    pos = calculate_layout(G_sub, layout_type='spring')
    degrees = dict(G_sub.degree())
    max_degree = max(degrees.values()) if degrees else 1
    
    node_colors = []
    node_sizes = []
    for node in G_sub.nodes():
        degree = degrees[node]
        normalized_degree = degree / max_degree
        if G_sub.nodes[node].get('node_type') == 'species':
            node_colors.append(SPECIES_COLOR)
            node_sizes.append(20 + normalized_degree * 80)
        else:
            node_colors.append(METABOLITE_COLOR)
            node_sizes.append(40 + normalized_degree * 160)

    edge_colors = [POSITIVE_EDGE if float(d.get('weight', 0)) > 0 else NEGATIVE_EDGE 
                  for u, v, d in G_sub.edges(data=True)]
    edge_widths = [1 + abs(float(d.get('weight', 0))) * 2 
                  for u, v, d in G_sub.edges(data=True)]
    
    fig, ax = plt.subplots(figsize=(12, 12))
    draw_network_base(G_sub, pos, ax, node_colors, node_sizes, edge_colors, edge_widths)
    save_figure(fig, output_path)


# ============================================================================
# 5. DIFFERENTIAL DISRUPTION NETWORKS
# ============================================================================

def visualize_differential_disruption_network(
    G_control: nx.Graph,
    G_disease: nx.Graph,
    differential_keystones: pd.DataFrame,
    differential_hubs: pd.DataFrame,
    output_path: Path,
    focus: str = 'depleted',
    top_n: int = 20,
    max_neighbors: int = 30
) -> None:
    """Visualize network disruption in disease."""
    # Get disrupted nodes
    if focus == 'depleted':
        disrupted_species = differential_keystones[
            differential_keystones['status'] == 'depleted_in_disease'
        ].nlargest(top_n, 'delta_keystone_score', keep='all')['species'].tolist()
        
        disrupted_metabolites = differential_hubs[
            differential_hubs['status'] == 'depleted_in_disease'
        ].nlargest(top_n, 'delta_hub_score', keep='all')['metabolite'].tolist()
    else:  # enriched
        disrupted_species = differential_keystones[
            differential_keystones['status'] == 'enriched_in_disease'
        ].nlargest(top_n, 'delta_keystone_score', keep='all')['species'].tolist()
        
        disrupted_metabolites = differential_hubs[
            differential_hubs['status'] == 'enriched_in_disease'
        ].nlargest(top_n, 'delta_hub_score', keep='all')['metabolite'].tolist()
    
    disrupted_nodes = set(disrupted_species + disrupted_metabolites)
    
    # Build ego network from control (to show what was lost/gained)
    nodes_to_include = set(disrupted_nodes)
    for node in disrupted_nodes:
        if node in G_control:
            neighbors = list(G_control.neighbors(node))
            if len(neighbors) > max_neighbors:
                neighbor_weights = [(n, abs(float(G_control[node][n].get('weight', 0)))) 
                                   for n in neighbors]
                neighbor_weights.sort(key=lambda x: x[1], reverse=True)
                neighbors = [n for n, w in neighbor_weights[:max_neighbors]]
            nodes_to_include.update(neighbors)
    
    # Create subgraph
    G_sub = G_control.subgraph(nodes_to_include).copy()
    
    print(f"  Differential {focus} network: {G_sub.number_of_nodes()} nodes, {G_sub.number_of_edges()} edges")
    
    if G_sub.number_of_nodes() == 0:
        print(f"  Warning: No nodes in differential network")
        return
    
    # Calculate layout
    pos = calculate_layout(G_sub, layout_type='spring')
    
    # Prepare node colors and sizes
    node_colors = []
    node_sizes = []
    
    disrupted_color = DEPLETED_COLOR if focus == 'depleted' else ENRICHED_COLOR
    
    for node in G_sub.nodes():
        if node in disrupted_nodes:
            node_colors.append(disrupted_color)
            node_sizes.append(200)
        elif G_sub.nodes[node].get('node_type') == 'species':
            node_colors.append(SPECIES_COLOR)
            node_sizes.append(30)
        else:
            node_colors.append(METABOLITE_COLOR)
            node_sizes.append(80)
    
    # Prepare edge colors and widths
    edge_colors = []
    edge_widths = []
    
    for u, v, d in G_sub.edges(data=True):
        weight = float(d.get('weight', 0))
        abs_weight = abs(weight)
        
        if weight > 0:
            edge_colors.append(POSITIVE_EDGE)
        else:
            edge_colors.append(NEGATIVE_EDGE)
        
        edge_widths.append(0.5 + abs_weight * 2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    draw_network_base(G_sub, pos, ax, node_colors, node_sizes, edge_colors, edge_widths)
    
    # Add legend
    label = 'Depleted in disease' if focus == 'depleted' else 'Enriched in disease'
    legend_elements = [
        mpatches.Patch(color=disrupted_color, label=label),
        mpatches.Patch(color=SPECIES_COLOR, label='Other species'),
        mpatches.Patch(color=METABOLITE_COLOR, label='Other metabolites'),
        plt.Line2D([0], [0], color=POSITIVE_EDGE, lw=2, label='Positive correlation'),
        plt.Line2D([0], [0], color=NEGATIVE_EDGE, lw=2, label='Negative correlation')
    ]
    create_legend(ax, legend_elements)
    
    save_figure(fig, output_path)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def generate_all_visualizations(
    results_dir: Path,
    output_dir: Path,
    datasets: Optional[List[str]] = None
) -> None:
    """Generate filtered network visualizations for all datasets.
    
    Args:
        results_dir: Directory containing analysis results
        output_dir: Directory to save figures
        datasets: Optional list of specific datasets to process
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all dataset directories
    if datasets is None:
        dataset_dirs = [d for d in results_dir.iterdir() 
                       if d.is_dir() and not d.name.startswith('cross')]
        datasets = [d.name for d in dataset_dirs]
    else:
        dataset_dirs = [results_dir / ds for ds in datasets]
    
    print("=" * 80)
    print("MASTER NETWORK VISUALIZATION PIPELINE (FILTERED)")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Datasets: {len(datasets)}")
    print("=" * 80)
    
    for dataset_dir in sorted(dataset_dirs):
        if not dataset_dir.exists():
            continue
            
        dataset_name = dataset_dir.name
        
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*80}")
        
        # Process control group
        if (dataset_dir / "control").exists():
            print("\n[CONTROL GROUP]")
            G_control, keystones_control, hubs_control, modules_control = \
                load_network_and_metadata(dataset_dir, "control")
            
            # Create organized subdirectory structure
            dataset_output_dir = output_dir / "03_networks_detailed" / "by_dataset" / dataset_name / "control"
            dataset_output_dir.mkdir(exist_ok=True, parents=True)
            
            # 1. Hub metabolite ego network
            print("  Generating hub metabolite ego network...")
            output_path = dataset_output_dir / f"NET01_hub_metabolites"
            visualize_hub_metabolite_ego_network(
                G_control, hubs_control, output_path, top_n=10, max_neighbors=50
            )
            
            # 2. Keystone species ego network
            print("  Generating keystone species ego network...")
            output_path = dataset_output_dir / f"NET02_keystone_species"
            visualize_keystone_species_ego_network(
                G_control, keystones_control, output_path, top_n=10, max_neighbors=50
            )
            
            # NOTE: Module and Core visualizations commented out to reduce clutter
            # output_path = dataset_output_dir / f"NET03_module_{module_id}"
            # output_path = dataset_output_dir / f"NET04_core"
        
        # Process disease group
        if (dataset_dir / "disease").exists():
            print("\n[DISEASE GROUP]")
            G_disease, keystones_disease, hubs_disease, modules_disease = \
                load_network_and_metadata(dataset_dir, "disease")
            
            # Create organized subdirectory structure
            dataset_output_dir = output_dir / "03_networks_detailed" / "by_dataset" / dataset_name / "disease"
            dataset_output_dir.mkdir(exist_ok=True, parents=True)
            
            # 1. Hub metabolite ego network
            print("  Generating hub metabolite ego network...")
            output_path = dataset_output_dir / f"NET01_hub_metabolites"
            visualize_hub_metabolite_ego_network(
                G_disease, hubs_disease, output_path, top_n=10, max_neighbors=50
            )
            
            # 2. Keystone species ego network
            print("  Generating keystone species ego network...")
            output_path = dataset_output_dir / f"NET02_keystone_species"
            visualize_keystone_species_ego_network(
                G_disease, keystones_disease, output_path, top_n=10, max_neighbors=50
            )
            
            # NOTE: Module and Core visualizations commented out to reduce clutter
        
        # Process differential networks (if both control and disease exist)
        if (dataset_dir / "control").exists() and (dataset_dir / "disease").exists():
            if (dataset_dir / "differential").exists():
                print("\n[DIFFERENTIAL NETWORKS]")
                
                # Create organized subdirectory structure for differential
                dataset_output_dir = output_dir / "03_networks_detailed" / "by_dataset" / dataset_name / "differential"
                dataset_output_dir.mkdir(exist_ok=True, parents=True)
                
                # Load differential data
                diff_keystones = pd.read_csv(dataset_dir / "differential" / "differential_keystones.csv")
                diff_hubs = pd.read_csv(dataset_dir / "differential" / "differential_hub_metabolites.csv")
                
                # Depleted network
                print("  Generating depleted network...")
                output_path = dataset_output_dir / f"NET05_depleted"
                visualize_differential_disruption_network(
                    G_control, G_disease, diff_keystones, diff_hubs,
                    output_path, focus='depleted', top_n=15, max_neighbors=30
                )
                
                # Enriched network
                print("  Generating enriched network...")
                output_path = dataset_output_dir / f"NET06_enriched"
                visualize_differential_disruption_network(
                    G_control, G_disease, diff_keystones, diff_hubs,
                    output_path, focus='enriched', top_n=15, max_neighbors=30
                )
    
    print("\n" + "=" * 80)
    print("NETWORK VISUALIZATION COMPLETE (FILTERED)")
    print(f"All figures saved to: {output_dir}")
    print("=" * 80)


def main():
    """Main entry point."""
    # Paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent.parent
    results_dir = project_dir / "output" / "results"
    output_dir = project_dir / "output" / "figures"
    
    # Generate all visualizations
    generate_all_visualizations(results_dir, output_dir)


if __name__ == "__main__":
    main()
