"""
Network visualizations for species-metabolite bipartite networks.

This script creates publication-quality visualizations of the bipartite networks
for control and disease groups across all datasets. Each network is saved as a 
separate figure for maximum clarity and flexibility in publications.

The networks are large and complex (hundreds to thousands of nodes, thousands 
to hundreds of thousands of edges), so visualization uses force-directed layouts
with careful scaling and color schemes.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Academic publication settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts for PDF
plt.rcParams['ps.fonttype'] = 42

# Color schemes for publication
SPECIES_COLOR = '#4292C6'  # Blue for species nodes
METABOLITE_COLOR = '#E34A33'  # Red/orange for metabolite nodes
POSITIVE_EDGE_COLOR = '#41AB5D'  # Green for positive correlations
NEGATIVE_EDGE_COLOR = '#807DBA'  # Purple for negative correlations


def load_network(graphml_path):
    """Load network from GraphML file.
    
    Args:
        graphml_path: Path to GraphML file
        
    Returns:
        NetworkX Graph object
    """
    G = nx.read_graphml(graphml_path)
    print(f"  Loaded network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def calculate_layout(G, layout_type='spring', iterations=50, k=None):
    """Calculate node positions using specified layout algorithm.
    
    For large networks, we use spring layout with limited iterations for 
    computational efficiency while maintaining reasonable visual quality.
    
    Args:
        G: NetworkX graph
        layout_type: Type of layout ('spring', 'kamada_kawai', 'spectral')
        iterations: Number of iterations for spring layout
        k: Optimal distance between nodes (None = auto)
        
    Returns:
        Dictionary of node positions
    """
    n_nodes = G.number_of_nodes()
    
    print(f"  Computing {layout_type} layout...")
    
    if layout_type == 'spring':
        # For very large networks, reduce iterations
        if n_nodes > 5000:
            iterations = min(30, iterations)
            k = k or 1.0 / np.sqrt(n_nodes)
        else:
            k = k or 0.5 / np.sqrt(n_nodes)
            
        pos = nx.spring_layout(
            G, 
            k=k,
            iterations=iterations,
            seed=42,  # Reproducible layouts
            scale=1.0
        )
    elif layout_type == 'kamada_kawai':
        # Only for smaller networks (very slow on large graphs)
        if n_nodes > 2000:
            print("    Warning: Kamada-Kawai slow on large networks, using spring instead")
            pos = nx.spring_layout(G, k=0.5/np.sqrt(n_nodes), iterations=30, seed=42)
        else:
            pos = nx.kamada_kawai_layout(G, scale=1.0)
    elif layout_type == 'spectral':
        pos = nx.spectral_layout(G, scale=1.0)
    else:
        raise ValueError(f"Unknown layout type: {layout_type}")
    
    return pos


def get_node_attributes(G):
    """Extract node attributes for visualization.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Tuple of (species_nodes, metabolite_nodes, node_colors, node_sizes)
    """
    species_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'species']
    metabolite_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'metabolite']
    
    # Color nodes by type
    node_colors = []
    for node in G.nodes():
        if G.nodes[node].get('node_type') == 'species':
            node_colors.append(SPECIES_COLOR)
        else:
            node_colors.append(METABOLITE_COLOR)
    
    # Size nodes by degree (connectivity)
    degrees = dict(G.degree())
    max_degree = max(degrees.values())
    min_degree = min(degrees.values())
    
    # Scale node sizes: metabolites larger (20-80), species smaller (5-30)
    node_sizes = []
    for node in G.nodes():
        degree = degrees[node]
        normalized_degree = (degree - min_degree) / (max_degree - min_degree) if max_degree > min_degree else 0.5
        
        if G.nodes[node].get('node_type') == 'metabolite':
            # Metabolites: 20 to 80
            size = 20 + normalized_degree * 60
        else:
            # Species: 3 to 20
            size = 3 + normalized_degree * 17
        
        node_sizes.append(size)
    
    print(f"  Nodes: {len(species_nodes)} species, {len(metabolite_nodes)} metabolites")
    
    return species_nodes, metabolite_nodes, node_colors, node_sizes


def get_edge_attributes(G, edge_alpha_scale=1.0):
    """Extract edge attributes for visualization.
    
    Args:
        G: NetworkX graph
        edge_alpha_scale: Scale factor for edge transparency (0-1)
        
    Returns:
        Tuple of (positive_edges, negative_edges, pos_colors, neg_colors, pos_widths, neg_widths)
    """
    positive_edges = []
    negative_edges = []
    pos_weights = []
    neg_weights = []
    
    for u, v, d in G.edges(data=True):
        weight = float(d.get('weight', 0))
        if weight > 0:
            positive_edges.append((u, v))
            pos_weights.append(abs(weight))
        else:
            negative_edges.append((u, v))
            neg_weights.append(abs(weight))
    
    # Scale edge widths and transparency by correlation strength
    def scale_edges(weights, min_width=0.05, max_width=1.5, base_alpha=0.15):
        """Scale edge visual properties by weight."""
        if not weights:
            return [], []
        
        weights = np.array(weights)
        
        # Width: scale by absolute correlation
        widths = min_width + (weights - weights.min()) / (weights.max() - weights.min() + 1e-10) * (max_width - min_width)
        
        # Alpha: stronger correlations more visible, scaled by overall network density
        alphas = base_alpha + (weights - weights.min()) / (weights.max() - weights.min() + 1e-10) * (0.5 - base_alpha)
        alphas = alphas * edge_alpha_scale
        
        return widths.tolist(), alphas.tolist()
    
    pos_widths, pos_alphas = scale_edges(pos_weights)
    neg_widths, neg_alphas = scale_edges(neg_weights)
    
    # Create RGBA colors with alpha
    pos_colors = [(0.255, 0.671, 0.365, alpha) for alpha in pos_alphas]  # Green with alpha
    neg_colors = [(0.502, 0.490, 0.729, alpha) for alpha in neg_alphas]  # Purple with alpha
    
    print(f"  Edges: {len(positive_edges)} positive, {len(negative_edges)} negative")
    
    return positive_edges, negative_edges, pos_colors, neg_colors, pos_widths, neg_widths


def visualize_network(G, output_path, layout_type='spring', figsize=(12, 12), dpi=300):
    """Create publication-quality network visualization.
    
    Args:
        G: NetworkX graph
        output_path: Path to save figure
        layout_type: Layout algorithm to use
        figsize: Figure size in inches
        dpi: Resolution for raster formats
    """
    print(f"\nVisualizing network: {output_path.stem}")
    
    # Calculate layout
    pos = calculate_layout(G, layout_type=layout_type)
    
    # Get node and edge attributes
    species_nodes, metabolite_nodes, node_colors, node_sizes = get_node_attributes(G)
    
    # Scale edge transparency by network density
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = 2 * n_edges / (n_nodes * (n_nodes - 1))
    
    # For dense networks, make edges more transparent
    if density > 0.01:
        edge_alpha_scale = 0.3
    elif density > 0.005:
        edge_alpha_scale = 0.5
    else:
        edge_alpha_scale = 0.8
    
    positive_edges, negative_edges, pos_colors, neg_colors, pos_widths, neg_widths = get_edge_attributes(
        G, edge_alpha_scale=edge_alpha_scale
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    ax.axis('off')
    
    # Draw edges first (background)
    # Draw negative edges
    if negative_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=negative_edges,
            edge_color=neg_colors,
            width=neg_widths,
            ax=ax,
            arrows=False
        )
    
    # Draw positive edges
    if positive_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=positive_edges,
            edge_color=pos_colors,
            width=pos_widths,
            ax=ax,
            arrows=False
        )
    
    # Draw nodes on top
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0.5,
        edgecolors='white',
        ax=ax
    )
    
    # Set limits with small margin
    x_values = [pos[node][0] for node in G.nodes()]
    y_values = [pos[node][1] for node in G.nodes()]
    x_margin = (max(x_values) - min(x_values)) * 0.05
    y_margin = (max(y_values) - min(y_values)) * 0.05
    
    ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
    ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
    
    # Save figure
    plt.tight_layout(pad=0.1)
    
    # Save as both PNG and PDF
    png_path = output_path.with_suffix('.png')
    pdf_path = output_path.with_suffix('.pdf')
    
    plt.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.close()
    
    print(f"  Saved: {png_path.name} and {pdf_path.name}")


def visualize_all_networks(results_dir, figures_dir, layout_type='spring'):
    """Generate visualizations for all networks in the analysis.
    
    Args:
        results_dir: Directory containing analysis results
        figures_dir: Directory to save figures
        layout_type: Layout algorithm to use
    """
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all dataset directories
    dataset_dirs = [d for d in results_dir.iterdir() if d.is_dir() and not d.name.startswith('cross')]
    
    print(f"Found {len(dataset_dirs)} datasets to visualize")
    print("=" * 80)
    
    for dataset_dir in sorted(dataset_dirs):
        dataset_name = dataset_dir.name
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*80}")
        
        # Visualize control network
        control_network_path = dataset_dir / "control" / "network.graphml"
        if control_network_path.exists():
            G_control = load_network(control_network_path)
            # Save to subdirectory
            output_subdir = figures_dir / "02_networks_basic"
            output_subdir.mkdir(exist_ok=True, parents=True)
            output_path = output_subdir / f"03A_network_{dataset_name}_control"
            visualize_network(G_control, output_path, layout_type=layout_type)
        else:
            print(f"  WARNING: Control network not found for {dataset_name}")
        
        # Visualize disease network
        disease_network_path = dataset_dir / "disease" / "network.graphml"
        if disease_network_path.exists():
            G_disease = load_network(disease_network_path)
            # Save to subdirectory
            output_subdir = figures_dir / "02_networks_basic"
            output_subdir.mkdir(exist_ok=True, parents=True)
            output_path = output_subdir / f"03B_network_{dataset_name}_disease"
            visualize_network(G_disease, output_path, layout_type=layout_type)
        else:
            print(f"  WARNING: Disease network not found for {dataset_name}")
    
    print("\n" + "=" * 80)
    print("Network visualization complete!")
    print(f"Figures saved to: {figures_dir}")
    print("=" * 80)


def main():
    """Main entry point."""
    # Paths relative to script location
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent.parent
    
    results_dir = project_dir / "output" / "results"
    figures_dir = project_dir / "output" / "figures"
    
    print("=" * 80)
    print("NETWORK VISUALIZATION PIPELINE")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Figures directory: {figures_dir}")
    print("=" * 80)
    
    # Generate all visualizations
    visualize_all_networks(
        results_dir=results_dir,
        figures_dir=figures_dir,
        layout_type='spring'  # Best for large bipartite networks
    )


if __name__ == "__main__":
    main()
