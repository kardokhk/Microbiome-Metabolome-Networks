# Microbiome-Metabolome Network Analysis Pipeline

## Overview

This pipeline performs comprehensive microbiome-metabolome network analysis with disease stratification, identifying keystone species and hub metabolites that are disrupted in disease states.

## Quick Start

The analysis workflow has been simplified into **two main steps**:

### Step 1: Run Complete Analysis

```bash
python run_analysis.py
```

This runs the entire analysis pipeline including:
- Disease-stratified network analysis (control vs disease)
- Hub metabolite identification
- Cross-study convergence analysis
- Module enrichment analysis
- Cross-disease comparison
- Metabolite functional annotation

**Outputs:**
- `output/results/{dataset}/control/` - Control network results
- `output/results/{dataset}/disease/` - Disease network results
- `output/results/{dataset}/differential/` - Differential analysis
- `output/results/cross_study/` - Cross-study convergence
- `output/results/cross_disease/` - Cross-disease comparison
- `logs/` - Comprehensive log files with timestamps

### Step 2: Generate Visualizations

```bash
python run_visualizations.py
```

This generates all publication-quality figures and tables:
- Study overview and dataset characteristics
- Network visualizations (hub metabolites, keystone species, modules)
- Differential analysis visualizations (control vs disease)
- Cross-study integration and key findings

**Outputs:**
- `output/figures/` - Publication-ready figures
- `output/tables/` - Summary tables and statistics

## Project Structure

```
.
├── run_analysis.py              # Step 1: Complete analysis pipeline
├── run_visualizations.py        # Step 2: All visualizations
├── core/                        # Core analysis modules
│   ├── analysis.py              # Keystone species analysis
│   ├── associations.py          # Correlation computation
│   ├── data_handler.py          # Data loading and stratification
│   ├── differential.py          # Differential analysis
│   ├── filtering.py             # Feature filtering
│   ├── logger.py                # Logging system
│   ├── modules.py               # Module detection
│   ├── networks.py              # Network construction
│   └── transformations.py       # Data transformations
├── scripts/                     # Analysis and visualization scripts
│   ├── cross_disease_comparison.py
│   ├── enhanced_hub_metabolite_analysis.py
│   ├── hub_metabolite_analysis.py
│   ├── metabolite_annotation.py
│   ├── metabolite_classifier.py
│   ├── module_enrichment_analysis.py
│   ├── utils/                   # Configuration and utilities
│   └── visualizations/          # Visualization scripts
├── data/                        # Input datasets
├── output/                      # Analysis outputs
│   ├── results/                 # Analysis results
│   ├── figures/                 # Figures
│   └── tables/                  # Tables
├── logs/                        # Log files
└── docs/                        # Documentation

```

## Logging

All analysis runs generate detailed log files in the `logs/` directory with timestamps:
- `logs/pipeline_YYYYMMDD_HHMMSS.log` - Complete execution log
- Includes timing information for each phase
- Records all decisions, filtering steps, and results
- Useful for debugging and understanding the analysis

## Requirements

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Analysis parameters are defined in `scripts/utils/config.py`:
- Dataset selection
- Filtering thresholds
- Parallel processing settings
- Output directories

## Datasets

The pipeline analyzes shotgun metagenomics datasets with paired microbiome (species-level) and metabolome data:
- YACHIDA_CRC_2019 - Colorectal cancer
- ERAWIJANTARI_GASTRIC_CANCER_2020 - Gastric cancer
- FRANZOSA_IBD_2019 - Inflammatory bowel disease
- iHMP_IBDMDB_2019 - IBD (multi-omics)
- MARS_IBS_2020 - Irritable bowel syndrome
- WANG_ESRD_2020 - End-stage renal disease

