"""Metabolite classification and annotation helper."""

import re
import pandas as pd
from typing import Dict, Tuple


class MetaboliteClassifier:
    """Classify metabolites into functional groups."""

    def __init__(self):
        """Initialize metabolite class definitions."""

        # KEGG-based classifications
        self.kegg_classes = {
            # Short-chain fatty acids (SCFAs)
            'SCFA': [
                'C00042', 'C00246', 'C00163', 'C00249', 'C00803',  # Succinate, Butyrate, Propionate, Valerate, Isovalerate
                'C01013', 'C02939',  # Isobutyrate, 2-methylbutyrate
            ],

            # Bile acids
            'Bile_acid': [
                'C00695', 'C05122', 'C01921', 'C03990', 'C04483',  # Cholate, Chenodeoxycholate, Deoxycholate, Lithocholate, Ursodeoxycholate
                'C02528', 'C17661', 'C15516',  # Taurocholate, Glycocholate, etc.
            ],

            # Amino acids (proteinogenic)
            'Amino_acid': [
                'C00041', 'C00082', 'C00183', 'C00079', 'C00078',  # Ala, Tyr, Val, Phe, Trp
                'C00025', 'C00037', 'C00047', 'C00049', 'C00062',  # Glu, Gly, Lys, Asp, Arg
                'C00065', 'C00073', 'C00097', 'C00123', 'C00135',  # Ser, Met, Cys, Leu, His
                'C00148', 'C00152', 'C00188', 'C00407',  # Pro, Asn, Thr, Ile
            ],

            # Branched-chain amino acids (BCAAs)
            'BCAA': [
                'C00183', 'C00123', 'C00407',  # Val, Leu, Ile
            ],

            # Aromatic amino acids
            'Aromatic_AA': [
                'C00079', 'C00082', 'C00078',  # Phe, Tyr, Trp
            ],

            # Amino acid metabolites
            'AA_metabolite': [
                'C00956', 'C05332', 'C00398', 'C00483', 'C01102',  # Phenyllactate, Phenethylamine, Tryptamine, Indole, Kynurenine
                'C00327', 'C02918', 'C00109',  # Citrulline, Ornithine, Putrescine
            ],

            # Nucleotides and derivatives
            'Nucleotide': [
                'C00002', 'C00008', 'C00020', 'C00015', 'C00044',  # ATP, ADP, AMP, UDP, GTP
                'C00063', 'C00144', 'C00147', 'C00286',  # CTP, GMP, Adenine, UMP
                'C00003', 'C00004', 'C00005', 'C00006',  # NAD+, NADH, NADP+, NADPH
            ],

            # Sugars and glycolysis intermediates
            'Sugar': [
                'C00031', 'C00095', 'C00208', 'C00124', 'C00267',  # Glucose, Fructose, Maltose, Galactose, Glucose-6-P
                'C00029', 'C00043', 'C00052',  # UDP-glucose, UDP-GlcNAc, UDP-galactose
            ],

            # TCA cycle intermediates
            'TCA_cycle': [
                'C00158', 'C00042', 'C00091', 'C00122', 'C00149',  # Citrate, Succinate, Fumarate, Malate, Oxaloacetate
                'C00026', 'C00417',  # 2-oxoglutarate, cis-Aconitate
            ],

            # Polyamines
            'Polyamine': [
                'C00109', 'C00315', 'C00134', 'C01137',  # Putrescine, Spermidine, Spermine, N-acetylputrescine
            ],

            # Vitamins and cofactors
            'Vitamin': [
                'C00378', 'C00378', 'C00101', 'C00255', 'C00120',  # Thiamine, B vitamins, Biotin, Riboflavin
                'C00864', 'C00919',  # Pantothenate, Choline
            ],

            # Lipids
            'Lipid': [
                'C00162', 'C00249', 'C00527', 'C00638',  # Fatty acids, Palmitate, Oleate, Stearate
            ],

            # Antioxidants
            'Antioxidant': [
                'C00127', 'C00051', 'C00157',  # Glutathione, GSH, Ascorbate
            ],

            # Indoles and tryptophan metabolites
            'Indole': [
                'C00463', 'C00954', 'C00078', 'C00643', 'C00780',  # Indole, Indole-3-acetate, Trp, Indoxyl, Skatole
            ],
        }

        # Name-based classifications (for non-KEGG metabolites)
        self.name_patterns = {
            'Bile_acid': [
                'cholate', 'chenodeoxycholate', 'deoxycholate', 'lithocholate',
                'ursodeoxycholate', 'taurocholate', 'glycocholate', 'DCA', 'LCA',
                'bile', 'cholic'
            ],
            'SCFA': [
                'acetate', 'propionate', 'butyrate', 'valerate', 'succinate',
                'isobutyrate', 'isovalerate', 'hexanoate'
            ],
            'Amino_acid': [
                'alanine', 'glycine', 'valine', 'leucine', 'isoleucine',
                'serine', 'threonine', 'cysteine', 'methionine', 'proline',
                'phenylalanine', 'tyrosine', 'tryptophan', 'histidine',
                'lysine', 'arginine', 'aspartate', 'glutamate', 'asparagine', 'glutamine'
            ],
            'Indole': [
                'indole', 'indoxyl', 'tryptamine', 'skatole', 'indoleacetic'
            ],
            'Polyamine': [
                'putrescine', 'spermidine', 'spermine', 'cadaverine'
            ],
            'Vitamin': [
                'thiamine', 'riboflavin', 'niacin', 'biotin', 'folate', 'cobalamin',
                'ascorbate', 'tocopherol', 'vitamin'
            ],
        }

    def extract_kegg_id(self, metabolite_name: str) -> str:
        """Extract KEGG ID from metabolite name.

        Args:
            metabolite_name: Metabolite name (e.g., 'C00041_Ala' or 'cholate')

        Returns:
            KEGG ID if found, else empty string
        """
        match = re.match(r'^"?(C\d{5})', metabolite_name)
        if match:
            return match.group(1)
        return ''

    def extract_common_name(self, metabolite_name: str) -> str:
        """Extract common name from metabolite string.

        Args:
            metabolite_name: Full metabolite name

        Returns:
            Common name in lowercase
        """
        # Remove quotes and KEGG prefix
        name = metabolite_name.strip('"').lower()

        # Pattern 1: "C00041_Ala" -> "ala"
        if '_' in name and name.startswith('c'):
            parts = name.split('_', 1)
            if len(parts) > 1:
                return parts[1].strip()

        # Pattern 2: "C18-neg_Cluster_0031: phenyllactate" -> "phenyllactate"
        if ':' in name:
            parts = name.split(':', 1)
            if len(parts) > 1:
                return parts[1].strip()

        # Pattern 3: Just the name
        return name

    def classify_metabolite(self, metabolite_name: str) -> Tuple[str, str]:
        """Classify a metabolite into functional class.

        Args:
            metabolite_name: Metabolite name

        Returns:
            Tuple of (class_name, classification_method)
        """
        # Try KEGG-based classification first
        kegg_id = self.extract_kegg_id(metabolite_name)
        if kegg_id:
            for class_name, kegg_ids in self.kegg_classes.items():
                if kegg_id in kegg_ids:
                    return (class_name, 'KEGG')

        # Try name-based classification
        common_name = self.extract_common_name(metabolite_name)
        for class_name, patterns in self.name_patterns.items():
            for pattern in patterns:
                if pattern in common_name:
                    return (class_name, 'name')

        # Unknown class
        return ('Unknown', 'unclassified')

    def classify_dataframe(self, df: pd.DataFrame, metabolite_col: str = 'metabolite') -> pd.DataFrame:
        """Add metabolite class column to dataframe.

        Args:
            df: DataFrame with metabolite column
            metabolite_col: Name of metabolite column

        Returns:
            DataFrame with added 'metabolite_class' and 'class_method' columns
        """
        classifications = df[metabolite_col].apply(self.classify_metabolite)
        df['metabolite_class'] = [c[0] for c in classifications]
        df['class_method'] = [c[1] for c in classifications]
        return df

    def get_class_summary(self) -> pd.DataFrame:
        """Get summary of all defined metabolite classes.

        Returns:
            DataFrame with class names and counts
        """
        summary = []
        for class_name, kegg_ids in self.kegg_classes.items():
            summary.append({
                'class': class_name,
                'n_kegg_ids': len(kegg_ids),
                'has_name_patterns': class_name in self.name_patterns
            })
        return pd.DataFrame(summary)


if __name__ == "__main__":
    # Test the classifier
    classifier = MetaboliteClassifier()

    test_metabolites = [
        'C00041_Ala',
        'C00042_Succinate',
        'C00695_Cholate',
        'C18-neg_Cluster_1258: cholate',
        'HILIC-pos_Cluster_0086: N-acetylputrescine',
        'C00398_Tryptamine',
        'Unknown_metabolite',
    ]

    print("Testing metabolite classifier:\n")
    for met in test_metabolites:
        class_name, method = classifier.classify_metabolite(met)
        print(f"{met:50s} -> {class_name:20s} ({method})")

    print("\n\nClass summary:")
    print(classifier.get_class_summary())
