import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import os
import tempfile
import logging
from tqdm import tqdm
from typing import List, Dict, Optional
import warnings

# Try to import optional dependencies
try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False
    warnings.warn("python-Levenshtein not installed. Near-duplicate checking will be slow or skipped.")

try:
    from anarci import anarci
    HAS_ANARCI = True
except ImportError:
    HAS_ANARCI = False
    warnings.warn("ANARCI not installed. CDR3 extraction will be approximated or skipped.")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AntibodyEDA:
    def __init__(self, output_dir: str = "eda_plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Set visualization style
        sns.set_theme(style="whitegrid", context="paper")

    def plot_affinity_distribution(self, df: pd.DataFrame, target_col: str = 'neg_log_kd'):
        """
        Plot distribution of -log(Kd). Check for bimodality, truncation, and skewness.
        Consider log-normalizing or clipping outliers beyond 3 std deviations.
        """
        logger.info(f"Analyzing {target_col} distribution...")
        
        plt.figure(figsize=(10, 6))
        sns.histplot(df[target_col], kde=True, bins=50)
        plt.title(f'Distribution of {target_col}')
        plt.xlabel('-log(Kd)')
        plt.ylabel('Count')
        
        # Calculate stats
        mean_val = df[target_col].mean()
        std_val = df[target_col].std()
        skewness = df[target_col].skew()
        
        plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
        plt.axvline(mean_val + 3*std_val, color='g', linestyle=':', label='+3 Std Dev')
        plt.axvline(mean_val - 3*std_val, color='g', linestyle=':', label='-3 Std Dev')
        
        plt.text(0.05, 0.95, f'Skewness: {skewness:.2f}', transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'affinity_distribution.png'), dpi=300)
        plt.close()
        
        # Outlier count
        outliers = df[(df[target_col] > mean_val + 3*std_val) | (df[target_col] < mean_val - 3*std_val)]
        logger.info(f"Found {len(outliers)} outliers beyond 3 std devs.")
        
    def plot_sequence_lengths(self, df: pd.DataFrame, vh_col: str, vl_col: str, ag_col: str):
        """
        Plot histograms for heavy chain, light chain, and antigen separately.
        Flag extreme outliers (e.g., antigen >500 AA).
        """
        logger.info("Analyzing sequence lengths...")
        
        lengths_df = pd.DataFrame({
            'Heavy (VH)': df[vh_col].dropna().str.len(),
            'Light (VL)': df[vl_col].dropna().str.len() if vl_col in df else None,
            'Antigen (Ag)': df[ag_col].dropna().str.len()
        })
        
        fig, axes = plt.subplots(1, 3 if vl_col in df else 2, figsize=(15, 5))
        if vl_col not in df:
            axes = list(axes) + [None] # Padding for logic below
            
        plot_idx = 0
        for col in lengths_df.columns:
            if lengths_df[col].isnull().all():
                continue
                
            ax = axes[plot_idx]
            sns.histplot(lengths_df[col], bins=30, ax=ax, kde=False)
            ax.set_title(f'{col} Length Distribution')
            ax.set_xlabel('Amino Acids')
            
            # Highlighting > 500 for antigens
            if 'Antigen' in col:
                long_ags = (lengths_df[col] > 500).sum()
                logger.info(f"Found {long_ags} antigens with length > 500 AA.")
                ax.axvline(500, color='r', linestyle='--', label='> 500 AA Threshold')
                ax.legend()
                
            plot_idx += 1
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sequence_lengths.png'), dpi=300)
        plt.close()

    def _run_cdhit(self, sequences: List[str], ids: List[str], identity: float = 0.9) -> Dict[str, int]:
        """Helper function to run CD-HIT and return cluster mappings."""
        # Note: CD-HIT must be installed and in PATH (e.g. `conda install -c bioconda cd-hit`)
        try:
            subprocess.run(["cd-hit", "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            logger.error("CD-HIT executable not found. Please install it (e.g., via conda).")
            return {}

        word_length = 5
        if identity < 0.7:
            word_length = 4
            
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = os.path.join(tmpdir, "input.fasta")
            out_path = os.path.join(tmpdir, "output")
            
            # Write Fasta
            with open(fasta_path, 'w') as f:
                for seq_id, seq in tqdm(zip(ids, sequences), total=len(sequences), desc="Writing Fasta for CD-HIT"):
                    f.write(f">{seq_id}\n{seq}\n")
                    
            # Run CD-HIT
            cmd = [
                "cd-hit",
                "-i", fasta_path,
                "-o", out_path,
                "-c", str(identity),
                "-n", str(word_length),
                "-M", "0", # unlimited memory
                "-T", "0"  # all threads
            ]
            
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                logger.error(f"CD-HIT failed: {res.stderr.decode('utf-8')}")
                return {}
                
            # Parse .clstr file
            cluster_map = {}
            current_cluster = -1
            with open(f"{out_path}.clstr", 'r') as f:
                for line in f:
                    if line.startswith(">Cluster"):
                        current_cluster = int(line.strip().split()[-1])
                    else:
                        # e.g.: 0    200aa, >seq_1... *
                        seq_id = line.split(">")[1].split("...")[0]
                        cluster_map[seq_id] = current_cluster
                        
            return cluster_map

    def cluster_antigens(self, df: pd.DataFrame, ag_col: str, id_col: str = 'id'):
        """Cluster antigens by sequence identity (CD-HIT at 90%)."""
        logger.info("Clustering antigens (CD-HIT 90%)...")
        
        # Filter uniques to speed up cd-hit
        unique_ags = df[[id_col, ag_col]].drop_duplicates(subset=[ag_col])
        cluster_map = self._run_cdhit(unique_ags[ag_col].tolist(), unique_ags[id_col].astype(str).tolist(), identity=0.9)
        
        if not cluster_map:
            logger.warning("Antigen clustering skipped due to CD-HIT error.")
            return
            
        # Map back to subset
        df['ag_cluster'] = df[id_col].astype(str).map(cluster_map)
        
        # Analyze distribution
        cluster_counts = df['ag_cluster'].value_counts()
        total_samples = len(df)
        
        logger.info(f"Generated {len(cluster_counts)} unique antigen families.")
        
        top_family_pct = (cluster_counts.iloc[0] / total_samples) * 100
        logger.info(f"Top family contains {top_family_pct:.2f}% of the dataset.")
        if top_family_pct > 50:
            logger.warning("Severe imbalance detected: >50% of samples belong to one antigen family.")
            
        # Plot top 20 families
        plt.figure(figsize=(10, 6))
        sns.barplot(x=range(min(20, len(cluster_counts))), y=cluster_counts.head(20).values)
        plt.title('Top 20 Antigen Families by Sample Count')
        plt.xlabel('Cluster Rank')
        plt.ylabel('Samples')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'antigen_clusters.png'), dpi=300)
        plt.close()

    def analyze_cdr3_diversity(self, df: pd.DataFrame, vh_col: str, id_col: str = 'id'):
        """
        Extract heavy CDR3 using ANARCI. Plot length and diversity. 
        Check near-duplicates using Levenshtein distance.
        """
        logger.info("Analyzing CDR3 diversity...")
        cdr3_list = []
        
        if HAS_ANARCI:    
            logger.info("Extracting CDR3s using ANARCI...")
            # For simplicity, passing one by one. Ideally batch this.
            for idx, seq in tqdm(zip(df[id_col], df[vh_col]), total=len(df), desc="Extracting CDR3 (ANARCI)"):
                if pd.isna(seq):
                    cdr3_list.append(None)
                    continue
                # run ANARCI
                results = anarci([("seq", seq)], scheme="imgt", output=False)
                # Parse results for CDR3 (IMGT positions 105 to 117)
                # Warning: ANARCI output parsing logic can be complex, simplifying here:
                try:
                    numbering = results[1][0][0][0] # numbering map
                    cdr3_seq = "".join([res[1] for res in numbering if 105 <= res[0][0] <= 117 and res[1] != '-'])
                    cdr3_list.append(cdr3_seq)
                except Exception:
                    cdr3_list.append(None)
        else:
            logger.warning("ANARCI not found. Assuming CDR3s are provided in a column 'cdr3h' fallback.")
            if 'cdr3h' in df.columns:
                cdr3_list = df['cdr3h'].tolist()
            else:
                logger.warning("No 'cdr3h' column found. Skipping CDR3 analysis.")
                return

        df['extracted_cdr3h'] = cdr3_list
        valid_cdr3s = df['extracted_cdr3h'].dropna()
        
        if len(valid_cdr3s) == 0:
            return
            
        # 1. Length Distribution
        lengths = valid_cdr3s.str.len()
        plt.figure(figsize=(8, 5))
        sns.histplot(lengths, discrete=True)
        plt.title('Heavy CDR3 Length Distribution')
        plt.xlabel('Length (AA)')
        plt.ylabel('Count')
        plt.savefig(os.path.join(self.output_dir, 'cdr3_lengths.png'), dpi=300)
        plt.close()
        
        # 2. Near duplicates check
        if HAS_LEVENSHTEIN and len(valid_cdr3s) > 1:
            logger.info("Checking for near-duplicates using Levenshtein (Sample size max 5000)...")
            # Subsample for speed if dataset is huge
            sample_cdr3s = valid_cdr3s.sample(min(5000, len(valid_cdr3s))).tolist()
            distances = []
            # Calculate distance to nearest neighbor for a random subset to estimate density
            sample_size = min(500, len(sample_cdr3s))
            for i in tqdm(range(sample_size), desc="Calculating Levenshtein dists"):
                min_dist = float('inf')
                for j in range(len(sample_cdr3s)):
                    if i != j:
                        d = Levenshtein.distance(sample_cdr3s[i], sample_cdr3s[j])
                        if d < min_dist:
                            min_dist = d
                distances.append(min_dist)
                
            near_dupes = sum(1 for d in distances if d <= 2) # e.g. dist <= 2
            logger.info(f"Estimated near duplicates (distance <= 2): {near_dupes}/{len(distances)} in sampled subset.")
            
            plt.figure(figsize=(8, 5))
            sns.histplot(distances, discrete=True)
            plt.title('Minimum Levenshtein Distance to Nearest CDR3 Neighbor')
            plt.xlabel('Distance')
            plt.savefig(os.path.join(self.output_dir, 'cdr3_distance.png'), dpi=300)
            plt.close()

    def check_pairwise_leakage(self, dftrain: pd.DataFrame, dftest: pd.DataFrame, seq_col: str, id_col: str = 'id'):
        """Cluster full sequences at 80% identity. Any cluster that spans both train and test is a leak."""
        logger.info("Checking training/test leakage at 80% identity...")
        
        # Combine
        dftrain_cp = dftrain.copy()
        dftest_cp = dftest.copy()
        dftrain_cp['split'] = 'train'
        dftest_cp['split'] = 'test'
        
        combined = pd.concat([dftrain_cp, dftest_cp])
        combined['clean_seq'] = combined[seq_col].str.replace(' ', '').str.replace('*', '')
        
        cluster_map = self._run_cdhit(combined['clean_seq'].tolist(), combined[id_col].astype(str).tolist(), identity=0.8)
        
        if not cluster_map:
            logger.warning("Leakage check skipped due to CD-HIT error.")
            return
            
        combined['cluster_80'] = combined[id_col].astype(str).map(cluster_map)
        
        # Check overlaps
        cluster_splits = combined.groupby('cluster_80')['split'].unique()
        leaking_clusters = [c for c, splits in cluster_splits.items() if len(set(splits)) > 1]
        
        leaked_train_samples = combined[(combined['cluster_80'].isin(leaking_clusters)) & (combined['split'] == 'train')]
        leaked_test_samples = combined[(combined['cluster_80'].isin(leaking_clusters)) & (combined['split'] == 'test')]
        
        if leaking_clusters:
            logger.warning(f"Found {len(leaking_clusters)} clusters spanning train and test!")
            logger.warning(f"This affects {len(leaked_train_samples)} training samples and {len(leaked_test_samples)} test samples.")
        else:
            logger.info("No data leakage detected between train and test sets at 80% identity.")
            
        return leaking_clusters

if __name__ == "__main__":
    # Example usage script
    import argparse
    
    parser = argparse.ArgumentParser(description="Run EDA on Antibody-Antigen Dataset")
    parser.add_argument("--csv", type=str, required=True, help="Path to the dataset CSV")
    parser.add_argument("--test_csv", type=str, help="Path to test dataset CSV (for leakage check)")
    parser.add_argument("--target", type=str, default="neg_log_kd", help="Target column name")
    parser.add_argument("--vh", type=str, default="vh_seq", help="Heavy chain column name")
    parser.add_argument("--vl", type=str, default="vl_seq", help="Light chain column name")
    parser.add_argument("--ag", type=str, default="antigen_seq", help="Antigen column name")
    parser.add_argument("--id", type=str, default="id", help="ID column name")
    parser.add_argument("--outdir", type=str, default="eda_reports", help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Create dummy data if file does not exist (for rapid testing)
    if not os.path.exists(args.csv):
        logger.error(f"File {args.csv} not found.")
        exit(1)
        
    df = pd.read_csv(args.csv)
    
    # Preprocess missing IDs
    if args.id not in df.columns:
        logger.warning(f"ID column '{args.id}' not found. Auto-generating unique IDs ('dataset_index').")
        df['dataset_index'] = df.index
        args.id = 'dataset_index'
        
    eda = AntibodyEDA(output_dir=args.outdir)
    
    # 1. Target distribution
    if args.target in df.columns:
        eda.plot_affinity_distribution(df, target_col=args.target)
        
    # 2. Lengths
    eda.plot_sequence_lengths(df, vh_col=args.vh, vl_col=args.vl, ag_col=args.ag)
    
    # 3. Antigen Families
    if args.ag in df.columns:
        eda.cluster_antigens(df, ag_col=args.ag, id_col=args.id)
        
    # 4. CDR3 Diversity
    if args.vh in df.columns:
        eda.analyze_cdr3_diversity(df, vh_col=args.vh, id_col=args.id)
        
    # 5. Leakage Check
    if args.test_csv and os.path.exists(args.test_csv):
        df_test = pd.read_csv(args.test_csv)
        # Create a full sequence concatenation for leakage check
        df['full_complex'] = df[args.vh].fillna('') + df.get(args.vl, pd.Series(['']*len(df))).fillna('') + df[args.ag].fillna('')
        df_test['full_complex'] = df_test[args.vh].fillna('') + df_test.get(args.vl, pd.Series(['']*len(df_test))).fillna('') + df_test[args.ag].fillna('')
        
        eda.check_pairwise_leakage(df, df_test, seq_col='full_complex', id_col=args.id)
    elif args.test_csv:
        logger.warning(f"Test file {args.test_csv} not found. Skipping leakage check.")
