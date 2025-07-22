import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import io

BASE = "https://raw.githubusercontent.com/MeijerW/ProteomeUI/main/Datafiles/"

st.title("Somitogenesis Gene Expression Explorer")

# Set Seaborn style
sns.set(style="whitegrid")
sns.set_context("paper")

# Color palettes
rna_palette = {
    "posterior": "#d5af34",
    "anterior": "#f9d777",
    "somite": "#f9e7b7"
}
prot_palette = {
    "posterior": "#8281be",
    "anterior": "#b2b2d9",
    "somite": "#d7d6ea"
}
# Spatial data files
RNA_URL = BASE + "RNA_preprocessed.csv"
PROT_URL = BASE + "Protein_preprocessed.csv"

# Spatiotemporal file templates
SPATIOTEMPORAL_REGIONS = ["anterior", "posterior", "somite"]

SPATIOTEMPORAL_RNA_TEMPLATE = BASE + "RNAseq_Spatiotemporal_{}.csv"
SPATIOTEMPORAL_PROT_TEMPLATE = BASE + "Protein_Spatiotemporal_{}.csv"

# Load spatial data
@st.cache_data
def load_spatial_data():
    rna = pd.read_csv(RNA_URL)
    prot = pd.read_csv(PROT_URL)
    rna['Type'] = 'RNA'
    prot['Type'] = 'Protein'
    return rna, prot

def average_replicates(df, prefix="TP_", reps=3):
    """
    Given a dataframe with replicate columns like TP_30_REP_1 ... TP_30_REP_n,
    average replicates per timepoint and return a DataFrame with columns as timepoints.
    """
    timepoints = [30, 60, 90, 120]
    averaged_data = {}
    
    for tp in timepoints:
        rep_cols = [f"{prefix}{tp}_REP_{i+1}" for i in range(reps)]
        # Some replicates might not exist, so filter cols that are in df
        rep_cols = [col for col in rep_cols if col in df.columns]
        if rep_cols:
            averaged_data[f"{tp}min"] = df[rep_cols].mean(axis=1)
        else:
            # No data for this timepoint, fill with NaNs
            averaged_data[f"{tp}min"] = np.nan

    return pd.DataFrame(averaged_data, index=df.index)


# Load spatiotemporal data

@st.cache_data
def load_spatiotemporal_data(region):
    rna_url = SPATIOTEMPORAL_RNA_TEMPLATE.format(region)
    prot_url = SPATIOTEMPORAL_PROT_TEMPLATE.format(region)
    
    # Read CSV instead of TSV
    rna = pd.read_csv(rna_url)  # no sep="\t" needed for CSV
    prot = pd.read_csv(prot_url)
    
    rna.set_index("ID", inplace=True)
    prot.set_index("ID", inplace=True)
    
    # Average replicates (RNA 3 reps, Protein 4 reps)
    rna_avg = average_replicates(rna, reps=3)
    prot_avg = average_replicates(prot, reps=4)
    
    return rna_avg, prot_avg, rna, prot

# Load spatial data immediately
rna_df, prot_df = load_spatial_data()
rna_df['group'] = rna_df['group'].str.lower()
prot_df['group'] = prot_df['group'].str.lower()

# Top-level tabs
main_tab1, main_tab2 = st.tabs(["Spatial Viewer", "Spatiotemporal Viewer"])

# ────────── Spatial Viewer ──────────
with main_tab1:
    subtab1, subtab2 = st.tabs(["Single Gene", "Heatmap (Multiple Genes)"])

    # Single Gene Boxplot
    with subtab1:
        st.markdown("### Single Gene Spatial Expression")
        st.markdown("Compare spatial expression of a gene across RNA and Protein levels.")
        gene_input = st.text_input("Enter gene name (e.g., tbx6):", value="")

        if gene_input:
            rna_plot = rna_df[rna_df["Gene"].str.lower() == gene_input.lower()]
            prot_plot = prot_df[prot_df["Gene"].str.lower() == gene_input.lower()]

            if rna_plot.empty and prot_plot.empty:
                st.warning(f"No data found for gene '{gene_input}'.")
            else:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

                sns.boxplot(
                    data=rna_plot,
                    x="group",
                    y="Z-score",
                    order=["posterior", "anterior", "somite"],
                    palette=rna_palette,
                    ax=axes[0]
                )
                axes[0].set_title("RNA Expression", fontsize=26, fontweight='bold')
                axes[0].set_xlabel("")
                axes[0].set_ylabel("Z-score", fontsize=16)
                axes[0].tick_params(axis='x', labelsize=16)

                sns.boxplot(
                    data=prot_plot,
                    x="group",
                    y="Z-score",
                    order=["posterior", "anterior", "somite"],
                    palette=prot_palette,
                    ax=axes[1]
                )
                axes[1].set_title("Protein Expression", fontsize=26, fontweight='bold')
                axes[1].set_xlabel("")
                axes[1].set_ylabel("")
                axes[1].tick_params(axis='x', labelsize=16)

                plt.tight_layout()
                st.pyplot(fig)
                # Save figure to a BytesIO buffer
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
                buf.seek(0)
                
                # Add download button
                st.download_button(
                    label="📥 Download this figure as PNG",
                    data=buf,
                    file_name= (gene_input + "_spatial_boxplot.png"),
                    mime="image/png"
)
    # Heatmap Tab 
        with subtab2:
            st.markdown("### Spatial Heatmap for Multiple Genes")
            st.markdown("Enter multiple gene names separated by commas (e.g., `tbx6, msgn1, dlc`). Capitalization does not matter.")
        
            gene_input = st.text_input("Genes for heatmap:", value="")
        
            if gene_input:
                gene_list = [g.strip().lower() for g in gene_input.split(",") if g.strip()]
        
                rna_subset = rna_df[rna_df["Gene"].str.lower().isin(gene_list)]
                prot_subset = prot_df[prot_df["Gene"].str.lower().isin(gene_list)]
        
                def prepare_avg(df):
                    grouped = df.groupby(['Gene', 'group'])['Z-score'].mean()
                    unstacked = grouped.unstack(fill_value=np.nan)
                    if isinstance(unstacked, pd.Series):
                        unstacked = unstacked.to_frame().T
                    return unstacked
        
                rna_avg = prepare_avg(rna_subset)
                prot_avg = prepare_avg(prot_subset)
        
                expected_regions = ["posterior", "anterior", "somite"]
                rna_avg = rna_avg.reindex(columns=expected_regions)
                prot_avg = prot_avg.reindex(columns=expected_regions)
        
                # Only keep genes with any data in RNA
                rna_avg = rna_avg.dropna(how='all')
                prot_avg = prot_avg.reindex(rna_avg.index).sort_index()
                
                # Check missing genes
                # Normalize index to lowercase for comparison
                found_genes_lower = {g.lower() for g in rna_avg.index.union(prot_avg.index)}
                genes_not_found = [g for g in gene_list if g not in found_genes_lower]
                
                if genes_not_found:
                    st.warning(f"The following genes were not found in either RNA or Protein datasets and will not be shown: {', '.join(genes_not_found)}")

                if rna_avg.empty:
                    st.warning("None of the entered genes were found in the RNA dataset.")
                else:
                    # Get clustering order
                    g = sns.clustermap(
                        rna_avg,
                        cmap="viridis",
                        row_cluster=True,
                        col_cluster=False,
                        cbar_pos=None,
                        figsize=(1, 1)  # dummy
                    )
                    plt.close()
        
                    gene_order = [rna_avg.index[i] for i in g.dendrogram_row.reordered_ind]
                    rna_ordered = rna_avg.loc[gene_order]
                    prot_ordered = prot_avg.loc[gene_order]
        
                    # Separate vmin/vmax for each
                    vmin_rna, vmax_rna = np.nanmin(rna_ordered.values), np.nanmax(rna_ordered.values)
                    vmin_prot, vmax_prot = np.nanmin(prot_ordered.values), np.nanmax(prot_ordered.values)
        
                    # Layout
                    fig = plt.figure(figsize=(12, len(gene_order) * 0.4 + 3))
                    gs = gridspec.GridSpec(2, 2, height_ratios=[20, 1], width_ratios=[1, 1], hspace=0.4, wspace=0.05)
        
                    ax1 = fig.add_subplot(gs[0, 0])
                    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
                    cax1 = fig.add_subplot(gs[1, 0])
                    cax2 = fig.add_subplot(gs[1, 1])
        
                    sns.heatmap(
                        rna_ordered,
                        cmap="viridis",
                        ax=ax1,
                        cbar=False,
                        vmin=vmin_rna,
                        vmax=vmax_rna,
                        yticklabels=True
                    )
                    ax1.set_title("RNA Expression (clustered)", fontsize=14)
                    ax1.set_xlabel("")
                    ax1.set_ylabel("")
                    ax1.set_yticklabels(rna_ordered.index, rotation=0)
                    
                    sns.heatmap(
                        prot_ordered,
                        cmap="viridis",
                        ax=ax2,
                        cbar=False,
                        vmin=vmin_prot,
                        vmax=vmax_prot,
                        yticklabels=True  # <--- Changed here
                    )
                    ax2.set_title("Protein Expression", fontsize=14)
                    ax2.set_xlabel("")
                    ax2.set_ylabel("")
                    ax2.yaxis.tick_right()
                    ax2.yaxis.set_label_position("right")
                    ax2.set_yticklabels(prot_ordered.index, rotation=0)
        
                    # Colorbars
                    sm_rna = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin_rna, vmax=vmax_rna))
                    sm_rna.set_array([])
                    cbar1 = fig.colorbar(sm_rna, cax=cax1, orientation='horizontal')
                    cbar1.set_label("Z-score (RNA)")
        
                    sm_prot = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin_prot, vmax=vmax_prot))
                    sm_prot.set_array([])
                    cbar2 = fig.colorbar(sm_prot, cax=cax2, orientation='horizontal')
                    cbar2.set_label("Z-score (Protein)")
        
                    st.pyplot(fig)
                    # Save figure to a BytesIO buffer
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
                    buf.seek(0)
                    
                    # Add download button
                    st.download_button(
                        label="📥 Download Heatmap as PNG",
                        data=buf,
                        file_name="spatial_heatmap.png",
                        mime="image/png"
                    )

# ────────── Spatiotemporal Viewer ──────────
with main_tab2:
    subtab3, subtab4 = st.tabs(["Single gene dynamics", "Multi-gene heatmap"])

    with subtab3:
        st.markdown("### Spatiotemporal Gene Expression")
        # Add RNA time series logic here

    with subtab4:
        st.markdown("### Spatiotemporal Heatmap for RNA and Protein")

        region = st.selectbox("Select Region", SPATIOTEMPORAL_REGIONS)
        gene_input = st.text_input("Enter gene names (comma-separated)", value="")
        meta_option = st.selectbox("Select metadata to display", ["P_VALUE", "PERIOD", "LAG"])

        if gene_input:
            gene_list = [g.strip().lower() for g in gene_input.split(",") if g.strip()]

            # Load spatiotemporal data
            rna_temp_df, prot_temp_df = load_spatiotemporal_data(region)
            rna_temp_df["ID"] = rna_temp_df["ID"].str.lower()
            prot_temp_df["ID"] = prot_temp_df["ID"].str.lower()

            rna_subset = rna_temp_df[rna_temp_df["ID"].isin(gene_list)]
            prot_subset = prot_temp_df[prot_temp_df["ID"].isin(gene_list)]

            found_genes = set(rna_subset["ID"]).union(prot_subset["ID"])
            not_found_genes = [g for g in gene_list if g not in found_genes]
            if not_found_genes:
                st.info(f"Genes not found: {', '.join(not_found_genes)}")

            if rna_subset.empty and prot_subset.empty:
                st.warning("None of the selected genes were found in this region.")
            else:
                # Timepoint columns
                tp_cols = [col for col in rna_temp_df.columns if col.startswith("TP_")]
                timepoints = sorted(set(col.split("_")[1] for col in tp_cols), key=int)

                def compute_zscores(df, cols):
                    expr = df.set_index("ID")[cols]
                    # Average per timepoint
                    avg_df = pd.DataFrame(index=expr.index)
                    for tp in timepoints:
                        reps = [c for c in expr.columns if f"TP_{tp}_" in c]
                        avg_df[tp] = expr[reps].mean(axis=1)
                    zscore = StandardScaler().fit_transform(avg_df)
                    return pd.DataFrame(zscore, index=avg_df.index, columns=avg_df.columns)

                rna_z = compute_zscores(rna_subset, tp_cols) if not rna_subset.empty else pd.DataFrame()
                prot_z = compute_zscores(prot_subset, tp_cols) if not prot_subset.empty else pd.DataFrame()

                all_genes = sorted(set(rna_z.index).union(prot_z.index))
                rna_z = rna_z.reindex(all_genes)
                prot_z = prot_z.reindex(all_genes)

                def extract_metadata(df, field):
                    return pd.DataFrame({field: df.set_index("ID")[field].reindex(all_genes)})

                rna_meta = extract_metadata(rna_subset, meta_option)
                prot_meta = extract_metadata(prot_subset, meta_option)

                # Plot
                fig = plt.figure(figsize=(16, len(all_genes) * 0.4 + 3))
                gs = gridspec.GridSpec(2, 4, height_ratios=[20, 1], width_ratios=[10, 1.5, 10, 1.5], hspace=0.4, wspace=0.3)

                # RNA heatmap + meta
                ax1 = fig.add_subplot(gs[0, 0])
                ax1_meta = fig.add_subplot(gs[0, 1], sharey=ax1)
                ax1_cb = fig.add_subplot(gs[1, 0])
                ax1_meta.set_title(f"{meta_option} (RNA)", fontsize=10)

                if not rna_z.empty:
                    sns.heatmap(rna_z, cmap="viridis", ax=ax1, cbar=False, yticklabels=True)
                    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=rna_z.min().min(), vmax=rna_z.max().max()))
                    fig.colorbar(sm, cax=ax1_cb, orientation='horizontal').set_label("RNA Z-score")

                if not rna_meta.empty:
                    sns.heatmap(rna_meta, cmap="coolwarm", annot=True, fmt=".2f", cbar=False, ax=ax1_meta, yticklabels=False)

                # Protein heatmap + meta
                ax2 = fig.add_subplot(gs[0, 2], sharey=ax1)
                ax2_meta = fig.add_subplot(gs[0, 3], sharey=ax1)
                ax2_cb = fig.add_subplot(gs[1, 2])
                ax2_meta.set_title(f"{meta_option} (Protein)", fontsize=10)

                if not prot_z.empty:
                    sns.heatmap(prot_z, cmap="viridis", ax=ax2, cbar=False, yticklabels=False)
                    sm2 = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=prot_z.min().min(), vmax=prot_z.max().max()))
                    fig.colorbar(sm2, cax=ax2_cb, orientation='horizontal').set_label("Protein Z-score")

                if not prot_meta.empty:
                    sns.heatmap(prot_meta, cmap="coolwarm", annot=True, fmt=".2f", cbar=False, ax=ax2_meta, yticklabels=False)

                st.pyplot(fig)
