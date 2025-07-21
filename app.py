import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

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

# Load data once, use throughout
RNA_URL = BASE + "RNA_preprocessed.csv"
PROT_URL = BASE + "Protein_preprocessed.csv"

@st.cache_data
def load_data():
    rna = pd.read_csv(RNA_URL)
    prot = pd.read_csv(PROT_URL)
    rna['Type'] = 'RNA'
    prot['Type'] = 'Protein'
    return rna, prot

rna_df, prot_df = load_data()
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

    # Heatmap Tab (placeholder)
        with subtab2:
            st.markdown("### Spatial Heatmap for Multiple Genes")
            st.markdown("Enter multiple gene names separated by commas (e.g., `tbx6, msgn1, dlc`). Capitalization does not matter")
    
            gene_input = st.text_input("Genes for heatmap:", value="")

            cluster_on = st.selectbox("Cluster based on:", ["RNA", "Protein"])
            
            if gene_input:
                gene_list = [g.strip().lower() for g in gene_input.split(",") if g.strip()]
            
                # Filter and aggregate RNA
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
            
                genes_in_rna = set(rna_subset['Gene'].str.lower())
                genes_in_prot = set(prot_subset['Gene'].str.lower())
                all_genes_found = genes_in_rna.union(genes_in_prot)
                rna_avg = rna_avg.reindex(all_genes_found).sort_index()
                prot_avg = prot_avg.reindex(all_genes_found).sort_index()
            
                # After you create gene_list, rna_avg, prot_avg, and all_genes_found:

                all_genes_entered = set(gene_list)
                genes_not_found = all_genes_entered - all_genes_found
                
                if genes_not_found:
                    st.warning(f"The following genes were not found in either dataset and will be skipped: {', '.join(sorted(genes_not_found))}")
                
                if rna_avg.empty and prot_avg.empty:
                    st.warning("None of the entered genes were found in either dataset.")
                else:
                    # ... continue with plotting code ...
                    if cluster_on == "RNA":
                        cluster_df = rna_avg
                        other_df = prot_avg
                        cluster_label = "RNA"
                        other_label = "Protein"
                    else:
                        cluster_df = prot_avg
                        other_df = rna_avg
                        cluster_label = "Protein"
                        other_label = "RNA"
            
                    if len(cluster_df) == 1:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
                        vmin = min(np.nanmin(cluster_df.values), np.nanmin(other_df.values))
                        vmax = max(np.nanmax(cluster_df.values), np.nanmax(other_df.values))
            
                        sns.heatmap(
                            cluster_df,
                            cmap="viridis",
                            vmin=vmin, vmax=vmax,
                            cbar=True,
                            ax=ax1,
                            yticklabels=False if cluster_on == "RNA" else True,
                        )
                        ax1.set_title(cluster_label, fontsize=16, fontweight='bold')
            
                        sns.heatmap(
                            other_df,
                            cmap="viridis",
                            vmin=vmin, vmax=vmax,
                            cbar=True,
                            ax=ax2,
                            yticklabels=True if cluster_on == "RNA" else False,
                        )
                        ax2.set_title(other_label, fontsize=16, fontweight='bold')
                        if cluster_on == "RNA":
                            ax2.yaxis.tick_right()
                            ax2.yaxis.set_label_position("right")
                            ax2.set_yticklabels(other_df.index, rotation=0)
                        else:
                            ax1.yaxis.tick_right()
                            ax1.yaxis.set_label_position("right")
                            ax1.set_yticklabels(other_df.index, rotation=0)
            
                        st.pyplot(fig)
            
                    else:
                        # Multiple genes: cluster on chosen dataset
                        g = sns.clustermap(
                            cluster_df,
                            cmap="viridis",
                            figsize=(6, 10),
                            row_cluster=True,
                            col_cluster=False,
                            cbar_pos=None
                        )
                        gene_order = [cluster_df.index[i] for i in g.dendrogram_row.reordered_ind]
                        plt.close()
            
                        cluster_ordered = cluster_df.loc[gene_order]
                        other_ordered = other_df.loc[gene_order]
            
                        vmin = min(np.nanmin(cluster_ordered.values), np.nanmin(other_ordered.values))
                        vmax = max(np.nanmax(cluster_ordered.values), np.nanmax(other_ordered.values))
            
                        fig = plt.figure(figsize=(12, len(gene_order)*0.4 + 3))
                        gs = gridspec.GridSpec(2, 2, height_ratios=[20, 1], hspace=0.3, wspace=0.1)
            
                        ax1 = fig.add_subplot(gs[0, 0])
                        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
                        cax1 = fig.add_subplot(gs[1, 0])
                        cax2 = fig.add_subplot(gs[1, 1])
            
                        sns.heatmap(
                            cluster_ordered,
                            cmap="viridis",
                            vmin=vmin, vmax=vmax,
                            ax=ax1,
                            cbar=False,
                            yticklabels=False
                        )
                        ax1.set_title(cluster_label, fontsize=16, fontweight='bold')
                        ax1.set_xlabel("")
                        ax1.set_ylabel("")
            
                        sns.heatmap(
                            other_ordered,
                            cmap="viridis",
                            vmin=vmin, vmax=vmax,
                            ax=ax2,
                            cbar=False,
                            yticklabels=True
                        )
                        ax2.set_title(other_label, fontsize=16, fontweight='bold')
                        ax2.set_xlabel("")
                        ax2.set_ylabel("")
                        ax2.yaxis.tick_right()
                        ax2.yaxis.set_label_position("right")
                        ax2.set_yticklabels(other_ordered.index, rotation=0)
            
                        norm = plt.Normalize(vmin=vmin, vmax=vmax)
                        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
                        sm.set_array([])
            
                        cbar1 = fig.colorbar(sm, cax=cax1, orientation='horizontal')
                        cbar1.set_label(f"Z-score ({cluster_label})")
                        cbar1.ax.xaxis.set_ticks_position('bottom')
            
                        cbar2 = fig.colorbar(sm, cax=cax2, orientation='horizontal')
                        cbar2.set_label(f"Z-score ({other_label})")
                        cbar2.ax.xaxis.set_ticks_position('bottom')
            
                        st.pyplot(fig)

# ────────── Spatiotemporal Viewer ──────────
with main_tab2:
    subtab3, subtab4 = st.tabs(["RNA Time Series", "Protein Time Series"])

    with subtab3:
        st.markdown("### Spatiotemporal RNA Expression")
        # Add RNA time series logic here

    with subtab4:
        st.markdown("### Spatiotemporal Protein Expression")
        # Add Protein time series logic here
