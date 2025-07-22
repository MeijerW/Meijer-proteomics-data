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
        
                rna_avg = rna_avg.dropna(how='all')
                prot_avg = prot_avg.reindex(rna_avg.index).sort_index()
        
                if rna_avg.empty:
                    st.warning("None of the entered genes were found in the RNA dataset.")
                else:
                    # Run clustermap to extract dendrogram ordering
                    g = sns.clustermap(
                        rna_avg,
                        row_cluster=True,
                        col_cluster=False,
                        cmap="viridis",
                        yticklabels=True,
                        figsize=(1, 1),
                        cbar_pos=None
                    )
                    gene_order = [rna_avg.index[i] for i in g.dendrogram_row.reordered_ind]
                    plt.close()
        
                    rna_ordered = rna_avg.loc[gene_order]
                    prot_ordered = prot_avg.loc[gene_order]
        
                    vmin = min(np.nanmin(rna_ordered.values), np.nanmin(prot_ordered.values))
                    vmax = max(np.nanmax(rna_ordered.values), np.nanmax(prot_ordered.values))
        
                    # Plot combined figure
                    fig = plt.figure(figsize=(12, len(gene_order)*0.3 + 2))
                    gs = gridspec.GridSpec(3, 2, height_ratios=[20, 1, 1], width_ratios=[1, 1], hspace=0.2, wspace=0.05)
        
                    # Dendrogram + RNA heatmap
                    g = sns.clustermap(
                        rna_avg,
                        row_cluster=True,
                        col_cluster=False,
                        cmap="viridis",
                        yticklabels=False,
                        figsize=(6, 6),
                        cbar_pos=None
                    )
                    plt.close(g.fig)
        
                    heatmap_ax = fig.add_subplot(gs[0, 0])
                    for child in g.ax_heatmap.get_children():
                        heatmap_ax.add_artist(child)
                    heatmap_ax.set_title("RNA Expression (clustered)", fontsize=14)
                    heatmap_ax.set_xlabel("")
                    heatmap_ax.set_ylabel("")
        
                    # RNA colorbar
                    cax1 = fig.add_subplot(gs[1, 0])
                    norm = plt.Normalize(vmin=vmin, vmax=vmax)
                    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
                    sm.set_array([])
                    cbar1 = fig.colorbar(sm, cax=cax1, orientation='horizontal')
                    cbar1.set_label("Z-score (RNA)")
        
                    # Protein heatmap
                    ax2 = fig.add_subplot(gs[0, 1], sharey=heatmap_ax)
                    sns.heatmap(
                        prot_ordered,
                        cmap="viridis",
                        vmin=vmin,
                        vmax=vmax,
                        ax=ax2,
                        cbar=False,
                        yticklabels=True
                    )
                    ax2.set_title("Protein Expression", fontsize=14)
                    ax2.set_xlabel("")
                    ax2.set_ylabel("")
                    ax2.yaxis.tick_right()
                    ax2.yaxis.set_label_position("right")
                    ax2.set_yticklabels(prot_ordered.index, rotation=0)
        
                    # Protein colorbar
                    cax2 = fig.add_subplot(gs[2, 1])
                    sm2 = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
                    sm2.set_array([])
                    cbar2 = fig.colorbar(sm2, cax=cax2, orientation='horizontal')
                    cbar2.set_label("Z-score (Protein)")
        
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
