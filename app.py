import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    
            if gene_input:
                gene_list = [g.strip().lower() for g in gene_input.split(",") if g.strip()]
    
                # Filter and aggregate RNA
                rna_subset = rna_df[rna_df["Gene"].str.lower().isin(gene_list)]
                rna_avg = rna_subset.groupby(["Gene", "group"])["Z-score"].mean().unstack(fill_value=None)
    
                # Filter and aggregate Protein
                prot_subset = prot_df[prot_df["Gene"].str.lower().isin(gene_list)]
                prot_avg = prot_subset.groupby(["Gene", "group"])["Z-score"].mean().unstack(fill_value=None)
    
                # Ensure consistent region order
                expected_regions = ["posterior", "anterior", "somite"]
                rna_avg = rna_avg.reindex(columns=expected_regions)
                prot_avg = prot_avg.reindex(columns=expected_regions)
    
                # Combine unique gene names found in either
                all_genes_found = set(rna_avg.index).union(set(prot_avg.index))
                rna_avg = rna_avg.reindex(all_genes_found).sort_index()
                prot_avg = prot_avg.reindex(all_genes_found).sort_index()
    
                if rna_avg.empty and prot_avg.empty:
                    st.warning("None of the entered genes were found in either dataset.")
                else:
                    import seaborn as sns
                    import matplotlib.pyplot as plt
    
                    # Get clustering-based order from RNA
                    g = sns.clustermap(
                        rna_avg,
                        cmap="viridis",
                        figsize=(6, 10),
                        row_cluster=True,
                        col_cluster=False,
                        cbar_pos=None
                    )
                    gene_order = [rna_avg.index[i] for i in g.dendrogram_row.reordered_ind]
                    plt.close()
    
                    # Reorder both
                    rna_ordered = rna_avg.loc[gene_order]
                    prot_ordered = prot_avg.loc[gene_order]
    
                    # Set up joint color scale
                    vmin = min(rna_ordered.min().min(), prot_ordered.min().min())
                    vmax = max(rna_ordered.max().max(), prot_ordered.max().max())
    
                    # Create joint figure with horizontal colorbar
                    fig, (ax1, ax2) = plt.subplots(
                        1, 2,
                        figsize=(12, len(gene_order) * 0.4 + 2),
                        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.05},
                        sharey=True
                    )
    
                    # Plot RNA
                    sns.heatmap(
                        rna_ordered,
                        cmap="viridis",
                        vmin=vmin, vmax=vmax,
                        ax=ax1,
                        cbar=False,
                        yticklabels=False  # Hide y-labels here
                    )
                    ax1.set_title("RNA (Clustered)", fontsize=16, fontweight='bold')
                    ax1.set_xlabel("Region")
                    ax1.set_ylabel("")
    
                    # Plot Protein (with gene names on y-axis)
                    heat = sns.heatmap(
                        prot_ordered,
                        cmap="viridis",
                        vmin=vmin, vmax=vmax,
                        ax=ax2,
                        cbar=True,
                        cbar_kws={"orientation": "horizontal", "shrink": 0.6, "pad": 0.25}
                    )
                    ax2.set_title("Protein (Same Order)", fontsize=16, fontweight='bold')
                    ax2.set_xlabel("Region")
                    ax2.set_ylabel("")
                    ax2.set_yticklabels(prot_ordered.index, rotation=0)
    
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
