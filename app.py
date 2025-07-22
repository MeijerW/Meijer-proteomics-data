import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import io

# --- SETTINGS
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
            

RNA_FILES = {
    "Anterior": "RNAseq_Spatiotemporal_anterior.csv",
    "Posterior": "RNAseq_Spatiotemporal_posterior.csv",
    "Somite": "RNAseq_Spatiotemporal_somite.csv",
}
PROT_FILES = {
    "Anterior": "Proteomics_Spatiotemporal_anterior.csv",
    "Posterior": "Proteomics_Spatiotemporal_posterior.csv",
    "Somite": "Proteomics_Spatiotemporal_somite.csv",
}


#---FUNCTIONS
@st.cache_data
def load_data():
    rna = pd.read_csv(RNA_URL)
    prot = pd.read_csv(PROT_URL)
    rna['Type'] = 'RNA'
    prot['Type'] = 'Protein'
    return rna, prot

@st.cache_data
def load_spatiotemporal_data():
    def load_files(file_dict):
        dfs = {}
        for region, filename in file_dict.items():
            df = pd.read_csv(BASE + filename, sep=',')  # Although file says .tsv, it's actually CSV
            df.set_index('ID', inplace=True)
            dfs[region] = df
        return dfs

    return load_files(RNA_FILES), load_files(PROT_FILES)

def prepare_long_df(df_dict, gene, datatype):
    all_data = []
    for region, df in df_dict.items():
        if gene not in df.index:
            continue
        row = df.loc[gene]
        expression = row.filter(like='TP_')
        melted = expression.reset_index()
        melted.columns = ['Condition', 'Expression']
        melted['Time'] = melted['Condition'].str.extract(r'TP_(\d+)_REP_\d+')
        melted['Rep'] = melted['Condition'].str.extract(r'REP_(\d+)')
        melted['Time'] = pd.Categorical(melted['Time'], categories=['30', '60', '90', '120'], ordered=True)
        melted['Region'] = region
        melted['Type'] = datatype
        all_data.append(melted)
    return pd.concat(all_data, ignore_index=True)

def plot_expression_grid(df, gene_name):
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True)

    for i, region in enumerate(["Posterior", "Anterior", "Somite"]):
        for j, datatype in enumerate(["RNA", "Protein"]):
            ax = axes[j, i]
            sub_df = df[(df['Region'] == region) & (df['Type'] == datatype)]
            if sub_df.empty:
                ax.set_visible(False)
                continue

            # Select color
            color = rna_palette[region.lower()] if datatype == "RNA" else prot_palette[region.lower()]

            # Boxplot
            sns.boxplot(
                data=sub_df,
                x="Time", y="Expression", color=color, ax=ax,
                order=['30', '60', '90', '120'], fliersize=0, width=0.6
            )
            # Overlay replicates
            sns.stripplot(
                data=sub_df,
                x="Time", y="Expression", color="black", ax=ax,
                order=['30', '60', '90', '120'], size=3, jitter=True
            )

            ax.set_title(f"{region}", fontsize=20)
            ax.set_xlabel("")
            ax.set_ylabel("Expression" if i == 0 else "")

            # Dynamic y-limit
            y_min, y_max = sub_df["Expression"].min(), sub_df["Expression"].max()
            y_pad = (y_max - y_min) * 0.1 if y_max > y_min else 1
            ax.set_ylim(y_min - y_pad, y_max + y_pad)


    fig.text(0.01, 0.6, "RNA", va="center", ha="right", fontsize=20, fontweight="bold", rotation=90)
    fig.text(0.01, 0.23, "Protein", va="center", ha="right", fontsize=20, fontweight="bold", rotation=90)
    fig.suptitle(f"Spatiotemporal Expression of {gene_name}", fontsize=26)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# Top-level tabs
main_tab1, main_tab2 = st.tabs(["Spatial Viewer", "Spatiotemporal Viewer"])

# ────────── Spatial Viewer ──────────
with main_tab1:
    subtab1, subtab2 = st.tabs(["Single Gene", "Heatmap (Multiple Genes)"])

    #Load spatial data directly
    rna_df, prot_df = load_data()
    rna_df['group'] = rna_df['group'].str.lower()
    prot_df['group'] = prot_df['group'].str.lower()
    
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
    subtab3, subtab4 = st.tabs(["Single Gene", "Heatmap (Multiple Genes)"])

    with subtab3:
        st.markdown("### Single gene dynamic expression")
        rna_dict, prot_dict = load_spatiotemporal_data()
        # Input gene
        gene_input = st.text_input("Enter gene name:", value="Lfng")
        
        if gene_input:
            rna_long = prepare_long_df(rna_dict, gene_input, "RNA")
            prot_long = prepare_long_df(prot_dict, gene_input, "Protein")
            combined_df = pd.concat([rna_long, prot_long], ignore_index=True)
        
            if combined_df.empty:
                st.warning(f"Gene '{gene_input}' not found in datasets.")
            else:
                fig = plot_expression_grid(combined_df, gene_input)
                st.pyplot(fig)
            
            # Save figure to a BytesIO buffer
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
            buf.seek(0)

            # Download button
            st.download_button(
                label="📥 Download this figure as PNG",
                data=buf,
                file_name=f"{gene_input}_spatiotemporal_expression.png",
                mime="image/png"
            )
   

    with subtab4:
        st.markdown("### Multi-gene Spatiotemporal Expression")


        # Select region
        region = st.selectbox("Select region", ["anterior", "posterior", "somite"])




