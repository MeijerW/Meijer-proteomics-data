import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

st.set_page_config(layout="wide")

st.title("🧬 Spatial RNA & Proteomics Expression Explorer")


@st.cache_data
def load_data():
#load spatial data
    BASE = "https://raw.githubusercontent.com/MeijerW/ProteomeUI/main/Datafiles/"
    rna = pd.read_csv(BASE + "RNAseq_Spatial_Normed_counts.csv", index_col=0)
    rna_meta = pd.read_csv(BASE + "RNAseq-metadata-spatial.csv", sep=';')
    prot = pd.read_csv(BASE + "Proteomics_Spatial_Short.csv", index_col=0)
    prot_meta = pd.read_csv(BASE + "Proteomics-metadata-spatial.csv", sep=';')
    return rna, rna_meta, prot, prot_meta


rna, rna_meta, prot, prot_meta = load_data()

# -- Preprocessing ----------------------------------------------

# Ensure sample names match column names
rna_samples = rna_meta['sample'].values
prot_samples = prot_meta['sample'].values

# Filter for the relevant samples only
rna_filtered = rna[rna_samples]
prot_filtered = prot[prot_samples]

# Z-score per gene (row-wise)
rna_z = rna_filtered.apply(zscore, axis=1)
prot_z = prot_filtered.apply(zscore, axis=1)

# Melt for plotting
def melt_with_meta(df_z, meta_df, source_label):
    df_long = df_z.T.reset_index().melt(id_vars="index", var_name="Gene", value_name="Z-score")
    df_long = df_long.rename(columns={"index": "sample"})
    df_long = df_long.merge(meta_df, on="sample", how="left")
    df_long["Source"] = source_label
    return df_long

rna_long = melt_with_meta(rna_z, rna_meta, "RNA")
prot_long = melt_with_meta(prot_z, prot_meta, "Protein")

# Combine both for easier plotting
combined_df = pd.concat([rna_long, prot_long], axis=0)

# -- User Input & Plotting --------------------------------------

gene_input = st.text_input("Enter a gene name :", placeholder="Tbx6")

if gene_input:
    gene = gene_input.strip()
    gene_df = combined_df[combined_df["Gene"].str.lower() == gene.lower()]

    if not gene_df.empty:
        cols = st.columns(2)

        for source, col in zip(["RNA", "Protein"], cols):
            plot_df = gene_df[gene_df["Source"] == source]

            if not plot_df.empty:
                with col:
                    st.subheader(f"{source} Expression (Z-score)")
                    fig, ax = plt.subplots(figsize=(5, 5))
                    sns.boxplot(data=plot_df, x="goup", y="Z-score", ax=ax)
                    sns.stripplot(data=plot_df, x="goup", y="Z-score", hue="replicate", dodge=True, ax=ax, palette="dark:.4", legend=False)
                    ax.set_title(f"{gene.upper()} in {source}")
                    ax.set_xlabel("Region")
                    ax.set_ylabel("Z-score")
                    st.pyplot(fig)
            else:
                with col:
                    st.warning(f"{gene} not found in {source} dataset.")
    else:
        st.warning(f"Gene '{gene}' not found in either dataset.")
