import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE = "https://raw.githubusercontent.com/MeijerW/ProteomeUI/main/Datafiles/"

RNA_URL = BASE + "RNA_preprocessed.csv"
PROT_URL = BASE + "Protein_preprocessed.csv"

@st.cache_data
def load_data():
    rna = pd.read_csv(RNA_URL)
    prot = pd.read_csv(PROT_URL)
    # Add 'Type' column for clarity if you want
    rna['Type'] = 'RNA'
    prot['Type'] = 'Protein'
    return rna, prot

rna_df, prot_df = load_data()
rna_df['group'] = rna_df['group'].str.lower()
prot_df['group'] = prot_df['group'].str.lower()

# Set Seaborn style
sns.set(style="whitegrid")
sns.set_context("paper")  # Use paper context for clean visuals

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

# Streamlit UI
st.title("Spatial Gene Expression Viewer")
st.markdown("Compare spatial expression of a gene across RNA and Protein levels.")

gene_input = st.text_input("Enter gene name (e.g., tbx6):", value="")

if gene_input:
    rna_plot = rna_df[rna_df["Gene"].str.lower() == gene_input.lower()]
    prot_plot = prot_df[prot_df["Gene"].str.lower() == gene_input.lower()]

    if rna_plot.empty and prot_plot.empty:
        st.warning(f"No data found for gene '{gene_input}'.")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        # Plot RNA
        sns.boxplot(
            data=rna_plot,
            x="group",
            y="Z-score",
            order=["posterior", "anterior", "somite"],
            palette=rna_palette,
            ax=axes[0]
        )
        axes[0].set_title("RNA Expression", fontsize=16, fontweight='bold')
        axes[0].set_xlabel("")  # Remove x-axis label
        axes[0].set_ylabel("Z-score", fontsize=12)
        axes[0].tick_params(axis='x', labelsize=12)

        # Plot Protein
        sns.boxplot(
            data=prot_plot,
            x="group",
            y="Z-score",
            order=["posterior", "anterior", "somite"],
            palette=prot_palette,
            ax=axes[1]
        )
        axes[1].set_title("Protein Expression", fontsize=16, fontweight='bold')
        axes[1].set_xlabel("")  # Remove x-axis label
        axes[1].set_ylabel("")  # Remove y-axis label to reduce clutter
        axes[1].tick_params(axis='x', labelsize=12)

        plt.tight_layout()
        st.pyplot(fig)
