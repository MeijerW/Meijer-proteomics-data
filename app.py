import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set base URL for GitHub raw files
BASE = "https://raw.githubusercontent.com/MeijerW/ProteomeUI/main/Datafiles/"

# Load preprocessed data
@st.cache_data
def load_data():
    rna = pd.read_csv(BASE + "RNA_preprocessed.csv")
    prot = pd.read_csv(BASE + "Protein_preprocessed.csv")
    return rna, prot

# Load data
@st.cache_data
def load_data():
    rna = pd.read_csv(RNA_URL)
    prot = pd.read_csv(PROT_URL)
    return rna, prot

rna_df, prot_df = load_data()

# Set page
st.title("Spatial Gene Expression Viewer")
st.markdown("Compare spatial expression of a gene across RNA and Protein levels.")

# Input box for gene name
gene_input = st.text_input("Enter gene name (e.g., tbx6):", value="tbx6")

if gene_input:
    # Filter by gene (case-insensitive match)
    rna_plot = rna_df[rna_df["Gene"].str.lower() == gene_input.lower()]
    prot_plot = prot_df[prot_df["Gene"].str.lower() == gene_input.lower()]

    if rna_plot.empty and prot_plot.empty:
        st.warning(f"No data found for gene '{gene_input}'.")
    else:
        # Create side-by-side boxplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        sns.boxplot(
            data=rna_plot, 
            x="region", 
            y="Z-score", 
            order=["posterior", "anterior", "somite"],
            ax=axes[0]
        )
        axes[0].set_title("RNA Expression")
        axes[0].set_xlabel("Region")
        axes[0].set_ylabel("Z-score")

        sns.boxplot(
            data=prot_plot, 
            x="region", 
            y="Z-score", 
            order=["posterior", "anterior", "somite"],
            ax=axes[1]
        )
        axes[1].set_title("Protein Expression")
        axes[1].set_xlabel("Region")

        st.pyplot(fig)
