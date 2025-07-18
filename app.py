import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gene Explorer", layout="wide")

#load data
df = pd.read_csv("https://raw.githubusercontent.com/MeijerW/ProteomeUI/edit/main/Datafiles/my_expression_data.csv")

#make heatmap
