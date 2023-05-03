import streamlit as st
from st_files_connection import FilesConnection
import pandas as pd

"# HuggingFace Dataset Explorer"

"""
This app is a quick prototype of using st.experimental_connection
with HfFileSystem to access HF DataSet data from Streamlit.

It's not doing anything super interesting since all of this can
be found easily on the HF Hub, but demonstrates how easy it is to set up.
"""

conn = st.experimental_connection('hf', type=FilesConnection)

# Enter a dataset and retrieve a list of data files
dataset_name = st.text_input(
    "Enter your dataset of interest",
    value="EleutherAI/lambada_openai"
    )
dataset = f'datasets/{dataset_name}'
file_options = [f['name'] for f in conn.fs.ls(f"{dataset}/data")]

datafile_selection = st.selectbox("Pick a data file", file_options)

# View the first few rows
@st.cache_data
def retrieve_data(selection):
    with conn.open(datafile_selection) as f:
        return pd.read_json(f, lines=True, nrows=50)
    

df = retrieve_data(datafile_selection)
st.dataframe(df)
