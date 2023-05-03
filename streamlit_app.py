import streamlit as st
from st_files_connection import FilesConnection
import pandas as pd

"# HuggingFace Dataset Explorer"

"""
This app is a quick prototype of using st.experimental_connection
with HfFileSystem to access HF DataSet data from Streamlit.

It's not doing anything super interesting since all of this can
be found easily on the HF Hub, but demonstrates how easy it is to set up.

There's probably some cool stuff to do with
[data editor](https://data-editor.streamlit.app/) here too!
"""

with st.echo("below"):
    conn = st.experimental_connection('hf', type=FilesConnection)

    # Enter a dataset and retrieve a list of data files
    dataset_name = st.text_input(
        "Enter your dataset of interest",
        value="EleutherAI/lambada_openai"
        )
    dataset = f'datasets/{dataset_name}'
    file_options = [f['name'] for f in conn.fs.ls(f"{dataset}/data")]

    datafile_selection = st.selectbox("Pick a data file", file_options)
    nrows = st.slider("Rows to retrieve", value=50)

    # View the first few rows
    # TODO: add native json support to FilesConnection.read()
    # so this can be simplified to a one liner
    @st.cache_data(ttl=3600)
    def retrieve_data(selection, nrows):
        with conn.open(selection) as f:
            return pd.read_json(f, lines=True, nrows=nrows)
        
    "## Dataset Preview"
    df = retrieve_data(datafile_selection, nrows)
    st.dataframe(df, use_container_width=True)
