import streamlit as st

st.set_page_config(
    page_title='HF Dataset Explorer',
    page_icon='🤗'
)

"# HuggingFace Dataset Explorer"

"""
This app is a quick prototype of using st.experimental_connection
with HfFileSystem to access HF DataSet data from Streamlit.
"""

with st.expander("Discussion / caveats"):
    """
    It's not doing anything super interesting since all of this can
    be found easily on the HF Hub, but demonstrates how easy it is to set up.

    There's probably some cool stuff to do with
    [data editor](https://data-editor.streamlit.app/) here too!

    Works with: datasets where data is in a `data/` folder and stored in csv, jsonl, or parquet files
    """

with st.echo("below"):
    import streamlit as st
    from st_files_connection import FilesConnection

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

    "## Dataset Preview"
    kwargs = dict(nrows=nrows, ttl=3600)

    # parquet doesn't neatly support nrows
    # could be fixed with something like this:
    # https://stackoverflow.com/a/69888274/20530083
    import pathlib
    if pathlib.Path(datafile_selection).suffix == '.parquet':
        del(kwargs['nrows'])
    
    df = conn.read(datafile_selection, **kwargs)
    st.dataframe(df.head(nrows), use_container_width=True)

    "## Full code for this app"
