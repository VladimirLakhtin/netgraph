import streamlit as st
import pandas as pd
from netgraph import main

def img_show():
    loaded_file = st.file_uploader(label="Load .xlsx or .csv file. File format: work id, duration, parents (delimiter ', ')", type=['xlsx', 'csv'])
    if loaded_file is not None:
        if '.csv' in loaded_file.name:
            df = pd.read_csv(loaded_file)
        else:
            df = pd.read_excel(loaded_file)
        img = main(df)
        return st.image(img)
    else: 
        return None

st.title('Draw Net Graph')
img_show()
