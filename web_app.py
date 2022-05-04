import streamlit as st
import pandas as pd
from netgraph import main
import numpy as np

def img_show():

    loaded_file = st.file_uploader(label="Load .xlsx or .csv file. File format: work id, duration, parents (delimiter ', ')", type=['xlsx', 'csv'])
    len_factor = st.select_slider(label='The degree of influence of the factor of the distance of events from each other horizontally on their vertical position', 
                                  options=np.arange(-5, 6, 0.5))
    family_factor = st.select_slider(label='the degree of influence of the factor of the number of parents of the event on its vertical position', 
                                     options=np.arange(-5, 6, 0.5))

    if loaded_file is not None:

        if '.csv' in loaded_file.name:
            df = pd.read_csv(loaded_file)

        else:
            df = pd.read_excel(loaded_file)
    
        img = main(df, len_factor, family_factor)
        return st.image(img)

    else: 
        return None

st.title('Draw Net Graph')
img_show()
