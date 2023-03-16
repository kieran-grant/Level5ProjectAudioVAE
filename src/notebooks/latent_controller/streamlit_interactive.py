import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from streamlit_plotly_events import plotly_events
import plotly.io as pio

pio.templates.default = "plotly"


@st.cache_data
def get_nearest_point(x, y, df):
    data = np.array([df.x, df.y]).T
    point = np.array([x, y])
    dists = np.linalg.norm(data - point, axis=1)
    idx =  np.argmin(dists)
    return idx

@st.cache_data
def get_df():
    df = pd.read_csv("./data.csv")
    return df

@st.cache_resource
def get_fig():
    df = get_df()

    return px.scatter(df, x="x", y="y", title='Plots', color='colour', color_discrete_sequence=df['colour'].to_list())

fig = get_fig()
selected_points = plotly_events(fig)

try:
    a = selected_points[0]
    # a = pd.DataFrame.from_dict(a,orient='index')
    index = get_nearest_point(a['x'], a['y'], get_df())
    a["index"] = int(index)
    a
except Exception as e:
    print("Something went wrongs", e)

