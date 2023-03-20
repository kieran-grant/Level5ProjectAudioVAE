import argparse

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from streamlit_plotly_events import plotly_events

pio.templates.default = "plotly"

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_id', type=str, default="fboro0y2")
n_supervised_choices = (0, 5, 10, 25, 100)
parser.add_argument('--n_supervised', choices=n_supervised_choices, default=0,
                    help='Number of supervised datapoints (default: %(default)s)')
parser.add_argument('--data_dir', type=str, default="/home/kieran/Level5ProjectAudioVAE/src/latent_controller/data")

args = parser.parse_args()


@st.cache_data
def get_nearest_point(x, y, df, n_sup):
    data = np.array([df[f'x_emb_n={n_sup}'], df[f'y_emb_n={n_sup}']]).T
    point = np.array([x, y])
    dists = np.linalg.norm(data - point, axis=1)
    idx = np.argmin(dists)
    return idx


@st.cache_data
def get_df(df_file):
    return pd.read_csv(df_file)


@st.cache_resource
def get_fig(df, n_supervised, colour):
    fig = px.scatter(df,
                     x=f"x_emb_n={n_supervised}",
                     y=f"y_emb_n={n_supervised}",
                     color=colour,
                     size_max=50
                     )

    # use custom colour
    # fig.update_traces(marker=dict(color=df[f'colour_n={n_supervised}']))
    fig.update_traces(opacity=.9)

    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(visible=False, showticklabels=False)

    fig.update_layout(plot_bgcolor="#ffffff",
                      margin=dict(l=10, r=10, t=20, b=20))

    return fig


def get_audio_for_index(idx, df):
    return df.iloc[idx]['audio_file']


DIR = f"{args.data_dir}/{args.checkpoint_id}/"

st.title("Latent Controller")

n_supervised = st.selectbox("Number of supervised points",
                            n_supervised_choices,
                            n_supervised_choices.index(args.n_supervised))

df = get_df(f"{DIR}/full_data.csv")

colour_options = [None]
# Add params to colour options
colour_options.extend([col for col in df.columns.tolist() if "p_" in col])
# Add params for brightness/depth
colour_options.extend([col for col in df.columns.tolist() if "_diff" in col])

colour = st.selectbox("Parameter coloring",
                      colour_options,
                      0)

fig = get_fig(df, n_supervised, colour)
selected_points = plotly_events(fig)

try:
    a = selected_points[0]
    # a = pd.DataFrame.from_dict(a,orient='index')
    index = get_nearest_point(a['x'], a['y'], df, n_supervised)
    filename = get_audio_for_index(index, df)

    # Show clean audio
    st.markdown("Clean audio")
    clean_audio = open(f"{DIR}/audio/audio_clean.wav", 'rb')
    clean_audio_bytes = clean_audio.read()
    st.audio(clean_audio_bytes, format='audio/wav')

    # Show effected audio
    st.markdown("Effected audio")
    effected_audio_file = open(f"{DIR}/audio/{filename}", 'rb')
    effected_audio_bytes = effected_audio_file.read()
    st.audio(effected_audio_bytes, format='audio/wav')
except IndexError:
    # selected points will be empty when page first loads
    pass
