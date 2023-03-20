from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from streamlit_plotly_events import plotly_events


@st.cache_data
def get_nearest_point(x: float, y: float, df: pd.DataFrame, n_sup: int):
    data = np.array([df[f'x_emb_n={n_sup}'], df[f'y_emb_n={n_sup}']]).T
    point = np.array([x, y])
    dists = np.linalg.norm(data - point, axis=1)
    return np.argmin(dists)


@st.cache_resource
def get_df(df_file: str):
    return pd.read_csv(df_file)


@st.cache_resource
def get_fig(df: pd.DataFrame, n_supervised: int, colour: Optional[str]):
    fig = px.scatter(df,
                     x=f"x_emb_n={n_supervised}",
                     y=f"y_emb_n={n_supervised}",
                     color=colour,
                     size_max=50,
                     hover_name='id'
                     )

    # use custom colour
    # fig.update_traces(marker=dict(color=df[f'colour_n={n_supervised}']))
    fig.update_traces(opacity=.9)

    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(visible=False, showticklabels=False)

    fig.update_layout(plot_bgcolor="#ffffff",
                      margin=dict(l=10, r=10, t=20, b=20))

    return fig


def get_audio_file_for_index(idx: int, df: pd.DataFrame):
    return df.iloc[idx]['audio_file']


def get_audio_for_file(filename: str):
    audio = open(filename, 'rb')
    return audio.read()


def get_audio_for_index(idx: int, df: pd.DataFrame):
    a_file = get_audio_file_for_index(idx, df)
    return get_audio_for_file(a_file)


def create_colour_options(df: pd.DataFrame):
    colour_options = [None]
    # Add params to colour options
    colour_options.extend([col for col in df.columns.tolist() if "p_" in col])
    # Add params for brightness/depth
    colour_options.extend([col for col in df.columns.tolist() if "_diff" in col])

    return colour_options


# ===== GLOBAL VARS =======
N_SUPERVISED_CHOICES = (0, 5, 10, 25, 100)
DATA_DIRECTORY = "/home/kieran/Level5ProjectAudioVAE/src/latent_controller/data"
EFFECT_TO_CHKPT_MAP = {
    "Overdrive": "fboro0y2",
    "RingMod": "c5rp55l2",
    "Delay": "gg4q2yj9",
    "Combo": "8283y9mm",
    "MultiBand": "th24l5fs",
}

pio.templates.default = "plotly"

# Title
st.title("Latent Controller")

# Dropdown box for effect
dafx = st.selectbox("Effect", EFFECT_TO_CHKPT_MAP.keys(), 0)
CHECKPOINT_ID = EFFECT_TO_CHKPT_MAP.get(dafx)

DIR = f"{DATA_DIRECTORY}/{CHECKPOINT_ID}/"

# Dropdown box for num supervised points
n_supervised = st.selectbox("Number of supervised points", N_SUPERVISED_CHOICES, 0)

df = get_df(f"{DIR}/full_data.csv")

colour_options = create_colour_options(df)

colour = st.selectbox("Parameter coloring",
                      colour_options,
                      0)

fig = get_fig(df, n_supervised, colour)
selected_points = plotly_events(fig)

try:
    a = selected_points[0]
    index = get_nearest_point(a['x'], a['y'], df, n_supervised)

    # Show clean audio
    st.markdown("Clean audio")
    st.audio(get_audio_for_file(f"{DIR}/audio/audio_clean.wav"))

    # Show effected audio
    st.markdown(f"Effected audio (ID={index})")
    f_name = get_audio_file_for_index(index, df)
    st.audio(get_audio_for_file(f"{DIR}/audio/{f_name}"), format='audio/wav')

except IndexError:
    # selected points will be empty when page first loads
    pass
