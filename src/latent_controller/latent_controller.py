from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from plotly import graph_objects as go
from streamlit_plotly_events import plotly_events

TEST_1_MARKER_LOOKUP = {
    "overdrive": {
        1: "x_emb_n=0",
        2: "y_emb_n=0",
        3: "p_drive"
    },
    "multiband": {
        1: "brightness_diff",
        2: "x_emb_n=0",
        3: "y_emb_n=0",
    },
}


@st.cache_data
def get_nearest_point(x: float, y: float, df: pd.DataFrame, n_sup: int):
    data = np.array([df[f'x_emb_n={n_sup}'], df[f'y_emb_n={n_sup}']]).T
    point = np.array([x, y])
    dists = np.linalg.norm(data - point, axis=1)
    return np.argmin(dists)


@st.cache_resource
def get_df(df_file: str):
    return pd.read_csv(df_file)


def get_marker(df, idx, label="", n_sup=0):
    return go.Scatter(
        x=[df.iloc[idx][f"x_emb_n={n_sup}"]],
        y=[df.iloc[idx][f"y_emb_n={n_sup}"]],
        mode="markers+text",
        marker=dict(
            color="red",
            size=15,
        ),
        textfont=dict(
            family="Courier New, monospace",
            size=22,
            color="crimson"
        ),
        text=label,
        name=label,
        textposition="bottom center"
    )


def get_line(df, start_idx, end_idx, n_sup):
    return go.Scatter(
        x=[df.iloc[start_idx][f"x_emb_n={n_sup}"], df.iloc[end_idx][f"x_emb_n={n_sup}"]],
        y=[df.iloc[start_idx][f"y_emb_n={n_sup}"], df.iloc[end_idx][f"y_emb_n={n_sup}"]],
        mode="lines",
        line=dict(
            color="red",
        ),
    )


def get_midpoint(df, idx_1, idx_2, n_sup):
    x1 = df.iloc[idx_1][f'x_emb_n={n_sup}']
    y1 = df.iloc[idx_1][f'y_emb_n={n_sup}']

    x2 = df.iloc[idx_2][f'x_emb_n={n_sup}']
    y2 = df.iloc[idx_2][f'y_emb_n={n_sup}']

    x_mid = (x1 + x2) / 2
    y_mid = (y1 + y2) / 2

    return get_nearest_point(x_mid, y_mid, df, n_sup=n_sup)


def get_markers_for_test(df, dafx_name, test_num, example_num, n_supervised):
    markers = []
    idx_min, idx_max, idx_mid = None, None, None
    # Interpolation test
    if test_num == 1 and example_num is not None and dafx_name.lower() in TEST_1_MARKER_LOOKUP:
        df_col = TEST_1_MARKER_LOOKUP[dafx_name.lower()][example_num]

        idx_min = df[df_col].argmin()
        idx_max = df[df_col].argmax()
        idx_mid = get_midpoint(df, idx_min, idx_max, n_sup=n_supervised)

        line = get_line(df, idx_min, idx_max, n_sup=n_supervised)

        a_marker = get_marker(df, idx_min, label='A', n_sup=n_supervised)
        b_marker = get_marker(df, idx_max, label='B', n_sup=n_supervised)
        c_marker = get_marker(df, idx_mid, label='C', n_sup=n_supervised)

        markers.append(line)
        markers.append(a_marker)
        markers.append(b_marker)
        markers.append(c_marker)

    elif test_num == 2 and dafx_name.lower() in TEST_1_MARKER_LOOKUP:
        x_mean = df[f"x_emb_n={n_supervised}"].mean()
        y_mean = df[f"y_emb_n={n_supervised}"].mean()

        idx_min = get_nearest_point(x_mean, y_mean, df, n_sup=n_supervised)

        mean_marker = get_marker(df, idx_min, label='X', n_sup=n_supervised)

        markers.append(mean_marker)

    return markers, idx_min, idx_max, idx_mid


def get_fig(df: pd.DataFrame,
            n_supervised: int,
            colour: Optional[str],
            dafx: str,
            test_num: Optional[int],
            example_num: Optional[int]):
    fig = px.scatter(df,
                     x=f"x_emb_n={n_supervised}",
                     y=f"y_emb_n={n_supervised}",
                     color=colour,
                     size_max=50,
                     hover_name='id'
                     )

    idx_max, idx_min, idx_mid = None, None, None
    if test_num is not None:
        markers, idx_min, idx_max, idx_mid = get_markers_for_test(df, dafx, test_num, example_num, n_supervised)
        for mark in markers:
            fig.add_trace(mark)
        fig.update_layout(showlegend=False)

    # use custom colour
    # fig.update_traces(marker=dict(color=df[f'colour_n={n_supervised}']))
    fig.update_traces(opacity=.8)

    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(visible=False, showticklabels=False)

    fig.update_layout(plot_bgcolor="#ffffff",
                      margin=dict(l=10, r=10, t=20, b=20))

    return fig, idx_min, idx_max, idx_mid


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
    "Ambience": "6d7hvfwc",
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

# Dropdown box for test
test_num = st.selectbox("Test", [None, 1, 2], 0)
if test_num == 1:
    example_num = st.selectbox("Example", [None, 1, 2, 3], 0)
else:
    example_num = None

fig, idx_min, idx_max, idx_mid = get_fig(df, n_supervised, colour, dafx, test_num, example_num)

selected_points = plotly_events(fig)

# Show clean audio
st.markdown("Clean audio")
st.audio(get_audio_for_file(f"{DIR}/audio/audio_clean.wav"))

# if idx_min is not None, then others are not None
if test_num == 1 and idx_min is not None:
    st.markdown(f"A audio")
    f_name = get_audio_file_for_index(idx_min, df)
    st.audio(get_audio_for_file(f"{DIR}/audio/{f_name}"), format='audio/wav')

    st.markdown(f"B audio")
    f_name = get_audio_file_for_index(idx_max, df)
    st.audio(get_audio_for_file(f"{DIR}/audio/{f_name}"), format='audio/wav')

    st.markdown(f"C audio")
    f_name = get_audio_file_for_index(idx_mid, df)
    st.audio(get_audio_for_file(f"{DIR}/audio/{f_name}"), format='audio/wav')
else:
    if test_num == 2 and idx_min is not None:
        st.markdown(f"X audio")
        f_name = get_audio_file_for_index(idx_min, df)
        st.audio(get_audio_for_file(f"{DIR}/audio/{f_name}"), format='audio/wav')
    try:
        a = selected_points[0]
        index = get_nearest_point(a['x'], a['y'], df, n_supervised)

        # Show effected audio
        st.markdown(f"Selected point (ID={index})")
        f_name = get_audio_file_for_index(index, df)
        st.audio(get_audio_for_file(f"{DIR}/audio/{f_name}"), format='audio/wav')

    except IndexError:
        # selected points will be empty when page first loads
        pass
