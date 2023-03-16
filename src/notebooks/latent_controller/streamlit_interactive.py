import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import scipy.io.wavfile as wavfile
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

def generate_dummy_audio(idx):
        # generate audio signal
        frequency =  50 * idx
        time = np.linspace(0, 5, 5* 24_000, False)
        signal = np.sin(frequency * 2 * np.pi * time)

        # scale signal to 16-bit integers
        signal *= 32767 / np.max(np.abs(signal))
        signal = signal.astype(np.int16)

        return signal

def create_audio_for_index(idx):
        signal = generate_dummy_audio(idx)
        filename = f"./plot_test_audio/audio{idx}.wav"
        # save signal as WAV file
        wavfile.write(filename, 24_000, signal)
        # save to dict
        return filename

fig = get_fig()
selected_points = plotly_events(fig)


a = selected_points[0]
# a = pd.DataFrame.from_dict(a,orient='index')
index = get_nearest_point(a['x'], a['y'], get_df())
filename = create_audio_for_index(index)
a["index"] = int(index)
a["filename"] = filename

audio_file = open(filename, 'rb')
audio_bytes = audio_file.read()

st.audio(audio_bytes, format='audio/wav')
st.write(a)

