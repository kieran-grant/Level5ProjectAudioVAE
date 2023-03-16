import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import scipy.io.wavfile as wavfile
import streamlit as st
import pandas as pd
from streamlit_plotly_events import plotly_events

sns.set(style='white', context='poster')


class NoOpUMAPWrapper:
    def __init__(self,
                 num_examples=1_000,
                 latent_size=128,
                 audio_dir="./plot_test_audio",
                 audio_length_sec=5,
                 sample_rate=24_000):

        self.latent_vectors = np.random.randn(num_examples, latent_size)
        self.embeddings = np.random.randn(num_examples, 2)

        self.audio_dir = audio_dir
        self.audio_length_sec = audio_length_sec
        self.sample_rate = sample_rate
        self.base_freq = 20

        self.colour_map = None
        self.audio_files = dict()

        # self.generate_colourmap()

    @staticmethod
    def _calculate_closest_index(x, y):
        dists = np.linalg.norm(x - y, axis=1)
        return np.argmin(dists)
    
    def get_dataframe(self):
        self.generate_colourmap()

        data = {
            "id": np.arange(len(self.embeddings)),
            "x": self.embeddings[:,0], 
            "y": self.embeddings[:,1], 
            "colour": self.colour_map
        }
        
        df = pd.DataFrame(data)

        return df

    def fit_transform(self, x):
        return self.embeddings

    def get_embedding_from_index(self, index):
        return self.embeddings[index]

    def get_latent_vector_from_index(self, index):
        return self.latent_vectors[index]

    def get_closest_embedding_to_point(self, x):
        index = self._calculate_closest_index(self.embeddings, x)

        return index, self.embeddings[index]

    def get_closest_latent_vector_to_point(self, x):
        index = self._calculate_closest_index(self.embeddings, x)

        return index, self.latent_vectors[index]

    def generate_dummy_audio(self, idx):
        # generate audio signal
        frequency = self.base_freq * idx
        time = np.linspace(0, self.audio_length_sec, self.audio_length_sec * self.sample_rate, False)
        signal = np.sin(frequency * 2 * np.pi * time)

        # scale signal to 16-bit integers
        signal *= 32767 / np.max(np.abs(signal))
        signal = signal.astype(np.int16)

        return signal

    def create_audio_for_index(self, idx):
        signal = self.generate_dummy_audio(idx)
        filename = f"{self.audio_dir}/audio{idx}.wav"
        # save signal as WAV file
        wavfile.write(filename, self.sample_rate, signal)
        # save to dict
        return filename

    def get_audio_file_for_point(self, x):
        idx, _ = self.get_closest_latent_vector_to_point(x)

        if idx not in self.audio_files:
            filename = self.create_audio_for_index(idx)
            self.audio_files[idx] = filename

        return self.audio_files[idx]

    @staticmethod
    def get_colour(x, y):
        assert (0 <= x <= 1)
        assert (0 <= y <= 1)

        # calculate the red, green, and blue values based on x and y coordinates
        red = int(x * 255)
        green = int(y * 255)
        blue = int(0.5 * 255)

        def clamp(color):
            return max(0, min(color, 255))

        hex_ = "#{0:02x}{1:02x}{2:02x}".format(clamp(red), clamp(green), clamp(blue))

        # return the RGBA value as hex
        return hex_

    def generate_colourmap(self):
        # Perform min/max normalisation across dimensions
        emb = (self.embeddings - self.embeddings.min(0)) / self.embeddings.ptp(0)
        self.colour_map = [self.get_colour(e[0], e[1]) for e in emb]


controller = NoOpUMAPWrapper()

if __name__ == "__main__":

    print("Fitting data...")
    # embeddings = umap_wrapper.fit_transform(latent_space)
    print("Data fitted!")

    df = controller.get_dataframe()

    # df = pd.DataFrame(controller.embeddings)
    fig = px.scatter(df, x='x', y='y', color='x')

    df.to_csv("./data.csv")

    # Can write inside of things using with!

    # st.plotly_chart(fig)
    print(plotly_events(fig, click_event=True))


