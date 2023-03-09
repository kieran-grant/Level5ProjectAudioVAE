import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.io.wavfile as wavfile

sns.set(style='white', context='poster')


class NoOpUMAPWrapper:
    def __init__(self,
                 num_examples=100,
                 latent_size=128,
                 audio_dir="./plot_test_audio",
                 audio_length_sec=5,
                 sample_rate=24_000):

        self.latent_vectors = np.random.rand(num_examples, latent_size)
        self.embeddings = np.random.rand(num_examples, 2)

        self.audio_dir = audio_dir
        self.audio_length_sec = audio_length_sec
        self.sample_rate = sample_rate
        self.base_freq = 20

        self.colour_map = None
        self.audio_files = dict()

    @staticmethod
    def _calculate_closest_index(x, y):
        dists = np.linalg.norm(x - y, axis=1)
        return np.argmin(dists)

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
        """
        Function should take in an embedding and find the closest matching embedding in the latent space (get vector from embedding)
        Check if audio file exists for that embedding
        if it does, return it
        Otherwise:
            Use feedforward section of model to predict output audio and save to file
            add file to dict and return audio

        :param embedding:
        :return:
        """
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

    def plot_embeddings(self):
        if self.colour_map is None:
            self.generate_colourmap()

        fig, ax = plt.subplots(1, 1)

        scatter = ax.scatter(self.embeddings[:, 0], self.embeddings[:, 1], c=self.colour_map)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        def onclick(event):
            if ax == event.inaxes:
                x = event.xdata
                y = event.ydata
                point = np.array([x, y])
                filename = self.get_audio_file_for_point(point)
                print(f"Audio saved to {filename}")

            # define hover event handler

        def on_hover(event):
            if event.inaxes == ax:
                point = np.array([event.xdata, event.ydata])
                point_id, _ = self.get_closest_embedding_to_point(point)
                tooltip.set_text(f"ID: {point_id}")
                tooltip.set_visible(True)
            else:
                tooltip.set_visible(False)
            # create tooltip annotation and hide it by default

        tooltip = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                              bbox=dict(boxstyle="round", fc="w"),
                              arrowprops=dict(arrowstyle="->"))

        tooltip.set_visible(False)

        # register hover event handler
        fig.canvas.mpl_connect("motion_notify_event", on_hover)

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()


if __name__ == "__main__":
    umap_wrapper = NoOpUMAPWrapper()

    print("Fitting data...")
    # embeddings = umap_wrapper.fit_transform(latent_space)
    print("Data fitted!")

    umap_wrapper.plot_embeddings()
