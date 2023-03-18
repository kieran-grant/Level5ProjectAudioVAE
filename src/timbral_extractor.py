import time
import torch
import numpy as np
import timbral_models
import warnings
from typing import List


class TimbralFeatureExtractor:
    def __init__(self,
                 sample_rate: int,
                 features: List = None,
                 phase_correction: bool = False,
                 verbose: bool = False,
                 default_value: float = 50., ):

        self._feature_map = {
            "hardness": timbral_models.timbral_hardness,
            "depth": timbral_models.timbral_depth,
            "brightness": timbral_models.timbral_brightness,
            "roughness": timbral_models.timbral_roughness,
            "sharpness": timbral_models.timbral_sharpness,
            "boominess": timbral_models.timbral_booming,
            "warmth": timbral_models.timbral_warmth,
            "reverb": timbral_models.timbral_reverb
        }

        self.sample_rate = sample_rate
        # If feature list is empty, calculate all features
        self.features = self._get_features(features)
        self.phase_correction = phase_correction
        self.verbose = verbose
        self.default_value = default_value

    def _get_features(self, features):
        if len(features) == 0:
            return list(self._feature_map.keys())
        return features

    def get_num_features(self):
        return len(self.features)

    def extract_batch_features(self,
                               signal_batch: torch.Tensor,
                               return_execution_times: bool = False):
        timbre_feats = []
        device = signal_batch.device
        for i in range(signal_batch.size()[0]):
            feats, _ = self.extract_features(signal_batch[i],
                                             return_execution_times)

            timbre_feats.append(torch.Tensor(feats))
        return torch.stack(timbre_feats).to(device)

    def extract_features(self,
                         signal: torch.Tensor,
                         return_execution_times: bool = False):

        # Ignore warnings for audio extraction
        warnings.filterwarnings("ignore")

        audio_samples = signal.clone().cpu().detach().squeeze().numpy()

        timbres = []
        times = []

        for feat in self.features:
            val, time = self.extract_feature(feature_name=feat,
                                             feature_fn=self._feature_map[feat],
                                             signal=audio_samples,
                                             return_execution_times=return_execution_times)
            timbres.append(val)
            times.append(time)

        # Reset warning messages
        warnings.filterwarnings("default")

        timbres = np.array(timbres)

        # Normalise
        timbres = torch.Tensor(timbres / 100)
        times = torch.Tensor(times)

        return timbres, times

    def extract_feature(self,
                        feature_name,
                        feature_fn,
                        signal,
                        return_execution_times=False):

        out_value = self.default_value
        execution_time = 0

        if return_execution_times:
            start = time.time()

        if self.verbose:
            print(f"Calculating {feature_name}...")
        try:
            out_value = feature_fn(signal,
                                   fs=self.sample_rate,
                                   dev_output=False,
                                   phase_correction=self.phase_correction,
                                   clip_output=True)
        except Exception as e:
            print(e)
            if self.verbose:
                print(f"Error calculating : {feature_name}, retuning default value ({self.default_value}). E: {e}")
        finally:
            if return_execution_times:
                end = time.time()
                execution_time = end - start
            return out_value, execution_time
