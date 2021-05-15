import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from numebot.data.data_constants import NC
from numebot.data.data_manager import DataManager
from numebot.file_names_getter import FileNamesGetter


class DataManagerExtended(DataManager):

    def __init__(self, file_names: FileNamesGetter, nrows: int, save_memory: bool=True):
        super().__init__(file_names=file_names, nrows=nrows, save_memory=save_memory)

        self._train_val = None

    @property
    def train_val(self):
        """
        Extended version of the training dataset, adding the validation data.
        """
        if self._train_val is None:
            self._train_val = pd.concat([self.training, self.val]).sort_values(by=NC.era)

        return self._train_val

    def noisy_tournament(self, n_times: int):
        feature_cols = [f for f in self.tournament.columns if f.startswith("feature")]

        return _make_noisy_dataframe(self.tournament, 
                                     columns=feature_cols + [NC.target],
                                     n_times=n_times)

    def noisy_training(self, n_times: int):
        feature_cols = [f for f in self.training.columns if f.startswith("feature")]
        
        return _make_noisy_dataframe(self.training, 
                                     columns=feature_cols + [NC.target],
                                     n_times=n_times)


def _make_noisy_dataframe(df: pd.DataFrame, columns=None, n_times=1):
    if columns is None:
        columns = df.columns

    new_df = pd.concat([df[columns]]*n_times)
    
    noise_shape = new_df.shape
    noise = (np.random.rand(*noise_shape) - 0.5) * 0.25

    new_df = new_df + noise

    new_df[:] = MinMaxScaler((0.0, 1.0)).fit_transform(new_df)
    
    return new_df
