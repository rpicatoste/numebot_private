import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from numebot.data.data_manager import DataManager


def noisy_tournament(data: DataManager, n_times: int):
    feature_cols = [f for f in data.tournament.columns if f.startswith("feature")]
    
    return _make_noisy_dataframe(data.tournament, 
                                    columns=feature_cols + ['target'], 
                                    n_times=n_times)


def noisy_training(data: DataManager, n_times: int):
    feature_cols = [f for f in data.training.columns if f.startswith("feature")]
    
    return _make_noisy_dataframe(data.training, 
                                    columns=feature_cols + ['target'], 
                                    n_times=n_times)


def _make_noisy_dataframe(df: pd.DataFrame, columns=None, n_times=1):
    if columns is None:
        columns = df.columns

    new_df = pd.concat([df[columns]]*n_times)
    
    noise_shape = new_df.shape
    noise = (np.random.rand(*noise_shape)-0.5)*0.25

    new_df = new_df + noise

    new_df[:] = MinMaxScaler((0.0, 1.0)).fit_transform(new_df)
    
    return new_df
