import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from numebot.models.example_model import ExampleModel


class ExampleModelWithNeutralization(ExampleModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def predict(self, numerai_data_set: pd.DataFrame, save_for_submission=False):
        print('\nPredicting from child class. First, running normal prediction ...')
        output = super().predict(numerai_data_set=numerai_data_set,
                                 save_for_submission=False)
        numerai_data_set['prediction'] = output
        print('Now, running feature neutralization ...')

        output = neutralize_from_forum(numerai_data_set, target='prediction')
        
        if save_for_submission:
            self.save_for_submission(output)

        return output


def neutralize_from_forum(df, target="target", by=None, proportion=1.0):
    if by is None:
        by = [x for x in df.columns if x.startswith('feature')]

    scores = df[target]
    exposures = df[by].values

    # constant column to make sure the series is completely neutral to exposures
    exposures = np.hstack((exposures, np.array([np.mean(scores)] * len(exposures)).reshape(-1, 1)))
    scores -= proportion * (exposures @ (np.linalg.pinv(exposures) @ scores.values))

    neutralized = scores / scores.std()
    print(type(neutralized))
    print(neutralized.shape)

    neutralized = pd.DataFrame(neutralized)
    print(neutralized.columns)

    neutralized[['prediction']] = MinMaxScaler().fit_transform(neutralized[[target]])
    
    return neutralized
