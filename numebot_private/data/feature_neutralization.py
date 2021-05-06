import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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
