from numebot.data.data_constants import NC
from numebot.data.data_manager import DataManager
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import pandas as pd

from numebot.models.numerai_model import NumeraiModel


class EraBoosted(NumeraiModel):


    def load_model(self):
        class_name = str(self.__class__).rstrip('>\'').split('.')[-1]
        print(f'Creating {class_name}')
        model_file = self.names.model_path(self.name, suffix='xgb')

        # This is the model that generates the included example predictions file.
        # Taking too long? Set learning_rate=0.1 and n_estimators=200 to make this run faster.
        # Remember to delete example_model.xgb if you change any of the parameters below.
        model = XGBRegressor(max_depth=5, 
                             learning_rate=0.01, 
                             n_estimators=2000, 
                             n_jobs=-1, 
                             colsample_bytree=0.1)

        if model_file.exists():
            print("Loading pre-trained model...")
            model.load_model(model_file)
            self.model_ready = True
        else:
            print(f'WARNING: Model for {self.name} is not trained: Run train!')
    
        return model


    def train_model(self, data: DataManager, proportion=0.5, trees_per_step=10, num_iters=200):

        X = data.train_val[data.features]
        y = data.train_val[NC.target]
        era_col = data.train_val[NC.era]

        model = XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=trees_per_step, n_jobs=-1, colsample_bytree=0.1)

        model.fit(X, y)
        new_df = X.copy()
        new_df["target"] = y
        new_df["era"] = era_col
        for i in range(num_iters-1):
            print(f"iteration {i}")
            # score each era
            print("predicting on train")
            preds = model.predict(X)
            new_df["pred"] = preds
            era_scores = pd.Series(index=new_df["era"].unique())
            print("getting per era scores")
            for era in new_df["era"].unique():
                era_df = new_df[new_df["era"] == era]
                era_scores[era] = spearmanr(era_df["pred"], era_df["target"])
            era_scores.sort_values(inplace=True)
            worst_eras = era_scores[era_scores <= era_scores.quantile(proportion)].index
            print(list(worst_eras))
            worst_df = new_df[new_df["era"].isin(worst_eras)]
            era_scores.sort_index(inplace=True)
            era_scores.plot(kind="bar")
            print("performance over time")
            plt.show()
            print("autocorrelation")
            #print(ar1(era_scores))
            print("mean correlation")
            print(np.mean(era_scores))
            print("sharpe")
            print(np.mean(era_scores)/np.std(era_scores))
            print("smart sharpe")
            #print(smart_sharpe(era_scores))
            model.n_estimators += trees_per_step
            booster = model.get_booster()
            print("fitting on worst eras")
            model.fit(worst_df[data.features], worst_df["target"], xgb_model=booster)


        print('Final number of estimators:', self.model.n_estimators)
        self.model = model

        if not self.testing:
            self.save_model()
            print('Model saved!')
        else:
            print('Testing mode: trained model not saved.')
        
        self.model_ready = True
        print('Training finished!')


def spearmanr(target, pred):
    return np.corrcoef(
        target,
        pred.rank(pct=True, method="first")
    )[0, 1]
