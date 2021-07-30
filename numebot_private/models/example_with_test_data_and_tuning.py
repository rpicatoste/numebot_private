from numebot.data.data_constants import NC
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from numebot.models.example_model import ExampleModel

from numebot_private.data.data_manager_extended import DataManagerExtended


# Inherits from ExampleModel!
class ExampleWithTestDataAndTuning(ExampleModel):

    def train_model(self, data: DataManagerExtended):
        print('Original test set:  ', data.test.shape)
        print("Training model...")

        model = XGBRegressor(max_depth=5, 
                             learning_rate=0.01, 
                             n_estimators=2000, 
                             colsample_bytree=0.1,
                             n_jobs=-1,)
        cv = GroupKFold(n_splits=5)
        distributions = dict(max_depth=[3, 4, 5, 6, 7], 
                             learning_rate=[0.1, 0.01, 0.001], 
                             n_estimators=[1000, 2000, 3000],
                             colsample_bytree=[0.5, 0.1, 0.2, 0.3])

        RandomizedSearchCV
        clf = RandomizedSearchCV(model, distributions, random_state=0)
        feature_names = [f for f in data.test.columns if f.startswith("feature")]
        search = clf.fit(data.test[feature_names], data.test[NC.target])
        
        self.save_model()
        print('Training finished!')

        # Best found 
        {'n_estimators': 2000,
        'max_depth': 3,
        'learning_rate': 0.01,
        'colsample_bytree': 0.3}
        
        self.model_ready = True

        return search