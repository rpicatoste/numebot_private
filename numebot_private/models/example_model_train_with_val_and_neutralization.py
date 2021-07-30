import pandas as pd

from numebot.data.data_manager import DataManager

from numebot_private.data.feature_neutralization import neutralize_from_forum
from numebot_private.models.example_model_train_with_val import ExampleModelTrainWithVal


# Inherits from ExampleModel!
class ExampleModelTrainWithValAndNeutralization(ExampleModelTrainWithVal):

    def predict(self, data: DataManager, to_be_saved_for_submission=False):
        numerai_data_set = data.tournament
        print('\nPredicting from child class. First, running normal prediction ...')
        output = super().predict(data=data,
                                 to_be_saved_for_submission=to_be_saved_for_submission)
        numerai_data_set['prediction'] = output
        print('Now, running feature neutralization ...')

        output = neutralize_from_forum(numerai_data_set, target='prediction')
        
        if to_be_saved_for_submission:
            self.save_for_submission(output)

        return output
