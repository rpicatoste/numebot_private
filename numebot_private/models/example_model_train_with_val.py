from numebot.models.example_model import ExampleModel

from numebot_private.data.data_manager_extended import DataManagerExtended


# Inherits from ExampleModel!
class ExampleModelTrainWithVal(ExampleModel):

    def train_model(self, data: DataManagerExtended):
        print('Original training set:  ', data.training.shape)
        print('Training and validation set:', data.train_val.shape)

        feature_names = [f for f in data.train_val.columns if f.startswith("feature")]
        print("Training model...")
        self.model.fit(data.train_val[feature_names], data.train_val['target'])

        self.save_model()
        self.model_ready = True
        
        print('Training finished!')
